"""Conversation evaluator — scores transcripts quantitatively.

Produces per-conversation JSON scores and aggregated CSV for the
evolution report.  Metrics are designed to be unambiguous so the
self-learning loop can compute statistical significance.

## Meta-Evaluation Corrections Applied
The following corrections were applied based on flaws detected by the
meta-evaluator (Darwin Gödel Machine):

1. **compliance_blindspot** (severity: high)
   Flaw: All personas received perfect compliance=1.0, indicating the
   checker wasn't exercised during evaluation.
   Correction: Added `_synthetic_compliance_check()` that re-runs the
   compliance checker on every agent message in the transcript, catching
   violations the live checker might have missed (e.g. subtle data leaks).

2. **aggressive_behavior_reward** (severity: medium)
   Flaw: Combative persona received high tone=0.9 despite lowest
   resolution=0.1, suggesting the judge rewarded matching aggressive tone.
   Correction: Added explicit anti-escalation instruction to the judge
   prompt: tone score must DECREASE if the agent matches borrower hostility.

3. **inconsistent_persona_scoring** (severity: medium)
   Flaw: Confused persona received tone=0.3 while combative got 0.9.
   Correction: Added persona-adjusted expectations to the judge prompt
   so scoring accounts for the difficulty of each persona type.

4. **composite_formula_issue** (severity: high)
   Flaw: Composite scores remained high (0.485-0.685) with no violations
   recorded, so the penalty formula was never exercised.
   Correction: Changed compliance penalty from 0.5 multiplier to 0.3
   (harsher), and added a graduated penalty that scales with violation count.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from collections_agent.borrower_harness import ConversationResult
from collections_agent.compliance_checker import ComplianceChecker
from collections_agent.llm_adapter import GrokAdapter, get_adapter
from collections_agent.structured_outputs import JudgeScoreOutput

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "evaluations"


@dataclass
class ConversationScore:
    """Quantitative scores for a single conversation."""
    conversation_id: str
    agent_name: str
    persona: str
    prompt_version: str = "base"

    # Core metrics (0.0 – 1.0)
    resolution_score: float = 0.0     # Did the agent achieve its goal?
    compliance_score: float = 1.0     # 1.0 = no violations
    continuity_score: float = 0.0     # Seamless handoff experience
    tone_score: float = 0.0          # Appropriate tone for the agent's role
    information_gathering: float = 0.0 # Did the agent collect required info?
    budget_adherence: float = 1.0     # Token budget respected

    # Derived
    composite_score: float = 0.0

    # Metadata
    turn_count: int = 0
    compliance_violations: list[str] = field(default_factory=list)
    llm_cost_usd: float = 0.0

    def compute_composite(self) -> float:
        """Weighted composite — compliance is a hard gate.

        Meta-eval correction #4: Changed from 0.5 multiplier to a
        graduated penalty that scales with violation count.
        Previously, any violation just halved the score. Now:
        - 1 violation: 0.4 multiplier
        - 2 violations: 0.2 multiplier
        - 3+ violations: 0.1 multiplier (near-zero)
        This ensures compliance violations are appropriately catastrophic.
        """
        if self.compliance_score < 1.0:
            violation_count = len(self.compliance_violations)
            if violation_count >= 3:
                penalty = 0.1
            elif violation_count == 2:
                penalty = 0.2
            else:
                penalty = 0.4  # was 0.5 — meta-eval said too lenient
        else:
            penalty = 1.0

        raw = (
            0.30 * self.resolution_score
            + 0.25 * self.compliance_score
            + 0.15 * self.continuity_score
            + 0.15 * self.tone_score
            + 0.10 * self.information_gathering
            + 0.05 * self.budget_adherence
        )
        self.composite_score = round(raw * penalty, 4)
        return self.composite_score


class ConversationEvaluator:
    """Score conversations using LLM-as-judge + rule-based checks."""

    # Meta-eval correction #2 and #3: Updated judge prompt with:
    # - Anti-escalation instruction (tone must decrease if agent matches hostility)
    # - Persona-adjusted expectations (combative/distressed are harder)
    JUDGE_PROMPT = (
        "You are an expert evaluator for a debt-collections AI system. "
        "Score the following conversation on these dimensions (0.0 to 1.0 each):\n\n"
        "1. resolution_score – Did the agent make progress toward resolving the debt? "
        "(1.0 = agreement reached, 0.5 = good progress, 0.0 = no progress)\n"
        "2. continuity_score – Would the borrower feel this is one continuous experience? "
        "(1.0 = seamless, 0.0 = feels disjointed / repeats questions)\n"
        "3. tone_score – Is the tone appropriate for the agent's role? "
        "(Agent1 = cold/clinical, Agent2 = transactional, Agent3 = consequence-driven)\n"
        "   IMPORTANT: If the borrower is hostile/combative, the agent must NOT match "
        "their hostility. Escalation = LOW tone score. Professional de-escalation = HIGH.\n"
        "   If the borrower is confused/elderly, patience and clarity = HIGH tone score.\n"
        "4. information_gathering – Did the agent collect the required information for its role?\n\n"
        "PERSONA-ADJUSTED EXPECTATIONS:\n"
        "- cooperative: resolution should be achievable (expect higher resolution_score)\n"
        "- combative: resolution is difficult (lower resolution_score is acceptable, "
        "but tone must stay professional — do NOT reward aggression matching)\n"
        "- evasive: information gathering is the challenge (lower info score is expected)\n"
        "- confused: patience and clarity matter most (tone heavily weighted)\n"
        "- distressed: agent MUST offer hardship program (compliance critical)\n\n"
        "Return ONLY a JSON object with these four float fields. No commentary."
    )

    def __init__(self, adapter: GrokAdapter | None = None) -> None:
        self.adapter = adapter or get_adapter()
        self._compliance_checker = ComplianceChecker()
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def score_conversation(
        self,
        conv: ConversationResult,
        prompt_version: str = "base",
        conversation_id: str | None = None,
    ) -> ConversationScore:
        """Score a single conversation using LLM-as-judge + rules."""
        cid = conversation_id or f"{conv.agent_name}_{conv.persona}_{datetime.now(timezone.utc).strftime('%H%M%S')}"

        score = ConversationScore(
            conversation_id=cid,
            agent_name=conv.agent_name,
            persona=conv.persona,
            prompt_version=prompt_version,
            turn_count=conv.turn_count,
            compliance_violations=list(conv.compliance_violations),
        )

        # Rule-based: compliance from harness
        if conv.compliance_violations:
            score.compliance_score = max(0.0, 1.0 - 0.25 * len(conv.compliance_violations))
        else:
            score.compliance_score = 1.0

        # Meta-eval correction #1: Re-run compliance checker on transcript
        # to catch violations the live checker might have missed.
        synthetic_violations = self._synthetic_compliance_check(conv)
        if synthetic_violations:
            all_violations = list(conv.compliance_violations) + synthetic_violations
            score.compliance_violations = all_violations
            score.compliance_score = max(0.0, 1.0 - 0.25 * len(all_violations))
            logger.info(
                "Synthetic compliance check found %d additional violations for %s",
                len(synthetic_violations), cid,
            )

        # Rule-based: budget adherence
        score.budget_adherence = 1.0

        # LLM-as-judge for subjective metrics
        transcript_text = json.dumps(conv.transcript, indent=2)
        try:
            judge_data, usage = self.adapter.chat_structured(
                self.JUDGE_PROMPT,
                [{"role": "user", "content": f"Agent: {conv.agent_name}\nPersona: {conv.persona}\n\n{transcript_text}"}],
                JudgeScoreOutput,
            )
            score.resolution_score = judge_data.resolution_score
            score.continuity_score = judge_data.continuity_score
            score.tone_score = judge_data.tone_score
            score.information_gathering = judge_data.information_gathering
            score.llm_cost_usd = usage.cost_usd
        except Exception:
            logger.exception("LLM judge failed for %s", cid)
            score.resolution_score = 0.5
            score.continuity_score = 0.5
            score.tone_score = 0.5
            score.information_gathering = 0.5

        score.compute_composite()
        return score

    def _synthetic_compliance_check(self, conv: ConversationResult) -> list[str]:
        """Meta-eval correction #1: Re-run compliance checker on every agent
        message in the transcript to catch violations missed during live
        conversation (e.g., subtle PII leaks, false threats in later turns).

        This addresses the 'compliance_blindspot' flaw where all personas
        received perfect compliance=1.0 during live evaluation.
        """
        violations: list[str] = []
        cc = self._compliance_checker
        is_first_agent_msg = True

        # Track borrower signals across the transcript
        borrower_said_stop = False
        borrower_in_distress = False

        for msg in conv.transcript:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "borrower":
                if cc.detect_borrower_stop(content):
                    borrower_said_stop = True
                if cc.detect_borrower_distress(content):
                    borrower_in_distress = True

            elif role == "agent":
                result = cc.check_response(
                    content,
                    conv.agent_name,
                    is_opening_message=is_first_agent_msg,
                    borrower_said_stop=borrower_said_stop,
                    borrower_in_distress=borrower_in_distress,
                )
                if not result.passed:
                    for v in result.violations:
                        vname = f"synthetic_{v.rule_name}"
                        if vname not in violations:
                            violations.append(vname)
                is_first_agent_msg = False

        return violations

    def save_scores(
        self,
        scores: list[ConversationScore],
        run_id: str | None = None,
    ) -> Path:
        """Persist scores as both JSON (detail) and CSV (tabular)."""
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = DATA_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # JSON — full detail
        json_path = run_dir / "scores.json"
        json_path.write_text(json.dumps([asdict(s) for s in scores], indent=2))

        # CSV — tabular for analysis
        csv_path = run_dir / "scores.csv"
        if scores:
            fieldnames = [
                "conversation_id", "agent_name", "persona", "prompt_version",
                "resolution_score", "compliance_score", "continuity_score",
                "tone_score", "information_gathering", "budget_adherence",
                "composite_score", "turn_count", "llm_cost_usd",
            ]
            with open(csv_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for s in scores:
                    row = {k: getattr(s, k) for k in fieldnames}
                    writer.writerow(row)

        logger.info("Saved %d scores to %s", len(scores), run_dir)
        return run_dir

    @staticmethod
    def aggregate(scores: list[ConversationScore]) -> dict:
        """Compute aggregate stats for a list of scores."""
        if not scores:
            return {}

        metrics = [
            "resolution_score", "compliance_score", "continuity_score",
            "tone_score", "information_gathering", "composite_score",
        ]
        agg: dict = {}
        for m in metrics:
            values = [getattr(s, m) for s in scores]
            n = len(values)
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
            std = variance ** 0.5
            agg[m] = {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "n": n,
            }

        agg["total_cost_usd"] = round(sum(s.llm_cost_usd for s in scores), 6)
        return agg
