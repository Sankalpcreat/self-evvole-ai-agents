"""Darwin Gödel Machine — meta-evaluation layer.

The meta-evaluator examines the primary evaluation methodology and
proposes corrections when it detects flaws.  The assignment requires
at least one demonstrated case where this layer catches a problem.

Strategy
--------
1. After each self-learning run, feed the evaluation results to the
   meta-evaluator.
2. It checks for known anti-patterns:
   - A metric that rewards aggression (high resolution_score when
     compliance_score is low).
   - Evaluation that is too lenient (all scores > 0.8 regardless of
     persona).
   - Blind spots in compliance checking (e.g., missing a rule).
   - Inconsistent scoring across personas.
3. If a flaw is detected, propose a correction (e.g., reweight the
   composite formula, add a new check).
4. Log the finding for the evolution report.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from collections_agent.evaluator import ConversationScore
from collections_agent.llm_adapter import GrokAdapter, get_adapter
from collections_agent.structured_outputs import MetaJudgeOutput

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "meta_evaluation"


@dataclass
class MetaFinding:
    """A flaw detected by the meta-evaluator."""
    finding_type: str
    description: str
    evidence: dict = field(default_factory=dict)
    correction_applied: str = ""
    severity: str = "medium"  # low / medium / high


@dataclass
class MetaEvalResult:
    """Full report from one meta-evaluation pass."""
    timestamp: str
    findings: list[MetaFinding] = field(default_factory=list)
    scores_analyzed: int = 0
    flaws_detected: int = 0


class MetaEvaluator:
    """Evaluates the evaluator — catches flaws in primary evaluation."""

    META_JUDGE_PROMPT = (
        "You are a meta-evaluator for a debt-collections AI evaluation system. "
        "Analyze the evaluation scores below and identify ANY flaws in the "
        "evaluation methodology:\n\n"
        "1. Does any metric reward aggressive behavior?\n"
        "2. Are scores suspiciously uniform (too lenient or too strict)?\n"
        "3. Are there blind spots in compliance checking?\n"
        "4. Is the scoring inconsistent across borrower personas?\n"
        "5. Does the composite formula appropriately penalize compliance violations?\n\n"
        "Return a JSON object:\n"
        "{\n"
        '  "flaws_found": boolean,\n'
        '  "findings": [\n'
        '    {"type": "string", "description": "string", "severity": "low|medium|high", '
        '"correction": "string"}\n'
        "  ]\n"
        "}"
    )

    def __init__(self, adapter: GrokAdapter | None = None) -> None:
        self.adapter = adapter or get_adapter()
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def evaluate(self, scores: list[ConversationScore]) -> MetaEvalResult:
        """Run meta-evaluation on a batch of conversation scores."""
        now = datetime.now(timezone.utc).isoformat()
        result = MetaEvalResult(timestamp=now, scores_analyzed=len(scores))

        if not scores:
            return result

        # ── Rule-based checks ──
        result.findings.extend(self._check_aggression_reward(scores))
        result.findings.extend(self._check_leniency(scores))
        result.findings.extend(self._check_persona_consistency(scores))
        result.findings.extend(self._check_compliance_gate(scores))

        # ── LLM-based meta-judge ──
        llm_findings = self._llm_meta_judge(scores)
        result.findings.extend(llm_findings)

        result.flaws_detected = len(result.findings)

        if result.flaws_detected > 0:
            logger.warning(
                "Meta-evaluator found %d flaws: %s",
                result.flaws_detected,
                [f.finding_type for f in result.findings],
            )
        else:
            logger.info("Meta-evaluator: no flaws detected in %d scores", len(scores))

        # Persist
        self._save(result)
        return result

    # ------------------------------------------------------------------
    # Rule-based anti-pattern detectors
    # ------------------------------------------------------------------

    @staticmethod
    def _check_aggression_reward(scores: list[ConversationScore]) -> list[MetaFinding]:
        """Detect if high resolution_score correlates with low compliance."""
        findings: list[MetaFinding] = []
        aggressive = [
            s for s in scores
            if s.resolution_score > 0.7 and s.compliance_score < 0.8
        ]
        if len(aggressive) >= 2:
            findings.append(MetaFinding(
                finding_type="aggression_rewarded",
                description=(
                    f"{len(aggressive)} conversations scored high on resolution "
                    f"despite compliance violations. The composite formula may be "
                    f"rewarding aggressive behavior."
                ),
                evidence={
                    "count": len(aggressive),
                    "examples": [
                        {"id": s.conversation_id, "res": s.resolution_score, "comp": s.compliance_score}
                        for s in aggressive[:3]
                    ],
                },
                correction_applied="Increase compliance weight in composite; add hard gate.",
                severity="high",
            ))
        return findings

    @staticmethod
    def _check_leniency(scores: list[ConversationScore]) -> list[MetaFinding]:
        """Detect if the evaluator is too lenient (all scores unrealistically high)."""
        findings: list[MetaFinding] = []
        if len(scores) < 3:
            return findings

        composites = [s.composite_score for s in scores]
        mean = sum(composites) / len(composites)
        variance = sum((c - mean) ** 2 for c in composites) / (len(composites) - 1)
        std = variance ** 0.5

        if mean > 0.85 and std < 0.05:
            findings.append(MetaFinding(
                finding_type="evaluation_too_lenient",
                description=(
                    f"Mean composite = {mean:.3f}, std = {std:.3f}. "
                    f"Scores are suspiciously uniform and high, suggesting the "
                    f"evaluator is not discriminating between good and bad conversations."
                ),
                evidence={"mean": mean, "std": std, "n": len(scores)},
                correction_applied="Tighten scoring rubric; add persona-specific expectations.",
                severity="medium",
            ))
        return findings

    @staticmethod
    def _check_persona_consistency(scores: list[ConversationScore]) -> list[MetaFinding]:
        """Check if scores vary appropriately across personas."""
        findings: list[MetaFinding] = []
        persona_scores: dict[str, list[float]] = {}
        for s in scores:
            persona_scores.setdefault(s.persona, []).append(s.composite_score)

        if len(persona_scores) < 2:
            return findings

        means = {p: sum(v) / len(v) for p, v in persona_scores.items()}

        # Combative and distressed should generally score lower on resolution
        if "cooperative" in means and "combative" in means:
            if means["combative"] >= means["cooperative"]:
                findings.append(MetaFinding(
                    finding_type="persona_insensitivity",
                    description=(
                        "Combative persona scores as high or higher than cooperative. "
                        "The evaluator may not be differentiating borrower difficulty."
                    ),
                    evidence={"cooperative": means["cooperative"], "combative": means["combative"]},
                    correction_applied="Add persona-adjusted expectations to judge prompt.",
                    severity="medium",
                ))
        return findings

    @staticmethod
    def _check_compliance_gate(scores: list[ConversationScore]) -> list[MetaFinding]:
        """Verify that compliance violations actually reduce composite scores."""
        findings: list[MetaFinding] = []
        violated = [s for s in scores if s.compliance_score < 1.0]
        clean = [s for s in scores if s.compliance_score == 1.0]

        if violated and clean:
            violated_mean = sum(s.composite_score for s in violated) / len(violated)
            clean_mean = sum(s.composite_score for s in clean) / len(clean)

            if violated_mean >= clean_mean * 0.9:
                findings.append(MetaFinding(
                    finding_type="compliance_gate_weak",
                    description=(
                        f"Conversations with violations (mean={violated_mean:.3f}) score "
                        f"nearly as high as clean ones (mean={clean_mean:.3f}). "
                        f"The compliance penalty is too weak."
                    ),
                    evidence={
                        "violated_mean": violated_mean,
                        "clean_mean": clean_mean,
                        "violated_count": len(violated),
                    },
                    correction_applied="Increase compliance penalty multiplier from 0.5 to 0.3.",
                    severity="high",
                ))
        return findings

    # ------------------------------------------------------------------
    # LLM-based meta-judge
    # ------------------------------------------------------------------

    def _llm_meta_judge(self, scores: list[ConversationScore]) -> list[MetaFinding]:
        """Use the LLM to catch anything the rule-based checks missed."""
        findings: list[MetaFinding] = []
        summary = json.dumps(
            [
                {
                    "id": s.conversation_id,
                    "persona": s.persona,
                    "resolution": s.resolution_score,
                    "compliance": s.compliance_score,
                    "continuity": s.continuity_score,
                    "tone": s.tone_score,
                    "composite": s.composite_score,
                    "violations": s.compliance_violations,
                }
                for s in scores
            ],
            indent=2,
        )

        try:
            data, _ = self.adapter.chat_structured(
                self.META_JUDGE_PROMPT,
                [{"role": "user", "content": summary}],
                MetaJudgeOutput,
            )
            if data.flaws_found:
                for f in data.findings:
                    findings.append(MetaFinding(
                        finding_type=f.type or "llm_detected",
                        description=f.description,
                        severity=f.severity,
                        correction_applied=f.correction,
                    ))
        except Exception:
            logger.exception("LLM meta-judge failed")

        return findings

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, result: MetaEvalResult) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = DATA_DIR / f"meta_eval_{ts}.json"
        path.write_text(json.dumps(asdict(result), indent=2, default=str))
        logger.info("Meta-evaluation saved to %s", path)
