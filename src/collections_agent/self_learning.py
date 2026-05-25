"""Self-learning loop — autonomous prompt improvement.

Pipeline per iteration:
  1. Run N full 3-agent system evaluations with current prompts.
  2. Score them at the SYSTEM level (continuity across handoffs).
  3. Propose a prompt mutation for the target agent (LLM-generated).
  4. Run N system evaluations with the candidate prompt.
  5. Compare old vs new scores with a statistical significance test.
  6. If significant improvement AND compliance preserved → adopt.
  7. Otherwise → reject.
  8. Log everything for the audit trail.

System-level evaluation means running Agent 1 → handoff → Agent 2 →
handoff → Agent 3 as a pipeline, scoring the full interaction.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from collections_agent.borrower_harness import PERSONAS, run_simulated_conversation
from collections_agent.compliance_checker import ComplianceChecker
from collections_agent.evaluator import ConversationEvaluator, ConversationScore
from collections_agent.llm_adapter import GrokAdapter, get_adapter
from collections_agent.models import BorrowerCase
from collections_agent.prompt_manager import PromptManager
from collections_agent.token_counter import (
    MAX_AGENT_CONTEXT_TOKENS,
    MAX_HANDOFF_TOKENS,
    count_tokens,
    truncate_to_budget,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "self_learning"


@dataclass
class IterationResult:
    """Audit record for one self-learning iteration."""
    iteration: int
    agent_name: str
    old_version: str
    new_version: str | None
    adopted: bool
    reason: str

    old_scores: dict = field(default_factory=dict)
    new_scores: dict = field(default_factory=dict)

    # Statistical test
    t_statistic: float = 0.0
    p_value: float = 1.0
    significant: bool = False
    threshold: float = 0.05

    # Compliance
    compliance_preserved: bool = True
    protected_sections_intact: bool = True

    # Cost
    iteration_cost_usd: float = 0.0


class SelfLearningLoop:
    """Runs the self-learning loop for a single agent.

    Critically, evaluation runs the FULL 3-agent pipeline, not
    isolated agents.  When mutating agent2's prompt, we still run
    agent1 → handoff → agent2(mutated) → handoff → agent3 to
    verify that the mutation doesn't break system-level behaviour.
    """

    MUTATION_PROMPT = (
        "You are an expert prompt engineer for a debt-collections AI system. "
        "Below is the current system prompt for {agent_name} and its evaluation scores.\n\n"
        "Your goal: propose a SMALL, targeted modification to improve the weakest metric "
        "while preserving compliance. Do NOT rewrite the entire prompt. "
        "Change at most 2–3 sentences.\n\n"
        "IMPORTANT: Sections between <!-- PROTECTED --> and <!-- /PROTECTED --> "
        "MUST NOT be modified. These contain compliance-critical rules.\n\n"
        "Return ONLY the complete new prompt text (not a diff). No commentary."
    )

    SUMMARIZE_PROMPT = (
        "Compact this conversation into a HANDOFF_STATE capsule for the next agent. "
        "Do not invent facts. Preserve identity verification status, financial situation, "
        "offers/terms, borrower objections, compliance flags, timeline order, and what "
        "the next agent must not re-ask. Use terse key:value lines. Keep under 500 tokens."
    )

    def __init__(
        self,
        *,
        conversations_per_eval: int = 10,
        significance_threshold: float = 0.05,
        min_improvement: float = 0.02,
        adapter: GrokAdapter | None = None,
    ) -> None:
        self.conversations_per_eval = conversations_per_eval
        self.significance_threshold = significance_threshold
        self.min_improvement = min_improvement
        self.adapter = adapter or get_adapter()
        self.evaluator = ConversationEvaluator(self.adapter)
        self.prompt_mgr = PromptManager()
        self.cc = ComplianceChecker()
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        agent_name: str,
        case_template: BorrowerCase,
        iterations: int = 5,
        budget_usd: float = 20.0,
    ) -> list[IterationResult]:
        """Run *iterations* improvement cycles for *agent_name*."""
        results: list[IterationResult] = []
        total_cost = 0.0

        for i in range(1, iterations + 1):
            if total_cost >= budget_usd:
                logger.warning("Budget exhausted ($%.2f/$%.2f). Stopping.", total_cost, budget_usd)
                break

            logger.info("── Self-learning iteration %d/%d for %s ──", i, iterations, agent_name)
            result = self._one_iteration(agent_name, case_template, i)
            results.append(result)
            total_cost += result.iteration_cost_usd
            logger.info(
                "Iteration %d: adopted=%s reason=%s cost=$%.4f",
                i, result.adopted, result.reason, result.iteration_cost_usd,
            )

        # Persist audit trail
        self._save_audit(agent_name, results)
        return results

    # ------------------------------------------------------------------
    # Single iteration
    # ------------------------------------------------------------------

    def _one_iteration(
        self,
        agent_name: str,
        case: BorrowerCase,
        iteration: int,
    ) -> IterationResult:
        current_prompt = self.prompt_mgr.load_prompt(agent_name)
        old_version = self._current_version_id(agent_name)

        # 1. Evaluate current prompt at system level
        old_scores = self._evaluate_system(agent_name, case)
        old_agg = self.evaluator.aggregate(old_scores)

        # 2. Propose mutation for the target agent
        candidate_prompt = self._propose_mutation(agent_name, current_prompt, old_agg)

        # 3. Validate candidate
        candidate_tokens = count_tokens(candidate_prompt)
        if candidate_tokens > MAX_AGENT_CONTEXT_TOKENS:
            return IterationResult(
                iteration=iteration,
                agent_name=agent_name,
                old_version=old_version,
                new_version=None,
                adopted=False,
                reason=f"Candidate exceeds token budget ({candidate_tokens}/{MAX_AGENT_CONTEXT_TOKENS})",
                old_scores=old_agg,
            )

        # Check protected sections
        if not self.prompt_mgr.validate_protected_preserved(current_prompt, candidate_prompt):
            return IterationResult(
                iteration=iteration,
                agent_name=agent_name,
                old_version=old_version,
                new_version=None,
                adopted=False,
                reason="Protected sections were modified",
                old_scores=old_agg,
                protected_sections_intact=False,
            )

        # 4. Temporarily swap the prompt and evaluate the full system
        # Save candidate as inactive version to test it
        temp_version = self.prompt_mgr.save_version(
            agent_name,
            candidate_prompt,
            {"status": "testing"},
            activate=False,
        )

        # Evaluate with candidate prompt (inject it during system eval)
        new_scores = self._evaluate_system(
            agent_name, case, override_prompt={agent_name: candidate_prompt}
        )
        new_agg = self.evaluator.aggregate(new_scores)

        # 5. Statistical test
        old_composites = [s.composite_score for s in old_scores]
        new_composites = [s.composite_score for s in new_scores]
        t_stat, p_val = self._welch_t_test(old_composites, new_composites)
        significant = p_val < self.significance_threshold

        old_mean = old_agg.get("composite_score", {}).get("mean", 0)
        new_mean = new_agg.get("composite_score", {}).get("mean", 0)
        improvement = new_mean - old_mean

        # 6. Check compliance in candidate scores
        any_violations = any(s.compliance_score < 1.0 for s in new_scores)

        # 7. Decision
        adopted = significant and improvement >= self.min_improvement and not any_violations

        # Compute iteration cost
        iter_cost = sum(s.llm_cost_usd for s in old_scores) + sum(s.llm_cost_usd for s in new_scores)

        if adopted:
            reason = f"Significant improvement: {old_mean:.4f} → {new_mean:.4f} (p={p_val:.4f})"
            self.prompt_mgr.save_version(
                agent_name,
                candidate_prompt,
                {"old_agg": old_agg, "new_agg": new_agg, "p_value": p_val, "t_stat": t_stat},
                activate=True,
            )
        else:
            if not significant:
                reason = f"Not significant: p={p_val:.4f} > {self.significance_threshold}"
            elif improvement < self.min_improvement:
                reason = f"Improvement too small: {improvement:.4f} < {self.min_improvement}"
            elif any_violations:
                reason = "Candidate introduced compliance violations"
            else:
                reason = "Rejected"

        return IterationResult(
            iteration=iteration,
            agent_name=agent_name,
            old_version=old_version,
            new_version=temp_version,
            adopted=adopted,
            reason=reason,
            old_scores=old_agg,
            new_scores=new_agg,
            t_statistic=t_stat,
            p_value=p_val,
            significant=significant,
            threshold=self.significance_threshold,
            compliance_preserved=not any_violations,
            protected_sections_intact=True,
            iteration_cost_usd=iter_cost,
        )

    # ------------------------------------------------------------------
    # System-level evaluation
    # ------------------------------------------------------------------

    def _evaluate_system(
        self,
        target_agent: str,
        case: BorrowerCase,
        override_prompt: dict[str, str] | None = None,
    ) -> list[ConversationScore]:
        """Run N full 3-agent pipelines and score the system interaction.

        If *override_prompt* is provided, the specified agent's prompt
        is replaced with the candidate (for A/B testing).
        """
        scores: list[ConversationScore] = []
        personas = list(PERSONAS.keys())
        override_prompt = override_prompt or {}

        for i in range(self.conversations_per_eval):
            persona = personas[i % len(personas)]
            try:
                score = self._run_full_pipeline(
                    case, persona, i, override_prompt
                )
                scores.append(score)
            except Exception:
                logger.exception("System eval failed for persona=%s i=%d", persona, i)

        return scores

    def _run_full_pipeline(
        self,
        case: BorrowerCase,
        persona: str,
        run_index: int,
        override_prompt: dict[str, str],
    ) -> ConversationScore:
        """Run Agent 1 → handoff → Agent 2 → handoff → Agent 3 and score."""
        all_transcripts: list[dict[str, str]] = []
        total_compliance_violations: list[str] = []
        total_turns = 0

        # ── Agent 1: Assessment Chat ──
        a1_prompt = self._compose_agent_prompt(
            "agent1",
            case,
            base_prompt=override_prompt.get("agent1"),
        )
        conv1 = run_simulated_conversation(
            agent_system_prompt=a1_prompt,
            agent_name="agent1",
            persona=persona,
            adapter=self.adapter,
            seed=42 + run_index,
        )
        all_transcripts.extend(conv1.transcript)
        total_compliance_violations.extend(conv1.compliance_violations)
        total_turns += conv1.turn_count

        # ── Handoff 1: Agent 1 → Agent 2 ──
        handoff1 = self._summarize_transcript(conv1.transcript)

        # ── Agent 2: Resolution Voice ──
        a2_base_prompt = self._compose_agent_prompt(
            "agent2",
            case,
            handoff_text=handoff1,
            base_prompt=override_prompt.get("agent2"),
        )
        conv2 = run_simulated_conversation(
            agent_system_prompt=a2_base_prompt,
            agent_name="agent2",
            persona=persona,
            adapter=self.adapter,
            seed=42 + run_index + 100,
        )
        all_transcripts.extend(conv2.transcript)
        total_compliance_violations.extend(conv2.compliance_violations)
        total_turns += conv2.turn_count

        # ── Handoff 2: Agent 1+2 → Agent 3 ──
        combined_transcript = conv1.transcript + conv2.transcript
        handoff2 = self._summarize_transcript(combined_transcript)

        # ── Agent 3: Final Notice Chat ──
        a3_prompt = self._compose_agent_prompt(
            "agent3",
            case,
            handoff_text=handoff2,
            base_prompt=override_prompt.get("agent3"),
        )
        conv3 = run_simulated_conversation(
            agent_system_prompt=a3_prompt,
            agent_name="agent3",
            persona=persona,
            adapter=self.adapter,
            seed=42 + run_index + 200,
        )
        all_transcripts.extend(conv3.transcript)
        total_compliance_violations.extend(conv3.compliance_violations)
        total_turns += conv3.turn_count

        # ── Score the full system interaction ──
        from collections_agent.borrower_harness import ConversationResult
        system_conv = ConversationResult(
            agent_name="system",
            persona=persona,
            transcript=all_transcripts,
            borrower_stop_requested=conv1.borrower_stop_requested or conv2.borrower_stop_requested or conv3.borrower_stop_requested,
            borrower_in_distress=conv1.borrower_in_distress or conv2.borrower_in_distress or conv3.borrower_in_distress,
            compliance_violations=total_compliance_violations,
            turn_count=total_turns,
        )

        score = self.evaluator.score_conversation(
            system_conv,
            conversation_id=f"system_{persona}_{run_index}",
        )
        return score

    def _get_agent_prompt(
        self,
        agent_name: str,
        case: BorrowerCase,
        handoff_text: str | None = None,
    ) -> str:
        """Load agent prompt and inject case context + optional handoff."""
        return self._compose_agent_prompt(agent_name, case, handoff_text=handoff_text)

    def _compose_agent_prompt(
        self,
        agent_name: str,
        case: BorrowerCase,
        handoff_text: str | None = None,
        base_prompt: str | None = None,
    ) -> str:
        """Compose an agent prompt with the same runtime context for A/B tests.

        Candidate prompts replace only the base prompt text. They still receive
        the live case context, handoff summary, and Agent 2 settlement policy.
        """
        prompt = base_prompt if base_prompt is not None else self.prompt_mgr.load_prompt(agent_name)
        context = (
            f"\n\n## Current Case (injected at runtime)\n"
            f"- Borrower ID: {case.borrower_id}\n"
            f"- Company: {case.company_name}\n"
            f"- Debt: ${case.debt_amount_cents / 100:,.2f}\n"
            f"- Account last 4: {case.account_last4}\n"
        )
        if agent_name == "agent2":
            # Inject settlement policy
            policy_path = Path(__file__).resolve().parent.parent.parent / "policy" / "settlement_policy.json"
            if policy_path.exists():
                policy = json.loads(policy_path.read_text())
                context += f"\n## Settlement Policy\n```json\n{json.dumps(policy, indent=2)}\n```\n"
        if handoff_text:
            context += f"\n## Prior Context (handoff)\n{handoff_text}\n"
        return prompt + context

    def _summarize_transcript(self, transcript: list[dict[str, str]]) -> str:
        """Summarize a transcript for handoff (≤500 tokens)."""
        transcript_text = json.dumps(transcript, indent=2)
        resp = self.adapter.chat(
            self.SUMMARIZE_PROMPT,
            [{"role": "user", "content": transcript_text}],
        )
        summary = truncate_to_budget(resp.text.strip(), MAX_HANDOFF_TOKENS)
        return summary

    # ------------------------------------------------------------------
    # Other helpers
    # ------------------------------------------------------------------

    def _propose_mutation(
        self,
        agent_name: str,
        current_prompt: str,
        eval_agg: dict,
    ) -> str:
        """Use the LLM to propose a small prompt improvement."""
        mutation_system = self.MUTATION_PROMPT.format(agent_name=agent_name)
        user_msg = (
            f"## Current Prompt\n\n{current_prompt}\n\n"
            f"## Current Scores\n\n{json.dumps(eval_agg, indent=2)}"
        )
        resp = self.adapter.chat(mutation_system, [{"role": "user", "content": user_msg}])
        return resp.text.strip()

    def _current_version_id(self, agent_name: str) -> str:
        versions = self.prompt_mgr.list_versions(agent_name)
        active = [v for v in versions if v.is_active]
        return active[0].version_id if active else "base"

    @staticmethod
    def _welch_t_test(a: list[float], b: list[float]) -> tuple[float, float]:
        """Welch's t-test for unequal variances.  Returns (t_stat, p_value)."""
        n1, n2 = len(a), len(b)
        if n1 < 2 or n2 < 2:
            return 0.0, 1.0

        m1 = sum(a) / n1
        m2 = sum(b) / n2
        v1 = sum((x - m1) ** 2 for x in a) / (n1 - 1)
        v2 = sum((x - m2) ** 2 for x in b) / (n2 - 1)

        se = math.sqrt(v1 / n1 + v2 / n2) if (v1 + v2) > 0 else 1e-9
        t_stat = (m2 - m1) / se

        # Welch–Satterthwaite degrees of freedom
        num = (v1 / n1 + v2 / n2) ** 2
        denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        df = num / denom if denom > 0 else 1

        # Approximate p-value using normal distribution for large df
        p_value = 2 * (1 - _normal_cdf(abs(t_stat)))
        return round(t_stat, 4), round(p_value, 4)

    def _save_audit(self, agent_name: str, results: list[IterationResult]) -> None:
        agent_dir = DATA_DIR / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = agent_dir / f"audit_{ts}.json"
        path.write_text(json.dumps([asdict(r) for r in results], indent=2, default=str))
        logger.info("Saved self-learning audit to %s", path)


def _normal_cdf(x: float) -> float:
    """Approximate CDF of the standard normal distribution."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
