"""Run the full evaluation + self-learning pipeline.

Single command:
    PYTHONPATH=src python -m collections_agent.run_pipeline

This script evaluates the FULL 3-agent pipeline with real handoffs:
  1. Run Agent 1 conversation → summarise transcript → handoff (≤500 tokens)
  2. Run Agent 2 conversation WITH Agent 1's handoff context → summarise → handoff
  3. Run Agent 3 conversation WITH full handoff context
  4. Score the ENTIRE system interaction (not individual agents)
  5. Run self-learning: when mutating one agent's prompt, re-run the full
     pipeline to verify handoff continuity is preserved.
  6. Run meta-evaluation on system-level scores.
  7. Save all results to data/ with cost breakdown.

Use --seed for reproducibility and --iterations to control loop depth.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from collections_agent.borrower_harness import PERSONAS, ConversationResult, run_simulated_conversation
from collections_agent.evaluator import ConversationEvaluator, ConversationScore
from collections_agent.llm_adapter import GrokAdapter, get_adapter
from collections_agent.meta_evaluator import MetaEvaluator
from collections_agent.models import BorrowerCase, HandoffSummary
from collections_agent.prompt_manager import PromptManager
from collections_agent.self_learning import SelfLearningLoop
from collections_agent.structured_outputs import JudgeScoreOutput
from collections_agent.token_counter import MAX_HANDOFF_TOKENS, count_tokens, truncate_to_budget
from collections_agent.voice_generator import generate_agent2_demo_audio

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# ---------------------------------------------------------------------------
# Summarisation helpers
# ---------------------------------------------------------------------------

SUMMARISE_PROMPT = (
    "You are a high-recall handoff compactor for a debt-collections AI system. "
    "Convert the JSON transcript into a compact HANDOFF_STATE capsule for the next agent. "
    "Do not answer the borrower and do not invent facts.\n\n"
    "Preserve exact high-signal state:\n"
    "- identity verification status\n"
    "- debt/account facts that were discussed\n"
    "- financial situation and payment capacity\n"
    "- offers made, deadlines, objections, borrower position\n"
    "- hardship, distress, stop-contact, dispute, or compliance flags\n"
    "- what happened first vs what happened later\n"
    "- what the next agent must not re-ask\n\n"
    "Use this compact format:\n"
    "HANDOFF_STATE v1\n"
    "timeline_1: ...\n"
    "timeline_2: ...\n"
    "identity_verified: ...\n"
    "financial_situation: ...\n"
    "offers_or_terms: ...\n"
    "borrower_position: ...\n"
    "objections: ...\n"
    "compliance_flags: ...\n"
    "next_agent_instruction: ...\n\n"
    "Return ONLY the capsule text. Keep it under 500 tokens."
)


def _summarise_transcript(
    adapter: GrokAdapter,
    transcript: list[dict[str, str]],
    extra_context: str = "",
) -> HandoffSummary:
    """Summarise a conversation transcript into a ≤500-token handoff."""
    content = json.dumps(transcript, indent=2)
    if extra_context:
        content = f"{extra_context}\n\n{content}"
    resp = adapter.chat(SUMMARISE_PROMPT, [{"role": "user", "content": content}])
    summary_text = truncate_to_budget(resp.text.strip(), MAX_HANDOFF_TOKENS)
    return HandoffSummary(text=summary_text, token_count=count_tokens(summary_text))


# ---------------------------------------------------------------------------
# Pipeline dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Full pipeline result for one borrower persona run."""
    persona: str
    seed: int
    agent1_transcript: list[dict[str, str]] = field(default_factory=list)
    agent1_handoff: str = ""
    agent2_transcript: list[dict[str, str]] = field(default_factory=list)
    agent2_handoff: str = ""
    agent3_transcript: list[dict[str, str]] = field(default_factory=list)
    handoff_token_counts: list[int] = field(default_factory=list)
    compliance_violations: list[str] = field(default_factory=list)
    system_score: ConversationScore | None = None


# ---------------------------------------------------------------------------
# Default case
# ---------------------------------------------------------------------------

def _default_case() -> BorrowerCase:
    return BorrowerCase(
        borrower_id="test-borrower-001",
        company_name="Riverline Collections",
        debt_amount_cents=500000,
        account_last4="4321",
        phone_number="+1-555-0100",
        chat_thread_id="thread-test-001",
    )


def _build_case_context(case: BorrowerCase) -> str:
    """Inject minimal case context into an agent prompt."""
    return (
        f"\n\n## Current Case\n"
        f"- Borrower ID: {case.borrower_id}\n"
        f"- Company: {case.company_name}\n"
        f"- Debt: ${case.debt_amount_cents / 100:,.2f}\n"
        f"- Account last 4: {case.account_last4}\n"
    )


def _build_settlement_policy_context() -> str:
    """Inject the local settlement policy into Agent 2 prompts."""
    policy_path = Path(__file__).resolve().parent.parent.parent / "policy" / "settlement_policy.json"
    if not policy_path.exists():
        return ""
    policy = json.loads(policy_path.read_text())
    return f"\n\n## Settlement Policy\n```json\n{json.dumps(policy, indent=2)}\n```\n"


# ---------------------------------------------------------------------------
# Full pipeline runner
# ---------------------------------------------------------------------------

def run_full_pipeline(
    adapter: GrokAdapter,
    case: BorrowerCase,
    persona: str,
    pm: PromptManager,
    seed: int,
) -> PipelineResult:
    """Run the complete Agent 1 → 2 → 3 pipeline for one persona.

    Returns a PipelineResult with all transcripts, handoffs, and token counts.
    """
    result = PipelineResult(persona=persona, seed=seed)
    case_ctx = _build_case_context(case)

    # ── Stage 1: Agent 1 (Assessment) ──
    agent1_prompt = pm.load_prompt("agent1") + case_ctx
    conv1 = run_simulated_conversation(
        agent_system_prompt=agent1_prompt,
        agent_name="agent1",
        persona=persona,
        adapter=adapter,
        seed=seed,
    )
    result.agent1_transcript = conv1.transcript
    result.compliance_violations.extend(conv1.compliance_violations)

    # Summarise Agent 1 → handoff for Agent 2
    handoff1 = _summarise_transcript(adapter, conv1.transcript)
    result.agent1_handoff = handoff1.text
    result.handoff_token_counts.append(handoff1.token_count)
    logger.info(
        "Agent 1 handoff: %d tokens (budget %d)",
        handoff1.token_count, MAX_HANDOFF_TOKENS,
    )

    # ── Stage 2: Agent 2 (Resolution) WITH Agent 1 handoff ──
    agent2_prompt = pm.load_prompt("agent2") + case_ctx + _build_settlement_policy_context()
    handoff_injection = (
        f"\n\n## Handoff from Agent 1 (Assessment)\n{handoff1.text}"
    )
    agent2_full_prompt = agent2_prompt + handoff_injection

    conv2 = run_simulated_conversation(
        agent_system_prompt=agent2_full_prompt,
        agent_name="agent2",
        persona=persona,
        adapter=adapter,
        seed=seed + 100,
    )
    result.agent2_transcript = conv2.transcript
    result.compliance_violations.extend(conv2.compliance_violations)

    # Summarise full history (Agent 1 handoff + Agent 2 transcript) → handoff for Agent 3
    combined_context = (
        f"## Prior context (Agent 1 assessment summary):\n{handoff1.text}"
    )
    handoff2 = _summarise_transcript(
        adapter, conv2.transcript, extra_context=combined_context,
    )
    result.agent2_handoff = handoff2.text
    result.handoff_token_counts.append(handoff2.token_count)
    logger.info(
        "Agent 2 handoff: %d tokens (budget %d)",
        handoff2.token_count, MAX_HANDOFF_TOKENS,
    )

    # ── Stage 3: Agent 3 (Final Notice) WITH full handoff context ──
    agent3_prompt = pm.load_prompt("agent3") + case_ctx
    handoff_injection3 = (
        f"\n\n## Handoff from prior stages\n{handoff2.text}"
    )
    agent3_full_prompt = agent3_prompt + handoff_injection3

    conv3 = run_simulated_conversation(
        agent_system_prompt=agent3_full_prompt,
        agent_name="agent3",
        persona=persona,
        adapter=adapter,
        seed=seed + 200,
    )
    result.agent3_transcript = conv3.transcript
    result.compliance_violations.extend(conv3.compliance_violations)

    return result


# ---------------------------------------------------------------------------
# System-level scorer
# ---------------------------------------------------------------------------

SYSTEM_JUDGE_PROMPT = (
    "You are an expert evaluator for a multi-agent debt-collections system. "
    "You are given the FULL interaction across three sequential agents:\n"
    "- Agent 1 (Assessment): identifies/verifies the borrower and assesses situation\n"
    "- Agent 2 (Resolution): negotiates payment via phone call\n"
    "- Agent 3 (Final Notice): sends final written notice with consequences\n\n"
    "Score the ENTIRE system on these dimensions (0.0 to 1.0 each):\n\n"
    "1. resolution_score – Did the system as a whole make progress toward "
    "resolving the debt across all three stages?\n"
    "2. continuity_score – Does the experience feel continuous? Does each agent "
    "pick up where the prior left off without repeating questions or losing context?\n"
    "3. tone_score – Is the tone appropriate at each stage and does it escalate "
    "naturally (clinical → transactional → consequence-driven)?\n"
    "   IMPORTANT: Do not reward the system for matching borrower hostility. "
    "Professional de-escalation should score higher than aggressive mirroring.\n"
    "4. information_gathering – Did the system collect all required information "
    "across the pipeline?\n\n"
    "PERSONA-ADJUSTED EXPECTATIONS:\n"
    "- cooperative: resolution should be achievable, so low resolution needs justification.\n"
    "- combative: resolution is difficult, but professionalism and no escalation matter.\n"
    "- evasive: information gathering is the main challenge.\n"
    "- confused: patience, clarity, and not repeating handoff facts matter most.\n"
    "- distressed: hardship handling and non-pressure are compliance-critical.\n\n"
    "Return ONLY a JSON object with these four float fields. No commentary."
)


def score_full_pipeline(
    adapter: GrokAdapter,
    pipeline: PipelineResult,
    evaluator: ConversationEvaluator,
    prompt_version: str = "base",
    conversation_id: str | None = None,
) -> ConversationScore:
    """Score the full 3-agent pipeline as a system."""
    cid = conversation_id or f"system_{pipeline.persona}_{pipeline.seed}"

    total_turns = 0
    for transcript_attr in ("agent1_transcript", "agent2_transcript", "agent3_transcript"):
        # We don't have separate ConversationResult objects here, so we count turns
        transcript = getattr(pipeline, transcript_attr)
        total_turns += len([m for m in transcript if m.get("role") == "agent"])

    score = ConversationScore(
        conversation_id=cid,
        agent_name="system",
        persona=pipeline.persona,
        prompt_version=prompt_version,
        turn_count=total_turns,
        compliance_violations=list(pipeline.compliance_violations),
    )
    if pipeline.compliance_violations:
        score.compliance_score = max(0.0, 1.0 - 0.25 * len(pipeline.compliance_violations))

    # Meta-eval correction: the full-pipeline scorer must re-run compliance over
    # saved transcripts too, otherwise repaired first-pass violations can vanish
    # from the final score and the composite penalty never gets exercised.
    synthetic_violations: list[str] = []
    for agent_name, transcript_attr in (
        ("agent1", "agent1_transcript"),
        ("agent2", "agent2_transcript"),
        ("agent3", "agent3_transcript"),
    ):
        transcript = getattr(pipeline, transcript_attr)
        conv = ConversationResult(
            agent_name=agent_name,
            persona=pipeline.persona,
            transcript=transcript,
            turn_count=len([m for m in transcript if m.get("role") == "agent"]),
        )
        synthetic_violations.extend(
            f"{agent_name}_{v}" for v in evaluator._synthetic_compliance_check(conv)
        )

    if synthetic_violations:
        all_violations = list(score.compliance_violations) + sorted(set(synthetic_violations))
        score.compliance_violations = all_violations
        score.compliance_score = max(0.0, 1.0 - 0.25 * len(all_violations))
        logger.info(
            "System synthetic compliance check found %d violations for %s",
            len(synthetic_violations), cid,
        )

    # Build a combined transcript for the LLM judge
    system_transcript = {
        "agent1_transcript": pipeline.agent1_transcript,
        "agent1_to_agent2_handoff": pipeline.agent1_handoff,
        "agent2_transcript": pipeline.agent2_transcript,
        "agent2_to_agent3_handoff": pipeline.agent2_handoff,
        "agent3_transcript": pipeline.agent3_transcript,
        "handoff_token_counts": pipeline.handoff_token_counts,
    }
    transcript_text = json.dumps(system_transcript, indent=2)

    # LLM-as-judge for the FULL system
    try:
        judge_data, usage = adapter.chat_structured(
            SYSTEM_JUDGE_PROMPT,
            [{"role": "user", "content": f"Persona: {pipeline.persona}\n\n{transcript_text}"}],
            JudgeScoreOutput,
        )
        score.resolution_score = judge_data.resolution_score
        score.continuity_score = judge_data.continuity_score
        score.tone_score = judge_data.tone_score
        score.information_gathering = judge_data.information_gathering
        score.llm_cost_usd = usage.cost_usd
    except Exception:
        logger.exception("System-level LLM judge failed for %s", cid)
        score.resolution_score = 0.5
        score.continuity_score = 0.5
        score.tone_score = 0.5
        score.information_gathering = 0.5

    # Budget adherence: check handoff tokens are within limits
    over_budget = any(tc > MAX_HANDOFF_TOKENS for tc in pipeline.handoff_token_counts)
    score.budget_adherence = 0.0 if over_budget else 1.0

    score.compute_composite()
    return score


# ---------------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------------

def run_baseline_evaluation(
    adapter: GrokAdapter,
    case: BorrowerCase,
    seed: int,
    repeats_per_persona: int = 1,
) -> list[ConversationScore]:
    """Run full pipeline baseline across all personas and score at system level."""
    pm = PromptManager()
    evaluator = ConversationEvaluator(adapter)
    all_scores: list[ConversationScore] = []

    logger.info("── Baseline evaluation: full 3-agent pipeline ──")
    for i, persona in enumerate(PERSONAS):
        for repeat in range(repeats_per_persona):
            run_seed = seed + i * 100 + repeat
            logger.info(
                "  Running full pipeline for persona: %s repeat=%d/%d seed=%d",
                persona,
                repeat + 1,
                repeats_per_persona,
                run_seed,
            )
            pipeline = run_full_pipeline(
                adapter=adapter,
                case=case,
                persona=persona,
                pm=pm,
                seed=run_seed,
            )
            score = score_full_pipeline(
                adapter=adapter,
                pipeline=pipeline,
                evaluator=evaluator,
                prompt_version="base",
                conversation_id=f"baseline_system_{persona}_r{repeat + 1}",
            )
            all_scores.append(score)
            logger.info(
                "  system/%s/r%d: composite=%.3f continuity=%.3f resolution=%.3f budget=%.1f",
                persona, repeat + 1, score.composite_score, score.continuity_score,
                score.resolution_score, score.budget_adherence,
            )

    # Save baseline
    run_dir = evaluator.save_scores(all_scores, run_id="baseline")
    agg = evaluator.aggregate(all_scores)
    logger.info("Baseline aggregate: %s", json.dumps(agg, indent=2))
    print(f"\n✅ Baseline saved to {run_dir}")
    print(f"   Scores: {len(all_scores)} full-pipeline conversations")
    print(f"   Cost: ${agg.get('total_cost_usd', 0):.4f}")
    return all_scores


# ---------------------------------------------------------------------------
# System-level self-learning
# ---------------------------------------------------------------------------

def run_self_learning(
    adapter: GrokAdapter,
    case: BorrowerCase,
    iterations: int,
    conversations_per_eval: int,
    budget: float,
) -> None:
    """Run the self-learning loop, evaluating at the SYSTEM level.

    When mutating one agent's prompt, the full pipeline is re-run so that
    handoff continuity is validated end-to-end.
    """
    pm = PromptManager()
    evaluator = ConversationEvaluator(adapter)
    total_cost = 0.0
    personas = list(PERSONAS.keys())

    for agent_name in ("agent1", "agent2", "agent3"):
        logger.info("══ System-level self-learning for %s (%d iterations) ══", agent_name, iterations)
        adopted_count = 0

        for iteration in range(1, iterations + 1):
            if total_cost >= budget:
                logger.warning("Budget exhausted ($%.2f/$%.2f). Stopping.", total_cost, budget)
                break

            logger.info("── Iteration %d/%d: mutating %s ──", iteration, iterations, agent_name)

            # 1. Run full pipeline with CURRENT prompts → baseline scores
            old_scores: list[ConversationScore] = []
            for j in range(min(conversations_per_eval, len(personas))):
                persona = personas[j % len(personas)]
                pipeline = run_full_pipeline(
                    adapter=adapter, case=case, persona=persona,
                    pm=pm, seed=42 + iteration * 100 + j,
                )
                score = score_full_pipeline(
                    adapter=adapter, pipeline=pipeline, evaluator=evaluator,
                    conversation_id=f"learn_{agent_name}_old_{iteration}_{persona}",
                )
                old_scores.append(score)

            old_agg = evaluator.aggregate(old_scores)
            old_mean = old_agg.get("composite_score", {}).get("mean", 0)

            # 2. Propose a mutation for the target agent
            current_prompt = pm.load_prompt(agent_name)
            mutation_prompt = (
                f"You are an expert prompt engineer for a debt-collections AI system. "
                f"Below is the current system prompt for {agent_name} and its "
                f"SYSTEM-LEVEL evaluation scores (across the full 3-agent pipeline).\n\n"
                f"Your goal: propose a SMALL, targeted modification to improve the "
                f"weakest metric while preserving compliance and handoff continuity. "
                f"Change at most 2–3 sentences.\n\n"
                f"IMPORTANT: Sections between <!-- PROTECTED --> and <!-- /PROTECTED --> "
                f"MUST NOT be modified.\n\n"
                f"Return ONLY the complete new prompt text. No commentary."
            )
            user_msg = (
                f"## Current Prompt\n\n{current_prompt}\n\n"
                f"## System-Level Scores\n\n{json.dumps(old_agg, indent=2)}"
            )
            resp = adapter.chat(mutation_prompt, [{"role": "user", "content": user_msg}])
            candidate_prompt = resp.text.strip()

            # 3. Validate protected sections
            if not pm.validate_protected_preserved(current_prompt, candidate_prompt):
                logger.info("Iteration %d: rejected — protected sections modified", iteration)
                continue

            # 4. Run full pipeline with CANDIDATE prompt → new scores
            # Temporarily swap the prompt
            original_load = pm.load_prompt

            def _patched_load(name: str, _agent=agent_name, _candidate=candidate_prompt) -> str:
                if name == _agent:
                    return _candidate
                return original_load(name)

            pm.load_prompt = _patched_load  # type: ignore[assignment]

            new_scores: list[ConversationScore] = []
            try:
                for j in range(min(conversations_per_eval, len(personas))):
                    persona = personas[j % len(personas)]
                    pipeline = run_full_pipeline(
                        adapter=adapter, case=case, persona=persona,
                        pm=pm, seed=42 + iteration * 100 + j,
                    )
                    score = score_full_pipeline(
                        adapter=adapter, pipeline=pipeline, evaluator=evaluator,
                        conversation_id=f"learn_{agent_name}_new_{iteration}_{persona}",
                    )
                    new_scores.append(score)
            finally:
                pm.load_prompt = original_load  # type: ignore[assignment]

            new_agg = evaluator.aggregate(new_scores)
            new_mean = new_agg.get("composite_score", {}).get("mean", 0)

            improvement = new_mean - old_mean
            iter_cost = sum(s.llm_cost_usd for s in old_scores) + sum(s.llm_cost_usd for s in new_scores)
            total_cost += iter_cost

            # 5. Decision: adopt if improvement ≥ 0.02 and continuity not degraded
            old_continuity = old_agg.get("continuity_score", {}).get("mean", 0)
            new_continuity = new_agg.get("continuity_score", {}).get("mean", 0)
            continuity_ok = new_continuity >= old_continuity - 0.05  # allow tiny dip

            if improvement >= 0.02 and continuity_ok:
                pm.save_version(
                    agent_name, candidate_prompt,
                    {"old_agg": old_agg, "new_agg": new_agg, "improvement": improvement},
                    activate=True,
                )
                adopted_count += 1
                logger.info(
                    "Iteration %d: ADOPTED — composite %.4f → %.4f (+%.4f), continuity %.4f → %.4f",
                    iteration, old_mean, new_mean, improvement, old_continuity, new_continuity,
                )
            else:
                pm.save_version(
                    agent_name, candidate_prompt,
                    {"old_agg": old_agg, "new_agg": new_agg, "rejected": True},
                    activate=False,
                )
                reason = "insufficient improvement" if improvement < 0.02 else "continuity degraded"
                logger.info(
                    "Iteration %d: REJECTED (%s) — composite %.4f → %.4f, continuity %.4f → %.4f",
                    iteration, reason, old_mean, new_mean, old_continuity, new_continuity,
                )

        print(f"\n✅ {agent_name}: {adopted_count}/{iterations} iterations adopted, cost=${total_cost:.4f}")


# ---------------------------------------------------------------------------
# Meta-evaluation
# ---------------------------------------------------------------------------

def run_meta_evaluation(adapter: GrokAdapter) -> None:
    """Run the meta-evaluator on baseline system-level scores."""
    meta = MetaEvaluator(adapter)

    baseline_dir = DATA_DIR / "evaluations" / "baseline"
    if not baseline_dir.exists():
        logger.warning("No baseline scores found. Run baseline evaluation first.")
        return

    scores_path = baseline_dir / "scores.json"
    if not scores_path.exists():
        return

    raw = json.loads(scores_path.read_text())
    scores = [ConversationScore(**s) for s in raw]

    result = meta.evaluate(scores)
    print(f"\n✅ Meta-evaluation: {result.flaws_detected} flaws detected in {result.scores_analyzed} scores")
    for f in result.findings:
        print(f"   [{f.severity}] {f.finding_type}: {f.description}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the full evaluation pipeline")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--iterations", type=int, default=3, help="Self-learning iterations per agent")
    parser.add_argument("--conversations", type=int, default=5, help="Conversations per evaluation")
    parser.add_argument(
        "--baseline-repeats",
        type=int,
        default=1,
        help="Baseline conversations per persona",
    )
    parser.add_argument("--budget", type=float, default=20.0, help="Total LLM budget in USD")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline evaluation")
    parser.add_argument("--skip-learning", action="store_true", help="Skip self-learning")
    parser.add_argument("--skip-meta", action="store_true", help="Skip meta-evaluation")
    parser.add_argument(
        "--generate-voice-demo",
        action="store_true",
        help="Generate the required Agent 2 xAI TTS audio artifact",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    adapter = get_adapter()
    case = _default_case()

    print("=" * 60)
    print("Riverline Self-Learning Pipeline (Full System Evaluation)")
    print(f"Seed: {args.seed} | Iterations: {args.iterations} | Budget: ${args.budget}")
    print("Pipeline: Agent 1 → handoff → Agent 2 → handoff → Agent 3")
    print("=" * 60)

    # 1. Baseline: full pipeline evaluation
    if not args.skip_baseline:
        print("\n📊 Phase 1: Full Pipeline Baseline Evaluation")
        run_baseline_evaluation(adapter, case, args.seed, args.baseline_repeats)

    # 2. Self-learning: system-level evaluation per mutation
    if not args.skip_learning:
        print("\n🧠 Phase 2: System-Level Self-Learning Loop")
        run_self_learning(adapter, case, args.iterations, args.conversations, args.budget)

    # 3. Meta-evaluation
    if not args.skip_meta:
        print("\n🔍 Phase 3: Meta-Evaluation (Darwin Gödel Machine)")
        run_meta_evaluation(adapter)

    # 4. Optional Agent 2 voice artifact
    if args.generate_voice_demo:
        print("\n🎙️ Phase 4: Agent 2 Voice Artifact")
        audio_path = generate_agent2_demo_audio()
        print(f"   Saved: {audio_path}")

    # 5. Cost summary
    usage = adapter.cumulative_usage
    print("\n" + "=" * 60)
    print("💰 Cost Summary")
    print(f"   Input tokens:  {usage['input_tokens']:,}")
    print(f"   Output tokens: {usage['output_tokens']:,}")
    print(f"   Total cost:    ${usage['cost_usd']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
