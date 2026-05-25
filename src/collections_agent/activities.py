"""Temporal activities — real LLM-powered agents.

Each agent runs a multi-turn conversation loop:
  1. Load system prompt + inject borrower context.
  2. Enforce 2 000-token agent budget.
  3. Loop: agent turn → compliance check → borrower turn (simulated or live).
  4. Extract structured result from conversation.
  5. Return typed result dataclass.

For the self-learning test harness the borrower is an LLM persona.
For live deployment the borrower messages come from the API.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from temporalio import activity

from collections_agent.compliance_checker import ComplianceChecker
from collections_agent.llm_adapter import GrokAdapter, get_adapter
from collections_agent.models import (
    AssessmentResult,
    BorrowerCase,
    FinalNoticeResult,
    HandoffSummary,
    ResolutionResult,
)
from collections_agent.prompt_manager import PromptManager
from collections_agent.structured_outputs import (
    AssessmentExtraction,
    FinalNoticeExtraction,
    ResolutionExtraction,
)
from collections_agent.token_counter import (
    MAX_AGENT_CONTEXT_TOKENS,
    MAX_HANDOFF_TOKENS,
    count_tokens,
    enforce_agent_budget,
    truncate_to_budget,
)

logger = logging.getLogger(__name__)

MAX_TURNS = 6  # per agent conversation
MAX_COMPLIANCE_RETRIES = 2  # retries per agent turn on compliance failure
PROVIDER_TOKEN_SAFETY_MARGIN = 1100  # xAI usage includes message/schema overhead beyond local text tokens.
MAX_TURN_MESSAGE_TOKENS = 150  # preserve latest turn, not full transcript, inside each provider call.

# Settlement policy path (relative to project root)
_SETTLEMENT_POLICY_PATH = (
    Path(__file__).resolve().parent.parent.parent / "policy" / "settlement_policy.json"
)

# Simulated borrower persona (simple, not the full borrower_harness)
_BORROWER_PERSONA = (
    "You are a borrower receiving a collections call. "
    "Respond naturally in 1-3 sentences. "
    "Be cooperative but ask clarifying questions."
)

_prompt_mgr: PromptManager | None = None
_compliance: ComplianceChecker | None = None


def _pm() -> PromptManager:
    global _prompt_mgr
    if _prompt_mgr is None:
        _prompt_mgr = PromptManager()
    return _prompt_mgr


def _cc() -> ComplianceChecker:
    global _compliance
    if _compliance is None:
        _compliance = ComplianceChecker()
    return _compliance


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _load_settlement_policy() -> str:
    """Load settlement_policy.json and return it as a formatted string."""
    try:
        with open(_SETTLEMENT_POLICY_PATH) as fh:
            policy = json.load(fh)
        return json.dumps(policy, indent=2)
    except Exception:
        logger.warning("Could not load settlement policy from %s", _SETTLEMENT_POLICY_PATH)
        return ""


def _build_system_prompt(agent_name: str, case: BorrowerCase, handoff_text: str | None = None) -> str:
    """Load the prompt, inject borrower context, and validate the budget."""
    base_prompt = _pm().load_prompt(agent_name)

    context_block = (
        "\n\n## Current Case (injected at runtime)\n"
        f"- Borrower ID: {case.borrower_id}\n"
        f"- Company: {case.company_name}\n"
        f"- Debt: ${case.debt_amount_cents / 100:,.2f}\n"
        f"- Account last 4: {case.account_last4}\n"
    )
    if handoff_text:
        context_block += f"\n## Prior Context (handoff)\n{handoff_text}\n"

    # Bug 3 fix: Inject settlement policy for Agent 2
    if agent_name == "agent2":
        policy_text = _load_settlement_policy()
        if policy_text:
            context_block += (
                "\n## Settlement Policy (you MUST only offer terms within these bounds)\n"
                f"```json\n{policy_text}\n```\n"
                "\nKey rules:\n"
                "- Lump-sum: max 30% discount (borrower pays at least 70% of balance), 48h commitment deadline\n"
                "- Payment plan: min $250 down payment, min $250/month, max 18 months, first payment within 48h\n"
                "- Hardship: if borrower signals hardship or distress, you MUST offer the hardship review program\n"
                "- NEVER offer discounts, deadlines, or terms outside this policy\n"
            )

    full_prompt = base_prompt + context_block

    # Enforce the budget (raises ValueError if exceeded)
    enforce_agent_budget(full_prompt, agent_name=agent_name)

    return full_prompt


def _message_tokens(messages: list[dict[str, str]]) -> int:
    """Conservative token count for chat messages."""
    return sum(count_tokens(m.get("role", "") + ": " + m.get("content", "")) for m in messages)


def _fit_messages_to_agent_budget(
    system_prompt: str,
    messages: list[dict[str, str]],
    agent_name: str,
) -> list[dict[str, str]]:
    """Drop oldest turn context until the provider call fits the 2000-token cap."""
    effective_budget = MAX_AGENT_CONTEXT_TOKENS - PROVIDER_TOKEN_SAFETY_MARGIN
    budget_for_messages = min(
        effective_budget - count_tokens(system_prompt),
        MAX_TURN_MESSAGE_TOKENS,
    )
    if budget_for_messages <= 0:
        enforce_agent_budget(system_prompt, agent_name=agent_name)
        return []

    fitted = list(messages)
    removed = 0
    while len(fitted) > 1 and _message_tokens(fitted) > budget_for_messages:
        fitted.pop(0)
        removed += 1

    if fitted and _message_tokens(fitted) > budget_for_messages:
        # Single oversized message: keep the latest instruction, but truncate it.
        latest = dict(fitted[-1])
        latest["content"] = truncate_to_budget(latest.get("content", ""), budget_for_messages)
        fitted = [latest]

    if removed:
        logger.info(
            "Trimmed %d old messages for %s to keep context <= %d tokens",
            removed,
            agent_name,
            effective_budget,
        )

    return fitted


def _chat_with_agent_budget(
    adapter: GrokAdapter,
    system_prompt: str,
    messages: list[dict[str, str]],
    agent_name: str,
):
    """Send a chat call after enforcing the full per-call context budget."""
    fitted = _fit_messages_to_agent_budget(system_prompt, messages, agent_name)
    return adapter.chat(system_prompt, fitted)


def _build_handoff_state(lines: list[tuple[str, object]]) -> HandoffSummary:
    """Build a source-backed handoff capsule and enforce the 500-token cap.

    This is intentionally deterministic. The raw transcript remains on the
    result object; the handoff is only the compact working state for the next
    agent.
    """
    rendered = ["HANDOFF_STATE v1"]
    for key, value in lines:
        if value is None or value == "" or value == []:
            value = "none"
        rendered.append(f"{key}: {value}")

    text = "\n".join(rendered)
    text = truncate_to_budget(text, MAX_HANDOFF_TOKENS)
    return HandoffSummary(text=text, token_count=count_tokens(text))


def _run_conversation(
    adapter: GrokAdapter,
    system_prompt: str,
    agent_name: str,
    opening_instruction: str = "Begin the conversation.",
) -> list[dict[str, str]]:
    """Run a multi-turn agent conversation.

    Returns the full transcript as a list of ``{"role": ..., "content": ...}``.

    In *simulation mode* (self-learning harness) the borrower is played
    by the LLM via a simple persona prompt.  In *live mode* messages arrive
    through the API.  This function handles simulation mode; live
    mode is handled by the FastAPI layer.
    """
    messages: list[dict[str, str]] = []
    transcript: list[dict[str, str]] = []
    cc = _cc()
    compliance_blocked_count = 0

    # Track borrower state across turns
    borrower_said_stop = False
    borrower_in_distress = False

    # ── Turn 1: Agent speaks (opening message) ──────────────────────
    messages.append({"role": "user", "content": opening_instruction})
    resp = _chat_with_agent_budget(adapter, system_prompt, messages, agent_name)
    agent_msg = resp.text

    # Compliance hard gate on opening message
    result = cc.check_response(agent_msg, agent_name, is_opening_message=True)
    if not result.passed:
        # Retry up to MAX_COMPLIANCE_RETRIES times
        for retry in range(MAX_COMPLIANCE_RETRIES):
            violation_names = [v.description for v in result.violations]
            rewrite_instruction = (
                f"Your previous response had compliance violations: {violation_names}. "
                "Rewrite your opening message to fix ALL of these violations. "
                "You MUST: identify yourself as an AI agent, disclose that the conversation "
                "is being recorded/logged, and avoid any false threats or PII exposure."
            )
            messages_retry = messages + [
                {"role": "assistant", "content": agent_msg},
                {"role": "user", "content": rewrite_instruction},
            ]
            resp = _chat_with_agent_budget(adapter, system_prompt, messages_retry, agent_name)
            agent_msg = resp.text
            result = cc.check_response(agent_msg, agent_name, is_opening_message=True)
            if result.passed:
                break
        if not result.passed:
            # Still failing after retries — terminate conversation
            compliance_blocked_count += 1
            logger.error(
                "Opening message blocked by compliance after %d retries for %s. "
                "Terminating conversation. Violations: %s",
                MAX_COMPLIANCE_RETRIES, agent_name,
                [v.description for v in result.violations],
            )
            transcript.append({
                "role": "system",
                "content": f"[CONVERSATION TERMINATED: compliance gate blocked opening message after {MAX_COMPLIANCE_RETRIES} retries]",
            })
            return transcript

    transcript.append({"role": "agent", "content": agent_msg})
    messages.append({"role": "assistant", "content": agent_msg})

    # ── Multi-turn loop ─────────────────────────────────────────────
    for turn in range(1, MAX_TURNS):
        # --- Borrower turn (simulated by LLM) ---
        borrower_messages = [
            {"role": "user", "content": (
                "Here is the conversation so far:\n"
                + json.dumps(transcript, indent=2)
                + "\n\nRespond as the borrower. Stay in character."
            )}
        ]
        borrower_resp = _chat_with_agent_budget(adapter, _BORROWER_PERSONA, borrower_messages, "borrower_sim")
        borrower_msg = borrower_resp.text

        transcript.append({"role": "borrower", "content": borrower_msg})
        messages.append({"role": "user", "content": borrower_msg})

        # Detect borrower signals
        if cc.detect_borrower_stop(borrower_msg):
            borrower_said_stop = True
        if cc.detect_borrower_distress(borrower_msg):
            borrower_in_distress = True

        # --- Check for natural conversation ending ---
        # If borrower says stop, agent must acknowledge and we end
        if borrower_said_stop:
            # Generate acknowledgment from agent
            stop_instruction = (
                "The borrower has asked to stop contact. You MUST acknowledge their request, "
                "confirm you will cease further communication, and end the conversation respectfully."
            )
            messages.append({"role": "user", "content": stop_instruction})
            resp = _chat_with_agent_budget(adapter, system_prompt, messages, agent_name)
            agent_msg = resp.text

            # Compliance check (with stop flag)
            result = cc.check_response(
                agent_msg, agent_name,
                borrower_said_stop=True,
                borrower_in_distress=borrower_in_distress,
            )
            if not result.passed:
                for retry in range(MAX_COMPLIANCE_RETRIES):
                    violation_names = [v.description for v in result.violations]
                    rewrite_instruction = (
                        f"Compliance violations found: {violation_names}. "
                        "Rewrite your response to fix ALL violations. "
                        "The borrower asked to stop — you MUST acknowledge and cease contact."
                    )
                    messages.append({"role": "assistant", "content": agent_msg})
                    messages.append({"role": "user", "content": rewrite_instruction})
                    resp = _chat_with_agent_budget(adapter, system_prompt, messages, agent_name)
                    agent_msg = resp.text
                    result = cc.check_response(
                        agent_msg, agent_name,
                        borrower_said_stop=True,
                        borrower_in_distress=borrower_in_distress,
                    )
                    if result.passed:
                        break
                if not result.passed:
                    compliance_blocked_count += 1
                    logger.error(
                        "Stop-acknowledgment blocked by compliance for %s", agent_name,
                    )

            transcript.append({"role": "agent", "content": agent_msg})
            transcript.append({
                "role": "system",
                "content": "[CONVERSATION ENDED: borrower requested stop contact]",
            })
            break

        # --- Agent turn ---
        resp = _chat_with_agent_budget(adapter, system_prompt, messages, agent_name)
        agent_msg = resp.text

        # Compliance hard gate
        result = cc.check_response(
            agent_msg, agent_name,
            borrower_said_stop=borrower_said_stop,
            borrower_in_distress=borrower_in_distress,
        )
        if not result.passed:
            # Retry up to MAX_COMPLIANCE_RETRIES times
            retried_ok = False
            for retry in range(MAX_COMPLIANCE_RETRIES):
                violation_names = [v.description for v in result.violations]
                rewrite_instruction = (
                    f"Your response had compliance violations: {violation_names}. "
                    "Rewrite your response to fix ALL of these violations. "
                    "Maintain professional tone and stay within policy bounds."
                )
                if borrower_in_distress:
                    rewrite_instruction += (
                        " The borrower is showing signs of distress — "
                        "you MUST offer the hardship review program."
                    )
                messages_with_retry = messages + [
                    {"role": "assistant", "content": agent_msg},
                    {"role": "user", "content": rewrite_instruction},
                ]
                resp = _chat_with_agent_budget(adapter, system_prompt, messages_with_retry, agent_name)
                agent_msg = resp.text
                result = cc.check_response(
                    agent_msg, agent_name,
                    borrower_said_stop=borrower_said_stop,
                    borrower_in_distress=borrower_in_distress,
                )
                if result.passed:
                    retried_ok = True
                    break

            if not result.passed:
                # Still failing — terminate the conversation
                compliance_blocked_count += 1
                logger.error(
                    "Agent turn %d blocked by compliance after %d retries for %s. "
                    "Terminating conversation. Violations: %s",
                    turn, MAX_COMPLIANCE_RETRIES, agent_name,
                    [v.description for v in result.violations],
                )
                transcript.append({
                    "role": "system",
                    "content": (
                        f"[CONVERSATION TERMINATED: compliance gate blocked agent response "
                        f"at turn {turn} after {MAX_COMPLIANCE_RETRIES} retries. "
                        f"Total blocked: {compliance_blocked_count}]"
                    ),
                })
                break

        transcript.append({"role": "agent", "content": agent_msg})
        messages.append({"role": "assistant", "content": agent_msg})

        # --- Detect natural conversation ending ---
        # Check if the agent's message signals a natural conclusion
        lower_agent = agent_msg.lower()
        natural_end_signals = [
            "thank you for your time",
            "we'll send you the confirmation",
            "agreement has been noted",
            "this concludes our",
            "have a good day",
            "goodbye",
            "take care",
        ]
        if any(signal in lower_agent for signal in natural_end_signals):
            transcript.append({
                "role": "system",
                "content": "[CONVERSATION ENDED: natural conclusion detected]",
            })
            break

    if compliance_blocked_count > 0:
        logger.warning(
            "Conversation for %s had %d compliance-blocked messages",
            agent_name, compliance_blocked_count,
        )

    return transcript


def _extract_assessment(adapter: GrokAdapter, transcript: list[dict[str, str]], case: BorrowerCase) -> dict:
    """Ask the LLM to extract structured assessment from a transcript."""
    extraction_prompt = (
        "You are a data extractor. Given the conversation transcript below, "
        "extract the assessment fields into the provided schema. Use only facts "
        "supported by the transcript."
    )
    transcript_text = json.dumps(transcript, indent=2)
    result, _ = adapter.chat_structured(
        extraction_prompt,
        [{"role": "user", "content": transcript_text}],
        AssessmentExtraction,
    )
    return result.model_dump()


def _extract_resolution(adapter: GrokAdapter, transcript: list[dict[str, str]]) -> dict:
    """Extract structured resolution data from a voice transcript."""
    extraction_prompt = (
        "You are a data extractor. Given the voice call transcript below, "
        "extract the resolution fields into the provided schema. Use only facts "
        "supported by the transcript."
    )
    transcript_text = json.dumps(transcript, indent=2)
    result, _ = adapter.chat_structured(
        extraction_prompt,
        [{"role": "user", "content": transcript_text}],
        ResolutionExtraction,
    )
    return result.model_dump()


def _extract_final_notice(adapter: GrokAdapter, transcript: list[dict[str, str]]) -> dict:
    """Extract structured final notice data from a chat transcript."""
    extraction_prompt = (
        "You are a data extractor. Given the final notice conversation below, "
        "extract the final-notice fields into the provided schema. Use only facts "
        "supported by the transcript."
    )
    transcript_text = json.dumps(transcript, indent=2)
    result, _ = adapter.chat_structured(
        extraction_prompt,
        [{"role": "user", "content": transcript_text}],
        FinalNoticeExtraction,
    )
    return result.model_dump()


# ────────────────────────────────────────────────────────────────────
# Activities
# ────────────────────────────────────────────────────────────────────

@activity.defn
def run_assessment_chat(case: BorrowerCase, attempt: int) -> AssessmentResult:
    """Agent 1 — cold assessment chat."""
    adapter = get_adapter()
    system_prompt = _build_system_prompt("agent1", case)

    opening = (
        f"A borrower (ID: {case.borrower_id}) has entered the collections pipeline. "
        f"This is assessment attempt {attempt}. Begin the conversation."
    )

    transcript = _run_conversation(adapter, system_prompt, "agent1", opening)

    # Extract structured data
    try:
        data = _extract_assessment(adapter, transcript, case)
    except Exception:
        logger.exception("Failed to extract assessment")
        data = {}

    return AssessmentResult(
        status=data.get("status", "no_response"),
        identity_verified=data.get("identity_verified", False),
        financial_situation=data.get("financial_situation", "unknown"),
        viable_path=data.get("viable_path", "voice_resolution"),
        hardship_signal=data.get("hardship_signal", False),
        distress_signal=data.get("distress_signal", False),
        stop_contact=data.get("stop_contact", False),
        debt_disputed=data.get("debt_disputed", False),
        transcript_id=f"{case.borrower_id}-assessment-attempt-{attempt}",
        brief_summary=data.get("brief_summary", ""),
        transcript=transcript,
    )


@activity.defn
def summarize_assessment(result: AssessmentResult) -> HandoffSummary:
    """Summarise Agent 1's conversation for Agent 2 (≤ 500 tokens)."""
    summary = _build_handoff_state([
        ("source_stage", "agent1_assessment_chat"),
        ("source_transcript_id", result.transcript_id),
        ("timeline_1", "Assessment chat gathered identity and financial facts before voice handoff."),
        ("identity_verified", result.identity_verified),
        ("financial_situation", result.financial_situation),
        ("viable_path", result.viable_path),
        ("debt_disputed", result.debt_disputed),
        ("hardship_signal", result.hardship_signal),
        ("distress_signal", result.distress_signal),
        ("stop_contact", result.stop_contact),
        ("stage_outcome", result.status),
        ("brief_summary", result.brief_summary),
        (
            "continuity_instruction",
            "Agent 2 must not re-introduce as a new agent or re-ask verified facts; continue from this state.",
        ),
    ])

    logger.info("Assessment handoff: %d tokens", summary.token_count)
    return summary


@activity.defn
def run_resolution_voice(case: BorrowerCase, handoff: HandoffSummary) -> ResolutionResult:
    """Agent 2 — transactional voice resolution.

    For bulk evaluation this runs as text (no actual voice call).
    Voice recording is generated separately for the demo artifact.
    """
    adapter = get_adapter()
    system_prompt = _build_system_prompt("agent2", case, handoff.text)

    opening = (
        "The borrower is now on a voice call. You already have context from the "
        "assessment chat. Present settlement options and push for commitment. Begin."
    )

    transcript = _run_conversation(adapter, system_prompt, "agent2", opening)

    try:
        data = _extract_resolution(adapter, transcript)
    except Exception:
        logger.exception("Failed to extract resolution")
        data = {}

    return ResolutionResult(
        status=data.get("status", "no_deal"),
        selected_offer=data.get("selected_offer"),
        borrower_position=data.get("borrower_position", "unknown"),
        objections=data.get("objections", []),
        latest_payment_capacity=data.get("latest_payment_capacity", "unknown"),
        hardship_signal=data.get("hardship_signal", False),
        distress_signal=data.get("distress_signal", False),
        stop_contact=data.get("stop_contact", False),
        call_session_id=f"{case.borrower_id}-resolution-call",
        transcript_id=f"{case.borrower_id}-resolution-transcript",
        brief_summary=data.get("brief_summary", ""),
        call_transcript=transcript,
    )


@activity.defn
def summarize_full_history(
    assessment: AssessmentResult,
    assessment_handoff: HandoffSummary,
    resolution: ResolutionResult,
) -> HandoffSummary:
    """Summarise the full history (chat + voice) for Agent 3 (≤ 500 tokens)."""
    summary = _build_handoff_state([
        ("source_stage", "agent1_assessment_chat + agent2_resolution_voice"),
        ("source_transcript_ids", [assessment.transcript_id, resolution.transcript_id]),
        ("timeline_1", "Agent 1 assessment chat: " + assessment_handoff.text.replace("\n", " | ")),
        ("timeline_2", "Agent 2 voice resolution call presented policy-bound options and captured borrower position."),
        ("assessment_status", assessment.status),
        ("identity_verified", assessment.identity_verified),
        ("financial_situation", assessment.financial_situation),
        ("resolution_status", resolution.status),
        ("selected_offer", resolution.selected_offer),
        ("borrower_position", resolution.borrower_position),
        ("objections", "; ".join(resolution.objections)),
        ("latest_payment_capacity", resolution.latest_payment_capacity),
        ("hardship_signal", assessment.hardship_signal or resolution.hardship_signal),
        ("distress_signal", assessment.distress_signal or resolution.distress_signal),
        ("stop_contact", assessment.stop_contact or resolution.stop_contact),
        ("brief_voice_summary", resolution.brief_summary),
        (
            "continuity_instruction",
            "Agent 3 must reference the prior call as already completed; do not re-verify or re-negotiate from scratch.",
        ),
    ])

    logger.info("Full-history handoff: %d tokens", summary.token_count)
    return summary


@activity.defn
def run_final_notice_chat(case: BorrowerCase, handoff: HandoffSummary) -> FinalNoticeResult:
    """Agent 3 — consequence-driven closer."""
    adapter = get_adapter()
    system_prompt = _build_system_prompt("agent3", case, handoff.text)

    opening = (
        "Following the voice call, no agreement was reached. "
        "Deliver the final notice with a hard deadline and clear consequences. Begin."
    )

    transcript = _run_conversation(adapter, system_prompt, "agent3", opening)

    try:
        data = _extract_final_notice(adapter, transcript)
    except Exception:
        logger.exception("Failed to extract final notice")
        data = {}

    return FinalNoticeResult(
        status=data.get("status", "no_resolution"),
        final_action=data.get("final_action", "flag_for_legal_or_write_off_review"),
        final_offer_referenced=data.get("final_offer_referenced"),
        expiry=data.get("expiry"),
        documented_next_step=data.get("documented_next_step", "legal_review"),
        borrower_response=data.get("borrower_response"),
        transcript_id=f"{case.borrower_id}-final-notice-transcript",
        brief_summary=data.get("brief_summary", ""),
        transcript=transcript,
    )


@activity.defn
def log_agreement(case: BorrowerCase, resolution: ResolutionResult) -> str:
    """Persist an agreement record."""
    record = {
        "borrower_id": case.borrower_id,
        "status": "agreement_logged",
        "offer": resolution.selected_offer,
        "summary": resolution.brief_summary,
    }
    logger.info("Agreement logged: %s", json.dumps(record))
    return json.dumps(record)


@activity.defn
def log_resolution(case: BorrowerCase, final_notice: FinalNoticeResult) -> str:
    """Persist a resolution record."""
    record = {
        "borrower_id": case.borrower_id,
        "status": "resolution_logged",
        "action": final_notice.final_action,
        "summary": final_notice.brief_summary,
    }
    logger.info("Resolution logged: %s", json.dumps(record))
    return json.dumps(record)


@activity.defn
def flag_for_legal_or_write_off(case: BorrowerCase, final_notice: FinalNoticeResult) -> str:
    """Flag borrower for legal action or write-off."""
    record = {
        "borrower_id": case.borrower_id,
        "status": "flagged_for_legal_or_write_off",
        "next_step": final_notice.documented_next_step,
        "summary": final_notice.brief_summary,
    }
    logger.info("Flagged for legal/write-off: %s", json.dumps(record))
    return json.dumps(record)
    
