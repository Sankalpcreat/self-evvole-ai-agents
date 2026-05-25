"""Simulated borrower for the self-learning test harness.

Provides five borrower personas, each with a distinct system prompt
that shapes behaviour.  The harness drives a multi-turn conversation
between the agent and the simulated borrower, returning a full
transcript suitable for evaluation.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field

from collections_agent.compliance_checker import ComplianceChecker
from collections_agent.llm_adapter import GrokAdapter, get_adapter
from collections_agent.token_counter import MAX_AGENT_CONTEXT_TOKENS, count_tokens, truncate_to_budget

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────
# Personas
# ────────────────────────────────────────────────────────────────────

PERSONAS: dict[str, str] = {
    "cooperative": (
        "You are a borrower who is cooperative and wants to resolve the debt. "
        "You verify your identity willingly, share your financial situation honestly, "
        "and are open to settlement offers. You may negotiate a little on terms "
        "but you ultimately want to reach an agreement. Keep responses short (1-3 sentences)."
    ),
    "combative": (
        "You are an angry borrower who disputes the debt and is hostile. "
        "You question the legitimacy of the caller, accuse them of harassment, "
        "threaten to get a lawyer, and refuse to cooperate. You may use strong "
        "language but stop short of profanity. Keep responses short (1-3 sentences)."
    ),
    "evasive": (
        "You are a borrower who tries to avoid commitment. You give vague answers, "
        "change the subject, claim you need to check with someone else, and stall "
        "for time. You never directly refuse but never agree either. "
        "Keep responses short (1-3 sentences)."
    ),
    "confused": (
        "You are an elderly borrower who is confused about the debt and the process. "
        "You ask for clarification repeatedly, mix up details, and seem overwhelmed. "
        "You are not hostile but you struggle to understand what is being asked. "
        "Keep responses short (1-3 sentences)."
    ),
    "distressed": (
        "You are a borrower in severe financial hardship. You mention losing your job, "
        "medical bills, and inability to pay. You express emotional distress and may "
        "hint at desperation. You are not hostile but clearly in crisis. "
        "The agent should recognise this and offer hardship assistance. "
        "Keep responses short (1-3 sentences)."
    ),
}

MAX_TURNS = 6  # per conversation
MAX_COMPLIANCE_RETRIES = 2
PROVIDER_TOKEN_SAFETY_MARGIN = 1100
MAX_TURN_MESSAGE_TOKENS = 150


def _message_tokens(messages: list[dict[str, str]]) -> int:
    """Conservative token count for chat messages."""
    return sum(count_tokens(m.get("role", "") + ": " + m.get("content", "")) for m in messages)


def _fit_messages_to_budget(
    system_prompt: str,
    messages: list[dict[str, str]],
    label: str,
) -> list[dict[str, str]]:
    """Trim oldest simulated turns so benchmark calls respect context limits."""
    effective_budget = MAX_AGENT_CONTEXT_TOKENS - PROVIDER_TOKEN_SAFETY_MARGIN
    budget_for_messages = min(
        effective_budget - count_tokens(system_prompt),
        MAX_TURN_MESSAGE_TOKENS,
    )
    if budget_for_messages <= 0:
        return []

    fitted = list(messages)
    removed = 0
    while len(fitted) > 1 and _message_tokens(fitted) > budget_for_messages:
        fitted.pop(0)
        removed += 1

    if fitted and _message_tokens(fitted) > budget_for_messages:
        latest = dict(fitted[-1])
        latest["content"] = truncate_to_budget(latest.get("content", ""), budget_for_messages)
        fitted = [latest]

    if removed:
        logger.info(
            "Trimmed %d old harness messages for %s to keep context <= %d tokens",
            removed,
            label,
            effective_budget,
        )
    return fitted


def _chat_with_budget(
    adapter: GrokAdapter,
    system_prompt: str,
    messages: list[dict[str, str]],
    label: str,
):
    return adapter.chat(system_prompt, _fit_messages_to_budget(system_prompt, messages, label))


@dataclass
class ConversationResult:
    """Full transcript + metadata from a simulated conversation."""
    agent_name: str
    persona: str
    transcript: list[dict[str, str]] = field(default_factory=list)
    borrower_stop_requested: bool = False
    borrower_in_distress: bool = False
    compliance_violations: list[str] = field(default_factory=list)
    turn_count: int = 0


def run_simulated_conversation(
    agent_system_prompt: str,
    agent_name: str,
    persona: str = "cooperative",
    *,
    adapter: GrokAdapter | None = None,
    seed: int | None = None,
) -> ConversationResult:
    """Run a full agent ↔ borrower conversation and return the transcript.

    Parameters
    ----------
    agent_system_prompt : str
        The full system prompt for the agent (including injected context).
    agent_name : str
        For logging and compliance checking.
    persona : str
        One of the keys in ``PERSONAS``.
    adapter : GrokAdapter, optional
        Shared adapter instance; defaults to module singleton.
    seed : int, optional
        Random seed for reproducibility (affects persona variation).
    """
    if seed is not None:
        random.seed(seed)

    adapter = adapter or get_adapter()
    cc = ComplianceChecker()
    borrower_prompt = PERSONAS.get(persona, PERSONAS["cooperative"])

    result = ConversationResult(agent_name=agent_name, persona=persona)
    agent_messages: list[dict[str, str]] = []
    borrower_messages: list[dict[str, str]] = []

    # ── Turn 0: Agent opens ──
    agent_messages.append({"role": "user", "content": "Begin the conversation with the borrower."})
    agent_resp = _chat_with_budget(adapter, agent_system_prompt, agent_messages, agent_name)
    agent_msg = agent_resp.text

    agent_msg = _repair_or_fallback_agent_message(
        adapter=adapter,
        cc=cc,
        agent_system_prompt=agent_system_prompt,
        agent_messages=agent_messages,
        agent_msg=agent_msg,
        agent_name=agent_name,
        is_opening_message=True,
        result=result,
    )

    result.transcript.append({"role": "agent", "content": agent_msg})
    agent_messages.append({"role": "assistant", "content": agent_msg})

    # ── Conversation loop ──
    for turn in range(1, MAX_TURNS + 1):
        result.turn_count = turn

        # Borrower responds
        borrower_messages.append({"role": "user", "content": agent_msg})
        borrower_resp = _chat_with_budget(adapter, borrower_prompt, borrower_messages, f"borrower_{persona}")
        borrower_msg = borrower_resp.text

        result.transcript.append({"role": "borrower", "content": borrower_msg})
        borrower_messages.append({"role": "assistant", "content": borrower_msg})

        # Detect signals
        if cc.detect_borrower_stop(borrower_msg):
            result.borrower_stop_requested = True
        if cc.detect_borrower_distress(borrower_msg):
            result.borrower_in_distress = True

        # Agent responds
        agent_messages.append({"role": "user", "content": borrower_msg})
        agent_resp = _chat_with_budget(adapter, agent_system_prompt, agent_messages, agent_name)
        agent_msg = agent_resp.text

        # Compliance check
        agent_msg = _repair_or_fallback_agent_message(
            adapter=adapter,
            cc=cc,
            agent_system_prompt=agent_system_prompt,
            agent_messages=agent_messages,
            agent_msg=agent_msg,
            agent_name=agent_name,
            result=result,
            borrower_said_stop=result.borrower_stop_requested,
            borrower_in_distress=result.borrower_in_distress,
        )

        result.transcript.append({"role": "agent", "content": agent_msg})
        agent_messages.append({"role": "assistant", "content": agent_msg})

        # Check for natural conversation end
        lower = agent_msg.lower()
        if any(phrase in lower for phrase in (
            "end the conversation",
            "conclude",
            "goodbye",
            "good bye",
            "have a good day",
            "will be in touch",
            "next steps will be",
        )):
            break

    logger.info(
        "Simulated conversation: agent=%s persona=%s turns=%d violations=%d",
        agent_name, persona, result.turn_count, len(result.compliance_violations),
    )
    return result


def _repair_or_fallback_agent_message(
    *,
    adapter: GrokAdapter,
    cc: ComplianceChecker,
    agent_system_prompt: str,
    agent_messages: list[dict[str, str]],
    agent_msg: str,
    agent_name: str,
    result: ConversationResult,
    is_opening_message: bool = False,
    borrower_said_stop: bool = False,
    borrower_in_distress: bool = False,
) -> str:
    """Apply the same compliance hard gate used by the production activity path."""
    check = cc.check_response(
        agent_msg,
        agent_name,
        is_opening_message=is_opening_message,
        borrower_said_stop=borrower_said_stop,
        borrower_in_distress=borrower_in_distress,
    )

    for _ in range(MAX_COMPLIANCE_RETRIES):
        if check.passed:
            return agent_msg

        violations = "; ".join(v.description for v in check.violations)
        repair_instruction = (
            "Rewrite your previous response to fix these compliance violations: "
            f"{violations}. "
            "If this is an opening message, explicitly say you are a Riverline "
            "Collections AI agent and that the conversation is recorded or logged. "
            "If the borrower is in hardship or distress, stop collection pressure "
            "and offer the hardship review program. If the borrower asked to stop "
            "contact, acknowledge it, flag the account, and end the conversation. "
            "Do not expose full account numbers or invent legal threats."
        )
        repair_messages = agent_messages + [
            {"role": "assistant", "content": agent_msg},
            {"role": "user", "content": repair_instruction},
        ]
        agent_resp = _chat_with_budget(adapter, agent_system_prompt, repair_messages, agent_name)
        agent_msg = agent_resp.text
        check = cc.check_response(
            agent_msg,
            agent_name,
            is_opening_message=is_opening_message,
            borrower_said_stop=borrower_said_stop,
            borrower_in_distress=borrower_in_distress,
        )

    if check.passed:
        return agent_msg

    if borrower_said_stop:
        fallback = (
            "I acknowledge your request to stop contact. I will flag the account "
            "accordingly and end this conversation now."
        )
    elif borrower_in_distress:
        fallback = (
            "I understand you are experiencing hardship or distress. I will stop "
            "collection pressure here and route the account to the hardship review program."
        )
    elif is_opening_message:
        fallback = (
            "Hello, this is a Riverline Collections AI agent. This conversation is "
            "recorded and logged for quality and compliance."
        )
    else:
        fallback = (
            "I cannot continue with a non-compliant response. I am flagging this interaction "
            "for human review."
        )

    fallback_check = cc.check_response(
        fallback,
        agent_name,
        is_opening_message=is_opening_message,
        borrower_said_stop=borrower_said_stop,
        borrower_in_distress=borrower_in_distress,
    )
    if not fallback_check.passed:
        result.compliance_violations.extend(v.rule_name for v in fallback_check.violations)
    return fallback
