"""Token counting and budget enforcement using tiktoken.

Grok does not publish its tokenizer, so we use cl100k_base (GPT-4)
as a conservative proxy.  The assignment requires token budgets to be
"enforced in code and evidenceable," so every call logs its counts.
"""

import logging

import tiktoken

logger = logging.getLogger(__name__)

_ENCODING = tiktoken.get_encoding("cl100k_base")

MAX_AGENT_CONTEXT_TOKENS = 2000
MAX_HANDOFF_TOKENS = 500


def count_tokens(text: str) -> int:
    """Count tokens in *text* using the cl100k_base encoding."""
    return len(_ENCODING.encode(text))


def enforce_agent_budget(
    system_prompt: str,
    handoff_text: str | None = None,
    agent_name: str = "unknown",
) -> dict:
    """Validate that system_prompt + handoff fits within the 2 000-token budget.

    Returns a breakdown dict for the audit trail.
    Raises ``ValueError`` if any budget is exceeded.
    """
    sys_tokens = count_tokens(system_prompt)
    handoff_tokens = count_tokens(handoff_text) if handoff_text else 0
    total = sys_tokens + handoff_tokens

    breakdown = {
        "agent": agent_name,
        "system_prompt_tokens": sys_tokens,
        "handoff_tokens": handoff_tokens,
        "total_tokens": total,
        "budget": MAX_AGENT_CONTEXT_TOKENS,
        "remaining": MAX_AGENT_CONTEXT_TOKENS - total,
    }

    if handoff_tokens > MAX_HANDOFF_TOKENS:
        logger.error(
            "Handoff budget exceeded for %s: %d/%d",
            agent_name,
            handoff_tokens,
            MAX_HANDOFF_TOKENS,
        )
        raise ValueError(
            f"Handoff budget exceeded for {agent_name}: "
            f"{handoff_tokens}/{MAX_HANDOFF_TOKENS} tokens"
        )

    if total > MAX_AGENT_CONTEXT_TOKENS:
        logger.error(
            "Agent context budget exceeded for %s: %d/%d",
            agent_name,
            total,
            MAX_AGENT_CONTEXT_TOKENS,
        )
        raise ValueError(
            f"Agent context budget exceeded for {agent_name}: "
            f"{total}/{MAX_AGENT_CONTEXT_TOKENS} tokens"
        )

    logger.info(
        "Token budget OK for %s: system=%d handoff=%d total=%d/%d",
        agent_name,
        sys_tokens,
        handoff_tokens,
        total,
        MAX_AGENT_CONTEXT_TOKENS,
    )
    return breakdown


def truncate_to_budget(text: str, max_tokens: int) -> str:
    """Truncate *text* so it fits within *max_tokens*."""
    tokens = _ENCODING.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated = _ENCODING.decode(tokens[:max_tokens])
    logger.warning("Truncated text from %d to %d tokens", len(tokens), max_tokens)
    return truncated
