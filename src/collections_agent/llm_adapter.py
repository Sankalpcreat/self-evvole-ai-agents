"""Grok / xAI LLM adapter using the OpenAI SDK Responses API.

Design decisions
----------------
* ``max_retries=0`` – Temporal handles retry logic, not the SDK.
* ``store=False``   – Don't persist conversations on xAI servers.
* Error classification – auth / validation → non-retryable
  (``ApplicationError``); rate-limits / 5xx → retryable (re-raised
  so Temporal retries the activity).
* Cumulative cost tracking for the $20 budget constraint.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel
from temporalio.exceptions import ApplicationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class LLMResponse:
    text: str
    usage: LLMUsage
    model: str


StructuredT = TypeVar("StructuredT", bound=BaseModel)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class GrokAdapter:
    """Thin wrapper around the xAI Responses API."""

    # Grok-4.3 pricing (May 2026)
    INPUT_COST_PER_M: float = 1.25
    OUTPUT_COST_PER_M: float = 2.50

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.client = OpenAI(
            api_key=api_key or os.environ["XAI_API_KEY"],
            base_url="https://api.x.ai/v1",
            max_retries=0,
        )
        self.model = model or os.getenv("GROK_MODEL", "grok-4.3")
        self._cumulative = LLMUsage()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> LLMResponse:
        """Send a chat request and return the response with usage info."""
        input_msgs: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            *messages,
        ]

        try:
            response = self.client.responses.create(
                model=self.model,
                input=input_msgs,
                store=False,
            )
        except Exception as exc:
            self._classify_and_raise(exc)
            raise  # unreachable – keeps mypy happy

        # Extract usage --------------------------------------------------
        in_tok = out_tok = 0
        if hasattr(response, "usage") and response.usage:
            in_tok = getattr(response.usage, "input_tokens", 0)
            out_tok = getattr(response.usage, "output_tokens", 0)

        cost = self._calc_cost(in_tok, out_tok)
        usage = LLMUsage(input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost)
        self._accum(usage)

        text = response.output_text or ""
        logger.info(
            "LLM call: model=%s in=%d out=%d cost=$%.6f",
            self.model,
            in_tok,
            out_tok,
            cost,
        )
        return LLMResponse(text=text, usage=usage, model=self.model)

    def chat_json(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> tuple[dict, LLMUsage]:
        """Chat expecting a JSON object back.  Strips markdown fences."""
        suffix = (
            "\n\nCRITICAL: Respond ONLY with a single valid JSON object. "
            "No markdown code fences, no commentary, no extra text."
        )
        resp = self.chat(system_prompt + suffix, messages)
        text = resp.text.strip()

        # Strip ```json ... ``` if the model wraps output
        if text.startswith("```"):
            lines = text.split("\n")
            end = len(lines)
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip().startswith("```"):
                    end = i
                    break
            text = "\n".join(lines[1:end])

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ApplicationError(
                f"LLM returned invalid JSON: {text[:300]}",
                non_retryable=True,
            ) from exc

        return parsed, resp.usage

    def chat_structured(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        schema: type[StructuredT],
    ) -> tuple[StructuredT, LLMUsage]:
        """Chat expecting a Pydantic-validated structured output.

        This uses xAI/OpenAI-compatible structured outputs rather than asking
        the model to produce JSON by convention. The provider constrains the
        output to the schema and the SDK returns a parsed Pydantic object.
        """
        input_msgs: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            *messages,
        ]

        try:
            response = self.client.responses.parse(
                model=self.model,
                input=input_msgs,
                text_format=schema,
                store=False,
            )
        except Exception as exc:
            self._classify_and_raise(exc)
            raise

        in_tok = out_tok = 0
        if hasattr(response, "usage") and response.usage:
            in_tok = getattr(response.usage, "input_tokens", 0)
            out_tok = getattr(response.usage, "output_tokens", 0)

        cost = self._calc_cost(in_tok, out_tok)
        usage = LLMUsage(input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost)
        self._accum(usage)

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise ApplicationError(
                f"Structured output parse returned no parsed {schema.__name__}",
                non_retryable=True,
            )

        logger.info(
            "Structured LLM call: model=%s schema=%s in=%d out=%d cost=$%.6f",
            self.model,
            schema.__name__,
            in_tok,
            out_tok,
            cost,
        )
        return parsed, usage

    @property
    def cumulative_usage(self) -> dict:
        """Return cumulative token and cost totals."""
        return {
            "input_tokens": self._cumulative.input_tokens,
            "output_tokens": self._cumulative.output_tokens,
            "cost_usd": round(self._cumulative.cost_usd, 6),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _calc_cost(self, in_tok: int, out_tok: int) -> float:
        return (in_tok * self.INPUT_COST_PER_M + out_tok * self.OUTPUT_COST_PER_M) / 1_000_000

    def _accum(self, usage: LLMUsage) -> None:
        self._cumulative.input_tokens += usage.input_tokens
        self._cumulative.output_tokens += usage.output_tokens
        self._cumulative.cost_usd += usage.cost_usd

    @staticmethod
    def _classify_and_raise(exc: Exception) -> None:
        """Auth / validation → non-retryable; everything else → retryable."""
        msg = str(exc).lower()
        if any(
            tok in msg
            for tok in (
                "401",
                "403",
                "invalid_api_key",
                "authentication",
                "api key",
                "incorrect api key",
            )
        ):
            raise ApplicationError(
                f"Auth error (non-retryable): {exc}",
                non_retryable=True,
            ) from exc
        if "400" in msg and "invalid" in msg:
            raise ApplicationError(
                f"Validation error (non-retryable): {exc}",
                non_retryable=True,
            ) from exc
        # Rate-limits (429), timeouts, 5xx → let Temporal retry
        raise


# ---------------------------------------------------------------------------
# Module-level singleton (lazily initialised in the worker process)
# ---------------------------------------------------------------------------

_adapter: GrokAdapter | None = None


def get_adapter() -> GrokAdapter:
    """Return (or create) the module-level ``GrokAdapter``."""
    global _adapter
    if _adapter is None:
        _adapter = GrokAdapter()
    return _adapter
