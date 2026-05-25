"""Pydantic schemas for xAI/Grok structured outputs.

These models are used with ``client.responses.parse(..., text_format=Model)``
so extractor and judge responses conform to a schema before they enter the
workflow or evaluation artifacts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AssessmentExtraction(BaseModel):
    status: Literal[
        "assessed",
        "no_response",
        "stop_contact",
        "identity_unverified",
        "hardship_referral",
    ]
    identity_verified: bool
    financial_situation: str
    viable_path: Literal[
        "voice_resolution",
        "structured_payment_plan",
        "lump_sum",
        "hardship_referral",
    ]
    hardship_signal: bool = False
    distress_signal: bool = False
    stop_contact: bool = False
    debt_disputed: bool = False
    brief_summary: str = Field(description="Maximum two sentence factual summary")


class ResolutionExtraction(BaseModel):
    status: Literal[
        "deal_agreed",
        "no_deal",
        "call_not_answered",
        "call_dropped",
        "stop_contact",
        "hardship_referral",
        "human_review",
    ]
    selected_offer: str | None = None
    borrower_position: str
    objections: list[str] = Field(default_factory=list)
    latest_payment_capacity: str = "unknown"
    hardship_signal: bool = False
    distress_signal: bool = False
    stop_contact: bool = False
    brief_summary: str = Field(description="Maximum two sentence factual summary")


class FinalNoticeExtraction(BaseModel):
    status: Literal[
        "resolved",
        "no_resolution",
        "stop_contact",
        "hardship_referral",
        "human_review",
    ]
    final_action: str
    final_offer_referenced: str | None = None
    expiry: str | None = None
    documented_next_step: str | None = None
    borrower_response: str | None = None
    brief_summary: str = Field(description="Maximum two sentence factual summary")


class JudgeScoreOutput(BaseModel):
    resolution_score: float = Field(ge=0.0, le=1.0)
    continuity_score: float = Field(ge=0.0, le=1.0)
    tone_score: float = Field(ge=0.0, le=1.0)
    information_gathering: float = Field(ge=0.0, le=1.0)


class MetaJudgeFindingOutput(BaseModel):
    type: str
    description: str
    severity: Literal["low", "medium", "high"] = "medium"
    correction: str = ""


class MetaJudgeOutput(BaseModel):
    flaws_found: bool
    findings: list[MetaJudgeFindingOutput] = Field(default_factory=list)
