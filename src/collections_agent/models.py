from dataclasses import dataclass, field
from typing import Literal


AssessmentStatus = Literal[
    "assessed",
    "no_response",
    "stop_contact",
    "identity_unverified",
    "hardship_referral",
]
ResolutionStatus = Literal[
    "deal_agreed",
    "no_deal",
    "call_not_answered",
    "call_dropped",
    "stop_contact",
    "hardship_referral",
    "human_review",
]
FinalNoticeStatus = Literal[
    "resolved",
    "no_resolution",
    "stop_contact",
    "hardship_referral",
    "human_review",
]


@dataclass
class BorrowerCase:
    borrower_id: str
    company_name: str
    debt_amount_cents: int
    account_last4: str
    phone_number: str
    chat_thread_id: str


@dataclass
class AssessmentResult:
    status: AssessmentStatus
    identity_verified: bool
    financial_situation: str
    viable_path: str
    hardship_signal: bool = False
    distress_signal: bool = False
    stop_contact: bool = False
    debt_disputed: bool = False
    transcript_id: str | None = None
    brief_summary: str = ""
    transcript: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ResolutionResult:
    status: ResolutionStatus
    selected_offer: str | None
    borrower_position: str
    objections: list[str] = field(default_factory=list)
    latest_payment_capacity: str = "unknown"
    hardship_signal: bool = False
    distress_signal: bool = False
    stop_contact: bool = False
    call_session_id: str | None = None
    transcript_id: str | None = None
    brief_summary: str = ""
    call_transcript: list[dict[str, str]] = field(default_factory=list)


@dataclass
class FinalNoticeResult:
    status: FinalNoticeStatus
    final_action: str
    final_offer_referenced: str | None = None
    expiry: str | None = None
    documented_next_step: str | None = None
    borrower_response: str | None = None
    transcript_id: str | None = None
    brief_summary: str = ""
    transcript: list[dict[str, str]] = field(default_factory=list)


@dataclass
class HandoffSummary:
    text: str
    token_count: int


@dataclass
class BorrowerWorkflowResult:
    borrower_id: str
    outcome: str
    assessment_attempts: int
    handoff_token_counts: list[int]
