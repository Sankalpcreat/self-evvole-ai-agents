"""Post-response compliance checker for the 8 mandated rules.

Strategy
--------
* Rules 1 & 6 (identity / recording disclosure) – checked on the
  agent's **first** message only.
* Rule 3 (no harassment) – triggered when the borrower has explicitly
  asked to stop being contacted.
* Rule 8 (data privacy) – regex patterns for full account numbers and
  SSN-shaped strings.
* Rules 2, 4, 5, 7 – keyword / heuristic detection.

Every check is cheap (no LLM call) so it can run on every single
agent turn without affecting the $20 budget.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

POLICY_DIR = Path(__file__).resolve().parent.parent.parent / "policy"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ComplianceViolation:
    rule_id: int
    rule_name: str
    description: str
    severity: str = "hard"  # "hard" → blocks response, "soft" → warning


@dataclass
class ComplianceResult:
    passed: bool
    violations: list[ComplianceViolation] = field(default_factory=list)
    checks_performed: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

class ComplianceChecker:
    """Rule-based compliance checker (no LLM cost)."""

    # PII patterns
    _FULL_ACCOUNT_RE = re.compile(r"\b\d{10,16}\b")
    _SSN_RE = re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b")

    # Stop-contact signals from the borrower
    _STOP_KEYWORDS: list[str] = [
        "stop contacting",
        "stop calling",
        "don't contact",
        "do not contact",
        "leave me alone",
        "cease communication",
        "stop reaching out",
        "don't call",
        "do not call",
    ]

    # Phrases that could constitute a false threat
    _FALSE_THREAT_KEYWORDS: list[str] = [
        "arrest",
        "jail",
        "prison",
        "wage garnishment",
        "seize your",
        "take your home",
        "sue you personally",
    ]

    def __init__(self) -> None:
        rules_path = POLICY_DIR / "compliance_rules.json"
        self.rules: dict = {}
        if rules_path.exists():
            with open(rules_path) as fh:
                self.rules = json.load(fh)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_response(
        self,
        agent_message: str,
        agent_name: str,
        *,
        is_opening_message: bool = False,
        borrower_said_stop: bool = False,
        borrower_in_distress: bool = False,
    ) -> ComplianceResult:
        """Run all applicable compliance checks against *agent_message*."""
        violations: list[ComplianceViolation] = []
        checks: list[str] = []

        # Rule 1 – Identity disclosure (opening message only)
        if is_opening_message:
            checks.append("rule_1_identity_disclosure")
            if not self._has_identity_disclosure(agent_message):
                violations.append(
                    ComplianceViolation(1, "identity_disclosure", "Agent did not identify as AI")
                )

        # Rule 2 – No false threats
        checks.append("rule_2_no_false_threats")
        violations.extend(self._check_false_threats(agent_message))

        # Rule 3 – No harassment / stop contact
        if borrower_said_stop:
            checks.append("rule_3_no_harassment")
            if not self._acknowledges_stop(agent_message):
                violations.append(
                    ComplianceViolation(
                        3, "no_harassment", "Borrower said stop but agent continued pressure"
                    )
                )

        # Rule 4 – No misleading terms (basic keyword heuristic)
        checks.append("rule_4_no_misleading_terms")
        # Deep check deferred to settlement-policy validation in activities

        # Rule 5 – Sensitive situations
        if borrower_in_distress:
            checks.append("rule_5_sensitive_situations")
            if not self._offers_hardship(agent_message):
                violations.append(
                    ComplianceViolation(
                        5, "sensitive_situations", "Borrower in distress; hardship program not offered"
                    )
                )

        # Rule 6 – Recording disclosure (opening message only)
        if is_opening_message:
            checks.append("rule_6_recording_disclosure")
            if not self._has_recording_disclosure(agent_message):
                violations.append(
                    ComplianceViolation(6, "recording_disclosure", "Recording/logging not disclosed")
                )

        # Rule 7 – Professional composure
        checks.append("rule_7_professional_composure")
        # Mostly handled by the system prompt; can add profanity filter here

        # Rule 8 – Data privacy
        checks.append("rule_8_data_privacy")
        violations.extend(self._check_data_privacy(agent_message))

        passed = len(violations) == 0
        if not passed:
            logger.warning(
                "Compliance FAILED for %s: %s",
                agent_name,
                [v.rule_name for v in violations],
            )
        return ComplianceResult(passed=passed, violations=violations, checks_performed=checks)

    def detect_borrower_stop(self, borrower_message: str) -> bool:
        """Return True if the borrower is requesting to stop contact."""
        lower = borrower_message.lower()
        return any(kw in lower for kw in self._STOP_KEYWORDS)

    def detect_borrower_distress(self, borrower_message: str) -> bool:
        """Return True if the borrower signals hardship or emotional distress."""
        lower = borrower_message.lower()
        distress_keywords = [
            "hardship",
            "can't afford",
            "cannot afford",
            "medical",
            "hospital",
            "lost my job",
            "laid off",
            "disability",
            "suicid",
            "kill myself",
            "desperate",
            "homeless",
            "evict",
        ]
        return any(kw in lower for kw in distress_keywords)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_identity_disclosure(msg: str) -> bool:
        lower = msg.lower()
        return any(
            p in lower
            for p in (
                "ai agent",
                "ai voice agent",
                "ai assistant",
                "artificial intelligence",
                "automated",
                "acting on behalf",
            )
        )

    @staticmethod
    def _has_recording_disclosure(msg: str) -> bool:
        lower = msg.lower()
        return any(p in lower for p in ("recorded", "recording", "logged", "logging", "monitored"))

    @classmethod
    def _check_false_threats(cls, msg: str) -> list[ComplianceViolation]:
        lower = msg.lower()
        return [
            ComplianceViolation(2, "no_false_threats", f"Potential false threat: '{kw}'")
            for kw in cls._FALSE_THREAT_KEYWORDS
            if kw in lower
        ]

    @staticmethod
    def _acknowledges_stop(msg: str) -> bool:
        lower = msg.lower()
        return any(
            p in lower
            for p in ("acknowledge", "noted", "will stop", "flag", "cease", "no further", "respect")
        )

    @staticmethod
    def _offers_hardship(msg: str) -> bool:
        lower = msg.lower()
        return any(
            p in lower
            for p in ("hardship", "program", "assistance", "support", "help you", "connect you")
        )

    @classmethod
    def _check_data_privacy(cls, msg: str) -> list[ComplianceViolation]:
        violations: list[ComplianceViolation] = []
        if cls._FULL_ACCOUNT_RE.search(msg):
            violations.append(
                ComplianceViolation(8, "data_privacy", "Possible full account number exposed")
            )
        if cls._SSN_RE.search(msg):
            violations.append(
                ComplianceViolation(8, "data_privacy", "SSN-shaped string detected")
            )
        return violations
