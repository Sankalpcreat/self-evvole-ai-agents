"""Temporal workflow — borrower collections pipeline.

Orchestrates the 3-agent pipeline with outcome-based branching:
  Assessment (chat) → Resolution (voice) → Final Notice (chat)

Handles all terminal statuses:
  - stop_contact    → terminate immediately (compliance rule 3)
  - hardship_referral → terminate (compliance rule 5)
  - human_review    → flag and terminate
  - call_not_answered / call_dropped → handled within resolution activity
"""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from collections_agent.activities import (
        flag_for_legal_or_write_off,
        log_agreement,
        log_resolution,
        run_assessment_chat,
        run_final_notice_chat,
        run_resolution_voice,
        summarize_assessment,
        summarize_full_history,
    )
    from collections_agent.models import BorrowerCase, BorrowerWorkflowResult


# Timeouts per activity type
CHAT_TIMEOUT = timedelta(seconds=120)
VOICE_TIMEOUT = timedelta(minutes=5)
SUMMARY_TIMEOUT = timedelta(seconds=60)
LOG_TIMEOUT = timedelta(seconds=30)

ACTIVITY_RETRY = RetryPolicy(
    maximum_attempts=3,
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
)

# Terminal statuses that halt the pipeline
_TERMINAL_STATUSES = frozenset({"stop_contact", "hardship_referral", "human_review"})


@workflow.defn
class BorrowerCollectionsWorkflow:
    """One workflow per borrower.  Linear pipeline with outcome branching."""

    @workflow.run
    async def run(self, case: BorrowerCase) -> BorrowerWorkflowResult:

        # ── Stage 1: Assessment (Chat) ── max 3 attempts ──
        assessment_attempts = 0
        assessment = None

        for attempt in range(1, 4):
            assessment_attempts = attempt
            assessment = await workflow.execute_activity(
                run_assessment_chat,
                args=[case, attempt],
                start_to_close_timeout=CHAT_TIMEOUT,
                retry_policy=ACTIVITY_RETRY,
            )

            # Terminal statuses → exit the pipeline
            if assessment.status in _TERMINAL_STATUSES:
                return BorrowerWorkflowResult(
                    borrower_id=case.borrower_id,
                    outcome=f"terminated_{assessment.status}",
                    assessment_attempts=assessment_attempts,
                    handoff_token_counts=[],
                )

            if assessment.status == "assessed":
                break

            # "no_response" or "identity_unverified" → retry

        if assessment is None:
            raise RuntimeError("assessment activity did not return a result")

        # ── Summarise assessment for handoff (≤500 tokens) ──
        assessment_handoff = await workflow.execute_activity(
            summarize_assessment,
            assessment,
            start_to_close_timeout=SUMMARY_TIMEOUT,
            retry_policy=ACTIVITY_RETRY,
        )

        # ── Stage 2: Resolution (Voice) ──
        resolution = await workflow.execute_activity(
            run_resolution_voice,
            args=[case, assessment_handoff],
            start_to_close_timeout=VOICE_TIMEOUT,
            retry_policy=ACTIVITY_RETRY,
        )

        # Terminal statuses from voice
        if resolution.status in _TERMINAL_STATUSES:
            return BorrowerWorkflowResult(
                borrower_id=case.borrower_id,
                outcome=f"terminated_{resolution.status}",
                assessment_attempts=assessment_attempts,
                handoff_token_counts=[assessment_handoff.token_count],
            )

        # Deal agreed → log and exit
        if resolution.status == "deal_agreed":
            await workflow.execute_activity(
                log_agreement,
                args=[case, resolution],
                start_to_close_timeout=LOG_TIMEOUT,
                retry_policy=ACTIVITY_RETRY,
            )
            return BorrowerWorkflowResult(
                borrower_id=case.borrower_id,
                outcome="agreement_logged",
                assessment_attempts=assessment_attempts,
                handoff_token_counts=[assessment_handoff.token_count],
            )

        # No deal / call issues → proceed to final notice

        # ── Summarise full history for Agent 3 (≤500 tokens) ──
        full_handoff = await workflow.execute_activity(
            summarize_full_history,
            args=[assessment, assessment_handoff, resolution],
            start_to_close_timeout=SUMMARY_TIMEOUT,
            retry_policy=ACTIVITY_RETRY,
        )

        # ── Stage 3: Final Notice (Chat) ──
        final_notice = await workflow.execute_activity(
            run_final_notice_chat,
            args=[case, full_handoff],
            start_to_close_timeout=CHAT_TIMEOUT,
            retry_policy=ACTIVITY_RETRY,
        )

        # Terminal statuses from final notice
        if final_notice.status in _TERMINAL_STATUSES:
            return BorrowerWorkflowResult(
                borrower_id=case.borrower_id,
                outcome=f"terminated_{final_notice.status}",
                assessment_attempts=assessment_attempts,
                handoff_token_counts=[assessment_handoff.token_count, full_handoff.token_count],
            )

        if final_notice.status == "resolved":
            await workflow.execute_activity(
                log_resolution,
                args=[case, final_notice],
                start_to_close_timeout=LOG_TIMEOUT,
                retry_policy=ACTIVITY_RETRY,
            )
            outcome = "resolution_logged"
        else:
            await workflow.execute_activity(
                flag_for_legal_or_write_off,
                args=[case, final_notice],
                start_to_close_timeout=LOG_TIMEOUT,
                retry_policy=ACTIVITY_RETRY,
            )
            outcome = "flagged_for_legal_or_write_off"

        return BorrowerWorkflowResult(
            borrower_id=case.borrower_id,
            outcome=outcome,
            assessment_attempts=assessment_attempts,
            handoff_token_counts=[assessment_handoff.token_count, full_handoff.token_count],
        )
