"""Starter — kick off a single borrower workflow (CLI)."""

from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv
from temporalio.client import Client

from collections_agent.models import BorrowerCase
from collections_agent.workflows import BorrowerCollectionsWorkflow


async def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    host = os.getenv("TEMPORAL_HOST", "localhost:7233")
    namespace = os.getenv("TEMPORAL_NAMESPACE", "default")
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE", "collections")

    client = await Client.connect(host, namespace=namespace)

    case = BorrowerCase(
        borrower_id="borrower-001",
        company_name="Riverline Collections",
        debt_amount_cents=500000,
        account_last4="4321",
        phone_number="+1-555-0100",
        chat_thread_id="thread-001",
    )

    handle = await client.start_workflow(
        BorrowerCollectionsWorkflow.run,
        case,
        id=f"collections-{case.borrower_id}",
        task_queue=task_queue,
    )

    print(f"Started workflow: {handle.id}")
    result = await handle.result()
    print(f"Outcome: {result.outcome}")
    print(f"Assessment attempts: {result.assessment_attempts}")
    print(f"Handoff token counts: {result.handoff_token_counts}")


if __name__ == "__main__":
    asyncio.run(main())
