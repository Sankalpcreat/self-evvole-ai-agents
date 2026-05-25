"""Worker entrypoint — registers all activities and starts the Temporal worker."""

from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv
from temporalio.client import Client
from temporalio.worker import Worker

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
from collections_agent.workflows import BorrowerCollectionsWorkflow

logger = logging.getLogger(__name__)

ALL_ACTIVITIES = [
    run_assessment_chat,
    summarize_assessment,
    run_resolution_voice,
    summarize_full_history,
    run_final_notice_chat,
    log_agreement,
    log_resolution,
    flag_for_legal_or_write_off,
]


async def run_worker() -> None:
    load_dotenv()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    host = os.getenv("TEMPORAL_HOST", "localhost:7233")
    namespace = os.getenv("TEMPORAL_NAMESPACE", "default")
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE", "collections")

    logger.info("Connecting to Temporal at %s (ns=%s, queue=%s)", host, namespace, task_queue)
    client = await Client.connect(host, namespace=namespace)

    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=100)

    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[BorrowerCollectionsWorkflow],
        activities=ALL_ACTIVITIES,
        activity_executor=executor,
    )

    logger.info("Worker started — listening on queue '%s'", task_queue)
    await worker.run()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
