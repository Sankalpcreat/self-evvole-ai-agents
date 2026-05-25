"""FastAPI application — live conversation triggering + status.

Endpoints
---------
POST /trigger          Start a borrower workflow
GET  /status/{id}      Check workflow status
GET  /health           Health check
GET  /versions/{agent} List prompt versions
POST /evaluate         Run evaluation pipeline
POST /voice/agent2-demo Generate Agent 2 TTS demo artifact
GET  /cost             Get cumulative LLM cost
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from temporalio.client import Client as TemporalClient

from collections_agent.evaluator import ConversationEvaluator
from collections_agent.llm_adapter import get_adapter
from collections_agent.models import BorrowerCase, BorrowerWorkflowResult
from collections_agent.prompt_manager import PromptManager
from collections_agent.run_pipeline import (
    _default_case,
    run_baseline_evaluation,
    run_meta_evaluation,
    run_self_learning,
)
from collections_agent.voice_generator import DATA_DIR as AUDIO_DATA_DIR
from collections_agent.voice_generator import generate_agent2_demo_audio
from collections_agent.voice_session import (
    audio_path_for,
    create_voice_session,
    get_voice_session,
    process_voice_turn,
    voice_web_html,
)
from collections_agent.workflows import BorrowerCollectionsWorkflow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models for the API
# ---------------------------------------------------------------------------

class TriggerRequest(BaseModel):
    borrower_id: str
    company_name: str = "Riverline Collections"
    debt_amount_cents: int = 500000  # $5,000 default
    account_last4: str = "4321"
    phone_number: str = "+1-555-0100"


class TriggerResponse(BaseModel):
    workflow_id: str
    run_id: str
    status: str = "started"


class StatusResponse(BaseModel):
    borrower_id: str
    status: str
    outcome: str | None = None
    assessment_attempts: int | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    temporal_connected: bool = False


class EvaluateRequest(BaseModel):
    seed: int = 42
    run_learning: bool = False
    run_meta: bool = True
    iterations: int = 1
    conversations: int = 3
    budget_usd: float = 5.0


class EvaluateResponse(BaseModel):
    status: str
    scores_count: int
    aggregate: dict
    cost: dict


class VoiceDemoRequest(BaseModel):
    text: str | None = None
    voice_id: str = "rex"
    language: str = "en"


class VoiceDemoResponse(BaseModel):
    status: str
    audio_path: str
    voice_id: str


class VoiceSessionCreateRequest(BaseModel):
    borrower_id: str = "voice-demo-borrower-001"
    company_name: str = "Riverline Collections"
    debt_amount_cents: int = 500000
    account_last4: str = "4321"
    phone_number: str = "+1-555-0100"
    handoff_text: str | None = None
    voice_id: str = "rex"
    language: str = "en"


class VoiceSessionResponse(BaseModel):
    session_id: str
    status: str
    audio_url: str | None = None
    transcript: list[dict[str, str]]


class VoiceTurnResponse(BaseModel):
    session_id: str
    borrower_text: str
    agent_text: str
    audio_url: str
    transcript: list[dict[str, str]]
    status: str


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

_temporal_client: TemporalClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Connect to Temporal on startup."""
    global _temporal_client
    host = os.getenv("TEMPORAL_HOST", "localhost:7233")
    try:
        _temporal_client = await TemporalClient.connect(host)
        logger.info("Connected to Temporal at %s", host)
    except Exception:
        logger.warning("Could not connect to Temporal at %s", host, exc_info=True)
    yield
    # Cleanup
    _temporal_client = None


app = FastAPI(
    title="Riverline Collections API",
    description="Self-learning AI debt collections system",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        temporal_connected=_temporal_client is not None,
    )


@app.post("/trigger", response_model=TriggerResponse)
async def trigger_workflow(req: TriggerRequest):
    """Start a new borrower collections workflow."""
    if _temporal_client is None:
        raise HTTPException(503, "Temporal not connected")

    case = BorrowerCase(
        borrower_id=req.borrower_id,
        company_name=req.company_name,
        debt_amount_cents=req.debt_amount_cents,
        account_last4=req.account_last4,
        phone_number=req.phone_number,
        chat_thread_id=f"thread-{req.borrower_id}",
    )

    task_queue = os.getenv("TEMPORAL_TASK_QUEUE", "collections")

    handle = await _temporal_client.start_workflow(
        BorrowerCollectionsWorkflow.run,
        case,
        id=f"collections-{req.borrower_id}",
        task_queue=task_queue,
    )

    return TriggerResponse(
        workflow_id=handle.id,
        run_id=handle.result_run_id,
    )


@app.get("/status/{borrower_id}", response_model=StatusResponse)
async def get_status(borrower_id: str):
    """Check the status of a borrower's workflow."""
    if _temporal_client is None:
        raise HTTPException(503, "Temporal not connected")

    workflow_id = f"collections-{borrower_id}"
    try:
        handle = _temporal_client.get_workflow_handle(workflow_id)
        desc = await handle.describe()
        status = getattr(desc.status, "name", str(desc.status))

        # Try to get result if completed
        outcome = None
        attempts = None
        if status == "COMPLETED":
            try:
                result = await handle.result()
                if isinstance(result, dict):
                    outcome = result.get("outcome")
                    attempts = result.get("assessment_attempts")
                else:
                    outcome = result.outcome
                    attempts = result.assessment_attempts
            except Exception:
                pass

        return StatusResponse(
            borrower_id=borrower_id,
            status=status,
            outcome=outcome,
            assessment_attempts=attempts,
        )
    except Exception as exc:
        raise HTTPException(404, f"Workflow not found: {exc}") from exc


@app.get("/versions/{agent_name}")
async def list_versions(agent_name: str):
    """List all prompt versions for an agent."""
    pm = PromptManager()
    try:
        versions = pm.list_versions(agent_name)
        return [
            {
                "version_id": v.version_id,
                "created_at": v.created_at,
                "is_active": v.is_active,
                "eval_data": v.eval_data,
            }
            for v in versions
        ]
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/cost")
async def get_cost():
    """Return cumulative LLM API cost."""
    adapter = get_adapter()
    return adapter.cumulative_usage


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest):
    """Run the reproducible full-system evaluation pipeline."""
    adapter = get_adapter()
    case = _default_case()

    scores = await run_in_threadpool(
        run_baseline_evaluation,
        adapter,
        case,
        req.seed,
    )

    evaluator = ConversationEvaluator(adapter)
    aggregate = evaluator.aggregate(scores)

    if req.run_learning:
        await run_in_threadpool(
            run_self_learning,
            adapter,
            case,
            req.iterations,
            req.conversations,
            req.budget_usd,
        )

    if req.run_meta:
        await run_in_threadpool(run_meta_evaluation, adapter)

    return EvaluateResponse(
        status="completed",
        scores_count=len(scores),
        aggregate=aggregate,
        cost=adapter.cumulative_usage,
    )


@app.post("/voice/agent2-demo", response_model=VoiceDemoResponse)
async def create_agent2_voice_demo(req: VoiceDemoRequest):
    """Generate an xAI TTS audio artifact for Agent 2."""
    try:
        path = await run_in_threadpool(
            generate_agent2_demo_audio,
            req.text,
            None,
            voice_id=req.voice_id,
            language=req.language,
        )
    except Exception as exc:
        raise HTTPException(502, f"Voice generation failed: {exc}") from exc

    return VoiceDemoResponse(
        status="created",
        audio_path=str(path),
        voice_id=req.voice_id,
    )


@app.get("/voice/agent2-demo/file")
async def get_latest_agent2_voice_demo():
    """Download the latest generated Agent 2 demo audio artifact."""
    files = sorted(AUDIO_DATA_DIR.glob("agent2_demo_*.mp3"), reverse=True)
    if not files:
        raise HTTPException(404, "No Agent 2 voice demo has been generated yet")
    return FileResponse(files[0], media_type="audio/mpeg", filename=files[0].name)


@app.get("/voice/agent2-web", response_class=HTMLResponse)
async def agent2_voice_web():
    """Browser-based live Agent 2 voice surface."""
    return HTMLResponse(voice_web_html())


@app.post("/voice/sessions", response_model=VoiceSessionResponse)
async def start_voice_session(req: VoiceSessionCreateRequest):
    """Start a live web voice session for Agent 2 and return opening audio."""
    case = BorrowerCase(
        borrower_id=req.borrower_id,
        company_name=req.company_name,
        debt_amount_cents=req.debt_amount_cents,
        account_last4=req.account_last4,
        phone_number=req.phone_number,
        chat_thread_id=f"voice-thread-{req.borrower_id}",
    )
    try:
        session, _audio = await run_in_threadpool(
            create_voice_session,
            case,
            handoff_text=req.handoff_text,
            voice_id=req.voice_id,
            language=req.language,
        )
    except Exception as exc:
        raise HTTPException(502, f"Could not start voice session: {exc}") from exc

    return VoiceSessionResponse(
        session_id=session.session_id,
        status=session.status,
        audio_url=f"/voice/sessions/{session.session_id}/audio/agent2_turn_01.mp3",
        transcript=session.transcript,
    )


@app.get("/voice/sessions/{session_id}", response_model=VoiceSessionResponse)
async def get_agent2_voice_session(session_id: str):
    """Return live Agent 2 voice session state."""
    try:
        session = get_voice_session(session_id)
    except KeyError as exc:
        raise HTTPException(404, "Voice session not found") from exc
    latest_audio = None
    agent_turns = len([item for item in session.transcript if item.get("role") == "agent"])
    if agent_turns:
        latest_audio = f"/voice/sessions/{session.session_id}/audio/agent2_turn_{agent_turns:02d}.mp3"
    return VoiceSessionResponse(
        session_id=session.session_id,
        status=session.status,
        audio_url=latest_audio,
        transcript=session.transcript,
    )


@app.post("/voice/sessions/{session_id}/turn", response_model=VoiceTurnResponse)
async def process_agent2_voice_turn(session_id: str, file: UploadFile = File(...)):
    """Accept borrower audio, transcribe it, run Agent 2, and return TTS audio."""
    audio = await file.read()
    try:
        result = await run_in_threadpool(
            process_voice_turn,
            session_id,
            audio,
            file.filename or "borrower.ogg",
            file.content_type,
        )
    except KeyError as exc:
        raise HTTPException(404, "Voice session not found") from exc
    except Exception as exc:
        raise HTTPException(502, f"Voice turn failed: {exc}") from exc
    return VoiceTurnResponse(**result)


@app.get("/voice/sessions/{session_id}/audio/{filename}")
async def get_voice_session_audio(session_id: str, filename: str):
    """Download a generated Agent 2 audio turn."""
    try:
        path = audio_path_for(session_id, filename)
    except ValueError as exc:
        raise HTTPException(400, "Invalid audio path") from exc
    if not path.exists():
        raise HTTPException(404, "Audio file not found")
    return FileResponse(path, media_type="audio/mpeg", filename=path.name)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the API server."""
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    uvicorn.run(
        "collections_agent.api:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
