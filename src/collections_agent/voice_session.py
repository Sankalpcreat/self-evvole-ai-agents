"""Live web voice loop for Agent 2.

This is intentionally small: browser audio is uploaded to FastAPI, the server
transcribes with xAI STT, runs Agent 2 through Grok with policy context, then
returns xAI TTS audio for browser playback. API keys never leave the server.
"""

from __future__ import annotations

import json
import mimetypes
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

from collections_agent.compliance_checker import ComplianceChecker
from collections_agent.llm_adapter import GrokAdapter, get_adapter
from collections_agent.models import BorrowerCase
from collections_agent.prompt_manager import PromptManager
from collections_agent.token_counter import enforce_agent_budget
from collections_agent.voice_generator import generate_tts_bytes

STT_URL = "https://api.x.ai/v1/stt"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "voice_sessions"
MAX_COMPLIANCE_RETRIES = 2


@dataclass
class VoiceSession:
    session_id: str
    case: BorrowerCase
    handoff_text: str
    voice_id: str = "rex"
    language: str = "en"
    transcript: list[dict[str, str]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "active"


_sessions: dict[str, VoiceSession] = {}
_prompt_mgr = PromptManager()
_compliance = ComplianceChecker()


def default_agent2_handoff() -> str:
    """Return a compact demo handoff from Agent 1 to Agent 2."""
    return (
        "Identity verified using account ending 4321. Borrower owes $5,000. "
        "Borrower said they may be able to pay something but cannot pay the full "
        "balance today. No stop-contact request. No dispute recorded. Agent 2 "
        "should present policy-approved lump-sum, payment plan, and hardship-review "
        "options without re-verifying identity."
    )


def create_voice_session(
    case: BorrowerCase,
    *,
    handoff_text: str | None = None,
    voice_id: str = "rex",
    language: str = "en",
    adapter: GrokAdapter | None = None,
) -> tuple[VoiceSession, bytes]:
    """Create a live Agent 2 voice session and return opening audio bytes."""
    session = VoiceSession(
        session_id=str(uuid.uuid4()),
        case=case,
        handoff_text=handoff_text or default_agent2_handoff(),
        voice_id=voice_id,
        language=language,
    )
    _sessions[session.session_id] = session

    opening = _generate_agent_response(
        session,
        adapter or get_adapter(),
        "Start the Agent 2 resolution voice call. Do not ask for information already present in the handoff.",
    )
    audio = _save_turn_audio(session, opening)
    _persist_session(session)
    return session, audio


def get_voice_session(session_id: str) -> VoiceSession:
    """Return a voice session from memory or disk."""
    if session_id in _sessions:
        return _sessions[session_id]

    path = _session_dir(session_id) / "session.json"
    if not path.exists():
        raise KeyError(session_id)
    raw = json.loads(path.read_text())
    case = BorrowerCase(**raw["case"])
    session = VoiceSession(
        session_id=raw["session_id"],
        case=case,
        handoff_text=raw["handoff_text"],
        voice_id=raw.get("voice_id", "rex"),
        language=raw.get("language", "en"),
        transcript=raw.get("transcript", []),
        created_at=raw.get("created_at", datetime.now(timezone.utc).isoformat()),
        updated_at=raw.get("updated_at", datetime.now(timezone.utc).isoformat()),
        status=raw.get("status", "active"),
    )
    _sessions[session_id] = session
    return session


def process_voice_turn(
    session_id: str,
    audio_bytes: bytes,
    filename: str,
    content_type: str | None,
    *,
    adapter: GrokAdapter | None = None,
) -> dict:
    """Process one borrower audio turn and return transcript + audio metadata."""
    session = get_voice_session(session_id)
    borrower_text = transcribe_audio(audio_bytes, filename, content_type, language=session.language)
    session.transcript.append({"role": "borrower", "content": borrower_text})

    agent_text = _generate_agent_response(
        session,
        adapter or get_adapter(),
        "Respond to the borrower's latest spoken turn as Agent 2.",
    )
    audio = _save_turn_audio(session, agent_text)
    _persist_session(session)

    return {
        "session_id": session.session_id,
        "borrower_text": borrower_text,
        "agent_text": agent_text,
        "audio_path": str(audio),
        "audio_url": f"/voice/sessions/{session.session_id}/audio/{audio.name}",
        "transcript": session.transcript,
        "status": session.status,
    }


def transcribe_audio(
    audio_bytes: bytes,
    filename: str,
    content_type: str | None,
    *,
    language: str = "en",
    timeout_seconds: float = 900.0,
) -> str:
    """Transcribe uploaded borrower audio with xAI STT."""
    key = os.getenv("XAI_API_KEY")
    if not key:
        raise RuntimeError("XAI_API_KEY is required for xAI STT")
    if not audio_bytes:
        raise ValueError("Uploaded audio is empty")

    guessed_type = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    files = [
        ("format", (None, "true")),
        ("language", (None, language)),
        ("file", (filename, audio_bytes, guessed_type)),
    ]
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(
            STT_URL,
            headers={"Authorization": f"Bearer {key}"},
            files=files,
        )
    response.raise_for_status()
    data = response.json()
    text = str(data.get("text", "")).strip()
    if not text:
        raise RuntimeError("xAI STT returned an empty transcript")
    return text


def audio_path_for(session_id: str, filename: str) -> Path:
    """Return a generated audio path under the requested session directory."""
    path = _session_dir(session_id) / filename
    root = _session_dir(session_id).resolve()
    resolved = path.resolve()
    if root not in resolved.parents and resolved != root:
        raise ValueError("Invalid audio path")
    return resolved


def voice_web_html() -> str:
    """Return a minimal browser UI for the live Agent 2 voice loop."""
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent 2 Voice</title>
  <style>
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f6f7f9; color: #171a1f; }
    main { max-width: 920px; margin: 0 auto; padding: 28px; }
    h1 { font-size: 24px; margin: 0 0 18px; }
    .bar { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 18px; }
    button { border: 1px solid #c8ccd4; background: #fff; padding: 10px 14px; border-radius: 6px; cursor: pointer; font-weight: 600; }
    button:disabled { opacity: .45; cursor: not-allowed; }
    #status { min-height: 22px; color: #4b5563; }
    .panel { background: #fff; border: 1px solid #d8dce3; border-radius: 8px; padding: 16px; }
    .turn { border-top: 1px solid #e5e7eb; padding: 12px 0; }
    .turn:first-child { border-top: 0; }
    .role { font-size: 12px; font-weight: 700; text-transform: uppercase; color: #5b6472; margin-bottom: 4px; }
    audio { width: 100%; margin: 12px 0 18px; }
  </style>
</head>
<body>
<main>
  <h1>Agent 2 Resolution Voice</h1>
  <div class="bar">
    <button id="start">Start Session</button>
    <button id="record" disabled>Record Borrower</button>
    <button id="stop" disabled>Stop & Send</button>
  </div>
  <div id="status"></div>
  <audio id="player" controls></audio>
  <section class="panel" id="transcript"></section>
</main>
<script>
let sessionId = null;
let recorder = null;
let chunks = [];

const statusEl = document.getElementById("status");
const player = document.getElementById("player");
const transcriptEl = document.getElementById("transcript");
const startBtn = document.getElementById("start");
const recordBtn = document.getElementById("record");
const stopBtn = document.getElementById("stop");

function setStatus(text) { statusEl.textContent = text; }

function renderTranscript(items) {
  transcriptEl.innerHTML = "";
  for (const item of items || []) {
    const row = document.createElement("div");
    row.className = "turn";
    row.innerHTML = `<div class="role">${item.role}</div><div>${item.content}</div>`;
    transcriptEl.appendChild(row);
  }
}

async function playAudio(url) {
  player.src = url + `?t=${Date.now()}`;
  await player.play().catch(() => {});
}

startBtn.onclick = async () => {
  setStatus("Creating voice session...");
  const res = await fetch("/voice/sessions", { method: "POST", headers: {"Content-Type": "application/json"}, body: "{}" });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  sessionId = data.session_id;
  renderTranscript(data.transcript);
  await playAudio(data.audio_url);
  recordBtn.disabled = false;
  startBtn.disabled = true;
  setStatus("Session ready. Record the borrower response.");
};

recordBtn.onclick = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const preferred = ["audio/ogg;codecs=opus", "audio/mp4", "audio/webm;codecs=opus"].find(t => MediaRecorder.isTypeSupported(t));
  recorder = new MediaRecorder(stream, preferred ? { mimeType: preferred } : undefined);
  chunks = [];
  recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
  recorder.start();
  recordBtn.disabled = true;
  stopBtn.disabled = false;
  setStatus("Recording...");
};

stopBtn.onclick = async () => {
  setStatus("Uploading audio and waiting for Agent 2...");
  stopBtn.disabled = true;
  await new Promise(resolve => { recorder.onstop = resolve; recorder.stop(); });
  const blob = new Blob(chunks, { type: recorder.mimeType || "audio/ogg" });
  const form = new FormData();
  const ext = blob.type.includes("mp4") ? "mp4" : (blob.type.includes("webm") ? "webm" : "ogg");
  form.append("file", blob, `borrower.${ext}`);
  const res = await fetch(`/voice/sessions/${sessionId}/turn`, { method: "POST", body: form });
  if (!res.ok) {
    setStatus(await res.text());
    recordBtn.disabled = false;
    return;
  }
  const data = await res.json();
  renderTranscript(data.transcript);
  await playAudio(data.audio_url);
  recordBtn.disabled = false;
  setStatus("Agent response ready. Record another borrower response when needed.");
};
</script>
</body>
</html>"""


def _generate_agent_response(session: VoiceSession, adapter: GrokAdapter, instruction: str) -> str:
    system_prompt = _build_agent2_prompt(session.case, session.handoff_text)
    messages = _messages_for_agent(session, instruction)
    response = adapter.chat(system_prompt, messages)
    agent_text = response.text.strip()

    is_opening = not any(item.get("role") == "agent" for item in session.transcript)
    borrower_text = "\n".join(
        item.get("content", "") for item in session.transcript if item.get("role") == "borrower"
    )
    borrower_said_stop = _compliance.detect_borrower_stop(borrower_text)
    borrower_in_distress = _compliance.detect_borrower_distress(borrower_text)

    result = _compliance.check_response(
        agent_text,
        "agent2",
        is_opening_message=is_opening,
        borrower_said_stop=borrower_said_stop,
        borrower_in_distress=borrower_in_distress,
    )
    for _ in range(MAX_COMPLIANCE_RETRIES):
        if result.passed:
            break
        violations = "; ".join(v.description for v in result.violations)
        repair_messages = messages + [
            {"role": "assistant", "content": agent_text},
            {
                "role": "user",
                "content": (
                    "Rewrite your last Agent 2 voice response to fix these compliance "
                    f"violations: {violations}. Keep it concise and policy-bounded."
                ),
            },
        ]
        response = adapter.chat(system_prompt, repair_messages)
        agent_text = response.text.strip()
        result = _compliance.check_response(
            agent_text,
            "agent2",
            is_opening_message=is_opening,
            borrower_said_stop=borrower_said_stop,
            borrower_in_distress=borrower_in_distress,
        )

    if not result.passed:
        if borrower_said_stop:
            agent_text = (
                "I acknowledge your request to stop contact. I will flag the account "
                "accordingly and end this conversation now."
            )
            session.status = "stop_contact"
        elif borrower_in_distress:
            agent_text = (
                "I hear that you are experiencing hardship. I can route this account "
                "to the hardship review program before discussing payment terms further."
            )
            session.status = "hardship_referral"
        else:
            agent_text = (
                "This is a Riverline Collections AI agent. This call is recorded and logged. "
                "I cannot proceed with a non-compliant response, so I am flagging this call "
                "for human review."
            )
            session.status = "human_review"

    session.transcript.append({"role": "agent", "content": agent_text})
    session.updated_at = datetime.now(timezone.utc).isoformat()
    return agent_text


def _messages_for_agent(session: VoiceSession, instruction: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in session.transcript[-12:]:
        role = "assistant" if item.get("role") == "agent" else "user"
        messages.append({"role": role, "content": item.get("content", "")})
    messages.append({"role": "user", "content": instruction})
    return messages


def _build_agent2_prompt(case: BorrowerCase, handoff_text: str) -> str:
    prompt = _prompt_mgr.load_prompt("agent2")
    policy_path = Path(__file__).resolve().parent.parent.parent / "policy" / "settlement_policy.json"
    policy = json.loads(policy_path.read_text())
    context = (
        "\n\n## Current Case\n"
        f"- Borrower ID: {case.borrower_id}\n"
        f"- Company: {case.company_name}\n"
        f"- Debt: ${case.debt_amount_cents / 100:,.2f}\n"
        f"- Account last 4: {case.account_last4}\n"
        f"\n## Handoff from Agent 1\n{handoff_text}\n"
        f"\n## Settlement Policy\n```json\n{json.dumps(policy, indent=2)}\n```\n"
    )
    full_prompt = prompt + context
    enforce_agent_budget(full_prompt, agent_name="agent2")
    return full_prompt


def _save_turn_audio(session: VoiceSession, text: str) -> Path:
    audio = generate_tts_bytes(text, voice_id=session.voice_id, language=session.language)
    turn_number = len([item for item in session.transcript if item.get("role") == "agent"])
    path = _session_dir(session.session_id) / f"agent2_turn_{turn_number:02d}.mp3"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audio)
    return path


def _session_dir(session_id: str) -> Path:
    return DATA_DIR / session_id


def _persist_session(session: VoiceSession) -> None:
    path = _session_dir(session.session_id) / "session.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(session), indent=2))
