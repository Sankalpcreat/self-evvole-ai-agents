"""xAI text-to-speech helpers for the Agent 2 voice artifact.

The learning loop evaluates Agent 2 in text mode for cost and repeatability.
This module produces the required voice demo artifact from the same Agent 2
script using xAI's server-side TTS endpoint.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from typing import Any
from pathlib import Path

import httpx
from dotenv import load_dotenv

TTS_URL = "https://api.x.ai/v1/tts"
MAX_TTS_CHARS = 15_000
DEFAULT_VOICE_ID = "rex"
DEFAULT_LANGUAGE = "en"
DEFAULT_CODEC = "mp3"

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "audio"


def agent2_demo_script() -> str:
    """Return a compliant Agent 2 sample script suitable for TTS."""
    return (
        "Hello, this is a Riverline Collections AI agent. This call may be recorded "
        "and logged for quality and compliance. I am calling about the account ending "
        "in 4321 with a current balance of five thousand dollars. "
        "Based on the prior assessment, I can document one of two policy-approved "
        "paths today. First, a lump-sum settlement of three thousand five hundred "
        "dollars, which is seventy percent of the balance, if committed within "
        "forty-eight hours. Second, a payment plan with at least two hundred fifty "
        "dollars down, at least two hundred fifty dollars per month, and a maximum "
        "term of eighteen months. If you are experiencing hardship, I can also route "
        "the account to the hardship review program before any payment arrangement "
        "is finalized. Which of those options is realistic for you today?"
    )


def _default_output_path(codec: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "mp3" if codec == "mp3" else codec
    return DATA_DIR / f"agent2_demo_{ts}.{suffix}"


def generate_tts_audio(
    text: str,
    output_path: Path | str | None = None,
    *,
    voice_id: str = DEFAULT_VOICE_ID,
    language: str = DEFAULT_LANGUAGE,
    codec: str = DEFAULT_CODEC,
    api_key: str | None = None,
    timeout_seconds: float = 900.0,
) -> Path:
    """Generate TTS audio with xAI and save the raw audio bytes locally."""
    if not text.strip():
        raise ValueError("TTS text must not be empty")
    if len(text) > MAX_TTS_CHARS:
        raise ValueError(f"TTS text exceeds {MAX_TTS_CHARS} characters")

    key = api_key or os.getenv("XAI_API_KEY")
    if not key:
        raise RuntimeError("XAI_API_KEY is required for xAI TTS generation")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_path) if output_path else _default_output_path(codec)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "text": text,
        "voice_id": voice_id,
        "language": language,
        "output_format": {
            "codec": codec,
            "sample_rate": 24000,
        },
    }
    if codec == "mp3":
        payload["output_format"]["bit_rate"] = 128000

    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(
            TTS_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    response.raise_for_status()
    out_path.write_bytes(response.content)
    return out_path


def generate_tts_bytes(
    text: str,
    *,
    voice_id: str = DEFAULT_VOICE_ID,
    language: str = DEFAULT_LANGUAGE,
    codec: str = DEFAULT_CODEC,
    api_key: str | None = None,
    timeout_seconds: float = 900.0,
) -> bytes:
    """Generate TTS audio and return raw bytes without writing a file."""
    if not text.strip():
        raise ValueError("TTS text must not be empty")
    if len(text) > MAX_TTS_CHARS:
        raise ValueError(f"TTS text exceeds {MAX_TTS_CHARS} characters")

    key = api_key or os.getenv("XAI_API_KEY")
    if not key:
        raise RuntimeError("XAI_API_KEY is required for xAI TTS generation")

    payload: dict[str, Any] = {
        "text": text,
        "voice_id": voice_id,
        "language": language,
        "output_format": {
            "codec": codec,
            "sample_rate": 24000,
        },
    }
    if codec == "mp3":
        payload["output_format"]["bit_rate"] = 128000

    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(
            TTS_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    response.raise_for_status()
    return response.content


def generate_agent2_demo_audio(
    text: str | None = None,
    output_path: Path | str | None = None,
    *,
    voice_id: str = DEFAULT_VOICE_ID,
    language: str = DEFAULT_LANGUAGE,
) -> Path:
    """Generate the required Agent 2 voice demo artifact."""
    return generate_tts_audio(
        text or agent2_demo_script(),
        output_path,
        voice_id=voice_id,
        language=language,
        codec=DEFAULT_CODEC,
    )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generate xAI TTS audio for Agent 2")
    parser.add_argument("--text", help="Text to synthesize. Defaults to Agent 2 demo script.")
    parser.add_argument("--text-file", help="Path to a UTF-8 text file to synthesize.")
    parser.add_argument("--output", help="Output audio path. Defaults to data/audio/*.mp3")
    parser.add_argument("--voice-id", default=DEFAULT_VOICE_ID, help="xAI voice ID")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="BCP-47 language code")
    args = parser.parse_args()

    text = args.text
    if args.text_file:
        text = Path(args.text_file).read_text()

    path = generate_agent2_demo_audio(
        text=text,
        output_path=args.output,
        voice_id=args.voice_id,
        language=args.language,
    )
    print(path)


if __name__ == "__main__":
    main()
