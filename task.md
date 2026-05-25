# Riverline Grok Build Status

## Completed

- Grok/xAI adapter using the Responses API with local cost tracking.
- Explicit Agent 1 -> Agent 2 -> Agent 3 pipeline with handoff summaries.
- Compliance checker and bounded rewrite path for agent responses.
- Settlement policy injection for Agent 2.
- System-level baseline evaluation and learning-loop evaluation.
- Meta-evaluator scaffold for evaluator/checker flaws.
- FastAPI endpoints for workflow trigger/status, evaluation, cost, prompt versions, and Agent 2 voice artifact generation.
- xAI TTS helper for producing the Agent 2 demo audio recording.
- Live web voice loop for Agent 2: browser recording -> xAI STT -> Grok Agent 2 -> xAI TTS playback, with saved transcripts.
- Dockerfile and docker-compose services for Temporal, API, and worker.
- Local Grok/Riverline skill at `.agents/skills/grok-riverline/SKILL.md`.

## Remaining Before Submission

- Rotate any API key pasted in chat or logs.
- Run the pipeline with a fresh `XAI_API_KEY`:
  `PYTHONPATH=src python -m collections_agent.run_pipeline --seed 42 --generate-voice-demo`
- Try the live voice loop with the API running:
  `open http://localhost:8000/voice/agent2-web`
- Review `data/evaluations/`, `data/self_learning/`, and `data/audio/` artifacts.
- Record the 2-3 minute demo video.
- Write the handwritten decision journal.
- Deploy with `docker compose up --build -d` and verify `/health`, `/evaluate`, and `/voice/agent2-demo`.
