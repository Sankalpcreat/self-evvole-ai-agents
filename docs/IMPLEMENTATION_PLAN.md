# Riverline Assignment Implementation Plan

This plan tracks the remaining work needed to turn the current Temporal skeleton into a valid Riverline submission.

## Current State

- Temporal workflow skeleton exists and runs end to end.
- Agent activities are stubs, not real LLM/voice agents.
- Docker Compose runs Temporal infrastructure, but not yet the API, worker, or evaluator as services.
- Only the 500-token handoff budget is partially enforced; full 2000-token agent context enforcement is not done.

## Build Order

1. Add agent prompts in `prompts/*.md`.
2. Add policy files in `policy/*.json`.
3. Add real token counter and enforce both 2000-token agent context and 500-token handoff limits.
4. Add LLM adapter with provider retries disabled so Temporal controls retries.
5. Add structured handoff JSON schemas and persistence.
6. Add compliance checker.
7. Add simulated borrower harness.
8. Add evaluator with raw JSON/CSV per-conversation scores.
9. Add prompt versioning and rollback.
10. Add self-learning loop.
11. Add meta-evaluation example.
12. Add FastAPI endpoints for live triggering and status.
13. Add mock voice plus TTS audio recording for Agent 2.
14. Finish README, evolution report, technical writeup, demo script, and cost report.

## Key Architecture Decisions

- Temporal owns durable borrower lifecycle orchestration.
- Activities own all side effects: LLM calls, voice calls, persistence, summarization, compliance checks, and reporting.
- Prompts are Markdown for human-readable static behavior.
- Runtime state, handoffs, policies, evaluations, and costs are structured JSON/CSV.
- Full transcripts are stored outside Temporal workflow history; workflows pass compact IDs and summaries.
- Compliance rules are protected prompt sections that self-learning cannot modify.
- Compliance violations are hard gates: a prompt with any violation cannot be adopted.

## Critical Edge Cases

- Borrower asks to stop contact: acknowledge, flag account, terminate workflow.
- Borrower mentions hardship, medical emergency, or emotional distress: offer hardship program and stop pressure.
- Agent offers out-of-policy terms: reject output and record compliance failure.
- Agent reveals full PII: reject output and record privacy failure.
- Assessment fails after 3 attempts: Agent 2 receives a no-assessment handoff and must re-verify.
- Voice call not answered or dropped: capture status and avoid duplicate calls through idempotent call session IDs.
- Summary truncation drops compliance facts: summarizer must prioritize compliance flags before normal business facts.
- Prompt update exceeds token budget: reject update.
- Prompt update improves one agent but worsens system continuity: reject based on system-level evaluation.
- Meta-evaluation changes must be validated before adoption.

## Definition of Done

- `docker compose up` starts Temporal, database, API, worker, and evaluator-ready services.
- A borrower workflow can be triggered through an API or CLI.
- Agent 1, Agent 2, and Agent 3 use real prompts and an LLM adapter.
- Agent 2 has a mock or real voice path plus an audio recording artifact.
- Handoffs are compact, structured, token-counted, and preserve required facts.
- Evaluation can be rerun with one command and fixed seed/config.
- Reports include raw per-conversation scores, prompt versions, statistical analysis, rollback evidence, meta-evaluation evidence, and cost breakdown.
