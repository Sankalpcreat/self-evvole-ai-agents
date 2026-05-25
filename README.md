# Riverline Collections Agents

Self-learning AI debt collections system with three agents orchestrated by [Temporal](https://temporal.io), powered by [Grok](https://x.ai) (xAI).

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Temporal Workflow                    │
│                (1 per borrower)                       │
│                                                       │
│  ┌──────────┐    ┌──────────────┐    ┌────────────┐ │
│  │ Agent 1   │───▶│   Agent 2     │───▶│  Agent 3   │ │
│  │Assessment │    │  Resolution   │    │Final Notice│ │
│  │  (Chat)   │    │   (Voice)     │    │  (Chat)    │ │
│  └──────────┘    └──────────────┘    └────────────┘ │
│       │ 500 tok       │ 500 tok                      │
│       │ handoff        │ handoff                      │
│       ▼                ▼                              │
│  [Summarize]      [Summarize]                        │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
           ┌────────────────────────┐
           │   Self-Learning Loop    │
           │                        │
           │  Simulate → Score →    │
           │  Mutate → Test →       │
           │  Adopt/Reject          │
           │                        │
           │  Meta-Evaluator        │
           │  (Darwin Gödel Machine)│
           └────────────────────────┘
```

### The Three Agents

| Agent | Mode | Role | Tone |
|-------|------|------|------|
| Agent 1: Assessment | Chat | Verify identity, gather financials, determine resolution path | Cold, clinical |
| Agent 2: Resolution | Voice | Present settlement options, handle objections, push for commitment | Transactional |
| Agent 3: Final Notice | Chat | State consequences, make final offer with hard deadline | Consequence-driven |

### Cross-Modal Handoffs

Each handoff is compacted into a deterministic `HANDOFF_STATE v1` capsule of ≤ 500 tokens. This is intentionally closer to context compaction than generic summarisation: the next agent receives the smallest high-signal state needed to continue, while the raw transcript remains available in evaluation/audit artifacts.

The handoff capsule prioritises:
1. Compliance flags (hardship, stop-contact, distress)
2. Identity verification status
3. Financial situation and offers made
4. Borrower's emotional state and objections

Token counting uses `tiktoken` with the `cl100k_base` encoding as a conservative proxy for Grok's tokeniser.

### Context Budget

| Agent | System Prompt | Handoff | Total Budget |
|-------|--------------|---------|-------------|
| Agent 1 | ≤ 2000 tok | 0 tok | 2000 tok |
| Agent 2 | ≤ 1500 tok | ≤ 500 tok | 2000 tok |
| Agent 3 | ≤ 1500 tok | ≤ 500 tok | 2000 tok |

Budgets are enforced in `token_counter.py` and checked before borrower-facing LLM calls. Violations raise `ValueError` and are logged. Offline evaluator/judge calls may inspect full transcripts and are accounted separately from borrower-facing context windows.

## Quickstart

### Prerequisites

- Docker & Docker Compose
- xAI API key ([console.x.ai](https://console.x.ai))

### Run the full system

```bash
# 1. Set your API key
echo "XAI_API_KEY=your_key_here" > .env

# 2. Start everything
docker compose up --build -d

# 3. Wait for Temporal to be ready (~30 seconds)
sleep 30

# 4. Trigger a borrower workflow
curl -X POST http://localhost:8000/trigger \
  -H "Content-Type: application/json" \
  -d '{"borrower_id": "borrower-001", "debt_amount_cents": 500000}'

# 5. Check status
curl http://localhost:8000/status/borrower-001

# 6. View Temporal UI
open http://localhost:8080
```

### Run locally (without Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Start Temporal infrastructure
docker compose up postgres temporal temporal-ui -d

# In terminal 1 — worker
PYTHONPATH=src python -m collections_agent.worker

# In terminal 2 — API server
PYTHONPATH=src python -m collections_agent.api

# In terminal 3 — trigger a workflow
PYTHONPATH=src python -m collections_agent.starter
```

### Run the evaluation pipeline

```bash
# Full pipeline: baseline → self-learning → meta-evaluation
PYTHONPATH=src python -m collections_agent.run_pipeline \
  --seed 42 \
  --iterations 3 \
  --conversations 5 \
  --budget 20.0 \
  --generate-voice-demo

# Evidence baseline used in the report: 5 personas × 3 repeats
PYTHONPATH=src python -m collections_agent.run_pipeline \
  --seed 42 \
  --baseline-repeats 3 \
  --skip-learning \
  --budget 6 \
  --generate-voice-demo

# Skip specific phases
PYTHONPATH=src python -m collections_agent.run_pipeline --skip-learning
PYTHONPATH=src python -m collections_agent.run_pipeline --skip-baseline --skip-meta

# Generate only the Agent 2 voice demo artifact
PYTHONPATH=src python -m collections_agent.voice_generator
```

### Try the live Agent 2 web voice loop

Start the API server, then open:

```bash
open http://localhost:8000/voice/agent2-web
```

The browser records borrower audio, the backend transcribes it with xAI STT,
runs Agent 2 with Grok and the settlement policy, generates xAI TTS audio, and
saves the transcript under `data/voice_sessions/`.

## Self-Learning Loop

### How it works

1. **Evaluate current prompt** — run N simulated conversations across 5 borrower personas, score each with LLM-as-judge + rule-based checks.
2. **Propose mutation** — Grok generates a small, targeted prompt modification aimed at the weakest metric.
3. **Validate** — check token budget, verify `<!-- PROTECTED -->` sections are intact.
4. **Evaluate candidate** — run N conversations with the new prompt.
5. **Statistical test** — Welch's t-test on composite scores (p < 0.05).
6. **Decision** — adopt only if: statistically significant, improvement ≥ 2%, zero compliance violations.
7. **Audit** — every version saved with full evaluation data in `data/prompt_versions/`.

### Metrics

| Metric | Weight | Source |
|--------|--------|--------|
| Resolution score | 30% | LLM-as-judge |
| Compliance score | 25% | Rule-based (8 rules) |
| Continuity score | 15% | LLM-as-judge |
| Tone score | 15% | LLM-as-judge |
| Information gathering | 10% | LLM-as-judge |
| Budget adherence | 5% | Rule-based |

Compliance violations apply a 50% penalty multiplier to the composite score.

### Borrower Personas

| Persona | Behaviour |
|---------|-----------|
| Cooperative | Willing to resolve, verifies identity, open to offers |
| Combative | Disputes debt, hostile, threatens legal action |
| Evasive | Vague answers, stalls, never commits |
| Confused | Overwhelmed, asks for clarification, mixes up details |
| Distressed | Financial hardship, emotional distress, mentions crisis |

## Darwin Gödel Machine (Meta-Evaluation)

This is a bounded, assignment-specific DGM-inspired loop, not the full open-ended Darwin Gödel Machine paper implementation. The practical mapping is:

- Prompt candidates are proposed and evaluated empirically.
- Every prompt version is archived with evaluation data.
- Compliance is a hard gate before adoption.
- Rollback is supported through prompt version pointers.
- The meta-evaluator checks whether the evaluator itself is misleading.

The meta-evaluator runs after the self-learning loop and checks for anti-patterns:

1. **Aggression rewarded** — high resolution score despite compliance violations
2. **Evaluation too lenient** — uniformly high scores with low variance
3. **Persona insensitivity** — combative persona scores as high as cooperative
4. **Weak compliance gate** — violations don't sufficiently reduce composite

Plus an LLM-based meta-judge for anything the rules miss.

The latest report run detected three evaluator flaws: compliance blind spot, inconsistent persona scoring, and a composite formula issue. See `docs/evolution_report.md`.

### Objective-Hacking Risk

The DGM paper warns that a self-improving system can optimize the benchmark instead of the real objective. In this project, a bad prompt could repeat handoff facts just to raise continuity, over-route borrowers to hardship to avoid violations, or become so conservative that collections performance drops. Mitigations are protected prompt sections, compliance hard gates, full-system scoring, raw audit trails, and meta-evaluation. This remains a real limitation, not something fully solved.

## Compliance Rules

All 8 rules are enforced by `compliance_checker.py` on every agent turn:

1. ✅ Identity disclosure (AI, not human)
2. ✅ No false threats (arrest, garnishment, etc.)
3. ✅ No harassment (honour stop-contact requests)
4. ✅ No misleading terms (offers within policy ranges)
5. ✅ Sensitive situations (hardship referral when distressed)
6. ✅ Recording disclosure
7. ✅ Professional composure
8. ✅ Data privacy (no full account numbers, no SSNs)

Protected prompt sections (`<!-- PROTECTED -->`) cannot be modified by the self-learning loop.

## Project Structure

```
├── docker-compose.yml          # Full system (Temporal + API + Worker)
├── Dockerfile                  # App container
├── pyproject.toml              # Dependencies
├── .env.example                # Environment template
├── prompts/                    # Agent system prompts (Markdown)
│   ├── agent1_assessment.md
│   ├── agent2_resolution_voice.md
│   └── agent3_final_notice.md
├── policy/                     # Business rules (JSON)
│   ├── settlement_policy.json
│   ├── compliance_rules.json
│   └── context_budget.json
├── src/collections_agent/
│   ├── activities.py           # Temporal activities (real LLM calls)
│   ├── api.py                  # FastAPI endpoints
│   ├── borrower_harness.py     # Simulated borrower (5 personas)
│   ├── compliance_checker.py   # 8-rule compliance validator
│   ├── evaluator.py            # LLM-as-judge scoring + CSV/JSON output
│   ├── llm_adapter.py          # Grok/xAI client wrapper
│   ├── meta_evaluator.py       # Darwin Gödel Machine
│   ├── models.py               # Data models (dataclasses)
│   ├── prompt_manager.py       # Prompt versioning + rollback
│   ├── run_pipeline.py         # Single-command evaluation pipeline
│   ├── self_learning.py        # Self-learning loop
│   ├── starter.py              # CLI workflow trigger
│   ├── token_counter.py        # tiktoken budget enforcement
│   ├── worker.py               # Temporal worker entrypoint
│   └── workflows.py            # Temporal workflow definition
└── data/                       # Generated at runtime
    ├── evaluations/            # Per-conversation scores (JSON + CSV)
    ├── prompt_versions/        # Prompt audit trail
    ├── audio/                  # Agent 2 voice artifacts
    ├── self_learning/          # Self-learning audit logs
    └── meta_evaluation/        # Meta-eval findings
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/trigger` | Start a borrower collections workflow |
| `GET` | `/status/{borrower_id}` | Check workflow status and outcome |
| `GET` | `/health` | Health check + Temporal connection status |
| `GET` | `/versions/{agent_name}` | List prompt versions for an agent |
| `POST` | `/evaluate` | Run full-system baseline evaluation, optional learning, optional meta-eval |
| `POST` | `/voice/agent2-demo` | Generate an xAI TTS audio artifact for Agent 2 |
| `GET` | `/voice/agent2-demo/file` | Download the latest generated Agent 2 audio artifact |
| `GET` | `/voice/agent2-web` | Browser UI for a live Agent 2 voice session |
| `POST` | `/voice/sessions` | Start a live Agent 2 voice session |
| `POST` | `/voice/sessions/{session_id}/turn` | Upload borrower audio and receive Agent 2 TTS response |
| `GET` | `/cost` | Cumulative LLM API cost |

## Technology Stack

| Component | Technology |
|-----------|-----------|
| LLM | Grok 4.3 (xAI) via OpenAI SDK |
| Orchestration | Temporal |
| API | FastAPI |
| Token counting | tiktoken (cl100k_base) |
| Voice | Text-based simulation (bulk eval) |
| Containerisation | Docker Compose |
| Language | Python 3.11 |

## Cost Breakdown

| Component | Est. Cost per Conversation |
|-----------|---------------------------|
| Agent conversation (6 turns) | ~$0.005 |
| Handoff summarisation | ~$0.001 |
| Evaluation (LLM judge) | ~$0.002 |
| Self-learning mutation | ~$0.003 |
| **Per full workflow** | **~$0.015** |

The fresh 15-conversation evidence run cost `$1.3009` including simulations, judge calls, meta-evaluation, and TTS generation. The baseline scoring portion was `$0.1004`.

## Limitations

- **Voice agent** uses a web voice loop for live demos. Bulk evaluation remains text-based for cost and reproducibility.
- **Token counting** uses cl100k_base as a proxy for Grok's actual tokeniser, so counts are approximate (conservative).
- **Statistical power** — with 5-10 conversations per evaluation, small effects may not reach significance. We accept this as a feature (conservative adoption).
- **Single-model dependency** — the system assumes Grok 4.3 availability. Provider switching requires only changing `.env`.
- **Objective hacking** — prompts can learn to satisfy measured metrics rather than true borrower-safe collections behavior. The meta-evaluator searches for this, but production would still need human review and live complaint/stop-contact monitoring.
