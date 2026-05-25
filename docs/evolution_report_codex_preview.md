# Evolution Report

## Summary

This project implements a bounded, compliance-preserving, Darwin Gödel Machine-inspired loop. It does not claim to reproduce the full open-ended DGM paper. Instead, prompt candidates are proposed, evaluated quantitatively, archived with audit data, adopted or rejected, and then a meta-evaluator checks whether the evaluator itself is misleading.

Fresh evidence run on 2026-05-25:

- 15 full Agent 1 -> Agent 2 -> Agent 3 conversations
- 5 borrower personas, 3 repeats each
- Seed: `42`
- Baseline scoring cost: `$0.1004`
- Full run cost including simulations, meta-eval, and TTS: `$1.3009`
- Budget: under the `$20` assignment cap
- Latest Agent 2 voice artifact: `data/audio/agent2_demo_20260525T040652Z.mp3`

| Agent | Candidate Result | Adopted? | Decision Reason |
|-------|------------------|----------|-----------------|
| Agent 1 (Assessment) | 0.720 -> 0.640 | Rejected | Candidate reduced continuity and composite score |
| Agent 2 (Resolution) | 0.655 -> 0.9425 | Adopted | Candidate improved settlement structure and continuity |
| Agent 3 (Final Notice) | 0.680 -> 0.7225 | Adopted | Candidate improved continuity despite slight resolution drop |

## Baseline Evaluation

Each row is a full system-level pipeline: assessment chat, resolution voice-style turn, final notice chat, with real handoff capsules between stages.

![Baseline composite by persona](/Users/sankalpsingh/Desktop/backend/docs/report_assets/riverline_baseline_composite.png)

### Per-Persona Aggregates

| Persona | N | Resolution Mean | Compliance Mean | Continuity Mean | Tone Mean | Info Mean | Composite Mean | Composite Std |
|---------|---|-----------------|-----------------|-----------------|-----------|-----------|----------------|---------------|
| Cooperative | 3 | 0.5167 | 1.0000 | 0.4667 | 0.5500 | 0.7500 | 0.6825 | 0.0563 |
| Combative | 3 | 0.1333 | 1.0000 | 0.8500 | 0.8667 | 0.4833 | 0.6458 | 0.0255 |
| Evasive | 3 | 0.2000 | 1.0000 | 0.4833 | 0.7500 | 0.2000 | 0.5650 | 0.0557 |
| Confused | 3 | 0.2333 | 1.0000 | 0.7333 | 0.7167 | 0.2500 | 0.6125 | 0.0455 |
| Distressed | 3 | 0.5833 | 1.0000 | 0.5167 | 0.7833 | 0.5000 | 0.7200 | 0.1570 |

### Overall Aggregate

| Metric | Mean | Std | Min | Max | N |
|--------|------|-----|-----|-----|---|
| Resolution | 0.3333 | 0.2127 | 0.10 | 0.75 | 15 |
| Compliance | 1.0000 | 0.0000 | 1.00 | 1.00 | 15 |
| Continuity | 0.6100 | 0.2523 | 0.25 | 0.95 | 15 |
| Tone | 0.7333 | 0.1460 | 0.40 | 0.90 | 15 |
| Info Gathering | 0.4367 | 0.2588 | 0.10 | 0.80 | 15 |
| Composite | 0.6452 | 0.0889 | 0.505 | 0.8675 | 15 |

![Baseline metric matrix](/Users/sankalpsingh/Desktop/backend/docs/report_assets/riverline_metric_matrix.png)

### Observations

1. Resolution remains the weakest metric at `0.3333` mean. This is consistent with a conservative collections policy and difficult borrower personas.
2. Compliance scores are uniformly `1.0`, which looks good at first but is actually suspicious because live logs show compliance repairs being triggered. The meta-evaluator correctly flags this as a scoring blind spot.
3. Distressed borrowers have the highest mean composite but also the highest variance. These cases depend heavily on whether the agent recognizes hardship early.
4. Combative borrowers get high continuity/tone despite low resolution. That is useful evidence that the evaluator can over-reward composure/continuity when the actual collections outcome is poor.

## Handoff And Context Budget Evidence

The handoff layer now uses deterministic `HANDOFF_STATE v1` capsules rather than opaque summarization. The raw transcript remains available for evaluation/audit, but borrower-facing agents receive only the bounded state needed to continue coherently.

During the 2026-05-25 run:

- Agent 1 -> Agent 2 handoffs ranged from `131` to `234` tokens.
- Agent 2 -> Agent 3 handoffs ranged from `137` to `243` tokens.
- All handoffs stayed below the assignment cap of `500` tokens.
- Borrower-facing LLM calls stayed below the `2000` input-token budget.
- Offline judge calls exceeded 2000 tokens in some cases because they score complete transcripts; this is not the borrower-facing agent context window.

The implementation enforces this in code through:

- `src/collections_agent/token_counter.py`
- `src/collections_agent/activities.py`
- `policy/context_budget.json`

## Self-Learning Decisions

### Agent 1: Assessment

The proposed mutation made the opening more direct and used a more structured question flow.

| Metric | Old | Candidate | Delta |
|--------|-----|-----------|-------|
| Resolution | 0.35 | 0.30 | -0.05 |
| Compliance | 1.00 | 1.00 | 0.00 |
| Continuity | 0.65 | 0.40 | -0.25 |
| Tone | 0.85 | 0.70 | -0.15 |
| Info Gathering | 0.90 | 0.85 | -0.05 |
| Composite | 0.720 | 0.640 | -0.080 |

Decision: rejected. The mutation improved neither performance nor continuity enough to justify adoption.

### Agent 2: Resolution

The proposed mutation improved settlement presentation: lead with concrete options, anchor within policy bounds, and make deadlines explicit.

| Metric | Old | Candidate | Delta |
|--------|-----|-----------|-------|
| Resolution | 0.25 | 0.90 | +0.65 |
| Compliance | 1.00 | 1.00 | 0.00 |
| Continuity | 0.60 | 0.95 | +0.35 |
| Tone | 0.70 | 0.90 | +0.20 |
| Info Gathering | 0.85 | 0.95 | +0.10 |
| Composite | 0.655 | 0.9425 | +0.2875 |

Decision: adopted. Compliance was preserved and all primary metrics improved.

### Agent 3: Final Notice

The proposed mutation made Agent 3 reference prior stages more explicitly and present final consequences in a clearer structure.

| Metric | Old | Candidate | Delta |
|--------|-----|-----------|-------|
| Resolution | 0.35 | 0.30 | -0.05 |
| Compliance | 1.00 | 1.00 | 0.00 |
| Continuity | 0.70 | 0.90 | +0.20 |
| Tone | 0.60 | 0.75 | +0.15 |
| Info Gathering | 0.80 | 0.85 | +0.05 |
| Composite | 0.680 | 0.7225 | +0.0425 |

Decision: adopted. This is a deliberate trade-off: slightly less immediate resolution, better continuity for the “one continuous borrower experience” requirement.

![Prompt evolution](/Users/sankalpsingh/Desktop/backend/docs/report_assets/riverline_prompt_evolution.png)

## Meta-Evaluation

The latest meta-evaluation artifact is `data/meta_evaluation/meta_eval_20260525T040652Z.json`.

![Meta-evaluation findings](/Users/sankalpsingh/Desktop/backend/docs/report_assets/riverline_meta_eval_findings.png)

The meta-evaluator found 3 flaws. The latest artifact demonstrates the flaws; the code now applies the corrections in the evaluator and system-level pipeline scorer for future reruns.

| Severity | Flaw | What It Means | Correction Implemented |
|----------|------|---------------|------------------------|
| High | Compliance blind spot | All final scores show compliance `1.0`, even though live compliance repair logs show missing disclosures/data-privacy issues during generation. | `ConversationEvaluator._synthetic_compliance_check()` re-runs compliance checks over saved transcripts, and `score_full_pipeline()` now feeds those violations into system-level scores. |
| Medium | Inconsistent persona scoring | Combative/evasive cases can receive high tone and continuity despite very low resolution. | The judge prompts now include persona-adjusted expectations and anti-escalation scoring anchors. |
| High | Composite formula issue | Compliance penalty is never exercised when the violations array is empty. | The composite formula uses graduated compliance penalties, and synthetic transcript violations populate the violation list before scoring. |

This satisfies the assignment’s DGM requirement in a bounded way: the system does not only improve prompts; it also identifies weaknesses in the measurement system.

## Objective-Hacking Limitation

The DGM paper warns that systems can optimize the benchmark rather than the real objective. That risk exists here.

Concrete examples:

- A prompt could repeat handoff facts mechanically to raise continuity while sounding unnatural.
- A prompt could route every distressed borrower to hardship to avoid violations, even when a normal payment plan is appropriate.
- A prompt could become overly conservative, preserving compliance but reducing actual collections performance.
- A prompt could learn to satisfy the LLM judge’s phrasing preferences instead of helping the borrower reach a legitimate resolution.

Mitigations in this system:

- Compliance is a hard gate before prompt adoption.
- Protected prompt sections cannot be rewritten by self-learning.
- Evaluation is system-level, not just per-agent.
- Raw scores, prompt versions, meta-eval findings, and costs are archived.
- The meta-evaluator explicitly searches for compliance blind spots and misleading composite scores.

Residual risk remains. In production, I would add adversarial borrower scenarios, human review before deployment, complaint/stop-contact monitoring, and periodic re-scoring with a second judge model.

## Cost Breakdown

![Cost breakdown](/Users/sankalpsingh/Desktop/backend/docs/report_assets/riverline_cost_breakdown.png)

| Phase | Cost |
|-------|------|
| 15-conversation baseline scoring | $0.1004 |
| Latest meta-evaluation | $0.0040 |
| Latest full run including simulations and TTS | $1.3009 |
| Assignment budget | $20.0000 |
| Remaining budget after latest full run | $18.6991 |

The latest run used:

- Input tokens: `513,948`
- Output tokens: `263,398`
- Total cost: `$1.3009`

## Reproducibility

Fresh evidence command:

```bash
set -a; source .env; set +a
source .venv/bin/activate
PYTHONPATH=src python -u -m collections_agent.run_pipeline \
  --seed 42 \
  --baseline-repeats 3 \
  --skip-learning \
  --budget 6 \
  --generate-voice-demo
```

Full learning command:

```bash
set -a; source .env; set +a
source .venv/bin/activate
PYTHONPATH=src python -m collections_agent.run_pipeline \
  --seed 42 \
  --iterations 1 \
  --conversations 1 \
  --budget 20.0 \
  --generate-voice-demo
```

Raw artifacts:

- `data/evaluations/baseline/scores.csv`
- `data/evaluations/baseline/scores.json`
- `data/prompt_versions/agent1/20260524T054650Z.json`
- `data/prompt_versions/agent2/20260524T055054Z.json`
- `data/prompt_versions/agent3/20260524T055602Z.json`
- `data/meta_evaluation/meta_eval_20260525T040652Z.json`
- `data/audio/agent2_demo_20260525T040652Z.mp3`

## Limitations

1. Self-learning adoption still used small samples in the earlier run. The fresh baseline has `n=3` per persona, but the prompt-evolution comparisons should ideally also be rerun with `n>=3`.
2. The compliance checker catches many issues during generation. The meta-evaluator caught that the prior final scoring path over-reported compliance as perfect; the code now re-checks saved transcripts, but the full 15-run baseline has not been rerun after this final scorer patch.
3. Voice evaluation is mostly text-based for reproducibility and cost. The system includes live browser voice and a TTS artifact, but not telephony-grade latency/barge-in testing.
4. Token counting uses `tiktoken` as a conservative proxy for Grok’s tokenizer.
5. This is a bounded DGM-inspired loop, not open-ended self-code rewriting or archive search. That is intentional because debt collection is compliance-sensitive and the assignment has a `$20` learning-loop budget.
