# Agent Prompts

These Markdown files define static agent behavior. They are not runtime memory.

Runtime state must be passed separately as structured handoff JSON and validated before each model call.

Protected sections marked `PROTECTED` are compliance and role-boundary instructions. The self-learning loop may propose changes to tunable sections only; it must not edit protected sections.

## Runtime Injection Contract

Do not edit prompt files to include borrower-specific data. Agent activities should load these prompts and inject a separate runtime context block containing:

- `company_name`
- `borrower_case`
- `settlement_policy`
- `compliance_rules`
- `handoff_summary` when applicable
- `conversation_so_far` or `transcript_id`

This keeps stable agent behavior in Markdown and mutable case memory in structured JSON.

Expected prompt files:

- `agent1_assessment.md` - chat assessment agent
- `agent2_resolution_voice.md` - voice resolution agent
- `agent3_final_notice.md` - chat final notice agent

The token-budget layer must enforce:

- Agent 1: system prompt <= 2000 tokens
- Agent 2: system prompt + handoff <= 2000 tokens, handoff <= 500 tokens
- Agent 3: system prompt + handoff <= 2000 tokens, handoff <= 500 tokens
