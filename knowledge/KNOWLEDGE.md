# Knowledge base schema (Layer 3)

This directory is LeAgent's knowledge base — an [OKF](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)-shaped bundle of markdown pages with YAML frontmatter, maintained by the Knowledge Agent (`leagent/agents/knowledge_agent.py`) and readable/editable by humans and any agent (DESIGN.md §3.6).

## Layers

1. **Raw sources (immutable):** `runs/<id>/events.jsonl`, eval reports, train logs, the SQLite store. Agents read these, never edit them.
2. **The wiki (this directory):** pages the Knowledge Agent owns and updates.
3. **This file:** schema and conventions the agent follows.

## Layout

```
knowledge/
├── KNOWLEDGE.md        # this schema file — never auto-edited
├── tasks/<slug>.md     # one page per task suite (e.g. libero_spatial)
├── policies/<slug>.md  # one page per policy (e.g. smolvla, pi05)
└── experiments/<slug>.md  # lesson pages for flywheel experiments (M1, planned)
```

## Page format

```markdown
---
name: smolvla
type: policy            # task | policy | experiment
status: observed-once   # observed-once | replicated | human-confirmed
updated: "2026-07-04"
provenance:             # every claim traces to runs/cycles (Layer 1)
  - run: 20260704-000524-m0-libero-smolvla
    cycle: 0
---

# Policy: smolvla

## Observations
- 2026-07-04 · run … · cycle 0 · libero_spatial · policy smolvla · success 50.0% · decision promote

## Lessons
- (agent- or human-maintained bullet list of durable, actionable lessons)
```

## Rules

- Pages are **advisory context only** — promote/iterate/escalate/rollback stays a pure function of eval deltas (DESIGN.md §2, §4).
- `status` upgrades automatically `observed-once → replicated` at ≥2 provenance entries; `human-confirmed` is set only by a human and is never downgraded by the agent.
- Humans may edit any page (OKF producer/consumer independence); run `KnowledgeAgent.lint()` afterwards to validate schema and provenance.
- The Lessons section is rewritten by the configured LLM (see `llm:` in the loop config); with no LLM configured it is append-only via humans.
