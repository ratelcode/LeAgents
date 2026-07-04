# LeAgent

Agentic orchestration for the [LeRobot](https://github.com/huggingface/lerobot) robotics pipeline — an orchestrator drives an automated **collect → train → eval → improve** loop over [LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/lerobot-dataset-v3), with a deterministic loop controller, a constitution safety gate, verification gates before promotion, and (M2) a dashboard for visualizing the flow.

<p align="center"><img src="docs/architecture.svg" alt="LeAgent architecture — a deterministic loop controller dispatches data/train/eval/knowledge agents over LeRobotDataset v3.0 on the Hugging Face Hub, with a constitution safety gate, an OKF knowledge wiki feeding the proposer, and a flow dashboard reading the event log" width="880"></p>

Architecture, research grounding (verified 2023–2026 papers), and roadmap: **[DESIGN.md](DESIGN.md)**.

## Status — v0.0.4

| Milestone | Scope | Status |
|---|---|---|
| **M0** | Sim-only loop on LIBERO: seed dataset → SmolVLA fine-tune → `lerobot-eval` gate → promote/iterate/escalate/rollback | ✅ pipeline verified end-to-end on a real GPU (PushT + LIBERO smoke configs); full-scale training run pending |
| **M1** | DexFlyWheel-style self-improvement, RoboGene-style task curation, policy escalation, OKF knowledge layer (Karpathy-wiki-style, DESIGN.md §3.6) + provider-agnostic LLM proposer | 🚧 knowledge layer + LLM adapter landed |
| **M2** | Flow dashboard (Rerun episode replay, WandB curves, OTel agent traces) | 🚧 flow view v1 landed: runs → cycles → decisions live, eval chart, event log, knowledge browser (`leagent dash`) |
| M3 | Real robot: teleop collection, HIL-SERL adapter (requires lerobot ≥ 0.6.0, see CVE note in DESIGN.md §6) | planned |

What works today: the full loop state machine with budgets, the constitution gate, SQLite job store, JSONL event log, subprocess wrappers for `lerobot-train` / `lerobot-eval`, the OKF knowledge layer (`knowledge/` pages with provenance, updated every cycle, linted), and a provider-agnostic LLM adapter (`llm: anthropic:*|openai:*[@base_url]`, or none at all — every flow has a deterministic fallback). All covered by tests that run without a GPU or lerobot installed.

## Quickstart

```bash
pip install -e ".[dev]"
pytest

# dry run — no GPU, no lerobot; synthetic eval scores exercise the decision logic
leagent run -c configs/m0_libero.yaml --dry-run

# real run (Linux; needs a GPU and the LIBERO extras)
pip install -e ".[lerobot]"
leagent run -c configs/m0_libero.yaml

# inspect runs, cycles, decisions, blessed checkpoints
leagent status

# flow dashboard — runs, cycle pipeline, decisions, eval chart, events, knowledge
pip install -e ".[dash]"
leagent dash            # → http://127.0.0.1:8321
```

## How the loop decides

Each cycle trains a candidate checkpoint and evaluates it on LIBERO. The decision is a pure function of success-rate deltas vs. the blessed baseline (`leagent/orchestrator/decision.py`):

- **promote** — candidate beats baseline by ≥ `promote_delta`; it becomes the new blessed checkpoint
- **iterate** — small improvement; collect more data on failing task variations
- **escalate** — plateaued for `plateau_cycles`; move up the policy ladder (SmolVLA → π0.5)
- **rollback** — regression; the blessed checkpoint stands

Control flow is deterministic Python persisted to SQLite — LLM agents (task proposal, curation; M1) only make proposals *inside* the gates, never control-flow decisions.

## Layout

```
leagent/
├── orchestrator/   # loop controller, decision logic, constitution gate, proposer
├── agents/         # data / train / eval / improve agents (LeRobot CLI wrappers)
├── contracts/      # typed records: DatasetRef, CheckpointRecord, EvalReport
├── events/         # JSONL event bus (dashboard reads this in M2)
└── store/          # SQLite job store (runs, cycles, checkpoints)
configs/            # m0_libero.yaml, constitution.yaml
tests/              # loop e2e with fake runners — no GPU needed
```

## License

Apache-2.0
