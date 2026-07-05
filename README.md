# LeAgents

Agentic orchestration for the [LeRobot](https://github.com/huggingface/lerobot) robotics pipeline — an orchestrator drives an automated **collect → train → eval → improve** loop over [LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/lerobot-dataset-v3), with a deterministic loop controller, a constitution safety gate, verification gates before promotion, and (M2) a dashboard for visualizing the flow.

<p align="center"><img src="https://raw.githubusercontent.com/ratelcode/LeAgents/main/docs/architecture.svg" alt="LeAgents architecture — a deterministic loop controller dispatches data/train/eval/knowledge agents over LeRobotDataset v3.0 on the Hugging Face Hub, with a constitution safety gate, an OKF knowledge wiki feeding the proposer, and a flow dashboard reading the event log" width="880"></p>

Architecture, research grounding (verified 2023–2026 papers), and roadmap: **[DESIGN.md](DESIGN.md)**.

## Status — v0.0.5

| Milestone | Scope | Status |
|---|---|---|
| **M0** | Sim-only loop on LIBERO: seed dataset → SmolVLA fine-tune → `lerobot-eval` gate → promote/iterate/escalate/rollback | ✅ done — a successful autonomous run climbed **63% → 69% → 72%** on LIBERO spatial (see below); PushT/LIBERO smoke configs included |
| **M1** | DexFlyWheel-style self-improvement, RoboGene-style task curation, policy escalation, OKF knowledge layer (Karpathy-wiki-style, DESIGN.md §3.6) + provider-agnostic LLM proposer | 🚧 knowledge layer + LLM adapter landed; success-filtered rollout harvest → accumulating mix validated end-to-end in a real run |
| **M2** | Flow dashboard (Rerun episode replay, WandB curves, OTel agent traces) | 🚧 flow view v1 landed: runs → cycles → decisions live, eval chart, event log, knowledge browser (`leagents dash`) |
| M3 | Real robot: teleop collection, HIL-SERL adapter (requires lerobot ≥ 0.6.0, see CVE note in DESIGN.md §6) | planned |

What works today: the full loop state machine with budgets, the constitution gate, SQLite job store, JSONL event log, subprocess wrappers for `lerobot-train` / `lerobot-eval`, the OKF knowledge layer (`knowledge/` pages with provenance, updated every cycle, linted), the DexFlyWheel data path (success-filtered rollout harvesting → accumulated mix → adaptation training), and a provider-agnostic LLM adapter (`llm: gemini:*|anthropic:*|openai:*[@base_url]`, or none at all — every flow has a deterministic fallback). All covered by tests that run without a GPU or lerobot installed.

## A successful autonomous run — the data-growth flywheel (2026-07-05)

Once the data bug below was fixed, one M0 run on a single RTX 5070 Ti (16 GB) climbed monotonically on LIBERO spatial (100 eval episodes/cycle), fully autonomous, 5.9 GPU-hours, one `run` command:

| Cycle | Data | LIBERO spatial success | Decision |
|---|---|---|---|
| 0 | 200 episodes (20/task) | **63%** | promote (baseline) |
| 1 | 320 | **69%** | promote — the hardest task (bowl-on-cookie-box) rose most, 20% → 50% |
| 2 | 432 (~43/task — every spatial episode the dataset has) | **72%** | iterate — +3% is real but below `promote_delta`; the 69% checkpoint stays blessed |

More data lifted the number, with the expected diminishing return (+6, then +3). Cycle 2 drained the well: the seed dataset holds 432 libero_spatial episodes in total (the config asked for 500; the task-filtered selector capped at what exists), so from here the loop's remaining levers are self-generated data, policy escalation, and RL — which is exactly what M1 is for. Each promotion also **harvested success-filtered rollouts** from the freshly-blessed policy and merged them into a growing "self-play" mix (20 → 40 episodes) — the DexFlyWheel data path, running end-to-end without a human in the loop. (Adaptation *training* on that mix is held back after an over-adaptation regression — see the note in `configs/m0_libero_scale.yaml` — so this run measures the pure data-growth signal.)

The decision function did its job in both directions: it promoted the genuine gains and, in an earlier run where an aggressive adaptation stage overfit and dropped success to 20%, it **rolled back and protected the 63% baseline** rather than accepting the regression.

## The failure that came first (2026-07-04)

Before any of that worked, three whole runs sat at a flat **0%**, and the metrics could not tell us why. One M0 run on the same GPU — 3 cycles, 6.1 GPU-hours, zero human intervention, every decision/event/knowledge-page update logged:

| Cycle | Data | Train | LIBERO spatial eval (100 episodes) | Decision |
|---|---|---|---|---|
| 0 | 40 episodes | 20k steps from `smolvla_base` (loss → 0.030) | 0% success — arm reaches targets, never completes | **promote** (baseline) |
| 1 | 80 | 20k steps, continued from the blessed checkpoint | 0% | **iterate** |
| 2 | 160 | 20k steps | 0% | **iterate** — the `escalate_floor` guard correctly refused to escalate a 0%-plateau to a bigger policy |

This run validated the *loop* — budgets held, weights carried over, and the decision function behaved exactly as specified.

**Root-cause correction (same day):** the flat 0% was first attributed to the data budget (4→16 episodes per task vs. the verified ~50). Digging into a follow-up run that stayed at 0% with 20 per task exposed the real cause: **HuggingFaceVLA/libero is suite-ordered, and the `[0..N)` episode prefix belongs to *other* suites** (libero_spatial episodes live around indices 1261–1538) — every run had trained on tasks disjoint from the eval. In the metrics this silent failure was indistinguishable from under-training. Fix: episode selection is now task-filtered against the eval suite (`leagents.scripts.select_episodes`, `data.task_filter`), balanced per task, and the Data Agent fails loudly if the selection doesn't cover the suite. The table above stands as a record of the failure mode.

## Quickstart

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ratelcode/LeAgents/blob/main/notebooks/quickstart_colab.ipynb) — one autonomous cycle (PushT) on a free GPU, ~5 minutes, nothing to install locally.

```bash
pip install -e ".[dev]"
pytest

# instant demo — installs a bundled real 3-cycle run (events, decisions,
# eval videos, knowledge pages) so the dashboard is never empty. No GPU.
leagents demo

# dry run — no GPU, no lerobot; synthetic eval scores exercise the decision logic
leagents run -c configs/m0_libero.yaml --dry-run

# real run (Linux; needs a GPU and the LIBERO extras)
pip install -e ".[lerobot]"
leagents run -c configs/m0_libero.yaml

# inspect runs, cycles, decisions, blessed checkpoints
leagents status

# flow dashboard — runs, cycle pipeline, decisions, eval chart, events, knowledge
pip install -e ".[dash]"
leagents dash            # → http://127.0.0.1:8321
```

### No root, no Docker? Use pixi

On shared servers the only step that needs sudo is LIBERO's `egl-probe` build (system EGL headers). [pixi](https://pixi.sh) supplies Python, the EGL/OpenGL headers, a C++ toolchain, and CMake 3.x from conda-forge instead — zero root required:

```bash
pixi run test                  # loop tests, no GPU needed
pixi -e lerobot run doctor     # full environment checks (GPU/EGL/LIBERO)
pixi -e lerobot run smoke-pusht
```

### Docker

The image bakes in everything that takes debugging natively — EGL headers for LIBERO's `egl-probe` build, CMake < 4, libero's one-time init, headless sim env vars. GPU access needs [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html); secrets come from `.env` at runtime, never the image:

```bash
docker compose up dash    # flow dashboard → http://localhost:8321
docker compose up loop    # the autonomous cycle (GPU)
```

## How the loop decides

Each cycle trains a candidate checkpoint and evaluates it on LIBERO. The decision is a pure function of success-rate deltas vs. the blessed baseline (`leagents/orchestrator/decision.py`):

- **promote** — candidate beats baseline by ≥ `promote_delta`; it becomes the new blessed checkpoint
- **iterate** — small improvement; collect more data on failing task variations
- **escalate** — plateaued for `plateau_cycles`; move up the policy ladder (SmolVLA → π0.5)
- **rollback** — regression; the blessed checkpoint stands

Control flow is deterministic Python persisted to SQLite — LLM agents (task proposal, curation; M1) only make proposals *inside* the gates, never control-flow decisions.

## Layout

```
leagents/
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
