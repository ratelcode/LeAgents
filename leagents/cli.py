"""leagents CLI — M0 surface: `leagents run`, `leagents status`."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from leagents.agents import (
    DataAgent,
    EvalAgent,
    ImproveAgent,
    KnowledgeAgent,
    TrainAgent,
    dry_runner,
    subprocess_runner,
)
from leagents.config import LoopConfig
from leagents.events import Event, EventBus
from leagents.llm import make_llm
from leagents.orchestrator import (
    Constitution,
    CuratedProposer,
    DeterministicProposer,
    LLMProposer,
    LoopController,
)
from leagents.store import JobStore


def _print_event(event: Event) -> None:
    cycle = f"cycle {event.cycle}" if event.cycle is not None else "-"
    print(f"[{cycle}] {event.stage}/{event.kind} {json.dumps(event.payload, default=str)[:160]}")


def _dry_runner_with_synthetic_eval(cmd, log_path):
    """Dry runner that also fakes lerobot-eval output (pc_success=50) so a
    --dry-run exercises the full loop, including the decision logic."""
    result = dry_runner(cmd, log_path)
    if cmd and cmd[0] == "lerobot-eval":
        out = next((a.split("=", 1)[1] for a in cmd if a.startswith("--output_dir=")), None)
        if out:
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / "eval_info.json").write_text(
                json.dumps({"overall": {"pc_success": 50.0}, "dry_run": True})
            )
    return result


def cmd_run(args: argparse.Namespace) -> int:
    # lerobot-train/-eval live next to this interpreter (same venv); make sure
    # subprocesses resolve them even when the venv is not activated.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    cfg = LoopConfig.from_yaml(args.config)
    run_id = args.run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{cfg.run_name}"
    workdir = cfg.workdir / run_id
    workdir.mkdir(parents=True, exist_ok=True)

    bus = EventBus(workdir / "events.jsonl")
    bus.subscribe(_print_event)
    store = JobStore(cfg.workdir / "leagents.db")
    constitution = Constitution.from_yaml(cfg.constitution)
    runner = _dry_runner_with_synthetic_eval if args.dry_run else subprocess_runner

    llm = make_llm(cfg.llm)
    fallback = DeterministicProposer(
        cfg.seed_dataset,
        initial_episodes=cfg.data.initial_episodes,
        growth=cfg.data.growth,
        max_episodes=cfg.data.max_episodes,
    )
    knowledge_agent = (
        KnowledgeAgent(cfg.knowledge.root, bus, llm) if cfg.knowledge.enabled else None
    )
    if cfg.curation.enabled:
        proposer = CuratedProposer(
            llm, cfg.knowledge.root, fallback, cfg.data,
            candidates=cfg.curation.candidates, bus=bus,
            max_tokens=cfg.curation.max_tokens,
        )
    else:
        proposer = LLMProposer(llm, cfg.knowledge.root, fallback)
    controller = LoopController(
        cfg=cfg,
        store=store,
        bus=bus,
        data_agent=DataAgent(bus, runner, task_filter=cfg.data.task_filter),
        train_agent=TrainAgent(cfg.train, constitution, bus, runner),
        eval_agent=EvalAgent(cfg.eval, constitution, bus, runner),
        proposer=proposer,
        knowledge_agent=knowledge_agent,
        improve_agent=(
            ImproveAgent(cfg.improve, cfg.eval, constitution, bus, runner)
            if cfg.improve.enabled else None
        ),
    )
    if args.dry_run:
        print(f"DRY RUN — commands are logged to {workdir}, nothing is executed.")
        print("note: eval scores are synthetic (pc_success=50) to exercise the decision logic.")
    summary = controller.run(run_id, workdir)
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    db = Path(args.workdir) / "leagents.db"
    if not db.exists():
        print(f"no runs found (missing {db})")
        return 1
    store = JobStore(db)
    for run in store.list_runs():
        blessed = store.blessed_checkpoint(run["id"])
        print(f"run {run['id']}  status={run['status']}  stop={run['stop_reason'] or '-'}")
        for cycle in store.cycles_for(run["id"]):
            print(f"  cycle {cycle['idx']}: stage={cycle['stage']} decision={cycle['decision']}")
        if blessed:
            print(f"  blessed: {blessed['policy_type']} @ {blessed['path']}")
    return 0


def cmd_dash(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from leagents.dashboard.server import create_app
    except ImportError:
        print("the dashboard needs extra dependencies: pip install 'leagents[dash]'")
        return 1
    app = create_app(Path(args.workdir), Path(args.knowledge))
    print(f"LeAgents dashboard → http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from leagents.doctor import run_doctor

    return run_doctor()


def cmd_setup(args: argparse.Namespace) -> int:
    from leagents.doctor import run_setup

    return run_setup(install_lerobot=args.install_lerobot)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="leagents")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the collect→train→eval→decide loop")
    run_p.add_argument("-c", "--config", required=True, help="path to a LoopConfig YAML")
    run_p.add_argument("--run-id", default=None)
    run_p.add_argument("--dry-run", action="store_true", help="log commands without executing")
    run_p.set_defaults(fn=cmd_run)

    status_p = sub.add_parser("status", help="show runs, cycles, and blessed checkpoints")
    status_p.add_argument("-w", "--workdir", default="runs")
    status_p.set_defaults(fn=cmd_status)

    doctor_p = sub.add_parser("doctor", help="diagnose the environment; print exact fixes")
    doctor_p.set_defaults(fn=cmd_doctor)

    setup_p = sub.add_parser("setup", help="apply safe environment fixes (libero init, .env)")
    setup_p.add_argument("--install-lerobot", action="store_true",
                         help="also install leagents[lerobot] with the CMake compat env")
    setup_p.set_defaults(fn=cmd_setup)

    dash_p = sub.add_parser("dash", help="serve the flow dashboard (needs 'leagents[dash]')")
    dash_p.add_argument("-w", "--workdir", default="runs")
    dash_p.add_argument("-k", "--knowledge", default="knowledge")
    dash_p.add_argument("--host", default="127.0.0.1")
    dash_p.add_argument("--port", type=int, default=8321)
    dash_p.set_defaults(fn=cmd_dash)

    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
