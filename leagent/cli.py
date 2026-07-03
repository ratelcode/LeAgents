"""leagent CLI — M0 surface: `leagent run`, `leagent status`."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from leagent.agents import (
    DataAgent,
    EvalAgent,
    KnowledgeAgent,
    TrainAgent,
    dry_runner,
    subprocess_runner,
)
from leagent.config import LoopConfig
from leagent.events import Event, EventBus
from leagent.llm import make_llm
from leagent.orchestrator import (
    Constitution,
    DeterministicProposer,
    LLMProposer,
    LoopController,
)
from leagent.store import JobStore


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
                json.dumps({"aggregated": {"pc_success": 50.0}, "dry_run": True})
            )
    return result


def cmd_run(args: argparse.Namespace) -> int:
    cfg = LoopConfig.from_yaml(args.config)
    run_id = args.run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{cfg.run_name}"
    workdir = cfg.workdir / run_id
    workdir.mkdir(parents=True, exist_ok=True)

    bus = EventBus(workdir / "events.jsonl")
    bus.subscribe(_print_event)
    store = JobStore(cfg.workdir / "leagent.db")
    constitution = Constitution.from_yaml(cfg.constitution)
    runner = _dry_runner_with_synthetic_eval if args.dry_run else subprocess_runner

    llm = make_llm(cfg.llm)
    fallback = DeterministicProposer(cfg.seed_dataset)
    knowledge_agent = (
        KnowledgeAgent(cfg.knowledge.root, bus, llm) if cfg.knowledge.enabled else None
    )
    controller = LoopController(
        cfg=cfg,
        store=store,
        bus=bus,
        data_agent=DataAgent(bus),
        train_agent=TrainAgent(cfg.train, constitution, bus, runner),
        eval_agent=EvalAgent(cfg.eval, constitution, bus, runner),
        proposer=LLMProposer(llm, cfg.knowledge.root, fallback),
        knowledge_agent=knowledge_agent,
    )
    if args.dry_run:
        print(f"DRY RUN — commands are logged to {workdir}, nothing is executed.")
        print("note: eval scores are synthetic (pc_success=50) to exercise the decision logic.")
    summary = controller.run(run_id, workdir)
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    db = Path(args.workdir) / "leagent.db"
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="leagent")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the collect→train→eval→decide loop")
    run_p.add_argument("-c", "--config", required=True, help="path to a LoopConfig YAML")
    run_p.add_argument("--run-id", default=None)
    run_p.add_argument("--dry-run", action="store_true", help="log commands without executing")
    run_p.set_defaults(fn=cmd_run)

    status_p = sub.add_parser("status", help="show runs, cycles, and blessed checkpoints")
    status_p.add_argument("-w", "--workdir", default="runs")
    status_p.set_defaults(fn=cmd_status)

    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
