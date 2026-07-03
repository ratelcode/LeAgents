"""Eval Agent — `lerobot-eval` LIBERO gate (DESIGN.md §3.4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from leagent.agents.base import Runner, subprocess_runner
from leagent.config import EvalConfig
from leagent.contracts import CheckpointRecord, EvalReport
from leagent.events import Event, EventBus
from leagent.orchestrator.constitution import Constitution, ConstitutionError


class EvalError(Exception):
    pass


class EvalAgent:
    def __init__(
        self,
        cfg: EvalConfig,
        constitution: Constitution,
        bus: EventBus,
        runner: Runner = subprocess_runner,
    ):
        self.cfg = cfg
        self.constitution = constitution
        self.bus = bus
        self.runner = runner

    def build_command(self, checkpoint: CheckpointRecord, output_dir: Path) -> list[str]:
        return [
            "lerobot-eval",
            f"--policy.path={checkpoint.path}",
            f"--env.type={self.cfg.env_type}",
            f"--env.task={self.cfg.task_suite}",
            f"--eval.n_episodes={self.cfg.n_episodes}",
            f"--output_dir={output_dir}",
            *self.cfg.extra_args,
        ]

    def run(
        self, *, run_id: str, cycle: int, checkpoint: CheckpointRecord, workdir: Path
    ) -> EvalReport:
        output_dir = workdir / f"cycle_{cycle}" / "eval"
        cmd = self.build_command(checkpoint, output_dir)

        for verdict in (self.constitution.check_eval(self.cfg.env_type, self.cfg.n_episodes),
                        self.constitution.check_command(cmd)):
            if not verdict.allowed:
                self.bus.emit(Event(run_id, "eval", "constitution_denied", cycle,
                                    {"rule": verdict.rule, "reason": verdict.reason}))
                raise ConstitutionError(verdict)

        self.bus.emit(Event(run_id, "eval", "job_started", cycle, {"cmd": cmd}))
        result = self.runner(cmd, output_dir / "eval.log")
        self.bus.emit(Event(run_id, "eval", "job_finished", cycle,
                            {"exit_code": result.exit_code, "duration_s": result.duration_s,
                             "log": str(result.log_path)}))
        if result.exit_code != 0:
            raise EvalError(f"lerobot-eval exited {result.exit_code}, see {result.log_path}")

        success_rate, raw = self._parse_eval_info(output_dir / "eval_info.json")
        report = EvalReport(
            checkpoint=checkpoint.path,
            env_type=self.cfg.env_type,
            task_suite=self.cfg.task_suite,
            n_episodes=self.cfg.n_episodes,
            success_rate=success_rate,
            raw=raw,
        )
        self.bus.emit(Event(run_id, "eval", "report", cycle,
                            {"success_rate": success_rate, "task_suite": self.cfg.task_suite}))
        return report

    @staticmethod
    def _parse_eval_info(path: Path) -> tuple[float, dict[str, Any]]:
        """Parse lerobot-eval's eval_info.json; pc_success is a percentage."""
        if not path.exists():
            raise EvalError(f"eval output missing: {path}")
        data: dict[str, Any] = json.loads(path.read_text())
        aggregated = data.get("aggregated", data)
        for key in ("pc_success", "success_rate", "avg_success"):
            if key in aggregated:
                value = float(aggregated[key])
                return (value / 100.0 if value > 1.0 else value), data
        raise EvalError(f"no success metric found in {path}")
