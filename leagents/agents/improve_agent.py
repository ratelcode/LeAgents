"""Improvement Agent — DexFlyWheel step 3: success-filtered rollout collection
(M1, DESIGN.md §3.5).

Runs the blessed policy in sim via ``leagents.scripts.collect_rollouts`` (a
subprocess, like every other agent) and keeps only successful episodes as a
new LeRobotDataset. Residual-RL (flywheel step 2) and merging the rollout
dataset into the training mix are the remaining M1 work.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from leagents.agents.base import Runner, subprocess_runner
from leagents.config import EvalConfig, ImproveConfig
from leagents.contracts import CheckpointRecord, DatasetRef
from leagents.events import Event, EventBus
from leagents.orchestrator.constitution import Constitution, ConstitutionError


class ImproveError(Exception):
    pass


class ImproveAgent:
    def __init__(
        self,
        cfg: ImproveConfig,
        eval_cfg: EvalConfig,
        constitution: Constitution,
        bus: EventBus,
        runner: Runner = subprocess_runner,
    ):
        self.cfg = cfg
        self.eval_cfg = eval_cfg
        self.constitution = constitution
        self.bus = bus
        self.runner = runner

    def build_command(self, checkpoint: CheckpointRecord, out_dir: Path, repo_id: str) -> list[str]:
        cmd = [
            sys.executable, "-m", "leagents.scripts.collect_rollouts",
            f"--policy-path={checkpoint.path}",
            f"--env-type={self.eval_cfg.env_type}",
            f"--task={self.eval_cfg.task_suite}",
            f"--episodes={self.cfg.episodes}",
            f"--out={out_dir}",
            f"--repo-id={repo_id}",
            f"--device={self.cfg.device}",
            *self.cfg.extra_args,
        ]
        if self.cfg.task_text:
            cmd.append(f"--task-text={self.cfg.task_text}")
        return cmd

    def run(
        self, *, run_id: str, cycle: int, checkpoint: CheckpointRecord, workdir: Path,
        prev_mix: DatasetRef | None = None,
    ) -> tuple[DatasetRef, dict]:
        out_dir = workdir / f"cycle_{cycle}" / "rollouts"
        repo_id = f"local/{run_id}-cycle{cycle}-rollouts"
        cmd = self.build_command(checkpoint, out_dir, repo_id)

        for verdict in (self.constitution.check_eval(self.eval_cfg.env_type, self.cfg.episodes),
                        self.constitution.check_command(cmd)):
            if not verdict.allowed:
                self.bus.emit(Event(run_id, "improve", "constitution_denied", cycle,
                                    {"rule": verdict.rule, "reason": verdict.reason}))
                raise ConstitutionError(verdict)

        self.bus.emit(Event(run_id, "improve", "job_started", cycle, {"cmd": cmd}))
        result = self.runner(cmd, out_dir.parent / "rollouts.log")
        self.bus.emit(Event(run_id, "improve", "job_finished", cycle,
                            {"exit_code": result.exit_code, "duration_s": result.duration_s,
                             "log": str(result.log_path)}))
        if result.exit_code != 0:
            raise ImproveError(f"rollout collection exited {result.exit_code}, "
                               f"see {result.log_path}")

        summary = self._parse_summary(result.log_path)
        self.bus.emit(Event(run_id, "improve", "rollouts_collected", cycle, summary))
        ref = DatasetRef(
            repo_id=repo_id,
            root=str(out_dir),
            num_episodes=summary.get("kept"),
            notes=f"success-filtered rollouts from cycle {cycle} blessed checkpoint",
        )
        if summary.get("kept", 0) == 0:
            return (prev_mix if prev_mix else ref), summary
        if prev_mix is None:
            return ref, summary
        return self._merge(run_id, cycle, workdir, prev_mix, ref), summary

    def _merge(
        self, run_id: str, cycle: int, workdir: Path, a: DatasetRef, b: DatasetRef
    ) -> DatasetRef:
        """Accumulate rollouts across cycles: mix_{n} = mix_{n-1} ∪ rollouts_n
        (DexFlyWheel's growing dataset), via lerobot-edit-dataset merge."""
        mix_repo_id = f"local/{run_id}-rollouts-mix-c{cycle}"
        mix_root = workdir / f"cycle_{cycle}" / "rollouts_mix"
        cmd = [
            "lerobot-edit-dataset",
            "--operation.type=merge",
            f"--operation.repo_ids=[{a.repo_id}, {b.repo_id}]",
            f"--operation.roots=[{a.root}, {b.root}]",
            f"--new_repo_id={mix_repo_id}",
            f"--new_root={mix_root}",
            "--push_to_hub=false",
        ]
        self.bus.emit(Event(run_id, "improve", "job_started", cycle,
                            {"cmd": cmd, "stage": "merge"}))
        result = self.runner(cmd, mix_root.parent / "rollouts_merge.log")
        self.bus.emit(Event(run_id, "improve", "job_finished", cycle,
                            {"exit_code": result.exit_code, "duration_s": result.duration_s,
                             "log": str(result.log_path)}))
        if result.exit_code != 0:
            raise ImproveError(f"rollout merge exited {result.exit_code}, see {result.log_path}")
        total = (a.num_episodes or 0) + (b.num_episodes or 0)
        merged = DatasetRef(repo_id=mix_repo_id, root=str(mix_root),
                            num_episodes=total or None,
                            notes=f"rollout mix through cycle {cycle}")
        self.bus.emit(Event(run_id, "improve", "rollouts_merged", cycle,
                            {"repo_id": mix_repo_id, "episodes": total}))
        return merged

    @staticmethod
    def _parse_summary(log_path: Path | None) -> dict:
        """The collector prints a JSON summary as its LAST stdout line."""
        if log_path is None or not Path(log_path).exists():
            raise ImproveError(f"rollout log missing: {log_path}")
        for line in reversed(Path(log_path).read_text().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        raise ImproveError(f"no JSON summary found in {log_path}")
