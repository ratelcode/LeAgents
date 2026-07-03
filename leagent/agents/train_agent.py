"""Training Agent — subprocess wrapper over `lerobot-train` (DESIGN.md §3.3)."""

from __future__ import annotations

from pathlib import Path

from leagent.agents.base import Runner, subprocess_runner
from leagent.config import PolicyRung, TrainConfig
from leagent.contracts import CheckpointRecord, DatasetRef
from leagent.events import Event, EventBus
from leagent.orchestrator.constitution import Constitution, ConstitutionError


class TrainError(Exception):
    pass


class TrainAgent:
    def __init__(
        self,
        cfg: TrainConfig,
        constitution: Constitution,
        bus: EventBus,
        runner: Runner = subprocess_runner,
    ):
        self.cfg = cfg
        self.constitution = constitution
        self.bus = bus
        self.runner = runner

    def build_command(self, rung: PolicyRung, dataset: DatasetRef, output_dir: Path) -> list[str]:
        policy_arg = (
            f"--policy.path={rung.init}" if rung.init else f"--policy.type={rung.name}"
        )
        return [
            "lerobot-train",
            policy_arg,
            # lerobot >= 0.5 refuses to train without policy.repo_id unless hub
            # push is explicitly off; leagent manages checkpoints locally.
            "--policy.push_to_hub=false",
            f"--dataset.repo_id={dataset.repo_id}",
            f"--output_dir={output_dir}",
            f"--steps={self.cfg.steps}",
            f"--batch_size={self.cfg.batch_size}",
            *self.cfg.extra_args,
        ]

    def run(
        self, *, run_id: str, cycle: int, rung: PolicyRung, dataset: DatasetRef, workdir: Path
    ) -> CheckpointRecord:
        output_dir = workdir / f"cycle_{cycle}" / "train"
        cmd = self.build_command(rung, dataset, output_dir)

        for verdict in (self.constitution.check_train(self.cfg.steps),
                        self.constitution.check_command(cmd)):
            if not verdict.allowed:
                self.bus.emit(Event(run_id, "train", "constitution_denied", cycle,
                                    {"rule": verdict.rule, "reason": verdict.reason}))
                raise ConstitutionError(verdict)

        self.bus.emit(Event(run_id, "train", "job_started", cycle, {"cmd": cmd}))
        # log lives outside output_dir: lerobot-train refuses a pre-existing output_dir
        result = self.runner(cmd, output_dir.parent / "train.log")
        self.bus.emit(Event(run_id, "train", "job_finished", cycle,
                            {"exit_code": result.exit_code, "duration_s": result.duration_s,
                             "log": str(result.log_path)}))
        if result.exit_code != 0:
            raise TrainError(f"lerobot-train exited {result.exit_code}, see {result.log_path}")

        # lerobot-train checkpoint layout: output_dir/checkpoints/last/pretrained_model
        checkpoint = output_dir / "checkpoints" / "last" / "pretrained_model"
        if not checkpoint.exists():
            self.bus.emit(Event(run_id, "train", "warning", cycle,
                                {"msg": f"expected checkpoint missing: {checkpoint}"}))
        return CheckpointRecord(policy_type=rung.name, path=str(checkpoint), cycle=cycle)
