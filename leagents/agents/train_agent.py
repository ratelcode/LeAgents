"""Training Agent — subprocess wrapper over `lerobot-train` (DESIGN.md §3.3)."""

from __future__ import annotations

from pathlib import Path

from leagents.agents.base import Runner, subprocess_runner
from leagents.config import PolicyRung, TrainConfig
from leagents.contracts import CheckpointRecord, DatasetRef
from leagents.events import Event, EventBus
from leagents.orchestrator.constitution import Constitution, ConstitutionError


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

    def build_command(
        self,
        rung: PolicyRung,
        dataset: DatasetRef,
        output_dir: Path,
        init_override: str | None = None,
        steps: int | None = None,
    ) -> list[str]:
        init = init_override or rung.init
        policy_arg = f"--policy.path={init}" if init else f"--policy.type={rung.name}"
        episodes_args = (
            [f"--dataset.episodes={list(range(dataset.num_episodes))}"]
            if dataset.num_episodes and dataset.root is None
            else []
        )
        root_args = [f"--dataset.root={dataset.root}"] if dataset.root else []
        return [
            "lerobot-train",
            policy_arg,
            # lerobot >= 0.5 refuses to train without policy.repo_id unless hub
            # push is explicitly off; leagents manages checkpoints locally.
            "--policy.push_to_hub=false",
            f"--dataset.repo_id={dataset.repo_id}",
            *root_args,
            *episodes_args,
            f"--output_dir={output_dir}",
            f"--steps={steps or self.cfg.steps}",
            f"--batch_size={self.cfg.batch_size}",
            *self.cfg.extra_args,
        ]

    def run(
        self,
        *,
        run_id: str,
        cycle: int,
        rung: PolicyRung,
        dataset: DatasetRef,
        workdir: Path,
        init_override: str | None = None,
        steps: int | None = None,
        stage: str = "train",
    ) -> CheckpointRecord:
        output_dir = workdir / f"cycle_{cycle}" / stage
        cmd = self.build_command(rung, dataset, output_dir, init_override, steps)

        for verdict in (self.constitution.check_train(steps or self.cfg.steps),
                        self.constitution.check_command(cmd)):
            if not verdict.allowed:
                self.bus.emit(Event(run_id, "train", "constitution_denied", cycle,
                                    {"rule": verdict.rule, "reason": verdict.reason}))
                raise ConstitutionError(verdict)

        self.bus.emit(Event(run_id, "train", "job_started", cycle, {"cmd": cmd, "stage": stage}))
        # log lives outside output_dir: lerobot-train refuses a pre-existing output_dir
        result = self.runner(cmd, output_dir.parent / f"{stage}.log")
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
