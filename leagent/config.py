"""Loop configuration, loaded from YAML (see configs/m0_libero.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PolicyRung(BaseModel):
    """One rung of the policy escalation ladder (DESIGN.md §3.3).

    If ``init`` is set the Training Agent fine-tunes from that pretrained
    checkpoint (``--policy.path=init``); otherwise it trains ``name`` from
    scratch (``--policy.type=name``).
    """

    name: str
    init: str | None = None


class BudgetConfig(BaseModel):
    max_cycles: int = 3
    max_gpu_hours: float = 24.0


class ThresholdConfig(BaseModel):
    """Success-rate deltas (fractions in [0, 1]) driving the loop decision."""

    promote_delta: float = 0.05
    regression_delta: float = 0.05
    plateau_epsilon: float = 0.01
    plateau_cycles: int = 2


class TrainConfig(BaseModel):
    steps: int = 20_000
    batch_size: int = 64
    extra_args: list[str] = Field(default_factory=list)


class EvalConfig(BaseModel):
    env_type: str = "libero"
    task_suite: str = "libero_spatial"
    n_episodes: int = 10
    extra_args: list[str] = Field(default_factory=list)


class LoopConfig(BaseModel):
    run_name: str
    workdir: Path = Path("runs")
    seed_dataset: str
    policy_ladder: list[PolicyRung] = Field(
        default_factory=lambda: [
            PolicyRung(name="smolvla", init="lerobot/smolvla_base"),
            PolicyRung(name="pi05", init="lerobot/pi05_base"),
        ]
    )
    budgets: BudgetConfig = Field(default_factory=BudgetConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    constitution: Path = Path("configs/constitution.yaml")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LoopConfig":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(raw)
