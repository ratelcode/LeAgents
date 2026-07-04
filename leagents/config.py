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
    # A plateau only escalates the policy ladder if the baseline already
    # clears this floor. A near-zero plateau means the policy hasn't learned
    # anything yet (under-training / too little data) — a bigger model is
    # not the fix, so the loop iterates instead. 0 disables the guard.
    escalate_floor: float = 0.05


class DataConfig(BaseModel):
    """Progressive data schedule (flywheel-lite, DESIGN.md §3.5).

    With ``initial_episodes`` set, cycle N trains on the first
    ``initial_episodes * growth**N`` episodes (capped at ``max_episodes``) —
    each iterate genuinely adds data, so eval deltas measure data growth.
    None -> always use the full dataset.
    """

    initial_episodes: int | None = None
    growth: float = 2.0
    max_episodes: int | None = None


class TrainConfig(BaseModel):
    steps: int = 20_000
    batch_size: int = 64
    # Continue fine-tuning from the run's blessed checkpoint (same policy)
    # instead of restarting from the ladder's init every cycle.
    continue_from_blessed: bool = True
    extra_args: list[str] = Field(default_factory=list)


class EvalConfig(BaseModel):
    env_type: str = "libero"
    task_suite: str = "libero_spatial"
    n_episodes: int = 10
    batch_size: int = 10  # parallel eval envs; effective value is capped at n_episodes
    extra_args: list[str] = Field(default_factory=list)


class ImproveConfig(BaseModel):
    """DexFlyWheel rollout collection after each promotion (DESIGN.md §3.5).

    Off by default: it costs GPU time and only makes sense once the policy
    succeeds sometimes (success-filtered collection of a 0% policy keeps
    nothing).
    """

    enabled: bool = False
    episodes: int = 20  # successful episodes to keep per promotion
    device: str = "cuda"
    task_text: str | None = None  # language instruction stored per frame
    extra_args: list[str] = Field(default_factory=list)


class KnowledgeConfig(BaseModel):
    """OKF knowledge layer (DESIGN.md §3.6)."""

    enabled: bool = True
    root: Path = Path("knowledge")


class LoopConfig(BaseModel):
    run_name: str
    workdir: Path = Path("runs")
    seed_dataset: str
    # Provider-agnostic LLM spec (leagents/llm/adapter.py), e.g.
    # "anthropic:claude-sonnet-5", "openai:gpt-5.2",
    # "openai:qwen3@http://localhost:11434/v1". None -> deterministic fallbacks.
    llm: str | None = None
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    improve: ImproveConfig = Field(default_factory=ImproveConfig)
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
