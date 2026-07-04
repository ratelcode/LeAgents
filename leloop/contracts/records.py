"""Typed records exchanged between agents.

Datasets themselves live on the HF Hub in LeRobotDataset v3.0 format —
these records only carry references and results across stage boundaries.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Proposal(BaseModel):
    """A task/data proposal from the proposer — advisory input to the Data
    Agent, never a control-flow decision (DESIGN.md §4)."""

    action: str  # "reuse_seed" | "recollect_failing"
    dataset: str
    num_episodes: int | None = None  # None -> full dataset
    notes: str = ""


class DatasetRef(BaseModel):
    repo_id: str
    revision: str | None = None
    num_episodes: int | None = None
    notes: str = ""


class CheckpointRecord(BaseModel):
    policy_type: str
    path: str
    cycle: int
    blessed: bool = False


class EvalReport(BaseModel):
    checkpoint: str
    env_type: str
    task_suite: str
    n_episodes: int
    success_rate: float  # fraction in [0, 1]
    per_task: dict[str, float] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)
