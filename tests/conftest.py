import json
from pathlib import Path

import pytest

from leagents.agents.base import RunResult
from leagents.config import LoopConfig
from leagents.events import EventBus
from leagents.orchestrator.constitution import Constitution


def _arg(cmd, prefix: str) -> str:
    return next(a.split("=", 1)[1] for a in cmd if a.startswith(prefix))


def make_train_runner(duration_s: float = 3600.0, seen_cmds: list | None = None):
    """Fake lerobot-train: creates the expected checkpoint directory."""

    def runner(cmd, log_path: Path) -> RunResult:
        if seen_cmds is not None:
            seen_cmds.append(list(cmd))
        output_dir = Path(_arg(cmd, "--output_dir="))
        (output_dir / "checkpoints" / "last" / "pretrained_model").mkdir(
            parents=True, exist_ok=True
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("fake train\n")
        return RunResult(list(cmd), 0, duration_s, log_path)

    return runner


def make_eval_runner(scores: list[float], duration_s: float = 60.0):
    """Fake lerobot-eval: writes eval_info.json with scripted success rates."""
    calls = {"n": 0}

    def runner(cmd, log_path: Path) -> RunResult:
        output_dir = Path(_arg(cmd, "--output_dir="))
        score = scores[min(calls["n"], len(scores) - 1)]
        calls["n"] += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "eval_info.json").write_text(
            json.dumps(  # mirrors the lerobot 0.5 eval_info.json schema
                {
                    "overall": {"pc_success": score * 100.0},
                    "per_group": {"suite_a": {"pc_success": score * 100.0}},
                }
            )
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("fake eval\n")
        return RunResult(list(cmd), 0, duration_s, log_path)

    return runner


@pytest.fixture
def constitution() -> Constitution:
    return Constitution(
        {
            "allowed_env_types": ["libero", "metaworld", "pusht"],  # mirrors configs/constitution.yaml
            "max_train_steps": 200_000,
            "max_eval_episodes": 500,
            "forbidden_commands": ["lerobot-record", "lerobot-async-inference"],
            "forbidden_args": ["--robot", "--teleop"],
        }
    )


@pytest.fixture
def bus(tmp_path: Path) -> EventBus:
    return EventBus(tmp_path / "events.jsonl")


@pytest.fixture
def loop_config(tmp_path: Path) -> LoopConfig:
    return LoopConfig(
        run_name="test",
        workdir=tmp_path / "runs",
        seed_dataset="test-org/seed",
    )
