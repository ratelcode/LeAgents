import json
import sys
from pathlib import Path

import pytest

from leloop.agents import ImproveAgent, ImproveError
from leloop.agents.base import RunResult
from leloop.config import EvalConfig, ImproveConfig
from leloop.contracts import CheckpointRecord
from leloop.orchestrator.constitution import ConstitutionError


def make_rollout_runner(summary: dict, chatter: str = "attempt 1: suite/0 success=True\n"):
    """Fake collect_rollouts subprocess: writes chatter + JSON summary line."""

    def runner(cmd, log_path: Path) -> RunResult:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(chatter + json.dumps(summary) + "\n")
        return RunResult(list(cmd), 0, 120.0, log_path)

    return runner


def _ckpt() -> CheckpointRecord:
    return CheckpointRecord(policy_type="smolvla", path="/ckpt/blessed", cycle=0)


def _agent(constitution, bus, runner, improve_cfg=None, eval_cfg=None):
    return ImproveAgent(
        improve_cfg or ImproveConfig(enabled=True, episodes=5, device="cpu"),
        eval_cfg or EvalConfig(env_type="pusht", task_suite="PushT-v0", n_episodes=4),
        constitution, bus, runner,
    )


def test_command_shape(constitution, bus, tmp_path):
    agent = _agent(constitution, bus, make_rollout_runner({"kept": 3, "attempts": 5}))
    cmd = agent.build_command(_ckpt(), tmp_path / "rollouts", "local/r-c0-rollouts")
    assert cmd[:3] == [sys.executable, "-m", "leloop.scripts.collect_rollouts"]
    assert "--policy-path=/ckpt/blessed" in cmd
    assert "--env-type=pusht" in cmd and "--task=PushT-v0" in cmd
    assert "--episodes=5" in cmd and "--device=cpu" in cmd


def test_run_returns_dataset_ref_and_summary(constitution, bus, tmp_path):
    agent = _agent(constitution, bus,
                   make_rollout_runner({"kept": 3, "attempts": 7, "success_rate": 0.43}))
    ref, summary = agent.run(run_id="r1", cycle=2, checkpoint=_ckpt(), workdir=tmp_path)
    assert ref.repo_id == "local/r1-cycle2-rollouts"
    assert ref.num_episodes == 3
    assert summary["attempts"] == 7


def test_missing_summary_raises(constitution, bus, tmp_path):
    agent = _agent(constitution, bus, make_rollout_runner({}, chatter="no json here\n"))

    def bad_runner(cmd, log_path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("no json here\n")
        return RunResult(list(cmd), 0, 1.0, log_path)

    agent.runner = bad_runner
    with pytest.raises(ImproveError, match="no JSON summary"):
        agent.run(run_id="r1", cycle=0, checkpoint=_ckpt(), workdir=tmp_path)


def test_denied_env_refused(constitution, bus, tmp_path):
    agent = _agent(constitution, bus, make_rollout_runner({"kept": 1, "attempts": 1}),
                   eval_cfg=EvalConfig(env_type="aloha_real", task_suite="x", n_episodes=1))
    with pytest.raises(ConstitutionError):
        agent.run(run_id="r1", cycle=0, checkpoint=_ckpt(), workdir=tmp_path)


def test_collector_helpers():
    torch = pytest.importorskip("torch")
    from leloop.scripts.collect_rollouts import (
        chw_float_to_hwc_uint8,
        features_from_rollout,
        first_done_index,
    )

    done = torch.tensor([False, False, True, True])
    assert first_done_index(done) == 2
    assert first_done_index(torch.tensor([False, False])) == 1

    img = torch.zeros(3, 4, 5)
    img[0] = 1.0
    hwc = chw_float_to_hwc_uint8(img)
    assert hwc.shape == (4, 5, 3) and hwc[0, 0, 0] == 255 and hwc[0, 0, 1] == 0

    obs = {"observation.image": torch.zeros(1, 6, 3, 8, 9),
           "observation.state": torch.zeros(1, 6, 2)}
    action = torch.zeros(1, 5, 7)
    features = features_from_rollout(obs, action)
    assert features["observation.image"] == {
        "dtype": "video", "shape": (8, 9, 3), "names": ["height", "width", "channels"]
    }
    assert features["observation.state"]["shape"] == (2,)
    assert features["action"]["shape"] == (7,)


def test_loop_dispatches_improve_on_promote(loop_config, constitution, tmp_path):
    from leloop.agents import DataAgent, EvalAgent, TrainAgent
    from leloop.events import EventBus
    from leloop.orchestrator import DeterministicProposer, LoopController
    from leloop.store import JobStore
    from tests.conftest import make_eval_runner, make_train_runner

    bus = EventBus(tmp_path / "events.jsonl")
    calls = []

    class SpyImprove:
        def run(self, **kwargs):
            calls.append(kwargs)

    controller = LoopController(
        cfg=loop_config,
        store=JobStore(tmp_path / "leloop.db"),
        bus=bus,
        data_agent=DataAgent(bus),
        train_agent=TrainAgent(loop_config.train, constitution, bus, make_train_runner()),
        eval_agent=EvalAgent(loop_config.eval, constitution, bus,
                             make_eval_runner([0.0, 0.5])),  # cycle0 0% -> no harvest
        proposer=DeterministicProposer(loop_config.seed_dataset),
        improve_agent=SpyImprove(),
    )
    loop_config.budgets.max_cycles = 2
    controller.run("run-improve-hook", tmp_path / "wd")

    # cycle 0 promotes at 0% (baseline) -> skipped; cycle 1 promotes at 50% -> harvested
    assert len(calls) == 1
    assert calls[0]["cycle"] == 1
