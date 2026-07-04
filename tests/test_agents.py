import pytest

from leagents.agents import EvalAgent, TrainAgent
from leagents.config import EvalConfig, PolicyRung, TrainConfig
from leagents.contracts import CheckpointRecord, DatasetRef
from leagents.orchestrator.constitution import ConstitutionError
from tests.conftest import make_eval_runner, make_train_runner


def test_train_command_finetunes_from_init(constitution, bus, tmp_path):
    agent = TrainAgent(TrainConfig(steps=100, batch_size=8), constitution, bus,
                       make_train_runner())
    cmd = agent.build_command(
        PolicyRung(name="smolvla", init="lerobot/smolvla_base"),
        DatasetRef(repo_id="org/data"),
        tmp_path / "out",
    )
    assert cmd[0] == "lerobot-train"
    assert "--policy.path=lerobot/smolvla_base" in cmd
    assert "--policy.push_to_hub=false" in cmd  # lerobot >= 0.5 refuses otherwise
    assert "--dataset.repo_id=org/data" in cmd
    assert "--steps=100" in cmd


def test_train_command_from_scratch_uses_type(constitution, bus, tmp_path):
    agent = TrainAgent(TrainConfig(), constitution, bus, make_train_runner())
    cmd = agent.build_command(PolicyRung(name="act"), DatasetRef(repo_id="org/data"),
                              tmp_path / "out")
    assert "--policy.type=act" in cmd


def test_train_command_init_override_beats_ladder_init(constitution, bus, tmp_path):
    agent = TrainAgent(TrainConfig(), constitution, bus, make_train_runner())
    cmd = agent.build_command(
        PolicyRung(name="smolvla", init="lerobot/smolvla_base"),
        DatasetRef(repo_id="org/data"),
        tmp_path / "out",
        init_override="/runs/prev/checkpoints/last/pretrained_model",
    )
    assert "--policy.path=/runs/prev/checkpoints/last/pretrained_model" in cmd
    assert "--policy.path=lerobot/smolvla_base" not in cmd


def test_train_command_episode_subset(constitution, bus, tmp_path):
    agent = TrainAgent(TrainConfig(), constitution, bus, make_train_runner())
    cmd = agent.build_command(
        PolicyRung(name="smolvla", init="lerobot/smolvla_base"),
        DatasetRef(repo_id="org/data", num_episodes=3),
        tmp_path / "out",
    )
    assert "--dataset.episodes=[0, 1, 2]" in cmd


def test_train_command_explicit_episode_indices_win(constitution, bus, tmp_path):
    agent = TrainAgent(TrainConfig(), constitution, bus, make_train_runner())
    cmd = agent.build_command(
        PolicyRung(name="smolvla", init="lerobot/smolvla_base"),
        DatasetRef(repo_id="org/data", episodes=[1261, 1262, 1300], num_episodes=3),
        tmp_path / "out",
    )
    assert "--dataset.episodes=[1261, 1262, 1300]" in cmd
    assert "--dataset.episodes=[0, 1, 2]" not in cmd


def test_train_run_returns_checkpoint(constitution, bus, tmp_path):
    agent = TrainAgent(TrainConfig(steps=100), constitution, bus, make_train_runner())
    record = agent.run(
        run_id="r", cycle=0,
        rung=PolicyRung(name="smolvla", init="lerobot/smolvla_base"),
        dataset=DatasetRef(repo_id="org/data"), workdir=tmp_path,
    )
    assert record.policy_type == "smolvla"
    assert record.path.endswith("checkpoints/last/pretrained_model")


def test_train_denied_by_constitution(constitution, bus, tmp_path):
    agent = TrainAgent(TrainConfig(steps=1_000_000), constitution, bus, make_train_runner())
    with pytest.raises(ConstitutionError):
        agent.run(
            run_id="r", cycle=0, rung=PolicyRung(name="smolvla"),
            dataset=DatasetRef(repo_id="org/data"), workdir=tmp_path,
        )


def test_eval_parses_success_rate(constitution, bus, tmp_path):
    agent = EvalAgent(EvalConfig(n_episodes=5), constitution, bus,
                      make_eval_runner(scores=[0.65]))
    report = agent.run(
        run_id="r", cycle=0,
        checkpoint=CheckpointRecord(policy_type="smolvla", path="/ckpt", cycle=0),
        workdir=tmp_path,
    )
    assert report.success_rate == pytest.approx(0.65)
    assert report.per_task == {"suite_a": pytest.approx(0.65)}
    assert report.task_suite == "libero_spatial"


def test_eval_denied_env(constitution, bus, tmp_path):
    agent = EvalAgent(EvalConfig(env_type="aloha_real"), constitution, bus,
                      make_eval_runner(scores=[0.5]))
    with pytest.raises(ConstitutionError):
        agent.run(
            run_id="r", cycle=0,
            checkpoint=CheckpointRecord(policy_type="smolvla", path="/ckpt", cycle=0),
            workdir=tmp_path,
        )
