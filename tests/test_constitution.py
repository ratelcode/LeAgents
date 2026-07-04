import pytest

from leloop.orchestrator.constitution import Constitution


@pytest.fixture
def rules(constitution: Constitution) -> Constitution:
    return constitution


def test_allows_sim_train_command(rules):
    verdict = rules.check_command(["lerobot-train", "--policy.path=lerobot/smolvla_base"])
    assert verdict.allowed


def test_denies_real_hardware_args(rules):
    verdict = rules.check_command(["lerobot-train", "--robot.type=so101"])
    assert not verdict.allowed
    assert verdict.rule == "forbidden_args"


def test_denies_cve_affected_command(rules):
    verdict = rules.check_command(["lerobot-async-inference", "--port=8080"])
    assert not verdict.allowed
    assert verdict.rule == "forbidden_commands"


def test_denies_over_budget_train(rules):
    assert not rules.check_train(1_000_000).allowed
    assert rules.check_train(20_000).allowed


def test_eval_env_allowlist(rules):
    assert rules.check_eval("libero", 10).allowed
    assert not rules.check_eval("aloha_real", 10).allowed
    assert not rules.check_eval("libero", 10_000).allowed
