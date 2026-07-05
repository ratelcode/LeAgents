import random

import pytest

from leagents.rl.env_adapter import (
    ResidualRollout,
    classify_step,
    init_state_split,
)


def test_classify_ongoing_step_bootstraps():
    o = classify_step(reward=0.0, info={"is_success": False}, step_index=5, max_steps=280)
    assert not o.done and not o.terminated and not o.truncated and o.bootstrap


def test_classify_success_terminates_without_bootstrap():
    o = classify_step(reward=1.0, info={"is_success": True}, step_index=42, max_steps=280)
    assert o.success and o.terminated and o.done and not o.bootstrap and not o.truncated


def test_classify_time_limit_truncates_and_bootstraps():
    o = classify_step(reward=0.0, info={"is_success": False}, step_index=279, max_steps=280)
    assert o.truncated and o.done and o.bootstrap and not o.terminated


def test_classify_success_on_last_step_is_terminal_not_truncated():
    o = classify_step(reward=1.0, info={"is_success": True}, step_index=279, max_steps=280)
    assert o.terminated and not o.truncated and not o.bootstrap


def test_classify_falls_back_to_reward_when_no_is_success():
    assert classify_step(1.0, {}, 3, 280).success  # reward>0 => success
    assert not classify_step(0.0, {}, 3, 280).success


def test_init_state_split_disjoint_and_ranged():
    train, harvest = init_state_split(100, (0, 60), (60, 100))
    assert train[0] == 0 and train[-1] == 59 and len(train) == 60
    assert harvest[0] == 60 and harvest[-1] == 99
    with pytest.raises(ValueError):  # overlap
        init_state_split(100, (0, 70), (60, 100))
    with pytest.raises(ValueError):  # out of range
        init_state_split(100, (0, 60), (60, 120))


class _FakeEnv:
    """Scripted env: succeeds after ``success_after`` steps of each episode.
    Records the init states it was reset with."""

    def __init__(self, success_after, max_steps_guard=1000):
        self.success_after = success_after
        self.reset_states = []
        self._ep_step = 0

    def reset(self, init_state):
        self.reset_states.append(init_state)
        self._ep_step = 0
        return {"obs_for_init": init_state, "step": 0}

    def step(self, action):
        self._ep_step += 1
        success = self._ep_step >= self.success_after
        reward = 1.0 if success else 0.0
        return {"obs_for_init": None, "step": self._ep_step}, reward, {"is_success": success}


class _FakePolicy:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def select_action(self, obs, state, *, global_step, explore, rng):
        return f"exec@{global_step}", f"base@{global_step}"


def test_rollout_resets_on_episode_end_and_cycles_init_states():
    env = _FakeEnv(success_after=3)
    policy = _FakePolicy()
    roll = ResidualRollout(env, extract_state=lambda o: o["step"], max_steps=280)
    trans = roll.collect(policy, n_steps=7, init_states=[10, 20], rng=random.Random(0))

    assert len(trans) == 7
    # episodes end after 3 steps (success) -> transitions 2 and 5 are terminal
    assert [t.done for t in trans] == [False, False, True, False, False, True, False]
    assert [t.bootstrap for t in trans] == [True, True, False, True, True, False, True]
    # reset once at start + once per completed episode (2) = 3; policy reset matches
    assert env.reset_states == [10, 20, 10]  # cycles the init states
    assert policy.resets == 3


def test_rollout_advances_global_step_for_epsilon_ramp():
    env = _FakeEnv(success_after=100)  # never succeeds within the window
    policy = _FakePolicy()
    roll = ResidualRollout(env, extract_state=lambda o: o["step"], max_steps=280)
    trans = roll.collect(policy, n_steps=4, init_states=[0], global_step_start=1000)
    # a_base/a_exec carry the global step the policy saw — monotonically increasing
    assert [t.a_base for t in trans] == [f"base@{1000 + i}" for i in range(4)]


def test_rollout_requires_init_states():
    roll = ResidualRollout(_FakeEnv(3), extract_state=lambda o: o, max_steps=280)
    with pytest.raises(ValueError):
        roll.collect(_FakePolicy(), n_steps=1, init_states=[])
