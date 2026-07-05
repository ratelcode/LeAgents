import random

import pytest

from leagents.rl.config import ResidualRLConfig

torch = pytest.importorskip("torch")

from leagents.rl.composition import ComposedPolicy  # noqa: E402


def _counting_base(chunk):
    calls = {"n": 0}

    def predict(_obs):
        calls["n"] += 1
        return chunk

    return predict, calls


def _zero_residual(_state, a_base):
    return torch.zeros_like(a_base)


def _saturating_residual(_state, a_base):
    return torch.full_like(a_base, 100.0)  # tanh -> ~1


def test_zero_residual_composes_to_base():
    chunk = torch.tensor([[0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.4]]).repeat(5, 1)
    base, _ = _counting_base(chunk)
    p = ComposedPolicy(base, _zero_residual, n_action_steps=5, alpha=0.5,
                       exploration_ramp_steps=100)
    a_exec, a_base = p.select_action(None, None, explore=False)
    assert torch.allclose(a_exec, a_base)
    # exploring can only pick composed(==base) or base -> still base
    p.reset()
    a_exec, a_base = p.select_action(None, None, global_step=50, explore=True,
                                     rng=random.Random(0))
    assert torch.allclose(a_exec, a_base)


def test_residual_is_alpha_tanh_bounded():
    base, _ = _counting_base(torch.zeros(5, 7))  # a_base = 0
    p = ComposedPolicy(base, _saturating_residual, n_action_steps=5, alpha=0.5,
                       exploration_ramp_steps=0)
    a_exec, _ = p.select_action(None, None, explore=False)
    assert torch.allclose(a_exec, torch.full((7,), 0.5), atol=1e-4)  # 0 + 0.5*tanh(100)


def test_composed_action_clipped_to_bounds():
    base, _ = _counting_base(torch.full((5, 7), 0.9))
    p = ComposedPolicy(base, _saturating_residual, n_action_steps=5, alpha=0.5,
                       exploration_ramp_steps=0, action_low=-1.0, action_high=1.0)
    a_exec, _ = p.select_action(None, None, explore=False)
    assert torch.allclose(a_exec, torch.ones(7))  # 0.9 + 0.5 -> clip 1.0


def test_base_called_once_per_n_action_steps():
    chunk = torch.arange(5 * 7, dtype=torch.float32).reshape(5, 7)
    base, calls = _counting_base(chunk)
    p = ComposedPolicy(base, _zero_residual, n_action_steps=5, alpha=0.5,
                       exploration_ramp_steps=0)
    bases = [p.select_action(None, None, explore=False)[1] for _ in range(7)]
    assert calls["n"] == 2  # refill at step 0 and step 5 only
    for i in range(5):
        assert torch.allclose(bases[i], chunk[i])
    assert torch.allclose(bases[5], chunk[0]) and torch.allclose(bases[6], chunk[1])


def test_reset_forces_a_refill():
    base, calls = _counting_base(torch.zeros(5, 7))
    p = ComposedPolicy(base, _zero_residual, n_action_steps=5, alpha=0.5,
                       exploration_ramp_steps=0)
    p.select_action(None, None, explore=False)
    assert calls["n"] == 1
    p.reset()
    p.select_action(None, None, explore=False)
    assert calls["n"] == 2  # reset cleared the cache


def test_epsilon_ramp_linear_and_clamped():
    base, _ = _counting_base(torch.zeros(5, 7))
    p = ComposedPolicy(base, _zero_residual, n_action_steps=5, alpha=0.5,
                       exploration_ramp_steps=100)
    assert p.epsilon(0) == 0.0
    assert p.epsilon(50) == 0.5
    assert p.epsilon(100) == 1.0
    assert p.epsilon(500) == 1.0
    # a zero ramp means "always composed"
    p0 = ComposedPolicy(base, _zero_residual, n_action_steps=5, alpha=0.5,
                        exploration_ramp_steps=0)
    assert p0.epsilon(0) == 1.0


def test_exploration_gate_picks_base_or_composed_by_draw():
    class FixedRng:
        def __init__(self, v):
            self.v = v

        def random(self):
            return self.v

    base, _ = _counting_base(torch.zeros(5, 7))  # base=0, composed=0.5
    p = ComposedPolicy(base, _saturating_residual, n_action_steps=5, alpha=0.5,
                       exploration_ramp_steps=100)
    # eps(50)=0.5; draw 0.4 < eps -> composed
    a_exec, _ = p.select_action(None, None, global_step=50, explore=True, rng=FixedRng(0.4))
    assert torch.allclose(a_exec, torch.full((7,), 0.5), atol=1e-4)
    p.reset()
    # draw 0.6 >= eps -> base only
    a_exec, a_base = p.select_action(None, None, global_step=50, explore=True, rng=FixedRng(0.6))
    assert torch.allclose(a_exec, a_base)


def test_config_defaults_validation_and_disjointness():
    c = ResidualRLConfig()
    assert not c.enabled and c.alpha == 0.5 and c.n_action_steps == 5
    with pytest.raises(ValueError):  # overlapping train/harvest ranges
        ResidualRLConfig(train_init_states=(0, 70), harvest_init_states=(60, 100))
    with pytest.raises(ValueError):  # alpha out of (0, 1]
        ResidualRLConfig(alpha=1.5)
    with pytest.raises(ValueError):  # range outside [0, 100]
        ResidualRLConfig(train_init_states=(0, 120))


def test_resolve_target_tasks_auto_and_explicit():
    per = {0: 0.9, 5: 0.2, 8: 0.4, 3: 0.8}
    auto = ResidualRLConfig(target_tasks="auto", success_floor=0.5)
    assert auto.resolve_target_tasks(per) == [5, 8]  # below floor, weakest first
    explicit = ResidualRLConfig(target_tasks=[7, 2])
    assert explicit.resolve_target_tasks(per) == [7, 2]
