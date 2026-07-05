"""P2.1 unit tests (CPU, fakes — no GPU/LIBERO): residual SAC wrappers, the
a_base buffer bookkeeping, RLPD 50/50 mixing, the copied SAC update, and the
full residual_sac_train loop against a scripted env; plus the clean-interpreter
CVE check for the new lerobot-touching modules."""

import random
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")

from leagents.rl.composition import ComposedPolicy  # noqa: E402
from leagents.rl.config import ResidualRLConfig  # noqa: E402
from leagents.rl.env_adapter import Transition  # noqa: E402
from leagents.rl.sac_wrappers import (  # noqa: E402
    build_residual_sac,
    make_residual_forward,
    residual_observation,
    zero_residual,
)
from leagents.rl.train import (  # noqa: E402
    ActingPolicy,
    attach_next_a_base,
    add_transitions,
    make_sac_optimizers,
    residual_sac_train,
    sac_update_step,
    sample_mixed,
)

STATE_DIM, ACTION_DIM = 8, 7


def small_policy(**kwargs):
    return build_residual_sac(device="cpu", residual_alpha=0.5, num_critics=2, **kwargs)


# --------------------------------------------------------------------------
# actor / composition
# --------------------------------------------------------------------------


def test_small_init_deterministic_residual_is_near_zero_so_composed_equals_base():
    policy = small_policy()
    det = make_residual_forward(policy, deterministic=True)
    state = torch.randn(STATE_DIM)
    a_base = torch.rand(ACTION_DIM) * 1.6 - 0.8

    residual = det(state, a_base)
    assert residual.shape == (ACTION_DIM,)
    delta = 0.5 * torch.tanh(residual)
    assert torch.all(delta.abs() < 0.02), f"residual too large at init: {delta}"

    composed = ComposedPolicy(
        lambda obs: a_base.repeat(50, 1), det,
        n_action_steps=5, alpha=0.5, exploration_ramp_steps=0)
    a_exec, a_b = composed.select_action(None, state)
    assert torch.equal(a_b, a_base)
    assert torch.allclose(a_exec, a_base, atol=0.02)


def test_zero_residual_composes_to_base_exactly():
    a_base = torch.rand(ACTION_DIM) * 2 - 1
    composed = ComposedPolicy(
        lambda obs: a_base.repeat(50, 1), zero_residual,
        n_action_steps=5, alpha=0.5, exploration_ramp_steps=0)
    a_exec, _ = composed.select_action(None, torch.zeros(STATE_DIM))
    assert torch.equal(a_exec, torch.clamp(a_base, -1, 1))


def test_stochastic_residual_forward_varies_deterministic_does_not():
    policy = small_policy()
    state, a_base = torch.zeros(STATE_DIM), torch.zeros(ACTION_DIM)
    stoch = make_residual_forward(policy, deterministic=False)
    det = make_residual_forward(policy, deterministic=True)
    assert not torch.equal(stoch(state, a_base), stoch(state, a_base))
    assert torch.equal(det(state, a_base), det(state, a_base))


def test_compose_recovers_a_base_from_last_dims_and_clips():
    policy = small_policy()
    state = torch.zeros(1, STATE_DIM)
    a_base = torch.full((1, ACTION_DIM), 0.9)
    obs = residual_observation(state, a_base)
    assert obs["observation.state"].shape == (1, STATE_DIM + ACTION_DIM)
    assert torch.equal(policy.base_action_from_obs(obs), a_base)
    # residual +1 at alpha 0.5 from a_base 0.9 -> 1.4, clipped to 1.0
    composed = policy._compose(obs, torch.ones(1, ACTION_DIM))
    assert torch.equal(composed, torch.ones(1, ACTION_DIM))


# --------------------------------------------------------------------------
# a_base bookkeeping + buffer semantics (design §1.4, §2.2)
# --------------------------------------------------------------------------


def _transition(i, *, reward=0.0, done=False, bootstrap=True):
    return Transition(
        state=torch.full((STATE_DIM,), float(i)),
        a_base=torch.full((ACTION_DIM,), float(i)),
        a_exec=torch.full((ACTION_DIM,), float(i) + 0.5),
        reward=reward,
        next_state=torch.full((STATE_DIM,), float(i + 1)),
        done=done,
        bootstrap=bootstrap,
    )


def test_attach_next_a_base_bookkeeping():
    transitions = [
        _transition(0),                                            # mid-episode
        _transition(1, reward=1.0, done=True, bootstrap=False),    # success terminal
        _transition(2),                                            # mid-episode (new ep)
        _transition(3, done=True, bootstrap=True),                 # time-limit truncation
        _transition(4),                                            # collection cut (no successor)
    ]
    paired = attach_next_a_base(transitions)
    assert torch.equal(paired[0][1], transitions[1].a_base)     # next step's a_base
    assert torch.equal(paired[1][1], torch.zeros(ACTION_DIM))   # masked, never read
    assert torch.equal(paired[2][1], transitions[3].a_base)
    assert torch.equal(paired[3][1], transitions[3].a_base)     # smoothness approximation
    assert torch.equal(paired[4][1], transitions[4].a_base)     # cut: same approximation


def _buffer(transitions):
    from lerobot.rl.buffer import ReplayBuffer

    buf = ReplayBuffer(capacity=64, device="cpu", state_keys=["observation.state"],
                       storage_device="cpu", use_drq=False)
    add_transitions(buf, transitions)
    return buf


def test_add_transitions_shapes_and_done_semantics():
    success = _buffer([_transition(0, reward=1.0, done=True, bootstrap=False)])
    batch = success.sample(1)
    assert batch["state"]["observation.state"].shape == (1, STATE_DIM + ACTION_DIM)
    assert batch["action"].shape == (1, ACTION_DIM)
    assert batch["done"].item() == 1.0      # success => no bootstrap
    assert batch["truncated"].item() == 0.0
    # the state vector layout is [state(8), a_base(7)]
    assert torch.equal(batch["state"]["observation.state"][0, STATE_DIM:],
                       torch.zeros(ACTION_DIM))

    truncated = _buffer([_transition(0, done=True, bootstrap=True)])
    batch = truncated.sample(1)
    assert batch["done"].item() == 0.0      # truncation DOES bootstrap
    assert batch["truncated"].item() == 1.0


def test_sample_mixed_is_symmetric_50_50():
    online = _buffer([_transition(i, reward=0.0) for i in range(8)])
    demo = _buffer([_transition(i, reward=1.0) for i in range(8)])
    batch = sample_mixed(online, demo, batch_size=8)
    assert batch["reward"].shape == (8,)
    assert batch["reward"].sum().item() == 4.0  # exactly half from the demo buffer
    # without a demo buffer, all online
    assert sample_mixed(online, None, 8)["reward"].sum().item() == 0.0


# --------------------------------------------------------------------------
# the copied SAC update
# --------------------------------------------------------------------------


def test_sac_update_step_runs_with_finite_losses_and_updates_params():
    torch.manual_seed(0)
    policy = small_policy()
    optimizers = make_sac_optimizers(policy)
    transitions = [_transition(i % 4, reward=float(i % 2)) for i in range(16)]
    buf = _buffer(transitions)
    before = [p.detach().clone() for p in policy.critic_ensemble.parameters()]

    info = sac_update_step(policy, optimizers, buf.sample(16), update_actor=True)

    for key in ("loss_critic", "loss_actor", "loss_temperature"):
        assert key in info and torch.isfinite(torch.tensor(info[key])), info
    after = list(policy.critic_ensemble.parameters())
    assert any(not torch.equal(b, a.detach()) for b, a in zip(before, after))


# --------------------------------------------------------------------------
# the full loop against a scripted env + fake base
# --------------------------------------------------------------------------


class _FakeBase:
    """Chunk predictor: constant small chunk; counts resets/forwards."""

    def __init__(self):
        self.resets = 0
        self.forwards = 0
        self.chunk = torch.linspace(-0.2, 0.2, 50 * ACTION_DIM).reshape(50, ACTION_DIM)

    def reset(self):
        self.resets += 1

    def predict_chunk(self, obs):
        self.forwards += 1
        return self.chunk


class _FakeTaskEnv:
    """Succeeds after ``success_after`` steps; obs carries an 8-d state."""

    max_steps = 10

    def __init__(self, success_after=4):
        self.success_after = success_after
        self._t = 0

    def reset(self, init_state):
        self._t = 0
        return {"state8": [float(init_state)] * STATE_DIM}

    def step(self, action):
        self._t += 1
        success = self._t >= self.success_after
        return ({"state8": [float(self._t)] * STATE_DIM},
                1.0 if success else 0.0,
                {"is_success": success})


def _extract_state(obs):
    return torch.tensor(obs["state8"], dtype=torch.float32)


def _tiny_cfg(**overrides):
    defaults = dict(
        enabled=True,
        train_init_states=(0, 2),
        harvest_init_states=(2, 4),
        env_steps=12,
        warmup_env_steps=6,
        critic_burnin_updates=2,
        steps_per_iter=6,
        exploration_ramp_steps=8,
        n_action_steps=3,
        utd=2,
        batch_size=16,
        buffer_capacity=256,
    )
    defaults.update(overrides)
    return ResidualRLConfig(**defaults)


def test_residual_sac_train_full_loop_with_fakes():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    result = residual_sac_train(
        cfg, _FakeBase(), _FakeTaskEnv(), _extract_state, device="cpu", seed=0)

    assert result.env_steps == 12
    assert result.updates == 12 * cfg.utd + cfg.critic_burnin_updates
    assert result.metrics["loss_critic"] and all(
        torch.isfinite(torch.tensor(v)) for v in result.metrics["loss_critic"])
    assert result.metrics["loss_actor"]  # actor moved after burn-in
    # inner gate ran on the held-out states and produced rates in [0, 1]
    assert 0.0 <= result.base_success <= 1.0
    assert 0.0 <= result.composed_success <= 1.0
    assert result.gate_passed == (result.composed_gain >= cfg.min_composed_gain)
    assert result.residual_state_dict  # the harvestable artifact


def test_residual_sac_train_uses_demo_transitions():
    torch.manual_seed(0)
    demos = [_transition(i % 3, reward=float(i % 3 == 2), done=i % 3 == 2,
                         bootstrap=i % 3 != 2) for i in range(9)]
    cfg = _tiny_cfg(env_steps=6, warmup_env_steps=4, critic_burnin_updates=1)
    result = residual_sac_train(
        cfg, _FakeBase(), _FakeTaskEnv(), _extract_state, demos,
        device="cpu", evaluate_gate=False, seed=0)
    assert result.env_steps == 6
    assert result.gate_passed is None  # gate skipped on request


def test_residual_sac_train_requires_warmup_or_demos():
    with pytest.raises(ValueError, match="warmup"):
        residual_sac_train(
            _tiny_cfg(), _FakeBase(), _FakeTaskEnv(), _extract_state,
            device="cpu", warmup_env_steps=0)


def test_acting_policy_resets_base_and_composition():
    base = _FakeBase()
    composed = ComposedPolicy(base.predict_chunk, zero_residual,
                              n_action_steps=3, alpha=0.5, exploration_ramp_steps=0)
    acting = ActingPolicy(base, composed)
    acting.select_action(None, torch.zeros(STATE_DIM), explore=True, rng=random.Random(0))
    acting.reset()
    assert base.resets == 1 and composed._chunk is None


# --------------------------------------------------------------------------
# CVE-2026-25874: the new lerobot-touching modules stay transport-free
# --------------------------------------------------------------------------


def test_p21_modules_import_no_cve_surface_in_clean_interpreter():
    code = (
        "import sys; "
        "import leagents.rl.train, leagents.rl.sac_wrappers, leagents.rl.libero_driver; "
        "from leagents.rl.safety import forbidden_loaded; "
        "hits = forbidden_loaded(sys.modules); "
        "sys.stderr.write(repr(hits)); "
        "sys.exit(1 if hits else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"P2.1 modules imported CVE surface: {r.stderr}"
