"""Single-process residual SAC training (design §2.1, §4; P2.1 core).

``residual_sac_train`` runs the RLPD recipe in ONE process, reusing lerobot's
transport-free pieces (``ReplayBuffer``, ``concatenate_batch_transitions``, the
SAC modules via ``sac_wrappers``) and this package's P2.0 scaffolding
(``ComposedPolicy``, ``ResidualRollout``, ``classify_step`` semantics).

CVE-2026-25874 (design §3.1): the SAC update lives in ``lerobot/rl/learner.py``,
which imports the gRPC/pickle transport, so the ~150-line update math
(learner.py:393-554 + make_optimizers_and_scheduler:761-810 in lerobot 0.5.1)
is COPIED here — ``sac_update_step`` / ``make_sac_optimizers`` /
``check_nan_in_transition`` — and this module must never import
``lerobot.rl.learner``/``actor``/``learner_service``/``lerobot.transport``.
``assert_cve_safe()`` runs at the start of training and periodically in-loop.

Loop shape (design §2.1):
  warmup   — fill the buffer executing ``a_base + α·tanh(noise)`` (a stochastic
             untrained residual with the exploration ramp disabled ≡ ResFiT's
             base+noise warmup), then critic burn-in with the actor held ≈0
             (small-gain init; actor updates skipped).
  main     — per iteration: collect on ``train_init_states`` via
             ResidualRollout (ε-ramp active), then UTD gradient steps per env
             step, 50/50 online/demo sampling (RLPD).
  gate     — deterministic composed vs pure base success on the DISJOINT
             ``harvest_init_states`` (design §5.1, ``min_composed_gain``).
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from lerobot.rl.buffer import ReplayBuffer, concatenate_batch_transitions
from lerobot.utils.constants import ACTION, OBS_STATE

from leagents.rl.composition import ComposedPolicy
from leagents.rl.config import ResidualRLConfig, init_state_indices
from leagents.rl.env_adapter import ResidualRollout, Transition, classify_step
from leagents.rl.safety import assert_cve_safe
from leagents.rl.sac_wrappers import (
    ResidualSACPolicy,
    build_residual_sac,
    make_residual_forward,
    zero_residual,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# helpers copied out of lerobot 0.5.1 lerobot/rl/learner.py (CVE §3.1 — that
# module imports the transport and must not be imported)
# --------------------------------------------------------------------------


def check_nan_in_transition(
    observations: dict[str, torch.Tensor],
    actions: torch.Tensor,
    next_state: dict[str, torch.Tensor],
    raise_error: bool = False,
) -> bool:
    """Copied from lerobot/rl/learner.py:1048 (transport-free math)."""
    nan_detected = False
    for key, tensor in observations.items():
        if torch.isnan(tensor).any():
            logger.error(f"observations[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in observations[{key}]")
    for key, tensor in next_state.items():
        if torch.isnan(tensor).any():
            logger.error(f"next_state[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in next_state[{key}]")
    if torch.isnan(actions).any():
        logger.error("actions contain NaN values")
        nan_detected = True
        if raise_error:
            raise ValueError("NaN detected in actions")
    return nan_detected


def make_sac_optimizers(
    policy: ResidualSACPolicy,
    *,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    temperature_lr: float = 3e-4,
) -> dict[str, torch.optim.Optimizer]:
    """Copied from lerobot/rl/learner.py make_optimizers_and_scheduler (:761),
    minus the discrete-critic branch (continuous residual, design §2.1)."""
    optimizer_actor = torch.optim.Adam(
        params=[
            p
            for n, p in policy.actor.named_parameters()
            if not policy.config.shared_encoder or not n.startswith("encoder")
        ],
        lr=actor_lr,
    )
    optimizer_critic = torch.optim.Adam(params=policy.critic_ensemble.parameters(), lr=critic_lr)
    optimizer_temperature = torch.optim.Adam(params=[policy.log_alpha], lr=temperature_lr)
    return {"actor": optimizer_actor, "critic": optimizer_critic, "temperature": optimizer_temperature}


def sac_update_step(
    policy: ResidualSACPolicy,
    optimizers: dict[str, torch.optim.Optimizer],
    batch: dict,
    *,
    update_actor: bool,
) -> dict[str, float]:
    """One SAC optimization step — the update math copied out of lerobot 0.5.1
    lerobot/rl/learner.py:393-554 (single-process: no queues/iterators/transport;
    no discrete critic; observation features None — v1 residual has no vision
    encoder, which is exactly when lerobot's get_observation_features returns None).
    """
    clip = policy.config.grad_clip_norm
    actions = batch[ACTION]
    rewards = batch["reward"]
    observations = batch["state"]
    next_observations = batch["next_state"]
    done = batch["done"]
    check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)

    forward_batch = {
        ACTION: actions,
        "reward": rewards,
        "state": observations,
        "next_state": next_observations,
        "done": done,
        "observation_feature": None,
        "next_observation_feature": None,
    }

    critic_output = policy.forward(forward_batch, model="critic")
    loss_critic = critic_output["loss_critic"]
    optimizers["critic"].zero_grad()
    loss_critic.backward()
    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
        parameters=policy.critic_ensemble.parameters(), max_norm=clip
    ).item()
    optimizers["critic"].step()

    info = {"loss_critic": loss_critic.item(), "critic_grad_norm": critic_grad_norm}

    if update_actor:
        actor_output = policy.forward(forward_batch, model="actor")
        loss_actor = actor_output["loss_actor"]
        optimizers["actor"].zero_grad()
        loss_actor.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=policy.actor.parameters(), max_norm=clip
        ).item()
        optimizers["actor"].step()
        info["loss_actor"] = loss_actor.item()
        info["actor_grad_norm"] = actor_grad_norm

        temperature_output = policy.forward(forward_batch, model="temperature")
        loss_temperature = temperature_output["loss_temperature"]
        optimizers["temperature"].zero_grad()
        loss_temperature.backward()
        info["temperature_grad_norm"] = torch.nn.utils.clip_grad_norm_(
            parameters=[policy.log_alpha], max_norm=clip
        ).item()
        optimizers["temperature"].step()
        info["loss_temperature"] = loss_temperature.item()
        info["temperature"] = policy.temperature

    policy.update_target_networks()
    return info


# --------------------------------------------------------------------------
# buffer bookkeeping (design §1.4): every stored transition carries a_base,
# and next_state carries the NEXT step's a_base so the TD target can compose.
# --------------------------------------------------------------------------


def attach_next_a_base(transitions: Sequence[Transition]) -> list[tuple[Transition, torch.Tensor]]:
    """Pair each transition with ``a_base(s')`` for the TD target's composed
    next-action. Must be called per ``ResidualRollout.collect`` batch (episode
    order intact). Pure function.

    - mid-episode: the NEXT transition's ``a_base`` (computed online, free);
    - success terminal (``bootstrap=False``): zeros — the (1 - done) mask kills
      the target, the value is never read;
    - time-limit truncation / collection cut (``done or last``, bootstrap=True):
      the CURRENT ``a_base`` — a chunk-smoothness approximation affecting at
      most one transition per episode (documented; adjacent chunk actions are
      near-identical, and truncations only occur on failed episodes).
    """
    out: list[tuple[Transition, torch.Tensor]] = []
    for i, t in enumerate(transitions):
        a_base = torch.as_tensor(t.a_base, dtype=torch.float32)
        if not t.bootstrap:
            next_a_base = torch.zeros_like(a_base)
        elif not t.done and i + 1 < len(transitions):
            next_a_base = torch.as_tensor(transitions[i + 1].a_base, dtype=torch.float32)
        else:
            next_a_base = a_base
        out.append((t, next_a_base))
    return out


def add_transitions(buffer: ReplayBuffer, transitions: Sequence[Transition]) -> None:
    """Store transitions with the 15-d ``[state, a_base]`` observation vector.

    Buffer ``done`` is "true terminal" = ``not bootstrap`` (success), so
    lerobot's ``(1 - done)`` TD mask bootstraps through our time-limit
    truncations and never through success — the classify_step semantics (§2.2).
    """
    for t, next_a_base in attach_next_a_base(transitions):
        state = torch.as_tensor(t.state, dtype=torch.float32)
        a_base = torch.as_tensor(t.a_base, dtype=torch.float32)
        next_state = torch.as_tensor(t.next_state, dtype=torch.float32)
        buffer.add(
            state={OBS_STATE: torch.cat([state, a_base])},
            action=torch.as_tensor(t.a_exec, dtype=torch.float32),
            reward=float(t.reward),
            next_state={OBS_STATE: torch.cat([next_state, next_a_base])},
            done=not t.bootstrap,
            truncated=bool(t.done and t.bootstrap),
        )


def sample_mixed(online: ReplayBuffer, demo: ReplayBuffer | None, batch_size: int) -> dict:
    """RLPD symmetric sampling: half online, half demo (design §2.1.1); all
    online when no demo buffer is available."""
    if demo is None or len(demo) == 0:
        return online.sample(batch_size)
    half = batch_size // 2
    batch = online.sample(half)
    return concatenate_batch_transitions(batch, demo.sample(batch_size - half))


# --------------------------------------------------------------------------
# acting policy + evaluation
# --------------------------------------------------------------------------


class ActingPolicy:
    """ResidualRollout's policy protocol over (frozen base, ComposedPolicy):
    reset() must clear BOTH the base's internal queue and the chunk cache."""

    def __init__(self, base, composed: ComposedPolicy):
        self.base = base
        self.composed = composed

    def reset(self) -> None:
        self.base.reset()
        self.composed.reset()

    def select_action(self, obs, state, *, global_step: int = 0, explore: bool = False, rng=None):
        return self.composed.select_action(
            obs, state, global_step=global_step, explore=explore, rng=rng)


def evaluate_success(
    env,
    policy: ActingPolicy,
    extract_state: Callable,
    init_states: Sequence[int],
    max_steps: int,
) -> float:
    """Deterministic success rate over the given init states (one episode
    each), using the same classify_step semantics as training."""
    successes = 0
    for idx in init_states:
        obs = env.reset(idx)
        policy.reset()
        for step in range(max_steps):
            state = extract_state(obs)
            a_exec, _ = policy.select_action(obs, state, explore=False)
            obs, reward, info = env.step(a_exec)
            outcome = classify_step(reward, info, step, max_steps)
            if outcome.done:
                successes += int(outcome.success)
                break
    return successes / len(init_states)


# --------------------------------------------------------------------------
# the training entrypoint
# --------------------------------------------------------------------------


@dataclass
class ResidualTrainResult:
    policy: ResidualSACPolicy
    residual_state_dict: dict
    env_steps: int
    updates: int
    wall_clock_s: float
    metrics: dict = field(default_factory=dict)
    base_success: float | None = None
    composed_success: float | None = None
    composed_gain: float | None = None
    gate_passed: bool | None = None


def residual_sac_train(
    cfg: ResidualRLConfig,
    base,
    env,
    extract_state: Callable,
    demo_transitions: Sequence[Transition] | None = None,
    *,
    device: str = "cpu",
    env_steps: int | None = None,
    warmup_env_steps: int | None = None,
    critic_burnin_updates: int | None = None,
    steps_per_iter: int | None = None,
    evaluate_gate: bool = True,
    seed: int = 0,
    log_every_iters: int = 1,
) -> ResidualTrainResult:
    """Train a residual over the frozen ``base`` on ``env`` (design §2.1, §4.1).

    ``base``: chunk predictor (``predict_chunk(obs) -> (chunk, action_dim)``, ``reset()``).
    ``env``: ResidualRollout's env protocol + ``max_steps`` (e.g. LiberoResidualEnv).
    ``demo_transitions``: seed-task demos with precomputed ``a_base``
    (``libero_driver.precompute_demo_a_base``) for the RLPD 50/50 anchor.

    Returns the trained residual and the inner-gate metric: deterministic
    composed vs pure base success on the held-out ``harvest_init_states``.
    """
    assert_cve_safe()
    t_start = time.monotonic()
    torch.manual_seed(seed)
    rng = random.Random(seed)

    env_steps = cfg.env_steps if env_steps is None else env_steps
    warmup_env_steps = cfg.warmup_env_steps if warmup_env_steps is None else warmup_env_steps
    critic_burnin_updates = (
        cfg.critic_burnin_updates if critic_burnin_updates is None else critic_burnin_updates)
    steps_per_iter = cfg.steps_per_iter if steps_per_iter is None else steps_per_iter
    if warmup_env_steps <= 0 and not demo_transitions:
        raise ValueError("warmup_env_steps must be > 0 when no demo transitions are given "
                         "(the first update would sample an empty buffer)")

    train_states = init_state_indices(cfg.train_init_states)
    harvest_states = init_state_indices(cfg.harvest_init_states)
    max_steps = int(env.max_steps)

    policy = build_residual_sac(
        device=device,
        residual_alpha=cfg.alpha,
        num_critics=cfg.num_critics,
        num_subsample_critics=cfg.num_subsample_critics,
        discount=cfg.discount,
        residual_init_gain=cfg.residual_init_gain,
        init_log_std=cfg.init_log_std,
    )
    optimizers = make_sac_optimizers(
        policy, actor_lr=cfg.actor_lr, critic_lr=cfg.critic_lr, temperature_lr=cfg.temperature_lr)

    rollout = ResidualRollout(env, extract_state, max_steps=max_steps)
    online = ReplayBuffer(
        capacity=cfg.buffer_capacity, device=device, state_keys=[OBS_STATE],
        storage_device="cpu", use_drq=False)
    demo: ReplayBuffer | None = None
    if demo_transitions:
        demo = ReplayBuffer(
            capacity=max(len(demo_transitions), 1), device=device, state_keys=[OBS_STATE],
            storage_device="cpu", use_drq=False)
        add_transitions(demo, demo_transitions)

    stochastic_forward = make_residual_forward(policy, deterministic=False)
    metrics: dict = {"loss_critic": [], "loss_actor": [], "loss_temperature": []}

    def record(info: dict) -> None:
        for k in metrics:
            if k in info:
                metrics[k].append(info[k])
        for k in ("loss_critic", "loss_actor", "loss_temperature"):
            if k in info and not torch.isfinite(torch.tensor(info[k])):
                raise RuntimeError(f"non-finite {k}={info[k]} — aborting residual training")

    # -- warmup: fill the buffer with a_base + α·tanh(noise) (ramp disabled ⇒
    # the composed action is always executed; the untrained small-init actor
    # IS base+noise), design §2.1.5.
    if warmup_env_steps > 0:
        warm = ActingPolicy(base, ComposedPolicy(
            base.predict_chunk, stochastic_forward,
            n_action_steps=cfg.n_action_steps, alpha=cfg.alpha, exploration_ramp_steps=0))
        add_transitions(online, rollout.collect(warm, warmup_env_steps, train_states, rng=rng))
        logger.info("warmup: %d env steps collected (buffer=%d)", warmup_env_steps, len(online))

    # -- critic burn-in: actor held at its ≈0 init (no actor updates).
    for _ in range(critic_burnin_updates):
        record(sac_update_step(
            policy, optimizers, sample_mixed(online, demo, cfg.batch_size), update_actor=False))
    if critic_burnin_updates:
        logger.info("critic burn-in: %d updates, loss_critic=%.4f",
                    critic_burnin_updates, metrics["loss_critic"][-1])

    # -- main RLPD loop: collect on train init states with the ε-ramp, then
    # UTD updates per env step (lerobot's utd loop shape: utd-1 critic-only
    # updates + 1 update that may also step the actor/temperature).
    acting = ActingPolicy(base, ComposedPolicy(
        base.predict_chunk, stochastic_forward,
        n_action_steps=cfg.n_action_steps, alpha=cfg.alpha,
        exploration_ramp_steps=cfg.exploration_ramp_steps))

    global_step = 0
    optimization_step = 0
    iteration = 0
    while global_step < env_steps:
        n = min(steps_per_iter, env_steps - global_step)
        transitions = rollout.collect(
            acting, n, train_states, global_step_start=global_step, rng=rng)
        add_transitions(online, transitions)
        global_step += n

        for _ in range(n):
            for _ in range(cfg.utd - 1):
                record(sac_update_step(
                    policy, optimizers, sample_mixed(online, demo, cfg.batch_size),
                    update_actor=False))
            record(sac_update_step(
                policy, optimizers, sample_mixed(online, demo, cfg.batch_size),
                update_actor=optimization_step % cfg.policy_update_freq == 0))
            optimization_step += 1

        assert_cve_safe()  # cheap; the loop must stay transport-free while running
        iteration += 1
        if iteration % log_every_iters == 0:
            successes = sum(t.reward > 0 for t in transitions)
            logger.info(
                "iter %d: env_steps=%d/%d eps=%.2f successes_in_batch=%d "
                "loss_critic=%.4f loss_actor=%s buffer=%d",
                iteration, global_step, env_steps, acting.composed.epsilon(global_step),
                successes, metrics["loss_critic"][-1],
                f"{metrics['loss_actor'][-1]:.4f}" if metrics["loss_actor"] else "n/a",
                len(online))

    # -- inner gate (design §5.1): deterministic composed vs pure base on the
    # DISJOINT held-out harvest init states, at the same chunk cadence.
    base_sr = composed_sr = gain = None
    gate = None
    if evaluate_gate:
        det = ActingPolicy(base, ComposedPolicy(
            base.predict_chunk, make_residual_forward(policy, deterministic=True),
            n_action_steps=cfg.n_action_steps, alpha=cfg.alpha, exploration_ramp_steps=0))
        pure_base = ActingPolicy(base, ComposedPolicy(
            base.predict_chunk, zero_residual,
            n_action_steps=cfg.n_action_steps, alpha=cfg.alpha, exploration_ramp_steps=0))
        composed_sr = evaluate_success(env, det, extract_state, harvest_states, max_steps)
        base_sr = evaluate_success(env, pure_base, extract_state, harvest_states, max_steps)
        gain = composed_sr - base_sr
        gate = gain >= cfg.min_composed_gain
        logger.info("inner gate: composed=%.3f base=%.3f gain=%.3f (min %.3f) -> %s",
                    composed_sr, base_sr, gain, cfg.min_composed_gain,
                    "PASS" if gate else "SKIP HARVEST")

    return ResidualTrainResult(
        policy=policy,
        residual_state_dict={k: v.detach().cpu() for k, v in policy.actor.state_dict().items()},
        env_steps=global_step,
        updates=optimization_step * cfg.utd + critic_burnin_updates,
        wall_clock_s=time.monotonic() - t_start,
        metrics=metrics,
        base_success=base_sr,
        composed_success=composed_sr,
        composed_gain=gain,
        gate_passed=gate,
    )
