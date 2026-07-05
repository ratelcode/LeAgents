"""Residual SAC actor/critic over lerobot's transport-free SAC modules (design §1.2, §1.4).

Builds, from ``lerobot.policies.sac.modeling_sac`` (verified CVE-safe — imports
zero ``lerobot.transport``/``lerobot.rl.{actor,learner,*}`` modules):

- a **residual actor**: lerobot's ``Policy`` (tanh-squashed Gaussian) over the
  observation ``[proprio_state(8), a_base(7)]`` — encoder(15→256 latent, LayerNorm)
  + 2×256 MLP + mean/std heads = the 3-layer/256 residual MLP of PLD/ResFiT. The
  actor outputs the **raw residual** ``u ∈ (-1,1)^7``; scaling by α and summing
  with ``a_base`` is composition's job (``ComposedPolicy`` at act time, and
  ``ResidualSACPolicy._compose`` inside the losses).
- a **critic ensemble** over the **full executed action** ``Q(s, a_exec)``
  (LayerNorm CriticHeads, E critics, optional min-over-random-subset targets).

``ResidualSACPolicy`` subclasses ``SACPolicy`` overriding exactly the two spots
where the parent feeds the actor's own sample into the critic — for a residual,
the critic must see the *composed* action ``clamp(a_base + α·u, -1, 1)``:
- ``compute_loss_critic``: the TD target's next-action is composed with the
  next state's ``a_base`` (stored per transition, design §1.4);
- ``compute_loss_actor``: Q is evaluated at the composed action.
The temperature loss and everything else reuse the parent unchanged.

The 15-d observation vector layout is ``[state(8), a_base(7)]`` — ``a_base`` is
always the LAST ``action_dim`` entries, which is how ``_compose`` recovers it.
"""

from __future__ import annotations

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from leagents.rl.safety import assert_cve_safe


def residual_observation(state: torch.Tensor, a_base: torch.Tensor) -> dict[str, torch.Tensor]:
    """The SAC observation dict for a (batched or single) ``(state, a_base)``
    pair: ``{"observation.state": [state, a_base]}`` — ``a_base`` LAST."""
    return {OBS_STATE: torch.cat(
        [torch.atleast_2d(state.float()), torch.atleast_2d(a_base.float())], dim=-1)}


class ResidualSACPolicy(SACPolicy):
    """lerobot ``SACPolicy`` whose actor is a *residual*: the critic (and the
    TD target) evaluate the composed executed action, not the raw actor sample.

    The replay buffer's ``action`` field must hold the EXECUTED action
    ``a_exec`` (already composed + clipped at collection time), so the parent's
    ``q_preds`` path needs no change."""

    def __init__(self, config: SACConfig, residual_alpha: float):
        super().__init__(config)
        if not 0.0 < residual_alpha <= 1.0:
            raise ValueError(f"residual_alpha={residual_alpha} must be in (0, 1]")
        self.residual_alpha = residual_alpha
        self._action_dim = config.output_features[ACTION].shape[0]

    def base_action_from_obs(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Recover ``a_base`` from the observation vector (the last action_dim dims)."""
        return observations[OBS_STATE][..., -self._action_dim:]

    def _compose(self, observations: dict[str, torch.Tensor], residual: torch.Tensor) -> torch.Tensor:
        """``a_exec = clamp(a_base + α·u, -1, 1)`` — must match ComposedPolicy
        exactly (same α, same clip), with u the tanh-squashed actor output."""
        a_base = self.base_action_from_obs(observations)
        return torch.clamp(a_base + self.residual_alpha * residual, -1.0, 1.0)

    # -- the two overridden loss computations -------------------------------
    # Copied from lerobot 0.5.1 modeling_sac.SACPolicy and modified ONLY where
    # the actor's sample enters the critic (composition). Discrete-critic
    # branches dropped: the residual is continuous over all 7 dims (design §2.1).

    def compute_loss_critic(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        done,
        observation_features: torch.Tensor | None = None,
        next_observation_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import einops
        import torch.nn.functional as F  # noqa: N812

        with torch.no_grad():
            next_residual, next_log_probs, _ = self.actor(next_observations, next_observation_features)
            # RESIDUAL: the target policy's executed action composes the
            # residual with the NEXT state's base action (stored in next_obs).
            next_action_exec = self._compose(next_observations, next_residual)

            q_targets = self.critic_forward(
                observations=next_observations,
                actions=next_action_exec,
                use_target=True,
                observation_features=next_observation_features,
            )

            # min-over-random-subset (REDQ/RLPD) if configured
            if self.config.num_subsample_critics is not None:
                indices = torch.randperm(self.config.num_critics)
                indices = indices[: self.config.num_subsample_critics]
                q_targets = q_targets[indices]

            min_q, _ = q_targets.min(dim=0)
            if self.config.use_backup_entropy:
                min_q = min_q - (self.temperature * next_log_probs)

            # ``done`` here is "true terminal" (success) — time-limit truncated
            # transitions are stored with done=0 so they DO bootstrap (§2.2).
            td_target = rewards + (1 - done) * self.config.discount * min_q

        # actions from the buffer are the executed a_exec — no composition needed
        q_preds = self.critic_forward(
            observations=observations,
            actions=actions,
            use_target=False,
            observation_features=observation_features,
        )
        td_target_duplicate = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])
        critics_loss = (
            F.mse_loss(input=q_preds, target=td_target_duplicate, reduction="none").mean(dim=1)
        ).sum()
        return critics_loss

    def compute_loss_actor(
        self,
        observations,
        observation_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual_pi, log_probs, _ = self.actor(observations, observation_features)
        # RESIDUAL: Q is evaluated at the composed executed action.
        actions_pi = self._compose(observations, residual_pi)

        q_preds = self.critic_forward(
            observations=observations,
            actions=actions_pi,
            use_target=False,
            observation_features=observation_features,
        )
        min_q_preds = q_preds.min(dim=0)[0]
        actor_loss = ((self.temperature * log_probs) - min_q_preds).mean()
        return actor_loss


def build_residual_sac(
    *,
    state_dim: int = 8,
    action_dim: int = 7,
    device: str = "cpu",
    residual_alpha: float = 0.5,
    num_critics: int = 2,
    num_subsample_critics: int | None = None,
    discount: float = 0.99,
    residual_init_gain: float = 0.01,
    init_log_std: float = -1.0,
    grad_clip_norm: float = 40.0,
) -> ResidualSACPolicy:
    """Build the residual SAC policy: actor over ``[state, a_base]`` (state_dim
    + action_dim inputs), critic ensemble over ``(same obs, a_exec)``.

    Small-gain init (Silver'18 / ResiP): the actor's mean head is scaled by
    ``residual_init_gain`` so the deterministic residual is ≈0 at init (composed
    == base), and the std head is pinned near ``exp(init_log_std)`` so early
    stochastic residuals are moderate noise around the base action.
    """
    assert_cve_safe()
    config = SACConfig(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim + action_dim,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        device=device,
        storage_device="cpu",
        num_critics=num_critics,
        num_subsample_critics=num_subsample_critics,
        discount=discount,
        grad_clip_norm=grad_clip_norm,
        num_discrete_actions=None,  # continuous residual over all 7 dims (v1, §2.1)
        use_torch_compile=False,  # short runs; compile warmup would dominate
        dataset_stats=None,  # SACPolicy applies no normalization internally (0.5.1)
    )
    policy = ResidualSACPolicy(config, residual_alpha=residual_alpha)
    with torch.no_grad():
        policy.actor.mean_layer.weight.mul_(residual_init_gain)
        policy.actor.mean_layer.bias.mul_(residual_init_gain)
        policy.actor.std_layer.weight.mul_(residual_init_gain)
        policy.actor.std_layer.bias.fill_(init_log_std)
    policy.to(device)
    return policy


def make_residual_forward(policy: ResidualSACPolicy, *, deterministic: bool = False):
    """A ``ComposedPolicy``-compatible ``residual_forward(state, a_base)`` that
    returns the actor's PRE-tanh output (7,) on CPU — ``ComposedPolicy`` applies
    the same ``α·tanh`` the losses use, so acting and training compose identically.

    ``deterministic=True`` returns the pre-tanh mean (eval / inner gate);
    otherwise a pre-tanh sample ``mean + std·ε`` — exactly the actor's
    ``TanhMultivariateNormalDiag.rsample()`` before the tanh."""
    actor = policy.actor

    @torch.no_grad()
    def residual_forward(state, a_base) -> torch.Tensor:
        device = next(actor.parameters()).device
        obs = residual_observation(
            torch.as_tensor(state, dtype=torch.float32),
            torch.as_tensor(a_base, dtype=torch.float32),
        )
        obs = {k: v.to(device) for k, v in obs.items()}
        obs_enc = actor.encoder(obs)
        h = actor.network(obs_enc)
        mean = actor.mean_layer(h)
        if deterministic:
            pre_tanh = mean
        else:
            std = torch.exp(actor.std_layer(h))
            std = torch.clamp(std, actor.std_min, actor.std_max)
            pre_tanh = mean + std * torch.randn_like(std)
        return pre_tanh.squeeze(0).cpu()

    return residual_forward


def zero_residual(state, a_base) -> torch.Tensor:
    """Residual that is exactly 0 — composes to the pure base policy (used to
    measure the base side of the inner gate at the same chunk cadence)."""
    del state
    return torch.zeros_like(torch.as_tensor(a_base, dtype=torch.float32))
