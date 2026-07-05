"""ComposedPolicy: frozen base + per-step residual over a cached action chunk.

This is the correctness-critical, genuinely-new piece the design calls out
(§1.3). It is deliberately free of lerobot/SmolVLA/GPU dependencies: the base
and residual are injected as callables, so the composition math — the part
that must be exactly right — is unit-testable on CPU with fakes.

    a_t = a_base_{i} + alpha * tanh(residual(state, a_base_{i}))     (clipped)

with two behaviours the literature is emphatic about (design §1.3):
- CHUNK CACHING: the base (a ~450M forward) is called once per ``n_action_steps``
  and its chunk cached; the cheap residual corrects every step. This is the
  difference between a feasible and an infeasible run (A2C2: base ~101ms vs
  residual ~4.7ms).
- PROGRESSIVE EXPLORATION: during training, execute base-only with probability
  (1 - epsilon), ramping epsilon 0 -> 1 linearly; an aggressive early residual
  is a documented cause of total failure (Policy Decorator).
"""

from __future__ import annotations

import random
from typing import Callable

import torch

# obs -> action chunk (T, action_dim); called once per n_action_steps and cached
BasePredictChunk = Callable[[object], torch.Tensor]
# (state, base_action) -> unbounded residual (action_dim,); the cheap MLP
ResidualForward = Callable[[object, torch.Tensor], torch.Tensor]


class ComposedPolicy:
    def __init__(
        self,
        base_predict_chunk: BasePredictChunk,
        residual_forward: ResidualForward,
        *,
        n_action_steps: int,
        alpha: float,
        exploration_ramp_steps: int,
        action_low: float = -1.0,
        action_high: float = 1.0,
    ):
        if n_action_steps < 1:
            raise ValueError("n_action_steps must be >= 1")
        self._base_predict_chunk = base_predict_chunk
        self._residual_forward = residual_forward
        self.n_action_steps = n_action_steps
        self.alpha = alpha
        self.exploration_ramp_steps = exploration_ramp_steps
        self.action_low = action_low
        self.action_high = action_high
        self.reset()

    def reset(self) -> None:
        """Clear the chunk cache. MUST be called on env reset (the base's own
        chunk queue is keyed to episode boundaries)."""
        self._chunk: torch.Tensor | None = None
        self._i = 0  # index into the current chunk

    def epsilon(self, global_step: int) -> float:
        """Linear base-only -> composed ramp over ``exploration_ramp_steps``.
        epsilon(0)=0 (execute base), epsilon(>=ramp)=1 (execute composed)."""
        if self.exploration_ramp_steps <= 0:
            return 1.0
        return min(1.0, max(0.0, global_step / self.exploration_ramp_steps))

    def base_action(self, obs: object) -> torch.Tensor:
        """The current step's base action, refilling (and caching) the chunk
        only when the previous one is exhausted — one base forward per
        ``n_action_steps`` steps."""
        if self._chunk is None or self._i >= self.n_action_steps or self._i >= self._chunk.shape[0]:
            self._chunk = self._base_predict_chunk(obs)
            self._i = 0
        a_base = self._chunk[self._i]
        self._i += 1
        return a_base

    def select_action(
        self,
        obs: object,
        state: object,
        *,
        global_step: int = 0,
        explore: bool = False,
        rng: random.Random | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(a_exec, a_base)``. ``a_base`` is returned so the SAC critic
        and actor loss can condition on it for every stored transition (the
        buffer stores it — design §1.4). With ``explore=False`` (harvest/eval)
        the full composed action is always executed; with ``explore=True`` the
        progressive-exploration gate applies."""
        a_base = self.base_action(obs)
        delta = self.alpha * torch.tanh(self._residual_forward(state, a_base))
        a_composed = torch.clamp(a_base + delta, self.action_low, self.action_high)
        if not explore:
            return a_composed, a_base
        eps = self.epsilon(global_step)
        draw = (rng or random).random()
        a_exec = a_composed if draw < eps else a_base
        return a_exec, a_base
