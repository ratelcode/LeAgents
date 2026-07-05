"""Config surface for residual RL (design §4.3). Off by default; extends
``ImproveConfig`` as ``improve.residual_rl``."""

from __future__ import annotations

from pydantic import BaseModel, model_validator


class ResidualRLConfig(BaseModel):
    """Residual-RL stage: on promotion, train a residual on the frozen base to
    reach success on stuck tasks, harvest the composed policy, then discard the
    residual (it is a data generator, never a promoted checkpoint)."""

    enabled: bool = False

    # 'auto' = tasks below success_floor in the last eval report; or explicit ids.
    target_tasks: str | list[int] = "auto"
    success_floor: float = 0.5  # for target_tasks='auto'

    # The 100 fixed LIBERO init states are split so the residual cannot win by
    # memorising eval states: it TRAINS on one range, and is measured + harvests
    # on the DISJOINT range (design §3.2, §5.1). [start, end) half-open.
    train_init_states: tuple[int, int] = (0, 60)
    harvest_init_states: tuple[int, int] = (60, 100)

    env_steps: int = 150_000  # off-policy budget per targeted task (PLD~250k, ResFiT~75-200k)
    alpha: float = 0.5  # residual scale; a_t = a_base + alpha*tanh(residual) (PLD LIBERO value)
    exploration_ramp_steps: int = 15_000  # linear base-only -> composed ramp
    n_action_steps: int = 5  # base re-plan cadence (chunk cache); reactivity/compute trade-off
    utd: int = 4  # update-to-data ratio
    use_images: bool = False  # v1: proprio + base_action; escalate to True if it stalls

    # Inner gate: only harvest a task if the composed policy beats the base by
    # this much on the held-out harvest_init_states (design §5.1).
    min_composed_gain: float = 0.05

    @model_validator(mode="after")
    def _check(self) -> "ResidualRLConfig":
        lo, hi = self.train_init_states
        hlo, hhi = self.harvest_init_states
        for name, (a, b) in (("train_init_states", (lo, hi)),
                             ("harvest_init_states", (hlo, hhi))):
            if not (0 <= a < b <= 100):
                raise ValueError(f"{name}={(a, b)} must be a [start, end) range within [0, 100]")
        # disjoint so measured improvement is generalisation, not leakage
        if lo < hhi and hlo < hi:
            raise ValueError(
                f"train_init_states {self.train_init_states} and harvest_init_states "
                f"{self.harvest_init_states} must be disjoint (design §5.1)")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha={self.alpha} must be in (0, 1]")
        if self.n_action_steps < 1:
            raise ValueError("n_action_steps must be >= 1")
        return self

    def resolve_target_tasks(self, per_task_success: dict[int, float]) -> list[int]:
        """The task ids the residual should target: an explicit list, or (for
        'auto') the tasks whose last eval success is below ``success_floor``,
        weakest first. Pure function — the loop stays deterministic."""
        if isinstance(self.target_tasks, list):
            return list(self.target_tasks)
        return [t for t, _ in sorted(per_task_success.items(), key=lambda kv: kv[1])
                if per_task_success[t] < self.success_floor]


# convenience for callers that keep init-state ranges as explicit index lists
def init_state_indices(span: tuple[int, int]) -> list[int]:
    return list(range(span[0], span[1]))
