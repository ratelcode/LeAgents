"""Promotion decision logic (DESIGN.md §2).

A pure function so the loop's core control decision is unit-testable
and never delegated to an LLM.
"""

from __future__ import annotations

from enum import Enum

from leloop.config import ThresholdConfig


class Decision(str, Enum):
    PROMOTE = "promote"
    ITERATE = "iterate"
    ESCALATE = "escalate"
    ROLLBACK = "rollback"


def decide(
    candidate: float,
    baseline: float | None,
    history: list[float],
    thresholds: ThresholdConfig,
) -> Decision:
    """Decide the loop's next move from eval success rates (fractions in [0, 1]).

    ``candidate`` is the current cycle's checkpoint score, ``baseline`` the
    blessed checkpoint's score (None on the first cycle — the first working
    checkpoint becomes the baseline by definition), ``history`` the scores of
    prior non-promoted candidates on the current policy rung, oldest first.
    """
    if baseline is None:
        return Decision.PROMOTE
    delta = candidate - baseline
    if delta >= thresholds.promote_delta:
        return Decision.PROMOTE
    if delta <= -thresholds.regression_delta:
        return Decision.ROLLBACK
    window = history + [candidate]
    if len(window) >= thresholds.plateau_cycles and all(
        abs(score - baseline) < thresholds.plateau_epsilon
        for score in window[-thresholds.plateau_cycles :]
    ):
        # A plateau below the floor means the policy never got off the
        # ground — that is under-training, not a policy ceiling.
        if baseline < thresholds.escalate_floor:
            return Decision.ITERATE
        return Decision.ESCALATE
    return Decision.ITERATE
