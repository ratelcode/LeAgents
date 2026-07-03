"""Improvement Agent — DexFlyWheel cycle placeholder (M1, DESIGN.md §3.5).

The M1 implementation runs: residual RL on the frozen base policy →
success-filtered rollout collection → augmentation → expanded dataset
for the next cycle's base policy.
"""

from __future__ import annotations

from leagent.events import EventBus


class ImproveAgent:
    def __init__(self, bus: EventBus):
        self.bus = bus

    def run(self, **kwargs: object) -> None:
        raise NotImplementedError(
            "DexFlyWheel improvement cycle lands in M1 (see DESIGN.md §3.5, §8)"
        )
