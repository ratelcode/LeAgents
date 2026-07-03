"""Data Agent — M0 scope (DESIGN.md §3.2, §8).

M0 collects no new data: the agent resolves the seed dataset from the
proposal and records provenance for the cycle. RoboGene-style curation,
MimicGen-style amplification, and GenAug-style augmentation land in M1.
"""

from __future__ import annotations

from leagent.contracts import DatasetRef, Proposal
from leagent.events import Event, EventBus


class DataAgent:
    def __init__(self, bus: EventBus):
        self.bus = bus

    def run(self, *, run_id: str, cycle: int, proposal: Proposal) -> DatasetRef:
        ref = DatasetRef(repo_id=proposal.dataset, notes=proposal.notes)
        self.bus.emit(
            Event(
                run_id=run_id,
                stage="collect",
                kind="dataset_resolved",
                cycle=cycle,
                payload={"action": proposal.action, "repo_id": ref.repo_id, "notes": ref.notes},
            )
        )
        return ref
