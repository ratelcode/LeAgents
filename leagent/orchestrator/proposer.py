"""Task proposal (DESIGN.md §3.1).

M0 uses a deterministic proposer; the LLM-backed proposer (Claude Agent
SDK) lands with M1 task curation. The interface stays identical so the
loop does not change when intelligence is swapped in.
"""

from __future__ import annotations

from typing import Protocol

from leagent.contracts import EvalReport, Proposal


class Proposer(Protocol):
    def propose(self, cycle: int, last_report: EvalReport | None) -> Proposal: ...


class DeterministicProposer:
    def __init__(self, seed_dataset: str, failing_threshold: float = 0.5):
        self.seed_dataset = seed_dataset
        self.failing_threshold = failing_threshold

    def propose(self, cycle: int, last_report: EvalReport | None) -> Proposal:
        if last_report is None or not last_report.per_task:
            return Proposal(action="reuse_seed", dataset=self.seed_dataset)
        failing = sorted(
            task
            for task, score in last_report.per_task.items()
            if score < self.failing_threshold
        )
        return Proposal(
            action="recollect_failing",
            dataset=self.seed_dataset,
            notes=f"focus tasks: {failing[:5]}" if failing else "no failing tasks",
        )
