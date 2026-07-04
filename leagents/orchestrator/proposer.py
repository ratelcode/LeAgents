"""Task proposal (DESIGN.md §3.1).

M0 ships a deterministic proposer. LLMProposer (M1) reads knowledge
pages (§3.6) and asks the configured provider-agnostic LLM, falling back
to the deterministic proposer whenever the LLM is absent or unusable —
so the loop runs identically with or without a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from leagents.contracts import EvalReport, Proposal
from leagents.llm import LLMClient


class Proposer(Protocol):
    def propose(self, cycle: int, last_report: EvalReport | None) -> Proposal: ...


class DeterministicProposer:
    def __init__(
        self,
        seed_dataset: str,
        initial_episodes: int | None = None,
        growth: float = 2.0,
        max_episodes: int | None = None,
        failing_threshold: float = 0.5,
    ):
        self.seed_dataset = seed_dataset
        self.initial_episodes = initial_episodes
        self.growth = growth
        self.max_episodes = max_episodes
        self.failing_threshold = failing_threshold

    def _episodes_for(self, cycle: int) -> int | None:
        """Progressive data schedule: each cycle reveals more of the seed data."""
        if self.initial_episodes is None:
            return None
        n = int(self.initial_episodes * self.growth**cycle)
        return min(n, self.max_episodes) if self.max_episodes else n

    def propose(self, cycle: int, last_report: EvalReport | None) -> Proposal:
        num_episodes = self._episodes_for(cycle)
        if last_report is None or not last_report.per_task:
            return Proposal(
                action="reuse_seed", dataset=self.seed_dataset, num_episodes=num_episodes
            )
        failing = sorted(
            task
            for task, score in last_report.per_task.items()
            if score < self.failing_threshold
        )
        return Proposal(
            action="recollect_failing",
            dataset=self.seed_dataset,
            num_episodes=num_episodes,
            notes=f"focus tasks: {failing[:5]}" if failing else "no failing tasks",
        )


_PROPOSER_SYSTEM = (
    "You propose the next data-collection step for a robot-learning loop. Reply with ONLY a "
    'JSON object: {"action": "reuse_seed"|"recollect_failing", "dataset": "<hub repo id>", '
    '"num_episodes": <int or null for the full dataset>, "notes": "<short rationale>"}. '
    "Proposals are advisory; safety gates apply downstream."
)


class LLMProposer:
    """Knowledge-aware proposer (M1). Advisory only — constitution and loop
    gates still apply to whatever it proposes."""

    def __init__(
        self,
        llm: LLMClient,
        knowledge_root: Path,
        fallback: DeterministicProposer,
        max_pages: int = 6,
    ):
        self.llm = llm
        self.knowledge_root = Path(knowledge_root)
        self.fallback = fallback
        self.max_pages = max_pages

    def _knowledge_context(self) -> str:
        if not self.knowledge_root.exists():
            return ""
        pages = sorted(
            (p for p in self.knowledge_root.rglob("*.md") if p.name != "KNOWLEDGE.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[: self.max_pages]
        return "\n\n".join(f"<page path={p}>\n{p.read_text()}\n</page>" for p in pages)

    def propose(self, cycle: int, last_report: EvalReport | None) -> Proposal:
        prompt = (
            f"Cycle: {cycle}\n"
            f"Seed dataset: {self.fallback.seed_dataset}\n"
            f"Last eval report: {last_report.model_dump_json() if last_report else 'none'}\n\n"
            f"Knowledge pages (most recently updated first):\n{self._knowledge_context()}"
        )
        reply = self.llm.complete(prompt, system=_PROPOSER_SYSTEM)
        try:
            start, end = reply.index("{"), reply.rindex("}") + 1
            data = json.loads(reply[start:end])
            return Proposal.model_validate(data)
        except (ValueError, TypeError):  # no/empty/unparseable reply -> deterministic
            return self.fallback.propose(cycle, last_report)
