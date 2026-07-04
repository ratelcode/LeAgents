"""RoboGene-style curation pipeline (M1, DESIGN.md §3.2).

Three LLM stages — propose K candidate data plans, score them, refine the
winner — with deterministic guards between stages (episode caps, dataset
pinning), falling back to the deterministic proposer whenever any stage
yields nothing usable. Curation is advisory only: the constitution and the
loop's pure decision function still apply downstream.
"""

from __future__ import annotations

import json
from pathlib import Path

from leagents.config import DataConfig
from leagents.contracts import EvalReport, Proposal
from leagents.events import Event, EventBus
from leagents.llm import LLMClient
from leagents.orchestrator.proposer import DeterministicProposer

_ACTIONS = {"reuse_seed", "recollect_failing"}

_PROPOSE_SYSTEM = (
    "You curate data collection for a robot imitation-learning loop. Given eval results, "
    "knowledge-wiki excerpts, and the data schedule, output ONLY a JSON array of {k} "
    'candidate plans: [{{"action": "reuse_seed"|"recollect_failing", "dataset": "<repo id>", '
    '"num_episodes": <int>, "notes": "<specific, actionable rationale>"}}]. Diverse '
    "candidates; no prose, no code fences."
)
_SCORE_SYSTEM = (
    "Score each candidate data plan 0-10 for a robot-learning loop: feasibility within the "
    "stated limits, expected improvement on failing tasks, and novelty versus what was "
    'already tried. Output ONLY a JSON array: [{"index": <int>, "score": <number>, '
    '"reason": "<short>"}].'
)
_REFINE_SYSTEM = (
    "Improve the winning data plan: make the notes concrete (which tasks, why this episode "
    "count) and correct anything that conflicts with the limits. Output ONLY the single "
    "JSON object with the same keys."
)


def extract_json(text: str, open_char: str = "{", close_char: str = "}"):
    """Parse the first JSON value in a reply, tolerating code fences and prose."""
    start = text.index(open_char)
    end = text.rindex(close_char) + 1
    return json.loads(text[start:end])


class CuratedProposer:
    def __init__(
        self,
        llm: LLMClient,
        knowledge_root: Path,
        fallback: DeterministicProposer,
        data_cfg: DataConfig,
        candidates: int = 3,
        bus: EventBus | None = None,
        max_tokens: int = 8192,  # reasoning models think first; small budgets truncate
    ):
        self.llm = llm
        self.knowledge_root = Path(knowledge_root)
        self.fallback = fallback
        self.data_cfg = data_cfg
        self.candidates = candidates
        self.bus = bus
        self.max_tokens = max_tokens

    # -- helpers -----------------------------------------------------------
    def _emit(self, kind: str, cycle: int, payload: dict) -> None:
        if self.bus:
            self.bus.emit(Event("curation", "curation", kind, cycle, payload))

    def _knowledge_context(self, limit: int = 4) -> str:
        if not self.knowledge_root.exists():
            return "(no knowledge pages yet)"
        pages = sorted(
            (p for p in self.knowledge_root.rglob("*.md") if p.name != "KNOWLEDGE.md"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )[:limit]
        return "\n\n".join(f"<page {p.name}>\n{p.read_text()[:2000]}\n</page>" for p in pages)

    def _context(self, cycle: int, last_report: EvalReport | None) -> str:
        schedule = self.fallback._episodes_for(cycle)
        return (
            f"Cycle: {cycle}\n"
            f"Seed dataset (the ONLY allowed dataset): {self.fallback.seed_dataset}\n"
            f"Schedule suggestion for this cycle: {schedule} episodes"
            f" (hard cap: {self.data_cfg.max_episodes})\n"
            f"Last eval: {last_report.model_dump_json(exclude={'raw'}) if last_report else 'none'}\n\n"
            f"Knowledge pages (newest first):\n{self._knowledge_context()}"
        )

    def _sanitize(self, raw: dict, cycle: int) -> Proposal | None:
        """Deterministic guardrails — the LLM proposes, these rules dispose."""
        if not isinstance(raw, dict) or raw.get("action") not in _ACTIONS:
            return None
        episodes = raw.get("num_episodes")
        if isinstance(episodes, (int, float)):
            episodes = max(1, int(episodes))
            if self.data_cfg.max_episodes:
                episodes = min(episodes, self.data_cfg.max_episodes)
        else:
            episodes = self.fallback._episodes_for(cycle)
        return Proposal(
            action=raw["action"],
            dataset=self.fallback.seed_dataset,  # pinned: no dataset switching in M1
            num_episodes=episodes,
            notes=str(raw.get("notes", ""))[:500],
        )

    # -- pipeline ------------------------------------------------------------
    def propose(self, cycle: int, last_report: EvalReport | None) -> Proposal:
        context = self._context(cycle, last_report)
        try:
            reply = self.llm.complete(
                context, system=_PROPOSE_SYSTEM.format(k=self.candidates),
                max_tokens=self.max_tokens,
            )
            candidates = [
                p for p in (self._sanitize(c, cycle) for c in extract_json(reply, "[", "]"))
                if p is not None
            ]
            if not candidates:
                raise ValueError("no valid candidates")
            self._emit("curation_candidates", cycle,
                       {"count": len(candidates),
                        "notes": [c.notes[:80] for c in candidates]})

            best = candidates[0]
            if len(candidates) > 1:
                listing = "\n".join(
                    f"{i}: {c.model_dump_json()}" for i, c in enumerate(candidates)
                )
                scores_reply = self.llm.complete(
                    f"{context}\n\nCandidates:\n{listing}", system=_SCORE_SYSTEM,
                    max_tokens=self.max_tokens,
                )
                scores = extract_json(scores_reply, "[", "]")
                ranked = sorted(
                    (s for s in scores if isinstance(s.get("index"), int)
                     and 0 <= s["index"] < len(candidates)),
                    key=lambda s: -float(s.get("score", 0)),
                )
                if ranked:
                    best = candidates[ranked[0]["index"]]
                    self._emit("curation_scored", cycle,
                               {"winner": ranked[0]["index"],
                                "scores": [(s["index"], s.get("score")) for s in ranked]})

            final = best
            try:  # refine is best-effort: a bad refinement never loses the winner
                refined_reply = self.llm.complete(
                    f"{context}\n\nWinning plan:\n{best.model_dump_json()}",
                    system=_REFINE_SYSTEM, max_tokens=self.max_tokens,
                )
                refined = self._sanitize(extract_json(refined_reply), cycle)
                if refined is not None:
                    final = refined
            except Exception:
                pass
            self._emit("curation_final", cycle, {"proposal": final.model_dump()})
            return final
        except Exception as exc:  # any unusable stage -> deterministic fallback
            self._emit("curation_fallback", cycle, {"error": repr(exc)[:200]})
            return self.fallback.propose(cycle, last_report)
