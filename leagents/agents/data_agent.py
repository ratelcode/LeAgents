"""Data Agent (DESIGN.md §3.2).

M0/M1 scope: resolve the seed dataset for each cycle. When a task filter is
configured, episode selection is delegated to
``leagents.scripts.select_episodes`` (a subprocess, like every agent tool),
which matches episode task strings against the eval suite and balances
per-task coverage — a bare ``[0..n)`` prefix silently trains on the wrong
suite when the dataset is suite-ordered (learned the hard way; see README).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from leagents.agents.base import Runner, subprocess_runner
from leagents.contracts import DatasetRef, Proposal
from leagents.events import Event, EventBus


class DataError(Exception):
    pass


class DataAgent:
    def __init__(self, bus: EventBus, runner: Runner = subprocess_runner,
                 task_filter: str | None = None):
        self.bus = bus
        self.runner = runner
        self.task_filter = task_filter

    def run(self, *, run_id: str, cycle: int, proposal: Proposal,
            workdir: Path | None = None) -> DatasetRef:
        if self.task_filter and proposal.num_episodes and workdir is not None:
            ref = self._select_episodes(run_id, cycle, proposal, workdir)
        else:
            ref = DatasetRef(
                repo_id=proposal.dataset,
                num_episodes=proposal.num_episodes,
                notes=proposal.notes,
            )
        self.bus.emit(
            Event(run_id, "collect", "dataset_resolved", cycle,
                  {"action": proposal.action, "repo_id": ref.repo_id,
                   "num_episodes": ref.num_episodes,
                   "task_filter": self.task_filter,
                   "episode_range": ([min(ref.episodes), max(ref.episodes)]
                                     if ref.episodes else None),
                   "notes": ref.notes})
        )
        return ref

    def _select_episodes(self, run_id: str, cycle: int, proposal: Proposal,
                         workdir: Path) -> DatasetRef:
        cmd = [
            sys.executable, "-m", "leagents.scripts.select_episodes",
            f"--repo-id={proposal.dataset}",
            f"--suite={self.task_filter}",
            f"--limit={proposal.num_episodes}",
        ]
        log_path = workdir / f"cycle_{cycle}" / "select_episodes.log"
        self.bus.emit(Event(run_id, "collect", "job_started", cycle, {"cmd": cmd}))
        result = self.runner(cmd, log_path)
        self.bus.emit(Event(run_id, "collect", "job_finished", cycle,
                            {"exit_code": result.exit_code,
                             "duration_s": result.duration_s,
                             "log": str(result.log_path)}))
        if result.exit_code != 0:
            # fail LOUD: silently training on the wrong tasks is exactly the
            # failure mode this stage exists to prevent
            raise DataError(f"episode selection failed (exit {result.exit_code}), "
                            f"see {result.log_path}")
        summary = self._parse_summary(result.log_path)
        return DatasetRef(
            repo_id=proposal.dataset,
            episodes=summary["episodes"],
            num_episodes=summary["selected"],
            notes=f"{proposal.notes} | suite={self.task_filter}, "
                  f"{summary['tasks_covered']}/{summary['suite_tasks']} tasks covered",
        )

    @staticmethod
    def _parse_summary(log_path: Path | None) -> dict:
        if log_path is None or not Path(log_path).exists():
            raise DataError(f"selection log missing: {log_path}")
        for line in reversed(Path(log_path).read_text().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        raise DataError(f"no JSON summary found in {log_path}")
