"""Deterministic cycle controller: COLLECT → TRAIN → EVAL → DECIDE.

Control flow is plain Python state persisted to SQLite — never an LLM
and never framework-checkpoint magic (DESIGN.md §4). Agents are injected
so the full loop runs under tests and --dry-run with fake runners.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from leagent.config import LoopConfig
from leagent.contracts import EvalReport
from leagent.events import Event, EventBus
from leagent.orchestrator.decision import Decision, decide
from leagent.orchestrator.proposer import Proposer
from leagent.store import JobStore

if TYPE_CHECKING:  # runtime import would be circular: agents depend on orchestrator
    from leagent.agents.data_agent import DataAgent
    from leagent.agents.eval_agent import EvalAgent
    from leagent.agents.train_agent import TrainAgent


@dataclass
class RunSummary:
    run_id: str
    cycles_run: int = 0
    decisions: list[str] = field(default_factory=list)
    final_policy: str | None = None
    baseline_success_rate: float | None = None
    stop_reason: str = "budget: max_cycles"

    def to_dict(self) -> dict:
        return asdict(self)


class LoopController:
    def __init__(
        self,
        cfg: LoopConfig,
        store: JobStore,
        bus: EventBus,
        data_agent: DataAgent,
        train_agent: TrainAgent,
        eval_agent: EvalAgent,
        proposer: Proposer,
    ):
        self.cfg = cfg
        self.store = store
        self.bus = bus
        self.data_agent = data_agent
        self.train_agent = train_agent
        self.eval_agent = eval_agent
        self.proposer = proposer
        self._job_seconds = 0.0
        bus.subscribe(self._track_budget)

    def _track_budget(self, event: Event) -> None:
        if event.kind == "job_finished":
            self._job_seconds += float(event.payload.get("duration_s", 0.0))

    def run(self, run_id: str, workdir: Path) -> RunSummary:
        self.store.create_run(run_id, self.cfg.model_dump_json())
        summary = RunSummary(run_id=run_id)
        ladder = self.cfg.policy_ladder
        rung_idx = 0
        baseline: float | None = None
        history: list[float] = []
        last_report: EvalReport | None = None

        try:
            for cycle in range(self.cfg.budgets.max_cycles):
                if self._job_seconds / 3600.0 >= self.cfg.budgets.max_gpu_hours:
                    summary.stop_reason = "budget: max_gpu_hours"
                    break

                rung = ladder[rung_idx]
                self.store.start_cycle(run_id, cycle)
                self.bus.emit(Event(run_id, "orchestrator", "cycle_started", cycle,
                                    {"policy": rung.name}))

                # COLLECT
                proposal = self.proposer.propose(cycle, last_report)
                dataset = self.data_agent.run(run_id=run_id, cycle=cycle, proposal=proposal)

                # TRAIN
                self.store.set_stage(run_id, cycle, "train")
                checkpoint = self.train_agent.run(
                    run_id=run_id, cycle=cycle, rung=rung, dataset=dataset, workdir=workdir
                )
                checkpoint_id = self.store.add_checkpoint(
                    run_id, cycle, checkpoint.policy_type, checkpoint.path
                )

                # EVAL
                self.store.set_stage(run_id, cycle, "eval")
                report = self.eval_agent.run(
                    run_id=run_id, cycle=cycle, checkpoint=checkpoint, workdir=workdir
                )
                self.store.attach_eval(checkpoint_id, report.model_dump())

                # DECIDE
                decision = decide(report.success_rate, baseline, history, self.cfg.thresholds)
                self.bus.emit(Event(run_id, "decide", "decision", cycle,
                                    {"decision": decision.value,
                                     "success_rate": report.success_rate,
                                     "baseline": baseline}))
                self.store.finish_cycle(run_id, cycle, decision.value)
                summary.decisions.append(decision.value)
                summary.cycles_run = cycle + 1

                if decision is Decision.PROMOTE:
                    baseline = report.success_rate
                    history = []
                    self.store.bless(run_id, checkpoint_id)
                elif decision is Decision.ESCALATE:
                    history = []
                    if rung_idx + 1 < len(ladder):
                        rung_idx += 1
                        self.bus.emit(Event(run_id, "orchestrator", "policy_escalated", cycle,
                                            {"to": ladder[rung_idx].name}))
                    else:
                        summary.stop_reason = "policy ladder exhausted"
                        last_report = report
                        break
                else:  # ITERATE or ROLLBACK: baseline and blessed checkpoint stand
                    history.append(report.success_rate)

                last_report = report

            summary.final_policy = ladder[rung_idx].name
            summary.baseline_success_rate = baseline
            self.store.finish_run(run_id, "finished", summary.stop_reason)
            self.bus.emit(Event(run_id, "orchestrator", "run_finished",
                                payload=summary.to_dict()))
            return summary
        except Exception as exc:
            self.store.finish_run(run_id, "failed", repr(exc))
            self.bus.emit(Event(run_id, "orchestrator", "run_failed", payload={"error": repr(exc)}))
            raise
