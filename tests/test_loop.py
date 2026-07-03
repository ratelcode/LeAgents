"""End-to-end loop tests with fake runners — no lerobot, no GPU."""

from leagent.agents import DataAgent, EvalAgent, TrainAgent
from leagent.events import EventBus
from leagent.orchestrator import DeterministicProposer, LoopController
from leagent.store import JobStore
from tests.conftest import make_eval_runner, make_train_runner


def _controller(loop_config, constitution, tmp_path, eval_scores):
    bus = EventBus(tmp_path / "events.jsonl")
    store = JobStore(tmp_path / "leagent.db")
    controller = LoopController(
        cfg=loop_config,
        store=store,
        bus=bus,
        data_agent=DataAgent(bus),
        train_agent=TrainAgent(loop_config.train, constitution, bus, make_train_runner()),
        eval_agent=EvalAgent(loop_config.eval, constitution, bus,
                             make_eval_runner(eval_scores)),
        proposer=DeterministicProposer(loop_config.seed_dataset),
    )
    return controller, store


def test_improving_run_promotes(loop_config, constitution, tmp_path):
    controller, store = _controller(
        loop_config, constitution, tmp_path, eval_scores=[0.50, 0.53, 0.60]
    )
    summary = controller.run("run-improve", tmp_path / "wd")

    assert summary.decisions == ["promote", "iterate", "promote"]
    assert summary.cycles_run == 3
    assert summary.baseline_success_rate == 0.60
    blessed = store.blessed_checkpoint("run-improve")
    assert blessed is not None and blessed["cycle_idx"] == 2


def test_plateau_escalates_policy_ladder(loop_config, constitution, tmp_path):
    controller, _ = _controller(
        loop_config, constitution, tmp_path, eval_scores=[0.50, 0.505, 0.502]
    )
    summary = controller.run("run-plateau", tmp_path / "wd")

    assert summary.decisions == ["promote", "iterate", "escalate"]
    assert summary.final_policy == "pi05"  # escalated off smolvla


def test_regression_rolls_back_and_keeps_baseline(loop_config, constitution, tmp_path):
    controller, store = _controller(
        loop_config, constitution, tmp_path, eval_scores=[0.50, 0.30, 0.30]
    )
    summary = controller.run("run-regress", tmp_path / "wd")

    assert summary.decisions[0] == "promote"
    assert "rollback" in summary.decisions[1:]
    blessed = store.blessed_checkpoint("run-regress")
    assert blessed is not None and blessed["cycle_idx"] == 0  # cycle-0 checkpoint stands
    assert summary.baseline_success_rate == 0.50


def test_gpu_hour_budget_stops_loop(loop_config, constitution, tmp_path):
    loop_config.budgets.max_gpu_hours = 1.5  # each fake train job burns 1h
    controller, _ = _controller(
        loop_config, constitution, tmp_path, eval_scores=[0.50, 0.53, 0.60]
    )
    summary = controller.run("run-budget", tmp_path / "wd")

    assert summary.cycles_run < 3
    assert summary.stop_reason == "budget: max_gpu_hours"


def test_events_are_logged_as_jsonl(loop_config, constitution, tmp_path):
    controller, _ = _controller(loop_config, constitution, tmp_path, eval_scores=[0.5])
    controller.run("run-events", tmp_path / "wd")
    lines = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    assert len(lines) > 0
    import json

    kinds = {json.loads(line)["kind"] for line in lines}
    assert {"cycle_started", "job_started", "job_finished", "decision"} <= kinds
