"""End-to-end loop tests with fake runners — no lerobot, no GPU."""

from leloop.agents import DataAgent, EvalAgent, KnowledgeAgent, TrainAgent
from leloop.events import EventBus
from leloop.orchestrator import DeterministicProposer, LoopController
from leloop.store import JobStore
from tests.conftest import make_eval_runner, make_train_runner


def _controller(loop_config, constitution, tmp_path, eval_scores, knowledge_root=None,
                seen_train_cmds=None):
    bus = EventBus(tmp_path / "events.jsonl")
    store = JobStore(tmp_path / "leloop.db")
    controller = LoopController(
        cfg=loop_config,
        store=store,
        bus=bus,
        data_agent=DataAgent(bus),
        train_agent=TrainAgent(loop_config.train, constitution, bus,
                               make_train_runner(seen_cmds=seen_train_cmds)),
        eval_agent=EvalAgent(loop_config.eval, constitution, bus,
                             make_eval_runner(eval_scores)),
        proposer=DeterministicProposer(loop_config.seed_dataset),
        knowledge_agent=KnowledgeAgent(knowledge_root, bus) if knowledge_root else None,
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


def test_later_cycles_continue_from_blessed_checkpoint(loop_config, constitution, tmp_path):
    cmds: list = []
    controller, _ = _controller(
        loop_config, constitution, tmp_path, eval_scores=[0.50, 0.60, 0.70],
        seen_train_cmds=cmds,
    )
    controller.run("run-continue", tmp_path / "wd")

    # cycle 0 starts from the ladder init; cycles 1-2 from the prior blessed checkpoint
    assert "--policy.path=lerobot/smolvla_base" in cmds[0]
    assert any("cycle_0/train/checkpoints/last" in arg for arg in cmds[1])
    assert any("cycle_1/train/checkpoints/last" in arg for arg in cmds[2])


def test_continue_from_blessed_can_be_disabled(loop_config, constitution, tmp_path):
    loop_config.train.continue_from_blessed = False
    cmds: list = []
    controller, _ = _controller(
        loop_config, constitution, tmp_path, eval_scores=[0.50, 0.60],
        seen_train_cmds=cmds,
    )
    controller.run("run-fresh", tmp_path / "wd")
    assert all("--policy.path=lerobot/smolvla_base" in cmd for cmd in cmds)


def test_knowledge_pages_written_each_cycle(loop_config, constitution, tmp_path):
    root = tmp_path / "knowledge"
    controller, _ = _controller(
        loop_config, constitution, tmp_path, eval_scores=[0.50, 0.53, 0.60],
        knowledge_root=root,
    )
    controller.run("run-knowledge", tmp_path / "wd")

    policy_page = root / "policies" / "smolvla.md"
    task_page = root / "tasks" / "libero_spatial.md"
    assert policy_page.exists() and task_page.exists()
    body = policy_page.read_text()
    assert "status: replicated" in body  # 3 cycles -> >= 2 provenance entries
    assert body.count("· cycle") == 3


def test_events_are_logged_as_jsonl(loop_config, constitution, tmp_path):
    controller, _ = _controller(loop_config, constitution, tmp_path, eval_scores=[0.5])
    controller.run("run-events", tmp_path / "wd")
    lines = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    assert len(lines) > 0
    import json

    kinds = {json.loads(line)["kind"] for line in lines}
    assert {"cycle_started", "job_started", "job_finished", "decision"} <= kinds
