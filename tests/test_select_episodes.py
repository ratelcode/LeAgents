import json

from leagents.agents.data_agent import DataAgent, DataError
from leagents.agents.base import RunResult
from leagents.contracts import Proposal
from leagents.events import EventBus
from leagents.scripts.select_episodes import (
    files_covering,
    missing_episodes,
    normalize,
    select_balanced,
)

import pytest

SUITE = ["Pick up the RED bowl", "push the plate  forward"]
EPISODES = [
    (0, "totally different long-horizon task"),   # wrong suite (the old bug)
    (1, "pick up the red bowl"),
    (2, "push the plate forward"),
    (3, "pick up the red bowl"),
    (4, "another unrelated task"),
    (5, "push the plate forward"),
    (6, "pick up the red bowl"),
]


def test_normalize_case_and_whitespace():
    assert normalize("  Pick  UP the\tbowl ") == "pick up the bowl"


def test_selection_excludes_other_suites_and_balances():
    selected, counts = select_balanced(EPISODES, SUITE, limit=4)
    assert 0 not in selected and 4 not in selected  # wrong-suite episodes excluded
    assert selected == [1, 2, 3, 5]
    assert sorted(counts.values()) == [2, 2]  # balanced across the two tasks


def test_selection_caps_at_available():
    selected, _ = select_balanced(EPISODES, SUITE, limit=100)
    assert selected == [1, 2, 3, 5, 6]


SPANS = [
    ("data/chunk-000/file-300.parquet", 1220, 1260),
    ("data/chunk-000/file-301.parquet", 1261, 1266),
    ("data/chunk-000/file-302.parquet", 1267, 1272),  # gap: nothing selected here
    ("data/chunk-000/file-303.parquet", 1273, 1278),
]


def test_files_covering_intersects_selected_set_not_just_range():
    selected = [1261, 1275]  # skips file-302's span entirely
    assert files_covering(selected, SPANS) == [
        "data/chunk-000/file-301.parquet",
        "data/chunk-000/file-303.parquet",
    ]


def test_files_covering_boundary_episodes():
    assert files_covering([1260], SPANS) == ["data/chunk-000/file-300.parquet"]
    assert files_covering([1266, 1267], SPANS) == [
        "data/chunk-000/file-301.parquet",
        "data/chunk-000/file-302.parquet",
    ]


def test_missing_episodes_detects_uncovered():
    assert missing_episodes([1, 2, 3], {1, 3}) == [2]
    assert missing_episodes([1, 2], {1, 2, 99}) == []


def test_no_match_returns_empty():
    selected, counts = select_balanced(EPISODES, ["unknown task"], limit=10)
    assert selected == [] and counts == {}


def _selection_runner(summary: dict, exit_code: int = 0):
    def runner(cmd, log_path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("noise\n" + json.dumps(summary) + "\n")
        return RunResult(list(cmd), exit_code, 3.0, log_path)

    return runner


def test_data_agent_uses_selected_indices(tmp_path):
    summary = {"episodes": [1261, 1290], "selected": 2, "suite_tasks": 10,
               "tasks_covered": 2, "per_task": {}}
    agent = DataAgent(EventBus(tmp_path / "e.jsonl"),
                      _selection_runner(summary), task_filter="libero_spatial")
    ref = agent.run(run_id="r", cycle=0,
                    proposal=Proposal(action="reuse_seed", dataset="org/seed",
                                      num_episodes=2),
                    workdir=tmp_path)
    assert ref.episodes == [1261, 1290]
    assert "libero_spatial" in ref.notes


def test_data_agent_fails_loud_on_selection_error(tmp_path):
    agent = DataAgent(EventBus(tmp_path / "e.jsonl"),
                      _selection_runner({}, exit_code=1), task_filter="libero_spatial")
    with pytest.raises(DataError):
        agent.run(run_id="r", cycle=0,
                  proposal=Proposal(action="reuse_seed", dataset="org/seed",
                                    num_episodes=2),
                  workdir=tmp_path)


def test_data_agent_without_filter_keeps_old_behavior(tmp_path):
    agent = DataAgent(EventBus(tmp_path / "e.jsonl"))
    ref = agent.run(run_id="r", cycle=0,
                    proposal=Proposal(action="reuse_seed", dataset="org/seed",
                                      num_episodes=5),
                    workdir=tmp_path)
    assert ref.episodes is None and ref.num_episodes == 5
