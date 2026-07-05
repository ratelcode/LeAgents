import json
import sys

from leagents.cli import _dry_runner_with_synthetic_eval


def test_dry_runner_fakes_eval_info(tmp_path):
    out = tmp_path / "eval"
    cmd = ["lerobot-eval", f"--output_dir={out}"]
    _dry_runner_with_synthetic_eval(cmd, tmp_path / "eval.log")
    info = json.loads((out / "eval_info.json").read_text())
    assert info["overall"]["pc_success"] == 50.0 and info["dry_run"]


def test_dry_runner_fakes_select_summary(tmp_path):
    # a task-filtered --dry-run needs a JSON selection summary or the Data
    # Agent's _parse_summary raises DataError
    log = tmp_path / "cycle_0" / "select_episodes.log"
    cmd = [sys.executable, "-m", "leagents.scripts.select_episodes",
           "--repo-id=org/data", "--suite=libero_spatial", "--limit=4"]
    _dry_runner_with_synthetic_eval(cmd, log)
    summary = json.loads(log.read_text().splitlines()[-1])
    assert summary["episodes"] == [0, 1, 2, 3]
    assert summary["selected"] == 4 and summary["dry_run"]
