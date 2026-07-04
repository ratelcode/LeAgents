"""Bundled sample run (issue #5) — a real 3-cycle M0 run on LIBERO spatial
(promote → iterate → iterate, the run documented in README's "First
full-scale autonomous run"), trimmed to ~0.5 MB: events, decisions, eval
reports, six rollout videos, and the knowledge pages it produced.

``leagents demo`` installs it into a workdir so ``leagents dash`` shows a
full flow view seconds after install — GPU or not.
"""

from __future__ import annotations

import json
import shutil
from importlib.resources import files
from pathlib import Path

from leagents.store import JobStore


def _data_root() -> Path:
    return Path(str(files("leagents.demo").joinpath("data")))


def install_demo(workdir: Path, knowledge_root: Path) -> tuple[str, bool]:
    """Copy the bundled run into ``workdir`` and replay its rows through the
    job store. Idempotent: returns (run_id, installed) where installed is
    False if the run was already present."""
    data = _data_root()
    manifest = json.loads((data / "run.json").read_text())
    run_id = manifest["run_id"]

    workdir.mkdir(parents=True, exist_ok=True)
    store = JobStore(workdir / "leagents.db")
    if any(r["id"] == run_id for r in store.list_runs()):
        return run_id, False

    run_dir = workdir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(data / "events.jsonl", run_dir / "events.jsonl")
    for video in manifest["videos"]:
        dest = run_dir / video["dest"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(data / video["src"], dest)

    store.create_run(run_id, manifest["run"]["config_json"])
    for cycle in manifest["cycles"]:
        store.start_cycle(run_id, cycle["idx"])
        store.finish_cycle(run_id, cycle["idx"], cycle["decision"])
    for ckpt in manifest["checkpoints"]:
        ckpt_id = store.add_checkpoint(run_id, ckpt["cycle_idx"],
                                       ckpt["policy_type"], ckpt["path"])
        if ckpt["eval"]:
            store.attach_eval(ckpt_id, ckpt["eval"])
        if ckpt["blessed"]:
            store.bless(run_id, ckpt_id)
    store.finish_run(run_id, manifest["run"]["status"],
                     manifest["run"]["stop_reason"])

    for page in (data / "knowledge").rglob("*.md"):
        dest = knowledge_root / page.relative_to(data / "knowledge")
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(page, dest)

    return run_id, True
