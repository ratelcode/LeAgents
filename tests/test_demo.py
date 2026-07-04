import json

from leagents.demo import install_demo
from leagents.store import JobStore


def test_demo_installs_full_run(tmp_path):
    workdir = tmp_path / "runs"
    knowledge = tmp_path / "knowledge"
    run_id, installed = install_demo(workdir, knowledge)

    assert installed
    store = JobStore(workdir / "leagents.db")
    runs = store.list_runs()
    assert [r["id"] for r in runs] == [run_id]
    assert runs[0]["status"] == "finished"

    cycles = store.cycles_for(run_id)
    assert len(cycles) == 3
    assert cycles[0]["decision"] == "promote"

    ckpts = store.checkpoints_for(run_id)
    assert len(ckpts) == 3
    assert store.blessed_checkpoint(run_id) is not None

    # every eval_json video path resolves to a bundled mp4
    for ckpt in ckpts:
        raw = json.loads(ckpt["eval_json"])["raw"]
        paths = raw["overall"]["video_paths"]
        assert paths, "demo checkpoints must reference bundled videos"
        for rel in paths:
            assert (tmp_path / rel).exists(), rel

    assert (workdir / run_id / "events.jsonl").exists()
    assert (knowledge / "tasks" / "libero_spatial.md").exists()


def test_demo_is_idempotent(tmp_path):
    workdir = tmp_path / "runs"
    knowledge = tmp_path / "knowledge"
    install_demo(workdir, knowledge)
    run_id, installed = install_demo(workdir, knowledge)

    assert not installed
    assert len(JobStore(workdir / "leagents.db").list_runs()) == 1
