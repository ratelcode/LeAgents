from leagents.store import JobStore


def test_run_cycle_checkpoint_roundtrip(tmp_path):
    store = JobStore(tmp_path / "leagents.db")
    store.create_run("run1", "{}")
    store.start_cycle("run1", 0)
    store.set_stage("run1", 0, "train")

    ckpt_a = store.add_checkpoint("run1", 0, "smolvla", "/ckpt/a")
    store.attach_eval(ckpt_a, {"success_rate": 0.5})
    store.bless("run1", ckpt_a)
    store.finish_cycle("run1", 0, "promote")

    ckpt_b = store.add_checkpoint("run1", 1, "smolvla", "/ckpt/b")
    store.bless("run1", ckpt_b)  # blessing is exclusive

    blessed = store.blessed_checkpoint("run1")
    assert blessed is not None and blessed["path"] == "/ckpt/b"

    store.finish_run("run1", "finished", "budget: max_cycles")
    runs = store.list_runs()
    assert runs[0]["status"] == "finished"
    cycles = store.cycles_for("run1")
    assert cycles[0]["decision"] == "promote" and cycles[0]["stage"] == "done"
