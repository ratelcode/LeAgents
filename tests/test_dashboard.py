import json

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from leagent.dashboard.server import create_app  # noqa: E402
from leagent.store import JobStore  # noqa: E402


@pytest.fixture
def client(tmp_path):
    workdir = tmp_path / "runs"
    store = JobStore(workdir / "leagent.db")
    store.create_run("run1", json.dumps({"budgets": {"max_cycles": 3}}))
    store.start_cycle("run1", 0)
    ckpt = store.add_checkpoint("run1", 0, "smolvla", "/ckpt/a")
    store.attach_eval(ckpt, {"success_rate": 0.42})
    store.bless("run1", ckpt)
    store.finish_cycle("run1", 0, "promote")
    store.finish_run("run1", "finished", "budget: max_cycles")

    events_dir = workdir / "run1"
    events_dir.mkdir(parents=True)
    (events_dir / "events.jsonl").write_text(
        json.dumps({"run_id": "run1", "stage": "train", "kind": "job_finished",
                    "cycle": 0, "payload": {"duration_s": 3600.0}, "ts": 1.0}) + "\n"
        + json.dumps({"run_id": "run1", "stage": "decide", "kind": "decision",
                      "cycle": 0, "payload": {"decision": "promote"}, "ts": 2.0}) + "\n"
    )

    knowledge = tmp_path / "knowledge"
    (knowledge / "policies").mkdir(parents=True)
    (knowledge / "policies" / "smolvla.md").write_text("---\nname: smolvla\n---\n\n# Policy\n")

    return TestClient(create_app(workdir, knowledge))


def test_index_serves_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "leagent" in response.text


def test_runs_endpoint(client):
    runs = client.get("/api/runs").json()
    assert len(runs) == 1
    assert runs[0]["id"] == "run1"
    assert runs[0]["blessed"]["path"] == "/ckpt/a"
    assert runs[0]["cycles"][0]["decision"] == "promote"


def test_run_detail_parses_eval_and_gpu_seconds(client):
    detail = client.get("/api/runs/run1").json()
    assert detail["checkpoints"][0]["success_rate"] == 0.42
    assert detail["gpu_seconds"] == 3600.0
    assert detail["config"]["budgets"]["max_cycles"] == 3
    assert client.get("/api/runs/nope").status_code == 404


def test_events_cursor_pagination(client):
    first = client.get("/api/runs/run1/events?after=0").json()
    assert len(first["events"]) == 2 and first["next"] == 2
    second = client.get(f"/api/runs/run1/events?after={first['next']}").json()
    assert second["events"] == [] and second["next"] == 2


def test_knowledge_pages_listed(client):
    pages = client.get("/api/knowledge").json()
    assert [p["path"] for p in pages] == ["policies/smolvla.md"]
    assert "smolvla" in pages[0]["content"]
