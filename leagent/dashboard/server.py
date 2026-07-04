"""Flow dashboard v1 (M2, DESIGN.md §5).

FastAPI + one static page, no build step. Reads exactly the artifacts the
loop already writes — the SQLite job store, per-run events.jsonl, and the
knowledge pages — and renders runs → cycles → decisions live via polling.
A React shell can replace the page if it outgrows this; the API stays.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from leagent.store import JobStore

_STATIC = Path(__file__).parent / "static" / "index.html"


def _gpu_seconds(events_path: Path) -> float:
    total = 0.0
    if events_path.exists():
        for line in events_path.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("kind") == "job_finished":
                total += float(event.get("payload", {}).get("duration_s", 0.0))
    return total


def create_app(workdir: Path, knowledge_root: Path):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse

    workdir = Path(workdir)
    knowledge_root = Path(knowledge_root)
    db_path = workdir / "leagent.db"
    app = FastAPI(title="leagent dashboard")

    def store() -> JobStore:
        # one short-lived connection per request: sqlite connections are not
        # shareable across FastAPI's worker threads
        return JobStore(db_path)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _STATIC.read_text()

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        if not db_path.exists():
            return []
        s = store()
        return [
            {
                **run,
                "cycles": s.cycles_for(run["id"]),
                "blessed": s.blessed_checkpoint(run["id"]),
            }
            for run in s.list_runs()
        ]

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict[str, Any]:
        s = store()
        run = next((r for r in s.list_runs() if r["id"] == run_id), None)
        if run is None:
            raise HTTPException(404, f"unknown run {run_id!r}")
        checkpoints = []
        for ckpt in s.checkpoints_for(run_id):
            eval_data = json.loads(ckpt["eval_json"]) if ckpt.get("eval_json") else None
            checkpoints.append(
                {
                    **{k: v for k, v in ckpt.items() if k != "eval_json"},
                    "success_rate": eval_data.get("success_rate") if eval_data else None,
                }
            )
        return {
            **run,
            "config": json.loads(run["config_json"]),
            "cycles": s.cycles_for(run_id),
            "checkpoints": checkpoints,
            "blessed": s.blessed_checkpoint(run_id),
            "gpu_seconds": _gpu_seconds(workdir / run_id / "events.jsonl"),
        }

    @app.get("/api/runs/{run_id}/events")
    def run_events(run_id: str, after: int = 0, limit: int = 200) -> dict[str, Any]:
        path = workdir / run_id / "events.jsonl"
        if not path.exists():
            return {"events": [], "next": after}
        lines = path.read_text().splitlines()
        window = lines[after : after + limit]
        events = []
        for line in window:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return {"events": events, "next": after + len(window)}

    @app.get("/api/knowledge")
    def knowledge() -> list[dict[str, str]]:
        if not knowledge_root.exists():
            return []
        return [
            {"path": str(page.relative_to(knowledge_root)), "content": page.read_text()}
            for page in sorted(knowledge_root.rglob("*.md"))
        ]

    return app
