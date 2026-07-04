"""Knowledge Agent — OKF knowledge layer (DESIGN.md §3.6, M1).

Two operations mirroring the Karpathy-LLM-Wiki workflow:
- ingest: after each cycle, distill the eval outcome into task/policy
  pages (markdown + YAML frontmatter, per Google's OKF spec shape).
- lint: health-check pass over the wiki (schema, provenance).

Pages are advisory context only — loop control stays deterministic.
Layer-1 artifacts (events.jsonl, SQLite, eval reports) are never edited;
this agent writes only under the knowledge root.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml

from leagents.contracts import EvalReport
from leagents.events import Event, EventBus
from leagents.llm import LLMClient, NullLLM

_PAGE_TYPES = {"task", "policy", "experiment"}
_STATUSES = {"observed-once", "replicated", "human-confirmed"}

_LESSONS_HEADING = "## Lessons"

_LESSONS_SYSTEM = (
    "You maintain one page of a robotics run-knowledge wiki. Given the page and a new "
    "observation, rewrite ONLY the Lessons section as a short markdown bullet list of "
    "durable, actionable lessons. No preamble, bullets only."
)


def _slug(name: str) -> str:
    return name.lower().replace(" ", "-").replace("/", "-")


def _read_page(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text()
    if text.startswith("---\n"):
        _, frontmatter, body = text.split("---\n", 2)
        return yaml.safe_load(frontmatter) or {}, body.lstrip("\n")
    return {}, text


def _write_page(path: Path, meta: dict[str, Any], body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip()
    path.write_text(f"---\n{frontmatter}\n---\n\n{body.strip()}\n")


class KnowledgeAgent:
    def __init__(self, root: Path, bus: EventBus, llm: LLMClient | None = None):
        self.root = Path(root)
        self.bus = bus
        self.llm = llm or NullLLM()

    # -- ingest ----------------------------------------------------------
    def ingest(
        self,
        *,
        run_id: str,
        cycle: int,
        policy: str,
        report: EvalReport,
        decision: str,
    ) -> list[Path]:
        observation = (
            f"- {time.strftime('%Y-%m-%d')} · run {run_id} · cycle {cycle} · "
            f"{report.task_suite} · policy {policy} · "
            f"success {report.success_rate:.1%} · decision {decision}"
        )
        touched = [
            self._upsert_page(
                self.root / "policies" / f"{_slug(policy)}.md",
                page_type="policy",
                title=f"Policy: {policy}",
                name=policy,
                observation=observation,
                provenance={"run": run_id, "cycle": cycle},
            ),
            self._upsert_page(
                self.root / "tasks" / f"{_slug(report.task_suite)}.md",
                page_type="task",
                title=f"Task: {report.task_suite}",
                name=report.task_suite,
                observation=observation,
                provenance={"run": run_id, "cycle": cycle},
            ),
        ]
        self.bus.emit(
            Event(run_id, "knowledge", "knowledge_updated", cycle,
                  {"pages": [str(p) for p in touched]})
        )
        return touched

    def _upsert_page(
        self,
        path: Path,
        *,
        page_type: str,
        title: str,
        name: str,
        observation: str,
        provenance: dict[str, Any],
    ) -> Path:
        if path.exists():
            meta, body = _read_page(path)
        else:
            meta = {"name": name, "type": page_type, "status": "observed-once",
                    "provenance": []}
            body = f"# {title}\n\n## Observations\n\n{_LESSONS_HEADING}\n"

        meta.setdefault("provenance", []).append(provenance)
        if meta.get("status") != "human-confirmed":  # human verdicts are never downgraded
            meta["status"] = "replicated" if len(meta["provenance"]) >= 2 else "observed-once"
        meta["updated"] = time.strftime("%Y-%m-%d")

        head, sep, lessons = body.partition(_LESSONS_HEADING)
        body = head.rstrip("\n") + f"\n{observation}\n\n" + sep + lessons

        lessons_update = self.llm.complete(
            f"PAGE:\n{body}\n\nNEW OBSERVATION:\n{observation}", system=_LESSONS_SYSTEM
        ).strip()
        if lessons_update:
            head, _, _ = body.partition(_LESSONS_HEADING)
            body = head + f"{_LESSONS_HEADING}\n{lessons_update}\n"

        _write_page(path, meta, body)
        return path

    # -- lint --------------------------------------------------------------
    def lint(self) -> list[dict[str, str]]:
        """Health check over all pages; findings feed the next experiments."""
        findings: list[dict[str, str]] = []
        for path in sorted(self.root.rglob("*.md")):
            if path.name == "KNOWLEDGE.md":
                continue
            try:
                meta, _ = _read_page(path)
            except Exception as exc:
                findings.append({"page": str(path), "problem": f"unparseable: {exc!r}"})
                continue
            if meta.get("type") not in _PAGE_TYPES:
                findings.append({"page": str(path), "problem": f"bad type {meta.get('type')!r}"})
            if meta.get("status") not in _STATUSES:
                findings.append({"page": str(path),
                                 "problem": f"bad status {meta.get('status')!r}"})
            if not meta.get("provenance"):
                findings.append({"page": str(path), "problem": "missing provenance"})
        self.bus.emit(Event("knowledge-lint", "knowledge", "lint_report",
                            payload={"findings": findings}))
        return findings
