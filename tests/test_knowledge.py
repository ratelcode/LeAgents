import yaml

from leloop.agents import KnowledgeAgent
from leloop.contracts import EvalReport
from leloop.events import EventBus


class FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply

    def complete(self, prompt: str, *, system=None, max_tokens: int = 1024) -> str:
        return self.reply


def _report(suite="libero_spatial", rate=0.5) -> EvalReport:
    return EvalReport(checkpoint="/ckpt", env_type="libero", task_suite=suite,
                      n_episodes=10, success_rate=rate)


def _frontmatter(path) -> dict:
    _, meta, _ = path.read_text().split("---\n", 2)
    return yaml.safe_load(meta)


def _agent(tmp_path, llm=None):
    return KnowledgeAgent(tmp_path / "knowledge", EventBus(tmp_path / "events.jsonl"), llm)


def test_ingest_creates_policy_and_task_pages(tmp_path):
    agent = _agent(tmp_path)
    pages = agent.ingest(run_id="r1", cycle=0, policy="smolvla",
                         report=_report(), decision="promote")

    assert [p.name for p in pages] == ["smolvla.md", "libero_spatial.md"]
    meta = _frontmatter(pages[0])
    assert meta["type"] == "policy"
    assert meta["status"] == "observed-once"
    assert meta["provenance"] == [{"run": "r1", "cycle": 0}]
    body = pages[0].read_text()
    assert "success 50.0%" in body and "decision promote" in body and "## Lessons" in body


def test_second_ingest_appends_and_marks_replicated(tmp_path):
    agent = _agent(tmp_path)
    agent.ingest(run_id="r1", cycle=0, policy="smolvla", report=_report(rate=0.5),
                 decision="promote")
    (page, _) = agent.ingest(run_id="r2", cycle=1, policy="smolvla",
                             report=_report(rate=0.6), decision="promote")

    meta = _frontmatter(page)
    assert meta["status"] == "replicated"
    assert len(meta["provenance"]) == 2
    body = page.read_text()
    assert "success 50.0%" in body and "success 60.0%" in body


def test_human_confirmed_status_is_never_downgraded(tmp_path):
    agent = _agent(tmp_path)
    (page, _) = agent.ingest(run_id="r1", cycle=0, policy="smolvla",
                             report=_report(), decision="promote")
    page.write_text(page.read_text().replace("status: observed-once",
                                             "status: human-confirmed"))
    agent.ingest(run_id="r2", cycle=1, policy="smolvla", report=_report(),
                 decision="iterate")
    assert _frontmatter(page)["status"] == "human-confirmed"


def test_llm_rewrites_lessons_section(tmp_path):
    agent = _agent(tmp_path, llm=FakeLLM("- always use 50 episodes per variation"))
    (page, _) = agent.ingest(run_id="r1", cycle=0, policy="smolvla",
                             report=_report(), decision="promote")
    assert "- always use 50 episodes per variation" in page.read_text()


def test_lint_passes_on_agent_written_pages(tmp_path):
    agent = _agent(tmp_path)
    agent.ingest(run_id="r1", cycle=0, policy="smolvla", report=_report(),
                 decision="promote")
    assert agent.lint() == []


def test_lint_flags_missing_provenance_and_bad_type(tmp_path):
    agent = _agent(tmp_path)
    bad = tmp_path / "knowledge" / "tasks" / "bad.md"
    bad.parent.mkdir(parents=True)
    bad.write_text("---\nname: bad\ntype: nonsense\nstatus: observed-once\n---\n\n# Bad\n")
    problems = {f["problem"] for f in agent.lint()}
    assert any("bad type" in p for p in problems)
    assert any("missing provenance" in p for p in problems)


def test_lint_skips_schema_file(tmp_path):
    agent = _agent(tmp_path)
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "knowledge" / "KNOWLEDGE.md").write_text("# schema, no frontmatter")
    assert agent.lint() == []
