from leagents.orchestrator import DeterministicProposer, LLMProposer
from tests.test_knowledge import FakeLLM


def _proposer(tmp_path, reply: str) -> LLMProposer:
    return LLMProposer(FakeLLM(reply), tmp_path / "knowledge",
                       DeterministicProposer("org/seed"))


def test_valid_llm_reply_is_used(tmp_path):
    proposer = _proposer(
        tmp_path,
        'Here you go: {"action": "recollect_failing", "dataset": "org/seed", '
        '"notes": "focus on drawer tasks"}',
    )
    proposal = proposer.propose(1, None)
    assert proposal.action == "recollect_failing"
    assert proposal.notes == "focus on drawer tasks"


def test_garbage_reply_falls_back_to_deterministic(tmp_path):
    proposal = _proposer(tmp_path, "I cannot answer that.").propose(0, None)
    assert proposal.action == "reuse_seed"
    assert proposal.dataset == "org/seed"


def test_null_llm_falls_back(tmp_path):
    proposal = _proposer(tmp_path, "").propose(0, None)
    assert proposal.action == "reuse_seed"


def test_knowledge_pages_reach_the_prompt(tmp_path):
    seen = {}

    class SpyLLM(FakeLLM):
        def complete(self, prompt, *, system=None, max_tokens=1024):
            seen["prompt"] = prompt
            return ""

    root = tmp_path / "knowledge" / "policies"
    root.mkdir(parents=True)
    (root / "smolvla.md").write_text("---\nname: smolvla\n---\n\n# Policy: smolvla\n")
    LLMProposer(SpyLLM(""), tmp_path / "knowledge",
                DeterministicProposer("org/seed")).propose(0, None)
    assert "Policy: smolvla" in seen["prompt"]
