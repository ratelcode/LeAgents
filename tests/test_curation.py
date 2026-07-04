import json

from leagents.config import DataConfig
from leagents.contracts import EvalReport
from leagents.orchestrator import CuratedProposer, DeterministicProposer
from leagents.orchestrator.curation import extract_json


class SequencedLLM:
    """Returns scripted replies in order; repeats the last one when exhausted."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[tuple[str, str | None]] = []

    def complete(self, prompt, *, system=None, max_tokens=1024):
        self.prompts.append((prompt, system))
        return self.replies.pop(0) if self.replies else ""


def _proposer(llm, tmp_path, max_episodes=500, initial=200):
    fallback = DeterministicProposer("org/seed", initial_episodes=initial,
                                     growth=1.6, max_episodes=max_episodes)
    return CuratedProposer(llm, tmp_path / "knowledge", fallback,
                           DataConfig(initial_episodes=initial, growth=1.6,
                                      max_episodes=max_episodes))


def _report(per_task=None) -> EvalReport:
    return EvalReport(checkpoint="/c", env_type="libero", task_suite="libero_spatial",
                      n_episodes=100, success_rate=0.0, per_task=per_task or {})


def test_extract_json_tolerates_code_fences():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('noise [ {"i": 0} ] trailing', "[", "]") == [{"i": 0}]


def test_full_pipeline_picks_scored_winner(tmp_path):
    llm = SequencedLLM([
        json.dumps([
            {"action": "reuse_seed", "dataset": "org/seed", "num_episodes": 200,
             "notes": "baseline"},
            {"action": "recollect_failing", "dataset": "org/seed", "num_episodes": 320,
             "notes": "focus failing"},
        ]),
        json.dumps([{"index": 1, "score": 9, "reason": "targets failures"},
                    {"index": 0, "score": 4, "reason": "no focus"}]),
        json.dumps({"action": "recollect_failing", "dataset": "org/seed",
                    "num_episodes": 320, "notes": "focus the 3 failing spatial tasks"}),
    ])
    proposal = _proposer(llm, tmp_path).propose(1, _report())
    assert proposal.action == "recollect_failing"
    assert proposal.num_episodes == 320
    assert "failing" in proposal.notes
    assert len(llm.prompts) == 3  # propose -> score -> refine


def test_guardrails_cap_episodes_and_pin_dataset(tmp_path):
    llm = SequencedLLM([
        json.dumps([{"action": "reuse_seed", "dataset": "evil/other",
                     "num_episodes": 99999, "notes": "x"}]),
        json.dumps({"action": "reuse_seed", "dataset": "evil/other",
                    "num_episodes": 99999, "notes": "x"}),
    ])
    proposal = _proposer(llm, tmp_path, max_episodes=500).propose(0, None)
    assert proposal.dataset == "org/seed"  # pinned
    assert proposal.num_episodes == 500  # capped


def test_garbage_reply_falls_back(tmp_path):
    proposal = _proposer(SequencedLLM(["I cannot help with that."]),
                         tmp_path).propose(0, None)
    assert proposal.action == "reuse_seed"
    assert proposal.dataset == "org/seed"
    assert proposal.num_episodes == 200  # deterministic schedule


def test_invalid_actions_dropped_then_fallback(tmp_path):
    llm = SequencedLLM([
        json.dumps([{"action": "buy_robots", "dataset": "org/seed",
                     "num_episodes": 10, "notes": "no"}]),
    ])
    proposal = _proposer(llm, tmp_path).propose(2, None)
    assert proposal.action == "reuse_seed"  # fallback path
    assert proposal.num_episodes == 500  # 200 * 1.6^2 = 512 -> capped at 500


def test_refine_failure_keeps_scored_winner(tmp_path):
    llm = SequencedLLM([
        json.dumps([{"action": "reuse_seed", "dataset": "org/seed",
                     "num_episodes": 250, "notes": "solid"}]),
        "not json at all",  # refine garbage (single candidate -> no score stage)
    ])
    proposal = _proposer(llm, tmp_path).propose(0, None)
    # single candidate skips scoring; refine failing must not lose the winner
    assert proposal.action == "reuse_seed"
    assert proposal.num_episodes == 250
