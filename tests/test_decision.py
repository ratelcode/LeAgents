from leagents.config import ThresholdConfig
from leagents.orchestrator.decision import Decision, decide

T = ThresholdConfig(
    promote_delta=0.05, regression_delta=0.05, plateau_epsilon=0.01, plateau_cycles=2,
    escalate_floor=0.05,
)


def test_first_cycle_promotes_to_establish_baseline():
    assert decide(0.1, None, [], T) is Decision.PROMOTE


def test_clear_improvement_promotes():
    assert decide(0.60, 0.50, [], T) is Decision.PROMOTE


def test_regression_rolls_back():
    assert decide(0.40, 0.50, [], T) is Decision.ROLLBACK


def test_small_improvement_iterates():
    assert decide(0.53, 0.50, [], T) is Decision.ITERATE


def test_plateau_escalates_policy():
    assert decide(0.505, 0.50, [0.502], T) is Decision.ESCALATE


def test_zero_plateau_iterates_instead_of_escalating():
    # observed in the 2026-07-04 mini-M0 run: an all-zero plateau means
    # under-training, not a policy ceiling — do not burn GPU on a bigger model
    assert decide(0.0, 0.0, [0.0], T) is Decision.ITERATE


def test_escalate_floor_zero_disables_guard():
    t = ThresholdConfig(escalate_floor=0.0)
    assert decide(0.0, 0.0, [0.0], t) is Decision.ESCALATE


def test_plateau_needs_enough_cycles():
    # only one near-baseline sample so far -> keep iterating
    assert decide(0.505, 0.50, [], T) is Decision.ITERATE


def test_noisy_history_does_not_plateau():
    # last window contains a non-plateau score
    assert decide(0.505, 0.50, [0.53], T) is Decision.ITERATE


def test_proposer_episode_growth():
    from leagents.orchestrator import DeterministicProposer

    proposer = DeterministicProposer("org/seed", initial_episodes=8, growth=2.0,
                                     max_episodes=20)
    assert proposer.propose(0, None).num_episodes == 8
    assert proposer.propose(1, None).num_episodes == 16
    assert proposer.propose(2, None).num_episodes == 20  # capped


def test_proposer_full_dataset_by_default():
    from leagents.orchestrator import DeterministicProposer

    assert DeterministicProposer("org/seed").propose(0, None).num_episodes is None
