from leagent.config import ThresholdConfig
from leagent.orchestrator.decision import Decision, decide

T = ThresholdConfig(
    promote_delta=0.05, regression_delta=0.05, plateau_epsilon=0.01, plateau_cycles=2
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


def test_plateau_needs_enough_cycles():
    # only one near-baseline sample so far -> keep iterating
    assert decide(0.505, 0.50, [], T) is Decision.ITERATE


def test_noisy_history_does_not_plateau():
    # last window contains a non-plateau score
    assert decide(0.505, 0.50, [0.53], T) is Decision.ITERATE
