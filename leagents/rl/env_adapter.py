"""LIBERO episode logic for residual RL — the verified env quirks, as pure
functions (design §2.2, §3.2).

The single most bug-prone part of an off-policy loop on LIBERO is getting the
terminal/bootstrap bookkeeping right, because LIBERO's env has three verified
traps:
- ``reward = 1.0`` iff ``_check_success()`` and ``info["is_success"]`` is set
  EVERY step (not just terminal);
- ``truncated`` is always ``False`` from the inner env — the 280-step
  ``libero_spatial`` limit is the CALLER's to enforce;
- on its own termination the env AUTO-RESETS inside ``step()`` and advances the
  init-state pointer, so the post-terminal ``next_state`` is not the terminal
  state of the episode you were in — the loop must drive resets explicitly and
  never step a done env again.

So we ignore the inner env's terminated/truncated and classify each step here.
Nothing in this module imports lerobot (the real env is injected), so the logic
is unit-tested on CPU with a fake env.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Protocol


@dataclass(frozen=True)
class StepOutcome:
    reward: float
    success: bool
    terminated: bool  # episode ended by task success (true terminal — do NOT bootstrap)
    truncated: bool   # episode ended by our time limit (bootstrap: value continues)
    done: bool
    bootstrap: bool   # include V(s') in the TD target?


def classify_step(reward: float, info: dict, step_index: int, max_steps: int) -> StepOutcome:
    """Classify one env step. ``step_index`` is 0-based within the episode;
    ``max_steps`` is the caller-owned limit (280 for libero_spatial). Success is
    read from ``info['is_success']`` (authoritative, per-step), falling back to
    ``reward > 0`` (reward is 1.0 iff success). Pure function."""
    success = bool(info.get("is_success", reward > 0.0))
    terminated = success  # the ONLY true terminal (the inner env's flag is unreliable)
    truncated = (not success) and (step_index + 1) >= max_steps
    done = terminated or truncated
    # bootstrap on a time-limit cut (the value function continues past it) but
    # never on success (the episode genuinely ended) — the standard fix for
    # time-limited MDPs, and correct around LIBERO's auto-reset.
    bootstrap = not terminated
    return StepOutcome(float(reward), success, terminated, truncated, done, bootstrap)


def init_state_split(n_states: int, train_span: tuple[int, int],
                     harvest_span: tuple[int, int]) -> tuple[list[int], list[int]]:
    """Train and harvest init-state index lists from half-open [start, end)
    spans, asserting they are disjoint and within range (design §5.1 — the
    residual must not be measured on the states it trained on). Pure function."""
    train = list(range(*train_span))
    harvest = list(range(*harvest_span))
    for name, idx in (("train", train), ("harvest", harvest)):
        if not idx or idx[0] < 0 or idx[-1] >= n_states:
            raise ValueError(f"{name}_span {train_span if name == 'train' else harvest_span} "
                             f"out of range for {n_states} init states")
    if set(train) & set(harvest):
        raise ValueError("train and harvest init states must be disjoint")
    return train, harvest


@dataclass
class Transition:
    """One stored transition (design §1.4): ``a_base`` is kept alongside so the
    critic/actor can condition on it for every sampled transition, including
    demo transitions (precomputed offline)."""
    state: object
    a_base: object
    a_exec: object
    reward: float
    next_state: object
    done: bool
    bootstrap: bool


class _Env(Protocol):
    def reset(self, init_state: int) -> object: ...  # returns obs
    def step(self, action: object) -> tuple[object, float, dict]: ...  # (obs, reward, info)


class _Policy(Protocol):
    def reset(self) -> None: ...
    def select_action(self, obs: object, state: object, *, global_step: int,
                      explore: bool, rng: object) -> tuple[object, object]: ...


def _cycle(items: list[int]) -> Iterator[int]:
    while True:
        yield from items


class ResidualRollout:
    """Drive a ComposedPolicy through an env, collecting transitions with
    correct bootstrap masks and resetting explicitly on episode end (cycling
    the given init states). Env/policy are injected, so this is tested with a
    fake env + fake policy; the real driver wraps lerobot's LIBERO env and
    extracts the 8-d proprio state (a lazy, lerobot-only concern)."""

    def __init__(self, env: _Env, extract_state: Callable[[object], object], max_steps: int):
        self.env = env
        self.extract_state = extract_state
        self.max_steps = max_steps

    def collect(self, policy: _Policy, n_steps: int, init_states: list[int], *,
                global_step_start: int = 0, rng: object = None) -> list[Transition]:
        if not init_states:
            raise ValueError("init_states must be non-empty")
        transitions: list[Transition] = []
        states = _cycle(init_states)
        obs = self.env.reset(next(states))
        policy.reset()
        gs = global_step_start
        ep_step = 0
        for _ in range(n_steps):
            state = self.extract_state(obs)
            a_exec, a_base = policy.select_action(
                obs, state, global_step=gs, explore=True, rng=rng)
            next_obs, reward, info = self.env.step(a_exec)
            outcome = classify_step(reward, info, ep_step, self.max_steps)
            transitions.append(Transition(
                state=state, a_base=a_base, a_exec=a_exec, reward=outcome.reward,
                next_state=self.extract_state(next_obs), done=outcome.done,
                bootstrap=outcome.bootstrap))
            gs += 1
            ep_step += 1
            if outcome.done:
                obs = self.env.reset(next(states))  # explicit reset — never step a done env
                policy.reset()
                ep_step = 0
            else:
                obs = next_obs
        return transitions
