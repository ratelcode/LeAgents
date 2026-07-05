"""CVE-2026-25874 import guard for the residual-RL code (design §3.1).

The vulnerability is an unauthenticated pickle RCE over lerobot's plaintext
gRPC (``lerobot/transport/utils.py`` ``pickle.load``), reached by the RL
actor/learner services and the async-inference server — fixed only in
lerobot >= 0.6.0. Phase 2 reuses lerobot's transport-FREE SAC math in a single
process and must never import that surface.

This is a code-level, always-on guard (not config-dependent — a missing or
edited constitution must not be able to disable the CVE check). The
constitution *also* lists these modules (``forbidden_imports``) for audit
visibility, and a test asserts the two lists agree.
"""

from __future__ import annotations

import sys
from typing import Iterable

# Prefixes that must never appear in ``sys.modules`` while residual RL runs.
FORBIDDEN_MODULE_PREFIXES: tuple[str, ...] = (
    "lerobot.rl.actor",
    "lerobot.rl.learner",
    "lerobot.rl.learner_service",
    "lerobot.transport",
)


class CveSafetyError(RuntimeError):
    pass


def forbidden_loaded(modules: Iterable[str]) -> list[str]:
    """The loaded module names that match a forbidden prefix. Pure function."""
    hits = []
    for name in modules:
        for prefix in FORBIDDEN_MODULE_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                hits.append(name)
                break
    return sorted(hits)


def assert_cve_safe(modules: Iterable[str] | None = None) -> None:
    """Raise if any CVE transport/RL-service module is imported. Call at the
    start of the residual-SAC loop (and it is asserted in tests)."""
    loaded = list(modules) if modules is not None else list(sys.modules)
    hits = forbidden_loaded(loaded)
    if hits:
        raise CveSafetyError(
            "CVE-2026-25874: residual RL must not import lerobot's gRPC/pickle "
            f"transport or RL services, but these are loaded: {hits}. Reuse only "
            "the transport-free SAC math (lerobot.policies.sac, lerobot.rl.buffer)."
        )
