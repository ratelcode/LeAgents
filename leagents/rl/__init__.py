"""Residual RL for the self-improvement flywheel (Phase 2, DexFlyWheel step 2).

Design: docs/DESIGN_phase2_residual_rl.md. A residual policy is trained on top
of the FROZEN base (SmolVLA) so the *composed* policy reaches success on init
states the base fails; rolling it out harvests genuinely new-coverage data.

CVE-2026-25874 boundary: nothing in this package may import lerobot's gRPC/
pickle transport or its actor/learner (see ``safety``). The residual-SAC loop
reuses only lerobot's transport-free SAC *math* in a single process.

P2.0 (this module, no GPU / no lerobot needed, fully unit-tested):
- ``composition.ComposedPolicy`` — per-step alpha*tanh residual over a cached
  base chunk, with the progressive exploration ramp.
- ``safety.assert_cve_safe`` — runtime guard that the CVE transport/RL modules
  are never imported.
- ``config.ResidualRLConfig`` — the config surface (wired into ImproveConfig).

Pending (P2.1+, need lerobot + a GPU + LIBERO — see the design's phased plan):
the SAC actor/critic wrappers over ``lerobot.policies.sac``, the single-process
``residual_sac_train`` loop, the LIBERO env adapter (sparse reward, init-state
train/harvest split), and the ``--residual-path`` collector flag.

Note: ``config`` and ``safety`` import only pydantic/stdlib, so ``leagents.config``
(which pulls in ``ResidualRLConfig``) stays torch-free. ``ComposedPolicy`` needs
torch and is imported lazily, so importing this package never requires torch.
"""

from leagents.rl.config import ResidualRLConfig
from leagents.rl.safety import (
    FORBIDDEN_MODULE_PREFIXES,
    CveSafetyError,
    assert_cve_safe,
)

__all__ = [
    "ComposedPolicy",  # lazy (needs torch) — see __getattr__
    "ResidualRLConfig",
    "FORBIDDEN_MODULE_PREFIXES",
    "CveSafetyError",
    "assert_cve_safe",
]


def __getattr__(name: str):  # PEP 562 — keep torch out of the import path
    if name == "ComposedPolicy":
        from leagents.rl.composition import ComposedPolicy

        return ComposedPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
