import subprocess
import sys

import pytest

from leagents.orchestrator.constitution import Constitution
from leagents.rl.safety import (
    FORBIDDEN_MODULE_PREFIXES,
    CveSafetyError,
    assert_cve_safe,
    forbidden_loaded,
)


def test_forbidden_loaded_matches_module_and_submodules_only():
    mods = [
        "lerobot.policies.sac.modeling_sac",  # allowed (transport-free math)
        "lerobot.rl.buffer",                  # allowed
        "lerobot.transport",                  # denied (exact)
        "lerobot.transport.utils",            # denied (submodule)
        "lerobot.rl.learner",                 # denied
        "lerobot.rl.learner_service",         # denied (distinct from learner)
        "lerobot.rl.actor",                   # denied
        "numpy",
    ]
    assert forbidden_loaded(mods) == [
        "lerobot.rl.actor",
        "lerobot.rl.learner",
        "lerobot.rl.learner_service",
        "lerobot.transport",
        "lerobot.transport.utils",
    ]


def test_assert_cve_safe_raises_only_on_forbidden():
    with pytest.raises(CveSafetyError):
        assert_cve_safe(["lerobot.rl.actor"])
    assert_cve_safe(["lerobot.policies.sac", "lerobot.rl.buffer", "numpy"])  # no raise


def test_importing_leagents_rl_pulls_in_no_cve_module():
    # run in a clean interpreter so the result is independent of what other
    # tests imported into this process's sys.modules
    code = (
        "import sys, leagents.rl; "
        "from leagents.rl.safety import forbidden_loaded; "
        "hits = forbidden_loaded(sys.modules); "
        "sys.stderr.write(repr(hits)); "
        "sys.exit(1 if hits else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"leagents.rl imported CVE modules: {r.stderr}"


def test_importing_leagents_config_stays_torch_free():
    # leagents.config pulls in ResidualRLConfig; it must not drag in torch
    # (the no-lerobot/no-torch test environment imports config everywhere)
    code = (
        "import sys, leagents.config; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, "importing leagents.config imported torch"


def test_constitution_forbidden_imports_covers_safety_and_flags_loaded():
    c = Constitution.from_yaml("configs/constitution.yaml")
    listed = c.rules.get("forbidden_imports", [])
    assert set(FORBIDDEN_MODULE_PREFIXES) <= set(listed)
    denied = c.check_imports(["lerobot.transport.utils"])
    assert not denied.allowed and denied.rule == "forbidden_imports"
    assert c.check_imports(["lerobot.policies.sac", "numpy"]).allowed
