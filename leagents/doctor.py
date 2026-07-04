"""Environment diagnosis (`leagents doctor`) and self-repair (`leagents setup`).

Every check here encodes a failure actually hit while standing up the stack
on a real machine (see DESIGN.md §3.3/§6 and knowledge/): EGL headers,
CMake 4.x vs egl-probe, libero's interactive first import, CUDA/torch
matching, headless-sim env vars. `doctor` diagnoses and prints exact fixes;
`setup` applies the ones that are safe to automate.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

OK, WARN, FAIL = "ok", "warn", "fail"
_ICONS = {OK: "✓", WARN: "!", FAIL: "✗"}

APT_EGL = "sudo apt install -y libegl1-mesa-dev libgles2-mesa-dev"
CMAKE_ENV = "CMAKE_POLICY_VERSION_MINIMUM=3.5"
LIBERO_CONFIG = Path.home() / ".libero" / "config.yaml"


@dataclass
class Check:
    name: str
    level: str  # ok | warn | fail
    detail: str
    fix: str | None = None


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_python() -> Check:
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.version_info >= (3, 10):
        return Check("python", OK, f"{version}")
    return Check("python", FAIL, f"{version} < 3.10", "install Python >= 3.10")


def check_torch() -> Check:
    if not _has_module("torch"):
        return Check("torch / GPU", WARN, "torch not installed (dry-run/tests still work)",
                     'pip install "leagents[lerobot]"')
    import torch

    if torch.cuda.is_available():
        return Check("torch / GPU", OK,
                     f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    return Check("torch / GPU", WARN, f"torch {torch.__version__}, CUDA not available",
                 "training/eval will be CPU-only; check the NVIDIA driver")


def check_lerobot() -> Check:
    if not _has_module("lerobot"):
        return Check("lerobot", WARN, "not installed (needed for real runs)",
                     f'{CMAKE_ENV} pip install "leagents[lerobot]"')
    import lerobot

    version = getattr(lerobot, "__version__", "?")
    if version.startswith(("0.4.", "0.5.")):
        return Check("lerobot", OK,
                     f"{version} — keep the async-inference path disabled "
                     "(CVE-2026-25874, fixed in >= 0.6.0; the constitution denies it)")
    return Check("lerobot", OK, version)


def check_libero(include_dir: Path = Path("/usr/include/EGL")) -> Check:
    """LIBERO import works, or predict whether installing it would succeed."""
    if _has_module("libero"):
        return Check("libero", OK, "installed")
    problems, fixes = [], []
    if not (include_dir / "egl.h").exists():
        problems.append("system EGL headers missing (egl-probe cannot build)")
        fixes.append(APT_EGL)
    cmake = shutil.which("cmake")
    if cmake:
        try:
            out = subprocess.run([cmake, "--version"], capture_output=True, text=True,
                                 timeout=10).stdout
            major = int(out.split()[2].split(".")[0])
            if major >= 4:
                problems.append("CMake >= 4 rejects egl-probe's old CMakeLists")
                fixes.append(f"install with {CMAKE_ENV} pip install \"leagents[lerobot]\"")
        except Exception:
            pass
    if problems:
        return Check("libero", FAIL, "; ".join(problems), " && ".join(fixes))
    return Check("libero", WARN, "not installed",
                 f'{CMAKE_ENV} pip install "leagents[lerobot]"')


def check_libero_config(config_path: Path = LIBERO_CONFIG) -> Check:
    if not _has_module("libero"):
        return Check("libero config", WARN, "skipped (libero not installed)")
    if config_path.exists():
        return Check("libero config", OK, str(config_path))
    return Check("libero config", FAIL,
                 "missing — libero's FIRST import blocks on an interactive prompt",
                 "leagents setup  (runs the non-interactive init)")


def check_headless_env(env: dict | None = None) -> Check:
    env = env if env is not None else dict(os.environ)
    missing = [f"{k}={v}" for k, v in (("MUJOCO_GL", "egl"), ("SDL_VIDEODRIVER", "dummy"))
               if env.get(k) != v]
    if not missing:
        return Check("headless sim env", OK, "MUJOCO_GL=egl, SDL_VIDEODRIVER=dummy")
    return Check("headless sim env", WARN, f"not set: {', '.join(missing)}",
                 "export " + " ".join(missing) + "  (needed for eval without a display)")


def check_disk(path: Path = Path.home(), min_free_gb: float = 30.0) -> Check:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / 1e9
    if free_gb >= min_free_gb:
        return Check("disk space", OK, f"{free_gb:.0f} GB free")
    return Check("disk space", WARN,
                 f"{free_gb:.0f} GB free — LIBERO episode shards are large",
                 "free space or set HF_LEROBOT_HOME to a bigger disk")


def check_llm_keys(env: dict | None = None) -> Check:
    env = env if env is not None else dict(os.environ)
    providers = [name for name, keys in (
        ("gemini", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
        ("anthropic", ("ANTHROPIC_API_KEY",)),
        ("openai", ("OPENAI_API_KEY",)),
    ) if any(env.get(k) for k in keys)]
    if providers:
        return Check("llm keys", OK, f"usable providers: {', '.join(providers)} "
                                     "(values never printed)")
    return Check("llm keys", WARN, "no provider keys in env — deterministic fallbacks only",
                 "source your .env (see .env.example); local Ollama/vLLM needs no key")


ALL_CHECKS = [check_python, check_torch, check_lerobot, check_libero,
              check_libero_config, check_headless_env, check_disk, check_llm_keys]


def run_doctor(checks=None) -> int:
    results = [check() for check in (checks or ALL_CHECKS)]
    width = max(len(r.name) for r in results)
    for r in results:
        print(f"[{_ICONS[r.level]}] {r.name.ljust(width)}  {r.detail}")
        if r.fix and r.level != OK:
            print(f"    {'fix:'.ljust(width)}  {r.fix}")
    fails = [r for r in results if r.level == FAIL]
    warns = [r for r in results if r.level == WARN]
    print(f"\n{len(results) - len(fails) - len(warns)} ok, "
          f"{len(warns)} warnings, {len(fails)} failures")
    return 1 if fails else 0


def run_setup(install_lerobot: bool = False, env_example: Path = Path(".env.example"),
              env_file: Path = Path(".env")) -> int:
    """Apply the fixes that are safe to automate; print the rest."""
    actions: list[str] = []

    if install_lerobot and not _has_module("libero"):
        cmd = [sys.executable, "-m", "pip", "install", "leagents[lerobot]"]
        print(f"installing lerobot extras with {CMAKE_ENV} (this is large) ...")
        result = subprocess.run(cmd, env={**os.environ,
                                          "CMAKE_POLICY_VERSION_MINIMUM": "3.5"})
        if result.returncode != 0:
            print(f"install failed — run the doctor and check EGL headers ({APT_EGL})")
            return 1
        actions.append("installed leagents[lerobot]")

    if _has_module("libero") and not LIBERO_CONFIG.exists():
        # libero prompts on first import; feed it the default answer once
        result = subprocess.run(
            [sys.executable, "-c", "import libero.libero"],
            input="N\n", text=True, capture_output=True,
        )
        if LIBERO_CONFIG.exists():
            actions.append(f"initialized {LIBERO_CONFIG}")
        else:
            print(f"libero init failed:\n{result.stderr[-500:]}")
            return 1

    if env_example.exists() and not env_file.exists():
        env_file.write_text(env_example.read_text())
        actions.append(f"created {env_file} from {env_example} — fill in your keys")

    for action in actions or ["nothing to fix"]:
        print(f"• {action}")
    print("\nverify with: leagents doctor")
    print("then smoke-test the pipeline: leagents run -c configs/smoke_pusht.yaml")
    return 0
