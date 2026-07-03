"""Shared subprocess runner for agents that wrap LeRobot CLIs.

Runners are injectable so tests (and --dry-run) exercise the full loop
without lerobot installed or a GPU present.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


@dataclass
class RunResult:
    cmd: list[str]
    exit_code: int
    duration_s: float
    log_path: Path | None = None


Runner = Callable[[Sequence[str], Path], RunResult]


def subprocess_runner(cmd: Sequence[str], log_path: Path) -> RunResult:
    """Run a command, teeing stdout+stderr to a log file.

    Training jobs stream output for hours — never buffer it in memory.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with log_path.open("w") as log:
        proc = subprocess.run(list(cmd), stdout=log, stderr=subprocess.STDOUT)
    return RunResult(list(cmd), proc.returncode, time.monotonic() - start, log_path)


def dry_runner(cmd: Sequence[str], log_path: Path) -> RunResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("DRY RUN: " + " ".join(cmd) + "\n")
    return RunResult(list(cmd), 0, 0.0, log_path)
