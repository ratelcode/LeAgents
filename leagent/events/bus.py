"""Append-only JSONL event log + in-process subscribers.

Every agent emits structured events here; the dashboard (M2) reads the
JSONL file, and subscribers let the CLI and the loop controller tail
events live (e.g. GPU-hour budget accounting).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class Event:
    run_id: str
    stage: str  # orchestrator | collect | train | eval | decide
    kind: str  # job_started | job_finished | decision | constitution_denied | warning | ...
    cycle: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class EventBus:
    def __init__(self, log_path: Path):
        self._log_path = log_path
        self._subscribers: list[Callable[[Event], None]] = []
        log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def log_path(self) -> Path:
        return self._log_path

    def subscribe(self, fn: Callable[[Event], None]) -> None:
        self._subscribers.append(fn)

    def emit(self, event: Event) -> None:
        with self._log_path.open("a") as f:
            f.write(json.dumps(asdict(event)) + "\n")
        for fn in self._subscribers:
            fn(event)
