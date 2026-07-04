"""Constitution safety gate (AutoRT-style, DESIGN.md §3.1).

M0 scope: static rule checks over agent commands and job parameters.
M0 is sim-only, so real-hardware entrypoints and the async-inference
gRPC path (CVE-2026-25874, DESIGN.md §6) are denied outright.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml


@dataclass
class Verdict:
    allowed: bool
    rule: str | None = None
    reason: str | None = None


class ConstitutionError(Exception):
    def __init__(self, verdict: Verdict):
        self.verdict = verdict
        super().__init__(f"constitution denied [{verdict.rule}]: {verdict.reason}")


class Constitution:
    def __init__(self, rules: dict[str, Any]):
        self.rules = rules

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Constitution":
        return cls(yaml.safe_load(Path(path).read_text()) or {})

    def check_command(self, cmd: Sequence[str]) -> Verdict:
        if cmd and cmd[0] in self.rules.get("forbidden_commands", []):
            return Verdict(False, "forbidden_commands", f"{cmd[0]!r} is not allowed in M0")
        for token in self.rules.get("forbidden_args", []):
            if any(arg.startswith(token) for arg in cmd):
                return Verdict(False, "forbidden_args", f"{token!r} is not allowed in M0")
        return Verdict(True)

    def check_train(self, steps: int) -> Verdict:
        max_steps = self.rules.get("max_train_steps")
        if max_steps is not None and steps > max_steps:
            return Verdict(False, "max_train_steps", f"{steps} > {max_steps}")
        return Verdict(True)

    def check_eval(self, env_type: str, n_episodes: int) -> Verdict:
        allowed = self.rules.get("allowed_env_types")
        if allowed and env_type not in allowed:
            return Verdict(False, "allowed_env_types", f"{env_type!r} not in {allowed}")
        max_eps = self.rules.get("max_eval_episodes")
        if max_eps is not None and n_episodes > max_eps:
            return Verdict(False, "max_eval_episodes", f"{n_episodes} > {max_eps}")
        return Verdict(True)
