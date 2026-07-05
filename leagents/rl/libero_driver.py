"""Real-LIBERO driver for residual RL (design §2.2, §3.2) — lazy lerobot imports.

Wraps one LIBERO task env behind ``ResidualRollout``'s injected-env interface
(``reset(init_state) -> obs``, ``step(action) -> (obs, reward, info)``) and
loads the frozen SmolVLA base as a chunk-predicting bundle for ``ComposedPolicy``.

Verified env quirks handled here (the terminal/bootstrap logic itself lives in
``env_adapter.classify_step`` and is NOT re-implemented):
- ``MUJOCO_GL=egl`` must be set before any mujoco import;
- ``~/.libero/config.yaml`` on this machine may point at the stale ``leagent``
  (no "s") venv — ``resolve_libero_config`` rewrites it against the installed
  ``libero`` package before the first libero import;
- exact init-state control: lerobot's ``LiberoEnv.reset`` consumes
  ``init_state_id`` (advancing it by ``_reset_stride``); we pin the stride to 0
  and set the id explicitly per reset, so the env's own reset flow (incl. the
  10 settle steps and controller mode) is reused unchanged;
- images render 256×256; the LiberoProcessorStep flip + 8-d state flatten and
  the ``--rename-map`` camera fix are applied through the same processor
  pipelines lerobot-eval uses (proven by Phase-1 eval/harvest).

CVE-2026-25874: imports only ``lerobot.envs.*``, ``lerobot.policies.*`` and
``lerobot.processor`` — all verified transport-free. Never ``lerobot.rl.actor/
learner`` or ``lerobot.transport``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Must precede any mujoco/robosuite import chain (design §3.2 quirk c).
os.environ.setdefault("MUJOCO_GL", "egl")

# The camera fix for checkpoints fine-tuned from smolvla_base on the LIBERO
# seed data (knowledge/tasks/libero_spatial.md): env cameras image/image2 →
# checkpoint features camera1/camera2.
DEFAULT_RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}


def resolve_libero_config(config_path: Path | None = None) -> Path:
    """Ensure ``~/.libero/config.yaml`` points at the *installed* libero package.

    The file on this machine may reference the stale ``leagent`` (no "s") venv
    path (design §3.2 quirk a). Rewriting it before the first ``libero.libero``
    import also prevents libero's interactive first-run prompt."""
    import importlib.util

    import yaml

    config_path = config_path or Path.home() / ".libero" / "config.yaml"
    spec = importlib.util.find_spec("libero")
    if spec is None or spec.origin is None:
        raise RuntimeError("libero is not installed in this environment")
    pkg_root = Path(spec.origin).parent / "libero"  # .../site-packages/libero/libero

    existing: dict[str, Any] = {}
    if config_path.exists():
        existing = yaml.safe_load(config_path.read_text()) or {}
    if existing and all(Path(p).exists() for p in existing.values()):
        return config_path  # already valid (possibly via the compat symlink)

    resolved = {
        "benchmark_root": str(pkg_root),
        "bddl_files": str(pkg_root / "bddl_files"),
        "init_states": str(pkg_root / "init_files"),
        "assets": str(pkg_root / "assets"),
        "datasets": str(pkg_root.parent / "datasets"),
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(resolved))
    return config_path


class LiberoResidualEnv:
    """One LIBERO task env implementing ``ResidualRollout``'s env protocol.

    ``reset(init_state)`` pins the env to an EXACT init-state index (the lever
    for the train/harvest split, §3.2); ``step`` returns ``(obs, reward, info)``
    with LIBERO's native per-step sparse reward and ``info['is_success']`` —
    terminal/bootstrap classification is the caller's (``classify_step``)."""

    def __init__(self, suite_name: str, task_id: int, *, seed: int = 0):
        resolve_libero_config()
        from libero.libero import benchmark
        from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, LiberoEnv

        suite = benchmark.get_benchmark_dict()[suite_name]()
        self._env = LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            obs_type="pixels_agent_pos",
            init_states=True,
        )
        self._env._reset_stride = 0  # we drive init-state selection explicitly
        self.suite_name = suite_name
        self.task_id = task_id
        self.max_steps = TASK_SUITE_MAX_STEPS[suite_name]
        self.task_description: str = self._env.task_description
        self.n_init_states: int = len(self._env._init_states)
        self._seed = seed

    def reset(self, init_state: int):
        # LiberoEnv indexes init states modulo len(); wrap-around would silently
        # LEAK train states into the harvest split, so fail loudly instead.
        # (Measured: libero_spatial ships 50 init states per task, not the 100
        # the design doc assumed — spans must fit [0, n_init_states).)
        if not 0 <= int(init_state) < self.n_init_states:
            raise ValueError(
                f"init_state {init_state} out of range [0, {self.n_init_states}) for "
                f"{self.suite_name}/{self.task_id} — check train/harvest spans")
        self._env.init_state_id = int(init_state)
        obs, _info = self._env.reset(seed=self._seed)
        return obs

    def step(self, action):
        import numpy as np

        a = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, _terminated, _truncated, info = self._env.step(a)
        # NOTE: on termination LIBERO auto-reset inside step(); classify_step
        # + explicit resets in ResidualRollout make the post-terminal obs unused.
        return obs, float(reward), info

    def close(self) -> None:
        self._env.close()


def extract_state(obs) -> "torch.Tensor":  # noqa: F821 (torch imported lazily)
    """8-d proprio ``[eef_pos(3), eef axis-angle(3), gripper_qpos(2)]`` from a
    raw LIBERO ``pixels_agent_pos`` observation — same quat→axis-angle math as
    lerobot's LiberoProcessorStep (reused, not re-implemented)."""
    import torch

    step = _libero_processor_step()
    rs = obs["robot_state"]
    pos = torch.as_tensor(rs["eef"]["pos"], dtype=torch.float32).reshape(3)
    quat = torch.as_tensor(rs["eef"]["quat"], dtype=torch.float32).reshape(1, 4)
    grip = torch.as_tensor(rs["gripper"]["qpos"], dtype=torch.float32).reshape(2)
    axis_angle = step._quat2axisangle(quat).squeeze(0)
    return torch.cat([pos, axis_angle, grip])


_PROCESSOR_STEP = None


def _libero_processor_step():
    global _PROCESSOR_STEP
    if _PROCESSOR_STEP is None:
        from lerobot.processor.env_processor import LiberoProcessorStep

        _PROCESSOR_STEP = LiberoProcessorStep()
    return _PROCESSOR_STEP


def _batch_np(obs):
    """Add a leading batch axis to every array leaf of a raw LIBERO obs, so the
    single env's obs matches the vector-env layout ``preprocess_observation``
    (and LiberoProcessorStep's (B,4) quat contract) expect."""
    import numpy as np

    if isinstance(obs, dict):
        return {k: _batch_np(v) for k, v in obs.items()}
    return np.asarray(obs)[None, ...]


class SmolVLABase:
    """The frozen base as a chunk predictor for ``ComposedPolicy``.

    Loads the checkpoint exactly the way Phase-1 eval/harvest do (``make_policy``
    with the env config + rename map — the checkpoint's declared features are
    the smolvla_base pretrain's, and this proven path is what reconciles them),
    then exposes ``predict_chunk(raw_obs) -> (chunk_size, 7)`` running the same
    processor chain as ``lerobot_eval.rollout``:
    ``preprocess_observation → task → LiberoProcessorStep → policy pre →
    predict_action_chunk → policy post (unnormalize)``."""

    def __init__(
        self,
        policy_path: str,
        *,
        device: str = "cuda",
        env_task: str = "libero_spatial",
        task_text: str,
        rename_map: dict[str, str] | None = None,
    ):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.factory import make_env_config, make_env_pre_post_processors
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        if not Path(policy_path).exists():
            raise FileNotFoundError(f"base policy path does not exist locally: {policy_path}")
        rename_map = DEFAULT_RENAME_MAP if rename_map is None else rename_map

        env_cfg = make_env_config("libero", task=env_task)
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
        policy_cfg.pretrained_path = policy_path
        policy_cfg.device = device
        self.policy = make_policy(policy_cfg, env_cfg=env_cfg, rename_map=rename_map)
        self.policy.eval()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_path,
            preprocessor_overrides={
                "device_processor": {"device": device},
                "rename_observations_processor": {"rename_map": rename_map},
            },
        )
        self.env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
        self.task_text = task_text
        self.device = device
        self.chunk_size = int(getattr(self.policy.config, "chunk_size", 50))

    def reset(self) -> None:
        self.policy.reset()

    def _policy_batch(self, batch: dict) -> dict:
        batch = self.env_preprocessor(batch)
        return self.preprocessor(batch)

    def predict_chunk(self, raw_obs) -> "torch.Tensor":  # noqa: F821
        """(chunk_size, 7) float32 CPU tensor in the env action space."""
        import torch

        from lerobot.envs.utils import preprocess_observation

        with torch.inference_mode():
            batch = preprocess_observation(_batch_np(raw_obs))
            batch["task"] = [self.task_text]
            batch = self._policy_batch(batch)
            chunk = self.policy.predict_action_chunk(batch)  # (1, chunk, 7), normalized
            chunk = self.postprocessor(chunk.squeeze(0))  # (chunk, 7), env action space
        return chunk.detach().to("cpu", torch.float32).clone()

    def predict_chunks_from_frames(self, frames: list[dict]) -> "torch.Tensor":  # noqa: F821
        """Batched chunk prediction from stored dataset frames (the offline
        ``a_base`` precompute, design §1.4). Frames are dataset-convention
        tensors (images CHW float [0,1], state 8-d) — the same policy
        preprocessor applies (rename step no-ops on absent env keys).
        Returns (B, chunk_size, 7) on CPU."""
        import torch

        with torch.inference_mode():
            keys = [k for k in frames[0] if k.startswith("observation.")]
            batch: dict[str, Any] = {
                k: torch.stack([torch.as_tensor(f[k]) for f in frames]) for k in keys
            }
            batch["task"] = [self.task_text] * len(frames)
            # NOTE: dataset frames must NOT go through the env preprocessor —
            # LiberoProcessorStep's image flip is not idempotent and stored
            # frames are already dataset-convention (flipped once at collection;
            # knowledge/tasks/libero_spatial.md), and the state is already the
            # flat 8-d vector. Only the policy preprocessor applies here.
            batch = self.preprocessor(batch)
            chunks = self.policy.predict_action_chunk(batch)  # (B, chunk, 7)
            chunks = self.postprocessor(chunks)
        return chunks.detach().to("cpu", torch.float32).clone()


def precompute_demo_a_base(
    base: SmolVLABase,
    dataset,
    task_text_in_dataset: str,
    *,
    n_action_steps: int,
    max_episodes: int | None = None,
    anchor_batch: int = 8,
) -> list:
    """One offline frozen-base pass over the demo episodes of one task,
    producing ``env_adapter.Transition``s with ``a_base`` stored per frame
    (design §1.4 / P2.0 bookkeeping) so the RL loop never pays a 450M forward
    per sampled batch.

    ``a_base`` mirrors the ONLINE chunk cadence: one base forward per
    ``n_action_steps`` frames (anchor), frames in between take the cached
    chunk's i-th action. Demo episodes are successes: reward 1.0 / done on the
    last frame (true terminal → no bootstrap), 0.0 elsewhere.
    """
    import torch

    from leagents.rl.env_adapter import Transition

    episodes = [
        (int(row["dataset_from_index"]), int(row["dataset_to_index"]))
        for row in dataset.meta.episodes
        if task_text_in_dataset in row["tasks"]
    ]
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise ValueError(f"no episodes with task {task_text_in_dataset!r} in dataset")

    transitions: list[Transition] = []
    for start, end in episodes:
        frames = [dataset[i] for i in range(start, end)]
        states = [torch.as_tensor(f["observation.state"], dtype=torch.float32) for f in frames]
        actions = [torch.as_tensor(f["action"], dtype=torch.float32) for f in frames]

        anchors = list(range(0, len(frames), n_action_steps))
        chunks: list[torch.Tensor] = []
        for i in range(0, len(anchors), anchor_batch):
            group = [frames[a] for a in anchors[i : i + anchor_batch]]
            chunks.extend(base.predict_chunks_from_frames(group))

        a_base = [chunks[t // n_action_steps][t % n_action_steps] for t in range(len(frames))]

        last = len(frames) - 1
        for t in range(len(frames)):
            terminal = t == last
            transitions.append(Transition(
                state=states[t],
                a_base=a_base[t],
                a_exec=actions[t],  # the expert's executed action
                reward=1.0 if terminal else 0.0,
                next_state=states[t + 1] if not terminal else states[t],
                done=terminal,
                bootstrap=not terminal,
            ))
    return transitions
