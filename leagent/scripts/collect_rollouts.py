"""Collect success-filtered policy rollouts into a LeRobotDataset (M1, DESIGN.md §3.5).

The DexFlyWheel primitive: run a (blessed) policy in simulation, keep only the
successful episodes, and write them as LeRobotDataset v3 episodes the next
cycle can train on. Stepping, processors, and success detection reuse
lerobot's own ``rollout()`` so behavior matches ``lerobot-eval`` exactly.

Run as a subprocess (the Improvement Agent wraps this):

    python -m leagent.scripts.collect_rollouts \\
        --policy-path runs/<run>/cycle_0/train/checkpoints/last/pretrained_model \\
        --env-type pusht --task PushT-v0 --episodes 20 \\
        --out rollouts/<run>/cycle_0 --repo-id local/rollouts-cycle0 --device cuda

The LAST stdout line is a JSON summary: {"kept": .., "attempts": .., ...}.
"""

from __future__ import annotations

import argparse
import json
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--env-type", required=True, help="e.g. pusht, libero")
    parser.add_argument("--task", required=True, help="e.g. PushT-v0, libero_spatial")
    parser.add_argument("--episodes", type=int, default=20, help="successful episodes to keep")
    parser.add_argument("--max-attempts", type=int, default=None,
                        help="rollout budget (default: episodes * 5)")
    parser.add_argument("--out", required=True, help="dataset root directory")
    parser.add_argument("--repo-id", required=True, help="dataset repo id (local unless pushed)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=None, help="default: env config fps")
    parser.add_argument("--task-text", default=None,
                        help="language instruction stored per frame (default: env/task string)")
    parser.add_argument("--rename-map", default=None, help='JSON, e.g. {"observation.images.image": ...}')
    parser.add_argument("--keep-failures", action="store_true",
                        help="also keep failed episodes (default: success-only)")
    return parser


def first_done_index(done) -> int:
    """Index of the first True in a 1-D cumulative done tensor (else last index)."""
    nonzero = done.nonzero()
    return int(nonzero[0, 0]) if len(nonzero) else int(done.shape[0] - 1)


def chw_float_to_hwc_uint8(img):
    """Invert lerobot's preprocess_observation image conversion for storage."""
    return (img.clamp(0.0, 1.0) * 255.0).round().to("cpu").byte().permute(1, 2, 0).numpy()


def features_from_rollout(observations: dict, action) -> dict:
    """Build a LeRobotDataset feature spec from one rollout's tensors."""
    features: dict[str, dict] = {}
    for key, value in observations.items():
        sample = value[0, 0]
        if sample.ndim == 3:  # image, stored preprocessed as (c, h, w) float
            c, h, w = sample.shape
            features[key] = {"dtype": "video", "shape": (h, w, c),
                             "names": ["height", "width", "channels"]}
        else:
            features[key] = {"dtype": "float32", "shape": tuple(sample.shape), "names": None}
    features["action"] = {"dtype": "float32", "shape": tuple(action[0, 0].shape), "names": None}
    return features


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    max_attempts = args.max_attempts or args.episodes * 5

    from pathlib import Path

    if not Path(args.policy_path).exists():
        # without this, lerobot falls back to treating the path as a Hub repo
        # id and dies with a confusing HFValidationError
        parser.error(f"--policy-path does not exist locally: {args.policy_path}")

    import torch  # heavy imports deferred so --help stays fast

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import rollout

    rename_map = json.loads(args.rename_map) if args.rename_map else None

    env_cfg = make_env_config(args.env_type, task=args.task)
    envs = [
        (suite, task_id, env)
        for suite, by_task in make_env(env_cfg, n_envs=1).items()
        for task_id, env in by_task.items()
    ]

    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    policy_cfg.device = args.device
    policy = make_policy(policy_cfg, env_cfg=env_cfg, rename_map=rename_map)
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": args.device},
        "rename_observations_processor": {"rename_map": rename_map or {}},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    fps = args.fps or int(getattr(env_cfg, "fps", 10))
    dataset: LeRobotDataset | None = None
    kept = attempts = 0

    while kept < args.episodes and attempts < max_attempts:
        suite, task_id, env = envs[attempts % len(envs)]
        with torch.inference_mode():
            data = rollout(
                env, policy, env_preprocessor, env_postprocessor, preprocessor,
                postprocessor, seeds=[args.seed + attempts], return_observations=True,
            )
        attempts += 1
        success = bool(data["success"][0].any())
        print(f"attempt {attempts}: {suite}/{task_id} success={success}", flush=True)
        if not success and not args.keep_failures:
            continue

        observations = data["observation"]
        if dataset is None:
            dataset = LeRobotDataset.create(
                repo_id=args.repo_id, fps=fps, root=args.out,
                features=features_from_rollout(observations, data["action"]),
            )
        n_frames = first_done_index(data["done"][0]) + 1
        task_text = args.task_text or f"{args.env_type}:{suite}/{task_id}"
        for i in range(n_frames):
            frame = {"task": task_text, "action": data["action"][0, i].to("cpu").numpy()}
            for key, value in observations.items():
                sample = value[0, i]
                frame[key] = (
                    chw_float_to_hwc_uint8(sample) if sample.ndim == 3
                    else sample.to("cpu").numpy()
                )
            dataset.add_frame(frame)
        dataset.save_episode()
        kept += 1

    if dataset is not None:
        dataset.finalize()

    print(json.dumps({
        "kept": kept, "attempts": attempts,
        "success_rate": kept / attempts if attempts else 0.0,
        "repo_id": args.repo_id, "root": args.out, "fps": fps,
    }))
    return 0 if (kept > 0 or args.episodes == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
