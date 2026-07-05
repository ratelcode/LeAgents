"""Train a residual policy over a frozen base on one LIBERO task (P2.1).

The single-process, CVE-safe residual-SAC entrypoint (design §2, §3): trains on
``train_init_states``, reports the inner-gate metric (deterministic composed vs
pure base success) on the DISJOINT ``harvest_init_states``. The residual is a
data-generator artifact for the harvest (design §4.2) — it is saved to ``--out``
for ``collect_rollouts --residual-path`` (P2.3), never promoted.

Smoke (a few minutes of GPU — proves plumbing, not learning):

    python -m leagents.scripts.train_residual \\
        --policy-path runs/<run>/cycle_N/train/checkpoints/last/pretrained_model \\
        --task-id 6 --env-steps 450 --warmup-env-steps 60 --critic-burnin-updates 40 \\
        --steps-per-iter 150 --exploration-ramp-steps 200 \\
        --train-span 0 60 --harvest-span 60 64 --out /tmp/residual_smoke

Full go/no-go (P2.1: ~100-150k env steps, 10-20 GPU-h — design §6):
same command with the ResidualRLConfig defaults (drop the overrides).

The LAST stdout line is a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True, help="frozen base checkpoint dir")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="dir for residual weights + summary json")
    # budget / cadence overrides (defaults = ResidualRLConfig)
    parser.add_argument("--env-steps", type=int, default=None)
    parser.add_argument("--warmup-env-steps", type=int, default=None)
    parser.add_argument("--critic-burnin-updates", type=int, default=None)
    parser.add_argument("--steps-per-iter", type=int, default=None)
    parser.add_argument("--exploration-ramp-steps", type=int, default=None)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--utd", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-span", type=int, nargs=2, default=None, metavar=("LO", "HI"))
    parser.add_argument("--harvest-span", type=int, nargs=2, default=None, metavar=("LO", "HI"))
    parser.add_argument("--no-gate-eval", action="store_true",
                        help="skip the held-out composed-vs-base evaluation")
    # RLPD demo anchor (optional): a local LeRobotDataset with episodes of this task
    parser.add_argument("--demo-root", default=None)
    parser.add_argument("--demo-repo-id", default=None)
    parser.add_argument("--demo-task-text", default=None,
                        help="task string as stored in the demo dataset "
                             "(default: 'libero:<suite>/<task-id>')")
    parser.add_argument("--demo-max-episodes", type=int, default=None)
    parser.add_argument("--rename-map", default=None,
                        help='JSON; default: the libero camera1/camera2 fix')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from leagents.rl.safety import assert_cve_safe

    assert_cve_safe()  # before AND after the lazy lerobot imports below

    import torch  # noqa: F401  (heavy imports deferred so --help stays fast)

    from leagents.rl.config import ResidualRLConfig
    from leagents.rl.libero_driver import (
        LiberoResidualEnv,
        SmolVLABase,
        extract_state,
        precompute_demo_a_base,
    )
    from leagents.rl.train import residual_sac_train

    overrides: dict = {"enabled": True}
    for field_name, value in (
        ("env_steps", args.env_steps),
        ("warmup_env_steps", args.warmup_env_steps),
        ("critic_burnin_updates", args.critic_burnin_updates),
        ("steps_per_iter", args.steps_per_iter),
        ("exploration_ramp_steps", args.exploration_ramp_steps),
        ("n_action_steps", args.n_action_steps),
        ("alpha", args.alpha),
        ("utd", args.utd),
        ("batch_size", args.batch_size),
        ("train_init_states", tuple(args.train_span) if args.train_span else None),
        ("harvest_init_states", tuple(args.harvest_span) if args.harvest_span else None),
    ):
        if value is not None:
            overrides[field_name] = value
    cfg = ResidualRLConfig(**overrides)

    env = LiberoResidualEnv(args.suite, args.task_id, seed=args.seed)
    rename_map = json.loads(args.rename_map) if args.rename_map else None
    base = SmolVLABase(
        args.policy_path, device=args.device, env_task=args.suite,
        task_text=env.task_description, rename_map=rename_map)

    demo_transitions = None
    if args.demo_root or args.demo_repo_id:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(args.demo_repo_id or "local/demo", root=args.demo_root)
        task_text = args.demo_task_text or f"libero:{args.suite}/{args.task_id}"
        demo_transitions = precompute_demo_a_base(
            base, dataset, task_text,
            n_action_steps=cfg.n_action_steps, max_episodes=args.demo_max_episodes)
        print(f"demo: {len(demo_transitions)} transitions with precomputed a_base", flush=True)

    assert_cve_safe()  # all lerobot surface loaded by now — must still be transport-free

    result = residual_sac_train(
        cfg, base, env, extract_state, demo_transitions,
        device=args.device, evaluate_gate=not args.no_gate_eval, seed=args.seed)

    summary = {
        "task_id": args.task_id,
        "env_steps": result.env_steps,
        "updates": result.updates,
        "wall_clock_s": round(result.wall_clock_s, 1),
        "loss_critic_last": result.metrics["loss_critic"][-1] if result.metrics["loss_critic"] else None,
        "loss_actor_last": result.metrics["loss_actor"][-1] if result.metrics["loss_actor"] else None,
        "base_success": result.base_success,
        "composed_success": result.composed_success,
        "composed_gain": result.composed_gain,
        "gate_passed": result.gate_passed,
    }
    if args.out:
        from pathlib import Path

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"actor": result.residual_state_dict,
             "alpha": cfg.alpha, "n_action_steps": cfg.n_action_steps},
            out / "residual.pt")
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        summary["out"] = str(out)

    env.close()
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
