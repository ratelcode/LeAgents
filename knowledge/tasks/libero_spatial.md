---
name: libero_spatial
type: task
status: replicated
provenance:
- run: 20260704-121559-m0-libero-smolvla
  cycle: 0
- run: 20260704-121559-m0-libero-smolvla
  cycle: 1
- run: 20260704-121559-m0-libero-smolvla
  cycle: 2
- run: 20260704-183753-m0-libero-scale
  cycle: 0
- run: 20260704-210759-m0-libero-scale
  cycle: 0
- run: 20260704-232610-m0-libero-scale
  cycle: 0
- run: 20260704-232610-m0-libero-scale
  cycle: 1
updated: '2026-07-05'
---

# Task: libero_spatial

## Observations
- 2026-07-04 · run 20260704-121559-m0-libero-smolvla · cycle 0 · libero_spatial · policy smolvla · success 0.0% · decision promote
- 2026-07-04 · run 20260704-121559-m0-libero-smolvla · cycle 1 · libero_spatial · policy smolvla · success 0.0% · decision iterate
- 2026-07-04 · run 20260704-121559-m0-libero-smolvla · cycle 2 · libero_spatial · policy smolvla · success 0.0% · decision iterate
- 2026-07-04 · run 20260704-183753-m0-libero-scale · cycle 0 · libero_spatial · policy smolvla · success 0.0% · decision promote
- 2026-07-04 · run 20260704-210759-m0-libero-scale · cycle 0 · libero_spatial · policy smolvla · success 63.0% · decision promote
- 2026-07-05 · run 20260704-232610-m0-libero-scale · cycle 0 · libero_spatial · policy smolvla · success 63.0% · decision promote
- 2026-07-05 · run 20260704-232610-m0-libero-scale · cycle 1 · libero_spatial · policy smolvla · success 20.0% · decision rollback

## Lessons
- 2026-07-05 · The DexFlyWheel adaptation stage as first configured (2000 steps on the 20-episode success-filtered rollout mix ALONE, run after main train and BEFORE eval) overfit and regressed cycle 1 from 63% → 20% — catastrophic forgetting: adapt loss fell to 0.156 vs the main train's ~0.37, per-task collapse was broad but non-uniform (drift toward the narrow harvested distribution). 20 eps × ~117 frames = 2332 frames; 2000 steps × batch 8 ≈ 7 epochs. The decision function correctly rolled back and protected the 63% baseline. Fix: adapting on rollouts ALONE for many steps is wrong — either disable adaptation and let data growth be the flywheel signal (adapt_steps=0, harvest still accumulates the mix), or adapt as a LIGHT touch (few hundred steps, lower LR) on rollouts BLENDED with real episodes. Never let an adaptation on a tiny self-generated set be the checkpoint eval/promote judges. (provenance: run 20260704-232610, cycle 1)
- 2026-07-04 · A flat 0% across whole runs was NOT under-training: HuggingFaceVLA/libero is suite-ordered and episode prefixes [0..N) are libero_10 episodes — spatial episodes live at indices ~1261–1538. Prefix-based `--dataset.episodes` silently trains on the wrong suite; always select via task filtering against the eval suite (leagents.scripts.select_episodes, data.task_filter). Sparse LIBERO reward means metrics cannot distinguish this from under-training — only task-string inspection can. (provenance: runs 20260704-121559, 20260704-183753)
- 2026-07-04 · Rollout harvesting on LIBERO needs three things train/eval already knew: (1) the `--rename-map` must be plumbed to EVERY stage that loads the policy — the collector without it dies in make_policy's feature check; (2) LIBERO raw observations nest `observation.robot_state` as a dict-of-dicts, and lerobot's `rollout(return_observations=True)` torch.stack()s appended observations → TypeError; fold the env-level LiberoProcessorStep into preprocess_observation for the stored frames; (3) that env step FLIPS images (H and W) and is not idempotent — the policy path must see exactly one application, so hand rollout an identity env step that only rebuilds transition scaffolding. A failed harvest must never kill the run (loop emits improve_failed and continues). (provenance: run 20260704-210759, cycle 0, 63% then ImproveError)
- 2026-07-04 · HuggingFaceVLA/libero's `meta.episodes` `data/file_index` mapping is STALE relative to its actual 377-file hub layout (meta says episode 137 → file-009; the hub's file-055 holds it, row counts matching the episode lengths exactly). Any shard fetch that trusts the mapping — including lerobot's own partial download — pulls the wrong files, and `--dataset.episodes` training dies with "Instruction 'train' corresponds to no data!" (or worse, silently trains on whatever subset happens to be cached). Resolve shards from the parquet files' own footer stats (episode_index row-group min/max via HTTP range requests) and verify coverage from local files after download. (provenance: run 20260704-205454, scale v3)
