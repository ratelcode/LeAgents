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
- run: 20260705-034343-m0-libero-scale
  cycle: 0
- run: 20260705-034343-m0-libero-scale
  cycle: 1
- run: 20260705-034343-m0-libero-scale
  cycle: 2
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
- 2026-07-05 · run 20260705-034343-m0-libero-scale · cycle 0 · libero_spatial · policy smolvla · success 63.0% · decision promote
- 2026-07-05 · run 20260705-034343-m0-libero-scale · cycle 1 · libero_spatial · policy smolvla · success 69.0% · decision promote
- 2026-07-05 · run 20260705-034343-m0-libero-scale · cycle 2 · libero_spatial · policy smolvla · success 72.0% · decision iterate

## Lessons
- 2026-07-06 · Phase 2 residual RL PASSED its go/no-go, and α is the make-or-break knob (measured on task 4, frozen v6 cycle-2 base). Composed = base + α·tanh(residual), off-policy SAC+RLPD (demo-anchored), 150k env-steps. **α=0.5 (PLD's cited LIBERO value) FAILS on SmolVLA@n=5**: the residual saturates to a ~0.32/dim constant OVERRIDE (27.7% of dims at the ±α bound), corrupting every OSC delta-action → composed 0% vs base 55% (self-reinforcing collapse: as success→0, reward→0, critic Q→0, no gradient to un-saturate). **α=0.1 PASSES big**: bounded to ±0.1 it can't override a good base action, only gently correct → composed **100% vs base 65% (+0.35)**, gate PASS. Two process lessons: (1) demo anchoring alone (α=0.5+demos) did NOT prevent the collapse — α is the primary fix. (2) Judge by the DETERMINISTIC gate (mean residual), NOT the stochastic-collection successes — mid-run the noisy exploration policy looked degraded (~16% collection success, loss_actor collapsing) yet the learned deterministic policy was 100%. Config default is now α=0.1. (provenance: runs residual_p21_task4 [α=0.5, 0%] and residual_p21_task4_a01 [α=0.1, 100%])
- 2026-07-08 · Residual RL improves the STUCK task, not just easy ones — the key Phase-2 result. task_id 5 "on the ramekin" is the run's persistent laggard (base ~40%, which data growth 200→432 never fixed). α=0.1 + demos residual → composed **60% vs base 40% (+0.20), gate PASS** (run residual_p22_task5). So the flywheel now has a mechanism to lift tasks the base can't reliably solve, WITHOUT new expert data. Design Risk #1 (a local residual can't create a behavior the base never attempts) did NOT bite because base ~40% DOES attempt the grasp — the residual nudges near-misses to success. It would still bite a truly ~0% task; 40% is above that floor. Scorecard: P2.1 task 4 (base 65%→100%, +0.35), P2.2 task 5 (base 40%→60%, +0.20).
- 2026-07-05 · Making a harvested rollout dataset BLENDABLE with the seed (so DexFlyWheel step 5 can be co-training, not a separate overfit-prone adapt) requires matching what lerobot's merge validation (aggregate.validate_all_metadata) checks: identical fps, robot_type, AND feature dicts. Four collector fixes were needed: (1) image dtype `image` not `video`; (2) fps = the SEED's label (10), not the env's (30) — for LIBERO the frame cadence is the same 1-obs-per-step, so the seed's fps label is the coherent one and no subsampling is needed; (3) robot_type = panda (was None); (4) real per-task LIBERO language instructions (was a synthetic `libero:<suite>/<id>` string, which weakens a language-conditioned policy). Implemented via `--match-dataset=<seed>` (copies fps + robot_type) + `libero_task_languages()`. Verified: a fresh 1-episode harvest matches the seed on all three merge-validated axes and merges cleanly (image dtype preserved). (provenance: run 20260705-034343 blessed ckpt, this session)
- 2026-07-05 · The DexFlyWheel adaptation stage as first configured (2000 steps on the 20-episode success-filtered rollout mix ALONE, run after main train and BEFORE eval) overfit and regressed cycle 1 from 63% → 20% — catastrophic forgetting: adapt loss fell to 0.156 vs the main train's ~0.37, per-task collapse was broad but non-uniform (drift toward the narrow harvested distribution). 20 eps × ~117 frames = 2332 frames; 2000 steps × batch 8 ≈ 7 epochs. The decision function correctly rolled back and protected the 63% baseline. Fix: adapting on rollouts ALONE for many steps is wrong — either disable adaptation and let data growth be the flywheel signal (adapt_steps=0, harvest still accumulates the mix), or adapt as a LIGHT touch (few hundred steps, lower LR) on rollouts BLENDED with real episodes. Never let an adaptation on a tiny self-generated set be the checkpoint eval/promote judges. (provenance: run 20260704-232610, cycle 1)
- 2026-07-05 · Correction/refinement of the index claim below: spatial episodes are NOT a contiguous block ending at 1538 — they span indices 1261–1692 INTERLEAVED with other suites, and the dataset holds **432 spatial episodes total** (~43/task, not 50/task). A 500-episode request caps at 432 (select_balanced caps at available), so run 20260705-034343's cycles actually trained on 200→320→432, and **the expert-data well is now dry for this suite** — further cycles differ only by training steps plus whatever self-generated data adds. (provenance: run 20260705-034343 cycle-1/2 select summaries: per-task counts 32×10, then 35–47)
- 2026-07-04 · A flat 0% across whole runs was NOT under-training: HuggingFaceVLA/libero is suite-ordered and episode prefixes [0..N) are libero_10 episodes — spatial episodes live at indices ~1261+ (see the 2026-07-05 refinement above). Prefix-based `--dataset.episodes` silently trains on the wrong suite; always select via task filtering against the eval suite (leagents.scripts.select_episodes, data.task_filter). Sparse LIBERO reward means metrics cannot distinguish this from under-training — only task-string inspection can. (provenance: runs 20260704-121559, 20260704-183753)
- 2026-07-04 · Rollout harvesting on LIBERO needs three things train/eval already knew: (1) the `--rename-map` must be plumbed to EVERY stage that loads the policy — the collector without it dies in make_policy's feature check; (2) LIBERO raw observations nest `observation.robot_state` as a dict-of-dicts, and lerobot's `rollout(return_observations=True)` torch.stack()s appended observations → TypeError; fold the env-level LiberoProcessorStep into preprocess_observation for the stored frames; (3) that env step FLIPS images (H and W) and is not idempotent — the policy path must see exactly one application, so hand rollout an identity env step that only rebuilds transition scaffolding. A failed harvest must never kill the run (loop emits improve_failed and continues). (provenance: run 20260704-210759, cycle 0, 63% then ImproveError)
- 2026-07-04 · HuggingFaceVLA/libero's `meta.episodes` `data/file_index` mapping is STALE relative to its actual 377-file hub layout (meta says episode 137 → file-009; the hub's file-055 holds it, row counts matching the episode lengths exactly). Any shard fetch that trusts the mapping — including lerobot's own partial download — pulls the wrong files, and `--dataset.episodes` training dies with "Instruction 'train' corresponds to no data!" (or worse, silently trains on whatever subset happens to be cached). Resolve shards from the parquet files' own footer stats (episode_index row-group min/max via HTTP range requests) and verify coverage from local files after download. (provenance: run 20260704-205454, scale v3)
