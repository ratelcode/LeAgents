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
updated: '2026-07-04'
---

# Task: libero_spatial

## Observations
- 2026-07-04 · run 20260704-121559-m0-libero-smolvla · cycle 0 · libero_spatial · policy smolvla · success 0.0% · decision promote
- 2026-07-04 · run 20260704-121559-m0-libero-smolvla · cycle 1 · libero_spatial · policy smolvla · success 0.0% · decision iterate
- 2026-07-04 · run 20260704-121559-m0-libero-smolvla · cycle 2 · libero_spatial · policy smolvla · success 0.0% · decision iterate
- 2026-07-04 · run 20260704-183753-m0-libero-scale · cycle 0 · libero_spatial · policy smolvla · success 0.0% · decision promote

## Lessons
- 2026-07-04 · A flat 0% across whole runs was NOT under-training: HuggingFaceVLA/libero is suite-ordered and episode prefixes [0..N) are libero_10 episodes — spatial episodes live at indices ~1261–1538. Prefix-based `--dataset.episodes` silently trains on the wrong suite; always select via task filtering against the eval suite (leagents.scripts.select_episodes, data.task_filter). Sparse LIBERO reward means metrics cannot distinguish this from under-training — only task-string inspection can. (provenance: runs 20260704-121559, 20260704-183753)
