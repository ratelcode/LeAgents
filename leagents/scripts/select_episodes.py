"""Select dataset episode indices whose task belongs to an eval suite.

Born from a silent failure: HuggingFaceVLA/libero is suite-ordered, so a
prefix selection (``--dataset.episodes=[0..N)``) trained three whole runs on
libero_10 episodes while evaluating on libero_spatial — indistinguishable
from under-training in the metrics. This script matches episode task strings
against the eval suite's language instructions and balances the selection
across tasks (the verified guidance is per-variation coverage).

Shard resolution does NOT trust ``meta.episodes``'s ``data/file_index``
column: on HuggingFaceVLA/libero that mapping is stale (it says episode 137
lives in file-009 while the hub's file-055 actually holds it), so both
lerobot's partial download and a mapping-based fetch pull the wrong shards
and training dies with "Instruction 'train' corresponds to no data!".
Instead we read each hub file's parquet FOOTER (a few KB over HTTP range
requests) for its episode_index min/max, binary-search the region holding
the selected episodes, download exactly those files, and then verify from
the local files that every selected episode is actually covered.

The LAST stdout line is a JSON summary the Data Agent parses.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def select_balanced(
    episode_tasks: list[tuple[int, str]],
    suite_languages: list[str],
    limit: int,
) -> tuple[list[int], dict[str, int]]:
    """Pick up to ``limit`` episode indices matching the suite, round-robin
    across tasks so coverage stays balanced. Pure function; dataset order is
    preserved within each task."""
    wanted = {normalize(lang) for lang in suite_languages}
    by_task: dict[str, list[int]] = {}
    for index, task in episode_tasks:
        key = normalize(task)
        if key in wanted:
            by_task.setdefault(key, []).append(index)

    selected: list[int] = []
    counts: dict[str, int] = {task: 0 for task in by_task}
    queues = {task: list(indices) for task, indices in by_task.items()}
    while len(selected) < limit and any(queues.values()):
        for task in list(queues):
            if queues[task] and len(selected) < limit:
                selected.append(queues[task].pop(0))
                counts[task] += 1
    return sorted(selected), counts


def files_covering(selected: list[int], spans: list[tuple[str, int, int]]) -> list[str]:
    """Files whose [ep_min, ep_max] span contains at least one selected
    episode. Pure function; preserves the order of ``spans``."""
    ordered = sorted(selected)
    chosen = []
    for path, ep_min, ep_max in spans:
        i = bisect.bisect_left(ordered, ep_min)
        if i < len(ordered) and ordered[i] <= ep_max:
            chosen.append(path)
    return chosen


def missing_episodes(selected: list[int], covered: set[int]) -> list[int]:
    """Selected episodes not present in the covered set. Pure function."""
    return [ep for ep in selected if ep not in covered]


def _remote_span(fs, repo_id: str, rel_path: str) -> tuple[int, int]:
    """episode_index min/max of a hub parquet file, from its footer only."""
    import pyarrow.parquet as pq

    with fs.open(f"datasets/{repo_id}/{rel_path}") as f:
        md = pq.ParquetFile(f).metadata
        col = next(k for k in range(md.num_columns)
                   if md.schema.column(k).name == "episode_index")
        stats = [md.row_group(g).column(col).statistics
                 for g in range(md.num_row_groups)]
        return min(s.min for s in stats), max(s.max for s in stats)


def resolve_needed_files(repo_id: str, selected: list[int]) -> list[str]:
    """Hub data files that hold the selected episodes, determined from the
    files' own footer stats (assumes episode-monotonic file order for the
    binary search; the caller's coverage check catches violations loudly)."""
    from huggingface_hub import HfApi, HfFileSystem

    files = sorted(p for p in HfApi().list_repo_files(repo_id, repo_type="dataset")
                   if p.startswith("data/") and p.endswith(".parquet"))
    if not files:
        return []
    fs = HfFileSystem()
    span_cache: dict[str, tuple[int, int]] = {}

    def span(rel: str) -> tuple[int, int]:
        if rel not in span_cache:
            span_cache[rel] = _remote_span(fs, repo_id, rel)
        return span_cache[rel]

    lo_target, hi_target = min(selected), max(selected)
    lo, hi = 0, len(files) - 1
    while lo < hi:  # first file whose max episode reaches the target range
        mid = (lo + hi) // 2
        if span(files[mid])[1] < lo_target:
            lo = mid + 1
        else:
            hi = mid
    spans = []
    for rel in files[lo:]:
        ep_min, ep_max = span(rel)
        if ep_min > hi_target:
            break
        spans.append((rel, ep_min, ep_max))
    return files_covering(selected, spans)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--suite", required=True, help="e.g. libero_spatial")
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--no-download", action="store_true",
                        help="skip ensuring the selected episodes' shards are local")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    try:
        from libero.libero import benchmark

        suite = benchmark.get_benchmark_dict()[args.suite]()
        languages = [suite.get_task(i).language for i in range(suite.n_tasks)]
    except KeyError:
        print(f"unknown suite {args.suite!r}", file=sys.stderr)
        return 1

    meta = LeRobotDatasetMetadata(args.repo_id)
    frame = meta.episodes
    frame = frame.to_pandas() if hasattr(frame, "to_pandas") else frame
    episode_tasks = [
        (int(row["episode_index"]), str(list(row["tasks"])[0]))
        for _, row in frame.iterrows()
    ]

    selected, counts = select_balanced(episode_tasks, languages, args.limit)

    fetched = 0
    needed: list[str] = []
    if selected and not args.no_download:
        # lerobot skips hub sync when the local root already exists, so a
        # partially-cached dataset would load ZERO rows for these episodes
        # ("Instruction 'train' corresponds to no data!"). Ensure the shards
        # holding the selected episodes are present in lerobot's cache —
        # resolved from footer stats, never from meta's stale file mapping.
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        needed = resolve_needed_files(args.repo_id, selected)
        cache_dir = meta.root.parent.parent.parent  # snapshots/<sha> -> hub cache root
        for rel_path in needed:
            if not (meta.root / rel_path).exists():
                hf_hub_download(args.repo_id, rel_path, repo_type="dataset",
                                cache_dir=cache_dir)
                fetched += 1

        covered: set[int] = set()
        for rel_path in needed:
            table = pq.read_table(meta.root / rel_path, columns=["episode_index"])
            covered.update(table["episode_index"].to_pylist())
        missing = missing_episodes(selected, covered)
        if missing:
            print(f"{len(missing)} selected episodes are not in the fetched "
                  f"shards (e.g. {missing[:5]}) — the dataset's file layout "
                  "does not match its footer stats", file=sys.stderr)
            return 1

    summary = {
        "episodes": selected,
        "selected": len(selected),
        "suite_tasks": len(languages),
        "tasks_covered": sum(1 for c in counts.values() if c),
        "per_task": counts,
        "shards_fetched": fetched,
        "shards_needed": len(needed),
    }
    if not selected:
        print(f"no episodes in {args.repo_id} match suite {args.suite!r} — "
              "training would be on the wrong tasks", file=sys.stderr)
        print(json.dumps(summary))
        return 1
    if summary["tasks_covered"] < len(languages):
        print(f"warning: only {summary['tasks_covered']}/{len(languages)} suite tasks "
              "have episodes", file=sys.stderr)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
