"""Select dataset episode indices whose task belongs to an eval suite.

Born from a silent failure: HuggingFaceVLA/libero is suite-ordered, so a
prefix selection (``--dataset.episodes=[0..N)``) trained three whole runs on
libero_10 episodes while evaluating on libero_spatial — indistinguishable
from under-training in the metrics. This script matches episode task strings
against the eval suite's language instructions and balances the selection
across tasks (the verified guidance is per-variation coverage).

The LAST stdout line is a JSON summary the Data Agent parses.
"""

from __future__ import annotations

import argparse
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--suite", required=True, help="e.g. libero_spatial")
    parser.add_argument("--limit", type=int, required=True)
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
    summary = {
        "episodes": selected,
        "selected": len(selected),
        "suite_tasks": len(languages),
        "tasks_covered": sum(1 for c in counts.values() if c),
        "per_task": counts,
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
