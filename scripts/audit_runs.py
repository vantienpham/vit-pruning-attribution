#!/usr/bin/env python
"""Report runs that are present but not finished.

    uv run --no-sync python scripts/audit_runs.py --runs out/runs --expect 7

``metrics.json`` is rewritten after every budget so that a wall-time kill does
not lose the whole run. The cost is that its presence does not mean the run
finished: a job killed after two budgets leaves a file that looks complete to
anything checking only for existence. This lists what is actually short, so a
resubmission can target it with ``--overwrite``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="out/runs")
    parser.add_argument(
        "--expect", type=int, default=6, help="number of budgets a finished run carries"
    )
    parser.add_argument(
        "--expect-datasets", type=int, default=5, help="probes each budget should carry"
    )
    args = parser.parse_args()

    root = Path(args.runs)
    complete, short, unreadable = [], [], []

    for path in sorted(root.rglob("metrics.json")):
        try:
            with open(path) as handle:
                metrics = json.load(handle)
        except json.JSONDecodeError:
            unreadable.append(path)
            continue

        budgets = metrics.get("budgets") or {}
        full = [
            key
            for key, entry in budgets.items()
            if len(entry.get("datasets") or {}) >= args.expect_datasets
        ]
        if len(full) >= args.expect:
            complete.append(path)
        else:
            short.append((path, len(full)))

    print(f"complete   : {len(complete)}")
    print(f"incomplete : {len(short)}")
    print(f"unreadable : {len(unreadable)}")

    for path, count in short:
        print(f"  {count}/{args.expect} budgets  {path.parent}")
    for path in unreadable:
        print(f"  corrupt  {path.parent}")

    if short or unreadable:
        print("\nRe-run the parents above with --overwrite.")

    # A quick census of what the campaign actually covers, which is easier to
    # read than a directory listing when several studies share a tree.
    census = Counter()
    for path in complete:
        with open(path) as handle:
            metrics = json.load(handle)
        census[(metrics.get("backbone"), metrics.get("objective"),
                metrics.get("calibration"), metrics.get("views"),
                metrics.get("allocation"))] += 1

    missing_seeds = [key for key, count in census.items() if count < 3]
    print(f"\nconfigurations with all three seeds: "
          f"{len(census) - len(missing_seeds)}/{len(census)}")
    for key in sorted(missing_seeds, key=str):
        print(f"  {census[key]}/3 seeds  {'  '.join(str(k) for k in key)}")

    return 1 if (short or unreadable) else 0


if __name__ == "__main__":
    raise SystemExit(main())
