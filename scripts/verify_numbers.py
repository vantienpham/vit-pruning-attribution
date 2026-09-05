#!/usr/bin/env python
"""Print every quantity the manuscript quotes, straight from the runs.

    uv run --no-sync python scripts/verify_numbers.py

Numbers drift during revision, and the ones in the abstract and the discussion
are the ones least likely to be regenerated when a table is. This prints them
next to their source so the two can be compared in one pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

DATASETS = ["imagenet100", "pets", "dtd", "flowers", "eurosat"]


def load(path: Path) -> List[Dict[str, str]]:
    with open(path) as handle:
        return list(csv.DictReader(handle))


def retention(
    rows: Sequence[Dict[str, str]],
    backbone: str,
    objective: str,
    budget: str,
    allocation: str = "global",
    views: str = "two-crop",
    calibration: str = "imagenet",
    metric: str = "linear",
):
    """Mean retained score over the five probes, and its spread over draws.

    Retention is formed inside a calibration draw and only then averaged over
    draws, matching ``make_tables.retention``. One draw gives one pruned
    network scored on all five probes, so the per-probe deviations within a
    draw are correlated and cannot be pooled as if they were independent.
    """
    base: Dict[Tuple[str, str], float] = {}
    cut: Dict[str, Dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["backbone"] != backbone or row["metric"] != metric:
            continue
        if row["dataset"] not in DATASETS:
            continue
        if row["budget"] == "s0":
            base[(row["dataset"], row["seed"])] = float(row["value"])
        if (row["objective"] == objective and row["budget"] == budget
                and row["allocation"] == allocation and row["views"] == views
                and row["calibration"] == calibration):
            cut[row["seed"]][row["dataset"]] = float(row["value"])

    draws = []
    for seed, scores in cut.items():
        if len(scores) < len(DATASETS):
            continue
        if any((d, seed) not in base for d in DATASETS):
            continue
        draws.append(statistics.fmean(scores[d] / base[(d, seed)] for d in DATASETS))
    if not draws:
        return None
    spread = statistics.stdev(draws) if len(draws) > 1 else 0.0
    return 100 * statistics.fmean(draws), 100 * spread


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", default="results/tables/measurements.csv")
    parser.add_argument("--ranking", default="out/ranking/dinov2/metrics.json")
    parser.add_argument("--cost", default="out/cost/dinov2-vitb14/metrics.json")
    parser.add_argument("--spectrum", default="out/spectrum")
    args = parser.parse_args()

    rows = load(Path(args.measurements))
    objectives = ["random", "magnitude", "lamp", "cosine", "mse", "cutvit",
                  "gram-a0.25", "gram-a0.5", "gram-a1", "gram-a2"]

    for backbone, allocation in [("dinov2-vitb14", "global"),
                                 ("dinov2-vitb14", "block-normalised"),
                                 ("dino-vitb16", "block-normalised"),
                                 ("deit-vitb16", "block-normalised")]:
        print(f"\n== {backbone}, {allocation} ==")
        for budget in ("s10", "s20", "s30"):
            cells = []
            for o in objectives:
                r = retention(rows, backbone, o, budget, allocation=allocation)
                cells.append(f"{o}={r[0]:.1f}+-{r[1]:.1f}" if r else f"{o}=--")
            print(f"  {budget}: " + "  ".join(cells))

    print("\n== views (dinov2, s20, global) ==")
    for views in ("two-crop", "noise", "identical"):
        cells = []
        for o in ("cosine", "cutvit", "gram-a1"):
            r = retention(rows, "dinov2-vitb14", o, "s20", views=views)
            cells.append(f"{o}={r[0]:.1f}+-{r[1]:.1f}" if r else f"{o}=--")
        print(f"  {views:10s} " + "  ".join(cells))

    print("\n== calibration corpus (dinov2, s20, global) ==")
    for corpus in ("imagenet", "pets", "dtd", "eurosat"):
        cells = []
        for o in ("cosine", "cutvit", "gram-a1"):
            r = retention(rows, "dinov2-vitb14", o, "s20", calibration=corpus)
            cells.append(f"{o}={r[0]:.1f}+-{r[1]:.1f}" if r else f"{o}=--")
        print(f"  {corpus:10s} " + "  ".join(cells))

    print("\n== allocation (dinov2, s20) ==")
    for allocation in ("global", "linear-decay", "block-normalised", "uniform"):
        cells = []
        for o in ("cosine", "cutvit", "gram-a1"):
            r = retention(rows, "dinov2-vitb14", o, "s20", allocation=allocation)
            cells.append(f"{o}={r[0]:.1f}+-{r[1]:.1f}" if r else f"{o}=--")
        print(f"  {allocation:17s} " + "  ".join(cells))

    ranking = Path(args.ranking)
    if ranking.exists():
        m = json.loads(ranking.read_text())
        print("\n== ranking correlations ==")
        for key in ("seed_noise", "objective_effect"):
            e = m[key]
            print(f"  {key:18s} flat={e['spearman_flat']:.3f} "
                  f"in-block={e['spearman_within_block']:.3f} "
                  f"keep={e['keep_overlap']:.3f} n={e['comparisons']}")

    cost = Path(args.cost)
    if cost.exists():
        m = json.loads(cost.read_text())
        print("\n== cost ==")
        for k, v in m["calibration"].items():
            print(f"  {k:12s} {v['seconds']:6.1f}s {v['peak_memory_gb']:5.2f}GB")
        print("  throughput: " + "  ".join(
            f"{k}={v['images_per_second']:.0f}img/s({v['parameters']/1e6:.1f}M)"
            for k, v in m["throughput"].items()))

    root = Path(args.spectrum)
    if root.exists():
        print("\n== spectra (effective rank, per image) ==")
        for path in sorted(root.glob("*/metrics.json")):
            axes = json.loads(path.read_text())["axes"]
            e = axes.get("spatial-per_image")
            if e:
                print(f"  {path.parent.name:18s} H={e['entropy_nats']:.2f} "
                      f"eff_rank={e['effective_rank']:.0f} "
                      f"99% at {e['rank_for']['0.99']} dirs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
