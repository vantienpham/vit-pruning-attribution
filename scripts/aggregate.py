#!/usr/bin/env python
"""Turn a directory of runs into the tables the manuscript reports.

    uv run --no-sync python scripts/aggregate.py --runs out/runs --out results/tables

Reads every ``metrics.json`` under ``--runs``, flattens it to one row per
(backbone, objective, calibration, seed, budget, dataset, metric), writes the
tidy CSV, and derives the per-table summaries. Aggregation over seeds always
reports the standard deviation alongside the mean: the margins claimed in this
literature are small enough that the spread across calibration draws decides
whether they mean anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

TIDY_FIELDS = [
    "backbone",
    "objective",
    "calibration",
    "views",
    "allocation",
    "seed",
    "budget",
    "parameter_sparsity",
    "flop_sparsity",
    "dataset",
    "metric",
    "value",
]


def iter_runs(root: Path) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    for path in sorted(root.rglob("metrics.json")):
        try:
            with open(path) as handle:
                yield path, json.load(handle)
        except json.JSONDecodeError:
            print(f"skipping unreadable {path}", file=sys.stderr)


def flatten(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per scalar measurement."""
    rows = []
    base = {
        "backbone": metrics.get("backbone"),
        "objective": metrics.get("objective"),
        "calibration": metrics.get("calibration"),
        "views": metrics.get("views"),
        "allocation": metrics.get("allocation"),
        "seed": metrics.get("seed"),
    }

    for budget, entry in (metrics.get("budgets") or {}).items():
        shared = {
            **base,
            "budget": budget,
            "parameter_sparsity": entry.get("parameter_sparsity"),
            "flop_sparsity": entry.get("flop_sparsity"),
        }
        for dataset, scores in (entry.get("datasets") or {}).items():
            for metric in ("knn", "linear"):
                if metric in scores:
                    rows.append({**shared, "dataset": dataset, "metric": metric,
                                 "value": scores[metric]})
        for metric in ("dense_r1", "dense_r5", "dense_spearman"):
            if metric in entry:
                rows.append({**shared, "dataset": "-", "metric": metric, "value": entry[metric]})
        if "throughput" in entry:
            rows.append({**shared, "dataset": "-", "metric": "images_per_second",
                         "value": entry["throughput"]["images_per_second"]})

    saliency = metrics.get("saliency") or {}
    for metric in ("seconds", "peak_memory_gb"):
        if metric in saliency:
            rows.append({**base, "budget": "-", "parameter_sparsity": None,
                         "flop_sparsity": None, "dataset": "-",
                         "metric": f"saliency_{metric}", "value": saliency[metric]})
    for name, value in (saliency.get("diagnostics") or {}).items():
        rows.append({**base, "budget": "-", "parameter_sparsity": None,
                     "flop_sparsity": None, "dataset": "-",
                     "metric": f"diag_{name}", "value": value})

    return rows


def summarise(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mean, sample standard deviation and count across seeds."""
    grouped: Dict[Tuple, List[float]] = defaultdict(list)
    for row in rows:
        if row["value"] is None:
            continue
        key = (
            row["backbone"],
            row["objective"],
            row["calibration"],
            row["views"],
            row["allocation"],
            row["budget"],
            row["dataset"],
            row["metric"],
        )
        grouped[key].append(float(row["value"]))

    out = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        backbone, objective, calibration, views, allocation, budget, dataset, metric = key
        out.append(
            {
                "backbone": backbone,
                "objective": objective,
                "calibration": calibration,
                "views": views,
                "allocation": allocation,
                "budget": budget,
                "dataset": dataset,
                "metric": metric,
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
        )
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #


def _cell(mean: Optional[float], std: Optional[float], scale: float, best: bool) -> str:
    if mean is None:
        return "--"
    value = f"{mean * scale:.1f}"
    if std is not None and std > 0:
        value += f"\\,\\textsubscript{{{std * scale:.1f}}}"
    return f"\\textbf{{{value}}}" if best else value


def latex_objective_table(
    summary: List[Dict[str, Any]],
    backbone: str,
    dataset: str,
    metric: str,
    objectives: List[str],
    budgets: List[str],
    calibration: str = "imagenet",
) -> str:
    """Rows are objectives, columns are sparsity budgets."""
    index = {
        (row["objective"], row["budget"]): row
        for row in summary
        if row["backbone"] == backbone
        and row["dataset"] == dataset
        and row["metric"] == metric
        and row["calibration"] in (calibration, None)
    }

    best = {}
    for budget in budgets:
        candidates = [
            (index[(objective, budget)]["mean"], objective)
            for objective in objectives
            if (objective, budget) in index
        ]
        if candidates:
            best[budget] = max(candidates)[1]

    lines = []
    for objective in objectives:
        cells = []
        for budget in budgets:
            row = index.get((objective, budget))
            cells.append(
                _cell(
                    row["mean"] if row else None,
                    row["std"] if row else None,
                    100.0,
                    best.get(budget) == objective,
                )
            )
        lines.append(f"{objective} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="out/runs")
    parser.add_argument("--out", default="results/tables")
    args = parser.parse_args()

    root = Path(args.runs)
    if not root.exists():
        print(f"no run directory at {root}", file=sys.stderr)
        return 1

    rows: List[Dict[str, Any]] = []
    run_count = 0
    for path, metrics in iter_runs(root):
        rows.extend(flatten(metrics))
        run_count += 1

    if not rows:
        print(f"no metrics found under {root}", file=sys.stderr)
        return 1

    out = Path(args.out)
    write_csv(out / "measurements.csv", rows, TIDY_FIELDS)

    summary = summarise(rows)
    write_csv(
        out / "summary.csv",
        summary,
        ["backbone", "objective", "calibration", "views", "allocation", "budget",
         "dataset", "metric", "mean", "std", "n"],
    )

    print(f"aggregated {run_count} runs")
    seeds = sorted({row["seed"] for row in rows if row["seed"] is not None})
    print(f"seeds present: {seeds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
