#!/usr/bin/env python
"""Render the manuscript's tables from the aggregated summary.

    uv run --no-sync python scripts/make_tables.py \
        --summary results/tables/summary.csv --out redaction/tables

Each table body is written as its own ``.tex`` fragment and pulled in with
\\input, so the numbers in the manuscript are never retyped and cannot drift
away from the runs that produced them.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: The rank Cut-ViT uses for the low-rank decomposition of the Gram matrices.
CUTVIT_RANK = 192

#: Display names, in the order the tables list them.
OBJECTIVE_LABELS = {
    "random": r"Random",
    "magnitude": r"Magnitude",
    "lamp": r"LAMP \cite{lee2021layer}",
    "cosine": r"Pointwise cosine \cite{simoncini2025elastic}",
    "mse": r"Pointwise MSE",
    "cutvit": r"Subspace, $\alpha\!=\!0$, $K\!=\!192$ \cite{yin2026cut}",
    "gram-a0.25": r"Spectral, $\alpha\!=\!0.25$",
    "gram-a0.5": r"Spectral, $\alpha\!=\!0.5$",
    "gram-a1": r"Spectral, $\alpha\!=\!1$",
    "gram-a2": r"Spectral, $\alpha\!=\!2$",
    "gram-a0-k192": r"$\alpha\!=\!0$",
    "gram-a0.25-k192": r"$\alpha\!=\!0.25$",
    "gram-a0.5-k192": r"$\alpha\!=\!0.5$",
    "gram-a1-k192": r"$\alpha\!=\!1$",
    "gram-a2-k192": r"$\alpha\!=\!2$",
    "gram-a1-spatial": r"Spatial Gram only",
    "gram-a1-channel": r"Channel Gram only",
    "gram-a1-only": r"No pointwise anchor",
    "gram-a1-pooled": r"Batch-pooled Gram",
}

#: Short names for tables where the row label competes with the numbers.
SHORT_LABELS = {
    "cosine": r"Pointwise cosine",
    "mse": r"Pointwise MSE",
    "cutvit": r"Subspace, $K\!=\!192$",
    "gram-a0.25": r"Spectral $\alpha\!=\!0.25$",
    "gram-a0.5": r"Spectral $\alpha\!=\!0.5$",
    "gram-a1": r"Spectral $\alpha\!=\!1$",
    "gram-a2": r"Spectral $\alpha\!=\!2$",
}

#: Names short enough for a single-column table that carries uncertainties.
COMPACT_LABELS = {
    "random": r"Random",
    "magnitude": r"Magnitude",
    "lamp": r"LAMP",
    "cosine": r"Pointwise cosine",
    "mse": r"Pointwise MSE",
    "cutvit": r"Subspace $\alpha\!=\!0$",
    "gram-a0.25": r"Spectral $\alpha\!=\!0.25$",
    "gram-a0.5": r"Spectral $\alpha\!=\!0.5$",
    "gram-a1": r"Spectral $\alpha\!=\!1$",
    "gram-a2": r"Spectral $\alpha\!=\!2$",
}

BACKBONE_LABELS = {
    "dinov2-vitb14": "DINOv2 ViT-B/14",
    "dino-vitb16": "DINO ViT-B/16",
    "deit-vitb16": "DeiT ViT-B/16",
}

DATASET_LABELS = {
    "imagenet100": "IN-100",
    "pets": "Pets",
    "dtd": "DTD",
    "flowers": "Flowers",
    "eurosat": "EuroSAT",
}

BUDGET_LABELS = {
    "s0": "0",
    "s05": "5",
    "s10": "10",
    "s15": "13",
    "s20": "18",
    "s30": "30",
    "s40": "41",
}


Row = Dict[str, str]


def write_body(path: Path, body: str) -> None:
    r"""Write a table body for \input inside a tabular.

    The fragment must not end with a row terminator: TeX mishandles a ``\\\\``
    that falls at the end of an \input file inside an alignment, and reports it
    as a misplaced \noalign on whatever rule follows. The manuscript supplies
    the final ``\\\\`` after the \input instead.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    trimmed = body.rstrip()
    if trimmed.endswith("\\\\"):
        trimmed = trimmed[:-2].rstrip()
    path.write_text(trimmed + "%\n")


def load(path: Path) -> List[Row]:
    with open(path) as handle:
        return list(csv.DictReader(handle))


def select(
    rows: Iterable[Row],
    **conditions,
) -> List[Row]:
    out = []
    for row in rows:
        if all(row.get(key) == value for key, value in conditions.items()):
            out.append(row)
    return out


def unpruned_baseline(rows: List[Row], backbone: str, metric: str) -> Dict[str, float]:
    """The unpruned score per dataset, used to express results as retention.

    Every objective shares the same unpruned network, so these are the same
    numbers whichever row they are read from.
    """
    baseline: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        if row["backbone"] == backbone and row["metric"] == metric and row["budget"] == "s0":
            baseline[row["dataset"]].append(float(row["mean"]))
    return {dataset: statistics.fmean(values) for dataset, values in baseline.items()}


def mean_over_datasets(
    rows: List[Row],
    datasets: Sequence[str],
    baseline: Optional[Dict[str, float]] = None,
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """Average a metric over datasets, keyed by (objective, budget).

    With ``baseline`` given, each dataset is first divided by its unpruned
    score, so the average is the fraction of the original accuracy retained.
    The five probes differ in difficulty by tens of points, and a raw average
    over them would be dominated by whichever happens to be hardest.

    The per-dataset standard deviations are pooled in quadrature, which is the
    spread of the mean across calibration draws under the assumption that the
    draws are independent between datasets. It is reported so the reader can see
    whether a margin exceeds the noise it is measured against.
    """
    grouped: Dict[Tuple[str, str], List[Tuple[float, float]]] = defaultdict(list)
    for row in rows:
        dataset = row["dataset"]
        if dataset not in datasets:
            continue
        scale = 1.0
        if baseline is not None:
            if dataset not in baseline or baseline[dataset] <= 0:
                continue
            scale = 1.0 / baseline[dataset]
        grouped[(row["objective"], row["budget"])].append(
            (float(row["mean"]) * scale, float(row["std"]) * scale)
        )

    out = {}
    for key, values in grouped.items():
        if len(values) != len(datasets):
            continue
        means = [m for m, _ in values]
        stds = [s for _, s in values]
        pooled = (sum(s**2 for s in stds) ** 0.5) / len(stds)
        out[key] = (statistics.fmean(means), pooled)
    return out


def fmt(value: Optional[Tuple[float, float]], bold: bool = False, digits: int = 1) -> str:
    if value is None:
        return "--"
    mean, std = value
    text = f"{100 * mean:.{digits}f}"
    if std > 0:
        text += rf"\std{{{100 * std:.{digits}f}}}"
    return rf"\textbf{{{text}}}" if bold else text


def objective_table(
    rows: List[Row],
    backbone: str,
    objectives: Sequence[str],
    budgets: Sequence[str],
    datasets: Sequence[str],
    metric: str = "linear",
    calibration: str = "imagenet",
    views: str = "two-crop",
    allocation: str = "global",
    highlight_from: int = 0,
    relative: bool = True,
    labels: Optional[Dict[str, str]] = None,
) -> str:
    """Objectives down the rows, sparsity budgets across the columns."""
    labels = OBJECTIVE_LABELS if labels is None else labels
    subset = select(
        rows,
        backbone=backbone,
        metric=metric,
        calibration=calibration,
        views=views,
        allocation=allocation,
    )
    baseline = unpruned_baseline(rows, backbone, metric) if relative else None
    table = mean_over_datasets(subset, datasets, baseline)

    best: Dict[str, str] = {}
    for budget in budgets[highlight_from:]:
        candidates = [
            (table[(objective, budget)][0], objective)
            for objective in objectives
            if (objective, budget) in table
        ]
        if candidates:
            best[budget] = max(candidates)[1]

    lines = []
    for objective in objectives:
        cells = [
            fmt(table.get((objective, budget)), bold=best.get(budget) == objective)
            for budget in budgets
        ]
        lines.append(f"{labels.get(objective, objective)} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def component_table(
    rows: List[Row],
    objectives: Sequence[str],
    variants: Sequence[Tuple[str, str, str]],
    budgets: Sequence[str],
    datasets: Sequence[str],
    backbone: str = "dinov2-vitb14",
    metric: str = "linear",
    relative: bool = True,
) -> str:
    """One row per (variant, objective), for the view/allocation/corpus studies.

    ``variants`` are ``(label, field, value)`` triples naming the pipeline
    component being varied.
    """
    baseline = unpruned_baseline(rows, backbone, metric) if relative else None
    lines = []
    for label, field, value in variants:
        conditions = {
            "backbone": backbone,
            "metric": metric,
            "calibration": "imagenet",
            "views": "two-crop",
            "allocation": "global",
        }
        conditions[field] = value
        table = mean_over_datasets(select(rows, **conditions), datasets, baseline)

        cells_by_objective = []
        for objective in objectives:
            cells_by_objective.append(
                " & ".join(fmt(table.get((objective, budget))) for budget in budgets)
            )
        lines.append(f"{label} & " + " & ".join(cells_by_objective) + r" \\")
    return "\n".join(lines)


def multi_backbone_table(
    rows: List[Row],
    arms: Sequence[Tuple[str, str]],
    objectives: Sequence[str],
    budgets: Sequence[str],
    datasets: Sequence[str],
    metric: str = "linear",
    relative: bool = True,
    labels: Optional[Dict[str, str]] = None,
) -> str:
    """The main comparison: objectives down the rows, arm x budget across.

    Each arm is a (backbone, allocation) pair, so the same objectives can be
    shown with and without the cross-block scale side by side. Cells are the
    fraction of the unpruned score retained, averaged over the five probes,
    with the standard deviation over three calibration draws.
    """
    tables = {}
    for arm in arms:
        backbone, allocation = arm
        subset = select(
            rows, backbone=backbone, metric=metric, calibration="imagenet",
            views="two-crop", allocation=allocation,
        )
        baseline = unpruned_baseline(rows, backbone, metric) if relative else None
        tables[arm] = mean_over_datasets(subset, datasets, baseline)

    best = {}
    for arm in arms:
        for budget in budgets:
            candidates = [
                (tables[arm][(o, budget)][0], o)
                for o in objectives
                if (o, budget) in tables[arm]
            ]
            if candidates:
                best[(arm, budget)] = max(candidates)[1]

    labels = COMPACT_LABELS if labels is None else labels
    lines = []
    for objective in objectives:
        cells = []
        for arm in arms:
            for budget in budgets:
                cells.append(
                    fmt(tables[arm].get((objective, budget)),
                        bold=best.get((arm, budget)) == objective)
                )
        lines.append(
            f"{labels.get(objective, objective)} & " + " & ".join(cells) + r" \\"
        )
    return "\n".join(lines)


def alpha_rank_table(
    rows: List[Row],
    alphas: Sequence[float],
    budgets: Sequence[str],
    datasets: Sequence[str],
    backbone: str = "dinov2-vitb14",
    metric: str = "linear",
    relative: bool = True,
) -> str:
    """The (exponent, rank) plane: rows are alpha, column groups are K.

    Separating the two is the point: the published objective fixes both at once,
    so a comparison against it cannot say which of them carries the difference.
    """
    subset = select(
        rows, backbone=backbone, metric=metric, calibration="imagenet",
        views="two-crop", allocation="global",
    )
    baseline = unpruned_baseline(rows, backbone, metric) if relative else None
    table = mean_over_datasets(subset, datasets, baseline)

    lines = []
    for alpha in alphas:  # noqa: B007
        truncated = f"gram-a{alpha:g}-k{CUTVIT_RANK}"
        full = "cutvit-basis" if alpha == 0 else f"gram-a{alpha:g}"
        cells = [fmt(table.get((truncated, budget))) for budget in budgets]
        cells += [
            fmt(table.get((full, budget))) if alpha != 0 else "n/a"
            for budget in budgets
        ]
        lines.append(rf"$\alpha\!=\!{alpha:g}$ & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def dense_table(
    rows: List[Row],
    objectives: Sequence[str],
    budgets: Sequence[str],
    backbone: str = "dinov2-vitb14",
    metrics: Sequence[str] = ("dense_r1", "dense_spearman"),
) -> str:
    """Preservation of the teacher's dense patch correspondences.

    Label-free, and the closest stand-in available here for the dense tasks
    (matching, video propagation) that this literature is usually scored on.
    """
    index: Dict[Tuple[str, str, str], Tuple[float, float]] = {}
    for row in select(
        rows, backbone=backbone, calibration="imagenet",
        views="two-crop", allocation="global", dataset="-",
    ):
        if row["metric"] in metrics:
            index[(row["objective"], row["budget"], row["metric"])] = (
                float(row["mean"]), float(row["std"])
            )

    best: Dict[Tuple[str, str], str] = {}
    for budget in budgets:
        for metric in metrics:
            candidates = [
                (index[(o, budget, metric)][0], o)
                for o in objectives
                if (o, budget, metric) in index
            ]
            if candidates:
                best[(budget, metric)] = max(candidates)[1]

    lines = []
    for objective in objectives:
        cells = []
        for budget in budgets:
            for metric in metrics:
                cells.append(
                    fmt(index.get((objective, budget, metric)),
                        bold=best.get((budget, metric)) == objective)
                )
        lines.append(f"{OBJECTIVE_LABELS.get(objective, objective)} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def cost_table(rows: List[Row], objectives: Sequence[str], backbone: str) -> str:
    """Calibration wall-clock and peak memory, measured under one protocol."""
    seconds = {
        row["objective"]: float(row["mean"])
        for row in select(rows, backbone=backbone, metric="saliency_seconds",
                          views="two-crop", calibration="imagenet", allocation="global")
    }
    memory = {
        row["objective"]: float(row["mean"])
        for row in select(rows, backbone=backbone, metric="saliency_peak_memory_gb",
                          views="two-crop", calibration="imagenet", allocation="global")
    }

    lines = []
    for objective in objectives:
        if objective not in seconds:
            continue
        lines.append(
            f"{SHORT_LABELS.get(objective, objective)} & "
            f"{seconds[objective]:.1f} & {memory[objective]:.2f}" + r" \\"
        )
    return "\n".join(lines)


def sparsity_row(rows: List[Row], backbone: str, budgets: Sequence[str]) -> str:
    """Measured parameter and FLOP sparsity for each budget."""
    from collections import defaultdict as dd

    # parameter_sparsity is carried on the tidy rows, not the summary, so read
    # it back from any measurement row for the budget.
    return " & ".join(BUDGET_LABELS[b] for b in budgets)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="results/tables/summary.csv")
    parser.add_argument("--out", default="redaction/tables")
    parser.add_argument("--metric", default="linear", choices=["linear", "knn"])
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="report raw accuracy instead of the fraction of the unpruned score",
    )
    args = parser.parse_args()
    relative = not args.absolute

    rows = load(Path(args.summary))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    datasets = [d for d in DATASET_LABELS if any(r["dataset"] == d for r in rows)]
    budgets = ["s0", "s05", "s10", "s15", "s20", "s30", "s40"]
    main_objectives = [
        "random", "magnitude", "lamp", "cosine", "mse", "cutvit",
        "gram-a0.25", "gram-a0.5", "gram-a1", "gram-a2",
    ]

    written = []

    # One table, three arms: the primary encoder under the raw global ranking
    # the published pipeline uses, the same encoder with the cross-block scale
    # removed, and a second encoder to show the conclusion is not specific to
    # one pretraining objective.
    body = multi_backbone_table(
        rows,
        [("dinov2-vitb14", "global"),
         ("dinov2-vitb14", "block-normalised"),
         ("dino-vitb16", "block-normalised")],
        main_objectives, ["s10", "s20", "s30"], datasets,
        metric=args.metric, relative=relative,
    )
    if body.strip():
        write_body(out / "objectives.tex", body)
        written.append(out / "objectives.tex")

    # Per-backbone bodies over the full budget grid, for the supplementary
    # record and for checking a claim against a single encoder.
    for backbone in BACKBONE_LABELS:
        body = objective_table(
            rows, backbone, main_objectives, budgets, datasets,
            metric=args.metric, highlight_from=1, relative=relative,
        )
        if body.strip():
            path = out / f"objectives-{backbone}.tex"
            write_body(path, body)
            written.append(path)

    body = alpha_rank_table(
        rows, [0.0, 0.5, 1.0, 2.0], ["s10", "s20"], datasets,
        metric=args.metric, relative=relative,
    )
    if body.strip():
        write_body(out / "alpha-rank.tex", body)
        written.append(out / "alpha-rank.tex")

    variant_objectives = ["cosine", "cutvit", "gram-a1"]
    # One sparsity keeps the table inside a single column measure; the
    # behaviour at the other budgets is described in the text.
    short_budgets = ["s20"]

    # One table for the three pipeline components, so the reader can compare
    # their effect sizes against each other rather than across three floats.
    groups = [
        (r"\textit{View protocol}", [
            ("~~Two crops$^\\dagger$", "views", "two-crop"),
            ("~~Noise", "views", "noise"),
            ("~~Identical", "views", "identical"),
        ]),
        (r"\textit{Depth allocation}", [
            ("~~Global$^\\dagger$", "allocation", "global"),
            ("~~Linear decay", "allocation", "linear-decay"),
            ("~~Block-norm.", "allocation", "block-normalised"),
            ("~~Uniform", "allocation", "uniform"),
        ]),
        (r"\textit{Calibration corpus}", [
            ("~~ImageNet$^\\dagger$", "calibration", "imagenet"),
            ("~~Pets", "calibration", "pets"),
            ("~~DTD", "calibration", "dtd"),
            ("~~EuroSAT", "calibration", "eurosat"),
        ]),
    ]
    blocks = []
    ncols = len(variant_objectives) * len(short_budgets)
    for heading, variants in groups:
        block = component_table(
            rows, variant_objectives, variants, short_budgets, datasets,
            metric=args.metric, relative=relative,
        )
        if block.strip():
            blocks.append(f"{heading}" + " & --" * ncols + r" \\" + "\n" + block)
    if blocks:
        write_body(out / "components.tex", "\n".join(blocks))
        written.append(out / "components.tex")

    body = dense_table(rows, main_objectives, ["s10", "s20"])
    if body.strip():
        write_body(out / "dense.tex", body)
        written.append(out / "dense.tex")

    body = cost_table(rows, ["cosine", "cutvit", "gram-a0.5", "gram-a1", "gram-a2"],
                      "dinov2-vitb14")
    if body.strip():
        write_body(out / "cost.tex", body)
        written.append(out / "cost.tex")

    for path in written:
        print(f"wrote {path}")
    if not written:
        print("no tables written: the summary has no matching rows yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
