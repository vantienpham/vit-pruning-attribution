#!/usr/bin/env python
"""Render the manuscript's figures from the pulled run directories.

    bash slurm/sync.sh pull
    uv run --no-sync python scripts/make_figures.py --runs out/runs --out redaction/figures

Reads ``metrics.json`` directly rather than the flattened CSV, because the
figures need the per-block widths and the spectral diagnostics that the tidy
table does not carry.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# A two-column page is narrow: keep type at the size the body text will be, so
# nothing has to be scaled down in the float and become unreadable.
plt.rcParams.update(
    {
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
    }
)

SERIES = [
    ("random", "Random", "#9e9e9e", ":", "s"),
    ("magnitude", "Magnitude", "#bcbd22", ":", "v"),
    ("cosine", "Pointwise cosine", "#1f77b4", "-", "o"),
    ("cutvit", r"Subspace ($\alpha$=0, K=192)", "#d62728", "-", "^"),
    ("gram-a0.5", r"Spectral $\alpha$=0.5", "#9467bd", "--", "D"),
    ("gram-a1", r"Spectral $\alpha$=1", "#2ca02c", "-", "P"),
    ("gram-a2", r"Spectral $\alpha$=2", "#ff7f0e", "--", "X"),
]

#: Legend text for the two-panel figure, where the column measure is 3.0in and
#: a longer label would force the legend below 6pt.
FIGURE_LABELS = {
    "cutvit": r"Subspace ($\alpha$=0)",
}

#: The single-column measure of the journal class, in inches: drawing at
#: exactly this width keeps 6pt legend text at 6pt in the float.
COLUMN_WIDTH = 216.24 / 72.27

BACKBONE_LABELS = {
    "dinov2-vitb14": "DINOv2 ViT-B/14",
    "dino-vitb16": "DINO ViT-B/16",
    "deit-vitb16": "DeiT ViT-B/16",
}


def load_runs(root: Path) -> List[Dict]:
    runs = []
    for path in sorted(root.rglob("metrics.json")):
        with open(path) as handle:
            runs.append(json.load(handle))
    return runs


def matching(runs: Iterable[Dict], **conditions) -> List[Dict]:
    return [r for r in runs if all(r.get(k) == v for k, v in conditions.items())]


def curve(
    runs: List[Dict],
    backbone: str,
    objective: str,
    datasets: Sequence[str],
    metric: str = "linear",
) -> Tuple[List[float], List[float], List[float]]:
    """Mean over datasets and seeds, against measured parameter sparsity."""
    selected = matching(
        runs,
        backbone=backbone,
        objective=objective,
        calibration="imagenet",
        views="two-crop",
        allocation="global",
    )
    by_budget: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for run in selected:
        for budget, entry in run.get("budgets", {}).items():
            scores = [
                entry["datasets"][d][metric]
                for d in datasets
                if d in entry.get("datasets", {})
            ]
            if len(scores) == len(datasets):
                by_budget[budget].append(
                    (entry["parameter_sparsity"], statistics.fmean(scores))
                )

    order = ["s0", "s05", "s10", "s15", "s20", "s30", "s40"]
    xs, ys, errs = [], [], []
    for budget in order:
        if budget not in by_budget:
            continue
        values = by_budget[budget]
        xs.append(statistics.fmean(x for x, _ in values))
        ys.append(100 * statistics.fmean(y for _, y in values))
        errs.append(
            100 * statistics.stdev([y for _, y in values]) if len(values) > 1 else 0.0
        )
    return xs, ys, errs


def figure_tradeoff(runs: List[Dict], datasets: Sequence[str], out: Path, metric: str) -> Optional[Path]:
    backbones = [b for b in BACKBONE_LABELS if matching(runs, backbone=b)]
    if not backbones:
        return None

    fig, axes = plt.subplots(
        1, len(backbones), figsize=(3.3 * len(backbones) / 1.6, 1.9), sharey=False
    )
    axes = [axes] if len(backbones) == 1 else list(axes)

    for ax, backbone in zip(axes, backbones):
        for objective, label, colour, style, marker in SERIES:
            xs, ys, errs = curve(runs, backbone, objective, datasets, metric)
            if not xs:
                continue
            ax.errorbar(
                [100 * x for x in xs], ys, yerr=errs, label=label,
                color=colour, linestyle=style, marker=marker, capsize=1.2, elinewidth=0.5,
            )
        ax.set_title(BACKBONE_LABELS[backbone])
        ax.set_xlabel("parameter sparsity (%)")
        ax.grid(alpha=0.25, linewidth=0.4)
    axes[0].set_ylabel("linear probe (%)" if metric == "linear" else "$k$-NN accuracy (%)")
    axes[-1].legend(frameon=False, loc="lower left", ncol=1)

    path = out / "tradeoff.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def _retained(runs, backbone, objective, allocation, datasets, budget, metric):
    """Mean retained score over datasets and seeds, or None."""
    unpruned, pruned = [], []
    for run in matching(runs, backbone=backbone, objective=objective,
                        calibration="imagenet", views="two-crop", allocation=allocation):
        budgets = run.get("budgets", {})
        if budget not in budgets or "s0" not in budgets:
            continue
        base = budgets["s0"].get("datasets", {})
        cut = budgets[budget].get("datasets", {})
        if not all(d in base and d in cut for d in datasets):
            continue
        unpruned.append([base[d][metric] for d in datasets])
        pruned.append([cut[d][metric] for d in datasets])
    if not pruned:
        return None
    ratios = [
        statistics.fmean(p / b for p, b in zip(pr, un) if b > 0)
        for pr, un in zip(pruned, unpruned)
    ]
    return 100 * statistics.fmean(ratios)


def figure_structure(runs, datasets, out: Path, metric: str = "linear") -> Optional[Path]:
    """Why the depth allocation dominates, in two panels.

    Left: how many MLP units each block keeps under a raw global ranking. Right:
    what removing the cross-block scale is worth, objective by objective.
    """
    # The left panel draws four of the seven series; the shared legend under
    # the figure names all of them, so the two panels can be read against each
    # other without a colour appearing that nothing explains.
    profile_objectives = ["random", "cosine", "cutvit", "gram-a1"]
    style = {name: (FIGURE_LABELS.get(name, label), colour, dash, marker)
             for name, label, colour, dash, marker in SERIES}

    profiles = {}
    for objective in profile_objectives:
        widths = [
            run["budgets"]["s20"]["widths"]["hidden_dims"]
            for run in matching(runs, backbone="dinov2-vitb14", objective=objective,
                                calibration="imagenet", views="two-crop",
                                allocation="global")
            if "s20" in run.get("budgets", {})
        ]
        if widths:
            profiles[objective] = [statistics.fmean(c) for c in zip(*widths)]

    paired = []
    for objective, label, colour, _dash, _marker in SERIES:
        g = _retained(runs, "dinov2-vitb14", objective, "global", datasets, "s20", metric)
        n = _retained(runs, "dinov2-vitb14", objective, "block-normalised", datasets,
                      "s20", metric)
        if g is not None and n is not None:
            paired.append((objective, g, n))

    if not profiles and not paired:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 1.58))
    fig.subplots_adjust(wspace=0.58, bottom=0.47, top=0.97)

    handles = {}
    # The alignment profiles very nearly coincide, which is the point of the
    # panel; distinct dashes and markers keep the hidden one legible under it.
    for objective, profile in profiles.items():
        label, colour, dash, marker = style[objective]
        line, = axes[0].plot(range(1, len(profile) + 1), profile, marker=marker,
                             linestyle=dash, color=colour, label=label,
                             markerfacecolor="none", markeredgewidth=0.7)
        handles[objective] = line
    axes[0].set_xlabel("block")
    axes[0].set_ylabel("MLP units kept")
    axes[0].grid(alpha=0.25, linewidth=0.4)

    if paired:
        lo = min(min(g, n) for _, g, n in paired) - 3
        hi = max(max(g, n) for _, g, n in paired) + 3
        axes[1].plot([lo, hi], [lo, hi], color="#999999", linewidth=0.6, linestyle="--")
        for objective, g, n in paired:
            label, colour, dash, marker = style[objective]
            point = axes[1].scatter([g], [n], color=colour, marker=marker, s=14,
                                    zorder=3, label=label)
            handles.setdefault(objective, point)
        axes[1].set_xlabel("global ranking (%)")
        axes[1].set_ylabel("block-normalised (%)")
        axes[1].grid(alpha=0.25, linewidth=0.4)
    else:
        axes[1].set_axis_off()

    ordered = [handles[name] for name, *_ in SERIES if name in handles]
    fig.legend(ordered, [h.get_label() for h in ordered], loc="lower center",
               ncol=3, frameon=False, fontsize=6, handlelength=1.6,
               columnspacing=0.8, handletextpad=0.4, bbox_to_anchor=(0.5, 0.0))

    path = out / "structure.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="out/runs")
    parser.add_argument("--out", default="redaction/figures",
                        help="where the figures the manuscript includes are written")
    parser.add_argument(
        "--diagnostics-out", default="results/figures",
        help="where figures not included in the manuscript go; keeping them out "
             "of redaction/figures stops an orphan reaching the submission bundle",
    )
    parser.add_argument("--metric", default="linear", choices=["linear", "knn"])
    parser.add_argument(
        "--datasets", nargs="+",
        default=["imagenet100", "pets", "dtd", "flowers", "eurosat"],
    )
    args = parser.parse_args()

    plt.rcParams["text.usetex"] = False
    runs = load_runs(Path(args.runs))
    if not runs:
        print(f"no runs under {args.runs}")
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    diagnostics = Path(args.diagnostics_out)
    diagnostics.mkdir(parents=True, exist_ok=True)

    for maker in (
        lambda: figure_tradeoff(runs, args.datasets, diagnostics, args.metric),
        lambda: figure_structure(runs, args.datasets, out, args.metric),
    ):
        path = maker()
        print(f"wrote {path}" if path else "skipped a figure: no matching runs yet")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
