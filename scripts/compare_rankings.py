#!/usr/bin/env python
"""Compare the importance rankings different objectives produce.

    uv run --no-sync python scripts/compare_rankings.py \
        --runs out/runs/objectives --run-dir out/ranking/dinov2

Downstream accuracy is an indirect and noisy way to ask whether two objectives
differ. The importance vectors are the direct object: if two objectives rank the
same units in the same order, no probe can separate them.

Two contrasts are reported, both held to one comparison per pair so that
neither is inflated by sharing a calibration draw:

  * **seed noise** -- one objective against itself on two different draws;
  * **objective effect** -- two objectives on the *same* draw.

If the second is not clearly larger than the first, the choice of objective
moves the ranking less than the choice of calibration images does.

Each is measured twice: over the flattened vector, which is dominated by the
cross-block scale, and within each block separately, which is the ranking the
objective is actually being asked to produce.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mp.runlog import RunDirectory  # noqa: E402


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Rank correlation between two score vectors."""
    ra = a.argsort().argsort().double()
    rb = b.argsort().argsort().double()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-12))


def within_block_spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean rank correlation computed separately inside each block."""
    return statistics.fmean(spearman(x, y) for x, y in zip(a, b))


def keep_overlap(a: torch.Tensor, b: torch.Tensor, ratio: float) -> float:
    """Jaccard overlap of the units each vector keeps at one sparsity."""
    flat_a, flat_b = a.flatten(), b.flatten()
    keep = flat_a.numel() - int(flat_a.numel() * ratio)
    top_a = set(flat_a.argsort(descending=True)[:keep].tolist())
    top_b = set(flat_b.argsort(descending=True)[:keep].tolist())
    union = len(top_a | top_b)
    return len(top_a & top_b) / union if union else 1.0


def load_importance(root: Path) -> Dict[Tuple[str, int], torch.Tensor]:
    """[blocks, hidden] MLP importance, keyed by (objective, seed)."""
    out: Dict[Tuple[str, int], torch.Tensor] = {}
    for metrics_path in sorted(root.rglob("metrics.json")):
        weights_path = metrics_path.parent / "importance.pt"
        if not weights_path.exists():
            continue
        with open(metrics_path) as handle:
            metrics = json.load(handle)
        if metrics.get("allocation") != "global":
            continue
        state = torch.load(weights_path, map_location="cpu")
        out[(metrics["objective"], metrics["seed"])] = state["mlp"].double()
    return out


def summarise(pairs: List[Tuple[torch.Tensor, torch.Tensor]], ratio: float) -> Dict[str, float]:
    if not pairs:
        return {}
    return {
        "spearman_flat": statistics.fmean(spearman(a.flatten(), b.flatten()) for a, b in pairs),
        "spearman_within_block": statistics.fmean(within_block_spearman(a, b) for a, b in pairs),
        "keep_overlap": statistics.fmean(keep_overlap(a, b, ratio) for a, b in pairs),
        "comparisons": len(pairs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="out/runs/objectives")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--ratio", type=float, default=0.25,
                        help="MLP pruning ratio at which the keep-overlap is measured")
    parser.add_argument(
        "--objectives", nargs="+",
        default=["cosine", "mse", "cutvit", "gram-a0.25", "gram-a0.5", "gram-a1", "gram-a2"],
        help="the alignment objectives; the data-free baselines are excluded because "
             "magnitude and LAMP do not depend on the calibration draw at all",
    )
    args = parser.parse_args()

    importance = load_importance(Path(args.runs))
    if not importance:
        print(f"no importance tensors under {args.runs}", file=sys.stderr)
        return 1

    seeds = sorted({seed for _, seed in importance})
    present = [o for o in args.objectives if any(o == n for n, _ in importance)]

    seed_pairs, objective_pairs = [], []
    for objective in present:
        for i, j in itertools.combinations(seeds, 2):
            if (objective, i) in importance and (objective, j) in importance:
                seed_pairs.append((importance[(objective, i)], importance[(objective, j)]))
    for a, b in itertools.combinations(present, 2):
        for seed in seeds:
            if (a, seed) in importance and (b, seed) in importance:
                objective_pairs.append((importance[(a, seed)], importance[(b, seed)]))

    metrics = {
        "ratio": args.ratio,
        "objectives": present,
        "seed_noise": summarise(seed_pairs, args.ratio),
        "objective_effect": summarise(objective_pairs, args.ratio),
        "per_pair": {},
    }
    for a, b in itertools.combinations(present, 2):
        pairs = [
            (importance[(a, s)], importance[(b, s)])
            for s in seeds
            if (a, s) in importance and (b, s) in importance
        ]
        metrics["per_pair"][f"{a}|{b}"] = summarise(pairs, args.ratio)

    RunDirectory(args.run_dir).write_metrics(metrics)
    RunDirectory(args.run_dir).write_config(vars(args))

    header = f"{'contrast':34s} {'flat rho':>9s} {'in-block rho':>13s} {'keep':>7s} {'n':>4s}"
    print(header)
    for name in ("seed_noise", "objective_effect"):
        e = metrics[name]
        if e:
            print(f"{name:34s} {e['spearman_flat']:9.3f} {e['spearman_within_block']:13.3f} "
                  f"{e['keep_overlap']:7.3f} {e['comparisons']:4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
