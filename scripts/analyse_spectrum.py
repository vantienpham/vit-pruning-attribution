#!/usr/bin/env python
"""Measure the Gram spectra the alignment objectives operate on.

    uv run --no-sync python scripts/analyse_spectrum.py \
        --backbone dinov2-vitb14 --run-dir out/spectrum/dinov2

The exponent argument of Section~4 turns on how concentrated these spectra are:
if the energy sits in a handful of directions, then a rank-$K$ projector with
$K$ far above that number spends most of its weight on directions that carry
almost nothing. This records the spectrum, its entropy, the effective rank
$\\exp(\\mathbb{H})$, and the cumulative explained variance, so that argument is
made against measurements rather than assertion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mp.data import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    build_transform,
    load_calibration_images,
    make_loader,
)
from mp.models import load_backbone, token_features  # noqa: E402
from mp.objectives import gram, top_eigenbasis  # noqa: E402
from mp.runlog import RunDirectory  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="dinov2-vitb14")
    parser.add_argument("--calibration", default="imagenet")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    run = RunDirectory(args.run_dir)
    run.write_config(vars(args))
    run.write_environment()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data_config = load_backbone(args.backbone, device=device, img_size=args.img_size)
    transform = build_transform(data_config)

    loader = make_loader(
        load_calibration_images(
            args.calibration, transform=transform, num_samples=args.samples,
            seed=0, root=args.data_root,
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Both poolings are measured: the per-image form is the one the published
    # equations define, the batch-pooled form is the one the released code
    # computes, and the two are not the same matrix.
    totals = {
        ("spatial", "per_image"): None,
        ("channel", "per_image"): None,
        ("spatial", "batch_pooled"): None,
        ("channel", "batch_pooled"): None,
    }
    count = 0

    for images in loader:
        images = images.to(device, non_blocking=True)
        features = token_features(model, images, tokens="patch").float()
        for axis, pooling in totals:
            g = gram(features, axis=axis, pooling=pooling)
            values, _ = top_eigenbasis(g, 10_000)
            # Normalise before averaging, so one high-energy batch does not
            # decide the shape of the mean spectrum.
            values = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            summed = values.reshape(-1, values.shape[-1]).sum(dim=0).cpu()
            weight = values.reshape(-1, values.shape[-1]).shape[0]
            key = (axis, pooling)
            totals[key] = (summed, weight) if totals[key] is None else (
                totals[key][0] + summed, totals[key][1] + weight
            )
        count += images.shape[0]

    metrics = {"backbone": args.backbone, "num_images": count, "axes": {}}

    for (axis, pooling), (summed, weight) in totals.items():
        spectrum = (summed / weight).double()
        spectrum = spectrum / spectrum.sum()
        entropy = float(-(spectrum * spectrum.clamp_min(1e-300).log()).sum())
        cumulative = spectrum.cumsum(0)

        metrics["axes"][f"{axis}-{pooling}"] = {
            "rank": int(spectrum.numel()),
            "entropy_nats": entropy,
            "effective_rank": float(torch.exp(torch.tensor(entropy))),
            "max_entropy_nats": float(torch.log(torch.tensor(float(spectrum.numel())))),
            "cevr": {
                str(k): float(cumulative[min(k, spectrum.numel()) - 1])
                for k in (1, 4, 8, 16, 32, 64, 128, 192, 256, 512)
                if k <= spectrum.numel()
            },
            # How many leading directions it takes to reach a given share of
            # the energy: the honest answer to "how big is the subspace".
            "rank_for": {
                f"{q:g}": int((cumulative < q).sum()) + 1
                for q in (0.9, 0.95, 0.99, 0.999)
            },
            "spectrum_head": [float(x) for x in spectrum[:512]],
        }

    run.write_metrics(metrics)

    for axis, entry in metrics["axes"].items():
        print(
            f"{axis:22s} rank={entry['rank']:4d}  "
            f"H={entry['entropy_nats']:.2f} nats (max {entry['max_entropy_nats']:.2f})  "
            f"effective rank={entry['effective_rank']:.1f}  "
            f"90%/99% energy at {entry['rank_for']['0.9']}/{entry['rank_for']['0.99']} "
            f"directions  CEVR@192={entry['cevr'].get('192', float('nan')):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
