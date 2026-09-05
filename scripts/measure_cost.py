#!/usr/bin/env python
"""Time the calibration pass and the pruned encoders, on an idle GPU.

    srun --exclusive --gres=gpu:1 ... \
      uv run --no-sync python scripts/measure_cost.py --run-dir out/cost/dinov2

The campaign packs several jobs onto one node, which is right for throughput
and wrong for timing: a shared card makes every objective look slower and makes
the ratios between them depend on what else was running. Cost is therefore
measured once, here, with the device to itself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mp.data import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    PairedViewDataset,
    build_transform,
    build_views,
    load_calibration_images,
    make_loader,
)
from mp.evaluate import measure_throughput  # noqa: E402
from mp.models import (  # noqa: E402
    BUDGETS,
    PrunableViT,
    count_parameters,
    encoder_flops,
    load_backbone,
)
from mp.objectives import build_objective  # noqa: E402
from mp.runlog import RunDirectory  # noqa: E402
from mp.saliency import estimate_importance, make_teacher  # noqa: E402

OBJECTIVES = ["cosine", "mse", "cutvit", "gram-a0.25", "gram-a0.5", "gram-a1", "gram-a2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="dinov2-vitb14")
    parser.add_argument("--objectives", nargs="+", default=OBJECTIVES)
    parser.add_argument("--calibration-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--budgets", nargs="+", default=["s0", "s10", "s20", "s30", "s40"])
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run = RunDirectory(args.run_dir)
    run.write_config(vars(args))
    run.write_environment()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data_config = load_backbone(args.backbone, device=device, img_size=args.img_size)
    transform = build_transform(data_config)

    calibration = PairedViewDataset(
        load_calibration_images(
            "imagenet", transform=None, num_samples=args.calibration_samples,
            seed=0, root=args.data_root,
        ),
        views=build_views("two-crop", data_config),
    )
    loader = make_loader(calibration, batch_size=args.batch_size, num_workers=args.num_workers)

    metrics = {"backbone": args.backbone, "calibration": {}, "throughput": {}}

    for name in args.objectives:
        fresh, _ = load_backbone(args.backbone, device=device, img_size=args.img_size)
        prunable = PrunableViT(fresh, device=device)
        teacher = make_teacher(fresh)

        # One untimed pass over a couple of batches, so allocator warm-up and
        # kernel autotuning are not charged to the first objective in the list.
        warm = make_loader(
            torch.utils.data.Subset(calibration, list(range(min(32, len(calibration))))),
            batch_size=args.batch_size, num_workers=2,
        )
        estimate_importance(prunable, teacher, build_objective(name), warm, device,
                            progress=False)

        prunable.enable_importance_gradients()
        report = estimate_importance(
            prunable, teacher, build_objective(name), loader, device, progress=False
        )
        metrics["calibration"][name] = report.to_dict()
        print(f"{name:14s} {report.seconds:7.1f} s  "
              f"{report.peak_memory_bytes / 1024**3:5.2f} GB", flush=True)

        del prunable, teacher, fresh
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Throughput of the resulting encoders, which is what a deployment sees.
    importance = {
        "mlp": torch.rand(len(model.blocks), model.blocks[0].mlp.fc1.out_features),
        "head": torch.rand(len(model.blocks), model.blocks[0].attn.num_heads),
    }
    num_tokens = (args.img_size // model.patch_embed.patch_size[0]) ** 2 + getattr(
        model, "num_prefix_tokens", 1
    )

    for budget_key in args.budgets:
        fresh, _ = load_backbone(args.backbone, device=device, img_size=args.img_size)
        pruned = PrunableViT(fresh, device=device)
        pruned.prune(BUDGETS[budget_key], importance=importance)
        entry = measure_throughput(
            pruned.model, device=device, input_size=tuple(data_config["input_size"])
        )
        entry["parameters"] = count_parameters(pruned.model)
        entry["encoder_flops"] = encoder_flops(pruned.model, num_tokens=num_tokens)
        metrics["throughput"][budget_key] = entry
        print(f"{budget_key:14s} {entry['images_per_second']:8.1f} img/s  "
              f"{entry['parameters'] / 1e6:6.1f} M", flush=True)

        del pruned, fresh
        if device.type == "cuda":
            torch.cuda.empty_cache()

    run.write_metrics(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
