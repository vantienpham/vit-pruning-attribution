#!/usr/bin/env python
"""Prune one backbone with one alignment objective, then score every budget.

    uv run --no-sync python scripts/run_prune.py \
        --backbone dinov2-vitb14 --objective gram-a1 --calibration imagenet \
        --seeds 0 1 2 --run-dir out/runs/dinov2-gram-a1-imagenet

The importance scores are computed once and reused for every sparsity level,
which is what makes one-shot pruning cheap: a single calibration pass yields a
whole family of subnetworks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mp.data import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    VIEW_PROTOCOLS,
    PairedViewDataset,
    build_transform,
    build_views,
    cap_splits,
    load_calibration_images,
    load_eval_dataset,
    make_loader,
)
from mp.evaluate import (  # noqa: E402
    dense_retrieval_score,
    evaluate_backbone,
    measure_throughput,
)
from mp.models import (  # noqa: E402
    ALLOCATIONS,
    BUDGETS,
    PrunableViT,
    count_parameters,
    encoder_flops,
    load_backbone,
)
from mp.objectives import build_objective, objective_names  # noqa: E402
from mp.runlog import RunDirectory  # noqa: E402
from mp.saliency import (  # noqa: E402
    estimate_importance,
    lamp_importance,
    magnitude_importance,
    make_teacher,
    random_importance,
    spectrum_diagnostics,
)

#: Importance baselines that need no alignment objective and no calibration pass.
DATA_FREE = {"magnitude", "random", "lamp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True)
    parser.add_argument(
        "--objective",
        required=True,
        help=f"one of {sorted(DATA_FREE)} or an alignment objective: {objective_names()}",
    )
    parser.add_argument("--calibration", default="imagenet")
    parser.add_argument(
        "--views",
        default="two-crop",
        choices=sorted(VIEW_PROTOCOLS),
        help="how the teacher and student inputs are made to differ; with "
        "identical views every alignment loss and its gradient are exactly zero",
    )
    parser.add_argument("--calibration-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="one calibration draw per seed; each writes its own run directory",
    )
    parser.add_argument("--budgets", nargs="+", default=["s10", "s20", "s30", "s40"])
    parser.add_argument("--eval-datasets", nargs="+", default=["cifar100", "pets", "dtd"])
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="extract features in float32; bfloat16 otherwise, which halves the "
        "cost of the dominant stage and is applied identically to every row",
    )
    parser.add_argument("--linear-epochs", type=int, default=100)
    parser.add_argument(
        "--max-eval-train",
        type=int,
        default=8000,
        help="cap on each probe's training split; feature extraction dominates "
        "the campaign cost and a linear probe saturates well below this",
    )
    parser.add_argument("--max-eval-test", type=int, default=5000)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--allocation",
        default="global",
        choices=list(ALLOCATIONS),
        help="how the global unit budget is spread across blocks",
    )
    parser.add_argument("--min-hidden-ratio", type=float, default=0.05)
    parser.add_argument("--min-head-ratio", type=float, default=0.2)
    parser.add_argument(
        "--measure-throughput",
        action="store_true",
        help="time the pruned encoders; only meaningful when the job owns the GPU",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in args.seeds:
        run_dir = Path(args.run_dir) / f"seed{seed}"
        if RunDirectory(run_dir).is_complete(args.budgets) and not args.overwrite:
            print(f"{run_dir} already has metrics.json; skipping")
            continue
        run_one(args, seed=seed, run_dir=run_dir, device=device)
    return 0


def run_one(args: argparse.Namespace, seed: int, run_dir: Path, device: torch.device) -> None:
    run = RunDirectory(run_dir)
    torch.manual_seed(seed)

    run.write_config({**vars(args), "seed": seed})
    run.write_environment()

    model, data_config = load_backbone(args.backbone, device=device, img_size=args.img_size)
    transform = build_transform(data_config)

    baseline_params = count_parameters(model)
    num_tokens = (args.img_size // model.patch_embed.patch_size[0]) ** 2 + getattr(
        model, "num_prefix_tokens", 1
    )
    baseline_flops = encoder_flops(model, num_tokens=num_tokens)

    metrics = {
        "backbone": args.backbone,
        "objective": args.objective,
        "calibration": args.calibration,
        "views": args.views,
        "allocation": args.allocation,
        "seed": seed,
        "baseline": {
            "parameters": baseline_params,
            "encoder_flops": baseline_flops,
            "num_tokens": num_tokens,
        },
        "budgets": {},
    }

    prunable = PrunableViT(
        model,
        device=device,
        min_hidden_ratio=args.min_hidden_ratio,
        min_head_ratio=args.min_head_ratio,
    )

    # ---- importance ------------------------------------------------------- #
    if args.objective in DATA_FREE:
        if args.objective == "magnitude":
            importance = magnitude_importance(prunable)
        elif args.objective == "lamp":
            importance = lamp_importance(prunable)
        else:
            importance = random_importance(prunable, seed=seed)
        metrics["saliency"] = {"seconds": 0.0, "peak_memory_gb": 0.0, "data_free": True}
    else:
        objective = build_objective(args.objective)
        metrics["objective_config"] = objective.config.to_dict()

        calibration = PairedViewDataset(
            load_calibration_images(
                args.calibration,
                transform=None,
                num_samples=args.calibration_samples,
                seed=seed,
                root=args.data_root,
            ),
            views=build_views(args.views, data_config),
        )
        calibration_loader = make_loader(
            calibration, batch_size=args.batch_size, num_workers=args.num_workers
        )
        print(
            f"calibrating on {len(calibration)} images from {args.calibration} "
            f"under the {args.views} view protocol"
        )

        teacher = make_teacher(model)
        report = estimate_importance(
            prunable,
            teacher=teacher,
            objective=objective,
            loader=calibration_loader,
            device=device,
        )
        metrics["saliency"] = report.to_dict()
        metrics["saliency"]["diagnostics"] = spectrum_diagnostics(
            teacher,
            calibration_loader,
            device=device,
            tokens=objective.config.tokens,
        )
        importance = prunable.importance_state()

    torch.save(importance, run_dir / "importance.pt")

    # ---- evaluation datasets ---------------------------------------------- #
    eval_splits = {
        name: cap_splits(
            load_eval_dataset(name, transform=transform, root=args.data_root),
            max_train=args.max_eval_train,
            max_test=args.max_eval_test,
        )
        for name in args.eval_datasets
    }

    dense_loader = make_loader(
        load_calibration_images(
            args.calibration,
            transform=transform,
            num_samples=256,
            seed=seed + 10_000,
            root=args.data_root,
        ),
        batch_size=16,
        num_workers=args.num_workers,
    )
    dense_teacher = make_teacher(model)

    # ---- one subnetwork per budget ---------------------------------------- #
    for budget_key in args.budgets:
        budget = BUDGETS[budget_key]
        pruned = PrunableViT(
            load_backbone(args.backbone, device=device, img_size=args.img_size)[0],
            device=device,
            min_hidden_ratio=args.min_hidden_ratio,
            min_head_ratio=args.min_head_ratio,
        )
        pruned.prune(budget, importance=importance, allocation=args.allocation)

        entry = {
            "mlp_ratio": budget.mlp_ratio,
            "head_ratio": budget.head_ratio,
            "parameters": count_parameters(pruned.model),
            "encoder_flops": encoder_flops(pruned.model, num_tokens=num_tokens),
            "widths": pruned.widths(),
            "datasets": {},
        }
        entry["parameter_sparsity"] = 1.0 - entry["parameters"] / baseline_params
        entry["flop_sparsity"] = 1.0 - entry["encoder_flops"] / baseline_flops

        for name, splits in eval_splits.items():
            entry["datasets"][name] = evaluate_backbone(
                pruned.model,
                splits,
                device=device,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                linear_epochs=args.linear_epochs,
                seed=seed,
                amp=not args.no_amp,
            )
            print(f"[{budget_key}] {name}: {entry['datasets'][name]}")

        entry.update(
            dense_retrieval_score(pruned.model, dense_teacher, dense_loader, device=device)
        )

        if args.measure_throughput:
            entry["throughput"] = measure_throughput(
                pruned.model, device=device, input_size=tuple(data_config["input_size"])
            )

        metrics["budgets"][budget_key] = entry
        run.write_metrics(metrics)  # checkpoint after every budget

        del pruned
        if device.type == "cuda":
            torch.cuda.empty_cache()

    run.write_metrics(metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k != "budgets"}, indent=2, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
