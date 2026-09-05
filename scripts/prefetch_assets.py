#!/usr/bin/env python3
"""Warm the caches this project needs. **Run on the login node.**

`scripts/prefetch.py` handles Hugging Face causal LMs; this one handles what
these experiments actually load: timm vision backbones and torchvision image
datasets. Compute nodes have no outbound internet, so anything not cached here
fails the job immediately under `HF_HUB_OFFLINE=1`.

    uv run --no-sync python scripts/prefetch_assets.py --all

Both guards from `prefetch.py` apply: this refuses to run under Slurm and
refuses to run from a host that cannot reach the network.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mp.data import DEFAULT_DATA_ROOT, EVAL_DATASETS  # noqa: E402
from mp.models import BACKBONES  # noqa: E402


def _reachable(host: str, port: int = 443, timeout: float = 5.0) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backbone", action="append", default=[], choices=sorted(BACKBONES))
    parser.add_argument("--dataset", action="append", default=[], choices=EVAL_DATASETS)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--all", action="store_true", help="every backbone and every dataset")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if os.environ.get("SLURM_JOB_ID"):
        print("REFUSING: running under Slurm; compute nodes have no internet.", file=sys.stderr)
        return 2
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        print("REFUSING: HF_HUB_OFFLINE=1, so nothing would download.", file=sys.stderr)
        return 2
    if not _reachable("huggingface.co"):
        print("REFUSING: cannot reach huggingface.co; run on the login node.", file=sys.stderr)
        return 2

    backbones = sorted(BACKBONES) if args.all else args.backbone
    datasets = EVAL_DATASETS if args.all else args.dataset

    if not backbones and not datasets:
        print("nothing to do: pass --backbone/--dataset, or --all", file=sys.stderr)
        return 2

    Path(args.data_root).mkdir(parents=True, exist_ok=True)
    failures = []

    for key in backbones:
        print(f"=== backbone: {key} ({BACKBONES[key]}) ===", flush=True)
        try:
            import timm
            import torch

            model = timm.create_model(BACKBONES[key], pretrained=True, num_classes=0)
            params = sum(p.numel() for p in model.parameters())
            print(f"  ok: {params / 1e6:.1f}M parameters", flush=True)
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as exc:
            failures.append(f"backbone {key}: {type(exc).__name__}: {exc}")
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    for name in datasets:
        print(f"=== dataset: {name} ===", flush=True)
        try:
            from torchvision import transforms as T

            from mp.data import load_eval_dataset

            splits = load_eval_dataset(
                name,
                transform=T.Compose([T.Resize(64), T.CenterCrop(64), T.ToTensor()]),
                root=args.data_root,
                allow_download=True,
            )
            print(
                f"  ok: {len(splits.train)} train / {len(splits.test)} test, "
                f"{splits.num_classes} classes",
                flush=True,
            )
        except Exception as exc:
            failures.append(f"dataset {name}: {type(exc).__name__}: {exc}")
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    if failures:
        print(f"\n{len(failures)} failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("\nAll cached. Jobs can now run with HF_HUB_OFFLINE=1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
