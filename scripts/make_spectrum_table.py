#!/usr/bin/env python
"""Render the Gram-spectrum table from the spectrum runs.

    uv run --no-sync python scripts/make_spectrum_table.py \
        --runs out/spectrum --out redaction/tables

Reports, per calibration corpus, the spectral entropy of each Gram matrix under
both poolings, the effective rank exp(H) that entropy implies, and the entropy
weight the axis rule derives from it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

#: Directory suffix to the corpus it was measured on.
CORPUS_LABELS = {
    "dinov2-vitb14": "ImageNet",
    "dinov2-pets": "Pets",
    "dinov2-dtd": "DTD",
    "dinov2-flowers": "Flowers",
    "dinov2-eurosat": "EuroSAT",
}

ORDER = ["dinov2-vitb14", "dinov2-pets", "dinov2-dtd", "dinov2-flowers", "dinov2-eurosat"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="out/spectrum")
    parser.add_argument("--out", default="redaction/tables")
    args = parser.parse_args()

    root = Path(args.runs)
    rows = []

    for key in ORDER:
        path = root / key / "metrics.json"
        if not path.exists():
            continue
        with open(path) as handle:
            axes = json.load(handle)["axes"]

        def entry(name: str) -> Dict:
            return axes.get(name, {})

        per_s, per_c = entry("spatial-per_image"), entry("channel-per_image")
        pool_s, pool_c = entry("spatial-batch_pooled"), entry("channel-batch_pooled")
        if not (per_s and per_c and pool_s and pool_c):
            continue

        weight_per = per_s["entropy_nats"] / (per_s["entropy_nats"] + per_c["entropy_nats"])
        weight_pool = pool_s["entropy_nats"] / (pool_s["entropy_nats"] + pool_c["entropy_nats"])

        rows.append(
            f"{CORPUS_LABELS[key]} & "
            f"{per_s['entropy_nats']:.2f} & {per_c['entropy_nats']:.2f} & "
            f"{per_s['effective_rank']:.0f} & {weight_per:.3f} & "
            f"{pool_s['entropy_nats']:.2f} & {pool_c['entropy_nats']:.2f} & "
            f"{weight_pool:.3f}" + r" \\"
        )

    if not rows:
        print("no spectrum runs found")
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # No trailing row terminator: TeX mishandles a "\\" at the end of an
    # \input file inside an alignment and blames the rule that follows.
    body = "\n".join(rows).rstrip()
    if body.endswith("\\\\"):
        body = body[:-2].rstrip()
    (out / "spectra.tex").write_text(body + "%\n")
    print(f"wrote {out / 'spectra.tex'}")
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
