#!/usr/bin/env python3
"""Submit a study to Slurm, self-throttled below the per-user job cap.

Run this **on the login node**, detached, so it survives the laptop:

    cd <REMOTE_DIR>
    nohup uv run --no-sync python scripts/submit_campaign.py \
        --study all --max-jobs 9 > logs/submit.log 2>&1 &

The account allows 12 concurrent jobs and 40 queued. Over the cap, jobs pend
with ``AssocMaxJobsLimit``, which is not contention: waiting does not clear it,
only one's own jobs draining does. So the submitter polls and holds itself
below the limit rather than submitting the whole matrix at once.

Units already carrying a ``metrics.json`` for every seed are skipped, so a
resubmission after a wall-time kill costs only the work that is missing.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

#: Slurm partitions to place jobs on, as one comma-separated list. These names
#: are specific to the cluster the study was run on and must be replaced.
#: Choose Ampere or newer: a ViT-B calibration pass fits comfortably on any
#: modern card, but bfloat16 feature extraction is native only from sm_80, and a
#: run that lands on Volta is both slower and numerically different from the
#: rest of the campaign.
PARTITIONS = os.environ.get("MP_PARTITIONS", "gpul40s,gpu40G,gpu80G,gpuh100p")

#: The probes each run is scored on: general object recognition, two
#: fine-grained sets, a texture set, and a satellite set well outside the
#: pretraining distribution.
EVAL_DATASETS = ["imagenet100", "pets", "dtd", "flowers", "eurosat"]

BUDGETS = ["s0", "s05", "s10", "s20", "s30", "s40"]
SEEDS = [0, 1, 2]


@dataclass
class Unit:
    """One Slurm job: one configuration, every seed."""

    study: str
    backbone: str
    objective: str
    calibration: str = "imagenet"
    views: str = "two-crop"
    allocation: str = "global"
    budgets: Sequence[str] = field(default_factory=lambda: BUDGETS)
    seeds: Sequence[int] = field(default_factory=lambda: SEEDS)

    @property
    def name(self) -> str:
        return (f"{self.backbone}__{self.objective}__{self.calibration}"
                f"__{self.views}__{self.allocation}")

    @property
    def run_dir(self) -> str:
        return f"out/runs/{self.study}/{self.name}"

    def is_complete(self, root: Path) -> bool:
        """Every seed present *and* carrying every budget.

        Existence alone is not the test. Metrics are rewritten after each
        budget so a requeue does not lose the run, so a seed stopped halfway
        leaves a file that looks finished to a check on the path.
        """
        for seed in self.seeds:
            path = root / self.run_dir / f"seed{seed}" / "metrics.json"
            if not path.exists():
                return False
            try:
                with open(path) as handle:
                    metrics = json.load(handle)
            except json.JSONDecodeError:
                return False
            if not set(self.budgets).issubset((metrics.get("budgets") or {}).keys()):
                return False
        return True


# --------------------------------------------------------------------------- #
# The studies
# --------------------------------------------------------------------------- #

#: Objectives compared in the main table: three references that need no
#: calibration, two pointwise losses, the published subspace objective, and the
#: spectral family at four exponents.
MAIN_OBJECTIVES = [
    "random",
    "magnitude",
    "lamp",
    "cosine",
    "mse",
    "cutvit",
    "gram-a0.25",
    "gram-a0.5",
    "gram-a1",
    "gram-a2",
]

MAIN_BACKBONES = ["dinov2-vitb14", "dino-vitb16", "deit-vitb16"]


#: The encoder every study covers. The other two are used only to check that a
#: conclusion is not specific to one pretraining objective.
PRIMARY_BACKBONE = MAIN_BACKBONES[0]
SECONDARY_BACKBONES = MAIN_BACKBONES[1:]


def study_objectives() -> List[Unit]:
    """A: which alignment objective produces the better ranking."""
    return [Unit("objectives", PRIMARY_BACKBONE, o) for o in MAIN_OBJECTIVES]


def study_objectives_normalised() -> List[Unit]:
    """A': the same comparison with the cross-block scale removed.

    Under a raw global ranking the outcome is decided largely by how unevenly an
    objective's gradient magnitude varies with depth, which is a property of the
    objective but not the one the comparison is meant to be about. Dividing each
    block by its own mean score removes that, and the two arms together say how
    much of any margin is attributable to the ranking itself.
    """
    return [
        Unit("objectives-normalised", PRIMARY_BACKBONE, objective,
             allocation="block-normalised")
        for objective in MAIN_OBJECTIVES
    ]


def study_alpha_rank() -> List[Unit]:
    """B: separate the spectral exponent from the rank truncation."""
    objectives = [f"gram-a{a:g}-k192" for a in (0.0, 1.0, 2.0)]
    objectives += ["gram-a1-spatial", "gram-a1-channel", "gram-a1-only", "gram-a1-pooled"]
    objectives += ["cutvit-scaled", "cutvit-entropy"]
    return [Unit("alpha-rank", "dinov2-vitb14", objective) for objective in objectives]


def study_views() -> List[Unit]:
    """C: how much the teacher/student view mismatch decides the outcome."""
    return [
        Unit("views", "dinov2-vitb14", objective, views=views)
        for objective, views in itertools.product(
            ["cosine", "cutvit", "gram-a1"], ["noise", "identical"]
        )
    ]


def study_allocation() -> List[Unit]:
    """E: how much of the outcome is decided by the cross-block score scale."""
    return [
        Unit("allocation", "dinov2-vitb14", objective, allocation=allocation)
        for objective, allocation in itertools.product(
            ["cosine", "cutvit", "gram-a1"],
            ["linear-decay", "block-normalised", "uniform"],
        )
    ]


def study_calibration() -> List[Unit]:
    """D: whether calibrating on the target domain is what carries the gain."""
    return [
        Unit("calibration", "dinov2-vitb14", objective, calibration=calibration)
        for objective, calibration in itertools.product(
            ["cosine", "cutvit", "gram-a1"], ["pets", "dtd", "eurosat"]
        )
    ]


def study_backbones() -> List[Unit]:
    """F: whether the ordering survives a change of pretraining objective.

    Last in the order deliberately: it broadens a conclusion the other studies
    have to establish first, so it is the part to lose if the queue runs out.
    """
    return [
        Unit(study, backbone, objective, allocation=allocation)
        for backbone in SECONDARY_BACKBONES
        for objective in MAIN_OBJECTIVES
        for study, allocation in (
            ("objectives", "global"),
            ("objectives-normalised", "block-normalised"),
        )
    ]


#: Order matters: the submitter works through this list, so the studies the
#: argument rests on come before the ones that broaden it.
STUDIES = {
    "objectives": study_objectives,
    "objectives-normalised": study_objectives_normalised,
    "views": study_views,
    "allocation": study_allocation,
    "alpha-rank": study_alpha_rank,
    "calibration": study_calibration,
    "backbones": study_backbones,
}


# --------------------------------------------------------------------------- #
# Submission
# --------------------------------------------------------------------------- #


def queued_jobs(user: str) -> int:
    out = subprocess.run(
        ["squeue", "-u", user, "-h", "-o", "%i"], capture_output=True, text=True, check=True
    )
    return len([line for line in out.stdout.splitlines() if line.strip()])


def queued_names(user: str) -> set:
    """Job names already in the queue.

    A restarted submitter must not resubmit work that is in flight: two jobs
    writing the same run directory interleave their metrics and corrupt both.
    A run is only complete once it is on disk, so the queue is the other half
    of the picture.
    """
    out = subprocess.run(
        ["squeue", "-u", user, "-h", "-o", "%j"], capture_output=True, text=True, check=True
    )
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def job_name(unit: "Unit") -> str:
    """A name unique to the unit, so the queue check cannot confuse two of them.

    Units in the same study differ by calibration corpus, view protocol or
    allocation as well as by objective, so the name has to carry all of them.
    """
    return f"mp-{unit.study}-{unit.name}"


def sbatch(
    unit: Unit, data_root: str, imagenet_root: str, time_limit: str, dry_run: bool
) -> str:
    command = [
        "sbatch",
        f"--job-name={job_name(unit)}",
        "-p", PARTITIONS,
        f"--time={time_limit}",
        "--gres=gpu:1",
        "--cpus-per-task=8",
        f"--export=ALL,MP_DATA_ROOT={data_root},MP_IMAGENET_ROOT={imagenet_root}",
        "slurm/run.slurm",
        "scripts/run_prune.py",
        "--backbone", unit.backbone,
        "--objective", unit.objective,
        "--calibration", unit.calibration,
        "--views", unit.views,
        "--allocation", unit.allocation,
        "--seeds", *[str(s) for s in unit.seeds],
        "--budgets", *unit.budgets,
        "--eval-datasets", *EVAL_DATASETS,
        "--data-root", data_root,
        "--run-dir", unit.run_dir,
    ]
    if dry_run:
        return " ".join(command)
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="all", choices=[*STUDIES, "all"])
    parser.add_argument("--max-jobs", type=int, default=9,
                        help="hold the queue below this; the account cap is 12")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--time-limit", default="0-08:00:00")
    parser.add_argument("--data-root", default=os.path.expanduser("~/datasets"))
    parser.add_argument(
        "--imagenet-root",
        default=os.environ.get("MP_IMAGENET_ROOT",
                               os.path.expanduser("~/datasets/ImageNet")),
        help="ImageNet root holding train/ and val/ in ImageFolder layout",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    user = os.environ.get("USER", "")

    names = list(STUDIES) if args.study == "all" else [args.study]
    units = [unit for name in names for unit in STUDIES[name]()]

    pending = [unit for unit in units if not unit.is_complete(REPO)]
    print(f"{len(units)} units, {len(units) - len(pending)} already complete, "
          f"{len(pending)} to submit", flush=True)

    if args.dry_run:
        for unit in pending:
            print(sbatch(unit, args.data_root, args.imagenet_root, args.time_limit, dry_run=True))
        return 0

    for i, unit in enumerate(pending, start=1):
        while queued_jobs(user) >= args.max_jobs:
            print(f"queue full ({queued_jobs(user)}); waiting", flush=True)
            time.sleep(args.poll_seconds)

        if job_name(unit) in queued_names(user):
            print(f"[{i}/{len(pending)}] already queued, skipping {unit.run_dir}", flush=True)
            continue

        print(f"[{i}/{len(pending)}] "
              f"{sbatch(unit, args.data_root, args.imagenet_root, args.time_limit, False)} "
              f"-> {unit.run_dir}", flush=True)
        time.sleep(2)

    print("all units submitted", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
