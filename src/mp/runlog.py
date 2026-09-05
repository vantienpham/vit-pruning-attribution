"""Run directories: config, environment and metrics as small JSON sidecars.

Every run writes three files beside whatever artefacts it produces, so a run can
be interpreted long after the job that made it, and so ``slurm/sync.sh pull``
can fetch the summaries of an entire campaign without touching the tensors.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch


def git_provenance(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Commit and branch, from ``.git_commit`` on the cluster or git locally.

    The remote tree is rsync'd without ``.git/``, so ``slurm/sync.sh push``
    leaves a stamp behind; without it a cluster run records no provenance.
    """
    root = repo_root or Path(__file__).resolve().parents[2]
    stamp = root / ".git_commit"

    if stamp.exists():
        fields = {}
        for line in stamp.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                fields[key] = value
        return {"source": ".git_commit", **fields}

    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"source": "git", "commit": commit, "dirty": str(dirty).lower()}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"source": "none"}


def environment() -> Dict[str, Any]:
    """What the run actually got, including the GPU and driver it landed on."""
    info: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
    }

    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        try:
            info["driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            ).splitlines()[0]
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    info["git"] = git_provenance()
    return info


class RunDirectory:
    """Creates ``run_dir`` and writes the three sidecars into it."""

    def __init__(self, run_dir: str | Path):
        self.path = Path(run_dir)
        self.path.mkdir(parents=True, exist_ok=True)

    def write_config(self, config: Dict[str, Any]) -> None:
        self._write("config.json", config)

    def write_environment(self) -> None:
        self._write("env.json", environment())

    def write_metrics(self, metrics: Dict[str, Any]) -> None:
        self._write("metrics.json", metrics)

    def is_complete(self, required_budgets: Optional[Iterable[str]] = None) -> bool:
        """Whether a previous job already finished this unit of work.

        Wall-time kills and requeues are the norm on a shared cluster, so a
        resubmitted campaign must skip what is on disk rather than redo it.
        Existence alone is not the test: metrics are rewritten after every
        budget so that a kill does not lose the whole run, which means a run
        stopped halfway leaves a file that looks finished. When the caller says
        which budgets it wants, they must all be present.
        """
        path = self.path / "metrics.json"
        if not path.exists():
            return False
        if required_budgets is None:
            return True
        try:
            with open(path) as handle:
                metrics = json.load(handle)
        except json.JSONDecodeError:
            return False
        present = set((metrics.get("budgets") or {}).keys())
        return set(required_budgets).issubset(present)

    def _write(self, name: str, payload: Dict[str, Any]) -> None:
        with open(self.path / name, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def read_metrics(run_dir: str | Path) -> Optional[Dict[str, Any]]:
    path = Path(run_dir) / "metrics.json"
    if not path.exists():
        return None
    with open(path) as handle:
        return json.load(handle)
