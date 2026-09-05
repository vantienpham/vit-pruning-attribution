# vit-pruning-attribution

Code and experiment definitions for **"The alignment objective is not what
decides training-free pruning of vision transformers."**

Training-free one-shot structured pruning of a vision transformer ranks MLP
hidden units and attention heads by the squared gradient of a teacher–student
alignment loss, accumulated over a few hundred unlabelled calibration images.
Recent work attributes its gains to the choice of that loss, and in particular
to aligning the leading eigenspaces of the feature **Gram matrices**.

This repository implements a **spectral family of Gram alignment objectives**,

```
L(α, K) = 1 − ⟨φ(G_t), φ(G_p)⟩_F / (‖φ(G_t)‖_F ‖φ(G_p)‖_F),
          φ(G; α, K) = U_K diag(σ_K^α) U_Kᵀ,
```

which contains the published **subspace / chordal-distance** constraint at
`α = 0` (where it equals the normalised squared chordal distance on the
Grassmannian) and **linear CKA** at `α = 1`, together with the campaign that
measures what actually decides the outcome.

The headline result is one of attribution. Once the cross-block gradient scale
is removed, seven objectives spanning pointwise matching, subspace alignment and
the spectral family fall within 4.4 points of each other, and the two components
no prior work isolates — the **depth allocation** and the teacher–student **view
protocol** — move the results by up to 25 points. Under the ranking published
methods use, every gradient-based objective falls below random pruning past 18%
sparsity.

## Two facts worth knowing before building on this

1. **The calibration pass needs two different views.** The unpruned network is an
   exact global minimum of every objective in the family, so a pass in which
   teacher and student read the same pixels has a gradient of *exactly* zero and
   yields a ranking made of floating-point residue. `--views` selects how the two
   views are made to differ; `identical` is retained as a control.
2. **Fractional exponents return NaN gradients through an eigendecomposition.**
   A transformer Gram matrix has an effective rank of a few dozen out of several
   hundred, so the `1/(σ_i − σ_j)` terms in the backward pass are enormous. The
   forward pass looks ordinary and `argsort` places NaN arbitrarily, so the
   pruning that follows produces a plausible network and a plausible number.
   Halves and quarters therefore use a Newton–Schulz iteration built from matrix
   products alone, and the saliency pass refuses a non-finite score.

## Quickstart

```bash
uv sync --extra dev
uv run --no-sync python -m pytest tests/ -q     # ~3 s, CPU only
```

The suite checks the claims the paper proves: that the `α = 0` member equals the
chordal subspace distance, that every member is basis-invariant, that the two
Gram entropies coincide (so the spectral-entropy axis weighting is the constant
½), and that identical views give a zero gradient.

One configuration, every seed and every budget:

```bash
uv run --no-sync python scripts/run_prune.py \
    --backbone dinov2-vitb14 --objective gram-a1 --allocation block-normalised \
    --calibration imagenet --views two-crop --seeds 0 1 2 \
    --budgets s0 s10 s20 s30 --eval-datasets imagenet100 pets dtd \
    --run-dir out/runs/example
```

## The tree

| path | what |
|---|---|
| `src/mp/objectives.py` | the objective family and the propositions it rests on |
| `src/mp/models.py` | backbone loading, structural pruning surgery, depth allocation |
| `src/mp/saliency.py` | the one-shot calibration pass and the data-free baselines |
| `src/mp/data.py` | calibration corpora, evaluation splits, view protocols |
| `src/mp/evaluate.py` | k-NN, linear probe, dense correspondence, throughput |
| `scripts/run_prune.py` | one configuration, every seed and budget |
| `scripts/submit_campaign.py` | the studies, self-throttled below a Slurm job cap |
| `scripts/aggregate.py` | run directories → tidy CSV → summary CSV |
| `scripts/make_tables.py`, `make_figures.py`, `make_spectrum_table.py` | summary → the `.tex` fragments the manuscript reads |
| `scripts/analyse_spectrum.py` | Gram spectra, entropy, effective rank |
| `scripts/compare_rankings.py` | objectives compared by their importance vectors, without a probe |
| `scripts/measure_cost.py` | calibration cost and throughput on an idle GPU |
| `scripts/audit_runs.py`, `verify_numbers.py` | which runs are unfinished; every quantity the paper quotes |
| `slurm/` | job runner, environment build, and the only sanctioned rsync |

**No number reaches the manuscript by hand.** Runs write JSON sidecars,
`aggregate.py` flattens them, and `make_*.py` render the `.tex` fragments the
paper reads. The generators default to writing under `redaction/`, which is
where the manuscript lives in the authors' tree; pass `--out` to redirect them.

## Reproducing the campaign

`scripts/submit_campaign.py` defines every study in the paper — the objective
comparison under both allocations, the exponent/rank plane, the view protocol,
the depth allocation and the calibration corpus — and submits them while holding
itself below a per-user Slurm job cap.

```bash
uv run --no-sync python scripts/submit_campaign.py --study all --dry-run
```

`slurm/` and the partition names in `submit_campaign.py` are specific to the
cluster this was run on and **must be adapted**: set `REMOTE_USER`,
`REMOTE_HOST` and `REMOTE_DIR` in `slurm/sync.sh`, and either edit `PARTITIONS`
or set `MP_PARTITIONS`. `cluster.example.md` is a template for recording what a
given cluster actually provides. `MP_DATA_ROOT` and `MP_IMAGENET_ROOT` locate
the datasets; ImageNet is expected in the usual `train/` and `val/` ImageFolder
layout.

Backbones resolve through `timm` and every evaluation dataset is public, so the
pipeline needs no gated download. Compute nodes without outbound internet should
warm the caches first with `scripts/prefetch_assets.py`, run on a node that can
reach the network.

## Citation

The paper is under review; this entry will be updated when it appears.

```bibtex
@misc{pham2026alignment,
  title  = {The alignment objective is not what decides training-free pruning
            of vision transformers},
  author = {Pham, Van Tien},
  year   = {2026}
}
```

## Licence

MIT, see [LICENSE](LICENSE).

The pipeline this study analyses was introduced by SnapViT (Simoncini et al.,
NeurIPS 2025) and Cut-ViT (Yin et al., ECCV 2026); neither is vendored here.
