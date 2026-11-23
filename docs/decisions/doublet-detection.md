# Doublet detection

Rationale extracted from `mapping/doublets.py` docstrings, 2026-08-06.

## Per-sample, not joint

Doublets are a within-sample artifact of droplet loading. SOLO used to run in
the integration stage, on a model trained jointly across many samples — which
made a sample's doublet calls depend on which *other* samples were in that run,
so the same cells got different calls in different integration targets.

Running per sample also puts the calls in `.uns['qc']` / `.obs` before
`QCPlotter` builds its removal mask, so `solo.mode` in `filter_config.yaml`
decides whether they remove cells, exactly like every other filter method.

A second thing falls out of the same model, which is why this is one step and
not two: SOLO needs an scVI model, and that model's latent space is the best
embedding available for judging whether a blob of cells is a real population. So
`X_umap_scVI` comes back alongside the scores and the interactive report can
offer it for lasso selection.

## The subprocess boundary

The model trains in a subprocess under its own conda env (`doublets.env`), for
the same reason CellBender does: scvi-tools does not support the pandas 3.x the
mapping stage runs on. The exchange is narrow — a small h5ad of counts for the
QC-passed cells on HVGs only, and a TSV back. `integration/doublet_model.py`
runs on the far side.

That model is disposable (`persist=False`): nothing is written to disk and
nothing touches the integration registry. `scvi_tune_registry.json` has no file
locking, and forty samples registering throwaway models during one mapping run
would race. Its parameters are recorded in
`.uns['qc']['solo']['model_params']` instead.

Autotuning (`doublets.tune`) is off by default because it runs **per sample** —
on a 40-sample cohort that is a different order of job. Covariates are never
passed, tuned or not: `SOLO.from_scvi_model` raises on a model carrying any.

## Why the threshold is not SOLO's 0.5

**SOLO's fixed 0.5 systematically over-calls.** Its classifier trains on a 2:1
simulated-doublet:singlet set (scvi-tools' `doublet_ratio=2`, i.e.
`num_doublets = 2 * n_obs` in `SOLO.create_doublets`), so the softmax posterior
is calibrated to a two-thirds-doublet prior and 0.5 sits inside the real singlet
mass.

The pipeline instead fits a 2-component 1D Gaussian mixture on the doublet
probabilities — confident singlets pile near 0, confident doublets near 1 — and
cuts at the Bayes boundary between the recovered modes.
`min_component_separation` and `min_component_weight` guard against splitting a
single population; when they fail the code falls back and records that it did,
so a fallback is distinguishable from a real fit.

The same valley logic is reused for Scrublet's scores: both score families are
doublet-probability-like on [0, 1], only the guard thresholds differ.

## Three methods, one policy

| Method | Where it runs | Fails the run? |
|---|---|---|
| SOLO | subprocess, scvi env | yes, on worker fault |
| Scrublet | in-process (pure CPU, part of scanpy) | no — recorded as a skip |
| consensus | intersection of the two | no |

Scrublet needs **raw counts over all genes** for the QC-passed cells, not SOLO's
HVG subset and not a normalised layer: it simulates doublets by summing pairs of
real transcriptomes and runs its own variability filter and PCA internally.
Handing it SOLO's input would fight its own preprocessing.

A fixed nominal threshold is passed to `sc.pp.scrublet` deliberately — its
auto-threshold path (`threshold=None`) pulls in `scikit-image`, which is not in
the mapping env.

**Consensus is the intersection**, and exists precisely so SOLO's over-calling
cannot drive removal on its own. Cells either method did not assess are not
consensus doublets, since agreement is undefined there.

## Two independent downgrades

`filter` → `advisory` happens for two unrelated reasons, and the recorded reason
must say which:

1. **sample_quality flagged the sample** up front — too small or too shallow to
   trust its own distribution (`_effective_mode`).
2. **SOLO called an implausible fraction** of its input doublets, above
   `max_plausible_rate` (`_apply_solo`, applied once the rate is known).

A downgraded method still runs and still records scores and figures; it just
removes nothing.

## HVG selection

`_select_hvgs` uses the pipeline's consensus selector (`shared/hvg.py`), which
aggregates eight ranking methods by median rank, rather than scanpy's
`flavor='seurat_v3'` alone. seurat_v3 fits a loess of log variance against log
mean and goes singular on small populations — `2_tip`, at 163 cells against
52,689 genes, failed with an opaque Fortran message. The consensus selector
still runs `logmv_seurat_v3` as one of its eight, but a failure there no longer
takes the whole selection down.

The last-resort fallback is variance of log1p counts — no curve fit, so it
cannot fail the same way. It deliberately does *not* fall back to all genes:
training a latent model on 52,689 genes for 163 cells is both slow and
meaningless.

## Model parameters come from the worker, not the config

With `tune: True`, Ray's search space — not `doublet_cfg` — picks
n_latent/n_hidden/n_layers, plus several parameters that have no config key at
all (dropout_rate, dispersion, gene_likelihood, use_layer_norm, use_batch_norm,
encode_covariates). Re-deriving them from the config recorded the values that
were *requested and then ignored*, sitting next to `tuned: True` — an artifact
naming an architecture that was never built.

`params_source` records which branch produced the architecture, so a reader can
tell a trained-and-reported record from a fallback.

## Operational notes

**Worker fault vs. skip.** A SOLO worker fault is not a skip: the sample carries
on (doublet calls are additive), `doublets.done` is *not* written so the next run
retries, and the reason returns to `main.py` so the run exits 1. Recorded under
`failed`/`failure_reason` so the report distinguishes "this broke" from "we chose
not to run this". A Scrublet failure is a recorded skip instead — it is
in-process, deterministic and additive, so an edge case there should not fail an
otherwise-good sample.

**`min_cells` is a technical floor, not a quality judgement.** Below ~10 cells
scvi has no validation split and autotune dies in ray; well above that a
per-sample model still fails to converge and calls ~100% doublets.

**Worker stderr is captured, not inherited.** Under SLURM the parent's stderr
goes to `slurm-<jobid>.err` while the stage log gets only "exit 1", so the actual
cause ends up in a different file from the failure it explains. The tail budget
was raised from 12 lines to 120 after it cost a diagnosis: on SMu005 (10,980
cells) the worker died in ray with "No best trial found for the given metric",
whose cause is whatever made the individual trials fail — printed well above the
final traceback. A tuning run emits one error block per failed trial, so the
budget must clear several tracebacks.

**Temp files are left behind on failure** — they are the only record of what the
worker was given. Same policy as CellBender's.

**The three filter blocks are independent.** `collect_removal_flags` ORs the flag
columns of every method in `filter` mode, so two of them in `filter` is a union,
and only the consensus in `filter` is the intersection.
