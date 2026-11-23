# Why contrastiveVI is swept rather than autotuned

Rationale extracted from `integration/orchestrator.py::_sweep_contrast`,
2026-08-06. `integration/contrastive.py` and `cvi_sweep.py` hold the
implementation.

## Autotuning selects the leak

The sweep is what `tune: true` would do if contrastiveVI could be autotuned. It
cannot.

Its objective is a sum of a target and a background ELBO **with no term
rewarding salient/background separation**. So raising `n_salient_latent`
strictly improves reconstruction, and any nonzero `wasserstein_penalty` strictly
worsens the objective. An `elbo_validation` search therefore selects maximum
capacity and zero penalty — which *is* the leak the penalty exists to prevent.

Every criterion in the sweep is consequently computed **outside** the objective;
see `contrastive.select_contrast_grid` and `rank_contrast_grid`.

## `base_model_params` is not optional, despite its default

Fixed 2026-08-04; it had never been passed by either caller.

A grid point overrides only `wasserstein_penalty` and `n_salient_latent`. Every
other architecture key falls through to `contrastive._MODEL_DEFAULTS` when no
base is given — and that default is `n_layers: 1`, while every CH5 target states
2.

So the grid was ranking penalties on a **one-layer** model, and the winner was
then merged into the target's own params and trained at **two**. Depth and the
Wasserstein penalty both govern how much shared variation reaches the salient
space, so selecting one at a depth you do not train at is not a second-order
mismatch. The base the winner will be trained with is passed now.

## A sweep that cannot answer must not also decide

On a grid where every point failed or collapsed (salient space reduced to a
group indicator), `best_grid_point` returns nothing and the target's own stated
`model_params` are used unchanged — the run continues, with a warning naming the
grid TSV on disk for whoever reads it.

The winning point, the grid size and the TSV path are written to the registry so
an automatic choice stays auditable.

## It runs only when training will actually happen

`_will_train_contrast` mirrors the reload condition in
`ContrastiveIntegrator.integrate()`. `integrate()` returns a cached model when
one exists and `--overwrite` was not passed, so sweeping first would burn a full
grid, throw the result away, and leave a fresh grid TSV on disk next to a model
that was not chosen from it.

**The two checks are deliberately in step: if one changes, the other must.**

## The sweep trains on a copy

`select_contrast_grid` copies per grid point, but the caller's `adata` must also
survive untouched for the final fit, since `_prep` overwrites `.X` and stamps
`_scvi_*` columns.

## `separation` alone cannot rank the grid

The 2026-08-03 CH5 sweep is the demonstration: at every `wasserstein_penalty >= 1`,
on **all five contrasts**, salient iLISI hit exactly 0 and `responder_frac`
exactly 1.

`separation` is background iLISI *minus* salient iLISI, so a salient iLISI of 0
maximises it by construction. Ranking on separation there was really ranking
background mixing, and it rewarded precisely the two-blob degeneracy the penalty
had caused. The score guards against the salient space being too **mixed** and
has no guard against it being too **separated**.

Hence the two-stage rule in `rank_contrast_grid`:

1. **Collapsed points sort last**, whatever their separation. Not dropped — they
   are the evidence that the penalty range was too high, and a grid where
   everything collapsed should look obviously wrong rather than silently return
   its least-bad row.
2. Everything else sorts on `quality = separation - leakage_ratio`, both roughly
   in [0, 1] and both wanted at their extremes: keep the arms apart in the
   salient space, keep cell identity out of it.

`quality` is a convenience, not an authority. The component columns are kept
alongside it precisely so a human can disagree.

## Leakage is measured on background cells only

Background cells received no perturbation, so nothing legitimate can make their
salient coordinates informative about anything — any structure found there is
leakage by construction. Running the same test on target cells cannot
distinguish leakage from a genuinely cell-type-restricted response, which is the
confound this design removes.

`leakage_ratio` (salient vs background NMI against cell identity) is the
meaningful figure rather than raw NMI: it asks how much of the identity signal
that legitimately lives in the background space has *also* reached the salient
space.

**Honest limit, worth carrying into the Methods:** `contrast_separation_score`
rewards the model for finding the split it was told to find. It is a screen
against degenerate fits, not evidence that the salient axis is biological. The
two things that would make it biological are external — interpretable
top-loading genes on the salient axis, and the contrast reproducing on
independent chemistry.

## Replicates, and why the median

Each grid point is fitted once per seed. Without seeding, every point was a
**single unseeded fit** (`scvi.settings.seed` defaults to None), and the spread
that hides is not small: re-running `10x_BAP` at identical settings moved
`leakage_ratio` from **0.353 to 0.556** while `separation` held at 0.324 vs
0.343.

So `leakage_ratio` — which dominates `quality` — was being read off one draw from
a distribution nobody had measured. `best_grid_point` therefore aggregates
replicates by **median**, and treats a point as collapsed if the *majority* of
its replicates collapsed. Picking the single best row would be picking the
luckiest fit.

`select_contrast_grid` **appends** rather than overwrites for the same reason: a
repeated point is not waste but replication, and the expensive contrasts take
hours per grid. The `run` column keeps replicates distinguishable.

`summarise_contrast_grid`'s `quality_range` is the column to read before
believing any ranking: where it is wider than the gap between the top rows, the
grid has not distinguished them, and the honest answer is a plateau rather than
an optimum.

## ContrastiveVI ignores covariates, silently

As of **scvi-tools 1.5.0.post1** the module does not read covariate keys at all,
and the failure is silent:

- `ContrastiveVI.setup_anndata` registers both joint covariate fields, so they
  reach the AnnDataManager and are written into the saved model's `setup_args` —
  which is what makes it look like it worked;
- `ContrastiveVI.__init__` reads `n_cats_per_cov` off the registry into a local
  variable and never passes it to the module. Dead code;
- no `n_continuous_cov` is computed anywhere in the package;
- `ContrastiveVAE.__init__` accepts no covariate dimensions at all, and its
  `cat_list` is `[n_batch]` — batch is the **only** nuisance variable the
  encoders and decoder ever see.

Confirmed on a trained model (`MtSZ_cVI_nuc_BAP`, 2026-08-03): with three
continuous covariates registered, the first encoder layer is (128, 52689),
exactly `n_vars` rather than `n_vars + 3`.

`module_supports_covariates()` probes for this rather than hardcoding it, so a
future scvi-tools that implements covariates starts working with no edit. When
it returns False the covariates are **dropped rather than forwarded** — 
forwarding is strictly worse, because `setup_anndata` would register them and
the saved model's `setup_args` would record a correction that never happened.
That is exactly how this went unnoticed on the first five CH5 contrasts.

`batch_key` is the one nuisance slot that does work. For variation that cannot
go there: check it post hoc by correlating the salient dimensions against it,
and/or raise `wasserstein_penalty`.

## The penalty ladder was cut off below the turnover twice

`DEFAULT_PENALTIES` in `cvi_sweep.py`, retuned 2026-08-03 and again 2026-08-04.

The first pass used `(0, 1, 5, 10, 50)` — the parameter has no published scale,
so a log-ish ladder was the honest opening. At every penalty >= 1 it returned
salient iLISI of exactly 0 and `responder_frac` of exactly 1, read at the time
as the salient space collapsing into two blobs, and the ladder was cut to below
1 on that basis.

**That reading was wrong**, and `salient_eff_dim` — added afterwards precisely
to test it — is what showed so: the salient space uses **8.6–9.0 of its 10
dimensions at every penalty including 1.0**. Nothing collapsed. iLISI 0 with
high effective dimensionality means the arms separate cleanly while target cells
keep rich internal structure: the desired outcome, not a degenerate one. The
`collapsed` flag correctly never fires on these contrasts.

Extended again to 20 on 2026-08-04, because 2.0 was also a ceiling rather than
an optimum: with 3 seeds a point, **all five contrasts selected the top of the
0–2 ladder** and `leakage_median` was still falling there (10x_BAP 0.115 →
0.099 → 0.077 across 0.5/1/2).

The turnover is most likely near 10. The very first sweep had already bracketed
it — 10x_BAP `leakage_ratio` ran 0.068 @5, 0.031 @10, 0.038 @50, a minimum
around 10 — but those were unseeded single fits made before `salient_eff_dim`
existed, so they cannot be checked for collapse after the fact and are not
trusted as evidence. 5/10/20 re-measures that region properly.

The sub-1 rungs are kept: that is where `leakage_ratio` moves fastest (0.556 →
0.111 on 10x_BAP), so dropping them would lose the shape of the curve even
though its minimum lies higher.
