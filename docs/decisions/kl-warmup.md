# Scaling KL warmup to the training length

Rationale extracted from `integration/integrate.py::normalize_kl_warmup`,
2026-08-06.

## The defect

scvi ramps the KL weight 0→1 over `n_epochs_kl_warmup` epochs (default **400**)
while capping `max_epochs` at 400 via its own cell-count heuristic
(`get_max_epochs_heuristic`, `min(round(20000/n * 400), 400)`).

On any sample the default warmup therefore finishes only at the final epoch —
the latent trains under-regularized throughout. On a small per-sample SOLO model
that manifests as **~100% of cells called doublets**
([scvi-tools #507](https://github.com/scverse/scvi-tools/issues/507)).

## The rule

Warmup is tied to the *effective* `max_epochs` — the configured value when
present, else scvi's heuristic for `n_obs` — so the cap tracks what the model
will actually train for.

`_KL_WARMUP_MAX_FRACTION = 0.5` leaves the second half of training fully
regularized. It is a starting point, not tuned.

Three branches:

- **unset** → derived as `0.5 * effective_max_epochs`, replacing scvi's fixed 400
- **set longer than that cap** → clamped, with a warning, so a user cannot pin a
  warmup that never completes by choosing too small a `max_epochs`
- **set within the cap** → left untouched

Skipped entirely when the caller uses step-based warmup, since
`n_steps_kl_warmup` takes precedence in scvi regardless of the epoch count.

## `"auto"` is DRVI's, and is opt-in by name

`n_epochs_kl_warmup: "auto"` is passed through untouched. That is DRVI's own
default — `DRVITrainingPlan` derives the warmup from `trainer.max_epochs`, i.e.
ramps across the whole run — and the string would otherwise die on the `int()`
coercion.

This is deliberately **opt-in per target**: leaving the key unset still takes the
`requested is None` branch and writes a number, so DRVI's upstream behaviour is
reachable only by asking for it by name. Any other string raises rather than
being guessed at.

## Why it is module-level, and applied in `__init__`

`normalize_kl_warmup` lives at module scope rather than being only a
`SingleCellIntegrate` method because the scArches query fine-tune
(`refmapping._map_query_ref`) funnels through `model.train()` the same way and
needs the same guarantee. `_normalize_kl_warmup` is a thin instance wrapper that
supplies the logger.

It is applied once in `__init__`, to `self.train_args`, **before** the search
space is built from it — so the tuning trials and the final fit inherit one
warmup that actually completes, rather than the trials searching under a
different regime than the model that wins.

## Why `get_max_epochs_heuristic` is imported rather than copied

Importing it from `scvi.model._utils` means the cap tracks any change scvi makes
to the heuristic. It is a private module, so the import is guarded and the
formula replicated in the fallback rather than left to raise at import time.
