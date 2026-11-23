# When autotune selects no model

Rationale extracted from `integration/integrate.py` docstrings and comments,
2026-08-06. See also `autotune-ray-runtime.md`.

## The failure is never downgraded to an untuned fit

`_integrate_scvi` calls `_tune_scvi` and uses whatever it returns; a tuning run
that returns no model **raises**. It is deliberately not caught and downgraded to
the fixed hyperparameters in the `else` branch.

Silently training untuned would turn a real, reproducible fault into a slightly
worse model and lose the only evidence of it. This fault is also not understood
yet: it took down SMu005 at 10,980 cells, where the "too few cells for a
validation split" explanation does not apply.

## Why `_log_tuning_failure` exists

`results.get_best_result` raises with *"no trial has reported this metric, or all
values reported for this metric are NaN"* — one message covering three very
different faults, naming neither which one occurred nor any trial.

That ambiguity is what made SMu005 undiagnosable: nothing in the log
distinguished "every trial crashed" from "every trial ran and returned NaN".
Ray's message is the symptom; the cause is in the individual trials.

So the three classes are separated and counted, one representative of each
logged with its own error (not all of them — 20 tracebacks would bury the
summary line):

| Class | How it is detected | What it points at |
|---|---|---|
| errored | the result carries `.error` | the environment or the data, not the search space: check GPU memory against the per-trial resources, and that the input matrix is counts |
| ran, never reported the metric | `metric not in .metrics` | the metric is a *validation* metric, only logged when a validation split exists. scvi-tools splits at `train_size=0.9` (`n_val = n - ceil(0.9n)`), so `n <= 9` gives an empty one. Above that, suspect the metric name or the training config, not the cell count |
| reported NaN/inf | `not np.isfinite(...)` | training diverged rather than failed. Suspect the search space — an over-large learning rate is the usual cause — or degenerate input |

The whole routine is best-effort and wrapped in a bare `except`: it runs on the
failure path and must not raise over ray's result API shifting, which would
replace the diagnosis with a second, less useful traceback.

The small-sample cause referenced above is the one `mapping/doublets.py` guards
with `min_cells`.

## Registry writes on the tuning path

`_save_opt_config` upserts rather than appends. A blind `add_entry` here was the
root cause of duplicate registry entries: it always created a second entry
alongside the one `_integrate_scvi`'s own save step updates via the same
`dataset_id`.

It is skipped entirely for a disposable (`persist=False`) model — this is the
registry write that would otherwise fire even with `save=False`, since tuning
records its optimum independently of the model save.
