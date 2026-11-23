# A config key that is not in `.obs` is an error

Rationale extracted from `integration/integrate.py::_prep_scvi`, 2026-08-06.

## What it used to do

`batch_key`, the categorical/continuous covariates and `labels_key` were all
resolved defensively. A key named in the config but absent from `.obs` was
logged at INFO and dropped, and `setup_anndata` filtered it out.

That meant a covariate whose name had drifted — a renamed `.obs` column, a typo,
a partial column lost during concatenation — produced a model that **trained
without it, with nothing in the output saying so**. The saved model's
`setup_args` recorded only the covariates that survived, so the loss was not
recoverable after the fact either.

`batch_key` was the worst case: `batch_key if batch_key in obs else None` at
`setup_anndata` meant the model trained with **no batch correction** while the
config said otherwise.

## What it does now

Every declared key is checked up front and a single `KeyError` names all the
missing ones at once, alongside the columns that are present. The
`if k in self.adata.obs` filtering at the `setup_anndata` call is gone — the
raise has already guaranteed presence, and the filters were what turned a
drifted name into a silently weaker model. What the config declares is what
`setup_anndata` registers, or the run stopped.

`labels_key` is checked separately but on the same reasoning, and is likewise
**not** downgraded to `None`. A caller passing one is either about to build a
scANVI model from this one — `SCANVI.from_scvi_model` inherits the source
model's setup args, and a label-free scVI forces the labels to be supplied
later — or about to tune on scib bio-conservation, where the benchmark reads its
`label_key` off the model's *registered* labels. With none, every cell carries
the same label and the whole bio-conservation half (NMI, ARI, silhouette,
isolated labels) is computed over one class. That does not fail loudly: it
reports NaN for every trial, and the run dies with the same "no trial reported
the metric" message that has no single cause (see
`autotune-failure-diagnosis.md`).

## Where it should have been caught first

`shared/validate.py` checks these at config time against the `.obs` union, so
reaching the raise in `_prep_scvi` means preflight was skipped
(`--skip-validation`) or the column went missing between concatenation and here.
Both are worth stopping for. The error message points at
`python -m mapping.validate --icfg ... --scfg ...` for exactly that reason —
catching it before a GPU job starts.
