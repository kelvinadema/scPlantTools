# What preflight checks, and why each check exists

Rationale extracted from `shared/validate.py`, 2026-08-06.

## Two passes

The **structural** pass walks `shared/config_spec.py` and checks types, enums and
required keys within a single file — the same spec the editor sees as JSON
Schema.

The **cross-file** pass checks what a per-file schema cannot express, which is
where this pipeline's expensive mistakes actually live: an `index_ref` that names
nothing, an index built for a different mapper, a covariate whose name has
drifted out of the `.obs` contract and is therefore silently dropped so the model
trains without it.

The point is to fail in the first second rather than after CellBender has spent
hours on a GPU node. Everything is read-only and cheap: no h5ad matrices are
loaded (obs column names come off the HDF5 header via h5py) and no subprocess
runs except an optional `piscem --version`.

**Errors block; warnings do not.** Warnings cover states that are legitimate
mid-workflow — most importantly a missing index, which `--build-indexes` is about
to create and which cached samples do not need at all.

## sample_type-keyed caps

A MAD feature's `max_value`/`min_value` may be a mapping keyed by `sample_type`
(`{nuclei: .., protoplast: ..}`) rather than a scalar, resolved per sample by
`FilterQCMetrics._resolve_sample_type_value`.

A sample whose `sample_type` is not a key of that mapping used to have the cap
**silently unset** — a warning and `None`. Those absolute caps are the only
filtering a `sample_quality`-flagged sample keeps, since its distribution-fitting
rules are downgraded to advisory, so losing one to a typo'd or unlisted
`sample_type` is the dangerous case rather than a cosmetic one.

Both ends hard-error now: the resolver raises at runtime, and preflight catches
it before any GPU time, naming every offending (sample, feature) pair at once.

Scoped to the `mad` block because that is the only place the mapping is read; no
other filter resolves it. The `sample_type` default (`protoplast`) matches
`FilterQCMetrics` and `preprocess.py`.

## Doublet detection is opt-in, so its checks are conditional

An absent `doublets:` block is perfectly normal and must not warn. The checks
only fire once some filter profile sets `solo.mode` to something other than
`off`.

What is then checked, in order, and why each is worth a preflight second:

- **`doublets.env` is set.** scvi-tools cannot be installed alongside the mapping
  stage (pandas 3.x), so the worker needs its own conda prefix.
- **It is a directory containing `bin/python`** — the conda *prefix*, not the
  binary.
- **scvi-tools is importable there**, rather than discovering it after mapping
  and CellBender have run.
- **A GPU is visible to it**, when `use_cuda` is on. The worker asks Lightning
  for a GPU accelerator and dies immediately when there is none — per sample,
  after mapping and CellBender have already spent the time, with the real reason
  buried in the worker's stderr.

`doublets.tune` warns rather than errors: autotuning is **per sample**, so its
cost scales with the cohort rather than being a fixed overhead, and the models
are discarded afterwards since they exist only to produce doublet calls. Worth
saying out loud before a 40-sample run rather than discovering it from the wall
clock.

## Every mapped sample's `.obs` is read, not just the first

The column set genuinely differs between samples: a filter that skipped — a
degenerate MAD feature, a miqc component separation below threshold — writes no
flag column. Checking one sample and generalising would both miss real drift and
invent it.

This is what makes the covariate check meaningful: a name that has drifted out
of the `.obs` contract is dropped silently at integration time, so the model
trains without the correction the config asked for. See
`integration-required-keys.md` for the runtime half of that guard.
