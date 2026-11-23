# What gets dropped, renamed and filtered before concatenation

Rationale extracted from `integration/orchestrator.py`, 2026-08-06. See also
`obs-sanitization.md` for the dtype half of the same step.

## Per-sample embeddings are dropped

`ad.concat(..., join="outer")` keeps any `.obsm` key present on every input, and
the mapping stage writes `X_umap_qc` (and `X_umap_scVI` when doublet detection
ran) for every sample — so they would all survive.

Each is its own arbitrary 2D coordinate space fit to **one sample**, so the
concatenated result would look like a valid embedding of the whole cohort and
mean nothing. That is the failure mode that produces a plausible figure rather
than an error, so they are dropped explicitly rather than trusted not to be
used.

The leiden labels alongside them (`leiden_qc`, `leiden_scVI`) are left in `.obs`
on purpose: they are per-sample cluster *labels*, which stay interpretable after
concatenation as long as nobody reads them as cohort-wide clusters, and they
carry the manual-review context from the QC report.

## Unused layers are dropped (`keep_layers`)

A mapping-stage artifact carries eight layers (`corrected_counts`, `counts`,
`raw_counts`, `ambiguous`, `spliced`, `unspliced`, `rspliced`, `runspliced`)
plus `X`, and `join="outer"` keeps every one. So a concatenated target holds
nine sparse matrices of the same shape in RAM at once and writes all nine to
disk — while the model reads exactly one (`corrected_counts`, which is also what
every downstream annotation/refine config names as its `counts_layer`).

On `MtInduced` that is 299,913 × 54,657 with ~8e8 stored values per layer: a
**69 GB artifact, ~90% of it never read**. That is the usual cause of an OOM
here, and it scales with cohort size, so it bites hardest on exactly the targets
that are already largest.

Opt-in via `keep_layers` on the target. `None` (the default) preserves the
keep-everything behaviour, so no existing target changes by being re-run.

This does change what the target's own output h5ad contains. That is the point —
but it means a stage wanting a dropped layer must re-derive it from the mapping
artifacts, which still carry all eight and are never written to by this stage.

## Barcodes are prefixed with the sample name

Unconditional, and separate from the `sample`-column backfill next to it.

The 10x barcode whitelist is shared and finite, so two independently run samples
can legitimately produce the same barcode string by coincidence. This used to be
nested inside `if "sample" not in adata.obs`, which is true only when a sample's
own artifact lacks a `sample` column — but the mapping stage stamps that column
on every terminal h5ad (`shared/io.py::write_h5ad`), so in practice the
condition was always False and **the line never ran**.

Found via MtSZ (pMock + nMOCK merged): 9 barcodes collided across the two
constituent samples; MtTip (77,883 cells): 76. The result was a non-unique
`adata.obs_names`, which pandas `.reindex()` (`annotation/review.py`'s seed
overlay) then refused to operate on: *"cannot reindex on an axis with duplicate
labels"*.

## The gene floor (`min_cells`)

Exists for the `do_hvg: False` case. The gene floor that normally runs with
integration is `hvg.py::initialize_views`' own `min_cells=10`, which lives
*inside* `HVGSelector` — and `integrate.py::_prep_scvi` only constructs it
`if hvg`. So turning HVG selection off removes the floor along with the
selection and hands the model every gene the mapping stage emitted, **~42% of
which are detected in under 10 cells** on the SZ cohort. Those carry no signal
for a split decoder to spend capacity on.

Applied once to the **concatenated** object rather than per sample: a gene
detected in 8 cells of each of 20 samples is detected in 160 cells of the
target, and filtering per sample would drop it. The cohort is the population the
model is fitted on, so it is the population the floor belongs to.

Two details that are easy to get wrong:

- **Count on the layer the model trains on, not on `.X`.** `_prep_scvi` is given
  `counts_layer` (normally `corrected_counts`), while `.X` after `ad.concat` is
  whatever the mapping artifacts happened to carry. Filtering on one matrix and
  training on another selects a gene set that does not describe the data the
  model sees.
- **`(X > 0)`, never `getnnz`.** `getnnz` counts *stored* entries, and
  `corrected_counts` is CellBender output that can carry explicitly stored
  zeros — `hvg.py::initialize_views` flags the same trap. A gene stored in
  10,000 cells but positive in 3 would otherwise pass.

## Artifact availability is pre-flighted

Every sample's artifact key is checked before anything is loaded, so a
40-sample run reports all the missing ones at once instead of failing on the
first. Reference samples loaded from an explicit `adata_path` are exempt: they
never go through the manifest.

Which artifact a model was trained on — the automatic `filtered_h5ad` or a
hand-reviewed `filtered_manual_h5ad` — has **no CLI override** on purpose. It is
part of what the model *is*, so it belongs in the config entry that documents
it rather than in an invocation nobody can reconstruct later. `--artifact-fallback`
is the exception, and mixing kinds across samples warns loudly.
