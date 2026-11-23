# The terminal UMAP is best-effort; the artifact is not

Rationale extracted from `mapping/qc_plot.py::QCPlotter.run`, 2026-08-06.

## What must survive, and what may not

`QCPlotter.run` writes the stage's actual contract — the filtered artifact, its
`metadata.tsv`, and the manifest. The final embedding (`X_umap_qc`,
`leiden_qc`) is additive under schema 1.2 and no consumer requires it.

So everything from the embedding to the write is wrapped: **a sample that cannot
be embedded must still produce all three.**

This block used to be the one unguarded path in the stage. On
`5_s_zone_exp`, where the cascade left **0 cells**, `DimReducer` →
`hvg.initialize_views` → `sc.pp.log1p` raised on the empty matrix, `main.py`'s
per-sample handler caught it, and the sample was lost entirely — no artifact, no
manifest, no report — over a figure.

## The cell floor is technical, not a quality judgement

`_MIN_CELLS_FOR_EMBEDDING = 20` is checked first so the common case gives a
clear reason rather than an opaque error from inside scanpy.

`hvg.py`'s own `initialize_views(min_cells=10)` drops every gene detected in
fewer than 10 cells, so below roughly this many rows the matrix has no columns
left and HVG/PCA fail with `Found array with 0 feature(s)` rather than anything
useful. PCA, neighbours and leiden each need more rows than that to mean
anything.

Samples under the floor still get their filtered artifact, metadata and HTML
report. Only the embedding is omitted, with the reason recorded in
`.uns['umap_qc_status']` so the report shows it in place of the figure. The
consequence worth knowing: without `X_umap_qc`/`leiden_qc` the interactive
report cannot lasso or cluster-select on that sample.

## Why the embedding is computed on a throwaway copy

`DimReducer` also leaves behind the neighbour graph (`.obsp`) and
`uns['neighbors']`, which stay out of the terminal artifact — at 15k cells the
graph is the only piece with real size cost, and nothing downstream needs it.

The 2D embedding and the leiden labels *are* copied back: two float columns and
one categorical, negligible against a multi-GB object, and they are what the
interactive QC report selects cells on. Lassoing a population, or picking a
cluster, is not possible from a PNG.

`umap_params` is **merged rather than overwritten** when recorded, because
`doublets.py` may already have written `X_umap_scVI` there and it is carried
over by the `.copy()` of `self.adata`.

## Stale figures are cleared before regenerating

`report.py`'s `_qc_figures` discovers figures by globbing the directory rather
than tracking what the run actually wrote. So a PNG left over from an earlier
run — a feature that no longer exists, a method that ran under a different
config, a threshold since changed — would keep appearing in every report
indefinitely, misrepresenting what the current run found.

`_FIGURE_PREFIXES` is therefore this stage's own figure prefixes only, kept in
sync with `report.py`'s `_METHOD_FIGURE_PREFIX` plus `filtered` for the final
UMAP. A mapping-stage figure like `{sample}_ReadFlow.png` deliberately does
**not** belong there: it is produced by an earlier stage with its own done flag
and must survive a QC-only rerun.

## Skipped filters

A filter that declined to threshold a feature has no flag column or cut-off to
draw, so `run()` guards those generically rather than having each plotter
hard-index the missing keys.

`_PLOTS_ON_SKIP` is the exception list. For **miqc** a skip is not an edge case
but the common case — its separation and degenerate-scale guards are
conservative by design — so blanket-suppressing the figure would mean the plot
never appears at all on most real samples.
