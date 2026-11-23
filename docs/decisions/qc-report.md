# The QC report is a pure consumer

Rationale extracted from `mapping/report.py`, 2026-08-06. See also
`qc-cascade.md`, `qc-plot-axes.md` and `qc-terminal-umap.md`.

## Nothing is recomputed

Everything rendered is read back from artifacts the earlier stages already
wrote — no re-plotting, no re-clustering, no re-thresholding. A report can
therefore be regenerated at any time without touching the pipeline, and it
cannot disagree with the artifact it describes.

Templating is **stdlib only**. jinja2 happens to be importable in the
environment as a CellBender transitive, but depending on that accidentally would
make the environment unreproducible, and the layout is fixed enough not to need
it.

The page skeleton, stylesheet and value formatters live in `shared/html.py`
because the annotation stage's review report renders the same way — an unknown
value that formats differently between two reports is two different claims about
the data.

## Counts are view-relative and must not be summed

This is the single easiest thing to get wrong here.

Filters run as a cascade, each seeded with the previous stages' flags, so a
method's `n_flagged` counts only the cells that *reached* it. Per-method counts
therefore **cannot be summed or stacked** into a total.

`cascade_stages` groups by method and reports each stage's removal against the
cells that reached it, which is the only honest reading. Within a stage — the
six MAD features — the flags overlap freely, so their counts are reported but
never totalled.

Cascade position is not stored explicitly: each stage is seeded with every
upstream stage's flags, so **the count of those flags orders the stages**. That
is what `sort_key` and `depth` use.

`describe_view` states the same fact on the figures. Without that line a reader
looking at, say, the GMM scatter reasonably assumes every cell in the sample is
on it — which matters a great deal once those figures become the surface for
selecting cells by hand. It renders nothing when the method saw everything, or
when the input count is unknown, rather than stating a reassuring "all cells"
the data does not support.

It names the **methods** that gated the view, not the flag columns: `mad, miqc`
reads far better than six mangled column names, and the columns are in the
detail table anyway.

## Advisory methods are excluded from the waterfall

They removed nothing, so including them would put rows in a removal waterfall
that did not remove anything and corrupt the staged arithmetic the table is
built on. They still appear in the per-method detail table, labelled as
advisory.

A missing `exec_mode` means `filter`, so artifacts written before execution
modes existed read exactly as they did before. The same
backwards-compatibility rule applies to `gate_applied` on the sample-quality
verdict (absent means the verdict always gated) and to `umap_qc_status` (absent
means "no reason on file", *not* "the embedding was skipped" — an older artifact
whose embedding did compute must not be reported as skipped).

## Banners, not table rows

`NON_FILTER_METHODS` are not filters — they have no `flag_col` and no
`n_flagged` in the same sense. They are sample-level verdicts *about* the
population every other filter acts on: `sample_quality` gating them up front,
`post_cascade` recording what survived.

Both are rendered as banners, separately and more prominently than a row in the
generic table, because they are meant to change how a human reads everything
below them.

## The interactive payload reads the pre-filter object

`{sample}_filtered.h5ad` holds only survivors, so a report built from it could
offer removal but never **rescue** — and rescuing cells the automatic cascade
got wrong is half the point, since the MAD degeneracy trap removes good cells on
exactly the samples worth reviewing by hand.

The pre-filter object holds every barcode that entered QC, which is also the set
the decision file addresses, so the two agree by construction.

It is read `backed='r'`: the file is multi-GB with all its layers, and the
payload needs `.obs` and `.obsm` only.

The **QC UMAP is the exception** — it lives on the filtered object, since it is
computed on survivors and cells the cascade removed have no coordinates by
construction. It is merged in by barcode, leaving non-survivors as NaN, which
the page draws as "not assessed" rather than at the origin.

## Two rendering traps worth keeping

- **`.uns` lists round-trip as numpy arrays.** `x or []` on one raises *"The
  truth value of an empty array is ambiguous"*. Convert with `as_list` first,
  test after. An empty `filtered_on_flags` is the *normal* case for the first
  cascade stage, so this fired on essentially every sample.
- **`bcrank`'s payload carries full-length per-cell arrays** (`ranks`,
  `smoothed_vals`) alongside its scalars. `parse_diagnostics` reads only
  scalars; rendering those arrays would embed one row per cell in the HTML.

## Figure captions are keyed, not derived

`_FIGURE_CAPTIONS` maps a figure stem to what it shows and what its axes are.
Without it the caption is the filename with underscores swapped for spaces,
which is how *"DoD umap dod"* and *"DoD umap filt"* ended up in reports meaning
nothing to anyone who had not read `_plot_dod`.

Each filter also stores its threshold differently — `thresholds` as a dict (mad,
regression), as a bare float (gmm), `threshold` singular (bcrank, iforest), or
nothing at all (dod) — so `describe_threshold` has no uniform accessor to lean
on and dispatches per method.
