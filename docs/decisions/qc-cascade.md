# The QC filter cascade

Rationale extracted from `mapping/qc_filter.py` docstrings, 2026-08-06.
See also `qc-miqc-transform.md` for the miQC transform decision.

## Three modes, and why `advisory` exists

`off` / `advisory` / `filter`, set per filter block in `filter_config.yaml`.

`advisory` computes and records everything a filtering run would — scores,
thresholds, flag column, `.uns` entry, figures — but its flags do not join the
removal mask. That is what gives the interactive report something to select on
for methods nobody wants making the decision automatically (iforest's
contamination quota, doublet calls).

`_filter_mode` **raises on an unrecognised value** rather than falling back to
`off`: a typo'd mode silently disabling a filter would let the run report success
on unfiltered data. This key replaced an older boolean `enabled:` — `filter` is
the equivalent of `enabled: true`, `off` of `enabled: false`.

## The sample_quality downgrade

`_assess_sample_quality` decides once, right after barcode rank, whether the
sample has enough surviving cells and depth to trust any filter that fits its
threshold from the sample's *own* distribution.

If it does not, `_effective_mode` downgrades every **distribution-dependent**
filter from `filter` to `advisory`. They still run and still record — the point
of flagging a sample is that a human looks at it, and they cannot judge a
threshold whose evidence was never computed.

`off` and `advisory` pass through untouched: a sample being marginal is not a
reason to start running a filter its config turned off.

**`mad` splits down the middle.** Its adaptive entries are distribution-dependent
and its `max_value`/`min_value` ones are not, so the two halves of one config
block resolve to different modes on a flagged sample. That is why
`_effective_mode` takes an explicit `distribution_dependent` override rather than
relying only on `DISTRIBUTION_DEPENDENT_FILTERS`.

Barcode rank runs *before* the verdict exists, which is why `_effective_mode`
reads `low_quality_sample` with a `getattr` default.

## Why MAD declines instead of guessing

MAD is the median absolute deviation from the median, so once more than half the
cells share a value it collapses to zero and `median ± k*0` degenerates into "is
this cell exactly the median" — flagging everything else.

Switching estimator does not help: Qn shares MAD's 50% breakdown, and IQR only
moves the cliff to 75%. The distribution has violated the unimodal assumption the
outlier model rests on, so `_degenerate_scale` declines rather than inventing a
threshold.

**`k` is not in sigma units.** `_mad` returns the raw median absolute deviation
with no 1.4826 consistency constant, so `k=3` is roughly 2σ for normal data, not
3σ — it flags ~4% of a clean sample where a reader would predict ~0.3%. This is a
choice, not a defect, but it is not what `k` looks like. See
`docs/resolved-defects.md`.

`_check_flag_fraction` is the backstop for the whole class: it warns when one
feature flags most of its input, since that is almost always a degenerate
threshold rather than real signal. It warns rather than acts, because a genuinely
poor sample may legitimately lose most of its cells.

## Ordering: adaptive first, absolute as cleanup

`_run_absolute_filter` applies the fixed `max_value`/`min_value` caps *after* the
adaptive MAD rule and miQC, as a cleanup on whatever they left behind. The
adaptive rules are the ones that adapt; a fixed cap is a floor under them, not a
competitor.

`_resolve_sample_type_value` lets a cap vary by `sample_type`, because a
nuclei prep and a whole-cell prep do not share an absolute gene-count ceiling.

## Distance-of-distance

`compute_dod_distances` implements Eq. 1 of Zhang et al., *Improved visualization
of high-dimensional data using the distance-of-distance transformation*
(doi:10.1371/journal.pcbi.1010764).

It is **quadratic in cell count**, hence the `max_cells` guard: roughly 5.6 GB at
50,000 cells. `_dod_filter` flags cells whose neighbourhood is rearranged by the
transform, on the reasoning that a cell whose neighbours change identity under a
distance re-ranking was not sitting in a stable neighbourhood to begin with.

## Recording

`_record_qc_metadata` stores everything under `.uns['qc'][method]`, and
`_is_per_feature` distinguishes entries that nest one payload per feature from
flat ones — the report's plot dispatch needs to know which shape it has.

`collect_removal_flags` is the single place that decides which `.obs` flag
columns actually remove cells: the union of flags from every method whose
*effective* mode is `filter`. A skip record carries no flag column, so it passes
over harmlessly.

`_record_post_cascade_viability` counts what is left once everything has run —
the cascade can be individually reasonable at every step and still leave nothing.
