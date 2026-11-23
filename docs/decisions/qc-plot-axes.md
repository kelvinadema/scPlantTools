# QC figures are drawn in raw units on a transformed axis

Rationale extracted from `mapping/qc_plot.py`, 2026-08-06. See also
`qc-cascade.md` and `qc-miqc-transform.md`.

## The problem

The adaptive MAD rule is fit in transformed space (`log_`/`logit_`), and the
figures used to be drawn there too — x axis labelled in logits, threshold at
`-3.098`.

Nothing else in the report speaks that language. The interactive panel offers
raw columns, the decision file records raw values, and `refilter` applies them
to the raw `.obs` column. So one cut-off appeared as two different numbers
depending on where you read it, and a hand-set threshold could not be compared
to the pipeline's by eye.

## The fix is a change of labels, not of picture

The values plotted are raw, but the axis carries the very transform MAD fit on
(`trans=`), so the distribution has exactly the shape it had before — a raw
linear axis would pile a count distribution into the leftmost bin and make the
figure useless. Ticks and threshold lines end up in units a person can type back
into the interactive panel.

`invert_transform` maps the recorded thresholds back to the raw scale for the
same reason.

**A transform is only claimed when a different raw column was actually
resolved.** `feature` is the column MAD worked on; `source_feature` is the raw
column behind it, equal to `feature` for the absolute-cap entries, which are
never transformed. Applying a log1p axis to already-log1p values would put every
cell in the wrong place.

## `logit_trans` is hand-rolled

mizani's own `logit_trans` is unusable here on two counts, verified against
**mizani 0.14.4**:

- it is `probability_trans('logistic')`, whose `transform` is the logistic
  *CDF* (i.e. expit — the inverse of what the name says);
- its `__init__` never sets `breaks_func`, so plotnine raises `AttributeError`
  the moment it asks for ticks.

`LOGIT_CLIP` is the same clip `_transform_features` applies before fitting, so
the axis places a cell exactly where MAD saw it. Without it the exact zeros
every fraction feature contains would map to `-inf` and plotnine would drop them
silently — and on a nuclei prep that is most of the sample.

## Break selection

Both break functions exist because the generic algorithms produce unreadable
labels on these axes.

**Fraction breaks are a fixed list**, not computed: the whole reason for a logit
axis is that the action is at 0.1% and 1%, and a generic break algorithm run
over logit space returns labels like `0.00734`.

**Count breaks are 1/2/5 × 10^k decades.** mizani's `log1p_trans` ships
`breaks_extended`, which works in data space and on a 200–2400 gene count
returns just 1000 and 2000 — two labels, both in the right-hand third, with
nothing to read the cut-off against on the left.

`_thin_breaks` is greedy with a minimum gap rather than "nearest candidate to
each of n evenly spaced targets". The latter happily picks two candidates that
are adjacent on the transformed axis, which on a logit axis renders as
`0.00050010` — two labels printed on top of each other.

## Figure naming

Adaptive figures are `MAD_adaptive_{raw}`; the absolute caps get their own
`absoluteCut_{raw}` prefix so `report.py` groups them under a separate heading.
The `mad:` config block produces both, and a fraction feature carries one entry
of each — they read identically without the distinction.

These are drawn wider than the default: the titles carry a
`{metric}_fraction_{layer}` feature name that the 6-inch default clips off the
left of the canvas.
