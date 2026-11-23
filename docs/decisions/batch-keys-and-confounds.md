# Batch keys, and what is safe to correct on

Rationale extracted from `integration/orchestrator.py` comments and docstrings,
2026-08-06. The columns themselves are documented in
`configs/sample_metadata.yaml`; this is the reasoning about which one to use.

## `batch` is not the batch key

On this cohort `batch`'s string is **one-to-one with condition**
(`10x_protoplast_treated_smeli96_sz`), so correcting on it removes the treatment
effect — the thing the experiment is about. It is a run label, not a nuisance
variable.

`tech` (10xv3 / bd_rhapsody) is **crossed** with condition wherever both
platforms ran the same timepoints, so it is the batch variable that can safely
be corrected. `prep` is the same idea one level down, and is correctable only on
the five BD susceptible-zone NF/mock samples where the handling variant is
crossed with NF-vs-mock.

`treatment` and `time_h` are the experimental design itself. They are **never**
batch keys; they are what a result is grouped and correlated against. Checking
whether a DRVI latent factor tracks the rhizobia timecourse needs `time_h` as a
number, not as a substring of `batch`.

Until 2026-08 the design existed only inside the `batch` string and in the
derived columns of `data/FQfiles/sample_summary.tsv` — nowhere a model or a plot
could read it. They are stamped onto `.obs` for every target now, not just the
ones that use them: a confound belongs in the data as a column rather than in
prose, and `batch_key: 'tech'` cannot be written at all until the column exists.

## The four `batch_key` forms

`shared/targets.py::parse_batch_key` normalizes a target's `batch_key:` into one
spec the run and the validator share. The first two are the historical forms:

| Written | Mode | Meaning |
|---|---|---|
| `batch_key: 'tech'` | `single` | one existing `.obs` column |
| `batch_key: null` | `none` | train with no correction |
| `batch_key: ['tech', 'prep']` | `combined` | derive one column joining the named columns' values, giving levels like `10xv3+2h_only` |
| `batch_key: {tip: 'sample', induced_all: ['tech','prep']}` | `mapped` | different keys for different parts of the cohort, resolved per sample, then combined |

**`combined` is for when two nuisance variables are each crossed with condition
but neither alone identifies the grouping to correct.**

**`mapped` exists for joint reference+query fits.** Per-sample correction is
right on an untreated reference — no treatment effect exists there to delete —
and catastrophic on a treated query, where `sample` is 1:1 with condition, so
correcting on it removes exactly the effect being measured. One column cannot
express that with a single key; a mapping can.

Keys of the mapping are group names or bare sample names, resolved through
`groups` exactly as `Samples`/`Reference`/`Query` are. **A sample matched by more
than one entry is an error**, not a precedence rule — silent precedence is how a
cohort ends up corrected on something nobody chose. `resolve_batch_assignments`
likewise raises on a sample no entry covers rather than falling back to a
default key.

A one-element list collapses to `single`, so `['tech']` and `'tech'` are the
same run and a target can gain and lose a second key without its `.obs` column
name changing under it. A repeated key is an error: repeating one would
duplicate its values inside every derived level without changing the grouping.

`batch_key` is in `NULLABLE_KEYS` because an explicit `batch_key: null` on a
target is a real instruction, not an omission. `resolve_target` otherwise skips
null values so a bare `key:` cannot accidentally clobber a default — right for
every key whose meaning is "a value", but wrong here, where reading it as
"unstated" would hand batch correction to a target that explicitly refused it.

## Materialising the column

`_apply_batch_key` materialises the `.obs` column named by the target's
`batch_key:`. `none`/`single` need nothing done — the column already exists, or
there is none. The `combined` and `mapped` forms name a column that does not
exist yet, and it comes into being there: **after** concatenation, so every
sample's values are present, and **before** the integrator sees the data.

Missing values are not filled. A component key absent from `.obs` raises: the
point of deriving the column is that the grouping is exact, and a level reading
`10xv3+nan` is a silent mis-correction of precisely the kind
`validate.py`'s `PARTIAL_STAMPED_OBS` check exists to stop. That check runs
first and should make the raise unreachable; it is there because the run must
not proceed on a half-built batch column if it ever is reached.

In `mapped` mode the **key names go into the value** (`sample=p1`, not `p1`),
because two samples resolved through different keys can otherwise collide on an
identical string and be silently merged into one batch level.

## Diagnostics are keyed on what the model actually corrected on

Fixed 2026-08-03. `run_diagnostics` was called with a hardcoded `batch_key='batch'`
regardless of the target's own key, which made the number actively misleading on
every target correcting on anything else.

Worked example: `MtSZ_DRVI_bd` trained on `prep` (2 levels) and its table
reported iLISI 0.073 / kBET 0.0004 over `batch` (10 levels) — the mixing of a
variable it deliberately did **not** correct on. Near-zero is the *correct*
outcome for that variable, but read as an integration score it says the fit
failed when it did not. This affected every non-`batch` target: `tech` on
`MtSZ_DRVI`, `prep` on `MtSZ_DRVI_bd`, `sample` on `MtTip_DRVI`, `batch_mixed`
on `MtTipInduced_DRVI`.

The `or 'batch'` fallback is deliberate and not just a None-guard: on a target
that asked for **no** correction (`batch_key: null` — `MtSZ_DRVI_10x`,
`MtSZ_DRVI_nobatch`) there is no corrected variable to score, and residual
`batch` structure is exactly the diagnostic those targets exist to expose.
scArches lands there too, having never had a batch_key.

Only the batch-mixing half of the diagnostics runs at this stage:
bio-conservation metrics need cell-type labels that do not exist until
annotation.

## contrastiveVI scores the background space

contrastiveVI writes two latents and only one is a meaningful batch-mixing
target. The **salient** space is *designed* to separate target from background,
and on these contrasts batch is one-to-one with that separation — so a "bad"
iLISI there is the model working correctly, and reporting it as an
integration-quality score would be actively misleading. The **background** space
is the one that genuinely should mix, so that is what is scored.
