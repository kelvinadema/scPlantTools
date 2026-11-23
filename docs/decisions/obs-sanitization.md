# Aligning `.obs` dtypes before `ad.concat`

Rationale extracted from `integration/orchestrator.py::_sanitize_obs`,
2026-08-06.

## The rule

A column keeps its real dtype only if it is present on **every** sample **and**
every one of those samples already agrees on its (non-categorical, non-object)
dtype. Everything else is stringified.

## Why presence alone is not enough

This is a real trap, not a hypothetical one. `_map_query_ref` loads a scArches
run's *reference* side as a single block straight from a previously trained
target's own combined `.h5ad` — itself already the output of an earlier
`_sanitize_obs` run against a different sample set. So a column there can
already be the stringified object dtype this function produces, while the
freshly-read query samples still carry it as a real bool.

Both "present everywhere" checks pass, but concatenating real bools from one
side with strings from the other produces a genuinely mixed object column —
which `write_h5ad` cannot serialize at all (h5py refuses an object array mixing
numbers and strings: `Can't implicitly convert non-string objects to strings`).
That is strictly worse than a merely imprecise dtype. Hence comparing dtypes
across samples, not just column presence.

## Why numeric and bool columns must be left alone

Blanket-stringifying `solo_doublet` turned it into `"True"`/`"False"`, which
`write_h5ad`'s own `strings_to_categoricals()` (on by default) then silently
promoted back to a real pandas Categorical on the in-memory object.
`report_embedding.py`'s `.sum()` over that column then raised:

    TypeError: 'Categorical' with dtype category does not support reduction 'sum'

Any numeric or bool QC column has the same exposure.

## Why the rest is stringified, and via `object`

Two different reasons landing on the same fix:

- **Categorical columns**, because frames whose categorical columns carry
  different category sets raise on concat otherwise.
- **Any column not present on every sample with one consistent dtype**, even a
  numeric or bool one. Doublet columns are documented as present only when that
  method ran (schema 1.4), so a genuine missing-column split is a real case too.
  Stringifying uniformly keeps it writable either way.

Cast to plain `object` first, **not** straight to `str`:

- a Categorical's own `fillna()` requires the fill value to already be a listed
  category, and `'Unknown'` generally is not one — calling it while still
  categorical is exactly what raised *"Cannot setitem on a Categorical with a
  new category (Unknown)"*;
- `object` also keeps a real NaN as NaN, instead of stringifying it to the
  literal text `"nan"` before the `fillna` gets a chance to replace it with
  `fill_value`.
