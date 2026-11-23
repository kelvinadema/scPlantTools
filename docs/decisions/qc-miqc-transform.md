# miQC: why neither axis is transformed

**Decision:** `mapping/qc_filter.py::_miqc_filter` fits on raw
`(complexity, fraction)`. No logit on the fraction, no log on complexity.

**Status:** settled. Guarded by
`tests/test_qc_miqc.py::test_raw_scale_skips_on_pure_noise`.

## What was there before

An earlier version of the function logit-transformed the fraction and — based on
a misreading of the paper — log-transformed complexity as well. Neither is in
the original. `flexmix(subsets_mito_percent ~ detected, k=2)` in the package's
own `mixtureModel.R` uses both variables completely untransformed; this was
checked against the source rather than inferred from the text.

## Why the logit broke

A logit disproportionately stretches the region near zero. That is exactly where
most cells sit for a low-incidence feature such as `ribosomal_rna_fraction`: few
genes contribute, so the bulk of cells cluster near zero with a long thin tail
that is pure count noise.

In that stretched space a difference between 0.1% and 2% — biologically
negligible — becomes a large separation. The mixture then has something to
split, and splits it, fabricating a "compromised" component out of noise instead
of finding a real one.

This was not hypothetical. It fired on a real, otherwise healthy sample.

### The measurements

| Setup | Logit fit | Raw fit |
|---|---|---|
| Synthetic fraction, **zero** real signal | flagged **69%** of cells | found no separation, correctly skipped |
| Synthetic fraction, **planted** compromised population | — | recovered **300/300**, 0 false negatives |

Both rows matter. The first shows the logit inventing structure; the second
shows that removing it does not cost sensitivity where there is something real
to find.

## The related choice: standardization inside the fit

Separate from the transform question, both axes *are* standardized (zero mean,
unit variance) inside the fit, with every fitted quantity converted back to raw
units immediately afterward.

With `covariance_type='full'` and no rescaling, the fitted covariance structure
— and therefore which axis the fit splits on — is dominated by whichever axis
carries the larger absolute variance. Complexity spans hundreds to thousands of
genes; a low-incidence fraction spans a few thousandths. Standardizing
complexity alone is not enough, because unit variance still dwarfs the
fraction's own.

Measured on synthetic data isolating the effect (two populations differing only
in fraction, complexity carrying no signal at all):

| Setup | Measured component-mean-fraction gap | True planted gap |
|---|---|---|
| Unscaled fit | **0.0002** (diluted to noise) | 0.03 |
| Both axes standardized | recovered the separation | 0.03 |

This is why `min_component_separation`, `posterior_threshold` and the
intact-line comparison all keep their raw-proportion-unit meaning while the fit
itself works in standardized space. The two are not in conflict — the
standardization is internal and reversed before anything is reported.

## Why this is not in the docstring

The docstring keeps the instruction ("do not add a transform to either axis")
and the one-line reason, because that is what a reader needs at the moment they
are tempted to add one. The narrative and the numbers are here, and the numbers
are additionally encoded as tests, because a percentage in prose cannot fail
when the code regresses.

---

*Extracted from `_miqc_filter`'s docstring on 2026-08-06. No content was
discarded — the operative facts stayed in the docstring, the evidence became
`tests/test_qc_miqc.py`, and the history is this file.*
