"""
Statistical properties of the QC filters.

WHAT THESE REPLACE. The docstrings in `qc_filter.py` carried ad-hoc claims from
implementation time -- "a logit fit flagged 69% of cells", "recovered 300/300".
Those pin one number, from one run, against one implementation: retune a
threshold and they fail while telling you nothing about correctness.

What is asserted here instead are properties that hold for **any** correct
implementation, at any tuning:

    monotonicity   a stricter parameter flags a SUBSET of what a looser one flags
    idempotency    filtering twice equals filtering once
    null control   clean data loses ~nothing
    recovery       planted signal is found -- asserted as RECALL, not a count
    invariance     permuting cells, or rescaling a feature, does not change
                   which cells are flagged
    direction      'upper' never flags below the median, and vice versa

Null and recovery are always **paired**. A filter that never flags passes the
null tests and fails recovery; one that always flags does the reverse. Only a
correct filter passes both, which is why neither is meaningful alone.
"""

from __future__ import annotations

import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from fixtures import synthetic          # noqa: E402


def _flags(adata, cols):
    """Boolean array for the single flag column a filter returned."""
    assert cols, 'filter returned no flag column'
    return adata.obs[cols[0]].to_numpy().astype(bool)


# ============================================================== monotonicity

def test_mad_higher_k_flags_a_subset(make_filter):
    """
    Raising `k` widens the acceptance band, so it can only flag fewer cells --
    and specifically a SUBSET, not merely a smaller count. The subset assertion
    also catches a filter that flags a different set of the same size.

    Two independent AnnData objects with the same seed: the filter writes its
    flags into `.obs`, so reusing one object would let the first call's columns
    leak into the second.
    """
    a_loose, _ = synthetic.planted_outliers(seed=2)
    a_strict, _ = synthetic.planted_outliers(seed=2)
    loose = _flags(a_loose, make_filter(a_loose)._mad_filter(['n_genes'], k=3))
    strict = _flags(a_strict, make_filter(a_strict)._mad_filter(['n_genes'], k=6))
    assert strict.sum() <= loose.sum()
    assert set(np.flatnonzero(strict)).issubset(set(np.flatnonzero(loose)))


def test_iforest_lower_contamination_flags_fewer(make_filter):
    a1 = synthetic.planted_outliers()[0]
    a2 = synthetic.planted_outliers()[0]
    hi = _flags(a1, make_filter(a1)._iforest_filter(['n_genes'], contamination=0.10))
    lo = _flags(a2, make_filter(a2)._iforest_filter(['n_genes'], contamination=0.01))
    assert lo.sum() < hi.sum()


# ================================================================ idempotency

def test_mad_filter_is_idempotent(make_filter, planted):
    adata, _ = planted
    f = make_filter(adata)
    first = _flags(adata, f._mad_filter(['n_genes'], k=3)).copy()
    second = _flags(adata, f._mad_filter(['n_genes'], k=3))
    assert np.array_equal(first, second)


# =============================================================== null control

def test_mad_flags_few_on_clean_data(make_filter, clean):
    """
    Null control: a clean unimodal sample should lose little.

    THE CEILING IS 8%, NOT ~0.3%, AND THAT IS DELIBERATE. `_mad` returns the RAW
    median absolute deviation, with no 1.4826 consistency constant, so `k=3`
    corresponds to roughly 2 sigma rather than 3 -- about 4% of a clean normal
    sample rather than 0.27%. That is a choice, not a defect (the configured
    thresholds were presumably tuned against this behaviour), but it means `k`
    does NOT read in sigma units. Measured on this fixture: ~3.7-4.7% across
    transform settings.

    If someone adds the 1.4826 scaling, this test keeps passing and the
    monotonicity tests keep passing -- but every configured `k` becomes ~1.5x
    stricter in effect. That is a science decision; the ceiling here is only a
    guard against a filter that starts eating clean samples wholesale.
    """
    flags = _flags(clean, make_filter(clean)._mad_filter(['n_genes'], k=3))
    assert flags.mean() < 0.08, 'clean sample lost {:.1%}'.format(flags.mean())


def test_miqc_declines_on_structureless_noise(make_filter, noise):
    """
    A low-incidence fraction with no real structure -- most cells near zero with
    a thin count-noise tail -- must yield no separation.

    This is the property behind the old "a logit fit flagged 69% of cells" note.
    The number was an artefact of a transform that has since been removed; the
    durable claim is that the raw-scale fit finds nothing here.
    """
    out = make_filter(noise)._miqc_filter('ribosomal_rna_fraction', 'n_genes')
    if out is not None:
        flags = _flags(noise, out)
        assert flags.mean() < 0.10, \
            'invented structure in pure noise: flagged {:.1%}'.format(flags.mean())


# ==================================================================== recovery

def test_mad_recovers_planted_outliers(make_filter, planted):
    """Recall on the planted set, not an exact flag count -- recall survives retuning."""
    adata, idx = planted
    flags = _flags(adata, make_filter(adata)._mad_filter(['n_genes'], k=3, direction='upper'))
    recall = flags[idx].mean()
    assert recall >= 0.95, 'recall {:.2f} on planted outliers'.format(recall)


def test_miqc_recovers_planted_compromised(make_filter, compromised):
    """The old '300/300, 0 false negatives' claim, as a recall floor."""
    adata, idx = compromised
    out = make_filter(adata)._miqc_filter('mitochondrial_fraction', 'n_genes')
    assert out is not None, 'miQC declined on a sample with real planted structure'
    recall = _flags(adata, out)[idx].mean()
    assert recall >= 0.80, 'recall {:.2f} on planted compromised cells'.format(recall)


def test_iforest_recovers_planted_outliers(make_filter):
    adata, idx = synthetic.planted_outliers(n_planted=100)
    flags = _flags(adata, make_filter(adata)._iforest_filter(['n_genes'], contamination=0.06))
    assert flags[idx].mean() >= 0.80


# =================================================================== invariance

def test_flags_are_permutation_invariant(make_filter):
    """
    Shuffling cell order must change which ROWS are flagged, not which CELLS.
    A filter that depends on input order is reading an artefact.
    """
    a1, _ = synthetic.planted_outliers(seed=1)
    flags1 = _flags(a1, make_filter(a1)._mad_filter(['n_genes'], k=3))
    called1 = set(a1.obs_names[flags1])

    a2, _ = synthetic.planted_outliers(seed=1)
    perm = np.random.default_rng(7).permutation(a2.n_obs)
    a2 = a2[perm].copy()
    flags2 = _flags(a2, make_filter(a2)._mad_filter(['n_genes'], k=3))
    called2 = set(a2.obs_names[flags2])

    assert called1 == called2


def test_miqc_is_scale_invariant(make_filter):
    """
    Both axes are standardised inside the fit and converted back, so multiplying
    the fraction by a constant must not change which cells are flagged.

    This is the durable form of the old "unscaled fit measured a gap of 0.0002
    against a planted 0.03" note: that number described the bug, this describes
    the fix, and it cannot go stale.
    """
    a1, idx = synthetic.planted_compromised(seed=3, scale=1.0)
    a2, _ = synthetic.planted_compromised(seed=3, scale=1000.0)
    o1 = make_filter(a1)._miqc_filter('mitochondrial_fraction', 'n_genes')
    o2 = make_filter(a2)._miqc_filter('mitochondrial_fraction', 'n_genes',
                                      min_component_separation=0.02 * 1000)
    assert (o1 is None) == (o2 is None), 'scaling changed whether the filter ran at all'
    if o1 is not None:
        assert np.array_equal(_flags(a1, o1), _flags(a2, o2)), \
            'flags changed under a pure rescaling of the feature'


def test_iforest_is_deterministic(make_filter):
    a1 = synthetic.planted_outliers(seed=5)[0]
    a2 = synthetic.planted_outliers(seed=5)[0]
    f1 = _flags(a1, make_filter(a1)._iforest_filter(['n_genes'], random_state=42))
    f2 = _flags(a2, make_filter(a2)._iforest_filter(['n_genes'], random_state=42))
    assert np.array_equal(f1, f2)


# ==================================================================== direction

def test_mad_upper_never_flags_below_median(make_filter, planted):
    adata, _ = planted
    v = adata.obs['n_genes'].to_numpy()
    flags = _flags(adata, make_filter(adata)._mad_filter(['n_genes'], k=3, direction='upper'))
    assert not (v[flags] < np.median(v)).any()


def test_mad_lower_never_flags_above_median(make_filter, planted):
    adata, _ = planted
    v = adata.obs['n_genes'].to_numpy()
    flags = _flags(adata, make_filter(adata)._mad_filter(['n_genes'], k=3, direction='lower'))
    assert not (v[flags] > np.median(v)).any()


def test_mad_both_is_the_union_of_upper_and_lower(make_filter):
    a_b, _ = synthetic.planted_outliers(seed=11)
    a_u, _ = synthetic.planted_outliers(seed=11)
    a_l, _ = synthetic.planted_outliers(seed=11)
    both = _flags(a_b, make_filter(a_b)._mad_filter(['n_genes'], k=3, direction='both'))
    up = _flags(a_u, make_filter(a_u)._mad_filter(['n_genes'], k=3, direction='upper'))
    lo = _flags(a_l, make_filter(a_l)._mad_filter(['n_genes'], k=3, direction='lower'))
    assert np.array_equal(both, up | lo)
