"""
Degenerate inputs: does the QC cascade decline gracefully, and say why?

These are the cheapest tests in the suite and the ones most likely to catch a
crash. None of them asserts a number -- each asserts that a filter handed input
that violates its assumptions **declines rather than inventing a threshold**, and
that the decline is recorded somewhere a human can see it.

The failure mode being guarded against is not an exception. It is a filter
silently producing a threshold from data that cannot support one, flagging most
of the sample, and the run continuing.
"""

from __future__ import annotations

import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from fixtures import synthetic          # noqa: E402


# --------------------------------------------------------------- _degenerate_scale

def test_degenerate_scale_flags_majority_identical(make_filter):
    """
    MAD is the median absolute deviation from the median, so once more than half
    the cells share a value it collapses to zero and `median +/- k*0` degenerates
    into "is this cell exactly the median" -- which flags everything else.
    """
    a = synthetic.degenerate('majority_identical')
    result = make_filter(a)._degenerate_scale(a.obs['n_genes'].to_numpy())
    assert result is not None, 'a 60%-identical feature must be declined'
    reason, share = result
    assert 'zero' in reason.lower() or 'single value' in reason.lower()
    assert share >= 0.5


def test_degenerate_scale_flags_constant(make_filter):
    a = synthetic.degenerate('constant')
    assert make_filter(a)._degenerate_scale(a.obs['n_genes'].to_numpy()) is not None


def test_degenerate_scale_flags_all_nan(make_filter):
    a = synthetic.degenerate('all_nan')
    result = make_filter(a)._degenerate_scale(a.obs['n_genes'].to_numpy())
    assert result is not None
    assert 'finite' in result[0].lower()


def test_degenerate_scale_passes_healthy_spread(make_filter, clean):
    """
    The negative control. Without this, a `_degenerate_scale` that returned a
    reason unconditionally would pass every test above.
    """
    assert make_filter(clean)._degenerate_scale(clean.obs['n_genes'].to_numpy()) is None


# ------------------------------------------------------------------ _mad_filter

def test_mad_does_not_flag_everything_on_majority_identical(make_filter):
    """
    The consequence the guard exists to prevent: a collapsed scale estimate must
    not turn into "flag every cell that is not exactly the median".
    """
    a = synthetic.degenerate('majority_identical')
    out = make_filter(a)._mad_filter(['n_genes'], k=3)
    if out:
        flagged = a.obs[out[0]].to_numpy().astype(bool).mean()
        assert flagged < 0.5, 'declined to {:.1%} -- the degenerate-scale guard did not hold'.format(flagged)


@pytest.mark.parametrize('kind', ['constant', 'all_nan', 'single_cell', 'empty', 'zero_variance'])
def test_mad_survives_degenerate_input(make_filter, kind):
    """No raise. Returning None or an unused flag column are both acceptable."""
    a = synthetic.degenerate(kind)
    make_filter(a)._mad_filter(['n_genes'], k=3)


# ----------------------------------------------------------------- _miqc_filter

def test_miqc_skips_below_min_cells_per_component(make_filter):
    """
    `min_cells_per_component` (default 20) is a floor below which the two-component
    fit is unstable regardless of what it reports.
    """
    a, _ = synthetic.planted_compromised(n_cells=10, n_planted=3)
    out = make_filter(a)._miqc_filter('mitochondrial_fraction', 'n_genes',
                                      min_cells_per_component=20)
    assert out is None, 'miQC must skip when a component cannot be populated'


def test_miqc_skips_when_separation_below_threshold(make_filter, clean):
    """
    `min_component_separation` guards against splitting a single population.
    Set it implausibly high and even real structure must be declined.
    """
    out = make_filter(clean)._miqc_filter('mitochondrial_fraction', 'n_genes',
                                          min_component_separation=0.95)
    assert out is None


# ------------------------------------------------------------------ other filters

def test_gmm_declines_below_min_k_cells(make_filter):
    """`_select_gmm_by_bic`'s `min_k_cells` (15) bounds components by cell count."""
    a = synthetic.degenerate('zero_variance', n_cells=10)
    X = a.obs[['n_genes']].to_numpy()
    # Returns (best_k, cov_type, bics), not a bare int.
    best_k, _cov, _bics = make_filter(a)._select_gmm_by_bic(X, max_components=3, min_k_cells=15)
    assert best_k is None or best_k <= 1, \
        'fitted {} components to 10 cells with min_k_cells=15'.format(best_k)


def test_iforest_survives_zero_variance(make_filter):
    a = synthetic.degenerate('zero_variance')
    make_filter(a)._iforest_filter(['n_genes'])


def test_regression_survives_constant_predictor(make_filter):
    a = synthetic.degenerate('constant')
    make_filter(a)._regression_filter('n_genes', 'total_counts')


# --------------------------------------------------------------------- recording

def test_decline_is_visible_not_silent(make_filter, caplog):
    """
    A skip must leave a trace. A filter that declines silently is
    indistinguishable, downstream, from one that ran and found nothing -- and
    those mean opposite things about the sample.
    """
    import logging
    a = synthetic.degenerate('majority_identical')
    with caplog.at_level(logging.INFO):
        make_filter(a)._mad_filter(['n_genes'], k=3)
    assert caplog.records, 'declining produced no log record at all'
