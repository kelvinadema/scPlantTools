"""
The QC filter contract: shapes, flag columns, and the mode cascade.

No statistics here. These assert the things a filter must do regardless of what
it computes -- and specifically the things its *consumers* depend on. The report
builder and the interactive selector both read flag columns out of `.obs` by
name; a filter and its reader disagreeing about that name is a whole class of
bug that produces no error at all, just an empty panel in a report.

`_effective_mode` is the other target. Its rule is subtle enough to be worth
pinning: a low-quality sample downgrades *distribution-dependent* filters from
`filter` to `advisory` -- they still run and still record, they just stop
removing cells -- while `off` and `advisory` pass through untouched. And `mad`
splits down the middle, because its adaptive entries are distribution-dependent
and its `max_value`/`min_value` ones are not.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from fixtures import synthetic          # noqa: E402


# ------------------------------------------------------------------- shapes

@pytest.mark.parametrize('method,kwargs', [
    ('_mad_filter', dict(features=['n_genes'], k=3)),
    ('_iforest_filter', dict(features=['n_genes'])),
    ('_gmm_filter', dict(features=['n_genes'])),
])
def test_filter_returns_flag_columns_that_exist_in_obs(make_filter, clean, method, kwargs):
    """
    The returned names are the contract. A filter that returns a column it did
    not write leaves every consumer reading a KeyError.
    """
    cols = getattr(make_filter(clean), method)(**kwargs)
    if not cols:
        pytest.skip('{} declined on this input'.format(method))
    for c in cols:
        assert c in clean.obs.columns, '{} returned {!r} but did not write it'.format(method, c)


@pytest.mark.parametrize('method,kwargs', [
    ('_mad_filter', dict(features=['n_genes'], k=3)),
    ('_iforest_filter', dict(features=['n_genes'])),
])
def test_flags_are_boolean_and_full_length(make_filter, clean, method, kwargs):
    cols = getattr(make_filter(clean), method)(**kwargs)
    if not cols:
        pytest.skip('declined')
    for c in cols:
        v = clean.obs[c].to_numpy()
        assert len(v) == clean.n_obs, '{} is not one value per cell'.format(c)
        assert v.dtype == bool or set(np.unique(v)) <= {0, 1, True, False}, \
            '{} is not boolean-valued'.format(c)


def test_filters_do_not_drop_or_reorder_cells(make_filter, clean):
    """
    Filters FLAG; they never subset. Removal happens once, downstream, from the
    merged flag list -- a filter that also dropped cells would make the counts
    in the report disagree with the object.
    """
    before = list(clean.obs_names)
    make_filter(clean)._mad_filter(['n_genes'], k=3)
    assert list(clean.obs_names) == before


def test_flag_column_names_are_prefixed(make_filter, clean):
    """
    `flag_` is what the report and interactive selector glob for. A filter that
    named its column something else would be invisible to both.
    """
    cols = make_filter(clean)._mad_filter(['n_genes'], k=3)
    for c in cols or []:
        assert c.startswith('flag_'), '{!r} will not be picked up as a QC flag'.format(c)


# -------------------------------------------------------------- _effective_mode

def _f(make_filter, clean, cfg, low_quality=False):
    f = make_filter(clean, cfg=cfg)
    f.low_quality_sample = low_quality
    return f


def test_mode_passes_through_when_sample_is_healthy(make_filter, clean):
    cfg = {'filters': {'mad': {'mode': 'filter'}}}
    assert _f(make_filter, clean, cfg)._effective_mode('mad') == 'filter'


def test_filter_downgrades_to_advisory_on_a_flagged_sample(make_filter, clean):
    """
    The point of flagging a sample is that a human looks at it -- so the filters
    must still RUN and record their evidence, just not act on it. Skipping them
    would leave the human judging a threshold whose evidence was never computed.
    """
    cfg = {'filters': {'mad': {'mode': 'filter'}}}
    got = _f(make_filter, clean, cfg, low_quality=True)._effective_mode(
        'mad', distribution_dependent=True)
    assert got == 'advisory'


def test_off_is_never_upgraded_on_a_flagged_sample(make_filter, clean):
    """A sample being marginal is not a reason to start running a disabled filter."""
    cfg = {'filters': {'mad': {'mode': 'off'}}}
    got = _f(make_filter, clean, cfg, low_quality=True)._effective_mode(
        'mad', distribution_dependent=True)
    assert got == 'off'


def test_advisory_stays_advisory_on_a_flagged_sample(make_filter, clean):
    cfg = {'filters': {'mad': {'mode': 'advisory'}}}
    got = _f(make_filter, clean, cfg, low_quality=True)._effective_mode(
        'mad', distribution_dependent=True)
    assert got == 'advisory'


def test_non_distribution_dependent_entries_are_not_downgraded(make_filter, clean):
    """
    The mad split: `max_value`/`min_value` entries are absolute thresholds, not
    inferred from this sample's distribution, so a questionable distribution is
    no reason to stop applying them. Two halves of one config block, two modes.
    """
    cfg = {'filters': {'mad': {'mode': 'filter'}}}
    got = _f(make_filter, clean, cfg, low_quality=True)._effective_mode(
        'mad', distribution_dependent=False)
    assert got == 'filter'


def test_effective_mode_works_before_quality_is_assessed(make_filter, clean):
    """
    Barcode rank runs before `_assess_sample_quality` sets `low_quality_sample`,
    so `_effective_mode` reads it with a getattr default. Removing that default
    would make the first filter of every run raise AttributeError.
    """
    f = make_filter(clean, cfg={'filters': {'barcode_rank': {'mode': 'filter'}}})
    assert not hasattr(f, 'low_quality_sample')
    assert f._effective_mode('barcode_rank') == 'filter'


def test_unrecognised_mode_raises_rather_than_disabling(make_filter, clean):
    """
    A typo'd mode must not fall back to 'off'. Silently disabling a filter would
    let a run report success on unfiltered data -- the same failure the
    all-NaN guard in `_iterate_features` raises to prevent.
    """
    f = make_filter(clean, cfg={'filters': {'mad': {'mode': 'fitler'}}})
    with pytest.raises(ValueError, match='expected one of'):
        f._filter_mode('mad')


def test_an_unconfigured_filter_defaults_to_off(make_filter, clean):
    """Absent from the config means not requested, which is 'off', not a crash."""
    assert make_filter(clean, cfg={'filters': {}})._filter_mode('mad') == 'off'


def test_mad_is_declared_distribution_dependent(make_filter, clean):
    """
    The downgrade rule keys off DISTRIBUTION_DEPENDENT_FILTERS. If `mad` fell out
    of that set, a flagged sample would keep removing cells on an adaptive
    threshold -- the exact case the downgrade exists to prevent.
    """
    from mapping.qc_filter import DISTRIBUTION_DEPENDENT_FILTERS
    for key in ('mad', 'miqc', 'regression', 'gmm', 'iforest'):
        assert key in DISTRIBUTION_DEPENDENT_FILTERS, \
            '{} is adaptive but is not marked distribution-dependent'.format(key)
