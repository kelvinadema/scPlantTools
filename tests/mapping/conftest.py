"""
Fixtures for the mapping-stage tests. Requires `scMappingV3.12`.

`FilterQCMetrics.__init__` reads a sample sheet and filter config off disk,
which the filter methods do not need. `filter_instance` therefore builds the
object with `object.__new__` and sets only the attributes the filters read
(adata, adata_view, filter_cfg, filter_profile, logger, transformed_features),
keeping these tests about the filters rather than config plumbing (covered by
tests/portable/test_config_resolution.py).

`adata_view` is set to None explicitly: in production every `_run_*_filter`
assigns it before a filter runs, and an instance without it would make the
filters operate on nothing.
"""

from __future__ import annotations

import logging
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fixtures import synthetic          # noqa: E402


@pytest.fixture(scope='session')
def FilterClass():
    from mapping.qc_filter import FilterQCMetrics
    return FilterQCMetrics


@pytest.fixture
def make_filter(FilterClass):
    """
    Build a FilterQCMetrics over a given AnnData.

    `profile` selects the filter profile name the mode helpers read;
    `cfg` is the filter config dict those helpers consult for `mode:`.
    """
    def _make(adata, cfg=None, profile='default'):
        f = object.__new__(FilterClass)
        f.adata = adata
        f.adata_view = None
        f.filter_cfg = cfg or {}
        f.filter_profile = profile
        f.logger = logging.getLogger('test.qc_filter')
        f.transformed_features = {}
        return f
    return _make


# --- data fixtures ---------------------------------------------------------

@pytest.fixture
def clean(): return synthetic.clean_normal()


@pytest.fixture
def planted():
    """(adata, planted_index) -- upper-tail outliers in n_genes."""
    return synthetic.planted_outliers()


@pytest.fixture
def compromised():
    """(adata, planted_index) -- the miQC case."""
    return synthetic.planted_compromised()


@pytest.fixture
def noise():
    """Low-incidence fraction with no real structure."""
    return synthetic.low_incidence_noise()
