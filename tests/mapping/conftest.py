"""
Fixtures for the mapping-stage tests. Requires `scMappingV3.12`.

THE HARNESS. `FilterQCMetrics.__init__` takes (sample_name, sample_base_dir,
sample_config, filter_config_path) and reads a sample sheet and filter config off
disk. The filter methods themselves need almost none of that: across
`_mad_filter`, `_gmm_filter`, `_miqc_filter`, `_iforest_filter`,
`_regression_filter` and their helpers, the only attributes read are

    adata  adata_view  filter_cfg  filter_profile  logger  transformed_features
    MAX_PLAUSIBLE_FLAG_FRACTION   (a class constant)

so `filter_instance` builds the object with `object.__new__` and sets exactly
those. That keeps the tests about the filters rather than about config plumbing,
which `tests/portable/test_config_resolution.py` already covers.

`adata_view` is set explicitly to None rather than left absent. `adata_active`
reads it with `getattr(self, 'adata_view', None)`, and its own docstring notes
the default never fires in production because every `_run_*_filter` assigns the
attribute first -- so an isolated instance that omits it returns None and the
filters silently operate on nothing. Setting it is what makes the harness behave
like a mid-cascade instance.
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
