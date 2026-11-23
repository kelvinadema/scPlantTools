"""
Two-layer label-space inputs -- `integration/orchestrator.py::derive_corrected_layers`
and `integration/integrate.py::{stack_layers, exclude_genes_from, mark_trained_features}`.

Every claim here is one the config doc for `derive_corrected_layers:` / `layer:` /
`exclude_genes:` makes in words: the split is sum-preserving and integral, a
corrected count with no raw counts behind it is an error rather than an
allocation, two layers become 2 x genes named features, an excluded gene loses
both of its features, and the saved gene-space object learns which features
trained. See docs/decisions/two-layer-label-spaces.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad

from integration.orchestrator import derive_corrected_layers, CORRECTED_SPLIT_LAYERS
from integration.integrate import (stack_layers, exclude_genes_from, mark_trained_features,
                                   TWO_LAYER_INPUT, FEATURE_SUFFIXES)


def _sample(n_cells=60, n_genes=25, seed=0, removed=0.05):
    """A synthetic mapping artifact: rs/ru Poisson, corrected = raw minus a
    binomial 'ambient' removal, so corrected <= raw everywhere."""
    rng = np.random.default_rng(seed)
    rs = rng.poisson(1.5, (n_cells, n_genes))
    ru = rng.poisson(0.8, (n_cells, n_genes))
    raw = rs + ru
    cc = raw - rng.binomial(raw, removed)
    a = ad.AnnData(X=sp.csr_matrix(cc.astype(float)))
    a.obs_names = ['c{:03d}'.format(i) for i in range(n_cells)]
    a.var_names = ['g{:03d}'.format(i) for i in range(n_genes)]
    a.layers['corrected_counts'] = sp.csr_matrix(cc.astype(float))
    a.layers['rspliced'] = sp.csr_matrix(rs.astype(float))
    a.layers['runspliced'] = sp.csr_matrix(ru.astype(float))
    return a, rs, ru, cc


# --------------------------------------------------- derive_corrected_layers

def test_the_split_sums_exactly_to_corrected_counts_and_is_integral():
    a, rs, ru, cc = _sample()
    derive_corrected_layers(a, 'syn', chunk_rows=7)
    s = a.layers[CORRECTED_SPLIT_LAYERS[0]].toarray()
    u = a.layers[CORRECTED_SPLIT_LAYERS[1]].toarray()
    assert np.array_equal(s + u, cc)
    assert np.all(s >= 0) and np.all(u >= 0)
    assert np.all(s == np.floor(s)) and np.all(u == np.floor(u))


def test_the_spliced_share_is_the_floored_raw_ratio():
    a, rs, ru, cc = _sample(seed=3)
    derive_corrected_layers(a, 'syn')
    s = a.layers[CORRECTED_SPLIT_LAYERS[0]].toarray()
    tot = rs + ru
    nz = cc > 0
    expected = np.floor(cc[nz] * rs[nz] / tot[nz])
    assert np.array_equal(s[nz], expected)
    # Neither half can exceed its own raw count when corrected <= raw.
    assert np.all(s <= rs) and np.all((cc - s) <= ru)


def test_chunking_does_not_change_the_result():
    a1, *_ = _sample(seed=5)
    a2, *_ = _sample(seed=5)
    derive_corrected_layers(a1, chunk_rows=1000)
    derive_corrected_layers(a2, chunk_rows=4)
    for key in CORRECTED_SPLIT_LAYERS:
        assert (a1.layers[key] != a2.layers[key]).nnz == 0


def test_a_corrected_count_with_no_raw_counts_behind_it_raises():
    # CellBender removes counts, never adds: this is a layer mismatch, not a
    # case to allocate.
    a, rs, ru, cc = _sample(seed=1)
    cc2 = cc.copy(); cc2[0, 0] = 3
    rs2 = rs.copy(); rs2[0, 0] = 0
    ru2 = ru.copy(); ru2[0, 0] = 0
    a.layers['corrected_counts'] = sp.csr_matrix(cc2.astype(float))
    a.layers['rspliced'] = sp.csr_matrix(rs2.astype(float))
    a.layers['runspliced'] = sp.csr_matrix(ru2.astype(float))
    with pytest.raises(ValueError, match='rspliced \\+ runspliced == 0'):
        derive_corrected_layers(a, 'syn')


def test_a_missing_source_layer_is_named():
    a, *_ = _sample()
    del a.layers['runspliced']
    with pytest.raises(KeyError, match='runspliced'):
        derive_corrected_layers(a)


def test_the_derivation_is_recorded_and_carries_no_explicit_zeros():
    a, rs, ru, cc = _sample(seed=2)
    record = derive_corrected_layers(a, 'syn')
    assert a.uns['corrected_layers'] is record
    assert record['method'] == 'floor_spliced_share'
    assert record['sum_corrected_rspliced'] + record['sum_corrected_runspliced'] == pytest.approx(cc.sum())
    assert record['n_corrected_gt_raw'] == 0
    for key in CORRECTED_SPLIT_LAYERS:
        m = a.layers[key]
        assert m.nnz == int((m.toarray() > 0).sum())


# ------------------------------------------------------------ stack_layers

def test_two_layers_become_one_feature_per_gene_per_layer():
    a, rs, ru, cc = _sample()
    derive_corrected_layers(a)
    two = stack_layers(a, list(CORRECTED_SPLIT_LAYERS))
    n = a.n_vars
    assert two.shape == (a.n_obs, 2 * n)
    assert list(two.var_names[:n]) == [g + FEATURE_SUFFIXES[0] for g in a.var_names]
    assert list(two.var_names[n:]) == [g + FEATURE_SUFFIXES[1] for g in a.var_names]
    assert list(two.var['gene']) == list(a.var_names) * 2
    assert (two.X[:, :n] != a.layers[CORRECTED_SPLIT_LAYERS[0]]).nnz == 0
    assert (two.X[:, n:] != a.layers[CORRECTED_SPLIT_LAYERS[1]]).nnz == 0
    assert (two.layers[TWO_LAYER_INPUT] != two.X).nnz == 0
    assert list(two.obs_names) == list(a.obs_names)
    assert two.uns['labelspace']['layers'] == list(CORRECTED_SPLIT_LAYERS)


def test_stack_layers_needs_two_distinct_present_layers():
    a, *_ = _sample()
    with pytest.raises(ValueError):
        stack_layers(a, ['rspliced'])
    with pytest.raises(ValueError):
        stack_layers(a, ['rspliced', 'rspliced'])
    with pytest.raises(KeyError, match='nope'):
        stack_layers(a, ['rspliced', 'nope'])


# ------------------------------------------------------- exclude_genes_from

def test_an_excluded_gene_is_gone_from_the_gene_space_object():
    a, *_ = _sample()
    sub, n = exclude_genes_from(a, {'g000', 'g001', 'not_a_gene'})
    assert n == 2
    assert 'g000' not in sub.var_names and 'g001' not in sub.var_names
    assert sub.n_vars == a.n_vars - 2


def test_an_excluded_gene_loses_both_of_its_features():
    a, *_ = _sample()
    derive_corrected_layers(a)
    two = stack_layers(a, list(CORRECTED_SPLIT_LAYERS))
    sub, n = exclude_genes_from(two, {'g003'}, feature_gene_column='gene')
    assert n == 2
    assert not any(v.startswith('g003' + s) for v in sub.var_names for s in FEATURE_SUFFIXES)
    assert sub.n_vars == two.n_vars - 2


def test_excluding_everything_is_refused():
    a, *_ = _sample()
    with pytest.raises(ValueError):
        exclude_genes_from(a, set(a.var_names))


# --------------------------------------------------- mark_trained_features

def test_the_saved_gene_space_object_learns_which_features_trained():
    a, *_ = _sample()
    derive_corrected_layers(a)
    two = stack_layers(a, list(CORRECTED_SPLIT_LAYERS))
    trained = ['g000__s', 'g001__s', 'g001__u', 'g005__u']
    full = a.copy()
    mark_trained_features(full, trained, two_layer=two.uns['labelspace'], excluded={'g007'})
    hs = full.var['hvg_spliced']; hu = full.var['hvg_unspliced']
    assert list(full.var_names[hs]) == ['g000', 'g001']
    assert list(full.var_names[hu]) == ['g001', 'g005']
    assert list(full.var_names[full.var['hvg']]) == ['g000', 'g001', 'g005']
    assert list(full.var_names[full.var['excluded']]) == ['g007']
    assert full.uns['labelspace']['n_hvg_spliced'] == 2
    assert full.uns['labelspace']['n_hvg_unspliced'] == 2
    assert full.uns['labelspace']['n_features_trained'] == 4


def test_single_layer_marking_is_unchanged():
    a, *_ = _sample()
    full = a.copy()
    mark_trained_features(full, ['g002', 'g004'])
    assert list(full.var_names[full.var['hvg']]) == ['g002', 'g004']
    assert 'hvg_spliced' not in full.var and 'excluded' not in full.var
