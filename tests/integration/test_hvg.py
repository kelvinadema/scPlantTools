"""Regression checks for equal-weight, gene-aligned HVG consensus across batches."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from shared.hvg import HVGSelector


def cohort():
    # The shared gene must retain support from both modalities.
    # Unequal cell counts must not give the larger modality more rank weight.
    x = np.vstack([np.tile([100, 90, 1, 0], (10, 1)),
                   np.tile([0, 90, 100, 0], (30, 1))]).astype(float)
    a = ad.AnnData(sparse.csr_matrix(x))
    a.var_names = ['p_only', 'shared', 'n_high', 'undetected']
    a.obs['modality'] = ['protoplast'] * 10 + ['nuclei'] * 30
    return a


@pytest.mark.parametrize('reverse', [False, True])
def test_both_modalities_vote_by_gene_and_export_matches_selection(tmp_path, reverse):
    a = cohort()
    if reverse:
        a = a[np.arange(a.n_obs)[::-1]].copy()
    out = tmp_path / 'hvg.tsv'
    selector = HVGSelector(a, out, n_hvg=2)
    selected = selector.select_hvg(methods=['mean_max'], batch_key='modality')
    ranks = a.var['hvg_mean_max_rank']
    assert ranks['p_only'] == 2.5  # (rank 1 + missing rank 4) / 2
    assert ranks['shared'] == 2
    assert ranks['n_high'] == 2
    assert np.isnan(ranks['undetected'])
    # The two best aggregate scores tie: deterministic gene-name tie break.
    assert list(selected.var_names) == ['n_high', 'shared']
    table = pd.read_csv(out, sep='\t').set_index('gene')
    pd.testing.assert_series_equal(table.hvg_mean_max_rank.reindex(a.var_names),
                                   ranks, check_names=False)
    assert list(table.dropna(subset=['consensus_rank']).index[:2]) == list(selected.var_names)
    assert set(a.var_names[a.var.hvg_consensus]) == set(selected.var_names)
    assert selected.uns['hvg_selection']['algorithm'].endswith('_v2')


def test_detection_floor_is_per_modality_and_invalid_genes_never_fill_slots(tmp_path):
    a = cohort()
    selector = HVGSelector(a, tmp_path / 'hvg.tsv', n_hvg=2000)
    selected = selector.select_hvg(methods=['mean_max'], batch_key='modality', min_cells=11)
    assert list(selected.var_names) == ['n_high', 'shared']
    assert a.var.loc['n_high', 'hvg_mean_max_rank'] == 2  # (missing 3 + rank 1) / 2
    assert a.var.loc['shared', 'hvg_mean_max_rank'] == 2.5
    assert a.var.loc[['p_only', 'undetected'], 'consensus_rank'].isna().all()


def test_methods_are_combined_after_modality_medians(tmp_path, monkeypatch):
    a = cohort()
    selector = HVGSelector(a, tmp_path / 'hvg.tsv', n_hvg=1)
    # Gene-indexed Series deliberately have different index orders in each batch.
    def method_one():
        if selector.adata_raw.obs.modality.iloc[0] == 'protoplast':
            return pd.Series([1., 2., 3.], index=['p_only', 'shared', 'n_high'])
        return pd.Series([1., 2.], index=['n_high', 'shared'])

    def method_two():
        if selector.adata_raw.obs.modality.iloc[0] == 'protoplast':
            return pd.Series([3., 1., 2.], index=['p_only', 'shared', 'n_high'])
        return pd.Series([2., 1.], index=['n_high', 'shared'])

    monkeypatch.setattr(selector, '_mean_max', method_one)
    monkeypatch.setattr(selector, '_mv_lowess_nc', method_two)
    selected = selector.select_hvg(methods=['mean_max', 'mv_lowess_nc'], batch_key='modality')
    # Across batches: p_only=(2.5,3.5), shared=(2,1), n_high=(2,2).
    # Across methods: p_only=3, shared=1.5, n_high=2.
    assert list(selected.var_names) == ['shared']
    assert list(a.var.dropna(subset=['consensus_rank']).sort_values('consensus_rank').index) == [
        'shared', 'n_high', 'p_only']


def test_failed_later_batch_cannot_leave_a_partial_method_vote(tmp_path, monkeypatch):
    a = cohort()
    a.var['hvg_obsolete_rank'] = [1., 2., 3., 4.]
    selector = HVGSelector(a, tmp_path / 'hvg.tsv', n_hvg=2)

    def fails_on_nuclei():
        if selector.adata_raw.obs.modality.iloc[0] == 'nuclei':
            raise ValueError('singular test fit')
        return pd.Series([3., 2., 1.], index=['p_only', 'shared', 'n_high'])

    monkeypatch.setattr(selector, '_mv_lowess_nc', fails_on_nuclei)
    selected = selector.select_hvg(methods=['mean_max', 'mv_lowess_nc'], batch_key='modality')
    assert 'mv_lowess_nc' in selector.failed_methods
    assert 'hvg_mv_lowess_nc_rank' not in a.var
    assert 'hvg_obsolete_rank' not in a.var
    assert list(selected.var_names) == ['n_high', 'shared']


def test_single_batch_rank_direction_override_and_repeated_calls(tmp_path):
    a = cohort()[:10].copy()
    selector = HVGSelector(a, tmp_path / 'hvg.tsv', n_hvg=3)
    first = selector.select_hvg(methods=['mean_max'], batch_key=None, n_hvg=1)
    assert list(first.var_names) == ['p_only']
    second = selector.select_hvg(methods=['mean_max'], batch_key=None)
    assert list(second.var_names) == ['p_only', 'shared', 'n_high']
    assert a.var.hvg_consensus.sum() == 3


def test_tied_medians_use_mean_then_gene_and_flags_agree(tmp_path):
    a = ad.AnnData(np.ones((10, 4)))
    a.var_names = ['z', 'b', 'a', 'w']
    a.var['hvg_one_rank'] = [1., 2., 2., 1.]
    a.var['hvg_two_rank'] = [2., 2., 2., 2.]
    a.var['hvg_three_rank'] = [6., 3., 3., 9.]
    a.var['hvg_consensus'] = [True, False, False, False]  # not a method
    selector = HVGSelector(a, tmp_path / 'unused.tsv')
    _, top = selector._consensus_hvg_rank(n_hvg=2)
    # All medians are 2; means (3, 7/3, 7/3, 4) then identifiers resolve ties.
    assert top == ['a', 'b']
    assert set(a.var_names[a.var.hvg_consensus]) == set(top)


def test_no_eligible_genes_fails_instead_of_selecting_nan_ranks(tmp_path):
    a = cohort()
    selector = HVGSelector(a, tmp_path / 'hvg.tsv')
    with pytest.raises(RuntimeError, match='Every requested method failed'):
        selector.select_hvg(methods=['mean_max'], batch_key='modality', min_cells=31)
    assert not (tmp_path / 'hvg.tsv').exists()


def test_export_method_ranks_without_consensus(tmp_path):
    a = cohort()
    out = tmp_path / 'hvg.tsv'
    selector = HVGSelector(a, out)
    ranked = selector.select_hvg(methods=['mean_max'], batch_key='modality', compute_consensus=False)
    assert ranked.shape == a.shape
    assert 'consensus_rank' not in ranked.var
    assert 'hvg_mean_max_rank' in pd.read_csv(out, sep='\t')


def test_all_eight_methods_run_on_synthetic_counts(tmp_path):
    rng = np.random.default_rng(42)
    rates = np.exp(rng.uniform(-2, 2, 200))
    x = rng.poisson(rates * rng.lognormal(0, 0.3, (100, 1))).astype(float)
    a = ad.AnnData(sparse.csr_matrix(x))
    a.obs['modality'] = ['protoplast'] * 50 + ['nuclei'] * 50
    selector = HVGSelector(a, tmp_path / 'hvg.tsv', n_hvg=20)
    selected = selector.select_hvg(batch_key='modality', min_cells=2)
    assert not selector.failed_methods
    assert len(selected.uns['hvg_selection']['successful_methods']) == 8
    assert selected.shape == (100, 20)
    assert a.var.hvg_consensus.sum() == 20
    # Provenance must remain serializable with the selected training input.
    selected.write_h5ad(tmp_path / 'selected.h5ad')
