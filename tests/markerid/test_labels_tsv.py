"""
External labels for the cluster-dependent methods -- `--labels-tsv`
(`markerid/markers.py::MarkerRun._attach_external_labels`, added 2026-09-13
for chapter 3's marker session).

The point of the option: a marker table computed on a DECLARED label set (a
refine output, a chapter's reviewed label file) rather than whichever Leiden
key the h5ad carries. What has to hold:

  * the ranking is grouped by the TSV's labels, under the TSV's names;
  * cells absent from the table, or labelled `Unassigned`, are dropped rather
    than ranked as a group of their own;
  * both cell-id conventions are matched (bare barcodes; `<sample>.<barcode>`);
  * groups below `--min-cells-per-group` are dropped, not ranked;
  * the planted structure is still recovered under the renamed labels.

Requires `scMarkersV3.12`.
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from fixtures import synthetic          # noqa: E402

from markerid.markers import (EXTERNAL_LABEL_KEY, MARKER_COLUMNS,   # noqa: E402
                              MarkerRun, identify_markers)
from markerid.methods import METHODS    # noqa: E402

RENAME = {'c0': 'cortex', 'c1': 'stele', 'c2': 'epidermis'}


@pytest.fixture(scope='module')
def planted():
    return synthetic.clustered_expression()


def _labels_frame(adata, prefix=None):
    cells = adata.obs_names.astype(str)
    if prefix:
        cells = prefix + '.' + cells
    return pd.DataFrame({'cell': cells,
                         'label': adata.obs['leiden'].astype(str).map(RENAME).to_numpy()})


def _run(adata, tsv, method='wilcoxon', **kwargs):
    run = MarkerRun(adata_path=None, sample='S1', method=method, latent_key='X_pca',
                    out_dir=kwargs.pop('out_dir'), adata=adata.copy(), min_cells=1,
                    n_jobs=1, labels_tsv=tsv, **kwargs)
    return run.prepare()


def test_ranking_is_grouped_by_the_tsv_labels(planted, tmp_path):
    adata, _ = planted
    tsv = tmp_path / 'labels.tsv'
    _labels_frame(adata).to_csv(tsv, sep='\t', index=False)
    run = _run(adata, tsv, out_dir=tmp_path)
    assert run.cluster_key == EXTERNAL_LABEL_KEY
    df = METHODS['wilcoxon'](run)
    for col in MARKER_COLUMNS:
        assert col in df.columns
    assert set(df['clusterName']) == set(RENAME.values())


def test_absent_and_excluded_cells_are_dropped(planted, tmp_path):
    adata, _ = planted
    frame = _labels_frame(adata)
    frame.loc[:19, 'label'] = 'Unassigned'          # 20 cells excluded by label
    frame = frame.iloc[:-10]                        # 10 cells absent from the table
    tsv = tmp_path / 'labels.tsv'
    frame.to_csv(tsv, sep='\t', index=False)
    run = _run(adata, tsv, out_dir=tmp_path)
    assert run.adata.n_obs == adata.n_obs - 30
    assert 'Unassigned' not in set(run.adata.obs[EXTERNAL_LABEL_KEY].astype(str))
    df = METHODS['wilcoxon'](run)
    assert 'Unassigned' not in set(df['clusterName'])


def test_prefixed_cell_ids_match_a_bare_object(planted, tmp_path):
    """Chapter label files carry `<sample>.<barcode>`; a mapping-stage object
    carries bare barcodes plus `.obs['sample']`."""
    adata, _ = planted
    adata = adata.copy()
    adata.obs['sample'] = 'S1'
    tsv = tmp_path / 'labels.tsv'
    _labels_frame(adata, prefix='S1').to_csv(tsv, sep='\t', index=False)
    run = _run(adata, tsv, out_dir=tmp_path)
    assert run.adata.n_obs == adata.n_obs
    assert set(run.adata.obs[EXTERNAL_LABEL_KEY].astype(str)) == set(RENAME.values())


def test_no_matching_cell_is_an_error_not_an_empty_table(planted, tmp_path):
    adata, _ = planted
    tsv = tmp_path / 'labels.tsv'
    pd.DataFrame({'cell': ['nope1', 'nope2'], 'label': ['a', 'b']}).to_csv(tsv, sep='\t', index=False)
    with pytest.raises(ValueError, match='no cell'):
        _run(adata, tsv, out_dir=tmp_path)


def test_small_groups_are_dropped_before_ranking(planted, tmp_path):
    adata, _ = planted
    frame = _labels_frame(adata)
    frame.loc[frame.index[:5], 'label'] = 'tiny'    # a 5-cell label
    tsv = tmp_path / 'labels.tsv'
    frame.to_csv(tsv, sep='\t', index=False)
    run = _run(adata, tsv, out_dir=tmp_path, min_cells_per_group=20)
    assert 'tiny' not in set(run.adata.obs[EXTERNAL_LABEL_KEY].astype(str))
    assert run.adata.n_obs == adata.n_obs - 5
    df = METHODS['wilcoxon'](run)
    assert 'tiny' not in set(df['clusterName'])


@pytest.mark.parametrize('method', ['wilcoxon', 't-test', 'cosg'])
def test_planted_markers_are_recovered_under_the_renamed_labels(planted, method, tmp_path):
    adata, markers = planted
    tsv = tmp_path / 'labels.tsv'
    _labels_frame(adata).to_csv(tsv, sep='\t', index=False)
    df = METHODS[method](_run(adata, tsv, method=method, out_dir=tmp_path))
    checked = 0
    for cluster, genes in markers.items():
        sub = df[df['clusterName'] == RENAME[cluster]]
        if sub.empty:
            continue
        planted_ranks = sub[sub['gene'].isin(genes)]['rank']
        assert planted_ranks.median() < sub['rank'].median()
        checked += 1
    assert checked == len(markers)


def test_identify_markers_accepts_the_option_and_writes_the_file(planted, tmp_path):
    adata, _ = planted
    tsv = tmp_path / 'labels.tsv'
    _labels_frame(adata).to_csv(tsv, sep='\t', index=False)
    df = identify_markers(None, 'S1', method='wilcoxon', latent_key='X_pca',
                          out_dir=tmp_path / 'wilcoxon', n_genes=10, adata=adata.copy(),
                          min_cells=1, labels_tsv=str(tsv))
    assert (tmp_path / 'wilcoxon' / 'S1.tsv').exists()
    assert set(df['clusterName']) == set(RENAME.values())
    assert df.groupby('clusterName')['rank'].max().max() == 9


def test_cli_parses_the_labels_options(monkeypatch, tmp_path):
    """The CLI flags reach `run_sample` as the `labels` dict."""
    from markerid import main as cli
    seen = {}

    def fake_run_sample(*args, **kwargs):
        seen.update(kwargs)
    monkeypatch.setattr(cli, 'run_sample', fake_run_sample)
    monkeypatch.setattr(cli, 'setup_logger', lambda **k: None)
    monkeypatch.setattr(cli, 'pipeline_log_path', lambda *a, **k: str(tmp_path / 'x.log'))
    tsv = tmp_path / 'labels.tsv'
    tsv.write_text('cell\tlabel\n')
    monkeypatch.setattr(sys, 'argv', [
        'markerid', '--method', 'wilcoxon', '--anndata', str(tmp_path / 'a.h5ad'),
        '--name', 'S1', '--labels-tsv', str(tsv), '--labels-col', 'cell_type',
        '--labels-exclude', 'Unassigned', 'Doublet', '--min-cells-per-group', '20',
        '--outdir', str(tmp_path)])
    cli.main()
    assert seen['labels'] == dict(labels_tsv=str(tsv), labels_col='cell_type',
                                  labels_cell_col='cell',
                                  labels_exclude=('Unassigned', 'Doublet'),
                                  min_cells_per_group=20)
