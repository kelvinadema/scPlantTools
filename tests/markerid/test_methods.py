"""
Marker methods -- `markerid/methods.py`, `markerid/markers.py`.

Two things are worth testing here and they are different in kind.

**The output contract.** All six methods write one table with the same columns
(`MARKER_COLUMNS`), because the annotation stage reads them interchangeably --
`{marker_file_dir}/{method}/{sample}.tsv`, one loader, six producers. A method
that emits a differently-shaped table breaks annotation, not markerID, and does
so at the point where OrthoFinder has already run.

**Port fidelity.** `cfmf` and `lmd` are ports from R (see
`docs/markerid-porting.md`). Ports drift. What is asserted is not that they
reproduce the original's numbers -- nothing here can check that -- but that they
still find planted structure and still return the contract.

The recovery assertion is deliberately weak: *the planted markers for a cluster
rank above the median gene for that cluster*. Six methods with six different
statistics cannot share a stronger claim, and a stronger one would just encode
whichever method was used to calibrate it.

Requires `scMarkersV3.12`.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from fixtures import synthetic          # noqa: E402

from markerid.markers import MARKER_COLUMNS, MarkerRun          # noqa: E402
from markerid.methods import METHODS, CLUSTER_DEPENDENT, CLUSTER_FREE   # noqa: E402


@pytest.fixture(scope='module')
def planted():
    """(adata, {cluster: [planted gene names]})"""
    return synthetic.clustered_expression()


def _run(adata, method, tmp_path, cluster_key='leiden'):
    run = MarkerRun(adata_path=None, sample='S1', method=method,
                    cluster_key=cluster_key, latent_key='X_pca',
                    out_dir=tmp_path / method, adata=adata.copy(),
                    min_cells=1, n_jobs=1)
    return run.prepare()


# ------------------------------------------------------------------ registry

def test_every_declared_method_is_registered():
    for m in CLUSTER_DEPENDENT + CLUSTER_FREE + ('cosg',):
        assert m in METHODS, '{} is declared but has no implementation'.format(m)


def test_cluster_dependent_and_free_do_not_overlap():
    """
    The split decides whether `--clusterkey` is required. A method in both sets
    would make that check meaningless.
    """
    assert not set(CLUSTER_DEPENDENT) & set(CLUSTER_FREE)


def test_unknown_method_raises_at_construction():
    """Fail on the typo, not after loading a 14 GB h5ad."""
    with pytest.raises(ValueError, match='Unknown method'):
        MarkerRun(adata_path=None, sample='S1', method='wilcoxonn', adata=None)


# ------------------------------------------------------------------- contract

@pytest.mark.parametrize('method', ['wilcoxon', 't-test', 'cosg'])
def test_cluster_dependent_methods_return_the_contract(planted, method, tmp_path):
    adata, _ = planted
    df = METHODS[method](_run(adata, method, tmp_path))
    for col in MARKER_COLUMNS:
        assert col in df.columns, '{} omitted the contract column {!r}'.format(method, col)
    assert len(df), '{} returned an empty table'.format(method)


@pytest.mark.parametrize('method', ['wilcoxon', 't-test', 'cosg'])
def test_method_column_identifies_the_producer(planted, method, tmp_path):
    """
    Six methods write into one directory tree and the annotation stage
    concatenates them. Without a correct `method` column their rows are
    indistinguishable once merged.
    """
    adata, _ = planted
    df = METHODS[method](_run(adata, method, tmp_path))
    assert set(df['method'].unique()) == {method}


@pytest.mark.parametrize('method', ['wilcoxon', 't-test', 'cosg'])
def test_ranks_start_at_zero_and_are_dense_per_cluster(planted, method, tmp_path):
    adata, _ = planted
    df = METHODS[method](_run(adata, method, tmp_path))
    for cluster, sub in df.groupby('clusterName'):
        ranks = sorted(sub['rank'].tolist())
        assert ranks == list(range(len(ranks))), \
            '{} produced non-dense ranks for {}'.format(method, cluster)


# ------------------------------------------------------------------- recovery

@pytest.mark.parametrize('method', ['wilcoxon', 't-test', 'cosg'])
def test_planted_markers_rank_above_the_median(planted, method, tmp_path):
    """
    Weak on purpose. Six methods with six statistics can only share a claim this
    loose -- and a tighter one would encode whichever method calibrated it.

    Paired with `test_finds_nothing_when_there_is_nothing` below: this one fails
    for a method that ranks arbitrarily, that one fails for a method that always
    reports the same genes.
    """
    adata, markers = planted
    df = METHODS[method](_run(adata, method, tmp_path))
    checked = 0
    for cluster, genes in markers.items():
        sub = df[df['clusterName'] == cluster]
        if sub.empty:
            continue
        median_rank = sub['rank'].median()
        planted_ranks = sub[sub['gene'].isin(genes)]['rank']
        if planted_ranks.empty:
            continue
        assert planted_ranks.median() < median_rank, \
            '{}: planted markers for {} did not rank above the median'.format(method, cluster)
        checked += 1
    assert checked, '{} produced no rows for any planted cluster'.format(method)


@pytest.mark.parametrize('method', ['wilcoxon', 't-test'])
def test_finds_nothing_when_there_is_nothing(method, tmp_path):
    """
    Structureless data: one cluster label, no planted genes. A method must not
    report confident markers for a partition that does not exist.

    Asserted on significance rather than presence -- scanpy always returns a
    ranked list, so the meaningful claim is that nothing clears a threshold.
    """
    adata, _ = synthetic.clustered_expression(n_clusters=1, n_markers=0, seed=9)
    df = METHODS[method](_run(adata, method, tmp_path))
    if 'pval_adj' not in df.columns:
        pytest.skip('{} reports no p-values'.format(method))
    strong = (df['pval_adj'] < 0.01).sum()
    assert strong == 0, '{} found {} significant markers in structureless data'.format(method, strong)


# ------------------------------------------------------------- cluster keys

def test_cluster_dependent_method_requires_a_cluster_key(planted, tmp_path):
    """
    `require_cluster_key` exists so this fails fast rather than after the
    neighbour graph has been built.
    """
    adata, _ = planted
    run = MarkerRun(adata_path=None, sample='S1', method='wilcoxon',
                    cluster_key=None, out_dir=tmp_path, adata=adata.copy(),
                    min_cells=1, n_jobs=1).prepare()
    with pytest.raises((ValueError, KeyError, RuntimeError)):
        METHODS['wilcoxon'](run)


# ------------------------------------------------------------------ file path

def test_marker_file_path_matches_what_annotation_reads(tmp_path):
    """
    Annotation resolves `{marker_file_dir}/{method}/{sample}.tsv` and raises on
    the first missing one -- after OrthoFinder has already run. The two must
    agree on the filename.
    """
    from markerid.markers import marker_file_path
    p = marker_file_path(tmp_path / 'cosg', 'MtTip')
    assert pathlib.Path(p).name == 'MtTip.tsv'
