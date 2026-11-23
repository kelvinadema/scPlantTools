#!/usr/bin/env python3
"""
The marker identification methods, behind one registry.

Every method takes a prepared `MarkerRun` and returns a DataFrame in the shared
contract (see markers.py::MARKER_COLUMNS):

    clusterName | rank | gene | score | method   (+ method-specific columns)

Cluster-dependent methods put the cluster label in `clusterName`;
cluster-free ones put a gene-module id there, so both feed the
annotation stage through the same path.
"""

import logging

import numpy as np
import pandas as pd

from .cfmf import cfmf_scores
from .lmd import lmd_scores
from .modules import assign_modules

logger = logging.getLogger('MarkerLogging')

CLUSTER_DEPENDENT = ('wilcoxon', 't-test')
CLUSTER_FREE = ('hotspot', 'lmd', 'cfmf')


def _module_frame(assignments, scores, method, prefix, extra=None):
    """Turn gene -> module plus a per-gene score into the output contract."""
    assigned = assignments[assignments >= 0]
    if assigned.empty:
        return pd.DataFrame(columns=['clusterName', 'rank', 'gene', 'score', 'method'])
    df = pd.DataFrame({
        'clusterName': [f'{prefix}_M{m}' for m in assigned.to_numpy()],
        'gene': assigned.index.to_numpy(),
        'score': scores.reindex(assigned.index).to_numpy()})
    if extra is not None:
        df = df.merge(extra, on='gene', how='left')
    df['method'] = method
    return df


def _rank_within_cluster(df, n_genes, ascending=False):
    """Rank genes inside each geneset and truncate. `ascending` for scores where
    lower is better (LMD)."""
    df = df.sort_values(['clusterName', 'score'], ascending=[True, ascending])
    df['rank'] = df.groupby('clusterName').cumcount()
    if n_genes is not None:
        df = df[df['rank'] < n_genes]
    front = ['clusterName', 'rank', 'gene', 'score', 'method']
    cols = front + [c for c in df.columns if c not in front]
    return df[cols].reset_index(drop=True)


# --------------------------------------------------------------- cluster-based

def run_scanpy(run, groups='all', reference='rest', n_genes=50, **kwargs):
    """scanpy rank_genes_groups (wilcoxon or t-test)."""
    import scanpy as sc

    cluster_key = run.require_cluster_key()
    sc.tl.rank_genes_groups(
        run.adata, groupby=cluster_key, method=run.method, pts=True,
        groups=groups, reference=reference, use_raw=False,
        layer=run.norm_layer, **kwargs)

    result = run.adata.uns['rank_genes_groups']
    records = [
        {'clusterName': group, 'rank': rank, 'gene': gene,
         'score': result['scores'][group][rank],
         'logfoldchange': result['logfoldchanges'][group][rank],
         'pval': result['pvals'][group][rank],
         'pval_adj': result['pvals_adj'][group][rank],
         'method': run.method}
        for group in result['names'].dtype.names
        for rank, gene in enumerate(result['names'][group])]
    df = pd.DataFrame(records)
    if n_genes is not None:
        df = df[df['rank'] < n_genes]
    return df.reset_index(drop=True)


def run_cosg(run, n_genes=50, mu=1.0, expressed_pct=0.1,
             remove_lowly_expressed=True, **kwargs):
    """COSG -- cosine-similarity marker genes (doi:10.1093/bib/bbab579)."""
    import cosg as cosg_module

    cluster_key = run.require_cluster_key()
    key = 'cosg'
    cosg_module.cosg(
        run.adata, groupby=cluster_key, mu=mu, n_genes_user=n_genes or 50,
        remove_lowly_expressed=remove_lowly_expressed, expressed_pct=expressed_pct,
        use_raw=False, layer=run.norm_layer, key_added=key, **kwargs)

    result = run.adata.uns[key]
    records = [
        {'clusterName': group, 'rank': rank, 'gene': gene,
         'score': result['scores'][group][rank], 'method': 'cosg'}
        for group in result['names'].dtype.names
        for rank, gene in enumerate(result['names'][group])]
    return pd.DataFrame(records).reset_index(drop=True)


# ---------------------------------------------------------------- cluster-free

def run_hotspot(run, n_genes=None, n_neighbors=30, model='danb', fdr_threshold=0.05,
                n_top_genes=1000, min_module_size=15, jobs=None, **kwargs):
    """
    Hotspot -- spatial autocorrelation on the latent space
    (doi:10.1016/j.cels.2021.04.005).

    Uses Hotspot's own module detection rather than modules.assign_modules: it
    is built on the same local-correlation statistic as the gene scores, so it
    is the method's own notion of a geneset.
    """
    import hotspot

    # Both phases run in parallel under the `forkserver` start method:
    # fork()ing a process with live OpenMP threads is a reliable SIGSEGV.
    # Results are bit-identical; memory, not CPU, caps `jobs`. See
    # docs/decisions/hotspot-parallelism.md.
    n_jobs = jobs if jobs is not None else max(int(getattr(run, 'n_jobs', 1) or 1), 1)
    if n_jobs > 1:
        import multiprocessing
        if multiprocessing.get_start_method(allow_none=True) != 'forkserver':
            # force=True because scanpy/anndata may already have set 'fork'.
            multiprocessing.set_start_method('forkserver', force=True)
        logger.info('[MARKERS] Hotspot: %d job(s), start method %r',
                    n_jobs, multiprocessing.get_start_method())

    # Applied at ANY job count: the kNN seed is what makes a hotspot run
    # reproducible at all, serial or parallel. See markerid/hotspot_patch.py,
    # including why marker files made before the patch cannot be reproduced.
    from .hotspot_patch import apply as _apply_hotspot_patch
    _apply_hotspot_patch(seed=getattr(run, 'random_state', 42))

    counts_layer = run.counts_layer_for_hotspot()
    umi_key = run.umi_counts_key()
    hs = hotspot.Hotspot(run.adata, layer_key=counts_layer, model=model,
                         latent_obsm_key=run.latent_key, umi_counts_obs_key=umi_key)
    hs.create_knn_graph(weighted_graph=False, n_neighbors=n_neighbors)

    autocorr = hs.compute_autocorrelations(jobs=n_jobs)
    significant = autocorr[autocorr['FDR'] < fdr_threshold]
    logger.info('[MARKERS] Hotspot: %d/%d genes with FDR < %.3g',
                len(significant), len(autocorr), fdr_threshold)
    if significant.empty:
        return pd.DataFrame(columns=['clusterName', 'rank', 'gene', 'score', 'method'])

    top = significant.sort_values('Z', ascending=False).head(n_top_genes).index
    # If this phase ever crashes, check whether it survives with n_jobs=1: a
    # SIGSEGV with no Python traceback shortly after "Computing pair-wise local
    # correlation..." is the fork-with-live-OpenMP-threads signature.
    hs.compute_local_correlations(top, jobs=n_jobs)
    assignments = hs.create_modules(min_gene_threshold=min_module_size,
                                    core_only=True, fdr_threshold=fdr_threshold)
    assignments = assignments.astype(int)

    extra = autocorr.loc[:, ['C', 'Z', 'Pval', 'FDR']].rename_axis('gene').reset_index()
    df = _module_frame(assignments, autocorr['Z'], 'hotspot', 'hotspot', extra=extra)
    run.module_scores = hs.calculate_module_scores() if not df.empty else None
    return _rank_within_cluster(df, n_genes, ascending=False)


def run_lmd(run, n_genes=None, knn=5, max_time=2 ** 20, n_top_genes=1000,
            min_module_size=10, n_metacells=5000, **kwargs):
    """
    LMD -- diffusion-based localization (doi:10.1038/s42003-025-08485-y).

    Lower LMDS means more localized, so genes are ranked ascending here and the
    top `n_top_genes` are the *smallest* scores.
    """
    expression = run.expression(genes_by_cells=True)
    scores, gene_mask, n_scales = lmd_scores(
        expression, run.latent(), knn=knn, max_time=max_time,
        n_metacells=n_metacells, random_state=run.random_state)

    genes = np.asarray(run.adata.var_names)[gene_mask]
    lmds = pd.Series(scores, index=genes, name='lmds')
    finite = lmds[np.isfinite(lmds)]
    if finite.empty:
        logger.warning('[MARKERS] LMD produced no finite scores')
        return pd.DataFrame(columns=['clusterName', 'rank', 'gene', 'score', 'method'])

    top = finite.nsmallest(min(n_top_genes, len(finite))).index
    sub = run.expression_subset(top)
    assignments = assign_modules(sub, run.latent(), top,
                                 min_module_size=min_module_size)
    df = _module_frame(assignments, lmds, 'lmd', 'lmd')
    df = df.rename(columns={'score': 'lmds'})
    df['score'] = df['lmds']
    run.n_scales = n_scales
    # ascending: a smaller LMDS is a better marker
    return _rank_within_cluster(df, n_genes, ascending=True)


def run_cfmf(run, n_genes=None, k=50, n_top_genes=1000, min_module_size=10,
             alternative='greater', fdr_threshold=0.05, **kwargs):
    """
    CFMF -- neighbourhood purity + silhouette (doi:10.1101/2025.10.26.683836).

    Ranked by silhouette rather than by FDR: the combined purity p-value is
    anti-conservative by construction (see docs/markerid-porting.md), so it is reported
    but not used to order genes.
    """
    stats = cfmf_scores(run.expression(genes_by_cells=False), run.latent(),
                        np.asarray(run.adata.var_names), k=k,
                        alternative=alternative, random_state=run.random_state)

    passing = stats[stats['FDR'] < fdr_threshold]
    if passing.empty:
        logger.warning('[MARKERS] CFMF: no gene passed FDR < %.3g; '
                       'falling back to silhouette ranking alone', fdr_threshold)
        passing = stats
    top = passing.nlargest(min(n_top_genes, len(passing)), 'silhouette')['gene']

    sub = run.expression_subset(top)
    assignments = assign_modules(sub, run.latent(), np.asarray(top),
                                 min_module_size=min_module_size)
    scores = stats.set_index('gene')['silhouette']
    extra = stats.drop(columns=['silhouette'])
    df = _module_frame(assignments, scores, 'cfmf', 'cfmf', extra=extra)
    return _rank_within_cluster(df, n_genes, ascending=False)


METHODS = {
    'wilcoxon': run_scanpy,
    't-test': run_scanpy,
    'cosg': run_cosg,
    'hotspot': run_hotspot,
    'lmd': run_lmd,
    'cfmf': run_cfmf,
}
