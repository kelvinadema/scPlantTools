#!/usr/bin/env python3
"""
Gene modules for the cluster-antagonistic methods.

LMD and CFMF both score genes individually -- they answer "is this gene
localized?", not "which genes go together". The annotation stage needs
*genesets*, so the top-scoring genes are grouped into co-expression modules
here and each module becomes one `clusterName`.

This is a shared post-hoc grouping, **not** either paper's own module
algorithm (LMD has `ClusterGenes`; CFMF has none). It is deliberately one
implementation for both so the two methods' outputs stay comparable. Hotspot is
not routed through here -- it has a native `create_modules` built on its own
local-correlation statistic, which is used instead.

Grouping is on kNN-smoothed expression: two genes localized to the same region
of the manifold correlate strongly once expression is averaged over each cell's
neighbourhood, while dropout suppresses that correlation in the raw matrix.
"""

import logging

import numpy as np
import pandas as pd

from scipy import sparse
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger('MarkerLogging')


def smooth_expression(expression, coords, n_neighbors=30):
    """Average each gene over each cell's kNN. `expression` is cells x genes."""
    n = coords.shape[0]
    k = min(n_neighbors, n - 1)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(
        coords, return_distance=False)
    rows = np.repeat(np.arange(n), idx.shape[1])
    A = sparse.csr_matrix(
        (np.full(idx.size, 1.0 / idx.shape[1], dtype=np.float32),
         (rows, idx.ravel())), shape=(n, n))
    return A @ expression


def assign_modules(expression, coords, genes, min_module_size=10,
                   n_neighbors=30, distance_threshold=0.5):
    """
    Group genes into co-expression modules.

    Returns a Series gene -> module id, with -1 for genes in a group smaller
    than `min_module_size` (unassigned), matching Hotspot's convention so both
    routes label unassigned genes the same way.
    """
    genes = np.asarray(genes)
    if len(genes) < min_module_size:
        logger.warning('[MARKERS] Only %d genes to cluster; no modules formed',
                       len(genes))
        return pd.Series(-1, index=genes, name='module')

    smoothed = smooth_expression(expression, coords, n_neighbors=n_neighbors)
    smoothed = np.asarray(
        smoothed.todense() if sparse.issparse(smoothed) else smoothed, dtype=np.float64)

    # Genes with no variance after smoothing cannot correlate with anything.
    sd = smoothed.std(axis=0)
    usable = sd > 0
    if usable.sum() < min_module_size:
        return pd.Series(-1, index=genes, name='module')

    corr = np.corrcoef(smoothed[:, usable], rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2.0, 0, None)

    linkage_matrix = linkage(squareform(dist, checks=False), method='average')
    labels = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')

    assigned = pd.Series(-1, index=genes, name='module')
    usable_genes = genes[usable]
    sizes = pd.Series(labels).value_counts()
    keep = set(sizes[sizes >= min_module_size].index)
    # Renumber from 1 so ids are contiguous and 1-based, as Hotspot's are.
    remap = {old: new for new, old in enumerate(sorted(keep), start=1)}
    assigned.loc[usable_genes] = [remap.get(l, -1) for l in labels]

    n_mod = len(remap)
    logger.info('[MARKERS] Formed %d gene modules (%d genes unassigned)',
                n_mod, int((assigned == -1).sum()))
    if n_mod == 0:
        logger.warning('[MARKERS] No module reached min_module_size=%d; try a '
                       'larger distance_threshold or more input genes',
                       min_module_size)
    return assigned
