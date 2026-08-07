#!/usr/bin/env python3
"""
Clustering-Free Marker Finding (CFMF) -- Python port.

Scores each gene by whether the cells expressing it sit together in the
embedding: a neighbourhood *purity* p-value and a *silhouette* on the
expressed/not-expressed split.

Ported from github.com/ShuminYin/project_CFMF (R). See docs/markerid-porting.md for
the correspondence, the two deliberate divergences (one-sided test; ranking by
statistic rather than trusting the combined p-value), and the reformulations
that make it tractable.

Both reformulations are exact, not approximations -- verified bit-identical to
the naive per-cell loop and to sklearn.silhouette_samples respectively:

  purity     the per-cell 2x2 Fisher test has fixed margins, so it is exactly a
             hypergeometric tail. One sparse matmul gives the positive-neighbour
             count for every (cell, gene); and since k and n are fixed, a gene
             has only k+1 possible overlaps, so a per-gene lookup table replaces
             one hypergeom call per expressing cell.
  silhouette the distance matrix does not depend on the gene, so `D @ B` yields
             every gene's within-group distance sums in a single matmul.

Preprint: doi:10.1101/2025.10.26.683836
"""

import logging

import numpy as np
import pandas as pd

from scipy import sparse
from scipy.stats import chi2, hypergeom
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger('MarkerLogging')

# The silhouette needs an n x n distance matrix. Exact at full resolution given
# the RAM (62,750 cells is ~15.7 GB in float32); above this the cells are
# subsampled, following the MAX_DOD_CELLS convention in qc_filter.py.
MAX_SILHOUETTE_CELLS = 25000


def neighbourhood_purity(binary, knn_index, alternative='greater'):
    """
    Per-gene neighbourhood purity, combined across expressing cells.

    `binary` is cells x genes (bool), `knn_index` is cells x k.

    For an expressing cell j, the original tests a 2x2 table of
    (is a kNN of j) x (is expressing) over the other n-1 cells. With k and the
    expressing count fixed those margins are fixed, so the one-sided test is
    the hypergeometric survival function; cell j is excluded from its own
    population, hence n-1 and m-1.

    Per-gene combination is Fisher's method, as published. Note the p-values
    being combined are strongly dependent (neighbouring cells share most of
    their kNN), so the result is anti-conservative -- see docs/markerid-porting.md.
    """
    if alternative not in ('greater', 'two-sided'):
        raise ValueError(f"alternative must be 'greater' or 'two-sided', got {alternative!r}")
    if alternative == 'two-sided':
        logger.warning('[MARKERS] CFMF: two-sided purity test reproduces the '
                       'original but also rewards genes whose expressing cells '
                       'avoid each other')

    n_cells, n_genes = binary.shape
    k = knn_index.shape[1]

    rows = np.repeat(np.arange(n_cells), k)
    A = sparse.csr_matrix(
        (np.ones(n_cells * k, dtype=np.float32), (rows, knn_index.ravel())),
        shape=(n_cells, n_cells))
    B = sparse.csr_matrix(binary.astype(np.float32))
    # positive neighbours per (cell, gene)
    overlap = np.asarray((A @ B).todense()).astype(np.int32)

    m = binary.sum(axis=0).astype(np.int64)
    xs = np.arange(k + 1)
    # (genes x k+1) table: every p-value a gene can produce.
    with np.errstate(divide='ignore', invalid='ignore'):
        table = hypergeom.sf(xs[None, :] - 1, n_cells - 1,
                             np.maximum(m - 1, 0)[:, None], k)
    if alternative == 'two-sided':
        lower = hypergeom.cdf(xs[None, :], n_cells - 1,
                              np.maximum(m - 1, 0)[:, None], k)
        table = np.minimum(1.0, 2.0 * np.minimum(table, lower))
    table = np.clip(np.nan_to_num(table, nan=1.0), 1e-300, 1.0)

    meta_p = np.ones(n_genes)
    for g in range(n_genes):
        pos = np.flatnonzero(binary[:, g])
        if pos.size == 0:
            continue
        pvals = table[g, overlap[pos, g]]
        stat = -2.0 * np.log(pvals).sum()
        meta_p[g] = chi2.sf(stat, 2 * pvals.size)
    return meta_p


def gene_silhouette(binary, coords, max_cells=MAX_SILHOUETTE_CELLS, random_state=42):
    """
    Median silhouette width over each gene's expressing cells.

    Exact: for a two-group split, a(i) and b(i) are group-wise mean distances,
    so `D @ B` gives every gene's positive-distance sums at once and the
    negative sums are the row totals minus those. Verified against
    sklearn.silhouette_samples to 3e-15.

    Cells are subsampled above `max_cells` because D is n x n; the statistic is
    a median over cells and tolerates it, but the subsample is recorded.
    """
    n_cells, n_genes = binary.shape
    rng = np.random.default_rng(random_state)
    if n_cells > max_cells:
        logger.info('[MARKERS] CFMF: subsampling %d -> %d cells for silhouette',
                    n_cells, max_cells)
        sel = rng.choice(n_cells, size=max_cells, replace=False)
        binary = binary[sel]
        coords = coords[sel]
        n_cells = max_cells

    D = cdist(coords, coords).astype(np.float32)
    row_total = D.sum(axis=1)
    B = binary.astype(np.float32)
    pos_sum = D @ B                                    # (cells x genes)

    m = B.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = pos_sum / np.maximum(m - 1, 1)[None, :]     # exclude self
        b = (row_total[:, None] - pos_sum) / np.maximum(n_cells - m, 1)[None, :]
        sil = (b - a) / np.maximum(a, b)
    sil[~np.isfinite(sil)] = 0.0

    out = np.full(n_genes, -1.0)
    for g in range(n_genes):
        pos = np.flatnonzero(binary[:, g])
        # A gene present in every cell or none has no two-group split.
        if pos.size and pos.size < n_cells:
            out[g] = float(np.median(sil[pos, g]))
    return out


def cfmf_scores(expression, coords, genes, k=50, min_cells=4,
                alternative='greater', max_silhouette_cells=MAX_SILHOUETTE_CELLS,
                random_state=42):
    """
    Run CFMF. `expression` is cells x genes (the normalised layer, thresholded
    at > 0 as the original does), `coords` is the cell embedding -- pass
    `X_scVI`.

    Returns a DataFrame: gene, silhouette, meta_p, FDR, pct, count.
    """
    binary = np.asarray(
        (expression > 0).todense() if sparse.issparse(expression) else expression > 0)
    count = binary.sum(axis=0)
    pct = count / binary.shape[0]
    # Original: count > 3 and pct < 1.
    keep = (count > min_cells - 1) & (pct < 1.0)
    if not keep.any():
        raise ValueError('CFMF: no gene passed the detection filter')
    logger.info('[MARKERS] CFMF: %d/%d genes pass the detection filter',
                int(keep.sum()), len(genes))

    binary = binary[:, keep]
    kept_genes = np.asarray(genes)[keep]

    coords = np.asarray(coords, dtype=np.float64)
    k = min(k, coords.shape[0] - 1)
    knn_index = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(
        coords, return_distance=False)[:, 1:]

    logger.info('[MARKERS] CFMF: computing neighbourhood purity (k=%d)', k)
    meta_p = neighbourhood_purity(binary, knn_index, alternative=alternative)
    logger.info('[MARKERS] CFMF: computing silhouettes')
    silhouette = gene_silhouette(binary, coords, max_cells=max_silhouette_cells,
                                 random_state=random_state)

    fdr = multipletests(meta_p, method='fdr_bh')[1]
    return pd.DataFrame({
        'gene': kept_genes,
        'silhouette': silhouette,
        'meta_p': meta_p,
        'FDR': fdr,
        'pct': pct[keep],
        'count': count[keep]})
