#!/usr/bin/env python3
"""
Localized Marker Detector (LMD) -- Python port.

Chau-style cluster-free marker detection: a gene's expression, normalised to a
probability distribution over cells, is diffused across the cell-cell graph.
A marker stays put; a ubiquitous or noisy gene washes out immediately.

Ported from github.com/KlugerLab/LocalizedMarkerDetector (R). See
docs/markerid-porting.md for the correspondence to the original, including the two
places it is easy to get wrong: the *unweighted* graph symmetrisation and the
fact that a **lower** LMDS means a better marker.

Paper: doi:10.1038/s42003-025-08485-y
"""

import logging

import numpy as np

from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger('MarkerLogging')

# Above this many cells the dense diffusion operator stops being affordable
# (n^2 floats), and the original's own CoarseGrain step is applied instead.
COARSE_GRAIN_ABOVE = 10000
DEFAULT_N_METACELLS = 5000


def kl_rows(P, Q):
    """
    Row-wise KL(p || q), matching the original's calcKL exactly.

    Terms with p == 0 are skipped, and any non-finite term is treated as 0 --
    the C++ does `if (p[j] == 0) continue;` then zeroes NaN/inf. Q may be a
    single row (broadcast) and is *not* renormalised, which is what makes
    KL(p || ones) equal -H(p).
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if Q.ndim == 1:
        Q = Q[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = P * (np.log(P) - np.log(Q))
    terms[P == 0] = 0.0
    terms[~np.isfinite(terms)] = 0.0
    return terms.sum(axis=1)


def construct_knn_graph(X, knn=5, self_loop=1.0, min_component=10,
                        adjust_disconnection=True):
    """
    Symmetric unweighted kNN affinity graph over cells.

    `W = min(A + A.T, 1)` in the original -- a logical OR of the directed
    adjacency, not the averaged `(A + A.T)/2` (which is present but commented
    out there). Components smaller than `min_component` are dropped and the
    remainder joined, since the diffusion needs a connected graph.

    Returns (W, keep) where `keep` indexes the retained cells.
    """
    n = X.shape[0]
    knn = min(knn, n - 1)
    idx = NearestNeighbors(n_neighbors=knn + 1).fit(X).kneighbors(
        X, return_distance=False)[:, 1:]
    rows = np.repeat(np.arange(n), knn)
    A = sparse.csr_matrix(
        (np.ones(n * knn, dtype=np.float64), (rows, idx.ravel())), shape=(n, n))
    W = A.maximum(A.T)                      # entries are 0/1, so max == min(A+A.T, 1)

    n_comp, labels = connected_components(W, directed=False)
    keep = np.arange(n)
    if n_comp > 1:
        sizes = np.bincount(labels)
        big = np.flatnonzero(sizes >= min_component)
        keep = np.flatnonzero(np.isin(labels, big))
        if len(keep) < n:
            logger.info('[MARKERS] LMD: dropping %d cells in components < %d',
                        n - len(keep), min_component)
        W = W[keep][:, keep]
        X = X[keep]
        n_comp, labels = connected_components(W, directed=False)

    if n_comp > 1 and adjust_disconnection:
        logger.info('[MARKERS] LMD: %d disconnected components, joining', n_comp)
        W = _connect_components(W, X, labels, n_comp)

    W = W.tolil()
    W.setdiag(self_loop)
    return W.tocsr(), keep


def _connect_components(W, X, labels, n_comp):
    """Join components by the shortest edge between each pair, greedily (an MST
    over components, which is what the original builds with igraph)."""
    W = W.tolil()
    comps = [np.flatnonzero(labels == c) for c in range(n_comp)]
    merged = {0}
    remaining = set(range(1, n_comp))
    while remaining:
        best = None
        for a in merged:
            tree = NearestNeighbors(n_neighbors=1).fit(X[comps[a]])
            for b in remaining:
                dist, pos = tree.kneighbors(X[comps[b]], return_distance=True)
                j = int(np.argmin(dist))
                d = float(dist[j, 0])
                if best is None or d < best[0]:
                    best = (d, comps[a][int(pos[j, 0])], comps[b][j], b)
        _d, i, j, b = best
        W[i, j] = W[j, i] = 1.0
        merged.add(b)
        remaining.discard(b)
    return W.tocsr()


def sinkhorn_knopp_sym(W, niter=100, tol=1e-8):
    """
    Symmetric Sinkhorn-Knopp scaling to a doubly stochastic operator.

    The `c = (c + r)/2` averaging is in the original and is not cosmetic: the
    symmetric iteration otherwise oscillates between two sequences.
    """
    dense = not sparse.issparse(W)
    n = W.shape[0]
    r = np.ones(n)
    c = np.ones(n)
    for _ in range(niter):
        r = 1.0 / np.maximum(W @ c, 1e-300)
        cnew = 1.0 / np.maximum(W.T @ r, 1e-300)
        cnew = (cnew + r) / 2.0
        if np.linalg.norm(cnew - c) < tol:
            c = cnew
            break
        c = cnew
    if dense:
        return r[:, None] * W * c[None, :]
    return sparse.diags(r) @ W @ sparse.diags(c)


def coarse_grain(X, expression, W, n_metacells, random_state=1):
    """
    Reduce cells to k-means metacells, as the original's CoarseGrain.

    `expression' = expression @ B.T` (B the 0/1 membership, so counts are
    summed within a metacell) and `W' = M W M.T` (M row-normalised, so the
    affinity is averaged). MiniBatchKMeans rather than Lloyd's: at 62k cells
    and 5k centres, full k-means is the dominant cost for no benefit here.
    """
    n_metacells = min(n_metacells, X.shape[0])
    logger.info('[MARKERS] LMD: coarse-graining %d cells -> %d metacells',
                X.shape[0], n_metacells)
    km = MiniBatchKMeans(n_clusters=n_metacells, random_state=random_state,
                         n_init=3, batch_size=4096).fit(X)
    labels = km.labels_
    n = X.shape[0]
    B = sparse.csr_matrix(
        (np.ones(n), (labels, np.arange(n))), shape=(n_metacells, n))
    counts = np.asarray(B.sum(axis=1)).ravel()
    M = sparse.diags(1.0 / np.maximum(counts, 1)) @ B

    expression_cg = expression @ B.T.tocsc()
    W_cg = (M @ W @ M.T).toarray()
    return np.asarray(expression_cg.todense() if sparse.issparse(expression_cg)
                      else expression_cg), W_cg, labels


def lmd_scores(expression, feature_space, knn=5, max_time=2 ** 20, self_loop=1.0,
               min_cell=5, n_metacells=DEFAULT_N_METACELLS,
               coarse_grain_above=COARSE_GRAIN_ABOVE, random_state=1):
    """
    LMD score per gene. **Lower is better** (more localized).

    `expression` is genes x cells (sparse or dense), `feature_space` is
    cells x dims -- pass `X_scVI` here, which is what the original's
    `feature_space` argument is for.

    Returns (scores, genes_kept_mask, n_scales).
    """
    expression = expression.tocsr() if sparse.issparse(expression) else np.asarray(expression)
    X = np.asarray(feature_space, dtype=np.float64)

    W, keep = construct_knn_graph(X, knn=knn, self_loop=self_loop)
    if len(keep) < X.shape[0]:
        expression = expression[:, keep]
        X = X[keep]

    # Genes must be present in at least min_cell cells (original's sanity check).
    detected = np.asarray((expression > 0).sum(axis=1)).ravel()
    gene_mask = detected >= min_cell
    expression = expression[gene_mask]
    logger.info('[MARKERS] LMD: %d genes retained (expressed in >= %d cells)',
                int(gene_mask.sum()), min_cell)

    if X.shape[0] > coarse_grain_above:
        expression, W, _labels = coarse_grain(
            X, expression, W, n_metacells, random_state=random_state)
    else:
        W = W.toarray()

    # rho: each gene row is a probability distribution over cells.
    rho = np.asarray(expression.todense() if sparse.issparse(expression) else expression,
                     dtype=np.float64)
    totals = rho.sum(axis=1, keepdims=True)
    nonzero = totals.ravel() > 0
    rho = np.divide(rho, np.maximum(totals, 1e-300))

    P = sinkhorn_knopp_sym(W).astype(np.float64)
    n_cells = P.shape[0]
    uniform = np.full(n_cells, 1.0 / n_cells)

    max_score0 = kl_rows(rho, uniform)             # scale factor
    cumulative = np.zeros(rho.shape[0])
    n_scales = 0
    for _t in range(int(np.floor(np.log2(max_time)))):
        # Converged when the operator has flattened to the uniform stationary
        # state; the original tests the diagonal against 1/n.
        if np.max(np.abs(np.diag(P) * n_cells - 1.0)) < 1e-2:
            break
        P = P @ P
        state = rho @ P
        cumulative += kl_rows(rho, state)
        n_scales += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        scores = cumulative / max_score0
    scores[~np.isfinite(scores)] = np.inf
    scores[~nonzero] = np.inf
    logger.info('[MARKERS] LMD: %d diffusion scales', n_scales)
    return scores, gene_mask, n_scales
