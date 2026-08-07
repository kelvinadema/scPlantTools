#!/usr/bin/env python3
"""
Two monkeypatches over hotspot 1.1.3, applied by methods.py::run_hotspot.

Both are behaviour-preserving in the sense that matters: they change how work is
distributed and how a random seed is chosen, never the statistic. Every number
still comes out of hotspot's own `_compute_hs_inner`.

Kept as patches rather than a vendored fork because hotspot is an unmodified
conda dependency and a fork would silently stop tracking upstream. The tradeoff
is that both patches are pinned to internals of one version -- see VERSION
SENSITIVITY at the bottom before upgrading hotspot.


PATCH 1 -- compute_hs feeds workers indices instead of dense vectors
--------------------------------------------------------------------
Measured 2026-08-05 on MtSZ_DRVI_10x_full_fixed (35,115 cells, 30,049 genes),
hotspot's two phases at jobs=8:

    autocorrelation (compute_hs)            31.0 min   vs 31.2 serial  -> 1.0x
    pairwise (compute_hs_pairs_centered_cond) 12.2 min vs 33.0 serial  -> 2.7x

Same machine, same worker count, same start method. The difference is entirely
in how each phase hands data to its workers:

  * the pairwise phase passes `counts` ONCE through `initargs` and then sends
    workers bare row indices (`pool.imap(_map_fun_parallel_pairs, range(n))`),
    so the parent does almost nothing per item and the workers saturate;

  * compute_hs instead pickles a DENSIFIED GENE VECTOR per gene through the
    pipe (`pool.imap(_map_fun_parallel, data_iter())`), and `data_iter` does
    `counts[i].toarray().ravel().astype('double')` serially in the parent.
    30,049 of those, one at a time, while eight workers idle.

Note this is NOT bandwidth: at that cell count the traffic is ~8 GB over 31
minutes, about 4.5 MB/s. It is the parent's serial densify-and-pickle loop, so
adding workers cannot help -- which is exactly what the 1.0x measured.

The patch below is the pairwise phase's design applied to compute_hs: `counts`
goes through `initargs` once, workers get `range(n)`, and the densification
happens inside the worker. Everything downstream -- the DataFrame assembly, the
Benjamini-Hochberg correction, the sort, the column selection -- is copied from
the original unchanged.

RESULTS ARE UNCHANGED, not approximately. `pool.imap` preserves input order, so
row i of the result frame is still gene i, and each row is still
`_compute_hs_inner` on the same dense vector with the same arguments. The
serial branch is kept too, and deliberately routed through the SAME worker
function, so jobs=1 and jobs=N differ only in who calls it.

THE COST IS MEMORY, and unlike the pairwise phase it is not bounded by
n_top_genes: `counts` here is the whole genes x cells sparse matrix, pickled per
worker under forkserver rather than shared copy-on-write. Roughly 1 GB per
worker on the 35k target and closer to 10 GB on the 343k one, so `jobs` needs
sizing against --mem on big targets rather than against the CPU count.


PATCH 2 -- seed the approximate kNN
-----------------------------------
`hotspot.knn.neighbors_and_weights` calls `NNDescent(coords, n_neighbors=...)`
with no `random_state`, and nothing upstream passes one, so pynndescent picks a
fresh seed per run. The neighbour graph therefore differs run to run, and with
it every Z score, every FDR call and every module.

That is not theoretical. Two runs of the same code on the same input on
2026-08-05 gave `24626/30049` and `24624/30049` genes at FDR < 0.05. Hotspot
output in this pipeline has never been reproducible, and this patch is what
makes the parallel-vs-serial comparison meaningful at all: without a fixed seed
the two runs differ for reasons that have nothing to do with parallelism.

The seed comes from `MarkerRun.random_state` (42 by default), which is the same
knob the rest of the marker stage already uses.

CONSEQUENCE WORTH KNOWING: hotspot marker files generated BEFORE this patch
(MtSZ_DRVI_10x_full_fixed, MtTip_DRVI_full_fixed and anything older) were
produced unseeded and cannot be reproduced exactly, by this code or any other.
Regenerate them if a run needs to be reproducible end to end.


VERSION SENSITIVITY
-------------------
Pinned to hotspot 1.1.3. Both patches rebind names in `hotspot.hotspot`, not in
the defining modules, because hotspot.py does `from .local_stats import
compute_hs` and `from .knn import neighbors_and_weights` -- patching the source
module alone would leave those bindings untouched. `apply()` verifies the
functions it is replacing still look the way it expects and raises rather than
silently no-opping.
"""

import logging
import multiprocessing

import pandas as pd
from scipy.sparse import issparse
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Set by the Pool initializer in each worker. A module-level dict rather than
# bare globals so the names cannot collide with hotspot's own g_* globals, which
# live in the same worker process once its initializer has run.
_STATE = {}

# Seed for patch 2, set by apply().
_SEED = None


def _init_worker(counts, neighbors, weights, num_umi, model, centered, Wtot2, D):
    """Pool initializer -- installs the read-only state every row needs."""
    _STATE['args'] = (counts, neighbors, weights, num_umi, model, centered, Wtot2, D)


def _hs_row(i):
    """One gene, by index. Densification happens HERE, in the worker."""
    from hotspot.local_stats import _compute_hs_inner

    counts, neighbors, weights, num_umi, model, centered, Wtot2, D = _STATE['args']
    vals = counts[i]
    if issparse(vals):
        vals = vals.toarray().ravel()
    vals = vals.astype('double')
    return _compute_hs_inner(vals, neighbors, weights, num_umi, model,
                             centered, Wtot2, D)


def _compute_hs_indexed(counts, neighbors, weights, num_umi, model, genes,
                        centered=False, jobs=1):
    """Drop-in for hotspot.local_stats.compute_hs. See PATCH 1 above."""
    from hotspot.knn import compute_node_degree

    neighbors = neighbors.values
    weights = weights.values
    num_umi = num_umi.values

    D = compute_node_degree(neighbors, weights)
    Wtot2 = (weights ** 2).sum()

    n_genes = counts.shape[0]
    init_args = (counts, neighbors, weights, num_umi, model, centered, Wtot2, D)

    if jobs > 1:
        with multiprocessing.Pool(processes=jobs, initializer=_init_worker,
                                  initargs=init_args) as pool:
            results = list(tqdm(pool.imap(_hs_row, range(n_genes)),
                                total=n_genes))
    else:
        # Same worker function, called in-process: jobs=1 and jobs=N must not
        # be two different implementations, or the comparison between them
        # stops testing what it is supposed to test.
        _init_worker(*init_args)
        results = list(tqdm(map(_hs_row, range(n_genes)), total=n_genes))

    # --- everything below is hotspot's own, unchanged -----------------------
    results = pd.DataFrame(results, index=genes,
                           columns=["G", "EG", "stdG", "Z", "C"])
    results["Pval"] = norm.sf(results["Z"].values)
    results["FDR"] = multipletests(results["Pval"], method="fdr_bh")[1]
    results = results.sort_values("Z", ascending=False)
    results.index.name = "Gene"
    return results[["C", "Z", "Pval", "FDR"]]


def _neighbors_and_weights_seeded(data, n_neighbors=30, neighborhood_factor=3,
                                  approx_neighbors=True):
    """Drop-in for hotspot.knn.neighbors_and_weights, seeded. See PATCH 2."""
    from pynndescent import NNDescent
    from sklearn.neighbors import NearestNeighbors
    from hotspot.knn import compute_weights

    coords = data.values

    if approx_neighbors:
        # The one changed line in this function: random_state.
        index = NNDescent(coords, n_neighbors=n_neighbors + 1,
                          random_state=_SEED)
        ind, dist = index.neighbor_graph
        ind, dist = ind[:, 1:], dist[:, 1:]
    else:
        # Exact neighbours are already deterministic; kept so the patch is a
        # true drop-in and `approx_neighbors=False` does not change meaning.
        nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                                algorithm="ball_tree").fit(coords)
        dist, ind = nbrs.kneighbors()

    weights = compute_weights(dist, neighborhood_factor=neighborhood_factor)

    ind = pd.DataFrame(ind, index=data.index)
    neighbors = ind
    weights = pd.DataFrame(weights, index=neighbors.index,
                           columns=neighbors.columns)

    return neighbors, weights


def apply(seed=42):
    """
    Install both patches. Idempotent, so calling it per hotspot run is fine.

    Raises if hotspot's internals are not the shape these patches expect --
    an upgrade that moves `_compute_hs_inner` or renames the bound functions
    should fail loudly here rather than produce numbers by some other route.
    """
    global _SEED
    _SEED = seed

    import hotspot.hotspot as hs_mod
    import hotspot.local_stats as ls_mod

    if not hasattr(ls_mod, '_compute_hs_inner'):
        raise RuntimeError(
            'hotspot.local_stats._compute_hs_inner is missing -- markerid/'
            'hotspot_patch.py is pinned to hotspot 1.1.3 internals and must be '
            'rechecked against the installed version before use.')

    for name, replacement in (('compute_hs', _compute_hs_indexed),
                              ('neighbors_and_weights', _neighbors_and_weights_seeded)):
        if not hasattr(hs_mod, name):
            raise RuntimeError(
                'hotspot.hotspot.{} is missing -- see the note above.'.format(name))
        if getattr(hs_mod, name) is not replacement:
            setattr(hs_mod, name, replacement)

    logger.info('[MARKERS] hotspot patched: indexed compute_hs, kNN seed=%s', seed)
