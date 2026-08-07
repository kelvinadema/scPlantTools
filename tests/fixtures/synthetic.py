"""
Synthetic AnnData builders for the QC tests.

WHY SYNTHETIC AND NOT A REAL SAMPLE. The claims worth testing are of the form
"this filter finds the compromised cells" and "this filter finds nothing when
there is nothing there". Neither is expressible against real data, because you
do not know which cells are compromised. Here you plant the signal and then
assert it is recovered -- and, just as importantly, assert that noise alone
recovers nothing.

Every builder takes an explicit `seed` and is deterministic.

WHAT IS DELIBERATELY NOT HERE. No `.X` count matrix beyond a trivial placeholder:
every filter under test reads `.obs` columns, not expression. Building a
realistic count matrix would add a large dependency on distributional
assumptions that the filters themselves are supposed to be robust to.
"""

from __future__ import annotations

import numpy as np


def _anndata(obs, seed=0):
    """Wrap an obs DataFrame in a minimal AnnData with a placeholder .X."""
    import anndata as ad
    import scipy.sparse as sp
    n = len(obs)
    rng = np.random.default_rng(seed)
    # 20 genes of low-rate counts. Placeholder: nothing under test reads it, but
    # anndata wants a matrix and some code paths check .n_vars.
    X = sp.csr_matrix(rng.poisson(0.5, size=(n, 20)).astype(np.float32))
    a = ad.AnnData(X=X, obs=obs)
    a.obs_names = ['cell{:05d}'.format(i) for i in range(n)]
    a.var_names = ['gene{:03d}'.format(i) for i in range(20)]
    return a


def clean_normal(n_cells=2000, seed=0):
    """
    A well-behaved unimodal sample. The null control.

    Any filter run over this should flag ~nothing. Paired with the `planted_*`
    builders: a filter that never flags passes this and fails those, and one
    that always flags does the reverse. Only a correct filter passes both.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    return _anndata(pd.DataFrame({
        'n_genes': rng.normal(2000, 300, n_cells).clip(50),
        'total_counts': rng.normal(8000, 1500, n_cells).clip(100),
        'mitochondrial_fraction': rng.beta(2, 100, n_cells),
    }), seed=seed)


def planted_outliers(n_cells=2000, n_planted=100, seed=0):
    """
    `clean_normal` plus `n_planted` cells pushed far into the upper tail.

    Returns (adata, planted_index). Assert **recall on planted_index**, not an
    exact flag count: recall survives a threshold retune, an exact count does not.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    a = clean_normal(n_cells, seed=seed)
    idx = rng.choice(n_cells, n_planted, replace=False)
    v = a.obs['n_genes'].to_numpy(copy=True)
    v[idx] = rng.normal(9000, 200, n_planted)          # ~23 sigma out
    a.obs['n_genes'] = v
    return a, np.sort(idx)


def planted_compromised(n_cells=2000, n_planted=300, seed=0, scale=1.0):
    """
    The miQC case: cells whose fraction metric rises as complexity falls.

    `scale` multiplies the fraction column. The filter standardises internally
    and converts back, so flags must be **identical** across scales -- that is
    the scale-invariance property, and it is what the old "0.0002 vs 0.03"
    note was really about.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    complexity = rng.normal(2000, 300, n_cells).clip(50)
    fraction = rng.beta(2, 100, n_cells)
    idx = rng.choice(n_cells, n_planted, replace=False)
    complexity[idx] = rng.normal(600, 100, n_planted).clip(50)   # low complexity
    fraction[idx] = rng.beta(20, 40, n_planted)                  # high fraction
    return _anndata(pd.DataFrame({
        'n_genes': complexity,
        'total_counts': complexity * 4,
        'mitochondrial_fraction': fraction * scale,
    }), seed=seed), np.sort(idx)


def low_incidence_noise(n_cells=2000, seed=0):
    """
    A low-incidence fraction with NO real structure: most cells near zero, a
    thin tail from pure count noise.

    This is the input a logit transform used to invent a "compromised" component
    out of. The property to assert is that the raw-scale fit finds no separation
    and declines -- not that some historical percentage is reproduced.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    return _anndata(pd.DataFrame({
        'n_genes': rng.normal(2000, 300, n_cells).clip(50),
        'total_counts': rng.normal(8000, 1500, n_cells).clip(100),
        'ribosomal_rna_fraction': rng.beta(0.5, 500, n_cells),
    }), seed=seed)


def degenerate(kind, n_cells=500, seed=0):
    """
    Inputs that violate the assumptions the outlier models rest on.

    `kind`:
      'majority_identical'  60% share one value -- MAD collapses to zero
      'constant'            every value identical
      'all_nan'             no finite values
      'single_cell'         n = 1
      'empty'               n = 0
      'zero_variance'       finite, but no spread
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    if kind == 'empty':
        v = np.array([], dtype=float)
    elif kind == 'single_cell':
        v = np.array([2000.0])
    elif kind == 'constant' or kind == 'zero_variance':
        v = np.full(n_cells, 2000.0)
    elif kind == 'all_nan':
        v = np.full(n_cells, np.nan)
    elif kind == 'majority_identical':
        v = rng.normal(2000, 300, n_cells)
        v[: int(n_cells * 0.6)] = 2000.0
    else:
        raise ValueError('unknown degenerate kind: {}'.format(kind))
    n = len(v)
    return _anndata(pd.DataFrame({
        'n_genes': v,
        'total_counts': v * 4 if n else v,
        'mitochondrial_fraction': np.full(n, 0.05),
    }), seed=seed)


def clustered_expression(n_cells=300, n_genes=200, n_clusters=3, n_markers=5,
                         seed=0, counts_layer='corrected_counts',
                         norm_layer='norm_log', latent_key='X_pca'):
    """
    An AnnData with planted cluster-specific marker genes.

    Returns (adata, markers) where `markers` maps cluster label -> the gene names
    planted as specific to it. Every marker method should rank its cluster's
    planted genes near the top; that is the only claim about marker methods that
    does not depend on the method's particular statistic.

    Built with the layers and obsm the markerID stage expects:
      layers[counts_layer]  raw-ish counts (what the cluster-free methods read)
      layers[norm_layer]    log1p-normalised (what scanpy ranks on)
      obsm[latent_key]      a latent space for the neighbour-graph methods
      obs['leiden']         the cluster labels
    """
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    per = n_cells // n_clusters
    labels, blocks, markers = [], [], {}

    for c in range(n_clusters):
        name = 'c{}'.format(c)
        labels += [name] * per
        X = rng.poisson(1.0, size=(per, n_genes)).astype(np.float32)
        # Planted markers: a disjoint block per cluster, strongly elevated.
        idx = list(range(c * n_markers, (c + 1) * n_markers))
        X[:, idx] += rng.poisson(60.0, size=(per, n_markers))
        markers[name] = ['gene{:03d}'.format(i) for i in idx]
        blocks.append(X)

    X = np.vstack(blocks)
    n = X.shape[0]
    a = ad.AnnData(X=sp.csr_matrix(X))
    a.obs_names = ['cell{:04d}'.format(i) for i in range(n)]
    a.var_names = ['gene{:03d}'.format(i) for i in range(n_genes)]
    a.obs['leiden'] = pd.Categorical(labels[:n])
    a.layers[counts_layer] = a.X.copy()

    norm = np.asarray(X, dtype=np.float32)
    totals = norm.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    a.layers[norm_layer] = sp.csr_matrix(np.log1p(norm / totals * 1e4))

    # A latent space that actually separates the clusters, so neighbour-graph
    # methods see the same structure the expression does.
    centres = rng.normal(0, 8, size=(n_clusters, 10))
    a.obsm[latent_key] = np.vstack([
        centres[c] + rng.normal(0, 1, size=(per, 10)) for c in range(n_clusters)])[:n]
    return a, markers
