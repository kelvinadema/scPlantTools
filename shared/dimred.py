#!/usr/bin/env python3

"""
Shared dimensionality-reduction/clustering (neighbors -> leiden -> UMAP) and UMAP
plotting, factored out so mapping QC, integration, and (structurally) annotation
share one correct implementation instead of three independent ones.

Root cause this replaces: `mapping/qc_plot.py`'s old `_plot_umap` computed
a PCA embedding and ran the standalone `umap-learn` package on it directly, then
*separately* ran `sc.pp.neighbors`/`sc.tl.leiden` on an unrelated representation
purely to generate cluster-color labels for the already-fixed scatter -- so
leiden's neighbor graph had nothing to do with what UMAP actually embedded.

Here, exactly one `sc.pp.neighbors(key_added=...)` call builds the graph, and
both `sc.tl.umap` and `sc.tl.leiden` are pointed at that same `neighbors_key`,
so the embedding and the clustering are always computed from the same graph.
"""

import logging

import numpy as np
import pandas as pd
import scanpy as sc

from pathlib import Path
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from plotnine import *

from shared.hvg import HVGSelector


def _mds_placeholder(distances, n_components=10, random_state=42):
    """
    Cheap classical-MDS embedding straight from a precomputed distance matrix.

    Written into `adata.obsm['X_pca']` purely so `sc.tl.umap` doesn't fall back
    to running a full PCA on `.X`: `sc.tl.umap` always resolves *some*
    representation via scanpy's `_choose_representation` to check the neighbor
    graph's connected components for its initialisation strategy (see
    `scanpy/tools/_utils.py::_get_pca_or_small_x`) -- when `sc.pp.neighbors`
    was given `distances=` directly, its recorded params carry no `use_rep`,
    so with no cached `X_pca` present, scanpy silently runs `sc.pp.pca` on the
    full (possibly huge) gene expression matrix just to get one. This is much
    cheaper: an MDS embedding of the n x n distance matrix already in hand,
    not a fresh PCA of n x n_genes -- and it uses the `X_pca` key name only
    because that's the key scanpy's own fallback-avoidance checks for, not
    because this is an actual PCA of gene expression; confined to the
    throwaway per-call AnnData copies the `distances` mode operates on.
    """
    n = distances.shape[0]
    k = min(n_components, n - 1)
    D2 = np.asarray(distances) ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    return TruncatedSVD(n_components=k, random_state=random_state).fit_transform(B)


def load_cached_hvg(staged, hvg_outfn, n_hvg, logger):
    """
    Reuse a consensus-HVG file from a previous run instead of recomputing.

    HVGSelector runs eight ranking methods and aggregates them by median
    rank, which is the slowest step in this path by a wide margin. The result
    is a deterministic function of the staged matrix, so recomputing it on a
    re-run buys nothing.

    Returns the gene-subset AnnData, or None when there is no usable cache
    and the caller should compute one. Deliberately conservative: any
    mismatch (missing file, unreadable, no recognisable gene column, too few
    of the listed genes still present) falls through to a fresh computation
    rather than proceeding on a stale or partial list.

    The file is keyed by path, and every caller passes a path that already
    encodes which population it describes, so two different cell subsets
    cannot collide on one cache entry.
    """
    if not hvg_outfn:
        return None
    path = Path(hvg_outfn)
    if not path.exists():
        return None
    try:
        table = pd.read_csv(path, sep='\t')
    except Exception as e:
        logger.warning('[DimReducer] Could not read cached HVG file {}: {} -- recomputing.'.format(path, e))
        return None

    column = next((c for c in ('gene', 'gene_name', 'genes', 'feature', 'index')
                   if c in table.columns), table.columns[0] if len(table.columns) else None)
    if column is None:
        logger.warning('[DimReducer] Cached HVG file {} has no usable gene column -- recomputing.'.format(path))
        return None

    # Build the lookup once. Inlining `set(staged.var_names)` into the
    # comprehension rebuilt it for every candidate gene -- 52,689 genes against
    # a 52,689-element set is ~2.8e9 operations, which turned a cache *hit* into
    # 13 minutes, far slower than the ~7s selection it was meant to avoid.
    known = set(staged.var_names)
    genes = [g for g in table[column].astype(str).tolist() if g in known]
    if len(genes) < min(n_hvg, staged.n_vars) * 0.5:
        logger.warning(
            '[DimReducer] Cached HVG file {} lists {} genes present in this object, far fewer '
            'than the {} requested -- it likely describes a different gene set. '
            'Recomputing.'.format(path, len(genes), n_hvg))
        return None

    genes = genes[:n_hvg]
    logger.info('[DimReducer] Reusing {} consensus HVGs from {} (delete it to force a '
                     'fresh selection).'.format(len(genes), path))
    return staged[:, genes].copy()


def _choose_n_pcs_by_variance(X, variance_cutoff=0.9):
    """
    Pick the smallest number of PCs whose cumulative explained variance ratio
    clears `variance_cutoff`, rather than an arbitrary fixed component count.

    Dispatches on sparsity/size: a small-enough sparse matrix is densified for
    exact `PCA`; a large sparse matrix uses `TruncatedSVD` (which never needs
    (and cannot efficiently take) a dense copy); a dense matrix always uses
    `PCA` directly. Every branch must assign `model` -- a prior version of
    this had no `else` for the plain-dense case, leaving `model` unassigned
    and raising `UnboundLocalError` at `model.fit(X)` below; dense input is
    latent-but-real here (not every caller's input layer is guaranteed sparse).
    """
    if sparse.issparse(X):
        if min(X.shape) < 500:
            X = X.toarray()
            model = PCA(n_components=min(X.shape) - 1)
        else:
            model = TruncatedSVD(n_components=min(X.shape) - 1)
    else:
        model = PCA(n_components=min(X.shape) - 1)
    model.fit(X)
    cumulative = np.cumsum(model.explained_variance_ratio_)
    n_pcs = np.searchsorted(cumulative, variance_cutoff) + 1
    return n_pcs, model


class DimReducer:
    """
    Computation only -- no plotting. Mutates `adata` in place: writes the
    neighbor graph, the UMAP embedding, and (optionally) leiden cluster labels,
    all under one shared key suffix so they can always be found together.
    """

    def __init__(self, adata, logger=None):
        self.adata = adata
        self.logger = logger or logging.getLogger('SingleCellMappingPipeline')

    def _resolve_n_neighbors(self, n_neighbors, n_obs):
        """Clamp `n_neighbors` to something the cell count can support.

        Args:
            n_neighbors (int or None): Requested value.
            n_obs (int): Cells available.

        Returns:
            int: The value actually used.
        """
        if n_neighbors is not None:
            return n_neighbors
        raw = int(round(np.sqrt(n_obs)))
        clipped = max(2, min(raw, n_obs - 1))
        if clipped != raw:
            self.logger.warning(
                '[DimReducer] Clipped n_neighbors from {} to {} ({} cells in this view)'.format(
                    raw, clipped, n_obs))
        return clipped

    def _hvg_pca_fallback(self, adata, layer, n_hvg, variance_cutoff, do_hvg, hvg_outfn):
        """
        No ready-made representation exists (mapping QC's normal case): select
        HVGs, then PCA on the HVG-subset object, writing the result into the
        real `adata.obsm['X_pca']` so `sc.pp.neighbors(use_rep='X_pca')` and any
        other downstream scanpy call finds it the standard way -- the old
        `_prepare_umap_input` only ever returned a bare, unattached array.

        HVG selection goes through `HVGSelector` (`shared/hvg.py`,
        the same consensus-rank selector integration already uses), not
        scanpy's `highly_variable_genes` -- per explicit instruction, replacing
        the earlier single-flavor approach.
        """
        staged = adata.copy()
        # HVGSelector's own `layer` constructor kwarg is accepted but never
        # actually read by any of its methods (verified by reading the class) --
        # every method operates on `.X` directly. Stage it here rather than
        # relying on that kwarg to do anything.
        staged.X = staged.layers[layer].copy()

        if do_hvg:
            hvg_adata = load_cached_hvg(staged, hvg_outfn, n_hvg, self.logger)
            if hvg_adata is None:
                selector = HVGSelector(staged, outfn=str(hvg_outfn), n_hvg=n_hvg, logger=self.logger)
                # The constructor supplies the default feature count. batch_key=None: this
                # fallback only ever runs on a single sample's own AnnData, which
                # has no multi-sample `.obs['batch']` concept -- batch_key='batch'
                # (the class default) would raise a KeyError.
                hvg_adata = selector.select_hvg(batch_key=None)
        else:
            hvg_adata = staged

        n_pcs, _ = _choose_n_pcs_by_variance(hvg_adata.X, variance_cutoff)
        sc.tl.pca(hvg_adata, n_comps=n_pcs)
        # Same cells/rows as `adata` (HVG selection subsets genes, not cells),
        # so this lines up by position without any reindexing.
        adata.obsm['X_pca'] = hvg_adata.obsm['X_pca']
        # Returned so compute() can record how many PCs the embedding used in
        # the params dict the report later renders.
        return n_pcs

    def compute(self, use_rep=None, layer=None, distances=None, mask=None, key=None,
                n_neighbors=None, metric='correlation', do_leiden=True, leiden_resolution=0.5,
                min_dist=0.0, random_state=42, n_hvg=2000, variance_cutoff=0.90, do_hvg=True,
                hvg_outfn=None, neighbors_kwargs=None, leiden_kwargs=None, umap_kwargs=None):
        """
        Resolve a representation, build one neighbor graph from it, then embed
        (`sc.tl.umap`) and, unless `do_leiden=False`, cluster (`sc.tl.leiden`)
        from that exact same graph.

        Representation is resolved in this order:
          1. `distances` (a precomputed full n x n pairwise distance matrix --
             DoD's case): `key` is required here (no representation name to
             derive a default suffix from, and DoD uses two different distance
             matrices in one invocation, so there is no safe shared default).
             `metric` is recorded as provenance only in this mode -- scanpy
             does not use it for computation when distances are precomputed.
          2. `use_rep`, if given and already present in `adata.obsm` (e.g.
             `X_scVI`, `X_scArches`, `X_pca`, or any future key -- this is what
             makes the module usable from annotation later without changes).
          3. `layer`, if no usable `use_rep` -- the HVG+PCA fallback above.

        Returns a dict: {'adata', 'use_rep', 'neighbors_key', 'umap_key',
        'leiden_key' (None if do_leiden=False), 'params'}. `adata` is always
        included (identical to `self.adata` unless `distances` + `mask` together
        required subsetting) so the caller never has to guess which object was
        actually mutated. `params` records the resolved neighbours/UMAP/Leiden
        settings actually used (n_neighbors after the sqrt(N) resolution,
        metric, min_dist, leiden_resolution, and -- on the HVG+PCA path -- n_pcs
        and n_hvg), so a caller can persist them for the report without having to
        re-derive what the defaults resolved to.
        """
        adata = self.adata

        if distances is not None:
            if key is None:
                raise ValueError(
                    "[DimReducer] `key` is required when `distances` is given -- there is no "
                    "representation name to derive a default suffix from.")
            if mask is not None:
                keep = ~np.asarray(mask, dtype=bool)
                idx = np.where(keep)[0]
                distances = distances[np.ix_(idx, idx)]
                adata = adata[keep].copy()
                self.adata = adata
            n_neighbors = self._resolve_n_neighbors(n_neighbors, adata.n_obs)
            neighbors_key = f"neighbors_{key}"
            neigh_kwargs = dict(neighbors_kwargs or {})
            # random_state is deliberately not passed here: scanpy ignores (and
            # warns about) it when `distances` is precomputed, since there is
            # no stochastic neighbor search left for a seed to control.
            neigh_kwargs.update(dict(n_neighbors=n_neighbors, distances=distances,
                                      key_added=neighbors_key))
            if metric is not None:
                neigh_kwargs.setdefault('metric', metric)
            sc.pp.neighbors(adata, **neigh_kwargs)
            # See _mds_placeholder's docstring: avoids sc.tl.umap silently
            # falling back to a full PCA of .X below.
            adata.obsm['X_pca'] = _mds_placeholder(distances)
            resolved_rep = None

        elif use_rep is not None and use_rep in adata.obsm:
            if key is None:
                key = use_rep[2:] if use_rep.startswith('X_') else use_rep
            n_neighbors = self._resolve_n_neighbors(n_neighbors, adata.n_obs)
            neighbors_key = f"neighbors_{key}"
            neigh_kwargs = dict(neighbors_kwargs or {})
            neigh_kwargs.update(dict(n_neighbors=n_neighbors, use_rep=use_rep, metric=metric,
                                      random_state=random_state, key_added=neighbors_key))
            sc.pp.neighbors(adata, **neigh_kwargs)
            resolved_rep = use_rep

        elif layer is not None:
            if key is None:
                key = 'pca'
            outfn = hvg_outfn or (Path.cwd() / 'hvg_consensus.tsv')
            n_pcs = self._hvg_pca_fallback(adata, layer=layer, n_hvg=n_hvg,
                                    variance_cutoff=variance_cutoff, do_hvg=do_hvg, hvg_outfn=outfn)
            # The recursion below does the actual neighbours/UMAP/Leiden on the
            # X_pca it just wrote, and returns the resolved params -- augment
            # them here with the HVG+PCA details only this outer call knows.
            result = self.compute(use_rep='X_pca', key=key, n_neighbors=n_neighbors, metric=metric,
                                 do_leiden=do_leiden, leiden_resolution=leiden_resolution,
                                 min_dist=min_dist, random_state=random_state,
                                 neighbors_kwargs=neighbors_kwargs, leiden_kwargs=leiden_kwargs,
                                 umap_kwargs=umap_kwargs)
            result['params'].update({'from_layer': layer, 'n_pcs': int(n_pcs), 'do_hvg': bool(do_hvg)})
            if do_hvg:
                result['params']['n_hvg'] = int(n_hvg)
            return result
        else:
            raise ValueError(
                "[DimReducer] compute() needs exactly one of: `distances`, `use_rep` (present "
                "in adata.obsm), or `layer`.")

        umap_key = f"X_umap_{key}"
        leiden_key = f"leiden_{key}" if do_leiden else None

        umap_call_kwargs = dict(umap_kwargs or {})
        umap_call_kwargs.update(dict(min_dist=min_dist, random_state=random_state,
                                      neighbors_key=neighbors_key, key_added=umap_key))
        sc.tl.umap(adata, **umap_call_kwargs)

        if do_leiden:
            leiden_call_kwargs = dict(leiden_kwargs or {})
            leiden_call_kwargs.update(dict(resolution=leiden_resolution, random_state=random_state,
                                            neighbors_key=neighbors_key, key_added=leiden_key))
            sc.tl.leiden(adata, **leiden_call_kwargs)

        # The settings this call actually resolved to, for a caller to persist so
        # the report can describe the embedding. All values are plain str/int/float
        # (no None -- anndata drops None-valued .uns keys) so they round-trip
        # through .uns and json.dumps unchanged.
        params = {
            'representation': (resolved_rep if resolved_rep is not None
                               else ('precomputed distances' if distances is not None else '')),
            'n_neighbors': int(n_neighbors),
            'metric': metric if metric is not None else '',
            'min_dist': float(min_dist),
            'random_state': int(random_state) if random_state is not None else '',
        }
        if do_leiden:
            params['leiden_resolution'] = float(leiden_resolution)

        return {'adata': adata, 'use_rep': resolved_rep, 'neighbors_key': neighbors_key,
                'umap_key': umap_key, 'leiden_key': leiden_key, 'params': params}


class UMAPPlotter:
    """
    Plotting only -- no computation. Always plots leiden-cluster coloring
    first (when a `leiden_key` is given), then one scatter per requested
    additional `.obs` column (batch, doublet score, mito fraction, ...), then
    an ad-hoc color vector not necessarily backed by an `.obs` column (DoD's
    boolean flag-array case).
    """

    def __init__(self, adata, save_dir, logger=None):
        self.adata = adata
        self.save_dir = Path(save_dir)
        self.logger = logger or logging.getLogger('SingleCellMappingPipeline')

    def _save_plot(self, plot, name, prefix="plot", dpi=600, width=6, height=4):
        """Write a figure to `{save_dir}/{prefix}_{name}.png`.

        Args:
            plot (plotnine.ggplot): The figure.
            name (str): Filename stem after the prefix.
            prefix (str): Filename prefix.
            dpi (int): Output resolution.
            width (float): Inches.
            height (float): Inches.
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)
        filename = self.save_dir / f"{prefix}_umap_{name}.png"
        plot.save(filename=str(filename), dpi=dpi, width=width, height=height)

    def _color_scale(self, series):
        """
        Numeric -> min-max scaled + a light-to-red gradient; exactly-2-level
        categorical -> a fixed black/red discrete scale; anything else -> the
        default discrete palette. Dedupes the branching that used to be
        implemented separately in `qc_plot.py::_plot_umap` and
        `shared/plot.py::scPlotter.plot_umap`.
        """
        if pd.api.types.is_numeric_dtype(series):
            scaled = MinMaxScaler().fit_transform(series.to_numpy().reshape(-1, 1)).ravel()
            return scaled, scale_color_gradient(low="lightgray", high="red")
        if series.nunique() == 2:
            return series, scale_color_manual(values=["black", "red"])
        return series, None

    def _scatter(self, base_df, color_values, title, color_scale):
        """One UMAP scatter, coloured by the given values.

        Args:
            base_df (pandas.DataFrame): Frame carrying the two UMAP axes.
            color_values (pandas.Series): Per-cell colour values.
            title (str): Figure title.
            color_scale: A plotnine scale, or None to let plotnine choose.

        Returns:
            plotnine.ggplot: The figure.
        """
        df = base_df.copy()
        df["color"] = color_values
        p = (ggplot(df, aes("UMAP1", "UMAP2", color="color"))
            + geom_point(size=self._size, alpha=self._alpha)
            + theme_bw()
            + ggtitle(title))
        if color_scale is not None:
            p += color_scale
        return p

    def plot(self, umap_key, leiden_key, extra_color_by=None, color=None,
             title_prefix="UMAP", prefix="plot", save=True, size=1.2, alpha=0.6):
        """Draw the UMAP coloured by cluster and by any extra columns asked for.

        Args:
            umap_key (str): `.obsm` key holding the 2-D embedding.
            leiden_key (str): `.obs` column holding cluster labels.
            extra_color_by (list[str], optional): Further `.obs` columns to colour by.
            color (str, optional): A single column to colour by instead.
            title_prefix (str): Prefix for each figure title.
            prefix (str): Filename prefix.
            save (bool): Whether to write the PNGs.
            size (float): Point size.
            alpha (float): Point alpha.

        Returns:
            list[plotnine.ggplot]: The figures, in the order drawn.
        """
        self._size, self._alpha = size, alpha
        adata = self.adata
        embedding = adata.obsm[umap_key]
        base_df = pd.DataFrame({"UMAP1": embedding[:, 0], "UMAP2": embedding[:, 1]},
                                index=adata.obs_names)

        color_cols = []
        if leiden_key is not None:
            color_cols.append(leiden_key)
        if extra_color_by is not None:
            color_cols.extend([extra_color_by] if isinstance(extra_color_by, str) else extra_color_by)

        plots = []
        for col in color_cols:
            if col not in adata.obs.columns:
                self.logger.warning(f"[UMAPPlotter] Column '{col}' not found in adata.obs -- skipping.")
                continue
            color_values, color_scale = self._color_scale(adata.obs[col])
            p = self._scatter(base_df, color_values, f"{title_prefix} - {col}", color_scale)
            if save:
                self._save_plot(p, name=f"{col}_{umap_key}", prefix=prefix)
            plots.append(p)

        if color is not None:
            series = pd.Series(np.asarray(color), index=adata.obs_names)
            color_values, color_scale = self._color_scale(series)
            p = self._scatter(base_df, color_values, title_prefix, color_scale)
            if save:
                self._save_plot(p, name=umap_key, prefix=prefix)
            plots.append(p)

        if not plots:
            # No leiden, no extras, no ad-hoc color -- still produce a plain
            # unfilled scatter rather than silently returning nothing.
            df = base_df.copy()
            p = ggplot(df, aes("UMAP1", "UMAP2")) + geom_point(color="black", size=size, alpha=alpha) \
                + theme_bw() + ggtitle(title_prefix)
            if save:
                self._save_plot(p, name=umap_key, prefix=prefix)
            plots.append(p)

        return plots
