#!/usr/bin/env python3
"""
Marker identification -- loading, preparation and dispatch.

This stage is standalone: it consumes an AnnData and writes a marker TSV, and
nothing here imports the annotation stage. The annotation pipeline consumes the
TSV (or calls `identify_markers` / the CLI to produce one), never the reverse.

Output contract, shared by every method and read by
`annotation.omg.OMGBuilder` via `markers.target` in
annotation_config.yaml:

    clusterName | rank | gene | score | method   (+ method-specific columns)

For wilcoxon / t-test / cosg, `clusterName` is a cell cluster. For hotspot /
lmd / cfmf it is a gene module -- those methods score genes without reference
to clusters, so their genesets are modules of co-localized genes.
"""

import logging

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from scipy import sparse

from .methods import CLUSTER_DEPENDENT, CLUSTER_FREE, METHODS

logger = logging.getLogger('MarkerLogging')

MARKER_COLUMNS = ('clusterName', 'rank', 'gene', 'score', 'method')
CLUSTER_COLUMN = 'clusterName'
GENE_COLUMN = 'gene'

# `.obs` column an external label table is written to (see `--labels-tsv`).
# A fixed name rather than the TSV's own column name so a run can be told
# apart from one on a pipeline cluster key by looking at the log alone.
EXTERNAL_LABEL_KEY = 'external_label'
DEFAULT_LABELS_EXCLUDE = ('Unassigned',)


class MarkerRun:
    """One sample, one method."""

    def __init__(self, adata_path, sample, method, cluster_key=None, latent_key=None,
                 counts_layer='corrected_counts', out_dir=None, norm_layer='norm_log',
                 max_pct=0.85, min_cells=10, random_state=42, n_jobs=4, adata=None,
                 labels_tsv=None, labels_col='label', labels_cell_col='cell',
                 labels_exclude=DEFAULT_LABELS_EXCLUDE, min_cells_per_group=0):
        """
        `labels_tsv` (added 2026-09-13, chapter 3 C5): a TSV with one row per
        cell (`labels_cell_col`, either the bare `.obs_names` or
        `<sample>.<barcode>`) and a label column (`labels_col`). Cells absent
        from the table or carrying a label in `labels_exclude` are DROPPED
        before gene filtering, the rest get the label in
        `.obs[EXTERNAL_LABEL_KEY]`, and that key becomes the cluster key for
        the cluster-dependent methods. This is how a marker table is computed on
        a declared, reviewed label set (a refine output, a chapter's label
        file) instead of the Leiden partition an h5ad happens to carry.
        `min_cells_per_group` (any cluster key) drops groups smaller than that
        before ranking -- scanpy will rank a 3-cell group and it means nothing.
        """
        if method not in METHODS:
            raise ValueError(
                f'Unknown method {method!r}. Expected one of {sorted(METHODS)}')

        self.method = method
        self.sample = sample
        self.cluster_key = cluster_key
        self.labels_tsv = Path(labels_tsv) if labels_tsv else None
        self.labels_col = labels_col
        self.labels_cell_col = labels_cell_col
        self.labels_exclude = tuple(labels_exclude or ())
        self.min_cells_per_group = int(min_cells_per_group or 0)
        if self.labels_tsv is not None:
            if cluster_key and cluster_key != EXTERNAL_LABEL_KEY:
                logger.warning('[MARKERS] --labels-tsv given; ignoring --clusterkey %r '
                               'in favour of %r', cluster_key, EXTERNAL_LABEL_KEY)
            self.cluster_key = EXTERNAL_LABEL_KEY
        self.latent_key = latent_key
        self.counts_layer = counts_layer
        self.norm_layer = norm_layer
        self.max_pct = max_pct
        self.min_cells = min_cells
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.out_dir = Path(out_dir) if out_dir else Path('results/markers') / method
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.outfn = f'{sample}.tsv'

        self.adata = adata if adata is not None else sc.read_h5ad(Path(adata_path))
        self.module_scores = None
        self.n_scales = None

    # ------------------------------------------------------------- preparation

    def prepare(self):
        """Load the object and check the keys this method needs are present."""
        self._coerce_numeric_obs()
        self._attach_external_labels()
        self._drop_small_groups()
        self._filter_and_normalize()
        self._resolve_latent()
        return self

    # ------------------------------------------------------ external labels

    def _attach_external_labels(self):
        """
        Join `labels_tsv` onto `.obs` as `EXTERNAL_LABEL_KEY`, dropping the
        cells it does not label. Idempotent: main.py hands the prepared object
        to the next method, and the join is recorded in `.uns['marker_labels']`.
        """
        if self.labels_tsv is None:
            return
        stamp = [str(self.labels_tsv), self.labels_col, list(self.labels_exclude)]
        if self.adata.uns.get('marker_labels') == stamp and \
                EXTERNAL_LABEL_KEY in self.adata.obs:
            logger.info('[MARKERS] Reusing external labels %r', EXTERNAL_LABEL_KEY)
            return
        if not self.labels_tsv.exists():
            raise FileNotFoundError(f'{self.sample}: labels TSV {self.labels_tsv} not found')
        table = pd.read_csv(self.labels_tsv, sep='\t', dtype=str)
        for col in (self.labels_cell_col, self.labels_col):
            if col not in table.columns:
                raise KeyError(f'{self.sample}: {self.labels_tsv} has no column {col!r} '
                               f'(columns: {list(table.columns)})')
        table = table.dropna(subset=[self.labels_cell_col, self.labels_col])
        labels = table.drop_duplicates(self.labels_cell_col).set_index(self.labels_cell_col)[self.labels_col]

        hit = _match_cells(self.adata, labels.index)
        matched = pd.Series(hit, index=self.adata.obs_names)
        found = matched.notna()
        values = labels.reindex(matched[found].to_numpy()).to_numpy()
        keep = found.to_numpy().copy()
        keep[found.to_numpy()] = ~pd.Series(values).isin(self.labels_exclude).to_numpy()
        n_absent = int((~found).sum())
        n_excluded = int(found.sum() - keep.sum())
        if not keep.any():
            raise ValueError(
                f'{self.sample}: no cell of {self.adata.n_obs} carries a usable label from '
                f'{self.labels_tsv} ({n_absent} absent from the table, {n_excluded} excluded). '
                'Check the cell-id convention (bare barcode vs <sample>.<barcode>).')
        logger.info('[MARKERS] External labels from %s: %d/%d cells labelled, %d absent, '
                    '%d excluded (%s)', self.labels_tsv.name, int(keep.sum()), self.adata.n_obs,
                    n_absent, n_excluded, ', '.join(self.labels_exclude) or '-')
        self.adata = self.adata[keep].copy()
        self.adata.obs[EXTERNAL_LABEL_KEY] = pd.Categorical(
            labels.reindex(matched[keep].to_numpy()).astype(str).to_numpy())
        self.adata.uns['marker_labels'] = stamp
        self.adata.uns.pop('marker_prep', None)     # cells changed: re-filter genes

    def _drop_small_groups(self):
        """Drop cells of groups with fewer than `min_cells_per_group` cells."""
        if not self.min_cells_per_group or not self.cluster_key \
                or self.cluster_key not in self.adata.obs:
            return
        counts = self.adata.obs[self.cluster_key].astype(str).value_counts()
        small = counts[counts < self.min_cells_per_group]
        if small.empty:
            return
        logger.warning('[MARKERS] %s: dropping %d group(s) with < %d cells before ranking: %s',
                       self.sample, len(small), self.min_cells_per_group,
                       ', '.join(f'{k} ({v})' for k, v in small.items()))
        keep = ~self.adata.obs[self.cluster_key].astype(str).isin(small.index).to_numpy()
        self.adata = self.adata[keep].copy()
        if isinstance(self.adata.obs[self.cluster_key].dtype, pd.CategoricalDtype):
            self.adata.obs[self.cluster_key] = \
                self.adata.obs[self.cluster_key].cat.remove_unused_categories()
        self.adata.uns.pop('marker_prep', None)

    def _coerce_numeric_obs(self):
        """
        Restore numeric dtypes to `.obs` columns stored as categorical.

        The integration stage currently writes every continuous QC metric as a
        string categorical -- `cell_complexity_corrected_counts` arrives with
        one category per cell. Hotspot dies on it ('Categorical' object has no
        attribute 'sum') and any covariate use is silently wrong, so numeric-
        looking categoricals are converted back on load.
        """
        converted = []
        for column in self.adata.obs.columns:
            dtype = self.adata.obs[column].dtype
            if not isinstance(dtype, pd.CategoricalDtype):
                continue
            values = pd.to_numeric(self.adata.obs[column].astype(str), errors='coerce')
            # Genuine categoricals (cluster labels, batch) must not be touched;
            # a numeric column stored this way has many distinct values.
            if values.notna().all() and len(dtype.categories) > 50:
                self.adata.obs[column] = values.astype(float)
                converted.append(column)
        if converted:
            logger.warning(
                '[MARKERS] Converted %d categorical .obs columns to numeric '
                '(integration wrote them as strings): %s%s',
                len(converted), ', '.join(converted[:3]),
                ' ...' if len(converted) > 3 else '')

    def _filter_and_normalize(self):
        """Drop uninformative genes, then build the normalised/log layer."""
        # main.py hands the prepared object to the next method so several
        # methods share one load; preparation is deterministic, so redoing it
        # would only re-normalise an already-normalised layer.
        prepared = self.adata.uns.get('marker_prep')
        if prepared == [self.counts_layer, self.norm_layer, self.min_cells,
                        self.max_pct]:
            logger.info('[MARKERS] Reusing prepared layer %r', self.norm_layer)
            return

        if self.counts_layer != 'X' and self.counts_layer not in self.adata.layers:
            raise KeyError(
                f'{self.sample}: layer {self.counts_layer!r} not found '
                f'({list(self.adata.layers)})')
        counts = self.adata.X if self.counts_layer == 'X' else \
            self.adata.layers[self.counts_layer]

        n_expressing = np.asarray((counts > 0).sum(axis=0)).ravel()
        detected = n_expressing >= self.min_cells
        ubiquitous = (n_expressing / self.adata.n_obs) > self.max_pct
        keep = detected & ~ubiquitous
        if not keep.any():
            raise ValueError(f'{self.sample}: no gene survived filtering')
        logger.info('[MARKERS] %d/%d genes retained (%d below min_cells=%d, '
                    '%d ubiquitous >%.0f%%)', int(keep.sum()), len(keep),
                    int((~detected).sum()), self.min_cells,
                    int((detected & ubiquitous).sum()), 100 * self.max_pct)

        self.adata = self.adata[:, keep].copy()
        source = self.adata.X if self.counts_layer == 'X' else \
            self.adata.layers[self.counts_layer]
        self.adata.layers[self.norm_layer] = source.copy()
        sc.pp.normalize_total(self.adata, target_sum=1e4, layer=self.norm_layer)
        # log1p stamps .uns['log1p']; clear it so a later re-prepare of a fresh
        # object does not warn about an already-transformed .X it never touched.
        self.adata.uns.pop('log1p', None)
        sc.pp.log1p(self.adata, layer=self.norm_layer)
        self.adata.uns['marker_prep'] = [self.counts_layer, self.norm_layer,
                                         self.min_cells, self.max_pct]

    def _resolve_latent(self):
        """
        Validate the latent space, or compute PCA on the normalised layer.

        Never PCA on raw counts: that embedding is dominated by library size and
        it drives the kNN graph every cluster-free method is built on.
        """
        if self.latent_key and self.latent_key not in self.adata.obsm:
            raise KeyError(
                f'{self.sample}: latent key {self.latent_key!r} not in .obsm '
                f'({list(self.adata.obsm)})')
        if not self.latent_key:
            logger.info('[MARKERS] No latent key given; computing PCA on %r',
                        self.norm_layer)
            sc.pp.pca(self.adata, layer=self.norm_layer, random_state=self.random_state)
            self.latent_key = 'X_pca'

        X = np.asarray(self.adata.obsm[self.latent_key], dtype=float)
        n_bad = int((~np.isfinite(X)).any(axis=1).sum())
        if n_bad:
            # Reference-mapped embeddings can carry NaN rows for query cells the
            # model never saw; sklearn's own error names neither key nor count.
            raise ValueError(
                f'{self.sample}: {n_bad}/{len(X)} cells '
                f'({100 * n_bad / len(X):.1f}%) have non-finite values in '
                f'.obsm[{self.latent_key!r}]. Choose a complete embedding, or '
                f'subset to the cells that have one.')

    # ---------------------------------------------------------------- accessors

    def latent(self):
        """The latent representation this method ranks against.

        Returns:
            np.ndarray: Cells x latent dimensions.
        """
        return np.asarray(self.adata.obsm[self.latent_key], dtype=np.float64)

    def expression(self, genes_by_cells=False):
        """Normalised expression, cells x genes by default."""
        X = self.adata.layers[self.norm_layer]
        return X.T.tocsr() if genes_by_cells and sparse.issparse(X) else (
            X.T if genes_by_cells else X)

    def expression_subset(self, genes):
        """The expression matrix restricted to `genes`.

        Args:
            genes (list[str]): Gene names.

        Returns:
            The subset matrix.
        """
        idx = self.adata.var_names.get_indexer(np.asarray(genes))
        return self.adata.layers[self.norm_layer][:, idx[idx >= 0]]

    def require_cluster_key(self):
        """The cluster column, raising if this method needs one and it is absent.

        Returns:
            str: The `.obs` column name.
        """
        if not self.cluster_key:
            raise ValueError(
                f'{self.sample}: method {self.method!r} is cluster-dependent and '
                f'needs --clusterkey (available: '
                f'{[c for c in self.adata.obs.columns if "leiden" in c.lower()]})')
        if self.cluster_key not in self.adata.obs:
            raise KeyError(
                f'{self.sample}: cluster key {self.cluster_key!r} not in .obs')
        return self.cluster_key

    def counts_layer_for_hotspot(self):
        """Hotspot models raw counts, not the normalised layer."""
        return None if self.counts_layer == 'X' else self.counts_layer

    def umi_counts_key(self):
        """The layer holding UMI counts for this run.

        Returns:
            str: Layer name.
        """
        key = f'{self.counts_layer}_total_counts'
        if key in self.adata.obs and pd.api.types.is_numeric_dtype(self.adata.obs[key]):
            return key
        return None

    # --------------------------------------------------------------------- run

    def run(self, n_genes=50, write=True, **kwargs):
        """Rank genes with the configured method and optionally write the TSV.

        Args:
            n_genes (int): Genes to keep per group.
            write (bool): Whether to write the output TSV.

        Returns:
            pandas.DataFrame: The ranking.
        """
        logger.info('[MARKERS] %s: running %s', self.sample, self.method)
        df = METHODS[self.method](self, n_genes=n_genes, **kwargs)
        df = self._finalize(df)
        if write:
            self.write(df)
        return df

    def _finalize(self, df):
        """Sort, trim and standardize the ranking's columns.

        Args:
            df (pandas.DataFrame): The raw ranking.

        Returns:
            pandas.DataFrame: The finalized ranking.
        """
        if df is None or df.empty:
            logger.warning('[MARKERS] %s: %s produced no markers',
                           self.sample, self.method)
            return pd.DataFrame(columns=list(MARKER_COLUMNS))
        missing = [c for c in MARKER_COLUMNS if c not in df.columns]
        if missing:
            raise RuntimeError(
                f'{self.method} returned a frame missing {missing}; the output '
                f'contract is {list(MARKER_COLUMNS)}')
        df[CLUSTER_COLUMN] = df[CLUSTER_COLUMN].astype(str)
        logger.info('[MARKERS] %s: %d rows across %d genesets', self.sample,
                    len(df), df[CLUSTER_COLUMN].nunique())
        return df

    def write(self, df):
        """Write the ranking to this run's TSV.

        Args:
            df (pandas.DataFrame): The finalized ranking.

        Returns:
            Path: The written file.
        """
        outpath = self.out_dir / self.outfn
        df.to_csv(outpath, sep='\t', index=False)
        logger.info('[MARKERS] Wrote %s', outpath)
        if self.module_scores is not None:
            score_path = self.out_dir / f'{self.sample}_module_scores.tsv'
            self.module_scores.to_csv(score_path, sep='\t')
            logger.info('[MARKERS] Wrote per-cell module scores to %s', score_path)
        return outpath


def _match_cells(adata, table_ids):
    """
    For every cell of `adata`, the id it has in `table_ids` (or None).

    Three conventions are tried, in order: the `.obs_names` as they are; the
    table ids with a `<sample>.` prefix stripped (table prefixed, object
    bare); the `.obs_names` prefixed with `.obs['sample']` (object bare, table
    prefixed). The first convention matching at least half the cells wins.
    """
    ids = pd.Index(pd.Series(table_ids).astype(str))
    names = pd.Index(adata.obs_names.astype(str))
    direct = names.isin(ids)
    if direct.mean() >= 0.5 or 'sample' not in adata.obs:
        return np.where(direct, names, None)
    prefixed = pd.Index(adata.obs['sample'].astype(str) + '.' + names)
    hit = prefixed.isin(ids)
    if hit.mean() >= 0.5:
        return np.where(hit, prefixed, None)
    return np.where(direct, names, None)


_RUN_KWARGS = ('norm_layer', 'max_pct', 'min_cells', 'random_state', 'n_jobs',
               'labels_tsv', 'labels_col', 'labels_cell_col', 'labels_exclude',
               'min_cells_per_group')


def identify_markers(adata_path, sample, method='wilcoxon', cluster_key=None,
                     latent_key=None, counts_layer='corrected_counts', out_dir=None,
                     n_genes=50, write=True, adata=None, **kwargs):
    """
    Public entrypoint, also used by the annotation stage.

    Returns the marker DataFrame; with `write=True` (default) it is also written
    to `{out_dir}/{sample}.tsv`, which is the file the annotation stage reads.
    """
    run = MarkerRun(adata_path=adata_path, sample=sample, method=method,
                    cluster_key=cluster_key, latent_key=latent_key,
                    counts_layer=counts_layer, out_dir=out_dir, adata=adata,
                    **{k: v for k, v in kwargs.items() if k in _RUN_KWARGS})
    run.prepare()
    method_kwargs = {k: v for k, v in kwargs.items() if k not in _RUN_KWARGS}
    return run.run(n_genes=n_genes, write=write, **method_kwargs)


def marker_file_path(out_dir, sample):
    """Where `identify_markers` writes, for callers that only need the path."""
    return Path(out_dir) / f'{sample}.tsv'


__all__ = ['MarkerRun', 'identify_markers', 'marker_file_path',
           'MARKER_COLUMNS', 'EXTERNAL_LABEL_KEY', 'CLUSTER_DEPENDENT',
           'CLUSTER_FREE', 'METHODS']
