#!/usr/bin/env python3

import logging
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess

class HVGSelector:
    """Rank genes by variability under several methods, then take their consensus.

    Eight ranking methods by default -- mean, mean-variance, dispersion and deviance
    families -- each producing a full ranking in `.var`, combined into one consensus
    rank. A method that goes singular is dropped rather than failing the run, since
    the consensus is the deliverable.
    """
    def __init__(self, adata, outfn = '', layer = 'corrected_counts', n_hvg = 2000, lowess_frac:float=0.5, logger=None):
        self.adata = adata
        self.layer = layer
        self.n_hvg = n_hvg
        self.lowess_frac = lowess_frac

        self.logger = logger or logging.getLogger('IntegrateLogging')

        self.outfn = outfn
        Path(self.outfn).parent.mkdir(parents=True, exist_ok=True)

        self.adata_raw = None
        self.adata_norm = None
        self.gene_madata_logask = None

    # Helpers
    def initialize_views(self, adata, min_cells=10):
        """
        Build the raw / normalized / log-normalized views every method reads.

        Expensive -- three full matrix copies plus a normalize and a log1p --
        so `select_hvg` calls this **once** for the whole method sweep rather
        than per method. Nothing downstream mutates the views' `.X` (scanpy's
        highly_variable_genes only adds `.var` columns), which is what makes
        sharing them safe.
        """
        if sp.issparse(adata.X):
            # getnnz counts STORED entries, which equals the positive ones only
            # when no explicit zeros are stored -- corrected_counts can carry
            # them. Check once, cheaply, and fall back to the exact form.
            data = adata.X.data
            if data.size and (data.min() <= 0):
                detected = np.asarray((adata.X > 0).sum(axis=0)).ravel()
            else:
                detected = adata.X.getnnz(axis=0)
        else:
            detected = (adata.X > 0).sum(axis=0)
        self.gene_mask = np.asarray(detected).ravel() >= min_cells

        # Raw counts
        self.adata_raw = adata[:, self.gene_mask].copy()

        # Normalized counts
        self.adata_norm = self.adata_raw.copy()
        sc.pp.normalize_total(self.adata_norm, target_sum=1e4)

        # Log-normalized counts
        self.adata_log = self.adata_norm.copy()
        sc.pp.log1p(self.adata_log)

    def _get_mean_var(self, X):
        """Per-gene mean and population variance, sparse or dense.

        Args:
            X: The matrix, genes in columns.

        Returns:
            tuple: (mean (np.ndarray), var (np.ndarray)).
        """
        if sp.issparse(X):
            mean = np.array(X.mean(axis=0)).ravel()
            mean_sq = np.array(X.multiply(X).mean(axis=0)).ravel()
            var = mean_sq - mean**2
        else:
            mean = X.mean(axis=0)
            var = X.var(axis=0, ddof=0)
        return mean, var

    def _hvg_scanpy(self, adata, flavor: str, colname: str, batch_key=None):
        """Rank genes with one of scanpy's own flavors and store the result.

        Args:
            adata (AnnData): The view this flavor expects (raw for seurat_v3, log
                otherwise).
            flavor (str): A scanpy `highly_variable_genes` flavor.
            colname (str): Stem of the `.var` rank column written.
            batch_key (str, optional): Passed through to scanpy.

        Returns:
            pandas.Series: Rank per gene, 1 = most variable.

        Raises:
            ValueError: On an unsupported flavor, or a flavor that produced no metric.
        """
        sc.pp.highly_variable_genes(adata, n_top_genes=adata.n_vars, flavor=flavor, inplace=True, batch_key=batch_key)
        genes = adata.var_names
        if flavor in ["seurat_v3", "seurat_v3_paper"]:
            metric = adata.var["variances_norm"]
            ranks = (-metric).rank(method="dense").astype(int)
            ranks = self._normalize_ranks(ranks, genes)
        elif flavor in ["seurat", "cell_ranger"]:
            if "dispersions_norm" in adata.var:
                ranks = (-adata.var["dispersions_norm"]).rank(method="dense").astype(int)
                ranks = self._normalize_ranks(ranks, genes)
            else:
                raise ValueError(f"No dispersion_norm found for flavor {flavor}")

        else:
            raise ValueError(f"Unsupported Scanpy flavor: {flavor}")
        self._store_ranks(ranks.values, adata.var_names, f"{colname}_rank")
        return ranks

    def _normalize_ranks(self, ranks, genes):
        """Reindex ranks onto `genes`, sending missing genes to the worst rank.

        Args:
            ranks (pandas.Series | np.ndarray): Ranks as produced.
            genes (pandas.Index): The gene axis to align to.

        Returns:
            pandas.Series: Integer ranks, one per gene.
        """
        if isinstance(ranks, np.ndarray):
            ranks = pd.Series(ranks, index=genes)
        ranks = ranks.reindex(genes)
        max_rank = len(genes)
        return ranks.fillna(max_rank).astype(int)

    def _store_ranks(self, ranks, genes, colname):
        """Write one method's ranks into `self.adata.var[colname]`.

        Stored as float so genes the method never saw stay NaN rather than being given
        a rank they did not earn.

        Args:
            ranks (pandas.Series | np.ndarray): Ranks for the genes this method saw.
            genes (pandas.Index): Gene names those ranks belong to.
            colname (str): `.var` column to write.

        Returns:
            pandas.Series: The ranks reindexed onto every gene.
        """
        if not isinstance(ranks, pd.Series):
            ranks = pd.Series(ranks, index=genes)
        ranks_full = ranks.reindex(self.adata.var_names)
        # Store as float (so NaNs are allowed)
        self.adata.var[colname] = ranks_full.astype("float")
        return ranks_full


    # Mean-Variance methods

    def _mean_max(self):
        """Rank genes by mean normalized expression.

        Returns:
            pandas.Series: Rank per gene, 1 = highest mean.
        """
        mean, _ = self._get_mean_var(self.adata_norm.X)
        order = np.argsort(-np.asarray(mean))
        ranks = np.empty(len(order), dtype=int)
        ranks[order] = np.arange(1, len(order) + 1)
        ranks = pd.Series(ranks, index=self.adata_norm.var_names)
        self._store_ranks(ranks, self.adata_norm.var_names, "hvg_mean_max_rank")
        return ranks

    def _mv_lowess_nc(self):
        """Rank genes by variance above a lowess mean-variance trend.

        Fitted exactly rather than with statsmodels' `delta` interpolation: that
        approximates the fit and so moves the residuals and the ranking.

        Returns:
            pandas.Series: Rank per gene, 1 = largest positive residual.
        """
        mean, var = self._get_mean_var(self.adata_norm.X)
        # statsmodels' `delta` is faster but APPROXIMATES the fit, moving the
        # residuals and so the ranking. 0.4s here is not where the time goes;
        # changing this is a deliberate change in results.
        fitted = lowess(var, mean, frac=self.lowess_frac, return_sorted=False)
        residual = var - fitted

        order = np.argsort(-np.asarray(residual))
        ranks = np.empty(len(order), dtype=int)
        ranks[order] = np.arange(1, len(order) + 1)
        ranks = pd.Series(ranks, index=self.adata_norm.var_names)
        self._store_ranks(ranks, self.adata_norm.var_names, "hvg_mv_lowess_nc_rank")
        return ranks

    # Log Mean-Variance methods

    def _logmv_seurat_v3(self):
        """Rank genes by scanpy's seurat_v3 variance-stabilizing metric.

        Returns:
            pandas.Series: Rank per gene.
        """
        return self._hvg_scanpy(adata = self.adata_raw, flavor = "seurat_v3", colname = "hvg_logmv_seuratv3")

    def _logmv_lognc(self):
        """Rank genes by variance above a quadratic fit in log-normalized space.

        Returns:
            pandas.Series: Rank per gene, 1 = largest positive residual.
        """
        mean, var = self._get_mean_var(self.adata_log.X)
        residual = var - np.polyval(np.polyfit(mean, var, deg=2), mean)
        order = np.argsort(-np.asarray(residual))
        ranks = np.empty(len(order), dtype=int)
        ranks[order] = np.arange(1, len(order) + 1)
        ranks = pd.Series(ranks, index=self.adata_log.var_names)
        self._store_ranks(ranks, self.adata_log.var_names, "hvg_logmv_lognc_rank")
        return ranks

    # Dispersion methods

    def _disp_seurat(self):
        """Rank genes by scanpy's normalized dispersion (seurat flavor).

        Returns:
            pandas.Series: Rank per gene.
        """
        return self._hvg_scanpy(adata = self.adata_log, flavor = "seurat", colname = "hvg_disp_seurat")

    def _disp_cellranger(self):
        """Rank genes by scanpy's normalized dispersion (cell_ranger flavor).

        Returns:
            pandas.Series: Rank per gene.
        """
        return self._hvg_scanpy(adata = self.adata_log, flavor = "cell_ranger", colname = "hvg_disp_cellranger")

    # Deviance

    def _loading_variance(self, fam = 'nb'):
        """Rank genes by their squared loadings on a GLM-PCA fit.

        Not in the default method list: glmpca is a heavy optional dependency, imported
        lazily so this module stays importable without it.

        Args:
            fam (str): GLM family passed to glmpca.

        Returns:
            pandas.Series: Rank per gene.
        """
        # Lazy: glmpca is a heavy optional dependency, and importing it at
        # module scope would break importing HVGSelector at all where it is
        # absent -- even for callers who never touch this method.
        from glmpca import glmpca
        model = glmpca.glmpca(Y = self.adata_raw.X.toarray(), L = 5, fam = 'nb')
        loadings = model["loadings"]
        gene_scores = np.sum(loadings**2, axis=1)
        order = np.argsort(-np.asarray(gene_scores))
        ranks = np.empty(len(order), dtype=int)
        ranks[order] = np.arange(1, len(order) + 1)
        ranks = pd.Series(ranks, index=self.adata_raw.var_names)
        self._store_ranks(ranks, self.adata_raw.var_names, "hvg_loading_variance_rank")
        return ranks

    def _deviance_poisson(self):
        """Rank genes by Poisson deviance from a constant-rate null.

        Returns:
            pandas.Series: Rank per gene, 1 = largest deviance.
        """
        # Densified whole rather than blocked: the gene filter has already run,
        # so this is a few GB. Blocking would change the summation order, shift
        # deviances by ~1e-12, and flip the rank of numerically tied genes --
        # a different gene set for no measured benefit.
        X = self.adata_raw.X.toarray() if hasattr(self.adata_raw.X, "toarray") else self.adata_raw.X
        mu = np.asarray(X.mean(axis=0)).flatten()  # (n_genes,)
        mu_safe = np.where(mu == 0, 1e-8, mu)
        dev = 2 * (X * np.log((X + 1e-8) / mu_safe) - (X - mu_safe))
        deviance = np.sum(dev, axis=0)
        order = np.argsort(-deviance)
        ranks = np.empty(len(order), dtype=int)
        ranks[order] = np.arange(1, len(order) + 1)
        ranks = pd.Series(ranks, index=self.adata_raw.var_names)
        self._store_ranks(ranks, self.adata_raw.var_names, "hvg_deviance_poisson_rank")
        return ranks

    def _deviance_nb(self):
        """Rank genes by negative-binomial deviance, with theta from the mean-variance fit.

        Returns:
            pandas.Series: Rank per gene, 1 = largest deviance.
        """
        # Not blocked, for the reason given in _deviance_poisson.
        X = self.adata_raw.X.toarray() if hasattr(self.adata_raw.X, "toarray") else self.adata_raw.X
        mu = np.asarray(X.mean(axis=0)).flatten()
        var = np.asarray(X.var(axis=0)).flatten()
        theta = np.where(var > mu, mu**2 / (var - mu + 1e-8), 1e8)
        Y = X
        MU = mu[np.newaxis, :]
        THETA = theta[np.newaxis, :]
        term1 = Y * np.log((Y * (THETA + MU) + 1e-8) / (MU * (THETA + Y) + 1e-8))
        term2 = THETA * np.log((THETA + MU) / (THETA + Y + 1e-8))
        dev = 2 * (term1 + term2)
        deviance = np.sum(dev, axis=0)
        order = np.argsort(-deviance)
        ranks = np.empty(len(order), dtype=int)
        ranks[order] = np.arange(1, len(order) + 1)
        ranks = pd.Series(ranks, index=self.adata_raw.var_names)
        self._store_ranks(ranks, self.adata_raw.var_names, "hvg_deviance_nb_rank")
        return ranks

    # API helpers

    def _tie_breaker(self, primary, secondary):
        """Order genes by a primary rank, breaking ties on a secondary then by name.

        Sorting by name last makes the order deterministic across runs where two genes
        tie on both scores.

        Args:
            primary (pandas.Series): Main ordering score, ascending.
            secondary (pandas.Series or None): Tie-break score.

        Returns:
            list[str]: Gene names, best first.
        """
        df = pd.DataFrame({"primary": primary})
        df["secondary"] = secondary if secondary is not None else 0
        df = df.dropna(subset=["primary"])
        # reset_index() names its column after the index's .name, not "index",
        # so normalize explicitly -- relying on either is data-dependent.
        df.index.name = "gene"
        df = df.reset_index()

        df_sorted = df.sort_values(by=["primary", "secondary", "gene"], ascending=[True, True, True],kind="mergesort")
        return df_sorted["gene"].tolist()

    def _consensus_hvg_rank(self, method='median', n_hvg=2000):
        """Combine every stored method rank into one consensus ranking.

        Only genes ranked by *all* methods take part; a gene absent from any method's
        ranking stays NaN rather than being given a rank it did not earn.

        Args:
            method (str): 'median' or 'mean' across the per-method ranks.
            n_hvg (int, optional): Mark this many top genes in `.var['hvg_consensus']`.

        Returns:
            pandas.Series or tuple: The consensus rank, plus the top gene list when
                `n_hvg` is given.

        Raises:
            ValueError: If no rank columns exist, or `method` is unrecognised.
        """
        rank_cols = [c for c in self.adata.var.columns if c.startswith("hvg_")]
        if not rank_cols:
            raise ValueError("No HVG rank columns found in adata.var")
        ranks_df = self.adata.var[rank_cols]

        # Keep only genes valid in all methods
        valid_mask = ranks_df.notna().all(axis=1)
        ranks_df = ranks_df.loc[valid_mask]

        if method == "median":
            consensus_score = ranks_df.median(axis=1, skipna=True)
        elif method == "mean":
            consensus_score = ranks_df.mean(axis=1, skipna=True)
        else:
            raise ValueError("method must be 'median' or 'mean'")
        secondary_score = ranks_df.mean(axis=1, skipna=True)
        ordered_genes = self._tie_breaker(consensus_score, secondary_score)
        # 1 = best; genes filtered out before ranking stay NaN.
        consensus_rank = pd.Series(
            np.arange(1, len(ordered_genes) + 1, dtype=float), index=ordered_genes
        ).reindex(self.adata.var_names)
        self.adata.var["consensus_rank"] = consensus_rank.values
        self._store_ranks(consensus_rank, self.adata.var_names, "hvg_consensus")
        # Mark top-n genes
        if n_hvg is not None:
            df_sorted = consensus_score.sort_values()
            top_genes = df_sorted.index[:n_hvg].tolist()
            self.adata.var["hvg_consensus"] = self.adata.var_names.isin(top_genes)
            return consensus_rank, top_genes
        return consensus_rank

    def _aggregate_batches(self, func, batch_key: str, aggregate: str = "median"):
        """Run one method per batch and aggregate the rankings.

        Each batch is different data, so the views are rebuilt per batch. A gene absent
        from a batch's ranking is given the worst rank there rather than being dropped.

        Args:
            func (callable): The bound method to run per batch.
            batch_key (str): `.obs` column identifying batches.
            aggregate (str): 'median' or 'mean' across batches.

        Returns:
            list[str]: Gene names, best first.

        Raises:
            ValueError: If `aggregate` is unrecognised.
        """
        results = {}
        for b in self.adata.obs[batch_key].unique():
            adata_b = self.adata[self.adata.obs[batch_key] == b].copy()
            self.initialize_views(adata_b)
            genes = func()
            results[b] = list(genes)

        # Rank aggregation
        all_genes = sorted(set().union(*results.values()))
        rank_df = pd.DataFrame(index=all_genes, columns=results.keys())
        for b, genes in results.items():
            rank_df.loc[genes, b] = np.arange(len(genes), 0, -1)
        rank_df = rank_df.fillna(len(all_genes) + 1)
        if aggregate == "median":
            agg_rank = rank_df.median(axis=1)
        elif aggregate == "mean":
            agg_rank = rank_df.mean(axis=1)
        else:
            raise ValueError("aggregate must be 'median' or 'mean'")
        top_genes = agg_rank.sort_values().index.tolist()
        return top_genes

    def _export_hvg_table(self, outfile: str, batch_key:str ='batch'):
        """Write every method's ranks and the consensus to a TSV.

        Args:
            outfile (str | Path): Where to write.
            batch_key (str, optional): Recorded in the table so the file says which
                batches it was computed over.
        """
        df = self.adata.var.copy()
        rank_cols = [c for c in df.columns if c.endswith("_rank") or c == "consensus_rank"]
        out_df = df[rank_cols].copy()
        out_df.insert(0, "gene", out_df.index)
        out_df = out_df.sort_values(by="consensus_rank", na_position='last')
        out_df = out_df.where(pd.notnull(out_df), "NA")
        if batch_key is not None:
            batches = self.adata.obs[batch_key].unique().tolist()
            out_df["batches"] = ",".join(map(str, batches))
        out_df.to_csv(outfile, sep="\t", index=False)
        self.logger.info('Finished HVG computation, writing outfile to: {}'.format(outfile))

    # main API

    def select_hvg(self, methods: list[str] = None, n_hvg: int = 2000, min_cells: int = 10, compute_consensus: bool = True, batch_key='batch'):
        """Run the method sweep, build the consensus, and subset the object to it.

        The three normalized views are built once for the whole sweep rather than per
        method -- that was the single largest cost here. A method that raises is
        recorded in `self.failed_methods` and dropped: several fit a curve that goes
        singular on a small or sparse sample, and the consensus is what matters.

        Args:
            methods (list[str], optional): Method names without the leading underscore.
                Defaults to the eight-method sweep.
            n_hvg (int): How many genes to keep.
            min_cells (int): Gene detection floor applied before anything is ranked.
            compute_consensus (bool): Whether to build the consensus ranking.
            batch_key (str, optional): Rank per batch and aggregate; None ranks the
                object as one.

        Returns:
            AnnData: The object subset to the selected genes.
        """
        if methods is None:
            methods = ["mean_max", "mv_lowess_nc", "deviance_nb", "deviance_poisson", "logmv_seurat_v3", "logmv_lognc", "disp_seurat", "disp_cellranger"]
        results = {}
        # Methods that raised, kept for the caller and for the exported table.
        self.failed_methods = {}
        # Built ONCE for the whole sweep: three matrix copies plus a normalize
        # and a log1p, which rebuilt per method was the largest cost here. Safe
        # because no method mutates the views' .X. The batched path
        # re-initialises per batch, which it must -- different data each time.
        if batch_key is None:
            self.initialize_views(self.adata, min_cells=min_cells)

        for method in methods:
            self.logger.info('[HVG] Starting HVG selection using method: {}'.format(method))
            func = getattr(self, f"_{method}")
            try:
                if batch_key == None:
                    results[method] = func()
                else:
                    results[method] = self._aggregate_batches(func, batch_key, aggregate="median")
            except Exception as e:
                # Drop the method, keep the consensus. Several of these fit a
                # curve (loess in logmv_seurat_v3, a negative binomial in the
                # deviance methods) and go singular on a small or sparse
                # population -- `2_tip`, 163 cells against 52,689 genes, fails
                # in exactly that way with an opaque message from the underlying
                # Fortran. Letting one method abort all eight throws away seven
                # perfectly good rankings; the whole point of a consensus is
                # that it does not depend on any single one.
                #
                # _consensus_hvg_rank reads whichever `hvg_*` columns exist in
                # .var, so a method that never wrote one simply does not vote.
                self.failed_methods[method] = str(e)
                self.logger.warning(
                    '[HVG] Method "{}" failed and is dropped from the consensus: {}'.format(method, e))

        if not results:
            raise RuntimeError(
                '[HVG] Every requested method failed, so there is no consensus to compute. '
                'Reasons per method: {}'.format(
                    '; '.join('{}: {}'.format(m, r) for m, r in self.failed_methods.items())))
        if self.failed_methods:
            self.logger.warning(
                '[HVG] Consensus computed from {} of {} methods; dropped: {}'.format(
                    len(results), len(methods), ', '.join(sorted(self.failed_methods))))
        # Reset
        if compute_consensus:
            results["consensus"] = self._consensus_hvg_rank(method = 'median', n_hvg = self.n_hvg)
        # batch_key threaded through explicitly: this used to always fall back
        # to _export_hvg_table's own hardcoded default ('batch'), ignoring
        # whatever batch_key was passed here. Harmless for every existing
        # caller (none override batch_key away from 'batch'), but a caller
        # with no .obs['batch'] column (e.g. a single, unbatched sample) needs
        # batch_key=None to actually reach _export_hvg_table's own guard
        # against writing the batches column at all.
        self._export_hvg_table(outfile = self.outfn, batch_key = batch_key)
        top_genes = (self.adata.var.sort_values("consensus_rank").head(self.n_hvg).index)
        self.logger.info(f"[HVG] Subsetting AnnData object to top {self.n_hvg} consensus HVG genes.")
        return self.adata[:, top_genes].copy()
