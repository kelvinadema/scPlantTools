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
        self.adata_log = None

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
        if self.adata_raw.n_vars == 0:
            # Scanpy's normalization/log transform rejects a zero-column matrix.
            # The batch aggregator still needs this batch's missing-rank penalty.
            self.adata_norm = self.adata_raw.copy()
            self.adata_log = self.adata_raw.copy()
            return

        # Normalized counts
        self.adata_norm = self.adata_raw.copy()
        sc.pp.normalize_total(self.adata_norm, target_sum=1e4)

        # Log-normalized counts
        self.adata_log = self.adata_norm.copy()
        sc.pp.log1p(self.adata_log)

    def _get_mean_var(self, X):
        """Per-gene (mean, population variance), sparse or dense; genes in columns."""
        if sp.issparse(X):
            mean = np.array(X.mean(axis=0)).ravel()
            mean_sq = np.array(X.multiply(X).mean(axis=0)).ravel()
            var = mean_sq - mean**2
        else:
            mean = X.mean(axis=0)
            var = X.var(axis=0, ddof=0)
        return mean, var

    def _hvg_scanpy(self, adata, flavor: str, batch_key=None):
        """Gene-indexed ranks (1 = most variable) from one of scanpy's own flavors.

        `adata` is the view the flavor expects (raw for seurat_v3, log otherwise).
        Raises ValueError on an unsupported flavor or one that produced no metric.
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
        return ranks

    def _normalize_ranks(self, ranks, genes):
        """Reindex ranks onto `genes` as integers, sending missing genes to the worst rank."""
        if isinstance(ranks, np.ndarray):
            ranks = pd.Series(ranks, index=genes)
        ranks = ranks.reindex(genes)
        max_rank = len(genes)
        return ranks.fillna(max_rank).astype(int)

    def _store_ranks(self, ranks, genes, colname):
        """Write one method's ranks into `self.adata.var[colname]`.

        Stored as float so genes the method never saw stay NaN rather than being given
        a rank they did not earn. Returns the ranks reindexed onto every gene.
        """
        if not isinstance(ranks, pd.Series):
            ranks = pd.Series(ranks, index=genes)
        ranks_full = ranks.reindex(self.adata.var_names)
        # Store as float (so NaNs are allowed)
        self.adata.var[colname] = ranks_full.astype("float")
        return ranks_full


    # Mean-Variance methods

    def _mean_max(self):
        """Rank genes by mean normalized expression (1 = highest mean)."""
        mean, _ = self._get_mean_var(self.adata_norm.X)
        order = np.argsort(-np.asarray(mean))
        ranks = np.empty(len(order), dtype=int)
        ranks[order] = np.arange(1, len(order) + 1)
        ranks = pd.Series(ranks, index=self.adata_norm.var_names)
        return ranks

    def _mv_lowess_nc(self):
        """Rank genes by variance above a lowess mean-variance trend (1 = largest residual)."""
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
        return ranks

    # Log Mean-Variance methods

    def _logmv_seurat_v3(self):
        """Rank genes by scanpy's seurat_v3 variance-stabilizing metric."""
        return self._hvg_scanpy(adata = self.adata_raw, flavor = "seurat_v3")

    def _logmv_lognc(self):
        """Rank genes by variance above a quadratic fit in log-normalized space (1 = largest residual)."""
        mean, var = self._get_mean_var(self.adata_log.X)
        residual = var - np.polyval(np.polyfit(mean, var, deg=2), mean)
        order = np.argsort(-np.asarray(residual))
        ranks = np.empty(len(order), dtype=int)
        ranks[order] = np.arange(1, len(order) + 1)
        ranks = pd.Series(ranks, index=self.adata_log.var_names)
        return ranks

    # Dispersion methods

    def _disp_seurat(self):
        """Rank genes by scanpy's normalized dispersion (seurat flavor)."""
        return self._hvg_scanpy(adata = self.adata_log, flavor = "seurat")

    def _disp_cellranger(self):
        """Rank genes by scanpy's normalized dispersion (cell_ranger flavor)."""
        return self._hvg_scanpy(adata = self.adata_log, flavor = "cell_ranger")

    # Deviance

    def _loading_variance(self, fam = 'nb'):
        """Rank genes by their squared loadings on a GLM-PCA fit.

        Not in the default method list. glmpca is a heavy optional dependency,
        imported lazily so this module stays importable without it.
        """
        from glmpca import glmpca
        model = glmpca.glmpca(Y = self.adata_raw.X.toarray(), L = 5, fam = 'nb')
        loadings = model["loadings"]
        gene_scores = np.sum(loadings**2, axis=1)
        order = np.argsort(-np.asarray(gene_scores))
        ranks = np.empty(len(order), dtype=int)
        ranks[order] = np.arange(1, len(order) + 1)
        ranks = pd.Series(ranks, index=self.adata_raw.var_names)
        return ranks

    def _deviance_poisson(self):
        """Rank genes by Poisson deviance from a constant-rate null (1 = largest deviance)."""
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
        return ranks

    def _deviance_nb(self):
        """Rank genes by negative-binomial deviance, theta from the mean-variance fit (1 = largest)."""
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
        return ranks

    # API helpers

    def _tie_breaker(self, primary, secondary):
        """Order genes by a primary rank, breaking ties on a secondary then by name.

        Sorting by name last makes the order deterministic across runs where two genes
        tie on both scores. Returns gene names, best first.
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
        ranking stays NaN rather than being given a rank it did not earn. With
        `n_hvg`, the top genes are marked in `.var['hvg_consensus']` and returned
        as (consensus rank, top genes); otherwise only the rank is returned.
        Raises ValueError if no rank columns exist or `method` is unrecognised.
        """
        rank_cols = [c for c in self.adata.var.columns
                     if c.startswith("hvg_") and c.endswith("_rank")]
        if not rank_cols:
            raise ValueError("No HVG rank columns found in adata.var")
        ranks_df = self.adata.var[rank_cols]

        # Keep only genes valid in all methods
        valid_mask = ranks_df.notna().all(axis=1)
        ranks_df = ranks_df.loc[valid_mask]
        if ranks_df.empty:
            raise ValueError("No genes have valid ranks across the successful HVG methods")

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
            top_genes = ordered_genes[:n_hvg]
            self.adata.var["hvg_consensus"] = self.adata.var_names.isin(top_genes)
            return consensus_rank, top_genes
        return consensus_rank

    def _aggregate_batches(self, func, batch_key: str, aggregate: str = "median", min_cells: int = 10):
        """Run one method per batch and aggregate the rankings.

        Retain the gene-indexed numerical ranks (1 = best), not their values as
        gene identifiers. Missing genes receive G + 1, where G is the number of
        genes ranked in at least one batch. Genes absent from every batch stay
        unranked. Fractional median ranks are preserved for the method consensus.
        `min_cells` is applied separately in every batch.
        """
        results = {}
        for b in self.adata.obs[batch_key].unique():
            adata_b = self.adata[self.adata.obs[batch_key] == b].copy()
            self.initialize_views(adata_b, min_cells=min_cells)
            # An empty batch gene set contributes only missing-rank penalties;
            # it must not cause an otherwise usable method to fail.
            results[b] = func() if self.adata_raw.n_vars else pd.Series(dtype=float)

        # Rank aggregation
        rank_df = pd.DataFrame(results).dropna(how="all")
        rank_df = rank_df.fillna(len(rank_df) + 1)
        if aggregate == "median":
            agg_rank = rank_df.median(axis=1)
        elif aggregate == "mean":
            agg_rank = rank_df.mean(axis=1)
        else:
            raise ValueError("aggregate must be 'median' or 'mean'")
        return agg_rank

    def _export_hvg_table(self, outfile: str, batch_key:str ='batch'):
        """Write every method's ranks and the consensus to a TSV, with the batches
        it was computed over when `batch_key` is set."""
        df = self.adata.var.copy()
        rank_cols = [c for c in df.columns if c.endswith("_rank") or c == "consensus_rank"]
        out_df = df[rank_cols].copy()
        out_df.insert(0, "gene", out_df.index)
        if "consensus_rank" in out_df:
            out_df = out_df.sort_values(by="consensus_rank", na_position='last')
        out_df = out_df.where(pd.notnull(out_df), "NA")
        if batch_key is not None:
            batches = self.adata.obs[batch_key].unique().tolist()
            out_df["batches"] = ",".join(map(str, batches))
        out_df.to_csv(outfile, sep="\t", index=False)
        self.logger.info('Finished HVG computation, writing outfile to: {}'.format(outfile))

    # main API

    def select_hvg(self, methods: list[str] = None, n_hvg: int = None, min_cells: int = 10, compute_consensus: bool = True, batch_key='batch'):
        """Run the method sweep, build the consensus, and subset the object to it.

        `methods` are names without the leading underscore (default: the eight-method
        sweep). A method that raises is recorded in `self.failed_methods` and
        dropped from the consensus. Unranked genes never fill unused `n_hvg` slots.
        `batch_key` ranks per batch and aggregates; None ranks the object as one.
        Returns the selected genes, or the full ranked object when
        `compute_consensus` is false.
        """
        n_hvg = self.n_hvg if n_hvg is None else n_hvg
        if not isinstance(n_hvg, (int, np.integer)) or n_hvg < 1:
            raise ValueError("n_hvg must be a positive integer")
        if not isinstance(min_cells, (int, np.integer)) or min_cells < 1:
            raise ValueError("min_cells must be a positive integer")
        if batch_key is not None:
            if batch_key not in self.adata.obs:
                raise KeyError(f"HVG batch key {batch_key!r} is missing from adata.obs")
            if self.adata.obs[batch_key].isna().any():
                raise ValueError(f"HVG batch key {batch_key!r} contains missing values")
        # Previous selections and failed/omitted methods must not cast votes.
        stale = [c for c in self.adata.var.columns
                 if (c.startswith("hvg_") and c.endswith("_rank"))
                 or c in ("consensus_rank", "hvg_consensus")]
        self.adata.var.drop(columns=stale, inplace=True)
        if methods is None:
            methods = ["mean_max", "mv_lowess_nc", "deviance_nb", "deviance_poisson", "logmv_seurat_v3", "logmv_lognc", "disp_seurat", "disp_cellranger"]
        results = {}
        # Methods that raised, kept for the caller and logged.
        self.failed_methods = {}
        # Built once for the whole sweep (see initialize_views). The batched path
        # re-initialises per batch, which it must -- different data each time.
        if batch_key is None:
            self.initialize_views(self.adata, min_cells=min_cells)

        for method in methods:
            self.logger.info('[HVG] Starting HVG selection using method: {}'.format(method))
            func = getattr(self, f"_{method}")
            try:
                if batch_key is None:
                    ranks = func()
                else:
                    ranks = self._aggregate_batches(func, batch_key, aggregate="median", min_cells=min_cells)
                if ranks.empty:
                    raise ValueError("No genes pass the per-batch detection floor")
                # Publish only after every batch succeeds, so a partial method
                # cannot leave behind the ranks of its last successful batch.
                stem = "logmv_seuratv3" if method == "logmv_seurat_v3" else method
                self._store_ranks(ranks, ranks.index, f"hvg_{stem}_rank")
                results[method] = ranks
            except Exception as e:
                # Drop the method, keep the consensus. Several of these fit a
                # curve (loess in logmv_seurat_v3, a negative binomial in the
                # deviance methods) that goes singular on a small or sparse
                # population (see docs/decisions/doublet-detection.md, "HVG
                # selection"); one failure must not throw away the other rankings.
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
        if compute_consensus:
            _, top_genes = self._consensus_hvg_rank(method='median', n_hvg=n_hvg)
        self.adata.uns['hvg_selection'] = {
            'algorithm': 'median_ranks_across_batches_then_methods_v2',
            'batch_key': batch_key or '',
            'min_cells_per_batch': int(min_cells),
            'n_hvg_requested': int(n_hvg),
            'successful_methods': list(results),
            'failed_methods': dict(self.failed_methods),
            'missing_batch_rank': 'number_of_genes_ranked_in_any_batch_plus_one',
            'compute_consensus': compute_consensus,
        }
        # batch_key passed through so batch_key=None (a single, unbatched
        # sample) skips the batches column.
        self._export_hvg_table(outfile = self.outfn, batch_key = batch_key)
        if not compute_consensus:
            return self.adata.copy()
        self.logger.info(f"[HVG] Subsetting AnnData object to top {len(top_genes)} consensus HVG genes.")
        return self.adata[:, top_genes].copy()
