#!/usr/bin/env python3

import logging

import numpy as np
import scanpy as sc

from pathlib import Path
from shared.utils import load_config, log_exceptions, GTFparser, write_done_flag, get_sample_output_paths
from shared.samples import load_samples
from shared.io import write_h5ad
from scipy import sparse
from scipy.sparse import issparse
from typing import Union

class ComputeQCMetrics():
    """Compute this sample's per-cell QC metrics into `.obs`."""
    def __init__(self, sample_name: str, sample_base_dir: Path, sample_config: Path, qc_config_path: Path):
        self.logger = logging.getLogger('SingleCellMappingPipeline')
        
        self.sample_name = sample_name
        self.sample_cfg = load_samples(sample_config)
        self.mapper = self.sample_cfg[self.sample_name]['mapper'].lower()
        self.config_paths = [sample_config, qc_config_path]

        self.qc_profile = self.sample_cfg[self.sample_name].get('qc_metrics', 'default')

        all_qc_configs = load_config(Path(qc_config_path)).get('QCconfigs', {})
        if self.qc_profile not in all_qc_configs:
            self.logger.warning(
                '[QC] QC metrics profile "{}" not found in qc_config.yaml for sample "{}"; falling back to "default"'.format(
                    self.qc_profile, self.sample_name))
            self.qc_profile = 'default'

        profile_cfg = all_qc_configs.get(self.qc_profile, {})
        self.metrics = profile_cfg.get('qc_metrics', {})
        self.gtf_path = profile_cfg.get('gtf_path')

        self.paths = get_sample_output_paths(Path(sample_base_dir) / self.sample_name, mapper = self.mapper)
        self.raw_dir = self.paths['raw']
        self.out_dir = self.paths['processed']

        self.adata = sc.read_h5ad(self.out_dir / '{}_qcfilters.h5ad'.format(self.sample_name))

        # Handler dict for function calling
        self._metric_handlers = {
            "feature_fraction": self._handle_feature_fraction,
            "layer_fraction": self._handle_layer_fraction,
            "cell_sparsity": (self._cell_sparsity, lambda cfg: (cfg.get("min_count", 1), cfg.get("min_cells", 1), cfg.get("layer", "X"))),
            "cell_complexity": (self._cell_complexity, lambda cfg: (cfg.get("layer", "X"),)), 
            "entropy": (self._entropy, lambda cfg: (cfg.get("layers", None),)),
            "fraction_top_n_genes": (self._fraction_top_n_genes, lambda cfg: (cfg.get('n', 10), cfg.get("layers", None),)),
            "nUMI": (self._total_counts_per_cell, lambda cfg: (cfg.get("layers", []),)),
            "nGenes": (self._n_genes_per_cell, lambda cfg: (cfg.get("layers", []),))}

    # Assign metric to specific functions
    def _get_metric_dispatchers(self):
        """Map each configurable metric name to the method that computes it.

        Returns:
            dict: {metric name: callable}.
        """
        return {
            "mitochondrial_fraction": "feature_fraction",
            "chloroplast_fraction": "feature_fraction",
            "ribosomal_rna_fraction": "feature_fraction",
            "ribosomal_protein_fraction": "feature_fraction",
            "ambient_fraction": "layer_fraction",
            "spliced_fraction": "layer_fraction",
            "cell_sparsity": "cell_sparsity",
            "cell_complexity": "cell_complexity",
            "entropy": "entropy",
            "fraction_top_n_genes": "fraction_top_n_genes",
            "nUMI": "nUMI",
            "nGenes": "nGenes",
        }

    def _handle_feature_fraction(self, metric_name: str, config: dict):
        """Compute a fraction-of-counts metric for a named gene set.

        Args:
            metric_name (str): Metric key from qc_config.yaml.
            config (dict): That metric's config block.
        """
        label = metric_name.replace("_fraction", "")
        geneset = self._get_genes_from_gtf(
            attribute=config.get("gtf_attribute", "gene_id"),
            value=config.get("gtf_value"),
            match_type=config.get("gtf_match_type", "prefix")
        )
        layer = config.get("layer", "X")
        self._feature_fraction(label, geneset, layer=layer)

    def _handle_layer_fraction(self, metric_name: str, config: dict):
        """Compute one layer's share of the counts across a set of layers.

        Args:
            metric_name (str): Metric key from qc_config.yaml.
            config (dict): That metric's config block.
        """
        layer_num = config.get("numerator", None)
        layer_denom = config.get("denominator", None)
        def ensure_list(x):
            """Coerce a scalar or sequence config value to a list."""
            if isinstance(x, list):
                return x
            elif x is None:
                return []
            else:
                return [x]
        # Get total counts for layers and store in .obs
        self._total_counts_per_cell(ensure_list(layer_num) + ensure_list(layer_denom))
        mask = config.get("mask_with_zero", False)
        self._layer_fraction(numerator = layer_num, denominator = layer_denom, mask_with_zero=mask)


    def _get_genes_from_gtf(self, attribute: str, value: str, match_type: str) -> list:
        """
        Look genes up in the GTF, reusing one parser across metrics.

        Each GTFparser opens a gffutils sqlite DB and every lookup scans all gene
        features, so building one per metric meant four opens and four full scans
        per sample -- and none of them were ever closed. get_qc_metrics closes this.
        """
        if getattr(self, '_gtf_parser', None) is None:
            self._gtf_parser = GTFparser(self.gtf_path)
        return self._gtf_parser.get_genes_by_attribute(attribute, value, match_type)

    def _close_gtf_parser(self) -> None:
        """Close the GTF database, if one was opened."""
        parser = getattr(self, '_gtf_parser', None)
        if parser is not None:
            parser.close()
            self._gtf_parser = None

    def _match_genes(self, genes):
        """
        Return the subset of var_names matching `genes` (a list of ids, or a prefix).

        Always returns a list, so callers can use `if not matched`. The prefix
        branch previously returned a boolean array, which would have raised
        "truth value of an array is ambiguous" in _feature_fraction.
        """
        if isinstance(genes, list):
            # set membership: `gene in genes` on a list made this O(n_var * n_genes)
            wanted = set(genes)
            return [gene for gene in self.adata.var_names if gene in wanted]
        elif isinstance(genes, str):
            return [gene for gene in self.adata.var_names if gene.startswith(genes)]
        else:
            raise ValueError('Pattern must be a list of gene names or a prefix string')

    def _safe_sum(self, X):
        """Row sums as a flat array, sparse or dense.

        Args:
            X: The matrix.

        Returns:
            np.ndarray: One total per cell.
        """
        total = X.sum(axis=1)
        return total.A1 if hasattr(total, "A1") else np.asarray(total).flatten()

    @property
    def _matrices(self):
        """The matrices this sample carries, keyed by layer name.

        Returns:
            dict: {layer name: matrix}, including `.X`.
        """
        return {**self.adata.layers, "X": self.adata.X}

    def _iterate_layers(self, layers, callback):
        """Run `callback` for each requested layer that exists.

        Args:
            layers (list[str]): Layer names.
            callback (callable): Called with (layer name, matrix).
        """
        for layer in layers:
            if layer in self._matrices and self._matrices[layer] is not None:
                callback(layer, self._matrices[layer])
            elif layer not in self.adata.layers:
                self.logger.warning(f"[QC] Layer '{layer}' not found in AnnData object.")
                continue

    @log_exceptions()
    def _feature_fraction(self, label: str, genes: list, layer: str='X'):
        """Fraction of a layer's counts falling in `genes`, per cell.

        Args:
            label (str): Names the `.obs` column written.
            genes (list[str]): Gene names forming the set.
            layer (str): Layer to compute over.
        """
        matched = self._match_genes(genes)
        if not matched:
            self.logger.warning(f'[QC] No genes matched for {label}')
            self.adata.obs[f'{label}_fraction_{layer}'] = 0
            return

        X_total = self._matrices.get(layer, self.adata.X)
        X_matched = self.adata[:, matched].X if layer == "X" else self.adata[:, matched].layers.get(layer)

        signal = self._safe_sum(X_matched)
        total = self._safe_sum(X_total)
        
        frac = np.full_like(total, np.nan, dtype=float)
        nonzero = total > 0
        frac[nonzero] = signal[nonzero] / total[nonzero]
        
        self.adata.obs[f"{label}_fraction_{layer}"] = frac
        self.logger.info(f'[QC] Computed {label} fraction for {self.adata.n_obs} cells using {len(matched)} genes based on layer {layer}')

    @log_exceptions()
    def _layer_fraction(self, numerator, denominator, mask_with_zero: Union[bool, str] = False):
        """
        Fraction of counts contributed by `numerator` relative to `denominator`.

        Args:
            numerator: layer name, or list of layer names to sum.
            denominator: layer name, or list of layer names to sum.
            mask_with_zero: False to disable; True to mask on 'corrected_counts';
                or a layer name. Cells with no counts in that layer get NaN.
                (Defaulted to 'corrected_counts' before, which no caller ever used
                -- _handle_layer_fraction passes config's value, default False.)
        """
        def get_total(layers):
            """Summed counts across the named layers, per cell."""
            if isinstance(layers, str):
                layers = [layers]
                
            totals = None
            for layer in layers:
                X = self.adata.layers.get(layer)
                if X is None:
                    self.logger.warning('[QC] Layer {} not found'.format(layer))
                    continue
                layer_sum = self._safe_sum(X)
                totals = layer_sum if totals is None else totals + layer_sum
            return totals

        num_counts = get_total(numerator)
        denom_counts = get_total(denominator)

        if num_counts is None or denom_counts is None:
            self.logger.warning("[QC] Ratio computation skipped due to missing layers.")
            return

        ratio = np.full_like(num_counts, np.nan, dtype=float)
        nonzero = denom_counts > 0
        ratio[nonzero] = num_counts[nonzero] / denom_counts[nonzero]

        if mask_with_zero is True:
            mask_layer = "corrected_counts"
        elif isinstance(mask_with_zero, str):
            mask_layer = mask_with_zero
        else:
            mask_layer = None

        if mask_layer:
            obs_key = f"{mask_layer}_total_counts"
            if obs_key not in self.adata.obs:
                self._total_counts_per_cell([mask_layer])
            if obs_key in self.adata.obs:
                zero_mask = self.adata.obs[obs_key].values <= 0
                ratio[zero_mask] = np.nan
                self.logger.info(f"[QC] Applied masking to {zero_mask.sum()} cells based on '{mask_layer}'")
            else:
                self.logger.warning(f"[QC] Could not find or compute '{obs_key}' for masking.")

        # get_total accepts a list, so numerator may be one; interpolating a list
        # straight into the name would produce "['a', 'b']_fraction".
        label = numerator if isinstance(numerator, str) else '_'.join(numerator)
        key = f"{label}_fraction"
        self.adata.obs[key] = ratio
        self.logger.info(f"[QC] Stored {key} in .obs")

    @log_exceptions()
    def _cell_sparsity(self, min_gene_count: int = 1, min_cells: int = 1, layer: str='X'):
        """
        Compute cell-wise sparsity: 1 - (# expressed genes in cell / # expressed genes in dataset).
        A gene is considered expressed if it is detected in at least min_cells with at least min_gene_count counts.

        Args:
            min_gene_count (int): Minimum count to consider a gene expressed in a cell.
            min_cells (int): Minimum number of cells a gene must be expressed in to be considered expressed in the dataset.
        """
        # Get the count matrix (cells x genes)
        X = self._matrices.get(layer, self.adata.X)
        # If sparse, convert to csc for efficient row/col ops
        if hasattr(X, 'tocsc'):
            X = X.tocsc()
        
        # Boolean matrix: gene is expressed in cell if count >= min_gene_count
        expressed = (X >= min_gene_count)
        gene_mask = np.array((expressed.sum(axis=0) >= min_cells)).ravel()
        n_expressed_genes = int(gene_mask.sum())

        if n_expressed_genes == 0:
            self.logger.warning('[QC] No genes passed the expression threshold for sparsity calculation.')
            self.adata.obs[f"cell_sparsity_{layer}"] = np.nan
            return

        cell_counts = np.array((expressed[:, gene_mask]).sum(axis=1)).ravel()

        # Sparsity: 1 - (# expressed genes in cell / # expressed genes in dataset)
        sparsity = 1 - cell_counts / n_expressed_genes

        self.adata.obs[f"cell_sparsity_{layer}"] = sparsity
        self.logger.info('[QC] Computed cell_sparsity for {} cells using {} expressed genes based on layer {}.'.format(self.adata.n_obs, n_expressed_genes, layer))

    @log_exceptions()
    def _cell_complexity(self, layer: str='X'):
        """
        Compute cell complexity for each cell: (# genes detected in cell) / (total # genes in dataset).
        """
        # Add genes per cell to adata.obs
        self._n_genes_per_cell(layers=[layer])
        self._total_counts_per_cell(layers=[layer])
        umi_key = "total_counts" if layer == "X" else f"{layer}_total_counts"
        gene_key = "n_genes" if layer == "X" else f"{layer}_n_genes"
        complexity = self.adata.obs[gene_key] / self.adata.obs[umi_key]
        self.adata.obs[f"cell_complexity_{layer}"] = complexity
        self.logger.info(f"[QC] Computed cell_complexity for {self.adata.n_obs} cells using layer '{layer}'")

    @log_exceptions()
    def _total_counts_per_cell(self, layers=None):
        """
        Compute total UMI counts per cell for each specified layer.
        Adds '{layer}_total_counts' to self.adata.obs for each layer.
        If layers is None, computes for .X (raw counts).
        Args:
            layers (list or None): List of layer names to compute totals for. If None, uses .X only.
        """
        def compute(layer_name, X_layer):
            """Write one layer's total counts per cell."""
            key = 'total_counts' if layer_name == 'X' else f'{layer_name}_total_counts'
            if key not in self.adata.obs:
                total = self._safe_sum(X_layer)
                self.adata.obs[key] = total
                self.logger.info(f'[QC] Computed {key} for {self.adata.n_obs} cells.')

        if layers is not None:
            self._iterate_layers(layers, compute)
        else:
            compute('X', self.adata.X)
            
    @log_exceptions()
    def _n_genes_per_cell(self, layers=None):
        """
        Compute number of genes detected (>0) per cell for each layer or .X.
        Stores as 'n_genes' (for .X) or '{layer}_n_genes' in .obs.
        """
        def compute(layer_name, X_layer):
            """Write one layer's detected-gene count per cell."""
            key = 'n_genes' if layer_name == 'X' else f'{layer_name}_n_genes'
            # Skip if already present, matching _total_counts_per_cell: both are
            # called opportunistically by _cell_complexity as well as directly.
            if key in self.adata.obs:
                return
            n_genes = (X_layer > 0).sum(axis=1)
            n_genes = n_genes.A1 if hasattr(n_genes, 'A1') else np.asarray(n_genes).flatten()
            self.adata.obs[key] = n_genes
            self.logger.info(f'[QC] Computed {key} for {self.adata.n_obs} cells.')

        if layers is not None:
            self._iterate_layers(layers, compute)
        else:
            compute('X', self.adata.X)
    
    @log_exceptions()
    def _entropy(self, layers=None):
        """
        Shannon entropy (bits) of each cell's count distribution, per layer.

        Rows are normalised to sum to 1 directly rather than by round-tripping
        through a full AnnData copy: the previous version copied the entire object
        -- every other layer included -- once per layer just to reuse
        sc.pp.normalize_total, then accumulated the entropy in a Python loop over
        every non-zero entry.
        """
        def compute(layer_name, X_layer):
            """Write one layer's per-cell expression entropy."""
            n_obs = X_layer.shape[0]

            if issparse(X_layer):
                X = X_layer.tocsr().astype(float, copy=True)
                totals = np.asarray(X.sum(axis=1)).ravel()
                # Cells with no counts stay at 0, matching normalize_total.
                scale = np.zeros_like(totals)
                np.divide(1.0, totals, out=scale, where=totals > 0)
                X = sparse.diags(scale) @ X

                coo = X.tocoo()
                # bincount over the row index is the vectorised form of
                # `for i, j, v in zip(...): entropy[i] -= v * log2(v + 1e-12)`.
                entropy = np.bincount(
                    coo.row,
                    weights=-coo.data * np.log2(coo.data + 1e-12),
                    minlength=n_obs)
            else:
                X = np.asarray(X_layer, dtype=float)
                totals = X.sum(axis=1)
                scale = np.zeros_like(totals)
                np.divide(1.0, totals, out=scale, where=totals > 0)
                X = X * scale[:, None]
                with np.errstate(divide='ignore', invalid='ignore'):
                    entropy = -np.sum(X * np.log2(X + 1e-12), axis=1)

            self.adata.obs[f"entropy_{layer_name}"] = np.asarray(entropy).flatten()
            self.logger.info(f"[QC] Computed entropy for layer '{layer_name}'")

        if layers is not None:
            self._iterate_layers(layers, compute)
        else:
            compute('X', self.adata.X)

    @log_exceptions()
    def _fraction_top_n_genes(self, n=50, layers=None):
        """
        Compute the fraction of counts in the top N most highly expressed genes for each cell.
        Sparse-safe version: works efficiently with both dense and sparse matrices.
        Stores result in .obs as 'fraction_top{n}' or '{layer}_fraction_top{n}'.
        """
        def compute(layer_name, X):
            """Write the share of a cell's counts held by its top N genes."""
            n_cells = X.shape[0]
            top_frac = np.zeros(n_cells, dtype=np.float32)
            total = self._safe_sum(X)

            if hasattr(X, 'tocsr'):
                X = X.tocsr()
                for i in range(n_cells):
                    row_data = X.data[X.indptr[i]:X.indptr[i+1]]
                    if row_data.size == 0 or total[i] == 0:
                        top_frac[i] = np.nan
                        continue
                    top_n = np.sort(row_data)[-n:]
                    top_frac[i] = top_n.sum() / total[i]
            else:
                # Dense matrix fallback
                X_dense = np.asarray(X)
                top_n_sum = np.sort(X_dense, axis=1)[:, -n:].sum(axis=1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    top_frac = top_n_sum / total
                    top_frac[total == 0] = np.nan

            key = f'fraction_top{n}' if layer_name == 'X' else f'{layer_name}_fraction_top{n}'
            self.adata.obs[key] = top_frac
            self.logger.info(f'[QC] Computed {key} for {self.adata.n_obs} cells.')

        if layers is not None:
            self._iterate_layers(layers, compute)
        else:
            compute('X', self.adata.X)

    def _is_enabled(self, key) -> bool:
        """Whether a metric is switched on in qc_config.yaml.

        Args:
            key (str): Metric name.

        Returns:
            bool: Whether to compute it.
        """
        return self.metrics.get(key, {}).get("enabled", False)

    @log_exceptions()
    def get_qc_metrics(self, force: bool = False):
        """Compute every enabled metric and write the annotated object.

        Args:
            force (bool): Re-run even if `qcmetrics.done` exists.

        Returns:
            AnnData: The object with QC columns in `.obs`.
        """
        self.logger.info(f"[QC] Starting QC computation for {self.sample_name}")
        done_path = self.paths["done"] / "qcmetrics.done"
        if done_path.exists() and not force:
            self.logger.info(f"[QC] Skipping {self.sample_name}; .done flag exists")
            return

        try:
            for metric_name, handler_key in self._get_metric_dispatchers().items():
                if not self._is_enabled(metric_name):
                    continue

                cfg = self.metrics.get(metric_name, {})
                handler = self._metric_handlers.get(handler_key)

                if handler is None:
                    self.logger.warning(f"[QC] No handler registered for metric type '{handler_key}'")
                    continue

                # Two cases: handler is (func, arg_extractor) or custom handler function
                if callable(handler):
                    handler(metric_name, cfg)
                elif isinstance(handler, tuple) and len(handler) == 2:
                    func, arg_fn = handler
                    args = arg_fn(cfg) if arg_fn else ()
                    func(*args)
                else:
                    self.logger.warning(f"[QC] Invalid handler format for '{handler_key}'")
        finally:
            # Release the sqlite connection even if a metric raised.
            self._close_gtf_parser()

        write_done_flag(done_path)
        write_h5ad(
            self.adata,
            self.paths["processed"] / f"{self.sample_name}_qcfilters.h5ad",
            sample=self.sample_name,
            mapper=self.mapper,
            stage='qc_metrics',
            config_paths=self.config_paths)
        self.logger.info(f'[QC] Finished QC for {self.sample_name}, wrote done flag')


