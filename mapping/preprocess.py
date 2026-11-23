#!/usr/bin/env python3

import logging
import json
import os
import scipy.sparse as sp

import scanpy as sc
import pandas as pd
import numpy as np

from pathlib import Path
from shared.utils import run_cmd, load_config, log_exceptions, write_done_flag, get_sample_output_paths
from shared.samples import resolve_sample
from shared.io import write_h5ad
from scipy.io import mmread

# CellBender is deliberately NOT imported here. It pins torch and does not install
# on this stage's interpreter (broadinstitute/CellBender #371, #386, #395), so it
# lives in its own environment and is invoked as a subprocess, output reader
# included. See ProcessMappingResults._read_cellbender_output.

class ProcessMappingResults():
    """Turn one sample's mapper output into an AnnData, then remove ambient RNA."""
    def __init__(self, sample_name : str, sample_config_path: str, mapping_config_path: str, raw_dir: str,
                 cellbender_cfg: dict=None):
        self.sample_name = sample_name
        self.sample_cfg = resolve_sample(self.sample_name, sample_config_path)
        self.config_paths = [sample_config_path, mapping_config_path]

        self.mapping_cfg = load_config(mapping_config_path)['mappers']
        self.mapper = self.sample_cfg['mapper'].lower()
        self.mapping_profile = self.sample_cfg.get('mapping_config', 'default')
        self.mapping_args = self.mapping_cfg.get(self.mapper, {}).get(self.mapping_profile, {})

        self.sample_type = self.sample_cfg.get('sample_type', 'protoplast')
        self.paths = get_sample_output_paths(Path(raw_dir) / self.sample_name, mapper = self.mapper)
        self.raw_dir = self.paths['raw']
        self.out_dir = self.paths['processed']

        self.logger = logging.getLogger('SingleCellMappingPipeline')
        self.cellbender_cfg = cellbender_cfg or {}

    def _resolve_cellbender(self):
        """
        Locate the cellbender binary and the interpreter of its environment.

        Both come from one `cellbender.env` prefix so they cannot drift apart: the
        binary runs the tool, and the interpreter runs its own output reader (see
        _read_cellbender_output). Falls back to whatever is on PATH, which only
        works if cellbender happens to share this interpreter.

        Returns (binary, python); python is None when no env prefix is configured.
        """
        env = self.cellbender_cfg.get('env') or os.environ.get('CELLBENDER_ENV')
        if not env:
            self.logger.warning(
                '[PREPROCESSING] No cellbender.env configured; falling back to "cellbender" on '
                'PATH. Set cellbender.env in pipeline_config.yaml to the conda prefix of the '
                'CellBender environment.')
            return 'cellbender', None

        prefix = Path(env)
        binary = prefix / 'bin' / 'cellbender'
        python = prefix / 'bin' / 'python'
        for path, what in ((binary, 'cellbender binary'), (python, 'python interpreter')):
            if not path.exists():
                raise FileNotFoundError(
                    '[PREPROCESSING] {} not found at "{}". cellbender.env should be the conda '
                    'environment prefix (the directory containing bin/), not the binary '
                    'itself.'.format(what, path))
        return str(binary), str(python)


    def process(self, force: bool=False):
        """Read the mapper's matrices, build the AnnData, and run CellBender.

        Returns the corrected object, or None when skipped because
        `preprocessing.done` exists (unless `force`).
        """
        done_path = self.paths['done'] / 'preprocessing.done'
        if done_path.exists() and not force:
            self.logger.info('[PREPROCESS] skipping sample "{}"; preprocessing.done exists (use --force-preprocessing to override)'.format(self.sample_name))
            return
            
        self.logger.info('[PREPROCESS] Starting preprocessing of sample {} using mapper {}'.format(self.sample_name, self.mapper))
        if self.mapper == 'alevin':
            adata = self._process_alevin()
        elif self.mapper == 'kb':
            adata = self._process_kb()
        elif self.mapper == 'starsolo':
            adata = self._proces_starsolo()
        else:
            raise NotImplementedError('[PREPROCESS] Unsupported mapper specified "{}"'.format(self.mapper))

        self.logger.info('[PREPROCESS] Finished reading matrices. Starting ambient RNA removal of sample "{}"'.format(self.sample_name))
        
        self._remove_ambient(adata)
        write_done_flag(done_path)
        self.logger.info('[PREPROCESS] Finished preprocessing for sample "{}", writing done flag to "{}"'.format(self.sample_name, done_path))
        return adata


    def _load_count_matrix(self, matrix_dir: Path, matrix_file='matrix.mtx', barcodes_file='barcodes.tsv', genes_file='genes.tsv'):
        """Read one Matrix Market triple from `matrix_dir`; returns (CSR matrix, barcodes, genes)."""
        
        self.logger.info('[PREPROCESS] Loading matrices from "{}"'.format(matrix_dir))
        
        mtx_path = matrix_dir/ matrix_file
        bc_path = matrix_dir/ barcodes_file
        genes_path = matrix_dir/ genes_file

        if not mtx_path.exists():
            raise FileNotFoundError(f"Matrix file not found: {mtx_path}")
        
        x = mmread(mtx_path).tocsr()
        
        with open(bc_path) as f:
            barcodes = [line.strip() for line in f]

        with open(genes_path) as f:
            genes = [line.strip().split()[0] for line in f]
        
        return x, barcodes, genes

    
    def _check_align_matrices(self, **named_matrices):
        """Pad every layer to a common shape so they can share one AnnData.

        Returns ({name: padded matrix}, {name: whether it was empty}).
        """
        matrices = list(named_matrices.values())
        names = list(named_matrices.keys())

        target_rows = max(m.shape[0] for m in matrices)
        target_cols = max(m.shape[1] for m in matrices)

        validated = {}
        zero_flags = {}

        for name, mtx in zip(names, matrices):
            if mtx.nnz == 0:
                self.logger.warning('[PREPROCESS] matrix {} of sample {}, is empty'.format(name, self.sample_name))
            zero_flags[name] = mtx.nnz == 0
            
            mtx = self._pad_rows(mtx, target_rows)
            mtx = self._pad_cols(mtx, target_cols)
            validated[name] = mtx

            self.logger.debug('[PREPROCESS] [DEBUG] Matrix {} of sample {} has shape: {}'.format(name, self.sample_name, mtx.shape))
        return validated, zero_flags

    
    def _pad_rows(self, mtx, target_rows: int):
        """Zero-pad a matrix to `target_rows`."""
        if mtx.shape[0] < target_rows:
            pad = target_rows - mtx.shape[0]
            return sp.vstack([mtx, sp.csr_matrix((pad, mtx.shape[1]))])
        return mtx


    def _pad_cols(self, mtx, target_cols: int):
        """Zero-pad a matrix to `target_cols`."""
        if mtx.shape[1] < target_cols:
            pad = target_cols - mtx.shape[1]
            return sp.hstack([mtx, sp.csr_matrix((mtx.shape[0], target_cols - mtx.shape[1]))])
        return mtx


    def _build_adata(self, x, barcodes, genes, layers: dict = None):
        """Assemble an AnnData from a matrix, its axes and any same-shape extra layers."""
        self.logger.debug('[PREPROCESS] [DEBUG] Count matrix shape: {}'.format(x.shape))
        self.logger.debug('[PREPROCESS] [DEBUG] # barcodes: {}'.format(len(barcodes)))
        self.logger.debug('[PREPROCESS] [DEBUG] # genes: {}'.format(len(genes)))
        adata = sc.AnnData(x, obs=pd.DataFrame(index=barcodes), var=pd.DataFrame(index=genes))
        if layers:
            for key, val in layers.items():
                adata.layers[key] = val
        return adata


    @log_exceptions()
    def _is_usa_mode(self, meta_path: Path):
        """Whether this alevin run produced a USA-mode matrix.

        Read from `af_quant/quant.json` rather than inferred from the shape. An
        unreadable file is logged and treated as not USA mode.
        """
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            return meta.get('usa_mode', False)
        except Exception as e:
            self.logger.warning('[PREPROCESS] Unable to parse "{}": "{}". Setting USA mode to False'.format(meta_path, e))
            return False
    
    @log_exceptions()
    def _process_alevin(self):
        """Read alevin-fry output, splitting a USA matrix into its three layers.

        Returns None (after logging an error) when `af_quant/quant.json` is missing.
        """
        af_path = self.paths['base']
        meta_path = af_path/'af_quant/quant.json'

        if not meta_path.exists():
            self.logger.error('[PREPROCESS] Unable to locate alevin metadata at "{}", cannot continue'. format(meta_path))
            return 
        
        if self._is_usa_mode(meta_path):
            self.logger.info('[PREPROCESS] Detected USA mode for sample "{}"'.format(self.sample_name))
            return self._load_alevin_usa_matrix(af_path)
        else:
            self.logger.info('[PREPROCESS] USA mode not detected for sample "{}", treating as total counts only'.format(self.sample_name))
            return self._load_alevin_standard(af_path)


    @log_exceptions()
    def _load_alevin_standard(self, af_path: Path):
        """Load standard (non-USA) Alevin-fry matrix into AnnData.
        """
        mtx_path = af_path/'af_quant/alevin'
        x, bc, genes = self._load_count_matrix(mtx_path, 'quants_mat.mtx', 'quants_mat_rows.txt', 'quants_mat_cols.txt')

        # build adata
        adata = self._build_adata(x, bc, genes)
        self.logger.info('Loaded standard anndata object with {} cells and {} genes'.format(adata.n_obs, adata.n_vars))
        self.logger.debug('[PREPROCESS] [DEBUG] Matrix shapes - x: {}'.format(x.shape))

        # Save as .h5ad
        outfile = self.paths['raw'] / f"{self.sample_name}.h5ad"
        adata.write_h5ad(outfile)
        return adata


    @log_exceptions()
    def _load_alevin_usa_matrix(self, af_path: Path):
        """
        Load an alevin-fry USA-mode matrix: layers spliced, unspliced, ambiguous,
        and the resolved rspliced/runspliced, with `.X` = rspliced + runspliced.
        """
        mtx_path = af_path/'af_quant/alevin'
        x, bc, genes_full = self._load_count_matrix(mtx_path, 'quants_mat.mtx', 'quants_mat_rows.txt', 'quants_mat_cols.txt')
        ngenes = len(genes_full) // 3
        self.logger.debug('[PREPROCESS] [DEBUG] {} genes detected'.format(ngenes))
        genes = genes_full[:ngenes]

        # Extract spliced, unspliced and ambiguous matrix
        s = x[:, 0:ngenes]
        u = x[:, ngenes:2*ngenes]
        a = x[:, 2*ngenes:3*ngenes]

        # Resolve ambiguous
        rs, ru = self._resolve_ambiguous_reads(u, s, a)

        # Build adata
        adata = self._build_adata(x = rs + ru, barcodes = bc, genes = genes, layers = {'spliced': s, 'unspliced': u, 'ambiguous': a, 'rspliced': rs, 'runspliced': ru})
        self.logger.info('[PREPROCESS] Loaded USA anndata object with {} cells and {} genes'.format(adata.n_obs, adata.n_vars))
        self.logger.debug('[PREPROCESS] [DEBUG] Matrix shapes - Spliced: {}, Unspliced: {}, Ambiguous: {}'.format(s.shape, u.shape, a.shape))

        # Save as .h5ad
        outfile = self.paths['raw'] / f"{self.sample_name}.h5ad"
        adata.write_h5ad(outfile)
        return adata


    def _process_kb(self):
        """Read kb output into an AnnData."""
        kb_path = self.paths['base']
        workflow = self.mapping_args.get('extra_mapping_args', [])
        if '--workflow' in workflow:
            idx = workflow.index('--workflow')
            selected = workflow[idx + 1] if idx + 1 < len(workflow) else 'standard'
            self.logger.info('[PREPROCESS] Preproccesing using profile "{}" for mapper "{}" '.format(selected , self.mapper))
        else:
            self.logger.warning('[PREPROCESS] workflow for "{}" in profile {} was not explicitly specified, defaulting to "standard"'.format(self.mapper, self.mapping_profile))
            selected = 'standard'
        
        if selected == 'nac':
            return self._load_kb_nac(kb_path)
        else:
            return self._load_kb_standard(kb_path)


    @log_exceptions()
    def _load_kb_standard(self, kb_path):
        """Load Kallisto-Bustools standard output.
        """
        mtx_path = kb_path/'counts_unfiltered'

        x, bc, genes = self._load_count_matrix(mtx_path, 'cells_x_genes.mature.mtx', 'cells_x_genes.barcodes.txt', 'cells_x_genes.genes.txt')
        adata = self._build_adata(x = x, barcodes = bc, genes = genes)
        self.logger.info('[PREPROCESS] [DEBUG] Loaded KB standard anndata object with {} cells and {} genes'.format(adata.n_obs, adata.n_vars))
        self.logger.debug('[PREPROCESS] [DEBUG] Matrix shapes - x: {}'.format(x.shape))
        
        # Save as .h5ad
        outfile = self.paths['raw'] / f"{self.sample_name}.h5ad"
        adata.write_h5ad(outfile)
        return adata


    @log_exceptions()
    def _load_kb_nac(self, kb_path):
        """Load Kallisto-Bustools NAC output with 3 matrices.
        """
        mtx_path = kb_path/'counts_unfiltered'

        s, bc, genes = self._load_count_matrix(mtx_path, 'cells_x_genes.mature.mtx', 'cells_x_genes.barcodes.txt', 'cells_x_genes.genes.txt')
        u, *_= self._load_count_matrix(mtx_path, 'cells_x_genes.nascent.mtx', 'cells_x_genes.barcodes.txt', 'cells_x_genes.genes.txt')
        a, *_ = self._load_count_matrix(mtx_path, 'cells_x_genes.ambiguous.mtx', 'cells_x_genes.barcodes.txt', 'cells_x_genes.genes.txt')

        rs, ru = self._resolve_ambiguous_reads(u, s, a)

        adata = self._build_adata(x = rs + ru, barcodes = bc, genes = genes, layers = {'spliced': s, 'unspliced': u, 'ambiguous': a, 'rspliced': rs, 'runspliced': ru})
        self.logger.info('[PREPROCESS] Loaded standard KB NAC anndata object with {} cells and {} genes'.format(adata.n_obs, adata.n_vars))
        self.logger.debug('[PREPROCESS] [DEBUG] Matrix shapes - Spliced: {}, Unspliced: {}, Ambiguous: {}'.format(s.shape, u.shape, a.shape))

        # Save as .h5ad
        outfile = self.paths['raw'] / f"{self.sample_name}.h5ad"
        adata.write_h5ad(outfile)
        return adata


    def _proces_starsolo(self):
        """Read STARsolo output into an AnnData."""
        ss_path = self.paths['base']
        workflow = self.mapping_args.get('extra_mapping_args', [])
        if '--soloFeatures' in workflow:
            idx = workflow.index('--soloFeatures')
            selected = workflow[idx + 1] if idx + 1 < len(workflow) else 'VeloCyto'
            self.logger.info('[PREPROCESS] Preproccesing using profile "{}" for mapper "{}" '.format(selected , self.mapper))
        else:
            self.logger.warning('[PREPROCESS] workflow for "{}" in profile {} was not explicitly specified, defaulting to "VeloCyto"'.format(self.mapper, self.mapping_profile))
            selected = 'VeloCyto'
        
        if selected == 'VeloCyto':
            return self._load_starsolo_velocyto(ss_path)
        elif selected == 'GeneFull':
            return self._load_starsolo_genefull(ss_path)
        else:
            return self._load_starsolo_gene(ss_path)


    @log_exceptions()
    def _load_starsolo_gene(self, solo_path: Path):
        """Read STARsolo's Gene matrix into an AnnData (under `solo_path/Solo.out`)."""
        mtx_path = solo_path/'Solo.out/Gene/raw'

        x, bc, genes = self._load_count_matrix(mtx_path, 'matrix.mtx', 'barcodes.tsv', 'features.tsv')
        adata = self._build_adata(x = x, barcodes = bc, genes = genes)

        self.logger.info('[PREPROCESS] Loaded standard STARsolo anndata object with {} cells and {} genes'.format(adata.n_obs, adata.n_vars))

        outfile = self.paths['raw'] / f"{self.sample_name}.h5ad"
        adata.write_h5ad(outfile)
        return adata


    @log_exceptions()
    def _load_starsolo_genefull(self, solo_path: Path):
        """Read STARsolo's GeneFull spliced/unspliced matrices into an AnnData, `.X` = their sum."""
        mtx_path = solo_path/'Solo.out/GeneFull/raw'

        s, bc, genes = self._load_count_matrix(mtx_path, 'spliced.mtx', 'barcodes.tsv', 'features.tsv')
        u, *_= self._load_count_matrix(mtx_path, 'unspliced.mtx', 'barcodes.tsv', 'features.tsv')

        adata = self._build_adata(x = s + u, barcodes = bc, genes = genes, layers = {'spliced': s, 'unspliced': u})
        self.logger.info('[PREPROCESS] Loaded GeneFull STARsolo anndata object with {} cells and {} genes'.format(adata.n_obs, adata.n_vars))
        
        outfile = self.paths['raw'] / f"{self.sample_name}.h5ad"
        adata.write_h5ad(outfile)
        return adata


    @log_exceptions()
    def _load_starsolo_velocyto(self, solo_path: Path):
        """Read STARsolo's Velocyto spliced/unspliced/ambiguous matrices into an AnnData,
        resolving the ambiguous reads as for alevin USA mode."""
        mtx_path = solo_path/'Solo.out/Velocyto/raw'

        s, bc, genes = self._load_count_matrix(mtx_path, 'spliced.mtx', 'barcodes.tsv', 'features.tsv')
        u, *_= self._load_count_matrix(mtx_path, 'unspliced.mtx', 'barcodes.tsv', 'features.tsv')
        a, *_ = self._load_count_matrix(mtx_path, 'ambiguous.mtx', 'barcodes.tsv', 'features.tsv')

        rs, ru = self._resolve_ambiguous_reads(u, s, a)
        adata = self._build_adata(x = rs + ru, barcodes = bc, genes = genes, layers = {'spliced': s, 'unspliced': u, 'ambiguous': a, 'rspliced': rs, 'runspliced': ru})
        self.logger.info('[PREPROCESS] Loaded velocyto STARsolo anndata object with {} cells and {} genes'.format(adata.n_obs, adata.n_vars))

        outfile = self.paths['raw'] / f"{self.sample_name}.h5ad"
        adata.write_h5ad(outfile)
        return adata


    def _compute_cell_gene_ratios(self, spliced, unspliced):
        """
        Compute per-cell-per-gene ratios (spliced / (spliced + unspliced)).
        Returns a sparse matrix of same shape.
        """
        total = spliced + unspliced
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_total = total.copy().astype(float)
            inv_total.data = 1.0 / inv_total.data  # reciprocal of nonzeros
            ratio = spliced.astype(float).multiply(inv_total)
        return ratio

    def _compute_gene_ratios(self, spliced, unspliced):
        """
        Compute per-gene ratios (length = n_genes).
        Used as fallback when per-cell info is uninformative.
        """
        gene_totals = np.array(spliced.sum(axis=0)).ravel() + np.array(unspliced.sum(axis=0)).ravel()
        gene_spliced = np.array(spliced.sum(axis=0)).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            gene_ratios = np.divide(gene_spliced, gene_totals, out=np.zeros_like(gene_spliced, dtype=float), where=gene_totals > 0)
        return gene_ratios

    def _compute_global_ratio(self, spliced, unspliced):
        """
        Compute dataset-wide ratio (scalar).
        Used as fallback when a gene has no informative reads.
        """
        total_spliced = spliced.sum()
        total_unspliced = unspliced.sum()
        total = total_spliced + total_unspliced
        return float(total_spliced) / total if total > 0 else np.nan

    def _resolve_with_fallbacks(self, spliced, unspliced, ambiguous, gene_ratios, global_ratio, sample_type):
        """
        Split ambiguous reads into spliced/unspliced parts, preserving totals.

        The main path uses each cell-gene's own spliced share. Ambiguous-only
        entries (no spliced or unspliced reads there) use the gene's share,
        falling back to the global share where the gene's is 0; if the global
        share is undefined, protoplasts get 1.0 (spliced) and other sample
        types 0.0. Returns (ambiguous spliced part, ambiguous unspliced part).
        """

        # --- Step 1: per-cell-per-gene ratios (main path) ---
        ratio = self._compute_cell_gene_ratios(spliced, unspliced)

        # --- Step 2: split ambiguous using ratio ---
        amb_spliced = ambiguous.multiply(ratio)
        amb_unspliced = ambiguous - amb_spliced

        # --- Step 3: round while preserving totals ---
        amb_spliced.data = np.rint(amb_spliced.data).astype(np.int64)
        amb_unspliced = ambiguous - amb_spliced  # recompute to preserve totals exactly
        amb_unspliced.eliminate_zeros()

        # --- Step 4: vectorized fallback for ambiguous-only entries ---
        # Convert to COO for easy indexing
        amb_coo = ambiguous.tocoo(copy=False)
        total = (spliced + unspliced).tocoo(copy=False)

        # Build a dict for fast lookup
        total_dict = {(i, j): v for i, j, v in zip(total.row, total.col, total.data)}

        # Identify ambiguous-only entries
        mask_indices = [(i, j, v) for i, j, v in zip(amb_coo.row, amb_coo.col, amb_coo.data)
                        if v > 0 and total_dict.get((i, j), 0) == 0]

        if mask_indices:
            rows, cols, vals = zip(*mask_indices)
            rows = np.array(rows)
            cols = np.array(cols)
            vals = np.array(vals)

            # Compute fallback ratios
            fallback_ratios = gene_ratios[cols]
            fallback_ratios = np.where(fallback_ratios > 0, fallback_ratios, global_ratio)
            if np.isnan(global_ratio):
                sample_ratio = 1.0 if sample_type == "protoplast" else 0.0
                fallback_ratios = np.where(np.isnan(fallback_ratios), sample_ratio, fallback_ratios)

            # Compute split
            sp_vals = np.rint(vals * fallback_ratios).astype(np.int64)
            unsp_vals = vals.astype(np.int64) - sp_vals

            # --- Replace ambiguous-only entries instead of adding ---
            amb_spliced = amb_spliced.tolil()
            amb_unspliced = amb_unspliced.tolil()
            amb_spliced[rows, cols] = sp_vals
            amb_unspliced[rows, cols] = unsp_vals
            amb_spliced = amb_spliced.tocsr()
            amb_unspliced = amb_unspliced.tocsr()

        return amb_spliced, amb_unspliced

    def _resolve_ambiguous_reads(self, u, s, a):
        """
        Resolve ambiguous reads into spliced/unspliced counts.

        Returns (resolved spliced, resolved unspliced); callers store them as the
        'rspliced'/'runspliced' layers. Asserts that no read is lost.
        """
        checked_mtx, flags = self._check_align_matrices(spliced=s, unspliced=u, ambiguous=a)
        spliced, unspliced, ambiguous = checked_mtx['spliced'], checked_mtx['unspliced'], checked_mtx['ambiguous']
        # Early exit if no ambiguous reads
        if flags['ambiguous']:
            self.logger.info(f'[PREPROCESS] No ambiguous reads detected for "{self.sample_name}", skipping resolve.')
            return spliced, unspliced

        gene_ratios = self._compute_gene_ratios(spliced, unspliced)
        global_ratio = self._compute_global_ratio(spliced, unspliced)
        # The class-wide resolved value (default 'protoplast'), so an unset
        # sample_type does not fall to the nuclei branch (ratio 0.0) here while
        # the rest of the code treats it as protoplast.
        sample_type = self.sample_type

        # Resolve ambiguous
        amb_spliced, amb_unspliced = self._resolve_with_fallbacks(spliced, unspliced, ambiguous, gene_ratios, global_ratio, sample_type)

        # Add resolved ambiguous to base counts
        resolved_spliced = (spliced + amb_spliced).tocsr()
        resolved_unspliced = (unspliced + amb_unspliced).tocsr()
        self.logger.info('rs: {}, ru: {}, s: {}, u: {}, a: {}'.format(resolved_spliced.sum(), resolved_unspliced.sum(), spliced.sum(), unspliced.sum(), ambiguous.sum()))
        self.logger.info('rs + ru: {}, s + u + a: {}'.format(resolved_spliced.sum() + resolved_unspliced.sum(), spliced.sum() + unspliced.sum() + ambiguous.sum()))
        assert float(resolved_spliced.sum() +  resolved_unspliced.sum()) == float(spliced.sum() + unspliced.sum() + ambiguous.sum()), "Ambiguous reads were lost"
        return resolved_spliced, resolved_unspliced

    @log_exceptions()
    def _read_cellbender_output(self, cb_output: Path, cb_python: str):
        """
        Load CellBender's output .h5 without importing cellbender into this process.

        CellBender's h5 is read by `cellbender.remove_background.downstream.
        anndata_from_h5`, which walks the file and routes arrays to obs/var/uns by
        shape. Rather than reimplement that here -- where a subtly wrong reader
        would silently corrupt counts -- the conversion is run by the interpreter
        of CellBender's own environment, so its reader stays authoritative and this
        stage only has to read an h5ad.

        Falls back to importing cellbender directly when no env is configured,
        which only works if it shares this interpreter.
        """
        if cb_python is None:
            from cellbender.remove_background.downstream import anndata_from_h5
            return anndata_from_h5(cb_output, analyzed_barcodes_only=False)

        cb_h5ad = self.paths['processed'] / '_temp_cb_output.h5ad'
        script = (
            'from cellbender.remove_background.downstream import anndata_from_h5\n'
            'a = anndata_from_h5({inp!r}, analyzed_barcodes_only=False)\n'
            'a.write_h5ad({out!r})\n').format(inp=str(cb_output), out=str(cb_h5ad))

        self.logger.info('[PREPROCESSING] Converting CellBender output via {}'.format(cb_python))
        run_cmd([cb_python, '-c', script])

        if not cb_h5ad.exists():
            raise FileNotFoundError(
                '[PREPROCESSING] CellBender output conversion produced no file at {}'.format(cb_h5ad))
        return sc.read_h5ad(cb_h5ad)

    def _remove_ambient(self, adata, input_layer: str=None, use_cuda: bool=True, learning_rate: float = 5e-5, inplace: bool=True):
        """Run CellBender on an in-memory AnnData object and reintegrate results.

        With `inplace` (the default), the original `.X` goes to
        `.layers['raw_counts']`, the corrected counts to `.X` and
        `.layers['corrected_counts']`, and the object is written as
        `{sample}_qcfilters.h5ad`. `cellbender_cfg` overrides `use_cuda` and
        `learning_rate`.
        """

        tmp_input = self.paths['raw'] / '_temp_cb_input.h5ad'
        cb_output = self.paths['processed'] / '_temp_cb_output.h5'

        if inplace:
            adata.layers['raw_counts'] = adata.X.copy()

        if input_layer:
            adata.layers['counts'] = adata.layers[input_layer].copy()
        else:
            adata.layers['counts'] = adata.X.copy()

        # Write temp input
        adata.write_h5ad(tmp_input)

        binary, cb_python = self._resolve_cellbender()
        use_cuda = self.cellbender_cfg.get('use_cuda', use_cuda)
        learning_rate = self.cellbender_cfg.get('learning_rate', learning_rate)

        cmd = [binary, 'remove-background', '--input', str(tmp_input), '--output', str(cb_output), '--learning-rate', str(learning_rate)]
        if use_cuda:
            cmd.append('--cuda')
        cmd += [str(a) for a in self.cellbender_cfg.get('extra_args', [])]
        self.logger.info('[PREPROCESSING] Running: {}'.format(' '.join(cmd)))
        # CellBender writes its training checkpoint as 'ckpt.tar.gz' relative to
        # the process cwd (hardcoded in cellbender/remove_background/run.py;
        # `--checkpoint` only names an input tarball to resume from). Running it
        # in this sample's processed_matrices/ keeps that multi-GB file out of
        # the repo root and next to cb_output, where _cleanup_cellbender_temp
        # removes it on success.
        run_cmd(cmd, cwd=str(self.paths['processed']))

        # Integrate output
        cb_adata = self._read_cellbender_output(cb_output, cb_python)
        self._cleanup_cellbender_temp(tmp_input, cb_output)
        if inplace:
            adata.X = cb_adata.X.copy()
            adata.obs = cb_adata.obs
            adata.var = cb_adata.var

            # Also save as explicit layer
            adata.layers['corrected_counts'] = cb_adata.X.copy()
            # Stamped here, after cb_adata.obs replaced the frame above. Written
            # under the same name qc_metrics/qc_filter will read and re-write --
            # preprocessing, QC metrics and QC filtering are additive passes over
            # one shared h5ad (see shared/io.py), not three separate artifacts.
            write_h5ad(
                adata,
                self.paths['processed'] / f"{self.sample_name}_qcfilters.h5ad",
                sample=self.sample_name,
                mapper=self.mapper,
                stage='preprocess',
                config_paths=self.config_paths)
            return adata
        else:
            cb_adata.write_h5ad(self.paths['processed'] / f"{self.sample_name}_CBout.h5ad")
            return cb_adata

    def _cleanup_cellbender_temp(self, tmp_input: Path, cb_output: Path) -> None:
        """
        Remove CellBender's transient input/output files after a successful run.

        CellBender writes several siblings alongside `--output cb_output`
        (.log, .pdf, _cell_barcodes.csv, _filtered.h5, _metrics.csv,
        _posterior.h5, _report.html) in addition to the .h5 itself, on top of
        `_read_cellbender_output`'s own `_temp_cb_output.h5ad` conversion and
        `tmp_input`; none of them are needed once `cb_adata` has been read
        back into memory here. Only called on the success path -- a failed
        run's temp files are left in place, since they are the only record of
        what CellBender was given and produced.

        `ckpt.tar.gz` does not share `cb_output`'s stem, so it is listed
        explicitly (see the comment at the call site for why it is there).
        """
        removed = []
        candidates = ([tmp_input, cb_output.parent / 'ckpt.tar.gz']
                       + sorted(cb_output.parent.glob(cb_output.stem + '*')))
        for path in candidates:
            path = Path(path)
            try:
                if path.exists():
                    path.unlink()
                    removed.append(str(path))
            except OSError as e:
                self.logger.warning('[PREPROCESSING] Could not remove temp file {}: {}'.format(path, e))
        if removed:
            self.logger.info('[PREPROCESSING] Removed {} CellBender temp file(s): {}'.format(
                len(removed), ', '.join(Path(p).name for p in removed)))
