#!/usr/bin/env python3

"""
Embed a run's UMAP + a summary sidecar into a mapping-stage sample's own qc/
directory and regenerate that sample's HTML report.

Applies whenever a run's resolved sample count is exactly 1 (see
orchestrator.py::run(), which calls this regardless of whether that single
sample was reached via --target or --sample) -- there's a natural single
mapping-stage report to embed into. Reuses whatever DimReducer already computed
(neighbors/UMAP/leiden) on the returned adata rather than recomputing it; this
module only re-plots into a second directory and writes the sidecar/report.
"""

import json
import logging

from datetime import datetime
from pathlib import Path

import pandas as pd
import scanpy as sc

from shared.utils import get_sample_output_paths, write_done_flag
from shared.dimred import UMAPPlotter


def embed_into_mapping_report(adata, sample_name: str, mapping_output_dir, sample_metadata_path,
                               mapper: str, method: str, model_class: str, tuned: bool,
                               logger: logging.Logger, force: bool = False) -> None:
    """Add this integration's UMAP and summary to a sample's mapping report.

    Only meaningful for a single-sample target, which is the only case with one
    natural report to embed into.

    Args:
        adata (AnnData): The integrated object.
        sample_name (str): The sample.
        mapping_output_dir (Path): Mapping results root.
        sample_metadata_path (Path): The sample sheet.
        mapper (str): Mapper the sample was run with.
        method (str): Integration method.
        model_class (str): Model class used.
        tuned (bool): Whether the model was autotuned.
        logger (logging.Logger): Where the progress lines go.
        force (bool): Re-embed even if the report already carries a section.
    """
    mapping_paths = get_sample_output_paths(Path(mapping_output_dir) / sample_name, mapper=mapper)
    done_path = mapping_paths['done'] / 'single_sample_integration.done'
    if done_path.exists() and not force:
        logger.info(f"[REPORT-EMBED] skipping '{sample_name}'; done flag exists (use --force to override)")
        return

    umap_key = f"X_umap_{method}"
    leiden_key = f"leiden_{method}"
    if umap_key not in adata.obsm:
        logger.warning(f"[REPORT-EMBED] '{umap_key}' not found in obsm for '{sample_name}'; skipping report embedding")
        return

    UMAPPlotter(adata, save_dir=mapping_paths['qc'], logger=logger).plot(
        umap_key=umap_key, leiden_key=leiden_key if leiden_key in adata.obs.columns else None,
        extra_color_by=[c for c in ('batch', 'solo_doublet') if c in adata.obs.columns],
        title_prefix=f"UMAP (X_{method})")

    # The doublet-score histogram is no longer drawn here. Doublet calling moved
    # into the mapping stage, so `.obs['solo_doublet*']` arrives with the
    # artifact this run loaded, and qc_plot.py::_plot_solo already renders it in
    # the sample's own QC section. Plotting it again from here would duplicate a
    # figure and, worse, imply integration had produced the calls.
    if 'solo_doublet' in adata.obs:
        doublet_col = adata.obs['solo_doublet']
        # Don't trust this to still be real bool: an artifact written before
        # orchestrator.py::_sanitize_obs stopped blanket-stringifying obs
        # columns can have this column cached on disk as the strings
        # "True"/"False" (themselves promoted to a pandas Categorical by
        # write_h5ad's own strings_to_categoricals()) -- on-disk data isn't
        # retroactively repaired by that fix. `.astype(bool)` would be wrong
        # here too: the non-empty string "False" is truthy in Python, so it
        # would silently count every cell as a doublet instead of raising.
        # Compare against the string form instead, which is correct whether
        # the column is real bool, object, or a stale Categorical of strings.
        if isinstance(doublet_col.dtype, pd.CategoricalDtype):
            doublet_col = doublet_col.astype(object)
        n_doublets = int((doublet_col.astype(str) == 'True').sum())
    else:
        n_doublets = None
    # Stamp which mapping artifact this was computed from. report.py globs these
    # figures off disk and cannot otherwise tell a figure produced from the
    # current QC output from one left behind by an integration run that predates
    # it -- the same stale-figure trap qc_plot.py clears its own prefixes to
    # avoid. Without this the report happily presents months-old scVI results as
    # current.
    parent_hash = ''
    try:
        filtered = mapping_paths['qc'] / '{}_filtered.h5ad'.format(sample_name)
        if filtered.exists():
            src = sc.read_h5ad(filtered, backed='r')
            parent_hash = str((src.uns.get('pipeline') or {}).get('config_hash', '') or '')
            if getattr(src, 'file', None) is not None:
                src.file.close()
    except Exception as e:
        logger.warning(f"[REPORT-EMBED] Could not read the mapping artifact's config_hash: {e}")

    summary = {
        'parent_config_hash': parent_hash,
        'generated': datetime.now().isoformat(timespec='seconds'),
        'model_class': model_class,
        'n_latent': int(adata.obsm[f"X_{method}"].shape[1]),
        'n_cells': int(adata.n_obs),
        'n_doublets': n_doublets,
        'doublet_fraction': (n_doublets / adata.n_obs) if n_doublets is not None and adata.n_obs else None,
        'tuned': bool(tuned)}
    summary_path = mapping_paths['qc'] / f"{sample_name}_scvi_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"[REPORT-EMBED] Wrote summary sidecar at {summary_path}")

    write_done_flag(done_path)

    from mapping.report import ReportBuilder
    try:
        ReportBuilder(sample_name, mapping_output_dir, sample_metadata_path).run(force=True)
    except Exception as e:
        logger.warning(f"[REPORT-EMBED] Could not regenerate report for '{sample_name}': {e} (run itself still succeeded)")
