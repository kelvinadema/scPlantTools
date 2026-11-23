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
    natural report to embed into. `force` re-embeds even when the done flag exists.
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

    # No doublet histogram here: the calls come from the mapping stage and
    # qc_plot.py::_plot_solo already renders them; drawing them again would
    # imply integration had produced them.
    if 'solo_doublet' in adata.obs:
        doublet_col = adata.obs['solo_doublet']
        # Older artifacts can hold this column as "True"/"False" strings (a
        # Categorical after write_h5ad); see docs/decisions/obs-sanitization.md.
        # `.astype(bool)` would count the truthy string "False" as a doublet, so
        # compare the string form, which is right for bool, object or Categorical.
        if isinstance(doublet_col.dtype, pd.CategoricalDtype):
            doublet_col = doublet_col.astype(object)
        n_doublets = int((doublet_col.astype(str) == 'True').sum())
    else:
        n_doublets = None
    # Stamp which mapping artifact this was computed from. report.py globs these
    # figures off disk and cannot otherwise tell a figure produced from the
    # current QC output from one left behind by an integration run that predates
    # it -- the same stale-figure trap qc_plot.py clears its own prefixes to
    # avoid.
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
