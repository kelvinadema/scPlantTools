#!/usr/bin/env python3

"""
Drop SOLO+Scrublet consensus doublets, producing a manual-artifact for integration.

    python -m mapping.remove_consensus_doublets -c configs/pipeline_config.yaml \
        --sample GSE123013_WT_rep1 GSE123013_WT_rep2 ...

`doublets.py` already computes `.obs['flag_doublet_consensus']` (SOLO AND Scrublet)
on every sample, but leaves it in `advisory` mode by default -- present, not removed
(see filter_config.yaml's `doublet_consensus:` block). This script is for a sample
set that should have the consensus cells removed without touching filter_config.yaml
and forcing a `--force-qc` rerun (and the QC/report regeneration that would trigger
for every other sample sharing that filter profile).

Reads `{sample}_filtered.h5ad` (the automatic artifact -- consensus cells are already
present, just not removed) and writes `{sample}_filtered_manual.h5ad` alongside it,
registering it in the sample's manifest.json under `filtered_manual_h5ad` -- the same
key `mapping.refilter` uses, so integration's `artifact_key` selection needs
no code change to pick this up.

Never overwrites `{sample}_filtered.h5ad` itself.
"""

import argparse
import logging
import sys

from pathlib import Path

import anndata as ad

from shared.utils import load_config, setup_logger, get_sample_output_paths, pipeline_log_path
from shared.samples import resolve_sample
from shared.io import write_h5ad, write_manifest

FLAG_COL = 'flag_doublet_consensus'


class ConsensusDoubletRemovalError(ValueError):
    """Raised when consensus doublets cannot be removed for a sample."""
    pass


def _remove_consensus_doublets(sample_name: str, pipeline_config_path, logger,
                                sample_sheet_override=None) -> Path:
    """Drop the cells both doublet callers agreed on, writing a new artifact.

    Args:
        sample_name (str): Sample name.
        pipeline_config_path (Path): pipeline_config.yaml.
        logger (logging.Logger): Where the progress lines go.
        sample_sheet_override (Path, optional): Use this sheet instead of the
            config's own.

    Returns:
        Path: The written artifact.
    """
    pipeline_cfg = load_config(pipeline_config_path) or {}
    if sample_sheet_override:
        # pipeline_config.yaml's configs.samples names exactly one sheet, but a
        # sample requested here may live in a different one (e.g. N0h in
        # sample_metadata.yaml while configs.samples currently points at
        # configs/external/external_metadata.yaml) -- see the memory note on
        # this pointer flipping between internal/external. Explicit override
        # rather than trusting whatever configs.samples happens to say.
        sample_sheet = sample_sheet_override
    else:
        cfg = pipeline_cfg.get('configs') or {}
        if 'samples' not in cfg:
            raise ConsensusDoubletRemovalError(
                "{} has no configs.samples entry; cannot resolve which sample sheet "
                "'{}' lives in. Pass --sample-sheet explicitly.".format(
                    pipeline_config_path, sample_name))
        sample_sheet = cfg['samples']
    config_paths = [pipeline_config_path, sample_sheet]

    entry = resolve_sample(sample_name, sample_sheet)
    mapper = entry.get('mapper', '').lower()
    batch = entry.get('batch')

    out_root = Path(pipeline_cfg.get('output_dir', 'results/mapping'))
    sample_dir = out_root / sample_name
    paths = get_sample_output_paths(sample_dir, mapper=mapper)
    qc_dir = paths['qc']
    input_path = qc_dir / '{}_filtered.h5ad'.format(sample_name)
    output_path = qc_dir / '{}_filtered_manual.h5ad'.format(sample_name)

    if not input_path.exists():
        raise ConsensusDoubletRemovalError(
            'No automatic artifact at {}; run mapping for this sample first.'.format(input_path))

    logger.info('[DOUBLET_REMOVAL] Reading {}'.format(input_path))
    adata = ad.read_h5ad(input_path)

    if FLAG_COL not in adata.obs:
        raise ConsensusDoubletRemovalError(
            "'{}' has no '{}' column -- doublet consensus was never computed for it "
            '(check .uns[\'qc\'][\'doublet_consensus\'] for why: usually SOLO or Scrublet '
            'skipped, so no consensus call exists).'.format(sample_name, FLAG_COL))

    flagged = adata.obs[FLAG_COL].astype(bool)
    keep = ~flagged.values
    n_flagged = int(flagged.sum())
    n_keep = int(keep.sum())

    if n_keep == 0:
        raise ConsensusDoubletRemovalError(
            'Every cell in {} is flagged as a consensus doublet. Refusing to write an '
            'empty artifact.'.format(sample_name))

    filtered = adata[keep].copy()
    logger.info('[DOUBLET_REMOVAL] {}: {} cells in, {} consensus doublets removed, '
                '{} remaining'.format(sample_name, adata.n_obs, n_flagged, n_keep))

    write_h5ad(
        filtered, output_path,
        sample=sample_name, mapper=mapper, stage='doublet_consensus_removal',
        config_paths=config_paths,
        extra={'n_cells': n_keep,
               'n_genes': int(filtered.n_vars),
               'n_cells_input': int(adata.n_obs),
               'n_consensus_doublets_removed': n_flagged,
               'parent_artifact': str(input_path)})

    manifest_path = sample_dir / 'manifest.json'
    artifacts = {}
    if manifest_path.exists():
        import json
        with open(manifest_path) as f:
            existing = dict((json.load(f).get('artifacts') or {}))
        # Rehydrate to absolute paths -- write_manifest re-relativises against
        # sample_dir, and a relative path handed back in would fail its
        # existence check and silently drop every artifact this run did not
        # produce (see refilter.py's identical comment).
        artifacts = {name: sample_dir / rel for name, rel in existing.items()}
    artifacts['filtered_manual_h5ad'] = output_path
    write_manifest(manifest_path, sample=sample_name, mapper=mapper,
                    artifacts=artifacts, base_dir=sample_dir, batch=batch,
                    config_paths=config_paths)
    logger.info("[DOUBLET_REMOVAL] Registered 'filtered_manual_h5ad' in {}".format(manifest_path))
    return output_path


def main(config_path: str, samples: list, sample_sheet_override: str = None) -> int:
    """CLI entry point: remove consensus doublets from one or more samples.

    Args:
        config_path (Path): pipeline_config.yaml.
        samples (list[str]): Samples to process.
        sample_sheet_override (Path, optional): Sheet to resolve against.

    Returns:
        int: Process exit status.
    """
    pipeline_cfg = load_config(config_path) or {}
    log_file = pipeline_log_path('remove_consensus_doublets', log_dir=pipeline_cfg.get('log_dir', 'logs'))
    setup_logger(log_file=log_file)
    logger = logging.getLogger('SingleCellMappingPipeline')

    failures = []
    for sample in samples:
        try:
            _remove_consensus_doublets(sample, config_path, logger,
                                       sample_sheet_override=sample_sheet_override)
        except Exception as e:
            logger.error('[DOUBLET_REMOVAL] Sample "{}" failed: {}: {}'.format(
                sample, type(e).__name__, e), exc_info=True)
            failures.append((sample, '{}: {}'.format(type(e).__name__, e)))

    if failures:
        logger.error('[DOUBLET_REMOVAL] {} of {} sample(s) failed:'.format(len(failures), len(samples)))
        for sample, err in failures:
            logger.error('  {}: {}'.format(sample, err))
    return len(failures)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-c', '--config', required=True, help='Path to pipeline_config.yaml')
    parser.add_argument('--sample', nargs='+', required=True,
                        help='Sample(s) to remove consensus doublets from')
    parser.add_argument('--sample-sheet', dest='sample_sheet', default=None,
                        help='Sample sheet to resolve --sample against, overriding '
                             "the config's configs.samples (e.g. when a sample lives in "
                             'a different sheet than the one currently active there)')
    args = parser.parse_args()

    sys.exit(1 if main(args.config, args.sample, args.sample_sheet) else 0)
