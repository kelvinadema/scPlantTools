import argparse
import logging
import os
import sys

import scanpy as sc

from shared.utils import load_config, setup_logger, get_sample_output_paths, pipeline_log_path
from shared.samples import load_samples, resolve_index_ref
from shared.cohorts import resolve_cohort
from shared.config_spec import KNOWN_MAPPERS as _KNOWN, IMPLEMENTED_MAPPERS as _IMPLEMENTED
from shared.validate import validate_mapping
from shared.io import write_manifest
from .index import IndexBuilder
from .mapping import SingleCellMapper
from .mappingutils import StatsExtractor
from .preprocess import ProcessMappingResults
from .qc_metrics import ComputeQCMetrics
from .qc_filter import FilterQCMetrics
from .doublets import DetectDoublets
from .qc_plot import QCPlotter
from .report import ReportBuilder, write_run_index

from pathlib import Path


# Mappers the config schema accepts. kb and starsolo are wired into the config
# indirection but not yet implemented end-to-end; validate_samples() gates them.
# Defined in shared/config_spec.py so this gate and shared/validate.py's cannot
# drift apart.
KNOWN_MAPPERS = set(_KNOWN)
IMPLEMENTED_MAPPERS = set(_IMPLEMENTED)


def validate_samples(samples: dict) -> None:
    """
    Reject any sample naming an unusable mapper, before the run does any work.

    Deliberately called from main() outside the per-sample try/except, which
    would otherwise swallow these as ordinary per-sample errors and let the run
    exit 0. Reports every offending sample at once rather than failing on the
    first, so a 40-sample sheet needs one round-trip to fix.
    """
    unknown = {}
    gated = {}

    for sample_name, sample_data in samples.items():
        mapper = sample_data.get('mapper', '').lower()
        if mapper not in KNOWN_MAPPERS:
            unknown[sample_name] = sample_data.get('mapper', '')
        elif mapper not in IMPLEMENTED_MAPPERS:
            gated[sample_name] = mapper

    if unknown:
        raise ValueError(
            'Unknown mapper(s) in sample config. Known mappers are {}. Offending samples: {}'.format(
                sorted(KNOWN_MAPPERS),
                ', '.join('{}={!r}'.format(s, m) for s, m in sorted(unknown.items()))))

    if gated:
        raise NotImplementedError(
            'Mapper(s) {} are not yet implemented end-to-end; only {} is currently supported. '
            'Offending samples: {}'.format(
                sorted(set(gated.values())),
                sorted(IMPLEMENTED_MAPPERS),
                ', '.join('{}={}'.format(s, m) for s, m in sorted(gated.items()))))


def main(config_path: str, cohort: str=None, build_indexes=False, force_list=None, force_mapping=False, force_preprocessing=False, force_qc=False, force_plotting=False, force_mapstats=False, force_report=False, force_doublets=False, skip_validation=False) -> int:
    """Run the mapping stage over every sample in the resolved cohort.

    Each sample is wrapped so one failure does not kill the run, but failures are
    collected and the process exits 1 if any occurred.

    Args:
        config_path (Path): pipeline_config.yaml.
        cohort (str, optional): Cohort name from the `cohorts:` block.
        build_indexes (bool): Build any index the cohort needs first.
        force_list (list[str]): Index names to rebuild unconditionally.
        force_mapping (bool): Ignore `mapping.done`.
        force_preprocessing (bool): Ignore `preprocessing.done`.
        force_qc (bool): Ignore the QC metric and filter sentinels.
        force_plotting (bool): Ignore `qcplot.done`.
        force_mapstats (bool): Ignore `mappingstats.done`.
        force_report (bool): Ignore `report.done`.
        force_doublets (bool): Ignore the doublet sentinel.
        skip_validation (bool): Skip preflight.

    Returns:
        int: 0 if every sample succeeded, else 1.
    """

    # Load top level config
    pipeline_cfg = load_config(config_path)
    cfg = dict(pipeline_cfg['configs'])

    # Which dataset this run operates on. With no `cohorts:` block this is a
    # no-op and `configs.samples` still selects it; with one, --cohort (or
    # `default_cohort:`) does, and the choice is logged so a run's own output
    # records which cohort produced it rather than leaving it to be inferred
    # from the sheet path. See shared/cohorts.py.
    selected = resolve_cohort(pipeline_cfg, cohort)
    if selected['samples']:
        cfg['samples'] = selected['samples']
    if not cfg.get('samples'):
        raise ValueError(
            'no sample sheet: pipeline_config.yaml sets neither configs.samples nor a '
            'cohort supplying one.')

    # Get each sublevel config. load_samples folds in the sheet's `defaults:`
    # block, so entries below read as if every key were written out per sample;
    # it is also what every stage class uses, so they cannot disagree about what
    # a sample says.
    samples = load_samples(cfg['samples'])
    mapper_cfg = load_config(cfg['mapping'])
    index_cfg = load_config(cfg['indexes'])
    tech_cfg = load_config(cfg['technologies'])
    
    # No fallback on purpose: this is a site-specific data location (this
    # cluster's nobackup tree), not something a fresh clone should silently
    # inherit -- a bare relative default would resolve against wherever cwd
    # happens to be, which is only correct by coincidence.
    out_path = selected.get('output_dir') or pipeline_cfg.get('output_dir')
    if not out_path:
        raise ValueError(
            'pipeline_config.yaml is missing required key "output_dir". It is an '
            'absolute path on this machine and has no default -- set it explicitly.')

    # CellBender runs from its own environment (it pins torch and will not install
    # alongside this stage); empty means "whatever is on PATH".
    cellbender_cfg = pipeline_cfg.get('cellbender', {}) or {}
    if cellbender_cfg.get('env'):
        # Publish the configured prefix so shared.io.tool_versions can read the
        # version out of that env for the provenance record, and any child process
        # inherits the same answer. Config stays the single source of truth.
        os.environ['CELLBENDER_ENV'] = str(cellbender_cfg['env'])

    # Doublet detection runs from its own environment for the same reason
    # (scvi-tools does not support this stage's pandas 3.x). Published for the
    # same reason too: without it tool_versions cannot reach scvi-tools and the
    # provenance record silently omits the version of a tool that changes the
    # doublet calls.
    doublet_cfg = pipeline_cfg.get('doublets', {}) or {}
    if doublet_cfg.get('env'):
        os.environ['DOUBLET_ENV'] = str(doublet_cfg['env'])

    # Initiate logging (configured exactly once, here). One file per run, in
    # its own logs/mapping/ subdirectory -- see shared.utils.pipeline_log_path
    # for why (every stage now shares this convention) and
    # batch_scripts/run_mapper.sh's logs/mapping-<jobid>.out/.err for the
    # matching SLURM-side naming.
    log_file = pipeline_log_path('mapping', log_dir=pipeline_cfg.get('log_dir', 'logs'))
    logger = setup_logger(log_file=log_file, level=logging.INFO)
    logger.info('[MAIN] Loaded configs, starting pipeline')

    # Fail the whole run up front on an unusable mapper, rather than per sample
    # inside the loop below where the try/except would mask it.
    # Full preflight over the whole config tree, before any work is done. This
    # is what catches an index_ref naming nothing, an index built for a
    # different mapper, or a FASTQ that has moved -- all of which would
    # otherwise surface hours in, after CellBender has spent GPU time.
    # Warnings are logged and the run continues; a missing index is a normal
    # pre---build-indexes state.
    if not skip_validation:
        report = validate_mapping(config_path, cohort=cohort)
        for warning in report.warnings:
            logger.warning('[VALIDATE] {}'.format(warning))
        for note in report.notes:
            logger.info('[VALIDATE] {}'.format(note))
        if not report.ok:
            for error in report.errors:
                logger.error('[VALIDATE] {}'.format(error))
            raise ValueError(
                'Config validation failed with {} error(s); see the [VALIDATE] lines above. '
                'Run `python -m mapping.validate -c {}` to re-check, or pass '
                '--skip-validation to run anyway.'.format(len(report.errors), config_path))
        logger.info('[VALIDATE] Config tree OK ({} warning(s))'.format(len(report.warnings)))

    validate_samples(samples)

    # Build indexes if required. tmp_dir is optional; IndexBuilder falls back to
    # $TMPDIR (set per-job by SLURM) then the system temp dir.
    if build_indexes:
        build_required_index(samples, index_cfg, mapper_cfg, force_list, work_dir=pipeline_cfg.get('tmp_dir'))

    failures = []

    for sample_name, sample_data in samples.items():
        logger.info('[MAPPING] Processing sample: {}'.format(sample_name))
        try:
            # Map samples
            mapper = SingleCellMapper(
                sample_name=sample_name,
                sample_config_path=cfg['samples'],
                mapper_config_path=cfg['mapping'],
                index_config_path=cfg['indexes'],
                tech_config_path=cfg['technologies'],
                out_path=out_path,
            )
            mapper.run(force = force_mapping)

            # Preprocess samples. cellbender_cfg carries the isolated environment
            # CellBender is invoked from; see ProcessMappingResults._resolve_cellbender.
            preprocessor = ProcessMappingResults(sample_name, sample_config_path = cfg['samples'], mapping_config_path = cfg['mapping'], raw_dir = out_path, cellbender_cfg = cellbender_cfg)
            preprocessor.process(force = force_preprocessing)

            # Get mapping statistics
            stats = StatsExtractor(sample_name, sample_config_path = cfg['samples'], raw_dir = out_path)
            stats.report_stats(force = force_mapstats)

            # Compute QC metrics
            qccomputer = ComputeQCMetrics(sample_name, out_path, cfg['samples'], cfg['qc'])
            qccomputer.get_qc_metrics(force = force_qc)

            # Filter QC metrics
            qcfilter = FilterQCMetrics(sample_name, out_path, cfg['samples'], cfg['filter'])
            qcfilter.run(force = force_qc)

            # Doublet detection. Between filtering and plotting on purpose: it
            # trains on the cells the cascade passed, and its flags have to be
            # in .uns['qc'] before QCPlotter builds the removal mask, so that
            # solo.mode decides whether doublets are removed exactly like any
            # other filter. Also emits X_umap_scVI for the QC report.
            doublets = DetectDoublets(sample_name, out_path, cfg['samples'], cfg['filter'],
                                      config_path)
            # A returned reason means the SOLO worker faulted (as opposed to
            # being deliberately skipped, which returns None). The sample still
            # finishes -- doublet calls are additive, and its filtered artifact
            # and report are worth keeping -- but it is counted as a failure so
            # the run exits non-zero and the fault cannot pass unnoticed.
            # doublets.py deliberately leaves `doublets.done` unwritten, so the
            # next run retries it without needing --force-doublets.
            doublet_failure = doublets.run(force = force_doublets)

            # Plot metrics
            plots = QCPlotter(sample_name, out_path, cfg['samples'])
            plots.run(force = force_plotting)

            # Build the report before the manifest: write_manifest omits artifacts
            # that don't exist yet, so the report has to be on disk to be listed.
            # Guarded separately -- the report is a read-only view of work already
            # done, and a rendering bug must not discard a successful sample.
            try:
                ReportBuilder(sample_name, out_path, cfg['samples']).run(force = force_report)
            except Exception as e:
                logger.warning('[REPORT] Could not build report for {}: {} (sample still succeeded)'.format(sample_name, e))

            write_sample_manifest(sample_name, sample_data, out_path, cfg)

            # Recorded after the artifacts are safely written, so a doublet
            # fault costs the sample its green status without costing it the
            # evidence needed to diagnose that fault.
            if doublet_failure:
                failures.append((sample_name, RuntimeError(
                    'doublet detection failed (the sample\'s other QC artifacts were still '
                    'written): {}'.format(doublet_failure))))

        except Exception as e:
            logger.exception('[MAPPING] Error processing {}: {} '.format(sample_name, e))
            failures.append((sample_name, e))

    # Run-level index. Guarded for the same reason as the per-sample reports: it
    # must not be able to change the run's exit code.
    try:
        write_run_index(out_path, samples, failures)
    except Exception as e:
        logger.warning('[REPORT] Could not build run index: {}'.format(e))

    n_ok = len(samples) - len(failures)
    if failures:
        logger.error('[MAIN] Finished with failures: {}/{} samples succeeded. Failed: {}'.format(
            n_ok, len(samples), ', '.join(name for name, _ in failures)))
    else:
        logger.info('[MAIN] Finished: {}/{} samples succeeded'.format(n_ok, len(samples)))

    return len(failures)


def write_sample_manifest(sample_name: str, sample_data: dict, out_path: str, cfg: dict) -> None:
    """
    Record what a completed sample produced, above the mapper directory.

    Lets a downstream stage discover a sample's outputs by globbing
    `{out_path}/*/manifest.json` without having to know which mapper ran or to
    re-read the mapping stage's sample config.
    """
    mapper = sample_data['mapper'].lower()
    sample_dir = Path(out_path) / sample_name
    paths = get_sample_output_paths(sample_dir, mapper=mapper, create=False)

    # Lift the doublet model's parameters out of .uns and into the manifest, so
    # "was this sample's doublet call from a tuned model, and with what latent
    # size" is answerable without opening a multi-GB h5ad. Read backed, and
    # never allowed to fail the sample -- this is a convenience copy of
    # something already recorded authoritatively in .uns['qc']['solo'].
    extra = {}
    try:
        filtered = paths['qc'] / '{}_filtered.h5ad'.format(sample_name)
        if filtered.exists():
            adata = sc.read_h5ad(filtered, backed='r')
            solo = (adata.uns.get('qc') or {}).get('solo') or {}
            if solo.get('model_params'):
                extra['doublet_model'] = dict(solo['model_params'])
                extra['doublet_model']['exec_mode'] = solo.get('exec_mode', '')
                extra['doublet_model']['n_flagged'] = int(solo.get('n_flagged', 0) or 0)
            elif solo.get('skipped'):
                extra['doublet_model'] = {'skipped': True,
                                          'skip_reason': str(solo.get('skip_reason', ''))}
            if getattr(adata, 'file', None) is not None:
                adata.file.close()
    except Exception as e:
        logging.getLogger('SingleCellMappingPipeline').warning(
            '[MAIN] Could not read doublet model parameters for the manifest: {}'.format(e))

    write_manifest(
        sample_dir / 'manifest.json',
        sample=sample_name,
        mapper=mapper,
        batch=sample_data.get('batch'),
        base_dir=sample_dir,
        extra=extra,
        config_paths=[cfg['samples'], cfg['mapping'], cfg['qc'], cfg['filter']],
        artifacts={
            'filtered_h5ad': paths['qc'] / '{}_filtered.h5ad'.format(sample_name),
            'metadata_tsv': paths['qc'] / 'metadata.tsv',
            'mapping_stats': paths['qc'] / 'MappingStats.tsv',
            'qc_report': paths['qc'] / '{}_report.html'.format(sample_name)})


def build_required_index(samples: dict, index_config: dict, mapper_config: dict, force_list: list, work_dir: str=None):
    """Build every index this cohort's samples resolve to, if missing.

    Resolution goes through `shared/samples.py::resolve_index_ref`, the same path
    the mapper uses, so a sample-level `index_ref` is visible here too.

    Args:
        samples (dict): Resolved sample sheet.
        index_config (dict): index_config.yaml contents.
        mapper_config (dict): mapping_config.yaml contents.
        force_list (list[str]): Index names to rebuild unconditionally.
        work_dir (Path, optional): Scratch root for the build.
    """
    # main() already called setup_logger(); just fetch the configured logger by name
    # rather than reconfiguring it here.
    logger = logging.getLogger('SingleCellMappingPipeline')
    # Extract required index names
    required_indexes = set()
    

    for sample_name, sample_data in samples.items():
        # Shared with SingleCellMapper._resolve_index. This used to consult only
        # the mapper profile, so a sample carrying its own `index_ref:` mapped
        # against an index that --build-indexes never knew to build.
        index_ref = resolve_index_ref(sample_data, mapper_config)

        if index_ref:
            required_indexes.add(index_ref)

    # Build each index if missing
    for index_name in sorted(required_indexes):
        entry = index_config["indexes"].get(index_name)
        if not entry:
            logger.info('[INDEX] {} not found in index config, please check index config'.format(index_name))
            continue  # was `return`, which silently skipped every remaining index

        # Setup builder
        builder = IndexBuilder(gtf_path=entry['gtf'], genome_path=entry['genome'], out_path=entry['idx_path'], work_dir=work_dir)

        # Generate index
        force = index_name in force_list
        builder.build_if_missing(mapper=entry['mapper'], extra_args=entry.get('extra_index_args', []), force=force)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Run single-cell mapping pipeline')
    parser.add_argument('-c', '--config', required=True, help='Path to pipeline_config.yaml')
    parser.add_argument('--cohort',
                        help='Which declared cohort to run on (see `cohorts:` in the pipeline '
                             'config). Makes the dataset explicit on the command line instead '
                             'of it depending on `configs.samples`. Omit to use '
                             '`default_cohort:`. Rejected if the config declares no cohorts.')
    parser.add_argument('--build-indexes',action='store_true',help='Build required, but missing, indexes before mapping')
    parser.add_argument('--force-index-rebuild', nargs = '*', default=list(), help='List of indexes to force rebuild')
    parser.add_argument('--force-mapping', action = 'store_true', help='Force remapping samples')
    parser.add_argument('--force-preprocessing', action = 'store_true', help='force preprocessing')
    parser.add_argument('--force-plotting', action = 'store_true', help='force plotting')
    parser.add_argument('--force-qc', action = 'store_true', help='force qc metric computation')
    parser.add_argument('--force-mapextraction', action = 'store_true', help='force reextraction of mapping stats')
    parser.add_argument('--force-report', action = 'store_true', help='force regeneration of the HTML QC report')
    parser.add_argument('--force-doublets', action = 'store_true', help='force re-running per-sample doublet detection')
    parser.add_argument('--skip-validation', action = 'store_true', help='skip the config preflight (not recommended)')
    args = parser.parse_args()

    n_failed = main(config_path=args.config, cohort=args.cohort, build_indexes=args.build_indexes, force_list=args.force_index_rebuild, force_mapping=args.force_mapping, force_preprocessing=args.force_preprocessing, force_qc=args.force_qc, force_plotting=args.force_plotting, force_mapstats=args.force_mapextraction, force_report=args.force_report, force_doublets=args.force_doublets, skip_validation=args.skip_validation)
    sys.exit(1 if n_failed else 0)