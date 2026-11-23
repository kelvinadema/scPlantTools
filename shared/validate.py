#!/usr/bin/env python3

"""
Preflight validation of the config tree.

Two passes. The *structural* pass walks shared/config_spec.py and checks types,
enums and required keys within a single file -- the same spec the editor sees as
JSON Schema. The *cross-file* pass checks the things a per-file schema cannot
express, which is where this pipeline's expensive mistakes actually live: an
`index_ref` that names nothing, an index built for a different mapper, a
covariate whose name has drifted out of the `.obs` contract and is therefore
silently dropped at integrate.py:79 so the model trains without it.

The point is to fail in the first second rather than after CellBender has spent
hours on a GPU node. Everything here is read-only and cheap: no h5ad matrices
are loaded (obs column names come off the HDF5 header via h5py), and no
subprocess runs except an optional `piscem --version`.

Errors block the run. Warnings do not -- they cover states that are legitimate
mid-workflow, most importantly a missing index (which `--build-indexes` is
about to create, and which cached samples do not need at all).
"""

import json
import os
import shutil
import subprocess

from pathlib import Path

from .utils import load_config
from .samples import load_samples, resolve_index_ref, SampleConfigError
from .cohorts import resolve_cohort, CohortConfigError
from .targets import (expand_group, expand_sample_list, load_integration_config,
                      parse_batch_key, parse_covariates, resolve_batch_assignments,
                      TargetConfigError, parse_layer, expand_env_path)
from .io import _env_package_version
from .config_spec import FILTER_MODES
from . import config_spec as spec


class Report:
    """Collected findings. Errors block; warnings inform."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.notes = []

    def error(self, msg):
        """Record a finding that blocks the run."""
        self.errors.append(msg)

    def warn(self, msg):
        """Record a finding that informs but does not block."""
        self.warnings.append(msg)

    def note(self, msg):
        """Record an informational line."""
        self.notes.append(msg)

    @property
    def ok(self):
        """True when nothing blocking was recorded."""
        return not self.errors

    def render(self):
        """The findings as indented lines, or a single all-clear line."""
        lines = []
        for label, items in (('ERROR', self.errors), ('WARN', self.warnings), ('note', self.notes)):
            for item in items:
                lines.append('  [{}] {}'.format(label, item))
        if not lines:
            return '  All checks passed.'
        return '\n'.join(lines)


def _structural(field, data, label, report):
    """Run the config_spec check for one field and record any errors.

    Args:
        field: The `config_spec` field descriptor.
        data: The value to check.
        label (str): Prefix for the messages.
        report (Report): Where findings go.
    """
    for err in spec.check(field, data, label):
        report.error(err)


def _check_paths(paths, label, report, kind='file', severity='error'):
    """Report any path in `paths` that is missing or empty.

    Args:
        paths (iterable): Paths to check.
        label (str): Prefix for the messages.
        report (Report): Where findings go.
        kind (str): Noun used in the message ('file' or 'directory').
        severity (str): 'error' or 'warn'.
    """
    emit = report.warn if severity == 'warn' else report.error
    for p in paths or []:
        path = Path(str(p))
        if not path.exists():
            emit('{}: {} does not exist: {}'.format(label, kind, path))
        elif kind == 'file' and path.is_file() and path.stat().st_size == 0:
            emit('{}: {} is empty: {}'.format(label, kind, path))


def _read_slot_mismatches(fq1, fq2):
    """Flag reads whose filename says R1 but which sit in the R2 slot, or vice versa.

    Args:
        fq1 (list[str]): Paths in the R1 slot.
        fq2 (list[str]): Paths in the R2 slot.

    Returns:
        list[str]: One message per mismatched path.
    """
    problems = []
    for slot, paths, wrong, right in (('fastq2', fq2, '_R1', '_R2'), ('fastq1', fq1, '_R2', '_R1')):
        for p in paths or []:
            base = Path(str(p)).name
            if wrong in base and right not in base:
                problems.append('{} entry looks like a {} file: {}'.format(slot, wrong.strip('_'), base))
    return problems


def _whitelist_paths(extra_args):
    """Pull whitelist paths out of a mapper's extra args.

    The BD Rhapsody entry stores its three per-block whitelists space-joined in a
    single list element on purpose (BDDataPrepper reads `extra_args[idx + 1]` and
    takes their cartesian product), so the value is split before checking.

    Args:
        extra_args (list): The mapper's extra argument list.

    Returns:
        list[str]: Whitelist paths named there.
    """
    found = []
    args = list(extra_args or [])
    for i, arg in enumerate(args):
        if str(arg) in ('--unfiltered-pl', '--whitelist') and i + 1 < len(args):
            found.extend(str(args[i + 1]).split())
    return found


def _piscem_version():
    """The installed piscem's version string, or None if it is not on PATH."""
    if not shutil.which('piscem'):
        return None
    try:
        out = subprocess.run(['piscem', '--version'], capture_output=True, text=True, timeout=15)
        return (out.stdout or out.stderr).strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _check_index_freshness(name, idx_path, report):
    """Warn when an index on disk predates the installed piscem.

    Args:
        name (str): Index name.
        idx_path (Path): Index directory.
        report (Report): Where findings go.
    """
    path = Path(idx_path)
    if not path.is_dir():
        return

    has_old = (path / 'piscem_idx.sshash').exists()
    has_new = (path / 'piscem_idx.ssi').exists()
    if not (has_old and not has_new):
        return

    built_by = ''
    for info_name in ('index_info.json', 'simpleaf_index_log.json'):
        info = path / info_name
        if not info.exists():
            continue
        try:
            blob = json.dumps(json.load(open(info)))
        except (OSError, ValueError):
            continue
        if 'piscem' in blob:
            built_by = ' ({} mentions piscem)'.format(info_name)
            break

    installed = _piscem_version() or 'not on PATH'
    report.warn(
        "index '{}' at {} has piscem_idx.sshash but no piscem_idx.ssi, so it was built by an "
        'old piscem{}; installed piscem is {}. Mapping against it fails with "failed to load '
        'dictionary". Rebuild with --build-indexes --force-index-rebuild {} (hours), and note '
        'that rebuilding changes results for every sample mapped against it, including ones '
        'already cached -- those need --force-mapping too.'.format(
            name, path, built_by, installed, name))


def validate_mapping(pipeline_config_path, report=None, cohort=None):
    """Validate the mapping stage's whole config tree.

    Args:
        cfg_path (Path): Path to pipeline_config.yaml.
        report (Report, optional): Reuse an existing report.

    Returns:
        Report: The findings.
    """
    report = report or Report()
    pipeline_path = Path(pipeline_config_path)

    if not pipeline_path.exists():
        report.error('pipeline config does not exist: {}'.format(pipeline_path))
        return report

    pipeline_cfg = load_config(pipeline_path) or {}
    _structural(spec.PIPELINE_CONFIG, pipeline_cfg, pipeline_path.name, report)

    cfg = dict(pipeline_cfg.get('configs') or {})

    # Resolve the cohort exactly as main() will, so preflight checks the sheet
    # the run will actually use. See shared/cohorts.py.
    try:
        selected = resolve_cohort(pipeline_cfg, cohort)
        if selected['samples']:
            cfg['samples'] = selected['samples']
        if selected['name']:
            report.note('cohort {!r} -> {}'.format(selected['name'], selected['samples']))
    except CohortConfigError as e:
        report.error(str(e))
        return report
    if not cfg.get('samples'):
        report.error('{}: no sample sheet -- neither configs.samples nor a cohort '
                     'supplies one.'.format(pipeline_path.name))
        return report

    for key in ('samples', 'mapping', 'indexes', 'technologies', 'qc', 'filter'):
        sub = cfg.get(key)
        if sub and not Path(sub).exists():
            report.error('{}: configs.{} points at a file that does not exist: {}'.format(
                pipeline_path.name, key, sub))
    if report.errors:
        return report  # nothing below can be trusted

    mapper_cfg = load_config(cfg['mapping']) or {}
    index_cfg = load_config(cfg['indexes']) or {}
    tech_cfg = load_config(cfg['technologies']) or {}
    qc_cfg = load_config(cfg['qc']) or {}
    filter_cfg = load_config(cfg['filter']) or {}

    _structural(spec.MAPPING_CONFIG, mapper_cfg, Path(cfg['mapping']).name, report)
    _structural(spec.INDEX_CONFIG, index_cfg, Path(cfg['indexes']).name, report)
    _structural(spec.TECHNOLOGY_CONFIG, tech_cfg, Path(cfg['technologies']).name, report)
    _structural(spec.QC_CONFIG, qc_cfg, Path(cfg['qc']).name, report)
    _structural(spec.FILTER_CONFIG, filter_cfg, Path(cfg['filter']).name, report)

    _structural(spec.SAMPLE_SHEET, load_config(cfg['samples']) or {}, Path(cfg['samples']).name, report)

    try:
        samples = load_samples(cfg['samples'])
    except SampleConfigError as e:
        report.error(str(e))
        return report

    _validate_indexes(index_cfg, report,
                      referenced=_referenced_indexes(samples, mapper_cfg))
    _validate_samples(samples, mapper_cfg, index_cfg, tech_cfg, qc_cfg, filter_cfg,
                      pipeline_cfg.get('output_dir', 'results/mapping'), report)
    _validate_cellbender(pipeline_cfg.get('cellbender') or {}, report)
    _check_done_flag_outputs(samples, pipeline_cfg.get('output_dir', 'results/mapping'), report)
    _validate_filter_modes(filter_cfg, report)
    _validate_sample_type_caps(samples, filter_cfg, report)
    _validate_doublets(pipeline_cfg.get('doublets') or {}, filter_cfg, report,
                       sample_names=list(samples))

    return report


def _referenced_indexes(samples, mapper_cfg):
    """Index names this run's sample sheet actually resolves to.

    `index_config.yaml` is a catalogue carrying entries no current sample uses, and
    checking those produces warnings about files nobody needs built. Resolution goes
    through the same `resolve_index_ref` the sample validator uses, so the two
    cannot disagree. A sample whose index cannot be resolved is skipped here --
    `_validate_samples` already raises precisely for that.

    Args:
        samples (dict): Resolved sample sheet.
        mapper_cfg (dict): mapping_config.yaml contents.

    Returns:
        set[str]: Index names in use.
    """
    referenced = set()
    for entry in (samples or {}).values():
        try:
            name = resolve_index_ref(entry or {}, mapper_cfg)
        except Exception:
            continue
        if name:
            referenced.add(name)
    return referenced


def _validate_indexes(index_cfg, report, referenced=None):
    """Check index entries.

    Args:
        index_cfg (dict): index_config.yaml contents.
        report (Report): Where findings go.
        referenced (set[str], optional): Restrict to these; None checks the whole
            catalogue.
    """
    all_indexes = (index_cfg.get('indexes') or {})
    if referenced is not None:
        skipped = sorted(set(all_indexes) - referenced)
        if skipped:
            report.note('not checking {} index entr{} no sample in this run uses: {}'.format(
                len(skipped), 'y' if len(skipped) == 1 else 'ies', ', '.join(skipped)))
        all_indexes = {k: v for k, v in all_indexes.items() if k in referenced}

    for name, entry in all_indexes.items():
        entry = entry or {}
        genome = entry.get('genome') or []
        gtf = entry.get('gtf') or []
        label = 'index_config.indexes.{}'.format(name)

        _check_paths(genome, label, report)
        _check_paths(gtf, label, report)

        if genome and gtf and len(genome) != len(gtf):
            report.error(
                '{}: {} genome file(s) but {} gtf file(s); a multi-reference merge pairs them '
                'positionally, so the lists must be the same length.'.format(
                    label, len(genome), len(gtf)))

        idx_path = entry.get('idx_path')
        if idx_path:
            if not Path(idx_path).exists():
                # A warning, not an error: --build-indexes creates it, and a
                # sample whose mapping.done is already written never needs it.
                report.warn('{}: idx_path does not exist yet: {} (build it with '
                            '--build-indexes)'.format(label, idx_path))
            else:
                _check_index_freshness(name, idx_path, report)


def _check_done_flag_outputs(samples, output_dir, report):
    """Catch a `.done` flag whose outputs are no longer on disk.

    Caching is a sentinel file and nothing else: `mappingstats.done` means "this
    ran", not "its results exist". If the outputs are later removed, every
    subsequent run skips the stage and the report silently renders "figure not
    available" forever, with no indication that a `--force-*` would bring it back.

    Warns rather than errors: a missing figure is not a reason to refuse to run, and
    the fix is a flag the message names.

    Args:
        samples (dict): Resolved sample sheet.
        output_dir (Path): Mapping results root.
        report (Report): Where findings go.
    """
    expected = {
        'mappingstats.done': [('MappingStats.tsv', '--force-mapextraction'),
                              ('{sample}_ReadFlow.png', '--force-mapextraction')],
        'qcplot.done': [('{sample}_filtered.h5ad', '--force-plotting')],
        'report.done': [('{sample}_report.html', '--force-report')],
    }
    for sample, entry in (samples or {}).items():
        mapper = (entry or {}).get('mapper', '').lower()
        if not mapper:
            continue
        base = Path(output_dir) / sample / mapper
        for flag, wanted in expected.items():
            if not (base / 'done' / flag).exists():
                continue
            missing = [(name.format(sample=sample), fix) for name, fix in wanted
                       if not (base / 'qc' / name.format(sample=sample)).exists()]
            if missing:
                report.warn(
                    "{}: {} exists but {} missing from {}. The stage is cached as done, so it "
                    "will keep being skipped -- re-run with {} to regenerate.".format(
                        sample, flag, ', '.join(n for n, _ in missing), base / 'qc',
                        missing[0][1]))


def _already_mapped(sample, mapper, output_dir):
    """True once this sample's `mapping.done` sentinel exists.

    Used to decide severity, not to skip checks: a sample whose reads have been
    consumed and cached does not need its FASTQs still on disk, so a moved or
    archived file is a warning there and an error for a sample yet to run.

    Args:
        sample (str): Sample name.
        mapper (str): Mapper name.
        output_dir (Path): Mapping results root.

    Returns:
        bool: Whether mapping is cached as done.
    """
    return (Path(output_dir) / sample / mapper / 'done' / 'mapping.done').exists()


def _validate_samples(samples, mapper_cfg, index_cfg, tech_cfg, qc_cfg, filter_cfg,
                      output_dir, report):
    """Check every sample entry: paths, profiles, mapper gating and collisions.

    Args:
        samples (dict): Resolved sample sheet.
        mapper_cfg (dict): mapping_config.yaml contents.
        index_cfg (dict): index_config.yaml contents.
        tech_cfg (dict): technology_config.yaml contents.
        qc_cfg (dict): qc_config.yaml contents.
        filter_cfg (dict): filter_config.yaml contents.
        output_dir (Path): Mapping results root, for the cached-sample severity rule.
        report (Report): Where findings go.
    """
    gated = {}
    seen_fastqs = {}
    seen_batches = {}

    qc_profiles = set(qc_cfg.get('QCconfigs') or {})
    filter_profiles = set(filter_cfg.get('QCfilterconfigs') or {})

    for name, entry in samples.items():
        label = 'sample {}'.format(name)

        for key in spec.REQUIRED_RESOLVED_SAMPLE_KEYS:
            if not entry.get(key):
                report.error("{}: missing required key '{}' (set it on the sample or in the "
                             "sheet's defaults: block)".format(label, key))
        for key in spec.ADVISORY_RESOLVED_SAMPLE_KEYS:
            if not entry.get(key):
                report.warn("{}: no '{}' set; falling back to the code default".format(label, key))

        mapper = (entry.get('mapper') or '').lower()
        if mapper and mapper not in spec.KNOWN_MAPPERS:
            report.error('{}: unknown mapper {!r}; known mappers are {}'.format(
                label, entry.get('mapper'), ', '.join(spec.KNOWN_MAPPERS)))
            continue
        if mapper and mapper not in spec.IMPLEMENTED_MAPPERS:
            gated[name] = mapper
            continue

        # -- mapper profile and index -------------------------------------
        profile = entry.get('mapping_config', 'default')
        profiles = (mapper_cfg.get('mappers') or {}).get(mapper) or {}
        if profile not in profiles and not entry.get('index_ref'):
            report.error(
                '{}: mapper profile mappers.{}.{} not found in mapping_config.yaml (available '
                'for {}: {}). Add the profile, or set index_ref directly on the sample.'.format(
                    label, mapper, profile, mapper, ', '.join(sorted(profiles)) or 'none'))
        else:
            index_ref = resolve_index_ref(entry, mapper_cfg)
            indexes = index_cfg.get('indexes') or {}
            if not index_ref:
                report.error(
                    '{}: no index_ref -- neither the sample nor profile mappers.{}.{} supplies '
                    'one.'.format(label, mapper, profile))
            elif index_ref not in indexes:
                report.error(
                    "{}: index_ref '{}' is not defined in index_config.yaml (available: {})".format(
                        label, index_ref, ', '.join(sorted(indexes)) or 'none'))
            else:
                index_mapper = (indexes[index_ref] or {}).get('mapper')
                if index_mapper != mapper:
                    report.error(
                        "{}: uses mapper '{}' but index '{}' was built for '{}'. An index is not "
                        'portable between mappers; point the sample at an index whose mapper '
                        'matches.'.format(label, mapper, index_ref, index_mapper))

        # -- chemistry ------------------------------------------------------
        tech = (entry.get('tech') or '').lower()
        techs = tech_cfg.get('technologies') or {}
        if tech and tech not in techs:
            report.error("{}: tech '{}' is not defined in technology_config.yaml (available: "
                         '{})'.format(label, tech, ', '.join(sorted(techs)) or 'none'))
        elif tech:
            tech_entry = (techs.get(tech) or {}).get(mapper)
            if tech_entry is None:
                report.error("{}: technology_config.yaml has no '{}' entry under technologies.{}; "
                             'that mapper/chemistry pair is undefined.'.format(label, mapper, tech))
            elif tech_entry.get('unsupported'):
                report.error("{}: mapper '{}' is marked unsupported for technology '{}'.".format(
                    label, mapper, tech))
            else:
                _check_paths(_whitelist_paths(tech_entry.get('extra_mapping_args')),
                             '{} (whitelist)'.format(label), report)

        # -- QC profiles ----------------------------------------------------
        qc_profile = entry.get('qc_metrics', 'default')
        if qc_profiles and qc_profile not in qc_profiles:
            report.warn("{}: qc_metrics profile '{}' not in qc_config.yaml (available: {}); the "
                        'stage falls back to "default"'.format(
                            label, qc_profile, ', '.join(sorted(qc_profiles))))
        filter_profile = entry.get('qc_filter', 'default')
        if filter_profiles and filter_profile not in filter_profiles:
            report.warn("{}: qc_filter profile '{}' not in filter_config.yaml (available: {}); the "
                        'stage falls back to "default"'.format(
                            label, filter_profile, ', '.join(sorted(filter_profiles))))

        # -- reads ----------------------------------------------------------
        fq1 = entry.get('fastq1') or []
        fq2 = entry.get('fastq2') or []
        if len(fq1) != len(fq2):
            report.error('{}: {} R1 file(s) but {} R2 file(s); they are paired positionally.'.format(
                label, len(fq1), len(fq2)))

        for problem in _read_slot_mismatches(fq1, fq2):
            report.error('{}: {}'.format(label, problem))

        # Reads already consumed into a cached mapping need not still be on disk.
        severity = 'warn' if _already_mapped(name, mapper, output_dir) else 'error'
        _check_paths(fq1, label, report, severity=severity)
        _check_paths(fq2, label, report, severity=severity)

        within = [p for p in set(map(str, fq1)) & set(map(str, fq2))]
        for dup in sorted(within):
            report.error('{}: {} is listed in both fastq1 and fastq2.'.format(label, dup))
        for slot, paths in (('fastq1', fq1), ('fastq2', fq2)):
            strs = [str(p) for p in paths]
            for dup in sorted({p for p in strs if strs.count(p) > 1}):
                report.error('{}: {} is listed twice in {}.'.format(label, dup, slot))

        for fq in set(map(str, fq1)) | set(map(str, fq2)):
            seen_fastqs.setdefault(fq, []).append(name)
        if entry.get('batch'):
            seen_batches.setdefault(entry['batch'], []).append(name)

    if gated:
        report.error(
            'mapper(s) {} are not implemented end-to-end; only {} is supported. Samples: '
            '{}'.format(', '.join(sorted(set(gated.values()))),
                        ', '.join(spec.IMPLEMENTED_MAPPERS),
                        ', '.join('{}={}'.format(s, m) for s, m in sorted(gated.items()))))

    for fq, owners in sorted(seen_fastqs.items()):
        if len(owners) > 1:
            report.warn('FASTQ {} is listed by more than one sample ({}) -- likely a copy-paste '
                        'error in the sheet.'.format(fq, ', '.join(sorted(owners))))
    for batch, owners in sorted(seen_batches.items()):
        if len(owners) > 1:
            report.note("batch '{}' is shared by {} samples ({}) -- fine if deliberate, since "
                        'batch is an integration covariate.'.format(
                            batch, len(owners), ', '.join(sorted(owners))))


def _validate_cellbender(cellbender_cfg, report):
    """Check that cellbender.env is a usable conda prefix.

    Args:
        cellbender_cfg (dict): The `cellbender:` block.
        report (Report): Where findings go.
    """
    env = os.environ.get('CELLBENDER_ENV') or cellbender_cfg.get('env')
    if not env:
        report.warn('no cellbender.env configured; "cellbender" will be taken from PATH and its '
                    'output reader imported into this interpreter, which fails on Python 3.12.')
        return

    prefix = Path(env)
    if not prefix.is_dir():
        report.error('cellbender.env is not a directory: {} (it must be the conda PREFIX, the '
                     'directory containing bin/, not the binary).'.format(prefix))
        return
    for needed in ('bin/cellbender', 'bin/python'):
        if not (prefix / needed).exists():
            report.error('cellbender.env {} has no {}; both are derived from the prefix and must '
                         'exist.'.format(prefix, needed))


def _env_has_cuda(env: str) -> bool:
    """Whether a GPU is visible to the interpreter in `prefix`.

    Args:
        prefix (str): Conda prefix to probe.

    Returns:
        bool: True if torch there reports CUDA available.
    """
    python = Path(env) / 'bin' / 'python'
    if not python.exists():
        return False
    try:
        proc = subprocess.run(
            [str(python), '-c', 'import torch; print(int(torch.cuda.is_available()))'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and proc.stdout.decode('utf-8', 'replace').strip() == '1'


def _validate_filter_modes(filter_cfg, report):
    """Check every filter block's `mode` against FILTER_MODES.

    Args:
        filter_cfg (dict): filter_config.yaml contents.
        report (Report): Where findings go.
    """
    for profile_name, profile in (filter_cfg.get('QCfilterconfigs') or {}).items():
        if not isinstance(profile, dict):
            continue
        for filter_name, block in (profile.get('filters') or {}).items():
            if not isinstance(block, dict):
                continue
            label = 'filter_config.yaml: QCfilterconfigs.{}.filters.{}'.format(
                profile_name, filter_name)
            if 'enabled' in block:
                report.error(
                    '{}: `enabled:` is no longer read; it was replaced by `mode:` with values '
                    '{}. `enabled: true` becomes `mode: filter`, `enabled: false` becomes '
                    '`mode: off`. Leaving `enabled:` in place means this filter does not '
                    'run.'.format(label, ', '.join(repr(m) for m in FILTER_MODES)))
            mode = block.get('mode')
            if mode is None:
                report.error('{}: missing `mode:` (one of {}).'.format(
                    label, ', '.join(repr(m) for m in FILTER_MODES)))
            elif mode not in FILTER_MODES:
                report.error('{}: mode {!r} is not one of {}.'.format(
                    label, mode, ', '.join(repr(m) for m in FILTER_MODES)))


def _validate_sample_type_caps(samples, filter_cfg, report):
    """Cross-check each sample's sample_type against the caps it resolves against.

    A sample whose `sample_type` is not a key of a sample_type-keyed
    `max_value`/`min_value` mapping would have that cap silently unset -- and those
    absolute caps are the only filtering a flagged sample keeps. Scoped to the `mad`
    block, the only place the mapping is read. See docs/decisions/preflight.md.

    Args:
        samples (dict): Resolved sample sheet.
        filter_cfg (dict): filter_config.yaml contents.
        report (Report): Where findings go.
    """
    configs = filter_cfg.get('QCfilterconfigs') or {}
    if not configs:
        return
    for name, entry in (samples or {}).items():
        sample_type = (entry or {}).get('sample_type', 'protoplast')
        requested = (entry or {}).get('qc_filter', 'default')
        # Mirror the stage's own fallback: an unknown profile resolves to
        # 'default' (already warned about in _validate_samples).
        profile_name = requested if requested in configs else 'default'
        profile = configs.get(profile_name)
        if not isinstance(profile, dict):
            continue
        features = (((profile.get('filters') or {}).get('mad') or {}).get('features') or [])
        for feat_cfg in features:
            if not isinstance(feat_cfg, dict):
                continue
            fname = feat_cfg.get('name', '<unnamed>')
            for bound in ('max_value', 'min_value'):
                value = feat_cfg.get(bound)
                if isinstance(value, dict) and sample_type not in value:
                    report.error(
                        "sample {}: sample_type '{}' is not a key of the {} for MAD feature "
                        "'{}' in filter profile '{}' (it defines {}). The sample_type-keyed "
                        "absolute cap would be silently unset for this sample, leaving the "
                        "organelle backstop off -- add a '{}' entry to that cap in "
                        "filter_config.yaml, or correct the sample's sample_type.".format(
                            name, sample_type, bound, fname, profile_name,
                            sorted(value), sample_type))


def _validate_doublets(doublets_cfg, filter_cfg, report, sample_names=None):
    """Check the doublet-detection env, but only when a profile asks for it.

    Doublet calling is opt-in per filter profile (`solo.mode`), so an absent
    `doublets:` block is normal and must not warn. See docs/decisions/preflight.md.

    Args:
        doublets_cfg (dict): The `doublets:` block.
        filter_cfg (dict): filter_config.yaml contents.
        report (Report): Where findings go.
        sample_names (iterable, optional): Used to size the tuning-cost warning.
    """
    wanted = any(
        (profile.get('filters') or {}).get('solo', {}).get('mode', 'off') != 'off'
        for profile in (filter_cfg.get('QCfilterconfigs') or {}).values()
        if isinstance(profile, dict))
    if not wanted:
        return

    env = os.environ.get('DOUBLET_ENV') or doublets_cfg.get('env')
    if not env:
        report.error('a filter profile enables solo (doublet detection) but no doublets.env is '
                     'configured. scvi-tools cannot be installed alongside the mapping stage '
                     '(pandas 3.x), so the doublet worker needs its own conda PREFIX here.')
        return

    prefix = Path(env)
    if not prefix.is_dir():
        report.error('doublets.env is not a directory: {} (it must be the conda PREFIX, the '
                     'directory containing bin/, not the binary).'.format(prefix))
        return
    if not (prefix / 'bin/python').exists():
        report.error('doublets.env {} has no bin/python.'.format(prefix))
        return

    # Confirm scvi-tools is actually importable there rather than discovering it
    # after mapping and CellBender have run.
    if not _env_package_version(str(prefix), 'scvi-tools'):
        report.error('doublets.env {} has no importable scvi-tools; doublet detection would fail '
                     'after mapping has already run.'.format(prefix))
        return

    # A GPU-less node is the failure this catches. The worker asks Lightning for
    # a GPU accelerator and dies immediately when there is none -- per sample,
    # after mapping and CellBender have already spent the time, with the real
    # reason buried in the worker's stderr. Ask once, up front, in the env that
    # will actually run it.
    # Autotuning is per sample, so its cost scales with the cohort rather than
    # being a fixed overhead. Worth saying out loud before a 40-sample run
    # rather than discovering it from the wall clock.
    if doublets_cfg.get('tune', False):
        n_samples = len(sample_names or ())
        trials = doublets_cfg.get('n_trials', 20)
        report.warn(
            'doublets.tune is on: hyperparameter search runs PER SAMPLE ({} trials each{}). '
            'This is a much longer job than the fixed-parameter default, and the models are '
            'discarded afterwards -- they exist only to produce doublet calls. Consider tuning '
            'only the samples you care about.'.format(
                trials, ' x {} samples'.format(n_samples) if n_samples else ''))

    if doublets_cfg.get('use_cuda', True) and not _env_has_cuda(str(prefix)):
        report.error(
            'doublets.use_cuda is on but no GPU is visible to {}. Doublet detection would fail '
            'for every sample. Submit to a GPU node (CellBender needs one anyway), or set '
            'doublets.use_cuda: false to train on CPU -- much slower, and worth reserving for '
            'small cohorts.'.format(prefix))


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def _obs_columns(input_dir, samples):
    """Collect `.obs` column names per mapped sample.

    Every mapped sample is read, not just the first: the column set genuinely
    differs between them, since a filter that skipped writes no flag column. Read
    off the HDF5 header via h5py, so no matrix is loaded.

    Args:
        input_dir (Path): Mapping results root.
        samples (dict): Resolved sample sheet.

    Returns:
        dict: {sample: set[str]}.
    """
    try:
        import h5py
    except ImportError:
        return None, 'h5py not importable'

    per_sample = {}
    for sample in samples:
        manifest_path = Path(input_dir) / sample / 'manifest.json'
        if not manifest_path.exists():
            continue
        try:
            manifest = json.load(open(manifest_path))
            rel = (manifest.get('artifacts') or {}).get('filtered_h5ad')
            if not rel:
                continue
            h5ad = Path(input_dir) / sample / rel
            if not h5ad.exists():
                continue
            with h5py.File(h5ad, 'r') as f:
                cols = [c.decode() if isinstance(c, bytes) else str(c)
                        for c in f['obs'].attrs.get('column-order', [])]
            if cols:
                per_sample[sample] = set(cols)
        except (OSError, ValueError, KeyError):
            continue

    if not per_sample:
        return None, 'no mapped sample found to read .obs from'
    return per_sample, '{} mapped sample(s)'.format(len(per_sample))


def _target_samples(entry, groups):
    """The sample names one integration target resolves to.

    Args:
        target (dict): A target entry.
        groups (dict): The `groups:` block, for expansion.

    Returns:
        list[str]: Resolved sample names.
    """
    out = []
    for key in ('Samples', 'Reference', 'Query'):
        try:
            out.extend(expand_sample_list(entry.get(key), groups) or [])
        except TargetConfigError:
            continue  # cycle already reported by the caller
    return out


def validate_integration(integration_config_path, sample_config_path, report=None):
    """Validate integration_config.yaml against the sample sheet and the .obs contract.

    Args:
        icfg_path (Path): Path to integration_config.yaml.
        scfg_path (Path): Path to the sample sheet.
        report (Report, optional): Reuse an existing report.

    Returns:
        Report: The findings.
    """
    report = report or Report()
    icfg_path = Path(integration_config_path)

    if not icfg_path.exists():
        report.error('integration config does not exist: {}'.format(icfg_path))
        return report

    icfg = load_integration_config(icfg_path) or {}
    _structural(spec.INTEGRATION_CONFIG, icfg, icfg_path.name, report)

    try:
        samples = load_samples(sample_config_path)
    except SampleConfigError as e:
        report.error(str(e))
        return report

    targets = icfg.get('IntegrateConfigs') or {}
    groups = icfg.get('groups') or {}
    defaults = icfg.get('defaults') or {}
    input_dir = icfg.get('input_dir', 'results/mapping')

    for gname in groups:
        try:
            members = expand_group(gname, groups)
        except TargetConfigError as e:
            report.error('integration groups: {}'.format(e))
            continue
        unknown = [m for m in members if m not in samples]
        if unknown:
            report.error("integration group '{}' names sample(s) not in the sample sheet: "
                         '{}'.format(gname, ', '.join(sorted(set(unknown)))))

    obs_per_sample, obs_source = _obs_columns(input_dir, list(samples))
    obs_union = set()
    if obs_per_sample is None:
        report.note('covariate names not checked against real .obs ({}).'.format(obs_source))
    else:
        obs_union = set().union(*obs_per_sample.values())
        report.note('covariate names checked against the union of .obs across {} ({} distinct '
                    'columns).'.format(obs_source, len(obs_union)))

    unmapped_seen = set()

    for tname, raw in targets.items():
        entry = {**defaults, **(raw or {})}
        label = "integration target '{}'".format(tname)

        method = entry.get('method')
        if not method:
            report.error("{}: no 'method' defined.".format(label))
        elif method not in spec.INTEGRATION_METHODS:
            report.error('{}: unsupported method {!r}; must be one of {}.'.format(
                label, method, ', '.join(spec.INTEGRATION_METHODS)))

        if method == 'scArches':
            ref_model = entry.get('Reference_model')
            if not ref_model:
                report.error("{}: method='scArches' requires 'Reference_model'.".format(label))
            elif ref_model not in targets:
                report.error("{}: Reference_model '{}' is not a target in this file (available: "
                             '{}).'.format(label, ref_model, ', '.join(sorted(targets))))
            elif {**defaults, **(targets[ref_model] or {})}.get('method') != 'scVI':
                report.error("{}: Reference_model '{}' is not a method='scVI' target; only an "
                             'scVI target has a trained model to map onto.'.format(label, ref_model))
            if not entry.get('Query'):
                report.error("{}: method='scArches' requires 'Query'.".format(label))
        elif method == 'contrastiveVI':
            background = entry.get('Background')
            target_side = entry.get('Target')
            if not background:
                report.error("{}: method='contrastiveVI' requires 'Background' (the control "
                             'samples defining shared variation).'.format(label))
            if not target_side:
                report.error("{}: method='contrastiveVI' requires 'Target' (the perturbed "
                             'samples).'.format(label))
            if background and target_side:
                try:
                    bg = set(expand_sample_list(background, groups) or [])
                    tg = set(expand_sample_list(target_side, groups) or [])
                except TargetConfigError:
                    bg, tg = set(), set()   # the group error is reported below
                overlap = sorted(bg & tg)
                if overlap:
                    report.error(
                        '{}: sample(s) {} appear in BOTH Background and Target. The same cells '
                        'cannot be the shared baseline and the perturbation at once -- this '
                        'trains without error and silently destroys the contrast.'.format(
                            label, ', '.join(overlap)))
            # Refused rather than silently ignored, and refused here rather than
            # at runtime so it costs no GPU time. contrastiveVI maximizes a sum
            # of a target ELBO and a background ELBO, with no term rewarding
            # salient/background separation: raising n_salient_latent strictly
            # improves target reconstruction, and a nonzero wasserstein_penalty
            # strictly worsens the objective. A search on elbo_validation
            # therefore selects maximum salient capacity and zero penalty --
            # exactly the leak that penalty exists to prevent.
            if entry.get('tune'):
                report.error(
                    '{}: tune is on, but a contrastiveVI target must not be autotuned. Its '
                    'objective contains no term rewarding salient/background separation, so '
                    'selecting on elbo_validation picks the largest salient space and a zero '
                    'Wasserstein penalty -- the degenerate fit. Set `tune: false` on this '
                    "target (this file's `defaults:` block turns it on) and rank a "
                    'n_salient_latent x wasserstein_penalty grid with '
                    'integration.contrastive.contrast_separation_score instead. '
                    '`early_stopping: true` is the supported use of elbo_validation '
                    'here.'.format(label))
            # Warned, not errored: the target still trains and its result is
            # valid -- it is just not corrected for what the config implies.
            # Checked statically here because this module must stay importable
            # without the integration env, so it cannot probe scvi directly;
            # integration.contrastive.module_supports_covariates() does
            # the version-aware probe at run time and drops them there.
            grid = entry.get('sweep_grid')
            if grid is not None and not isinstance(grid, dict):
                report.error("{}: 'sweep_grid' must be a mapping of parameter name to a list "
                             'of values, e.g. {{wasserstein_penalty: [0, 0.1]}}; got '
                             '{}.'.format(label, type(grid).__name__))
            elif grid:
                unknown = sorted(set(grid) - {'wasserstein_penalty', 'n_salient_latent',
                                              'n_background_latent', 'n_hidden', 'n_layers',
                                              'dropout_rate'})
                if unknown:
                    report.error(
                        "{}: 'sweep_grid' names parameter(s) ContrastiveVI does not take: {}. "
                        'Its constructor accepts n_hidden, n_background_latent, '
                        'n_salient_latent, n_layers, dropout_rate, use_observed_lib_size and '
                        'wasserstein_penalty.'.format(label, ', '.join(unknown)))
                for key, values in grid.items():
                    if not isinstance(values, (list, tuple)) or not values:
                        report.error("{}: sweep_grid['{}'] must be a non-empty list of "
                                     'values.'.format(label, key))
                if not entry.get('sweep'):
                    report.warn(
                        "{}: 'sweep_grid' is set but 'sweep' is not true, so the grid is "
                        'ignored and this target trains once at its stated model_params.'.format(
                            label))

            cov = entry.get('covariates')
            if isinstance(cov, dict):
                named = list(cov.get('categorical') or []) + list(cov.get('continuous') or [])
            else:
                named = list(cov or [])
            if named:
                report.warn(
                    '{}: {} covariate(s) named ({}) but contrastiveVI DOES NOT APPLY THEM. '
                    'scvi-tools registers covariate keys on this model and its module then '
                    'never reads them -- ContrastiveVAE takes no covariate dimensions and sees '
                    'only batch -- so they correct nothing while appearing in the saved '
                    "model's setup_args as though they had. Verified against 1.5.0.post1; the "
                    'run drops them and logs a warning. Remove them from this target, and put '
                    'the one nuisance variable you can correct in `batch_key`. For the rest, '
                    'check post hoc (correlate salient dimensions against them) or raise '
                    '`wasserstein_penalty`.'.format(label, len(named), ', '.join(named)))
        elif entry.get('sweep') or entry.get('sweep_grid'):
            # Refused rather than ignored: `sweep` on an scVI target reads as
            # "tune this", and silently doing nothing would leave someone
            # believing a search ran. scVI/DRVI have real autotune -- `tune:
            # true` with `n_trials` -- and that is what they should use.
            report.error(
                "{}: 'sweep'/'sweep_grid' is a contrastiveVI-only key, but this target's "
                "method is '{}'. scVI and DRVI targets autotune through ray with `tune: true` "
                'and `n_trials`; the sweep exists only because contrastiveVI cannot be '
                'autotuned on its own objective.'.format(label, method))

        if method == 'scVI' and not entry.get('Samples'):
            report.error("{}: method='scVI' requires 'Samples'.".format(label))

        for key in ('Samples', 'Reference', 'Query', 'Background', 'Target'):
            listed = entry.get(key)
            if listed is None:
                continue
            try:
                expanded = expand_sample_list(listed, groups) or []
            except TargetConfigError as e:
                report.error('{}: {}'.format(label, e))
                continue
            missing = [s for s in expanded if s not in samples]
            if missing:
                report.error('{}: {} names sample(s) not in the sample sheet: {}'.format(
                    label, key, ', '.join(sorted(set(missing)))))
            for s in expanded:
                if s in samples and s not in unmapped_seen:
                    if not (Path(input_dir) / s / 'manifest.json').exists():
                        unmapped_seen.add(s)

        # An unknown batch_key is not an error at runtime -- the model would
        # train with NO batch correction while the config says otherwise -- so
        # the typo is caught here. The stamped columns are accepted without
        # checking obs_union because the orchestrator writes them during
        # concatenation; the PARTIAL ones are checked explicitly below, since a
        # target mixing declaring and non-declaring samples yields NaN for the
        # rest and the correction silently means something else. See
        # docs/decisions/batch-keys-and-confounds.md.
        _STAMPED_OBS = ('batch', 'sample', 'tech', 'sample_type', 'prep', 'treatment', 'time_h')
        _PARTIAL_STAMPED_OBS = ('prep', 'treatment', 'time_h')

        # Label-space keys (2026-09-10; docs/decisions/two-layer-label-spaces.md).
        if 'layer' in entry:
            try:
                parse_layer(entry['layer'])
            except TargetConfigError as e:
                report.error('{}: {}'.format(label, e))
        hvg_bk = entry.get('hvg_batch_key')
        if hvg_bk and obs_per_sample is not None and hvg_bk not in obs_union \
                and hvg_bk not in _STAMPED_OBS:
            report.error(
                "{}: hvg_batch_key '{}' is not an .obs column of any mapped sample, nor one "
                'stamped at concatenation ({}); HVGSelector would raise on it.'.format(
                    label, hvg_bk, ', '.join(_STAMPED_OBS)))
        if entry.get('exclude_genes'):
            try:
                gene_list = expand_env_path(entry['exclude_genes'], 'exclude_genes')
            except TargetConfigError as e:
                report.error('{}: {}'.format(label, e))
            else:
                if not Path(gene_list).is_file():
                    report.error('{}: exclude_genes file does not exist: {}'.format(
                        label, gene_list))
        if 'batch_key' in entry:
            # One string, a list to be combined, or a per-cohort mapping -- all
            # three reduce to "these keys, over these samples", and every key is
            # then held to exactly the checks a lone batch_key always faced. A
            # combined key is not a way to smuggle `treatment` into a batch
            # column by pairing it with something legal.
            try:
                batch_spec = parse_batch_key(entry.get('batch_key'), groups)
            except TargetConfigError as exc:
                report.error('{}: {}'.format(label, exc))
                batch_spec = None

            target_samples = []
            for key in ('Samples', 'Reference', 'Query', 'Background', 'Target'):
                listed = entry.get(key)
                if listed is None:
                    continue
                try:
                    expanded = expand_sample_list(listed, groups) or []
                except TargetConfigError:
                    continue
                target_samples += [s for s in expanded if s not in target_samples]

            if batch_spec is not None and batch_spec['mode'] == 'mapped':
                try:
                    resolve_batch_assignments(batch_spec, target_samples)
                except TargetConfigError as exc:
                    report.error('{}: {}'.format(label, exc))

            for batch_key in (batch_spec or {}).get('components', ()):
                # Which samples this key actually has to hold for. For the mapped
                # form that is only the samples it was named for -- requiring
                # `prep` on a tip sample corrected by `sample` would be a false
                # error.
                if batch_spec['mode'] == 'mapped':
                    applies_to = [s for s in target_samples
                                  if batch_key in (batch_spec['assignments'].get(s) or ())]
                else:
                    applies_to = target_samples

                if batch_key in _PARTIAL_STAMPED_OBS:
                    undeclared = [s for s in applies_to
                                  if s in samples and (samples[s] or {}).get(batch_key) is None]
                    if undeclared:
                        report.error(
                            "{}: batch_key '{}' is only stamped onto samples that declare it in "
                            'the sample sheet, and {} do(es) not: {}. Those cells would carry a '
                            'missing value for the batch column, so the correction would not be '
                            'the one this target describes. Declare it on every sample in the '
                            'target, or use an explicit null.'.format(
                                label, batch_key, len(set(undeclared)),
                                ', '.join(sorted(set(undeclared)))))

                if obs_per_sample is None:
                    continue
                if batch_key not in obs_union and batch_key not in _STAMPED_OBS:
                    report.error(
                        "{}: batch_key '{}' is not an .obs column of any mapped sample, and is "
                        'not one of the columns the integration stage stamps itself ({}). An '
                        'unknown batch_key is dropped rather than raising, so the model would '
                        'train with no batch correction while this config says '
                        'otherwise.'.format(label, batch_key, ', '.join(_STAMPED_OBS)))
                elif batch_key in ('batch', 'sample'):
                    # 'batch' and 'sample' are the same variable in practice --
                    # `batch` is 1:1 with `sample` on every thesis cohort. A note
                    # rather than an error because per-sample correction is the
                    # RIGHT choice on an untreated reference atlas, where there
                    # is no treatment effect for it to delete; it is only fatal
                    # on a treated cohort. The validator cannot tell which
                    # without deciding what counts as treated, so it says so and
                    # leaves the call to the target's author.
                    treated = sorted({s for s in applies_to
                                      if (samples.get(s) or {}).get('treatment')
                                      not in (None, 'none')})
                    if treated:
                        report.note(
                            "{}: batch_key '{}' is one-to-one with the experimental condition on "
                            'the {} treated sample(s) it covers ({}). Correcting on it removes '
                            'the effect being measured there -- legitimate only if this cohort is '
                            'a reference whose treatment structure is not being read. Prefer '
                            "'tech'/'prep' where one is crossed with condition, or an explicit "
                            'null.'.format(label, batch_key, len(treated),
                                           ', '.join(treated[:6])))
                elif batch_key in ('treatment', 'time_h'):
                    report.error(
                        "{}: batch_key is '{}', which is a design variable, not a nuisance "
                        'one. Correcting on it removes exactly the effect the target exists '
                        'to measure -- the same failure as batch_key: batch, but explicit. '
                        "Use 'tech' or 'prep' where one is crossed with condition, or an "
                        'explicit null.'.format(label, batch_key))

        covariates = entry.get('covariates')
        if covariates and obs_per_sample is not None:
            if isinstance(covariates, str):
                covariates = [covariates]
            # Both accepted shapes reduce to one flat list of column names. The
            # object form ({categorical: [...], continuous: [...]}) was rejected
            # by config_spec until 2026-08-17; walking it as a plain list here
            # checked for .obs columns literally named 'categorical'.
            if isinstance(covariates, dict):
                parsed = parse_covariates(covariates)
                covariates = list(parsed['categorical']) + list(parsed['continuous'])
            for cov in covariates:
                # Stamped columns are written onto .obs at concatenation time
                # (orchestrator.py:210-225) and are legitimately absent from
                # every per-sample h5ad. Held to the same allowance batch_key
                # already gets a few lines above.
                if cov in _STAMPED_OBS:
                    continue
                if cov not in obs_union:
                    report.error(
                        "{}: covariate '{}' is not an .obs column of any mapped sample. scVI "
                        'covariates are filtered to those present at integrate.py:79, so it '
                        'would be silently dropped and the model would train without it.'.format(
                            label, cov))
                    continue
                # Present somewhere but not everywhere: the targets that include
                # a sample lacking it are the ones that lose it.
                lacking = sorted(s for s, cols in obs_per_sample.items() if cov not in cols)
                targeted = [s for s in _target_samples(entry, groups) if s in lacking]
                if targeted:
                    report.warn(
                        "{}: covariate '{}' is missing from the .obs of {}, which this target "
                        'includes; a filter that skipped on those samples writes no column, and '
                        'the covariate is dropped rather than erroring.'.format(
                            label, cov, ', '.join(targeted)))

    if unmapped_seen:
        report.warn('sample(s) referenced by integration targets have no manifest.json under {} '
                    'and have therefore not completed mapping: {}'.format(
                        input_dir, ', '.join(sorted(unmapped_seen))))

    return report


def emit_schemas(out_dir):
    """Regenerate `configs/schemas/*.schema.json` from config_spec.

    Args:
        out_dir (Path, optional): Where to write; defaults to configs/schemas.

    Returns:
        list[Path]: The files written.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for stem, (field, title) in sorted(spec.SPECS.items()):
        path = out / '{}.schema.json'.format(stem)
        path.write_text(json.dumps(spec.to_json_schema(field, title), indent=2) + '\n')
        written.append(path)
    return written
