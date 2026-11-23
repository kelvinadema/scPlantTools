#!/usr/bin/env python3

"""
Apply a human's manual QC decisions, producing a second filtered artifact.

    python -m mapping.refilter -c configs/pipeline_config.yaml --sample 2_tip

Reads `{sample}_decisions.json` (written by the interactive QC report's export
button), applies it on top of the automatic cascade, and writes
`{sample}_filtered_manual.h5ad` alongside the automatic `{sample}_filtered.h5ad`
-- never over it. Integration then picks which one to use via
`--artifact-key filtered_manual_h5ad`.

Three properties this is built around:

**Decisions are undoable.** The file is the complete state, not a delta, and
every barcode in it refers to `{sample}_qcfilters.h5ad` -- the pre-filter object
holding every cell that entered QC, flags and all. Undo is deleting an entry and
re-running; the result depends only on the file's current contents, never on
what was run before. That also means decisions can *rescue* cells the automatic
cascade removed, not just remove more, which matters because the automatic
filters are sometimes the ones that are wrong (see the MAD degeneracy trap).

**Re-running is keyed on the decisions, not a sentinel.** The decision file is an
input, so there is no `.done` flag: the run is skipped when the existing manual
artifact already carries a matching decision hash, and re-runs by itself the
moment the file is edited.

**The parent is checked.** A `--force-qc` rerun that regenerates
`_qcfilters.h5ad` invalidates a manual artifact derived from the old one, so the
parent's config hash is recorded and compared rather than silently ignored.
"""

import argparse
import hashlib
import json
import logging
import sys

import numpy as np
import pandas as pd
import scanpy as sc

from pathlib import Path

from shared.utils import load_config, setup_logger, get_sample_output_paths, pipeline_log_path
from shared.samples import resolve_sample
from shared.io import write_h5ad, write_manifest, PROVENANCE_KEY
from .qc_filter import collect_removal_flags
from .interactive import _automatic_bounds

DECISION_SCHEMA_VERSION = '1.0'

# Recognised `kind` values in a decision file. Anything else is an error rather
# than a no-op: a typo'd kind that silently did nothing would let the user
# believe a decision was applied when it was not.
_KINDS = ('threshold', 'exclude_barcodes', 'rescue_barcodes')


class DecisionError(ValueError):
    """Raised for a decision file that cannot be applied as written."""


def decision_hash(decisions: dict) -> str:
    """
    Stable hash of a decision file's meaning.

    Covers each decision with keys sorted and its `note` and `source_plot`
    fields left out, plus the file's `parent_artifact`; no other file-level
    field enters the hash. So re-running after a comment change is a no-op,
    while any change to what is actually filtered forces a rerun.
    """
    payload = {
        'decisions': sorted(
            (json.dumps({k: v for k, v in d.items()
                         if k not in ('note', 'source_plot')}, sort_keys=True)
             for d in decisions.get('decisions', []))),
        'parent_artifact': decisions.get('parent_artifact', ''),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class ManualRefilter():
    """Apply a decision file to one sample's pre-filter object."""

    def __init__(self, sample_name: str, pipeline_config_path: Path, decisions_path: Path = None):
        self.logger = logging.getLogger('SingleCellMappingPipeline')
        self.sample_name = sample_name

        self.pipeline_cfg = load_config(pipeline_config_path) or {}
        cfg = self.pipeline_cfg.get('configs') or {}
        if 'samples' not in cfg:
            raise DecisionError(
                "{} has no configs.samples entry; cannot resolve which sample sheet "
                "'{}' lives in.".format(pipeline_config_path, sample_name))
        self.sample_config = cfg['samples']
        self.config_paths = [pipeline_config_path, self.sample_config, cfg.get('filter')]

        self.logger.info('[REFILTER] "{}": resolving against sample sheet {}'.format(
            sample_name, self.sample_config))
        entry = resolve_sample(sample_name, self.sample_config)
        self.mapper = entry.get('mapper', '').lower()
        self.batch = entry.get('batch')

        out_root = Path(self.pipeline_cfg.get('output_dir', 'results/mapping'))
        self.sample_dir = out_root / sample_name
        self.paths = get_sample_output_paths(self.sample_dir, mapper=self.mapper)
        self.qc_dir = self.paths['qc']
        self.parent_path = self.paths['processed'] / '{}_qcfilters.h5ad'.format(sample_name)
        self.output_path = self.qc_dir / '{}_filtered_manual.h5ad'.format(sample_name)
        self.decisions_path = Path(decisions_path) if decisions_path else (
            self.qc_dir / '{}_decisions.json'.format(sample_name))

    # -- decisions ----------------------------------------------------------

    def _load_decisions(self) -> dict:
        """Read and validate the decision file.

        Raises DecisionError if it is absent, the wrong schema major version, for
        another sample, or carries an unknown decision kind (see `_KINDS`). An
        unknown kind is an error rather than a no-op, so a typo cannot silently
        drop a decision.
        """
        if not self.decisions_path.exists():
            raise DecisionError(
                "No decision file at {}. Open the sample's QC report, make your selections, and "
                "use its Export button -- or pass --decisions <path>.".format(self.decisions_path))
        with open(self.decisions_path) as f:
            decisions = json.load(f)

        version = str(decisions.get('schema_version', ''))
        if version and version.split('.')[0] != DECISION_SCHEMA_VERSION.split('.')[0]:
            raise DecisionError(
                'Decision file {} is schema {}, this build reads {}.'.format(
                    self.decisions_path, version, DECISION_SCHEMA_VERSION))
        if decisions.get('sample') and decisions['sample'] != self.sample_name:
            raise DecisionError(
                "Decision file {} is for sample '{}', not '{}'.".format(
                    self.decisions_path, decisions['sample'], self.sample_name))

        for i, entry in enumerate(decisions.get('decisions', [])):
            kind = entry.get('kind')
            if kind not in _KINDS:
                raise DecisionError(
                    'decisions[{}] has kind {!r}; expected one of {}. (An unrecognised kind is an '
                    'error rather than a no-op, so a typo cannot silently drop a '
                    'decision.)'.format(i, kind, ', '.join(repr(k) for k in _KINDS)))
        return decisions

    def _check_parent(self, adata, decisions: dict, allow_mismatch: bool = False) -> None:
        """Raise DecisionError when the decisions were made against a different
        parent config hash; only warn when `allow_mismatch`."""
        recorded = decisions.get('parent_config_hash')
        current = (adata.uns.get(PROVENANCE_KEY) or {}).get('config_hash')
        if not (recorded and current and recorded != current):
            return

        detail = (
            "Decision file was made against a different version of {parent}: it records parent "
            "config_hash {recorded} but the file on disk is {current}. This hash is over the "
            "*config file contents* that produced the QC object, so it differs whenever any "
            "config feeding QC has changed on disk -- which usually means QC was re-run and the "
            "barcodes and thresholds in the decision file may no longer mean what they did.".format(
                parent=self.parent_path, recorded=recorded, current=current))

        if allow_mismatch:
            self.logger.warning(
                '[REFILTER] {detail}\n'
                'Proceeding because --allow-parent-mismatch was given. This is only safe if the '
                'QC *data* in {parent} is unchanged (e.g. only the config file contents drifted, '
                'or QC was re-run with identical parameters) -- the decisions are being applied to '
                'the current object regardless. If QC was actually re-run with different filtering, '
                'the resulting artifact will not match what you reviewed in the report.'.format(
                    detail=detail, parent=self.parent_path))
            return

        raise DecisionError(
            '{detail} Re-export from the current report, or delete {decisions}. If you are certain '
            'the QC data is unchanged (only the report layout or an unrelated config changed), pass '
            '--allow-parent-mismatch to apply the decisions anyway.'.format(
                detail=detail, decisions=self.decisions_path))

    # -- application --------------------------------------------------------

    def _apply_threshold(self, adata, entry: dict, mask: np.ndarray) -> dict:
        """
        Flag cells outside a manual bound on one .obs feature.

        Bounds are in the feature's own raw units. Several features are
        thresholded by the automatic cascade in transformed space
        (`log_`/`logit_` columns), but a human setting a cut-off reads and
        thinks in the raw fraction or count, so the decision file records raw
        values and this applies them to the raw column.
        """
        feature = entry.get('feature')
        if feature not in adata.obs:
            raise DecisionError(
                "threshold decision names .obs column '{}', which is not present. Available "
                "columns are listed in the report's filter table.".format(feature))

        values = pd.to_numeric(adata.obs[feature], errors='coerce').values
        op = entry.get('op', 'max')
        bound = float(entry['value'])
        if op == 'max':
            flagged = values > bound
        elif op == 'min':
            flagged = values < bound
        else:
            raise DecisionError(
                "threshold decision on '{}' has op {!r}; expected 'max' or 'min'.".format(feature, op))
        # NaN compares False either way, which is what we want: a cell with no
        # value for this feature was not measured, not out of bounds.
        flagged = np.where(np.isnan(values), False, flagged)
        mask |= flagged
        return {'kind': 'threshold', 'feature': feature, 'op': op, 'value': bound,
                'n_flagged': int(flagged.sum())}

    def _apply_barcodes(self, adata, entry: dict, mask: np.ndarray, remove: bool) -> dict:
        """Apply one barcode-list decision to the removal mask, in place.

        A barcode the parent no longer has is warned about rather than fatal -- a
        decision file can legitimately outlive a re-run that changed which barcodes
        exist -- but it must be visible, since the selection is not being fully applied.

        `remove` selects exclusion versus rescue. Returns what was applied, for
        the run record.
        """
        barcodes = [str(b) for b in (entry.get('barcodes') or [])]
        index = pd.Index(adata.obs_names.astype(str))
        hits = index.isin(barcodes)
        n_unknown = len(set(barcodes) - set(index))
        if n_unknown:
            # Not fatal -- a decision file can outlive a re-run that changed
            # which barcodes exist -- but visible, since the selection is not
            # fully applied.
            self.logger.warning(
                '[REFILTER] {} of {} barcodes in a {} decision are not in {}; ignoring '
                'them.'.format(n_unknown, len(barcodes), entry.get('kind'), self.parent_path.name))
        if remove:
            mask |= hits
        else:
            mask &= ~hits
        return {'kind': entry.get('kind'), 'n_barcodes': len(barcodes),
                'n_matched': int(hits.sum()), 'n_unknown': n_unknown,
                'note': entry.get('note', '')}

    def _check_unassessed(self, adata, decisions: dict, mask_before, allow: bool) -> None:
        """
        Refuse decisions that admit cells the automatic cascade removed.

        Filters run as a cascade, so a cell the cascade dropped was never in the
        view of anything downstream of the filter that dropped it: its
        `flag_mad_*` and friends are `False` **by default**, meaning "never
        tested", not "passed". Masking such a cell back in produces an object
        where some cells went through the whole cascade and others through
        almost none of it -- and nothing about the result says so.

        Tightening is unaffected and needs no guard: every surviving cell was
        still assessed by every filter. Only admission is a problem.

        Two separate checks, because the two failures look different:

        **Loosened thresholds.** A manual bound weaker than the pipeline's is not
        an admission at all -- manual thresholds are OR'd into the mask, so a
        weaker one adds nothing and the decision is a **silent no-op**. The user
        drags the line, exports, re-runs, and the cell count does not move. So
        this is caught by comparing the bounds themselves, not the outcome.

        **Admitted cells.** `rescue_barcodes` genuinely does put cells back, so
        that one is caught on the outcome.

        The escape route is not `--allow-unassessed` but the config: set the
        bound in filter_config.yaml, re-run `--force-qc` (~20 s -- mapping and
        CellBender sit upstream and are untouched), and the cascade produces that
        population properly. The check then passes on its own, because the
        decision no longer disagrees with the cascade.
        """
        problems = []

        # 1. Manual thresholds weaker than an actively-filtering automatic bound.
        #    Advisory/downgraded bounds are ignored: they remove nothing, so
        #    there is nothing to loosen relative to.
        bounds = _automatic_bounds(adata.uns.get('qc', {}))
        for entry in decisions.get('decisions', []):
            if entry.get('kind') != 'threshold':
                continue
            feature, op = entry.get('feature'), entry.get('op', 'max')
            try:
                value = float(entry['value'])
            except (TypeError, ValueError, KeyError):
                continue
            for b in bounds.get(feature, []):
                if b.get('exec_mode') != 'filter':
                    continue
                auto = b.get('lower') if op == 'min' else b.get('upper')
                if auto is None:
                    continue
                looser = (op == 'min' and value < auto) or (op == 'max' and value > auto)
                if looser:
                    problems.append(
                        "  {feat} {opsym} {val:g} is more permissive than {meth}'s {opsym} {auto:g}"
                        .format(feat=feature, opsym='>=' if op == 'min' else '<=',
                                val=value, meth=b.get('method', '?'), auto=auto))

        # 2. Cells any decision actually put back (rescue_barcodes).
        admitted = mask_before & ~self._final_mask
        n_admitted = int(admitted.sum())
        if n_admitted:
            examples = ', '.join(str(b) for b in adata.obs_names[admitted][:5])
            problems.append(
                '  {n} cell(s) rescued that the cascade removed, e.g. {ex}'.format(
                    n=n_admitted, ex=examples))

        if not problems:
            return

        detail = (
            '[REFILTER] These decisions try to keep cells the automatic cascade removed:\n'
            '{rows}\n\n'
            'That cannot be applied as a decision. Cells the cascade dropped were never in the '
            'view of anything downstream of the filter that dropped them, so their flag columns '
            'are False by default -- "not tested", not "passed". A weaker manual threshold is '
            'also simply ignored, since manual bounds are OR\'d onto the automatic ones and can '
            'only ever tighten.\n\n'
            'Set the bound in filter_config.yaml instead and re-run QC, so every downstream '
            'filter re-fits on the population you want:\n'
            '    python -m mapping.main -c <config> --force-qc\n'
            '(the cascade takes ~20 s; mapping and CellBender are not re-run.)\n\n'
            'Pass --allow-unassessed to write the artifact anyway.'
        ).format(rows='\n'.join(problems))

        if allow:
            self.logger.warning(detail.replace(
                'Pass --allow-unassessed to write the artifact anyway.',
                'Proceeding because --allow-unassessed was given; the loosened bounds above had '
                'no effect and the rescued cells are only partially assessed.'))
            return
        raise DecisionError(detail)

    def _build_mask(self, adata, decisions: dict):
        """
        Compute the removal mask: automatic cascade, then manual decisions.

        Order is deliberate and is what makes rescue work. Automatic flags go in
        first, manual thresholds and exclusions add to them, and rescues are
        applied last so an explicit "keep this cell" beats everything -- both
        the automatic cascade and any manual threshold.
        """
        auto_flags = collect_removal_flags(adata.uns.get('qc', {}))
        mask = np.zeros(adata.n_obs, dtype=bool)
        for flag in auto_flags:
            if flag in adata.obs:
                mask |= adata.obs[flag].astype(bool).values
        n_auto = int(mask.sum())
        # Snapshot before any decision touches it: _check_unassessed compares
        # the two to find cells the decisions put back.
        self._auto_mask = mask.copy()
        self.logger.info('[REFILTER] Automatic cascade removes {} of {} cells ({} flag columns)'.format(
            n_auto, adata.n_obs, len(auto_flags)))

        applied = []
        for entry in decisions.get('decisions', []):
            kind = entry['kind']
            if kind == 'threshold':
                applied.append(self._apply_threshold(adata, entry, mask))
            elif kind == 'exclude_barcodes':
                applied.append(self._apply_barcodes(adata, entry, mask, remove=True))

        # Rescues last: an explicit keep overrides every rule above it.
        for entry in decisions.get('decisions', []):
            if entry['kind'] == 'rescue_barcodes':
                applied.append(self._apply_barcodes(adata, entry, mask, remove=False))

        return mask, applied, auto_flags, n_auto

    # -- entrypoint ---------------------------------------------------------

    def run(self, force: bool=False, allow_unassessed: bool=False,
            allow_parent_mismatch: bool=False) -> Path:
        """Apply the decision file and write `{sample}_filtered_manual.h5ad`; returns its path.

        Skipped when the existing artifact already carries this decision hash,
        unless `force`.
        """
        decisions = self._load_decisions()
        this_hash = decision_hash(decisions)

        if self.output_path.exists() and not force:
            try:
                existing = sc.read_h5ad(self.output_path, backed='r')
                prior = (existing.uns.get(PROVENANCE_KEY) or {}).get('decision_hash')
                if hasattr(existing, 'file') and existing.file is not None:
                    existing.file.close()
            except Exception:
                prior = None
            if prior == this_hash:
                self.logger.info(
                    '[REFILTER] "{}" already reflects these decisions (hash {}); nothing to do. '
                    'Edit {} or pass --force to re-run.'.format(
                        self.output_path.name, this_hash, self.decisions_path.name))
                return self.output_path

        if not self.parent_path.exists():
            raise DecisionError(
                'No pre-filter object at {}. Manual refiltering reads the object holding every '
                'cell that entered QC, so that decisions can put cells back as well as take them '
                'out; run the mapping stage for this sample first.'.format(self.parent_path))

        self.logger.info('[REFILTER] Reading {}'.format(self.parent_path))
        adata = sc.read_h5ad(self.parent_path)
        self._check_parent(adata, decisions, allow_mismatch=allow_parent_mismatch)

        mask, applied, auto_flags, n_auto = self._build_mask(adata, decisions)
        self._final_mask = mask
        self._check_unassessed(adata, decisions, self._auto_mask, allow=allow_unassessed)
        adata.obs['flag_manual'] = mask & ~self._auto_only(adata, auto_flags)

        keep = ~mask
        n_keep = int(keep.sum())
        if n_keep == 0:
            raise DecisionError(
                'These decisions remove every cell in {}. Refusing to write an empty '
                'artifact.'.format(self.sample_name))

        # Recorded as an ordinary filter so every existing consumer handles it
        # without special-casing: parse_diagnostics renders it as a row, and
        # _METHOD_ORDER has no entry for it so it sorts last -- which is exactly
        # where a manual override belongs in a cascade.
        adata.uns.setdefault('qc', {})
        adata.uns['qc']['manual'] = {
            'flag_col': 'flag_manual',
            'exec_mode': 'filter',
            'features': [d.get('feature', '') for d in applied if d.get('feature')],
            'n_flagged': int(adata.obs['flag_manual'].sum()),
            'decision_hash': this_hash,
            'decisions': json.dumps(applied),
            'run_context': {'filtered_on_flags': list(auto_flags)},
        }

        filtered = adata[keep].copy()
        self.logger.info(
            '[REFILTER] {}: {} cells in, {} automatic removals, {} after manual decisions '
            '({:+d} vs automatic)'.format(
                self.sample_name, adata.n_obs, n_auto, n_keep, n_keep - (adata.n_obs - n_auto)))

        write_h5ad(
            filtered, self.output_path,
            sample=self.sample_name, mapper=self.mapper, stage='manual_filter',
            config_paths=self.config_paths,
            extra={'n_cells': int(filtered.n_obs),
                   'n_genes': int(filtered.n_vars),
                   'n_cells_input': int(adata.n_obs),
                   'decision_hash': this_hash,
                   'decision_file': str(self.decisions_path),
                   'parent_artifact': str(self.parent_path)})

        self._update_manifest()
        return self.output_path

    def _auto_only(self, adata, auto_flags) -> np.ndarray:
        """Cells the automatic cascade alone would have removed."""
        auto = np.zeros(adata.n_obs, dtype=bool)
        for flag in auto_flags:
            if flag in adata.obs:
                auto |= adata.obs[flag].astype(bool).values
        return auto

    def _update_manifest(self) -> None:
        """
        Register the manual artifact so integration can select it by key.

        Merged into the existing manifest rather than rewritten from scratch:
        the automatic artifacts are still there and still valid, and this run
        did not produce them.
        """
        manifest_path = self.sample_dir / 'manifest.json'
        artifacts = {}
        if manifest_path.exists():
            with open(manifest_path) as f:
                existing = dict((json.load(f).get('artifacts') or {}))
            # Stored relative to the sample dir (that is what keeps the results
            # tree relocatable); write_manifest re-relativises, so hand it back
            # absolute paths or it would fail its existence check and silently
            # drop every artifact this run did not produce.
            artifacts = {name: self.sample_dir / rel for name, rel in existing.items()}
        artifacts['filtered_manual_h5ad'] = self.output_path
        write_manifest(manifest_path, sample=self.sample_name, mapper=self.mapper,
                       artifacts=artifacts, base_dir=self.sample_dir, batch=self.batch,
                       config_paths=self.config_paths)
        self.logger.info("[REFILTER] Registered 'filtered_manual_h5ad' in {}".format(manifest_path))


def main(config_path: str, samples: list, decisions: str=None, force: bool=False,
         allow_unassessed: bool=False, allow_parent_mismatch: bool=False) -> int:
    """CLI entry point: apply manual QC decisions to one or more samples.

    Arguments mirror the CLI flags. Returns the number of samples that failed.
    """
    # One log file per run under logs/refilter/; see shared.utils.pipeline_log_path.
    pipeline_cfg = load_config(config_path) or {}
    log_file = pipeline_log_path('refilter', log_dir=pipeline_cfg.get('log_dir', 'logs'))
    setup_logger(log_file=log_file)
    logger = logging.getLogger('SingleCellMappingPipeline')

    failures = []
    for sample in samples:
        try:
            ManualRefilter(sample, config_path, decisions).run(
                force=force, allow_unassessed=allow_unassessed,
                allow_parent_mismatch=allow_parent_mismatch)
        except Exception as e:
            # Name the exception type: a KeyError's str() is just the missing
            # key, quoted, with no other context. exc_info adds the traceback.
            logger.error('[REFILTER] Sample "{}" failed: {}: {}'.format(
                sample, type(e).__name__, e), exc_info=True)
            failures.append((sample, '{}: {}'.format(type(e).__name__, e)))

    if failures:
        logger.error('[REFILTER] {} of {} sample(s) failed:'.format(len(failures), len(samples)))
        for sample, err in failures:
            logger.error('  {}: {}'.format(sample, err))
    return len(failures)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-c', '--config', required=True, help='Path to pipeline_config.yaml')
    parser.add_argument('--sample', nargs='+', required=True,
                        help='Sample(s) to refilter from their decision files')
    parser.add_argument('--decisions', default=None,
                        help='Explicit decision file (single sample only; defaults to '
                             '{qc}/{sample}_decisions.json)')
    parser.add_argument('--force', action='store_true',
                        help='Re-apply even when the artifact already matches the decision hash')
    parser.add_argument('--allow-unassessed', dest='allow_unassessed', action='store_true',
                        help='Write the artifact even when decisions admit cells the cascade '
                             'removed. Those cells were never seen by downstream filters; prefer '
                             'setting the bound in filter_config.yaml and re-running --force-qc.')
    parser.add_argument('--allow-parent-mismatch', dest='allow_parent_mismatch',
                        action='store_true',
                        help="Apply decisions even when the parent QC object's config_hash no "
                             'longer matches the one recorded when they were exported. Only safe '
                             'when the QC *data* is unchanged (e.g. only the report layout or an '
                             'unrelated config changed); always logs a warning. If QC was actually '
                             're-run with different filtering, re-export from the current report '
                             'instead.')
    args = parser.parse_args()

    if args.decisions and len(args.sample) > 1:
        parser.error('--decisions applies to a single --sample')

    sys.exit(1 if main(args.config, args.sample, args.decisions, args.force,
                       args.allow_unassessed, args.allow_parent_mismatch) else 0)
