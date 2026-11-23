#!/usr/bin/env python3
"""
Stage 3 entrypoint -- refine one target from the reviewer's curated grouping.

    # propose a grouping from stage 1's own calls, for the reviewer to edit
    python -m refine.main --cfg configs/refine_config.yaml --propose

    # check the seed classes without touching a GPU (seconds)
    python -m refine.main --cfg configs/refine_config.yaml --dry-run

    # the run
    python -m refine.main --cfg configs/refine_config.yaml

    # what one method would have said on its own, for a suspicious call
    python -m refine.main --cfg configs/refine_config.yaml --method hotspot

**One run, not six.** The six marker methods are six ways of finding genesets,
not six answers to train separately; stage 2 is where a person decides which of
their genesets describe the same population, and what comes out is a single set
of cell types. The six-way comparison lives in the review page instead, where
it costs nothing.

Runs under `scIntegrationV1.12` on a GPU node.
"""

import argparse
import json
import logging
import sys

from pathlib import Path

import pandas as pd

from shared.config_spec import REFINE_CONFIG, check
from shared.groups import GroupError, propose, write as write_groups
from shared.utils import find_root_dir, load_config, setup_logger, pipeline_log_path

from .refine import RefineRun


def _ontology_extensions(cfg):
    """Extension OBO paths, resolved next to the ontology file (as stage 1 does)."""
    ont = cfg.get('ontology_file')
    out = []
    for e in (cfg.get('ontology_extensions') or []):
        p = Path(e)
        out.append(str(p) if p.is_absolute() or ont is None
                   else str(Path(ont).with_name(e)))
    return out


def _resolve_paths(cfg, root):
    """(annotation_dir, output_dir, log_dir), defaulting under the repo root."""
    annotation_dir = cfg.get('annotation_dir') or (root / 'results/annotation')
    output_dir = cfg.get('output_dir') or (root / 'results/refine')
    log_dir = cfg.get('log_dir') or (root / 'logs')
    return Path(annotation_dir), Path(output_dir), Path(log_dir)


def _already_done(run_path: Path, gate_hash: str) -> bool:
    """
    Has this exact grouping already been refined?

    Gated on the grouping's hash rather than a `.done` flag, because the
    grouping is an *input*: editing a note must be a no-op while moving a
    geneset between groups must re-run. Same rule as `mapping.refilter`.
    """
    if not run_path.exists() or not gate_hash:
        return False
    try:
        record = json.loads(run_path.read_text())
    except (OSError, ValueError):
        return False        # an unreadable record is not evidence of a finished run
    return record.get('group_hash', record.get('decision_hash')) == gate_hash


def do_propose(sample, target, annotation_dir, methods=None, force=False,
               logger=None) -> Path:
    """
    Write the starting grouping: genesets sharing an automatic top call, pooled.

    A proposal, not an answer: the merge is made explicit, across methods, in a
    file the reviewer edits.
    """
    sample_dir = Path(annotation_dir) / sample
    methods = methods or sorted(
        p.name for p in sample_dir.iterdir()
        if p.is_dir() and (p / '{}_decisions.tsv'.format(sample)).exists())
    if not methods:
        raise GroupError('No stage 1 decision templates under {}'.format(sample_dir))

    templates, annotations, evidence = {}, {}, {}
    for method in methods:
        frame = pd.read_csv(sample_dir / method / '{}_decisions.tsv'.format(sample),
                            sep='\t', dtype={'cluster': str})
        templates[method] = frame
        annotations[method] = sample_dir / method / '{}_significant_annotations.tsv'.format(sample)
        for _, row in frame.iterrows():
            evidence[(method, str(row['cluster']))] = {
                'n_seed_cells': row.get('n_seed_cells', ''),
                'decision': row.get('decision', ''),
                'top_po_name': row.get('top_po_name', '')}

    doc = propose(sample, templates, annotations=annotations)
    tsv = write_groups(sample_dir / '{}_groups.tsv'.format(sample), doc,
                       evidence=evidence, force=force)
    if logger:
        logger.info('[GROUPS] %s: %d geneset(s) across %d method(s) -> %d group(s) '
                    '(%d named, %d unnamed), %d ungrouped, %d dropped',
                    sample, sum(len(t) for t in templates.values()), len(methods),
                    len(doc.groups), len(doc.named), len(doc.unnamed),
                    len(doc.ungrouped), len(doc.dropped))
        logger.info('[GROUPS] -> %s', tsv)
        for _, row in doc.summary().iterrows():
            logger.info('[GROUPS]   %-28s %-12s %d method(s): %s',
                        row['group'], row['po_id'] or '(unnamed)', row['n_methods'],
                        row['members'])
    return tsv


def main(config_path, sample=None, method=None, dry_run=False, force=False,
         allow_stale=False, do_propose_only=False, force_groups=False) -> int:
    """Run stage 3 for one target, or every target in the config; returns the exit status.

    Arguments mirror the CLI flags below.
    """
    config = load_config(Path(config_path))
    errors = check(REFINE_CONFIG, config, Path(config_path).name)
    if errors:
        raise ValueError('{} is not valid:\n  {}'.format(config_path, '\n  '.join(errors)))
    cfg = config['RefineConfigs']

    root = find_root_dir()
    annotation_dir, output_dir, log_dir = _resolve_paths(cfg, root)

    # One log file per run under logs/refine/; see shared.utils.pipeline_log_path.
    log_file = pipeline_log_path('refine', log_dir=log_dir)
    logger = setup_logger(name='AnnotationLogging', log_file=log_file, level=logging.INFO)
    logger.info('[REFINE] Starting (log: %s)', log_file)

    targets = cfg.get('targets') or {}
    names = [sample] if sample else sorted(targets)
    unknown = [n for n in names if n not in targets]
    if unknown:
        raise ValueError('No target(s) {} in {}. Known: {}'.format(
            unknown, config_path, ', '.join(sorted(targets)) or '(none)'))

    failures, completed, skipped = [], [], []
    for name in names:
        target = targets[name]
        try:
            if do_propose_only:
                do_propose(name, target, annotation_dir,
                           methods=target.get('methods'), force=force_groups, logger=logger)
                completed.append('{} (proposed)'.format(name))
                continue

            run = RefineRun(
                sample=name, adata_path=target['adata'],
                annotation_dir=annotation_dir, output_dir=output_dir,
                method=method, methods=target.get('methods'),
                batch_key=target.get('batch_key'),
                counts_layer=target.get('counts_layer', 'corrected_counts'),
                covariates=target.get('covariates'),
                seeds=seeds_for(cfg, target), safeguards=cfg.get('safeguards'),
                model=cfg.get('model'), embedding=cfg.get('embedding'),
                ontology_file=cfg.get('ontology_file'),
                ontology_extensions=_ontology_extensions(cfg),
                allow_stale=allow_stale,
                # --force retrains; otherwise models on disk are reused
                # (see RefineRun._reuse_model).
                reuse_models=not force)

            run.prepare()

            run_path = run.out_dir / '{}_{}_run.json'.format(name, run.run)
            if not force and _already_done(run_path, run.gate_hash):
                logger.info('[REFINE] %s/%s: grouping unchanged since %s; skipping '
                            '(--force to re-run)', name, run.run, run_path.name)
                skipped.append('{}/{}'.format(name, run.run))
                continue

            if dry_run:
                _report_dry_run(run, logger)
                completed.append('{}/{} (dry run)'.format(name, run.run))
                continue

            run.execute()
            completed.append('{}/{}'.format(name, run.run))
        except Exception as exc:                      # noqa: BLE001 - collected below
            logger.exception('[REFINE] %s failed', name)
            failures.append('{}: {}: {}'.format(name, type(exc).__name__, exc))

    total = len(completed) + len(failures) + len(skipped)
    logger.info('[REFINE] %d/%d completed, %d skipped, %d failed',
                len(completed), total, len(skipped), len(failures))
    if failures:
        logger.error('[REFINE] failures:\n  %s', '\n  '.join(failures))
        return 1
    return 0


def seeds_for(cfg: dict, target: dict) -> dict:
    """
    The `seeds` block for one target: the global block, with the target's own
    `seeds` keys laid over it. Returns a new dict; neither input is modified.

    The right seeding rule can differ between targets in one config:
    `quantile_scope: geneset` suits a review group that unites a nuclei and a
    protoplast cluster, but over-seeds a group whose members include a cluster
    with no evidence for that cell type.
    """
    merged = dict(cfg.get('seeds') or {})
    merged.update(target.get('seeds') or {})
    return merged


def _report_dry_run(run, logger) -> None:
    """
    Everything the review page deliberately does not compute.

    The page shows only numbers stage 1 actually produced; a group's real seed
    count depends on the group-level assignment, which needs the score matrices
    and cannot happen in a browser. This is where that number comes from, and
    it is seconds rather than GPU-hours, so it is meant to be run after every
    editing session.
    """
    record = run.record
    logger.info('[REFINE] --- dry run: %s / %s ---', run.sample, run.run)
    logger.info('[REFINE] %d seed class(es), %d/%d cells seeded (%.1f%%)',
                record['n_seed_classes'], record['n_seed_cells'], record['n_cells'],
                100 * record['seed_fraction'])
    if record.get('support_histogram'):
        logger.info('[REFINE] method support of seeded cells: %s',
                    ', '.join('{} method(s): {}'.format(k, v)
                              for k, v in sorted(record['support_histogram'].items())))
    logger.info('[REFINE] %-44s %8s  %s', 'class', 'seeds', 'kind')
    for row in record['seeds']:
        logger.info('[REFINE] %-44s %8d  %s', row['label'], row['n_seed_cells'], row['kind'])
    if record.get('mode') == 'combined':
        logger.info('[REFINE] grouping: %d group(s) (%d named, %d unnamed), %d ungrouped, '
                    '%d dropped', record['n_groups'], record['n_named_groups'],
                    record['n_unnamed_groups'], record['n_ungrouped'], record['n_dropped'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Refine annotations from a curated cross-method grouping')
    parser.add_argument('--cfg', required=True, help='Path to refine_config.yaml')
    parser.add_argument('--sample', required=False,
                        help='Target to refine. Default: every target in the config.')
    parser.add_argument('--propose', dest='do_propose_only', action='store_true',
                        help='Write the starting grouping from stage 1\'s own top calls and '
                             'stop. This is the file the review page and the reviewer edit.')
    parser.add_argument('--force-groups', action='store_true',
                        help='Overwrite an edited grouping when proposing. Discards review work.')
    parser.add_argument('--method', required=False,
                        help='Refine ONE method from its own per-geneset decision file, '
                             'instead of the combined grouping. For checking what a single '
                             'method would have said; not the normal route.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Build the seed labels, report the classes and the per-cell '
                             'method support, and stop before training.')
    parser.add_argument('--force', action='store_true',
                        help='Re-run even when the grouping is unchanged since the last run')
    parser.add_argument('--allow-stale', action='store_true',
                        help='Accept a grouping written against older annotation tables')
    args = parser.parse_args()
    sys.exit(main(config_path=args.cfg, sample=args.sample, method=args.method,
                  dry_run=args.dry_run, force=args.force, allow_stale=args.allow_stale,
                  do_propose_only=args.do_propose_only, force_groups=args.force_groups))
