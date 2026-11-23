#!/usr/bin/env python3
"""
Marker identification CLI.

    python -m markerid.main \
        --anndata results/integration/scVI/MtTip_noHVG/MtTip_noHVG.h5ad \
        --name MtTip_noHVG --method hotspot --latentkey X_scVI

`--method` is repeatable, so several methods can be compared on one dataset in
a single load. Runs in the scMarkersV3.12 environment (envs/scMarkersV3.12.yml).
"""

import argparse
import logging
import os

from pathlib import Path

from shared.utils import setup_logger, pipeline_log_path
from .markers import MarkerRun, METHODS, DEFAULT_LABELS_EXCLUDE
from .methods import CLUSTER_DEPENDENT

logger = logging.getLogger('MarkerLogging')


# A manifest lists one value per line. Data files are excluded by suffix so a
# single --anndata path is not mistaken for one and parsed as text.
MANIFEST_SUFFIXES = ('.txt', '.tsv', '.csv', '.list')


def normalize_list(values):
    """Accept either repeated values or a single manifest file listing one per line."""
    if len(values) == 1 and os.path.isfile(values[0]) \
            and Path(values[0]).suffix.lower() in MANIFEST_SUFFIXES:
        with open(values[0]) as handle:
            return [line.strip() for line in handle if line.strip()]
    return values


def run_sample(adata_path, sample, methods, cluster_key, latent_key, counts_key,
               n_genes, out_root, n_jobs, labels=None):
    """Run every requested method against one dataset, loading it once.

    `labels` is the dict of external-label options (`labels_tsv`, `labels_col`,
    `labels_cell_col`, `labels_exclude`, `min_cells_per_group`) or None.
    """
    results = {}
    shared_adata = None
    for method in methods:
        out_dir = Path(out_root) / method if out_root else None
        run = MarkerRun(adata_path=adata_path, sample=sample, method=method,
                        cluster_key=cluster_key, latent_key=latent_key,
                        counts_layer=counts_key, out_dir=out_dir,
                        n_jobs=n_jobs, adata=shared_adata, **(labels or {}))
        run.prepare()
        # Preparation is deterministic given the arguments, so later methods
        # reuse the filtered/normalised object instead of re-reading the h5ad.
        shared_adata = run.adata
        results[method] = run.run(n_genes=n_genes)
    return results


def main():
    """CLI entry point: run every requested method on every sample.

    A failed sample is logged and the rest continue; any failure ends the
    process with SystemExit(1).
    """
    parser = argparse.ArgumentParser(description='Identify marker genes')
    parser.add_argument('--method', required=True, action='append',
                        choices=sorted(METHODS),
                        help='Marker method; repeat to run several')
    parser.add_argument('--anndata', required=True, nargs='+',
                        help='.h5ad paths, or one text file listing them')
    parser.add_argument('--name', required=True, nargs='+',
                        help='Sample names, or one text file listing them, '
                             'in the same order as --anndata')
    parser.add_argument('--clusterkey', help='.obs key with cluster labels '
                                             f'(required for {", ".join(CLUSTER_DEPENDENT)}, cosg)')
    # External labels: rank on a declared label file instead of an .obs
    # cluster key. Cells absent from the table, or labelled with one of
    # --labels-exclude, are dropped before gene filtering.
    parser.add_argument('--labels-tsv', '--labelstsv', dest='labels_tsv',
                        help='TSV with one row per cell; makes the cluster-dependent '
                             'methods rank on its label column (overrides --clusterkey)')
    parser.add_argument('--labels-col', '--labelscol', dest='labels_col', default='label',
                        help='label column of --labels-tsv (default: label)')
    parser.add_argument('--labels-cell-col', '--labelscellcol', dest='labels_cell_col',
                        default='cell',
                        help='cell-id column of --labels-tsv: bare barcodes or '
                             '<sample>.<barcode> (default: cell)')
    parser.add_argument('--labels-exclude', '--labelsexclude', dest='labels_exclude',
                        nargs='*', default=list(DEFAULT_LABELS_EXCLUDE),
                        help='labels whose cells are dropped (default: Unassigned)')
    parser.add_argument('--min-cells-per-group', '--mincellspergroup',
                        dest='min_cells_per_group', type=int, default=0,
                        help='drop groups with fewer cells than this before ranking '
                             '(any cluster key; default 0 = keep all)')
    parser.add_argument('--latentkey', help='.obsm key with the latent space, '
                                            'e.g. X_scVI (recommended)')
    parser.add_argument('--countskey', default='corrected_counts',
                        help="layer holding raw counts, or 'X'")
    parser.add_argument('--ngenes', type=int, default=50,
                        help='genes kept per geneset (0 for all)')
    parser.add_argument('--outdir', default='results/markers',
                        help='output root; each method writes to <outdir>/<method>/')
    parser.add_argument('--jobs', type=int,
                        default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    args = parser.parse_args()

    paths = normalize_list(args.anndata)
    names = normalize_list(args.name)
    if len(paths) != len(names):
        parser.error(f'--anndata ({len(paths)}) and --name ({len(names)}) '
                     f'must have the same length')

    # One log file per run under logs/markerID/; see shared.utils.pipeline_log_path.
    log_file = pipeline_log_path('markerID')
    setup_logger(name='MarkerLogging', log_file=log_file, level=logging.INFO)

    n_genes = args.ngenes or None
    labels = None
    if args.labels_tsv or args.min_cells_per_group:
        labels = dict(labels_tsv=args.labels_tsv, labels_col=args.labels_col,
                      labels_cell_col=args.labels_cell_col,
                      labels_exclude=tuple(args.labels_exclude or ()),
                      min_cells_per_group=args.min_cells_per_group)
    failed = []
    for path, name in zip(paths, names):
        logger.info('[MAIN] %s (%s)', name, path)
        try:
            run_sample(path, name, args.method, args.clusterkey, args.latentkey,
                       args.countskey, n_genes, args.outdir, args.jobs, labels=labels)
        except Exception:
            logger.exception('[MAIN] %s failed', name)
            failed.append(name)

    if failed:
        logger.error('[MAIN] %d/%d samples failed: %s',
                     len(failed), len(names), ', '.join(failed))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
