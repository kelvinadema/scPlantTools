#!/usr/bin/env python3
"""
Stage 3 -- one UMAP per trained model, not just scANVI's.

A refine run fits up to four latents and, until this module existed, embedded
exactly one of them:

    SCVI/                the base fit that seeds scANVI, untuned
    SCANVI/              the deliverable -- the only one that got a UMAP
    SCVI_scibtuned/      optional comparison arm, scib-tuned
    JointEmbeddingSCVI/  optional comparison arm, scib-tuned

The arms exist to be *compared* against the deliverable, and the comparison a
person actually makes is visual: does this latent separate the same populations,
and does it still mix the batches. Training curves and the batch-mixing table
answer neither. So the same neighbours/UMAP/Leiden computation
(`shared.dimred.DimReducer`) and the same figures (`shared.dimred.UMAPPlotter`)
that `RefineRun._embed` has always run on the scANVI latent are run here per
model, into that model's own directory.

Every UMAP is keyed by the LATENT it came from, never by the run alone:

    X_scANVI_{run}      -> .obsm['X_umap_{run}'],              SCANVI/
    X_scVI_{run}        -> .obsm['X_umap_scVI_{run}'],         SCVI/
    X_scVItuned_{run}   -> .obsm['X_umap_scVItuned_{run}'],    SCVI_scibtuned/
    X_jointSCVI_{run}   -> .obsm['X_umap_jointSCVI_{run}'],    JointEmbeddingSCVI/

scANVI keeps the bare `{run}` suffix it has always had, because that key is the
deliverable and is already referenced by the report, by `inject_refine_labels`
and by anything downstream that reads a finished run. The three new keys carry
the latent's own name for the reason `model_dir` exists at all: a figure or an
embedding that names only the run identifies no model.

The colourings are the ones `_predict` already wrote (seed, seed support,
annotation, confidence, flag) plus the batch key, so the four panels are
directly comparable -- the same cells, the same colours, a different manifold.

Two entrypoints, one implementation:

    embed_models(...)      called by RefineRun._embed during a run
    python -m refine.embed_models --cfg configs/refine_config.yaml --sample S
                           BACKFILL for a run that has already been written.
                           Reads only .obs/.obsm out of the finished h5ad (the
                           counts are irrelevant to a UMAP and the file is
                           ~11 GB), so it needs neither a GPU nor the models.

The backfill deliberately does NOT write back into the run's h5ad. Adding
`.obsm` keys to an existing AnnData on disk means rewriting the whole file, and
a half-written 11 GB deliverable costs a GPU-day to reproduce. The coordinates
land next to the figures as `{sample}_{key}_umap.tsv.gz` instead, and a re-run
of stage 3 (which reuses the models already on disk) is what puts them in the
h5ad.
"""

import argparse
import json
import logging
import sys

from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd

from shared.dimred import DimReducer, UMAPPlotter

logger = logging.getLogger('AnnotationLogging')


#: One model's embedding: where its figures go, which latent it produced, and
#: the suffix every key derived from that latent carries.
LatentSpec = namedtuple('LatentSpec', 'subdir latent key')

#: Order matters only for the log: the deliverable first, then the base fit it
#: was built from, then the comparison arms. `key` is what DimReducer turns into
#: `X_umap_{key}` / `leiden_{key}` / `neighbors_{key}`.
MODEL_LATENTS = (
    ('SCANVI', 'X_scANVI_{run}', '{run}'),
    ('SCVI', 'X_scVI_{run}', 'scVI_{run}'),
    ('SCVI_scibtuned', 'X_scVItuned_{run}', 'scVItuned_{run}'),
    ('JointEmbeddingSCVI', 'X_jointSCVI_{run}', 'jointSCVI_{run}'),
)

#: Everything except the deliverable. `RefineRun._embed` embeds SCANVI
#: unconditionally and these only when asked, so that turning the option off
#: returns the stage to exactly what it did before.
COMPARISON_MODELS = tuple(s[0] for s in MODEL_LATENTS if s[0] != 'SCANVI')


def latent_specs(run, models=None):
    """Resolve the templates above for one run.

    Args:
        run (str): The run name -- 'combined' for the normal route.
        models (list[str], optional): Subset of model directory names, in
            `MODEL_LATENTS` order regardless of the order given. Defaults to
            all four.

    Returns:
        list[LatentSpec]: The specs, whether or not the latent exists.

    Raises:
        ValueError: If a name is not one of the four models a run can fit.
    """
    known = {s[0] for s in MODEL_LATENTS}
    wanted = set(known if models is None else models)
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError('unknown model(s) {}; known: {}'.format(
            ', '.join(unknown), ', '.join(sorted(known))))
    return [LatentSpec(subdir, latent.format(run=run), key.format(run=run))
            for subdir, latent, key in MODEL_LATENTS if subdir in wanted]


def color_columns(adata, run, batch_key=None):
    """The `.obs` columns every model's UMAP is coloured by.

    The same list `RefineRun._embed` has always used for scANVI. Holding it
    identical across models is the point: a panel that differs in colouring is
    not a comparison of manifolds.

    Args:
        adata (AnnData): The object being plotted.
        run (str): The run name.
        batch_key (str, optional): The batch column, appended when present.

    Returns:
        list[str]: Column names present in `.obs`, in drawing order.
    """
    cols = ['seed_{}'.format(run), 'seed_support_{}'.format(run),
            'annotation_{}'.format(run), 'confidence_{}'.format(run),
            'flag_{}'.format(run)]
    if batch_key:
        cols.append(batch_key)
    return [c for c in cols if c in adata.obs.columns]


def embed_one(adata, spec, sample, run, out_dir, cfg=None, batch_key=None,
              do_leiden=True, log=None):
    """Neighbours + UMAP + Leiden on one latent, then that model's figures.

    Args:
        adata (AnnData): Carries the latent in `.obsm`; mutated in place with
            the resulting neighbours graph, UMAP and Leiden clustering.
        spec (LatentSpec): Which latent, which directory, which key suffix.
        sample (str): Sample name; the figure filename prefix.
        run (str): Run name. Passed rather than parsed back out of `spec.key`:
            the colour columns are named after the RUN and a run name may itself
            contain an underscore.
        out_dir (Path): The run directory -- figures go to `out_dir/spec.subdir`.
        cfg (dict, optional): The `embedding:` config block.
        batch_key (str, optional): Batch column to colour by.
        do_leiden (bool): Cluster on the same graph the UMAP came from.
        log (logging.Logger, optional): Defaults to the module logger.

    Returns:
        dict: `DimReducer.compute`'s result, plus 'subdir' and 'latent'.

    Raises:
        KeyError: If the latent is not in `.obsm`. Callers that iterate over
            models should check with `spec.latent in adata.obsm` first -- a
            missing arm is a disabled arm, not a fault.
    """
    log = log or logger
    cfg = dict(cfg or {})
    if spec.latent not in adata.obsm:
        raise KeyError("'{}' is not in .obsm".format(spec.latent))

    log.info('[REFINE] %s: embedding %s (%s) -> X_umap_%s',
             sample, spec.subdir, spec.latent, spec.key)
    result = DimReducer(adata, logger=log).compute(
        use_rep=spec.latent,
        key=spec.key,
        n_neighbors=cfg.get('n_neighbors'),
        metric=cfg.get('metric', 'correlation'),
        min_dist=cfg.get('min_dist', 0.0),
        leiden_resolution=cfg.get('leiden_resolution', 0.5),
        random_state=cfg.get('random_state', 42),
        do_leiden=do_leiden)

    # The figures are wrapped and the embedding is not, deliberately: the
    # coordinates are the expensive half (an hour of neighbours + UMAP on
    # ~150k cells) and a plotting fault must not throw them away. A caller
    # that needs the embedding itself to be fatal -- `RefineRun._embed` on the
    # deliverable -- still gets that, because `compute` above is not caught here.
    try:
        UMAPPlotter(result['adata'], Path(out_dir) / spec.subdir, logger=log).plot(
            umap_key=result['umap_key'], leiden_key=result['leiden_key'],
            extra_color_by=color_columns(result['adata'], run, batch_key),
            prefix=sample)
    except Exception as exc:                              # noqa: BLE001
        log.warning('[REFINE] %s: %s embedding plots failed: %s',
                    sample, spec.subdir, exc)

    result['subdir'] = spec.subdir
    result['latent'] = spec.latent
    return result


def embed_models(adata, sample, run, out_dir, models=None, cfg=None,
                 batch_key=None, do_leiden=True, log=None, record=None):
    """Embed and plot several models' latents, one directory each.

    Per model, collect-and-continue: an embedding that fails (a latent scanpy
    chokes on, a figure backend fault) must not cost the models after it. These
    are independent comparisons, and the deliverable's own UMAP is normally the
    first one done.

    Args:
        adata (AnnData): The run's object.
        sample (str): Sample name.
        run (str): Run name.
        out_dir (Path): The run directory.
        models (list[str], optional): Subset of model directory names.
        cfg (dict, optional): The `embedding:` config block.
        batch_key (str, optional): Batch column to colour by.
        do_leiden (bool): Cluster on each model's own graph.
        log (logging.Logger, optional): Defaults to the module logger.
        record (dict, optional): Mutated with one entry per model, for the run
            record.

    Returns:
        dict: {model dir name: DimReducer result}, only for models that ran.
    """
    log = log or logger
    results = {}
    for spec in latent_specs(run, models):
        if spec.latent not in adata.obsm:
            log.info("[REFINE] %s: no '%s' in .obsm; %s not embedded "
                     '(arm disabled or not fitted)', sample, spec.latent, spec.subdir)
            continue
        try:
            result = embed_one(adata, spec, sample=sample, run=run, out_dir=out_dir,
                               cfg=cfg, batch_key=batch_key, do_leiden=do_leiden, log=log)
        except Exception as exc:                          # noqa: BLE001
            log.error('[REFINE] %s: embedding %s failed (%s: %s); continuing with '
                      'the remaining model(s).', sample, spec.subdir,
                      type(exc).__name__, exc)
            if record is not None:
                record[spec.subdir] = {'latent': spec.latent, 'failed': True,
                                       'error': '{}: {}'.format(type(exc).__name__, exc)}
            continue
        results[spec.subdir] = result
        if record is not None:
            record[spec.subdir] = {'latent': spec.latent, 'umap_key': result['umap_key'],
                                   'leiden_key': result['leiden_key'] or '',
                                   **result['params']}
    return results


# --------------------------------------------------------------------- backfill

def load_obs_obsm(h5ad_path, log=None):
    """A finished run's `.obs` and `.obsm`, without its counts.

    A UMAP needs a latent and the columns to colour by; the counts matrix is
    ~11 GB of it and is never touched. `read_elem` on the two groups gives an
    AnnData of shape (n_obs, 0) that `sc.pp.neighbors(use_rep=...)`,
    `sc.tl.umap` and `sc.tl.leiden` all work on unchanged, because every one of
    them reads the representation out of `.obsm` and then only ever touches the
    neighbour graph.

    Args:
        h5ad_path (Path): The run's h5ad.
        log (logging.Logger, optional): Defaults to the module logger.

    Returns:
        AnnData: Cells x 0, carrying the full `.obs` and `.obsm`.
    """
    import anndata as ad
    import h5py
    from anndata.io import read_elem

    log = log or logger
    with h5py.File(str(h5ad_path), 'r') as handle:
        obs = read_elem(handle['obs'])
        # `_scvi_extra_continuous_covs` comes back as a DataFrame rather than an
        # array. Nothing here embeds or plots it, and keeping it would only give
        # AnnData's .obsm validation something to object to.
        elements = {k: read_elem(handle['obsm'][k]) for k in handle['obsm']}
    obsm = {k: v for k, v in elements.items() if isinstance(v, np.ndarray)}
    adata = ad.AnnData(obs=obs, obsm=obsm)
    log.info('[REFINE] loaded %d cells x %d obs column(s), %d embedding(s) from %s',
             adata.n_obs, obs.shape[1], len(obsm), h5ad_path)
    return adata


def write_coords(adata, result, sample, out_dir, log=None):
    """The 2-D coordinates and clusters, beside that model's figures.

    Why a sidecar and not the h5ad: see the module docstring. This file is what
    makes a backfilled UMAP reusable (re-plot it, colour it by something else,
    join it to the labels TSV on barcode) without re-running the embedding.

    Args:
        adata (AnnData): The embedded object.
        result (dict): One `embed_one` result.
        sample (str): Sample name.
        out_dir (Path): The run directory.
        log (logging.Logger, optional): Defaults to the module logger.

    Returns:
        Path: The file written.
    """
    log = log or logger
    embedding = np.asarray(adata.obsm[result['umap_key']])
    frame = pd.DataFrame({'UMAP1': embedding[:, 0], 'UMAP2': embedding[:, 1]},
                         index=adata.obs_names)
    frame.index.name = 'barcode'
    if result['leiden_key']:
        frame['leiden'] = adata.obs[result['leiden_key']].astype(str).to_numpy()
    path = Path(out_dir) / result['subdir'] / '{}_{}_umap.tsv.gz'.format(
        sample, result['umap_key'][len('X_umap_'):])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep='\t')
    log.info('[REFINE] %s: coordinates -> %s', result['subdir'], path)
    return path


def backfill(sample, run, out_dir, h5ad=None, models=None, cfg=None, batch_key=None,
             do_leiden=True, log=None):
    """Generate the missing UMAPs for a run that has already been written.

    Args:
        sample (str): Sample name.
        run (str): Run name.
        out_dir (Path): The refine output root -- `out_dir/sample/run` is the
            run directory.
        h5ad (Path, optional): The run's h5ad; defaults to the one the run wrote.
        models (list[str], optional): Which models. Defaults to the three
            comparison latents -- scANVI already has its figures.
        cfg (dict, optional): The `embedding:` config block.
        batch_key (str, optional): Batch column to colour by.
        do_leiden (bool): Cluster on each model's own graph.
        log (logging.Logger, optional): Defaults to the module logger.

    Returns:
        dict: {model dir name: result}, as `embed_models`.
    """
    log = log or logger
    run_dir = Path(out_dir) / sample / run
    h5ad = Path(h5ad) if h5ad else run_dir / '{}_{}.h5ad'.format(sample, run)
    if not h5ad.exists():
        raise FileNotFoundError('no run artifact at {}'.format(h5ad))

    adata = load_obs_obsm(h5ad, log=log)
    record = {}
    results = embed_models(adata, sample=sample, run=run, out_dir=run_dir,
                           models=models or list(COMPARISON_MODELS), cfg=cfg,
                           batch_key=batch_key, do_leiden=do_leiden,
                           log=log, record=record)
    for result in results.values():
        write_coords(adata, result, sample=sample, out_dir=run_dir, log=log)

    # Beside the run record rather than inside it: the run record describes what
    # the RUN did, and this ran afterwards. Merged into `embedding_models` by a
    # later re-run of the stage, which recomputes these in place.
    sidecar = run_dir / '{}_{}_umap_backfill.json'.format(sample, run)
    sidecar.write_text(json.dumps(record, indent=2, default=str))
    log.info('[REFINE] %s/%s: %d model(s) embedded -> %s',
             sample, run, len(results), sidecar)
    return results


def _cli_config(args):
    """Resolve sample/run/paths from the refine config, with CLI overrides.

    Reading the config rather than taking eight flags is what keeps a backfilled
    figure comparable to the one the run produced: same n_neighbors, same
    metric, same min_dist, same seed, same batch key.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        tuple: (embedding cfg dict, output_dir Path, batch_key str or None).
    """
    from shared.utils import find_root_dir, load_config

    if not args.cfg:
        return {}, Path(args.out_dir), args.batch_key

    cfg = load_config(Path(args.cfg))['RefineConfigs']
    target = (cfg.get('targets') or {}).get(args.sample)
    if target is None:
        raise ValueError('no target {!r} in {}. Known: {}'.format(
            args.sample, args.cfg, ', '.join(sorted(cfg.get('targets') or {})) or '(none)'))
    out_dir = Path(args.out_dir or cfg.get('output_dir')
                   or (find_root_dir() / 'results/refine'))
    return (dict(cfg.get('embedding') or {}), out_dir,
            args.batch_key or target.get('batch_key'))


def main(argv=None) -> int:
    """Backfill entrypoint. See the module docstring for the two ways in."""
    from shared.utils import setup_logger

    parser = argparse.ArgumentParser(
        description='Generate the per-model UMAPs for a finished refine run')
    parser.add_argument('--cfg', help='configs/refine_config.yaml, for the embedding '
                                      'settings and batch key the run itself used')
    parser.add_argument('--sample', required=True, help='target name')
    parser.add_argument('--run', default='combined', help="run name (default 'combined')")
    parser.add_argument('--out-dir', help='refine output root; default from --cfg')
    parser.add_argument('--h5ad', help="the run's h5ad; default the one the run wrote")
    parser.add_argument('--models', nargs='+', default=None,
                        help='model directories to embed (default: {})'.format(
                            ' '.join(COMPARISON_MODELS)))
    parser.add_argument('--batch-key', help='.obs column to colour by; default from --cfg')
    parser.add_argument('--no-leiden', action='store_true',
                        help="don't cluster on each model's own graph")
    args = parser.parse_args(argv)

    if not args.cfg and not args.out_dir:
        parser.error('one of --cfg or --out-dir is required')

    log = setup_logger(name='AnnotationLogging', level=logging.INFO)
    embedding_cfg, out_dir, batch_key = _cli_config(args)
    backfill(sample=args.sample, run=args.run, out_dir=out_dir, h5ad=args.h5ad,
             models=args.models, cfg=embedding_cfg, batch_key=batch_key,
             do_leiden=not args.no_leiden, log=log)
    return 0


if __name__ == '__main__':
    sys.exit(main())
