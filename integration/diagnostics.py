#!/usr/bin/env python3

"""
Post-training integration diagnostics for the scVI path: training-loss curves
(from ``model.history``) and label-free batch-mixing metrics (scib-metrics) on
the latent space.

Two independent questions, kept in separate functions because they need
different inputs and fail in different ways:

  * *Did the model converge / overfit?* -> ``plot_training_history``, from the
    trained ``model``'s ``.history``. scvi persists every attribute ending in
    ``_`` (including ``history_``) into ``model.pt`` and restores it on ``load``
    (verified against scvi 1.5.0), so a model reloaded from the registry on a
    ``--plot`` rerun still carries its curves -- these are recoverable without
    retraining.

  * *Did the batches actually mix?* -> ``batch_mixing_metrics``: iLISI and kBET
    on a latent ``.obsm`` keyed by ``batch``. Needs only the AnnData, so it works
    for any method that writes a latent (scVI or scArches).

**Bio-conservation metrics (NMI/ARI/silhouette vs cell type) are deliberately
absent.** They need cell-type labels this pipeline does not have until stage-3
annotation. Only the label-free *batch-mixing* half of integration quality is
computable pre-annotation; ``silhouette_batch`` is excluded for the same reason
(scib computes it per cell-type label). When labels exist, this is the natural
place to add a full ``scib_metrics.benchmark.Benchmarker`` pass.

Lives in ``integration/`` rather than ``shared/`` on purpose: it imports
scib-metrics/scvi, which exist only in the integration env, and ``shared/`` must
stay importable by the preflight validator outside that env.
"""

import logging
import os

import numpy as np
import pandas as pd

from pathlib import Path
from plotnine import (ggplot, aes, geom_line, geom_col, geom_text, facet_wrap,
                      theme_bw, labs, scale_color_manual, ylim, guides)


# scib's own convention for the neighbour count these metrics read (LISI's
# default perplexity is n_neighbors // 3, i.e. 30 at 90). Deliberately NOT the
# sqrt(N) the UMAP uses: iLISI/kBET are defined against a fixed local
# neighbourhood, not the embedding's, so they must not track the cell count.
_METRIC_N_NEIGHBORS = 90


def _split_metric_name(name: str):
    """
    ('elbo_train' | 'elbo_validation' | 'train_loss_epoch' | ...) -> (base, split).

    scvi's history keys are not uniformly suffixed -- most are ``<base>_train`` /
    ``<base>_validation`` but a few read ``train_loss_epoch`` / ``validation_loss``
    -- so pair on the presence of the word rather than a fixed suffix, and strip
    every train/validation/epoch/step token to recover a base that groups the two
    splits of one metric onto one facet.
    """
    low = name.lower()
    if 'validation' in low:
        split = 'validation'
    elif 'train' in low:
        split = 'train'
    else:
        split = 'other'
    base = name
    for token in ('_validation', 'validation_', '_train', 'train_', '_epoch', '_step'):
        base = base.replace(token, '')
    base = base.strip('_') or name
    return base, split


def _tidy_history(history):
    """
    ``model.history`` ({metric: epoch-indexed 1-col DataFrame}) -> long tidy frame
    with columns (epoch, value, metric, split), or None when there is nothing to
    plot. Robust to whatever set of keys a given model class logged -- no key is
    assumed present -- so it works unchanged for SCVI and JointEmbeddingSCVI.
    """
    if not history:
        return None
    frames = []
    for name, df in history.items():
        if df is None or len(df) == 0:
            continue
        series = df.iloc[:, 0]
        base, split = _split_metric_name(str(name))
        frames.append(pd.DataFrame({
            'epoch': np.asarray(series.index, dtype=float),
            'value': np.asarray(series.to_numpy(), dtype=float).ravel(),
            'metric': base,
            'split': split,
        }))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def plot_training_history(model, out_dir, target, logger, save_plot=True):
    """
    Persist the training history as a CSV and (when ``save_plot``) draw the
    loss curves, faceted per metric with train vs validation overlaid so the
    train/validation gap -- the overfitting signal -- reads directly off the
    figure. Returns the tidy frame, or None if the model logged no history.
    """
    tidy = _tidy_history(getattr(model, 'history', None))
    if tidy is None:
        logger.warning('[DIAG] %s: model.history is empty -- no training-loss curves.', target)
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / '{}_training_history.csv'.format(target)
    tidy.to_csv(csv, index=False)
    logger.info('[DIAG] %s: wrote training history to %s', target, csv)

    if save_plot:
        plot = (ggplot(tidy, aes('epoch', 'value', color='split'))
                + geom_line()
                + facet_wrap('~metric', scales='free_y')
                + scale_color_manual(values={'train': '#1f77b4', 'validation': '#d62728',
                                             'other': '#555555'})
                + theme_bw()
                + labs(title='{}: scVI training history'.format(target),
                       x='epoch', y='', color=''))
        png = out_dir / '{}_training_history.png'.format(target)
        plot.save(str(png), dpi=200, width=9, height=6, verbose=False)
        logger.info('[DIAG] %s: wrote training-history plot to %s', target, png)
    return tidy


def batch_mixing_metrics(adata, latent_key, batch_key='batch', n_neighbors=_METRIC_N_NEIGHBORS,
                         out_dir=None, target=None, logger=None, save_plot=True):
    """
    Label-free batch-mixing on ``adata.obsm[latent_key]`` keyed by ``batch_key``:
    iLISI (scaled 0-1, 1 = perfectly mixed) and kBET acceptance rate (0-1, higher
    = more mixing). Returns a flat dict of scalars, or None when the inputs make
    the metrics undefined (missing key, <2 batches).

    Both need a kNN graph over the latent; it is built once with pynndescent at
    scib's conventional neighbour count and shared by the two.
    """
    logger = logger or logging.getLogger('IntegrateLogging')
    # Imported lazily: scib-metrics pulls in jax, and this module is imported by
    # the orchestrator whether or not diagnostics run.
    from scib_metrics import ilisi_knn, kbet
    from scib_metrics.nearest_neighbors import pynndescent

    if latent_key not in adata.obsm:
        logger.warning('[DIAG] latent key %r not in .obsm -- skipping batch-mixing metrics.', latent_key)
        return None
    if batch_key not in adata.obs:
        logger.warning('[DIAG] batch key %r not in .obs -- skipping batch-mixing metrics.', batch_key)
        return None

    batches = adata.obs[batch_key].astype(str).to_numpy()
    n_batches = int(np.unique(batches).size)
    if n_batches < 2:
        logger.info('[DIAG] only %d batch(es) present -- batch-mixing metrics are undefined, '
                    'skipping (this is normal for a single-sample run).', n_batches)
        return None

    X = np.asarray(adata.obsm[latent_key], dtype=np.float32)
    k = int(min(n_neighbors, X.shape[0] - 1))
    n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or 1
    logger.info('[DIAG] Building %d-NN graph on %r for batch-mixing (%d cells, %d batches).',
                k, latent_key, X.shape[0], n_batches)
    neighbors = pynndescent(X, n_neighbors=k, random_state=0, n_jobs=n_jobs)

    result = {'latent_key': latent_key, 'batch_key': batch_key,
              'n_cells': int(X.shape[0]), 'n_batches': n_batches, 'n_neighbors': k,
              'ilisi': float('nan'), 'kbet_acceptance': float('nan')}
    try:
        result['ilisi'] = float(ilisi_knn(neighbors, batches))
    except Exception as e:
        logger.warning('[DIAG] iLISI computation failed on %r: %s', latent_key, e)
    try:
        kb = kbet(neighbors, batches)
        result['kbet_acceptance'] = float(kb[0] if isinstance(kb, (tuple, list)) else kb)
    except Exception as e:
        logger.warning('[DIAG] kBET computation failed on %r: %s', latent_key, e)

    logger.info('[DIAG] %s batch mixing on %r: iLISI=%.3f, kBET acceptance=%.3f '
                '(both 0-1, higher = better mixed).',
                target or '', latent_key, result['ilisi'], result['kbet_acceptance'])

    if out_dir is not None and target is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tsv = out_dir / '{}_batch_mixing.tsv'.format(target)
        pd.DataFrame([result]).to_csv(tsv, sep='\t', index=False)
        logger.info('[DIAG] %s: wrote batch-mixing metrics to %s', target, tsv)
        if save_plot:
            bars = pd.DataFrame({'metric': ['iLISI', 'kBET acceptance'],
                                 'score': [result['ilisi'], result['kbet_acceptance']]})
            bars['label'] = bars['score'].map(lambda v: '{:.3f}'.format(v))
            plot = (ggplot(bars, aes('metric', 'score', fill='metric'))
                    + geom_col(width=0.6)
                    + geom_text(aes(label='label'), nudge_y=0.03, size=10)
                    + ylim(0, 1)
                    + guides(fill=False)
                    + theme_bw()
                    + labs(title='{}: batch mixing on {}\n(1 = batches well mixed)'.format(
                        target, latent_key), x='', y='score'))
            png = out_dir / '{}_batch_mixing.png'.format(target)
            plot.save(str(png), dpi=200, width=5, height=4, verbose=False)
            logger.info('[DIAG] %s: wrote batch-mixing plot to %s', target, png)
    return result


def run_diagnostics(adata, method, out_dir, target, model=None, batch_key='batch',
                    latent_keys=None, logger=None, save_plot=True):
    """
    Entry point used after dimensionality reduction by the orchestrator and by refine.

    Training-loss curves run only when a `model` carrying history is supplied
    (the scVI path); batch-mixing runs on every latent `.obsm` for the method,
    or on `latent_keys` when given. Every piece is wrapped so a
    diagnostics failure warns and moves on rather than costing the trained model
    -- the model, its h5ad and the UMAP are the deliverables, diagnostics are an
    annotation on top. The batch-mixing scalars are stamped into
    ``.uns['integration_batch_mixing']`` so they travel with the saved h5ad.
    """
    logger = logger or logging.getLogger('IntegrateLogging')

    if model is not None:
        try:
            plot_training_history(model, out_dir, target, logger, save_plot=save_plot)
        except Exception as e:
            logger.warning('[DIAG] %s: training-history diagnostics failed: %s', target, e)

    if latent_keys is None:
        latent_keys = [k for k in adata.obsm if k.startswith('X_{}'.format(method))]
    metrics = {}
    for key in latent_keys:
        try:
            result = batch_mixing_metrics(adata, key, batch_key=batch_key, out_dir=out_dir,
                                          target=target, logger=logger, save_plot=save_plot)
            if result is not None:
                metrics[key] = result
        except Exception as e:
            logger.warning('[DIAG] %s: batch-mixing diagnostics failed on %r: %s', target, key, e)
    if metrics:
        adata.uns['integration_batch_mixing'] = metrics
    return metrics
