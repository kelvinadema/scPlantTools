#!/usr/bin/env python3

"""
Per-sample doublet model: train scVI via the shared integration code, run SOLO.

    python -m integration.doublet_model --input in.h5ad --output out.tsv

Runs under the **integration** conda env, invoked as a subprocess by
`mapping/doublets.py`. It lives in `integration/` rather than
`mapping/` for a specific reason: this file executes under pandas 2.3.x,
and everything in `mapping/` targets pandas 3.x. `integration/`
and `shared/` already run under this interpreter on every integration run, so
importing them here is safe -- importing anything from `mapping/` would
not be.

It reuses `SingleCellIntegrate` rather than re-implementing scVI setup. The
earlier standalone worker duplicated that logic on a rationale ("import nothing
from this repo") that was stricter than the real constraint, and duplicated code
is how the two drift apart.

Contract with the caller, deliberately narrow:

    in   an h5ad whose .X is counts for the cells and genes the caller chose;
         cell filtering and HVG selection happen on the caller's side, so the
         file crossing the boundary is small
    out  a TSV indexed by barcode: solo_doublet_score, solo_doublet, umap1,
         umap2, leiden

The model is **disposable**: `persist=False` means nothing is written to disk and
nothing touches `scvi_tune_registry.json` (originally because it had no file
locking and 40 samples registering models in one mapping run would race; it is
locked as of 2026-07-28, but a model deleted at the end of the sample still does
not belong in a registry that exists to locate artifacts). Its parameters go
back to the caller in the JSON summary and end up in `.uns['qc']['solo']`.

**No covariates, ever.** `SOLO.from_scvi_model` raises on a source model
carrying any registered categorical or continuous covariate. That is a
scvi-tools limitation, not a config choice, and it applies however this model is
trained -- including when `tune` is on.
"""

import argparse
import json
import logging
import sys
import tempfile
import warnings



def _log(msg):
    """Progress to stderr; stdout carries the machine-readable summary only."""
    print('[doublet-model] {}'.format(msg), file=sys.stderr, flush=True)


def _jsonable(obj):
    """
    Coerce a resolved model-parameter dict to something json.dumps accepts.

    Ray Tune hands back the winning config with numpy scalars in it (hyperopt
    samples `tune.choice` into np.int64/np.float64), which json.dumps refuses
    outright -- so without this the summary line is unparseable and the caller
    silently falls back to its own defaults, which is exactly the wrong-record
    problem this dict exists to fix.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, bool)) or obj is None:
        return obj
    item = getattr(obj, 'item', None)          # numpy scalar -> python scalar
    if callable(item):
        try:
            return item()
        except (ValueError, TypeError):
            pass
    if isinstance(obj, (int, float)):
        return obj
    return str(obj)


def main() -> int:
    """Worker entry point: train a throwaway model and write doublet calls.

    Runs in the integration env as a subprocess of the mapping stage, because
    scvi-tools cannot be installed alongside it.

    Returns:
        int: Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='h5ad with counts in .X')
    parser.add_argument('--output', required=True, help='TSV to write')
    parser.add_argument('--sample', default='sample', help='Used only to name the transient target')
    parser.add_argument('--n-latent', type=int, default=10)
    parser.add_argument('--n-hidden', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--n-epochs-kl-warmup', type=int, default=None,
                        help='KL warmup length in epochs, passed via plan_kwargs. Keep below '
                             'max_epochs (or scvi\'s cell-count heuristic) so warmup completes and '
                             'the latent settles -- an under-regularized latent can push SOLO '
                             'toward calling 100%% doublets (scvi-tools #507).')
    parser.add_argument('--tune', action='store_true',
                        help='Autotune hyperparameters instead of using the fixed defaults. '
                             'Costly: this runs per sample.')
    parser.add_argument('--n-trials', type=int, default=20)
    parser.add_argument('--tune-metric', default='elbo_validation')
    parser.add_argument('--doublet-threshold', type=float, default=0.5)
    parser.add_argument('--leiden-resolution', type=float, default=1.0)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scvi

    from .integrate import SingleCellIntegrate
    from shared.dimred import DimReducer

    logging.basicConfig(level=logging.INFO, format='[doublet-model] %(message)s', stream=sys.stderr)
    logger = logging.getLogger('doublet-model')
    scvi.settings.seed = args.random_state

    adata = sc.read_h5ad(args.input)
    _log('loaded {} cells x {} genes'.format(adata.n_obs, adata.n_vars))

    # scVI's likelihood is defined on counts. The caller sends whichever layer
    # was configured and CellBender's corrected output is float, so round here
    # rather than let scvi-tools warn per batch about non-integer input.
    X = adata.X
    data = X.data if hasattr(X, 'data') else X
    if data.size and not np.allclose(data, np.rint(data)):
        _log('input is non-integer (CellBender-corrected counts?); rounding for the NB likelihood')
        if hasattr(X, 'data'):
            X.data = np.rint(X.data)
        else:
            adata.X = np.rint(X)

    # SingleCellIntegrate keys its registry lookups and output folder off the
    # target name. With persist=False neither is written, but a scratch out_dir
    # keeps even an incidental write out of results/integration/.
    # scvi train() kwargs. max_epochs and the KL-warmup control both live here; a
    # warmup longer than max_epochs leaves the latent under-regularized (scvi-tools
    # #507), which on a small per-sample doublet model can feed the unconverged
    # 100%-doublet failure mode. n_epochs_kl_warmup reaches the FINAL fit through
    # this dict and the tuning trials through SingleCellIntegrate's search space,
    # so setting it here covers both tuned and untuned runs.
    train_args = {}
    if args.max_epochs is not None:
        train_args['max_epochs'] = args.max_epochs
    if args.n_epochs_kl_warmup is not None:
        train_args['plan_kwargs'] = {'n_epochs_kl_warmup': args.n_epochs_kl_warmup}

    with tempfile.TemporaryDirectory(prefix='doublet_') as scratch:
        integrator = SingleCellIntegrate(
            adata=adata,
            target='_doublet_{}'.format(args.sample),
            do_tune=args.tune,
            do_hvg=False,            # the caller already selected HVGs
            output_dir=scratch,
            covariates={'categorical': [], 'continuous': []},   # SOLO requires none
            logger=logger,
            model_class='SCVI',
            n_trials=args.n_trials,
            tune_metric=args.tune_metric,
            model_args={'model_params': {
                'n_latent': args.n_latent,
                'n_hidden': args.n_hidden,
                'n_layers': args.n_layers}},
            # train_args (max_epochs + plan_kwargs.n_epochs_kl_warmup) is built
            # just above. It was the route missing on both sides of this boundary:
            # --max-epochs used to be parsed, echoed into the JSON summary below
            # and recorded in .uns['qc']['solo']['max_epochs'] by the caller, but
            # never reached model.train(). The integration side gets the same from
            # a target's max_epochs:/plan_kwargs:.
            train_args=(train_args or None),
            persist=False,
            out_dir=scratch)

        _log('training scVI (tune={})'.format(args.tune))
        # batch_key=None: one sample, so there is no batch structure to model,
        # and a mapping artifact carries no .obs['batch'] until integration adds it.
        adata = integrator.integrate(batch_key=None, layer=None)
        model = integrator.model

        _log('training SOLO')
        solo = scvi.external.SOLO.from_scvi_model(model)
        solo.train(accelerator='gpu' if args.cuda else 'cpu')

        # soft=True gives per-class probabilities; SOLO appends simulated
        # doublets to its own training data but predicts on the real cells, so
        # this comes back aligned to `adata`.
        pred = solo.predict(soft=True)
        if isinstance(pred, pd.DataFrame):
            column = 'doublet' if 'doublet' in pred.columns else pred.columns[-1]
            score = pred[column].astype(float)
        else:
            score = pd.Series(np.asarray(pred, dtype=float), index=adata.obs_names)
        score.index = [str(i) for i in score.index]
        score = score.reindex([str(i) for i in adata.obs_names])
        if score.isna().any():
            _log('WARNING: {} cells got no SOLO score; emitting NaN'.format(int(score.isna().sum())))

        # Embedding from the same latent space, so the report's scVI UMAP and
        # the doublet calls describe one model rather than two. Routed through
        # the shared DimReducer (the same code that builds the mapping stage's
        # PCA UMAP, X_umap_qc) so this embedding uses the *same* neighbours/
        # UMAP/Leiden settings by construction: n_neighbors = round(sqrt(N)),
        # cosine metric, min_dist = 0.0. Before this it used scanpy's bare
        # defaults (n_neighbors=15, euclidean, min_dist=0.5), which spread the
        # cells out and made the scVI UMAP look blobbier than the latent
        # warranted -- an apples-to-oranges comparison with the crisp PCA UMAP.
        # leiden_kwargs preserves the previous igraph flavour exactly.
        if 'X_scVI' not in adata.obsm:
            adata.obsm['X_scVI'] = model.get_latent_representation()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            keys = DimReducer(adata, logger=logger).compute(
                use_rep='X_scVI', metric='cosine', min_dist=0.0, do_leiden=True,
                leiden_resolution=args.leiden_resolution, random_state=args.random_state,
                leiden_kwargs={'flavor': 'igraph', 'n_iterations': 2, 'directed': False})

        umap = np.asarray(adata.obsm[keys['umap_key']])
        out = pd.DataFrame({
            'solo_doublet_score': score.values,
            'solo_doublet': (score.values > args.doublet_threshold),
            'umap1': umap[:, 0],
            'umap2': umap[:, 1],
            'leiden': adata.obs[keys['leiden_key']].astype(str).values,
        }, index=[str(i) for i in adata.obs_names])
        out.index.name = 'barcode'
        out.to_csv(args.output, sep='\t')

        n_doublets = int(np.nansum(out['solo_doublet'].values))
        _log('wrote {} ({} doublets of {} cells)'.format(args.output, n_doublets, len(out)))
        # `model_params` is what the model was ACTUALLY built with, read back off
        # the integrator rather than echoed from argv. With --tune the search
        # space owns n_latent/n_hidden/n_layers (and dropout_rate, dispersion,
        # gene_likelihood, ...), and SingleCellIntegrate._integrate_scvi never
        # reads the model_args we passed in -- so echoing our own arguments here
        # reported a model that was never trained. The caller records this dict
        # verbatim; anything not in it was not chosen here.
        print(json.dumps({
            'n_cells': int(len(out)),
            'n_doublets': n_doublets,
            'scvi_version': str(scvi.__version__),
            'tuned': bool(args.tune),
            'model_params': _jsonable(integrator.resolved_model_params),
            'max_epochs': args.max_epochs,
            'n_epochs_kl_warmup': args.n_epochs_kl_warmup,
            # The resolved neighbours/UMAP/Leiden settings, so the caller can
            # record them in .uns and the report can describe this embedding.
            'umap_params': _jsonable(keys['params']),
        }))
    return 0


if __name__ == '__main__':
    sys.exit(main())
