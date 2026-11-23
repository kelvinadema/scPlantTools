#!/usr/bin/env python3

"""
contrastiveVI: one Background-vs-Target contrast per target.

A third integration method alongside scVI and scArches, and a *method* rather
than another `model_class`, because `scvi.external.ContrastiveVI` does not fit
the SCVI shape the rest of `integration/` is built on:

  * it subclasses `BaseModelClass`, not `scvi.model.SCVI`, so it inherits neither
    `ArchesMixin` nor the constructor/`train()` signature `SingleCellIntegrate`
    drives;
  * `train()` takes two REQUIRED positional arguments -- `background_indices` and
    `target_indices` -- while `SingleCellIntegrate._integrate_scvi` calls
    `model.train(**train_args)` with none;
  * it produces TWO latents, not one: a `background` space holding variation
    shared with the controls, and a `salient` space that should hold only
    target-specific variation.

(DRVI, by contrast, *does* subclass SCVI and is therefore registered as an
ordinary `model_class` in integrationutils.MODEL_CLASSES -- see the note there.)

Not autotuned, and that is a deliberate statistical decision rather than an
unimplemented feature: `select_contrast_grid` is what replaces it. See
docs/decisions/contrastive-sweep.md.
"""

import datetime
import functools
import inspect
import logging
import os

import numpy as np
import pandas as pd
import scanpy as sc
import scvi

from pathlib import Path
from scvi.external import ContrastiveVI

from shared.utils import find_root_dir
from .integrationutils import CacheUtils, integration_tool_versions
from .integrate import normalize_kl_warmup
from shared.hvg import HVGSelector

# Output folder name, fixed like scArches' rather than derived from a model
# class -- there is only one class on this path.
METHOD_NAME = 'contrastiveVI'

# The published defaults (Weinberger et al. 2023; the scvi-tools tutorial uses
# exactly these), used when a target states no `model_params`. Written out
# rather than left to scvi's constructor so the registry entry is a complete
# record of what was trained.
#
# `n_salient_latent` and `wasserstein_penalty` are the two that matter and the
# two that must NOT be autotuned -- set them from select_contrast_grid instead.
# See docs/decisions/contrastive-sweep.md.
_MODEL_DEFAULTS = {
    'n_background_latent': 10,
    'n_salient_latent': 10,
    'n_hidden': 128,
    'n_layers': 1,
    'dropout_rate': 0.1,
    'wasserstein_penalty': 0.0,
}

# .obs column recording which side of the contrast each cell is on. Kept in the
# saved h5ad so "which cells were background" survives in the data rather than
# only in the config that produced it; every downstream read keys off it.
GROUP_KEY = 'contrastive_group'
BACKGROUND, TARGET = 'background', 'target'


@functools.lru_cache(maxsize=1)
def module_supports_covariates() -> bool:
    """Whether this scvi-tools' ContrastiveVI module actually READS covariate keys.

    Probed rather than hardcoded so a future scvi-tools that implements them starts
    working here with no edit. As of scvi-tools 1.5.0.post1 the answer is no and the
    failure is silent -- see docs/decisions/contrastive-sweep.md.
    """
    try:
        module_cls = ContrastiveVI._module_cls
        params = set(inspect.signature(module_cls.__init__).parameters)
    except Exception:      # pragma: no cover - defensive against API churn
        return False
    return bool(params & {'n_cats_per_cov', 'n_continuous_cov',
                          'n_extra_categorical_covs', 'n_extra_continuous_covs'})


class ContrastiveIntegrator:
    """Fit one contrastiveVI contrast and attach both latents to the AnnData.

    `adata` covers the whole contrast (background samples then target samples); the
    two index vectors are positions into it, which is why the orchestrator
    concatenates both sides into a single object before constructing this.
    """

    def __init__(self, adata, background_samples: list, target_samples: list, target: str,
                 covariates: dict, logger, batch_key: str = None, do_hvg: bool = False,
                 layer: str = 'corrected_counts', model_args: dict = None,
                 train_args: dict = None, registry: CacheUtils = None,
                 overwrite: bool = False, out_dir=None, persist: bool = True):
        self.adata = adata
        self.background_samples = list(background_samples)
        self.target_samples = list(target_samples)
        self.target = target
        self.covariates = covariates
        self.logger = logger
        self.batch_key = batch_key
        self.hvg = do_hvg
        self.layer = layer
        self.overwrite = overwrite
        # A disposable fit: no registry read or write, nothing on disk. Without
        # it a 15-point sweep would upsert the target's entry 15 times and leave
        # the last grid point masquerading as the chosen fit.
        self.persist = persist
        self.registry = registry or CacheUtils()

        # Same shape as SingleCellIntegrate's: {'model_params': {...}}, so a
        # target's `model_params:` block reads identically across methods.
        supplied = (model_args or {}).get('model_params') or {}
        self.model_params = {**_MODEL_DEFAULTS, **supplied}

        self.train_args = dict(train_args or {})
        # ContrastiveVI trains through the stock scvi TrainingPlan, so it needs
        # the same warmup guarantee. See docs/decisions/kl-warmup.md.
        self.train_args = normalize_kl_warmup(self.train_args, self.adata.n_obs, self.logger)

        self.out_dir = Path(out_dir) if out_dir is not None else (
            find_root_dir() / 'results/integration' / METHOD_NAME / self.target)

        self.model = None

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def _split_indices(self):
        """Positional indices of the two sides, and the `.obs` column recording them.

        Raises rather than warns on an empty or overlapping side: an overlapping sample
        would land in both index vectors, which trains without error and silently
        destroys the contrast. Also raises on a missing 'sample' column or a named
        sample absent from the AnnData. Returns (background_indices, target_indices).
        """
        if 'sample' not in self.adata.obs:
            raise KeyError(
                "[contrastiveVI] .obs has no 'sample' column, so the background/target split "
                'cannot be resolved. The orchestrator backfills it during concatenation.')

        overlap = sorted(set(self.background_samples) & set(self.target_samples))
        if overlap:
            raise ValueError(
                "[contrastiveVI] target '{}': sample(s) {} appear in BOTH Background and "
                'Target. The same cells cannot be the shared baseline and the perturbation '
                'at once -- this trains without error and silently destroys the '
                'contrast.'.format(self.target, ', '.join(overlap)))

        samples = self.adata.obs['sample'].astype(str).to_numpy()
        present = set(np.unique(samples))
        missing = sorted((set(self.background_samples) | set(self.target_samples)) - present)
        if missing:
            raise ValueError(
                "[contrastiveVI] target '{}': sample(s) {} are named in Background/Target but "
                'absent from the loaded AnnData.'.format(self.target, ', '.join(missing)))

        is_background = np.isin(samples, self.background_samples)
        is_target = np.isin(samples, self.target_samples)

        for label, mask, names in ((BACKGROUND, is_background, self.background_samples),
                                   (TARGET, is_target, self.target_samples)):
            if not mask.any():
                raise ValueError(
                    "[contrastiveVI] target '{}': the {} side ({}) contributed no cells.".format(
                        self.target, label, ', '.join(names)))

        group = np.full(samples.shape, '', dtype=object)
        group[is_background] = BACKGROUND
        group[is_target] = TARGET
        self.adata.obs[GROUP_KEY] = pd.Categorical(group)

        background_indices = np.flatnonzero(is_background).tolist()
        target_indices = np.flatnonzero(is_target).tolist()
        self.logger.info(
            '[contrastiveVI] %s: %d background cell(s) from %s, %d target cell(s) from %s.',
            self.target, len(background_indices), ', '.join(self.background_samples),
            len(target_indices), ', '.join(self.target_samples))
        return background_indices, target_indices

    def _prep(self):
        """Register the AnnData with ContrastiveVI. Mirrors SingleCellIntegrate._prep_scvi.

        Covariates are dropped rather than forwarded when the module cannot read them,
        so the saved model does not claim a correction it never applied. See
        docs/decisions/contrastive-sweep.md.
        """
        if self.layer is not None:
            self.adata.X = self.adata.layers[self.layer].copy()

        categorical = [k for k in self.covariates.get('categorical') or []]
        continuous = [k for k in self.covariates.get('continuous') or []]

        if not module_supports_covariates():
            dropped = list(categorical) + list(continuous)
            if dropped:
                self.logger.warning(
                    '[contrastiveVI] %s: IGNORING %d covariate(s) -- %s. ContrastiveVI in '
                    'scvi-tools %s accepts covariate keys but its module never reads them, so '
                    'these correct NOTHING. Dropped rather than forwarded, so the saved model '
                    'does not claim a correction it did not apply. batch_key (%r) is the ONE '
                    'nuisance slot this model has and does work. For variation you cannot put '
                    'there: check it post hoc by correlating the salient dimensions against it, '
                    'and/or raise wasserstein_penalty -- the only lever this model gives you '
                    'against technical structure leaking into the salient space.',
                    self.target, len(dropped), ', '.join(dropped),
                    getattr(scvi, '__version__', '?'), self.batch_key)
            categorical, continuous = [], []

        for k in ([self.batch_key] if self.batch_key else []) + categorical:
            if k in self.adata.obs:
                self.adata.obs[k] = self.adata.obs[k].astype('category')
            else:
                self.logger.info('[contrastiveVI] {} is an unknown key, ignoring'.format(k))
        for k in continuous:
            if k not in self.adata.obs:
                self.logger.info('[contrastiveVI] {} is an unknown key, ignoring'.format(k))

        if self.hvg:
            hvg_path = self.out_dir / '{}_HVG.tsv'.format(self.target)
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.adata = HVGSelector(self.adata, hvg_path).select_hvg()
            if self.persist:
                self.registry.upsert_entry(self.target, {'hvg_path': str(hvg_path)})

        ContrastiveVI.setup_anndata(
            self.adata,
            layer=self.layer,
            batch_key=self.batch_key if (self.batch_key and self.batch_key in self.adata.obs) else None,
            categorical_covariate_keys=[k for k in categorical if k in self.adata.obs],
            continuous_covariate_keys=[k for k in continuous if k in self.adata.obs])

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def integrate(self):
        """Train (or reload) this contrast; returns the AnnData with both latents in `.obsm`."""
        entry = self.registry.find_entry(self.target) if self.persist else None
        if not self.overwrite and entry and 'model_path' in entry and Path(entry['model_path']).exists():
            adata_path = Path(entry.get('anndata_path', ''))
            self.logger.info('[contrastiveVI] Found existing model at %s, loading...',
                             entry['model_path'])
            try:
                self.adata = sc.read_h5ad(adata_path)
                self.model = ContrastiveVI.load(entry['model_path'], adata=self.adata)
                return self.adata
            except Exception as e:
                # Same tolerance the scArches reload branch has: a run killed
                # mid-write can leave an entry pointing at a truncated .h5ad,
                # and a target must not be permanently stuck on it.
                self.logger.warning(
                    "[contrastiveVI] Cached model/anndata for '%s' could not be reloaded (%s); "
                    'retraining from scratch.', self.target, e)

        background_indices, target_indices = self._split_indices()
        # _prep may subset genes (HVG), which rebuilds .obs but never reorders
        # rows, so the indices computed above stay valid.
        self._prep()

        self.logger.info('[contrastiveVI] Creating model with parameters: %s', self.model_params)
        model = ContrastiveVI(self.adata, **self.model_params)
        self.logger.info('[contrastiveVI] Training with train args: %s', self.train_args)
        model.train(background_indices=background_indices, target_indices=target_indices,
                    **self.train_args)
        self.model = model

        self._save(model)
        return self.adata

    def _save(self, model):
        """Attach both latents, then persist the model, h5ad and registry entry.

        The latents are attached before the `persist` gate: a disposable grid fit still
        needs them, since scoring it is the entire point.
        """
        # Both, under keys sharing the `X_contrastiveVI` prefix so
        # _dispatch_reduction UMAPs each. Reading either alone is a mistake: the
        # salient space is only interpretable against the background space
        # having absorbed the shared variation.
        for kind in ('salient', 'background'):
            self.adata.obsm['X_{}_{}'.format(METHOD_NAME, kind)] = \
                model.get_latent_representation(representation_kind=kind)

        if not self.persist:
            self.logger.info(
                '[contrastiveVI] persist=False: latents attached, nothing written to disk or '
                'the registry.')
            return

        self.out_dir.mkdir(parents=True, exist_ok=True)
        model.save(self.out_dir, overwrite=True)

        # Same atomic write the scArches path uses: a run killed mid-write must
        # leave no file at the trusted path rather than a truncated one.
        final_path = self.out_dir / '{}.h5ad'.format(self.target)
        tmp_path = self.out_dir / '.{}.h5ad.partial-{}'.format(self.target, os.getpid())
        self.adata.write_h5ad(tmp_path)
        os.replace(tmp_path, final_path)
        self.logger.info('[contrastiveVI] Saved anndata at path: %s', final_path)

        # Written last, only once the anndata write above completed -- see the
        # same ordering note in refmapping._map_query_ref.
        self.registry.upsert_entry(self.target, {
            'method': METHOD_NAME,
            'background_samples': self.background_samples,
            'target_samples': self.target_samples,
            'samples': self.adata.obs['sample'].unique().tolist(),
            'model_path': str(self.out_dir),
            'anndata_path': str(final_path),
            'model_params': dict(self.model_params),
            'batch_key': self.batch_key,
            'tool_versions': integration_tool_versions(),
            'timestamp': datetime.datetime.now().isoformat()})


# ----------------------------------------------------------------------
# Model selection -- what replaces autotuning
# ----------------------------------------------------------------------

def contrast_separation_score(adata, logger=None, out_dir=None, target=None):
    """Score one fitted contrast on whether the two latents did their separate jobs.

    The contract is two-sided: the salient space should SEPARATE the two arms (low
    iLISI) while the background space should not (high iLISI). This is a screen
    against degenerate fits, not evidence that the salient axis is biological --
    see docs/decisions/contrastive-sweep.md.

    Returns 'ilisi_salient', 'ilisi_background' and 'separation' = background
    minus salient, in [-1, 1]; at or below 0 the fit is degenerate. With both
    `out_dir` and `target`, also writes `{target}_contrast_separation.tsv`.
    """
    from .diagnostics import batch_mixing_metrics

    logger = logger or logging.getLogger('IntegrateLogging')
    if GROUP_KEY not in adata.obs:
        raise KeyError(
            "[contrastiveVI] .obs has no '{}' column -- this scores a fitted contrast, and "
            'the column is written by ContrastiveIntegrator._split_indices.'.format(GROUP_KEY))

    scores = {}
    for kind in ('salient', 'background'):
        key = 'X_{}_{}'.format(METHOD_NAME, kind)
        # batch_mixing_metrics is label-agnostic -- it mixes on whatever .obs
        # column it is handed -- so pointing it at the contrast group rather
        # than `batch` asks exactly the right question here.
        result = batch_mixing_metrics(adata, key, batch_key=GROUP_KEY, logger=logger,
                                      save_plot=False)
        scores[kind] = result['ilisi'] if result else float('nan')

    out = {'target': target,
           'ilisi_salient': scores['salient'],
           'ilisi_background': scores['background'],
           'separation': scores['background'] - scores['salient']}
    logger.info(
        '[contrastiveVI] %s: iLISI salient=%.3f (want LOW), background=%.3f (want HIGH), '
        'separation=%.3f (higher = cleaner split; <=0 means the fit is degenerate).',
        target or '', out['ilisi_salient'], out['ilisi_background'], out['separation'])

    if out_dir is not None and target is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tsv = out_dir / '{}_contrast_separation.tsv'.format(target)
        pd.DataFrame([out]).to_csv(tsv, sep='\t', index=False)
        logger.info('[contrastiveVI] %s: wrote separation score to %s', target, tsv)
    return out


# Shared by `contrast_leakage_score` and `select_contrast_grid` so a bare call
# to the scorer measures the same thing the sweep does; None would give a
# direct caller NaN for the two metrics that matter most.
DEFAULT_IDENTITY_KEY = 'leiden_scVI'
DEFAULT_QC_KEYS = ('mitochondrial_fraction_corrected_counts',
                   'chloroplast_fraction_corrected_counts',
                   'ribosomal_rna_fraction_corrected_counts')


def contrast_leakage_score(adata, identity_key: str = DEFAULT_IDENTITY_KEY,
                           qc_keys=DEFAULT_QC_KEYS,
                           n_clusters: int = 10, n_permutations: int = 20,
                           seed: int = 0, logger=None):
    """Does the salient space hold structure it has no right to hold?

    `contrast_separation_score` asks whether the two latents did their separate
    jobs; this asks whether the salient space is carrying CELL IDENTITY or TECHNICAL
    variation rather than treatment response. A fit can separate the contrast
    cleanly and still be a cell-type map.

    Measured on BACKGROUND cells only, which is what makes it interpretable: those
    cells received no perturbation, so any structure in their salient coordinates is
    leakage by construction. See docs/decisions/contrastive-sweep.md.

    `n_clusters` is k for the k-means partition of the salient space;
    `n_permutations` draws form the NMI null.

    Returns:
        dict: 'leakage_nmi', 'leakage_ratio' (salient vs background NMI -- the
            figure to read), 'qc_max_abs_rho', 'responder_frac',
            'salient_eff_dim' (participation ratio; ~1 means the space is one
            axis), and 'collapsed'. NaN for any metric whose inputs are absent.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score
    from scipy.stats import spearmanr

    logger = logger or logging.getLogger('IntegrateLogging')
    rng = np.random.default_rng(seed)
    out = {'leakage_nmi': float('nan'), 'leakage_null': float('nan'),
           'background_nmi': float('nan'), 'leakage_ratio': float('nan'),
           'qc_max_abs_rho': float('nan'), 'qc_worst_key': None,
           'responder_frac': float('nan'), 'salient_eff_dim': float('nan'),
           'collapsed': False}

    if GROUP_KEY not in adata.obs:
        raise KeyError("[contrastiveVI] .obs has no '{}' column.".format(GROUP_KEY))
    sal_key, bg_key = ('X_{}_salient'.format(METHOD_NAME), 'X_{}_background'.format(METHOD_NAME))
    if sal_key not in adata.obsm:
        raise KeyError('[contrastiveVI] .obsm has no {!r}.'.format(sal_key))

    group = adata.obs[GROUP_KEY].astype(str).to_numpy()
    is_bg, is_tg = group == BACKGROUND, group == TARGET
    salient = np.asarray(adata.obsm[sal_key])

    # Responder fraction -- how much of the target side actually moved.
    if is_bg.any() and is_tg.any():
        norms = np.linalg.norm(salient, axis=1)
        out['responder_frac'] = float((norms[is_tg] > np.quantile(norms[is_bg], 0.95)).mean())

    # On TARGET cells: background cells are meant to sit at the origin, so
    # including them would measure the gap between the arms rather than the
    # structure within the responding one.
    if is_tg.sum() > 2:
        centred = salient[is_tg] - salient[is_tg].mean(axis=0, keepdims=True)
        sv = np.linalg.svd(centred, compute_uv=False) ** 2
        total = float(sv.sum())
        if total > 0:
            out['salient_eff_dim'] = float(total ** 2 / float((sv ** 2).sum()))

    # Collapse: separated to the point of carrying nothing. BOTH conditions
    # must hold -- each alone has an innocent reading, and only together do they
    # say the salient space is one axis every target cell is far along, i.e. a
    # group indicator. See docs/decisions/contrastive-sweep.md.
    if (salient.shape[1] > 1 and np.isfinite(out['responder_frac'])
            and np.isfinite(out['salient_eff_dim'])):
        out['collapsed'] = bool(out['responder_frac'] > 0.95
                                and out['salient_eff_dim'] < 2.0)

    # Identity leakage, background cells only.
    if identity_key and identity_key in adata.obs and is_bg.sum() > n_clusters:
        ident = adata.obs[identity_key].astype(str).to_numpy()[is_bg]
        k = int(min(n_clusters, len(set(ident))))
        if k >= 2:
            def _nmi_against(matrix):
                part = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(matrix[is_bg])
                return float(normalized_mutual_info_score(part, ident)), part
            out['leakage_nmi'], sal_part = _nmi_against(salient)
            if bg_key in adata.obsm:
                out['background_nmi'], _ = _nmi_against(np.asarray(adata.obsm[bg_key]))
                if out['background_nmi'] > 0:
                    out['leakage_ratio'] = out['leakage_nmi'] / out['background_nmi']
            # Null: the NMI floor for this k and these class sizes is not 0.
            null = [normalized_mutual_info_score(sal_part, rng.permutation(ident))
                    for _ in range(n_permutations)]
            out['leakage_null'] = float(np.mean(null))

    # QC association of the salient axes.
    for key in (qc_keys or []):
        if key not in adata.obs:
            continue
        values = pd.to_numeric(adata.obs[key], errors='coerce').to_numpy()
        ok = np.isfinite(values)
        if ok.sum() < 10:
            continue
        for dim in range(salient.shape[1]):
            rho = spearmanr(salient[ok, dim], values[ok]).statistic
            if np.isfinite(rho) and (not np.isfinite(out['qc_max_abs_rho'])
                                     or abs(rho) > out['qc_max_abs_rho']):
                out['qc_max_abs_rho'], out['qc_worst_key'] = float(abs(rho)), key

    logger.info(
        '[contrastiveVI] leakage: nmi=%.3f (null %.3f, background %.3f, ratio %.2f) '
        'qc_max|rho|=%.3f (%s) responders=%.1f%%',
        out['leakage_nmi'], out['leakage_null'], out['background_nmi'],
        out['leakage_ratio'] if np.isfinite(out['leakage_ratio']) else float('nan'),
        out['qc_max_abs_rho'], out['qc_worst_key'],
        100 * out['responder_frac'] if np.isfinite(out['responder_frac']) else float('nan'))
    return out


def select_contrast_grid(adata, background_samples, target_samples, target, grid,
                         covariates=None, batch_key=None, layer='corrected_counts',
                         train_args=None, base_model_params=None,
                         identity_key=DEFAULT_IDENTITY_KEY, qc_keys=DEFAULT_QC_KEYS,
                         out_dir=None, logger=None, append=True, seeds=(0,)):
    """Fit a `n_salient_latent` x `wasserstein_penalty` grid and rank it.

    This is what replaces autotuning; every criterion is computed OUTSIDE the
    objective, because contrastiveVI's own objective rewards the degenerate fit.
    Each point is a disposable fit on a copy of `adata` -- nothing is written to
    disk and the target's registry entry is never touched. Choosing is separate:
    the `sweep: true` orchestrator path trains `best_grid_point`'s choice and
    records it in the registry; otherwise the parameters go into the config by
    hand. See docs/decisions/contrastive-sweep.md.

    `grid` is a list of `model_params` overrides. `base_model_params` (the
    architecture the winner will be trained with) is not optional in practice.
    `append` accumulates onto an existing `{target}_contrast_grid.tsv` in
    `out_dir`, since repeated points are replication. Returns the ranked
    results, one row per (point, seed).
    """
    logger = logger or logging.getLogger('IntegrateLogging')
    covariates = covariates or {'categorical': [], 'continuous': []}
    # One fit per point per seed, so best_grid_point can take a median across
    # replicates instead of a lucky fit.
    seeds = tuple(seeds) if seeds else (0,)
    jobs = [(point, seed) for point in grid for seed in seeds]

    rows = []
    for i, (point, seed) in enumerate(jobs, 1):
        params = {**(base_model_params or {}), **point}
        logger.info('[contrastiveVI] grid %d/%d for %s: %s seed=%s',
                    i, len(jobs), target, point, seed)
        # Set per fit rather than once, so a given (point, seed) reproduces
        # regardless of where it sits in the loop.
        scvi.settings.seed = int(seed)
        # A fresh view per point: _prep overwrites .X from `layer` and
        # setup_anndata stamps _scvi_* columns, so reusing one object across
        # fits would carry the previous point's registration into the next.
        work = adata.copy()
        try:
            integrator = ContrastiveIntegrator(
                work, background_samples=background_samples, target_samples=target_samples,
                target=target, covariates=covariates, logger=logger, batch_key=batch_key,
                layer=layer, model_args={'model_params': params}, train_args=train_args,
                persist=False)
            fitted = integrator.integrate()
            row = {**point, 'seed': int(seed)}
            row.update(contrast_separation_score(fitted, logger=logger))
            row.update(contrast_leakage_score(fitted, identity_key=identity_key,
                                              qc_keys=qc_keys, logger=logger))
            row.pop('target', None)
            rows.append(row)
        except Exception as e:
            # One bad grid point must not cost the whole sweep -- the failure is
            # recorded as a row so it is visible in the table rather than only
            # in the log.
            logger.warning('[contrastiveVI] grid point %s seed=%s failed: %s', point, seed, e)
            rows.append({**point, 'seed': int(seed), 'error': str(e)[:200]})
        finally:
            del work

    df = pd.DataFrame(rows)
    df['run'] = datetime.datetime.now().isoformat(timespec='seconds')

    tsv = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tsv = out_dir / '{}_contrast_grid.tsv'.format(target)

    # ACCUMULATE rather than overwrite: these contrasts take hours per grid, and
    # a repeated point is replication rather than waste. The `run` column keeps
    # replicates distinguishable; pass append=False for a clean table.
    if append and tsv is not None and tsv.exists():
        try:
            previous = pd.read_csv(tsv, sep='\t')
            before = len(previous)
            df = pd.concat([previous, df], ignore_index=True)
            logger.info('[contrastiveVI] %s: appended %d new point(s) to %d existing.',
                        target, len(rows), before)
        except Exception as e:
            logger.warning('[contrastiveVI] %s: could not read existing grid at %s (%s); '
                           'writing this run only.', target, tsv, e)

    df = rank_contrast_grid(df)
    if tsv is not None:
        df.to_csv(tsv, sep='\t', index=False)
        logger.info('[contrastiveVI] %s: wrote %d grid row(s) to %s', target, len(df), tsv)
    return df


def rank_contrast_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Order a grid best-first, and refuse to be fooled by a collapsed fit.

    Two-stage: collapsed points sort last whatever their separation (they maximise
    it by construction), and everything else sorts on
    `quality = separation - leakage_ratio`. Collapsed points are kept, not dropped --
    they are the evidence that the penalty range was too high. `quality` is a
    convenience, not an authority. See docs/decisions/contrastive-sweep.md.
    Returns a sorted copy with 'quality' and 'collapsed' present.
    """
    if df.empty:
        return df
    out = df.copy()
    if 'collapsed' not in out.columns:
        out['collapsed'] = False
    out['collapsed'] = out['collapsed'].fillna(False).astype(bool)
    if {'separation', 'leakage_ratio'}.issubset(out.columns):
        out['quality'] = out['separation'] - out['leakage_ratio']
    elif 'separation' in out.columns:
        out['quality'] = out['separation']
    else:
        out['quality'] = float('nan')
    # A failed grid point (an `error` row) has no metrics; keep it visible at the
    # bottom rather than letting NaN sort it anywhere.
    return out.sort_values(['collapsed', 'quality'], ascending=[True, False],
                           na_position='last').reset_index(drop=True)


def best_grid_point(df: pd.DataFrame, param_keys=('wasserstein_penalty', 'n_salient_latent')):
    """The chosen `model_params` from a ranked grid, or None if nothing is usable.

    Replicates are aggregated by MEDIAN rather than taken at face value, and a point
    counts as collapsed if the majority of its replicates collapsed. Returns None
    when every point collapsed rather than handing back the least bad one: that grid
    has not answered the question it was asked.
    """
    if df is None or df.empty:
        return None
    ranked = rank_contrast_grid(df)
    usable = ranked[ranked['quality'].notna()]
    if 'error' in usable.columns:
        usable = usable[usable['error'].isna()]
    keys = [k for k in param_keys if k in usable.columns]
    if usable.empty or not keys:
        return None

    agg = (usable.groupby(keys, dropna=False)
                 .agg(quality=('quality', 'median'),
                      collapsed=('collapsed', 'mean'),
                      n=('quality', 'size'))
                 .reset_index())
    agg = agg[agg['collapsed'] <= 0.5]
    if agg.empty:
        return None
    row = agg.sort_values('quality', ascending=False).iloc[0]
    return {k: (float(row[k]) if k == 'wasserstein_penalty' else int(row[k]))
            for k in keys if pd.notna(row[k])}


def summarise_contrast_grid(df: pd.DataFrame,
                            param_keys=('wasserstein_penalty', 'n_salient_latent')):
    """One row per parameter combination, with the spread across seeds.

    The point of replication is not a better point estimate, it is knowing whether
    two points differ by more than the noise. `quality_range` is the column to read
    before believing any ranking: where it is wider than the gap between the top
    rows, the grid has not distinguished them.
    """
    if df is None or df.empty:
        return df
    ranked = rank_contrast_grid(df)
    if 'error' in ranked.columns:
        ranked = ranked[ranked['error'].isna()]
    keys = [k for k in param_keys if k in ranked.columns]
    if not keys:
        return ranked

    out = (ranked.groupby(keys, dropna=False)
                 .agg(n=('quality', 'size'),
                      quality_median=('quality', 'median'),
                      quality_min=('quality', 'min'),
                      quality_max=('quality', 'max'),
                      separation_median=('separation', 'median'),
                      leakage_median=('leakage_ratio', 'median'),
                      leakage_min=('leakage_ratio', 'min'),
                      leakage_max=('leakage_ratio', 'max'),
                      eff_dim_median=('salient_eff_dim', 'median'),
                      collapsed_frac=('collapsed', 'mean'))
                 .reset_index())
    out['quality_range'] = out['quality_max'] - out['quality_min']
    out['leakage_range'] = out['leakage_max'] - out['leakage_min']
    return out.sort_values('quality_median', ascending=False).reset_index(drop=True)
