#!/usr/bin/env python3

import datetime
import logging
import os
import shutil
import tempfile
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import torch

# Must be set before `import ray`. Ray Tune permanently replaces
# `builtins.print`, which numba then cannot type, so every `sc.pp.neighbors`
# after a tuning run fails to JIT-compile. See
# docs/decisions/autotune-ray-runtime.md.
os.environ['RAY_TQDM_PATCH_PRINT'] = '0'

import ray

from scvi.autotune import AutotuneExperiment
from ray import tune
from pathlib import Path

# Imported rather than copied so it tracks any change scvi makes to the
# heuristic; the fallback covers a private-module move.
try:
    from scvi.model._utils import get_max_epochs_heuristic
except Exception:                                    # pragma: no cover - scvi internal move
    def get_max_epochs_heuristic(n_obs, epochs_cap=400):
        return min(round((20000 / max(n_obs, 1)) * 400), epochs_cap)
from shared.utils import find_root_dir
from .integrationutils import CacheUtils, resolve_model_class, integration_tool_versions, model_output_folder
from shared.hvg import HVGSelector

# Fraction of the effective max_epochs KL warmup must finish within. A starting
# point, not tuned. See normalize_kl_warmup.
_KL_WARMUP_MAX_FRACTION = 0.5


# ---------------------------------------------------------------------------
# Autotune search spaces
#
# Held as PLAIN DATA (lists, not tune.* objects) so the defaults below and a
# `search_space:` block in integration_config.yaml go through exactly one
# converter, `_to_sampler`, and cannot drift into meaning different things.
# The value grammar and how these defaults were chosen are in
# docs/decisions/autotune-search-spaces.md.
_BASE_SPACE = {
    "n_layers": [1, 2, 3],
    "n_latent": [10, 20, 30],
    "n_hidden": [128, 256],
    "dropout_rate": [0.1, 0.2],
    "dispersion": ["gene", "gene-batch", "gene-cell"],
    "gene_likelihood": ["zinb", "nb"],
}

# The normalization / covariate-encoding pins, which are NOT shared and sat in
# _BASE_SPACE incorrectly until 2026-08-17.
#
# They used to read `use_layer_norm: both`, `use_batch_norm: none`,
# `encode_covariates: True`, described as "DRVIModule's own defaults". The
# description was accurate; the placement was not. _BASE_SPACE is inherited by
# every class, so DRVI's convention was silently imposed on the other two.
# Measured against the actual constructors:
#
#   class               use_batch_norm  use_layer_norm  encode_covariates
#   scvi VAE (SCVI)     both            none            False
#   DRVIModule          none            both            not accepted (**kwargs)
#   JointEmbeddingVAE   not accepted    not accepted    not accepted (**kwargs)
#
# So plain SCVI was built with scvi's own normalization inverted and covariate
# encoding switched on -- three modelling decisions never chosen for it, and
# PINNED, so no search could revisit them. Nothing has used `model_class: SCVI`
# yet, which is why it went unnoticed.
#
# Split per class so a value is only set where the constructor consumes it.
# NOTHING ALREADY FITTED CHANGES: DRVI keeps exactly the values it had, and
# JointEmbeddingSCVI stops being handed three keys its module discards through
# **kwargs -- never applied, but they WERE recorded in resolved_model_params,
# so 42 registry entries claim settings that had no effect. Only SCVI's values
# change, and only for fits made from here on.
_SCVI_NORM_PINS = {
    "use_batch_norm": ["both"],
    "use_layer_norm": ["none"],
    "encode_covariates": [False],
}

_DRVI_NORM_PINS = {
    "use_layer_norm": ["both"],
    "use_batch_norm": ["none"],
    # DRVIModule takes this via **kwargs rather than by name. Kept anyway:
    # dropping it would change what resolved_model_params records for the 18
    # DRVI entries already in the registry, and a search-space edit is not the
    # place to silently rewrite their provenance.
    "encode_covariates": [True],
}

# Per model class: the base space plus that class's own overrides, so the keys
# every scVI-shaped model shares are stated once. The DRVI and
# JointEmbeddingSCVI ranges were narrowed on 2026-08-04 against the fits
# already in the tune registry -- the evidence, key by key, is in
# docs/decisions/autotune-search-spaces.md.
_SEARCH_SPACE_DEFAULTS = {
    # The registry still holds no TUNED plain-SCVI fit, so the searched ranges
    # are the scvi-tools documented space, unrevised. The three pins are not:
    # they are scvi's own constructor defaults, corrected 2026-08-17 -- see
    # _SCVI_NORM_PINS.
    "SCVI": {**_BASE_SPACE, **_SCVI_NORM_PINS},

    # No normalization pins: JointEmbeddingVAE accepts none of the three and
    # swallows them through **kwargs, so setting them only produced provenance
    # claiming settings that were never applied.
    "JointEmbeddingSCVI": {
        **_BASE_SPACE,
        "n_layers": [1, 2],
        "dispersion": ["gene-batch", "gene-cell"],
        "joint_embedding_weight": [1.0, 10.0, 100.0],
        "lambda_off_diag": [0.001, 0.01, 0.1],
        "reconstruction_weight": [0.25, 0.5, 1.0],
        "variance_weight": [0.0, 0.1],
    },

    # SysVI REPLACES the base space rather than widening it, because SysVAE
    # accepts almost none of it. It has no `dispersion` and no `gene_likelihood`
    # at all: its decoder emits a Normal and the reconstruction term is
    # `GaussianNLLLoss` on X, so the count-likelihood keys are meaningless here
    # and the normalization pins are not constructor arguments either.
    #
    # THE CYCLE AND PRIOR CONTROLS ARE PINNED, NOT SEARCHED, and it is the same
    # defect JointEmbeddingSCVI has: SysVAE.loss returns
    # `LossOutput(loss=..., reconstruction_loss=..., kl_local=...)` with the
    # cycle term in `extra_metrics['cycle_loss']`, so `elbo_validation` --
    # assembled as rec_loss + kl_local -- cannot see z_distance_cycle_weight at
    # all. Searching it would walk toward whichever value weakens the cycle
    # constraint, which is the same failure documented at length for
    # joint_embedding_weight in notebooks/03-sc-vs-sn/configs/integration.yaml.
    #
    # `prior` is pinned to sysVI's own 'vamp' for the same reason one step
    # removed: a VampPrior is strictly more flexible than a standard normal, so
    # it can only improve the likelihood the criterion reads, and a search over
    # it measures flexibility rather than integration quality. Pin, do not
    # search; override per target if you want to measure the difference.
    "SysVI": {
        "n_layers": [1, 2],
        "n_latent": [10, 20, 30],
        "n_hidden": [128, 256],
        "dropout_rate": [0.1, 0.2],
        # 'sample_feature' is SysVAE's own default.
        "out_var_mode": ["sample_feature"],
        "prior": ["vamp"],
        "n_prior_components": [5],
    },

    # DRVI accepts the whole base space (it inherits SCVI's __init__ and
    # DRVIModule takes **kwargs), so this widens rather than replaces it.
    "DRVI": {
        **_BASE_SPACE,
        **_DRVI_NORM_PINS,
        "n_layers": [2, 3, 4],
        # Kept wide deliberately: DRVI is built for an OVER-COMPLETE latent, so
        # under-sizing is the real risk, and this is the parameter the chapter
        # actually reads.
        "n_latent": [32, 64, 128],
        "n_hidden": [256, 512],
        "dispersion": ["gene-cell"],
        "gene_likelihood": ["pnb", "nb"],   # 'pnb' is DRVI's own default
        # Disentanglement controls: how the latent is split and how the
        # per-split decodings recombine.
        "n_split_latent": [None, 8],
        "split_aggregation": ["logsumexp", "mean"],
        "split_method": ["split_map", "split_mask"],
    },
}

# ---------------------------------------------------------------------------
# Model classes that depart from the scVI contract this module is built around.
# Both sets exist because the departure is SILENT otherwise: the run completes,
# the registry records what the config asked for, and the model was fitted on
# something else.

# Gaussian reconstruction likelihood, so the input must be LOG-NORMALIZED
# rather than counts. SysVAE.loss computes GaussianNLLLoss against
# REGISTRY_KEYS.X_KEY directly; handed raw counts it fits a Normal to an
# overdispersed count distribution and converges to something meaningless
# without erroring. `_ensure_gaussian_input` derives the transform.
_GAUSSIAN_INPUT_CLASSES = frozenset({'SysVI'})

# Layer `_ensure_gaussian_input` writes the derived values into. Named rather
# than reusing 'lognorm_cp10k' so it can never be confused with a layer the
# mapping stage wrote, and so its provenance is unambiguous in the saved h5ad.
_GAUSSIAN_INPUT_LAYER = 'sysvi_input_lognorm'

# Classes whose own train() refuses KL warmup. SysVI.train resets
# n_epochs_kl_warmup and n_steps_kl_warmup to 0 with a warning -- the authors
# found warmup degrades cycle-model integration. normalize_kl_warmup would
# otherwise SET a warmup for these targets, which scvi then discards, leaving
# the config and the registry both claiming a value that was never applied.
# See docs/decisions/kl-warmup.md for the invariant this is an exception to.
_NO_KL_WARMUP_CLASSES = frozenset({'SysVI'})

# Two-layer training input (`layer: [a, b]` on a target). The stacked
# (cells x 2*genes) matrix lives in this layer of a DERIVED AnnData whose var is
# the gene space doubled: one feature per (gene, layer), named with these
# suffixes. The gene-space object is kept aside and the latent is written back
# onto it. See docs/decisions/two-layer-label-spaces.md.
TWO_LAYER_INPUT = 'twolayer_input'
FEATURE_SUFFIXES = ('__s', '__u')


def stack_layers(adata, layers, suffixes=FEATURE_SUFFIXES):
    """A (cells x 2*genes) feature object from two count layers of a gene-space object.

    Feature i of the first block is gene i's count in `layers[0]`, named
    `<gene><suffixes[0]>`; the second block likewise for `layers[1]`. `.var`
    carries `gene` and `source_layer` so a feature maps back to its gene, and
    `.obs` is copied unchanged so batch/covariate columns register as usual.

    Args:
        adata (AnnData): Gene-space object carrying both layers.
        layers (list[str]): Exactly two distinct layer names.
        suffixes (tuple[str, str]): Feature-name suffixes per layer.

    Returns:
        AnnData: The feature object, with the stacked matrix in `.X` and in
            `.layers[TWO_LAYER_INPUT]` and the derivation in `.uns['labelspace']`.

    Raises:
        ValueError: If not exactly two distinct layers are given.
        KeyError: If a layer is absent.
    """
    layers = list(layers)
    if len(layers) != 2 or layers[0] == layers[1]:
        raise ValueError('stack_layers needs exactly two distinct layers, got {!r}'.format(layers))
    missing = [k for k in layers if k not in adata.layers]
    if missing:
        raise KeyError('stack_layers: layer(s) {} not on the object; present: {}'.format(
            ', '.join(missing), ', '.join(sorted(adata.layers))))
    blocks = [sp.csr_matrix(adata.layers[k]) for k in layers]
    X = sp.hstack(blocks, format='csr')
    n = adata.n_vars
    names = [g + suffixes[0] for g in adata.var_names] + [g + suffixes[1] for g in adata.var_names]
    var = pd.DataFrame({'gene': list(adata.var_names) * 2,
                        'source_layer': [layers[0]] * n + [layers[1]] * n},
                       index=pd.Index(names, name=adata.var_names.name))
    out = anndata.AnnData(X=X, obs=adata.obs.copy(), var=var)
    out.layers[TWO_LAYER_INPUT] = X.copy()
    out.uns['labelspace'] = {'layers': layers, 'suffixes': list(suffixes),
                             'n_genes': int(n), 'input_layer': TWO_LAYER_INPUT}
    return out


def exclude_genes_from(adata, genes, feature_gene_column=None):
    """A copy of `adata` without the listed genes.

    On a gene-space object the match is on `var_names`; on a two-layer feature
    object pass `feature_gene_column='gene'` so BOTH features of a listed gene go.

    Args:
        adata (AnnData): The object to subset.
        genes (Iterable[str]): Gene ids to remove.
        feature_gene_column (str, optional): `.var` column holding the gene id
            per feature.

    Returns:
        tuple: (subset AnnData, number of features removed).

    Raises:
        ValueError: If nothing would remain.
    """
    genes = set(genes)
    if feature_gene_column:
        names = adata.var[feature_gene_column].astype(str)
    else:
        names = pd.Series(adata.var_names.astype(str), index=adata.var_names)
    keep = ~names.isin(genes).to_numpy()
    n_dropped = int((~keep).sum())
    if not keep.any():
        raise ValueError('exclude_genes removed every feature; the list is not a subset.')
    return adata[:, keep].copy(), n_dropped


def mark_trained_features(full, trained_on, two_layer=None, excluded=None) -> None:
    """Stamp which genes the model actually trained on into `full.var`, in place.

    Args:
        full (AnnData): The gene-space object being saved.
        trained_on (Iterable[str]): The training object's `var_names` (gene ids,
            or suffixed feature names on the two-layer path).
        two_layer (dict, optional): The `.uns['labelspace']` record from
            `stack_layers`; None on the single-layer path.
        excluded (Iterable[str], optional): The `exclude_genes` ids, if any.
    """
    trained = pd.Index(list(trained_on))
    if two_layer is None:
        full.var['hvg'] = full.var_names.isin(trained)
    else:
        s, u = two_layer['suffixes']
        hs = pd.Index([g + s for g in full.var_names]).isin(trained)
        hu = pd.Index([g + u for g in full.var_names]).isin(trained)
        full.var['hvg_spliced'] = hs
        full.var['hvg_unspliced'] = hu
        full.var['hvg'] = hs | hu
        full.uns['labelspace'] = {**two_layer, 'n_features_trained': int(len(trained)),
                                  'n_hvg_spliced': int(hs.sum()), 'n_hvg_unspliced': int(hu.sum())}
    if excluded:
        full.var['excluded'] = full.var_names.isin(list(excluded))


def _looks_like_counts(matrix) -> bool:
    """True when every stored value is a non-negative integer.

    The test for "has this layer already been normalized". Deliberately a
    property of the DATA rather than of the layer's name: a target may point at
    any layer, and `corrected_counts` is integral while `lognorm_cp10k` is not.

    Args:
        matrix: A dense array or scipy sparse matrix.

    Returns:
        bool: Whether the values are consistent with raw counts.
    """
    data = matrix.data if hasattr(matrix, 'data') else np.asarray(matrix).ravel()
    data = np.asarray(data)
    if data.size == 0:
        return True
    return bool(np.all(data >= 0) and np.all(data == np.floor(data)))


# Ray Tune samplers reachable from config. A whitelist rather than any
# getattr(tune, name): the search space is executed, so an arbitrary attribute
# name out of a YAML file is not something to hand to getattr.
_SAMPLERS = ('choice', 'uniform', 'quniform', 'loguniform', 'qloguniform',
             'randint', 'qrandint', 'lograndint', 'randn', 'qrandn')


def _to_sampler(key: str, spec):
    """One search-space value, as written in code or YAML, to a ray sampler.

    Args:
        key (str): Parameter name, used only in the error messages.
        spec (list | tuple | dict): A list of values, or a single-key mapping
            naming a sampler in `_SAMPLERS` with its arguments.

    Returns:
        ray.tune.sample.Domain: The sampler.

    Raises:
        ValueError: If `spec` is an empty list or not a recognised form.
    """
    if isinstance(spec, (list, tuple)):
        if len(spec) == 0:
            raise ValueError(
                "search_space['{}'] is an empty list; give it at least one value "
                "(a 1-element list pins the parameter), or null to drop it.".format(key))
        return tune.choice(list(spec))
    if isinstance(spec, dict):
        if len(spec) != 1:
            raise ValueError(
                "search_space['{}'] must name exactly one sampler, got {}.".format(
                    key, sorted(spec)))
        name, args = next(iter(spec.items()))
        if name not in _SAMPLERS:
            raise ValueError(
                "search_space['{}'] names unknown sampler '{}'. Known: {}.".format(
                    key, name, ', '.join(_SAMPLERS)))
        args = list(args) if isinstance(args, (list, tuple)) else [args]
        return getattr(tune, name)(*args)
    raise ValueError(
        "search_space['{}'] must be a list of values, a single-key sampler mapping "
        "like {{loguniform: [1e-4, 1e-2]}}, or null to drop the key -- got {!r}.".format(
            key, spec))


def resolve_search_space(model_class: str, overrides: dict = None, logger=None) -> dict:
    """The model-params search space for `model_class`.

    A target's own `search_space:` is merged over the built-in defaults per key.
    See docs/decisions/autotune-search-spaces.md for the value grammar and how the
    defaults were chosen.

    Args:
        model_class (str): Key into `_SEARCH_SPACE_DEFAULTS`.
        overrides (dict, optional): A target's `search_space:` block. `None` as a
            value drops that key entirely, leaving scvi its own default; a
            1-element list pins the parameter.
        logger (logging.Logger, optional): Defaults to 'IntegrateLogging'.

    Returns:
        dict: {parameter name: ray sampler}.

    Raises:
        ValueError: If `model_class` has no default space.
    """
    log = logger if logger is not None else logging.getLogger('IntegrateLogging')
    if model_class not in _SEARCH_SPACE_DEFAULTS:
        raise ValueError(
            "No default search space for model_class '{}'. Known: {}.".format(
                model_class, ', '.join(sorted(_SEARCH_SPACE_DEFAULTS))))

    merged = dict(_SEARCH_SPACE_DEFAULTS[model_class])
    changed, dropped = [], []
    for key, spec in (overrides or {}).items():
        if spec is None:
            if merged.pop(key, None) is not None:
                dropped.append(key)
        else:
            changed.append(key)
            merged[key] = spec

    space = {key: _to_sampler(key, spec) for key, spec in merged.items()}

    if changed or dropped:
        log.info(
            '[INTEGRATION] search_space for {}: {} default key(s), config overrode {}{}.'.format(
                model_class, len(space),
                ', '.join(sorted(changed)) or 'nothing',
                '; dropped ' + ', '.join(sorted(dropped)) if dropped else ''))
    else:
        log.info('[INTEGRATION] search_space for {}: built-in default, {} key(s).'.format(
            model_class, len(space)))
    return space


def normalize_kl_warmup(train_kwargs: dict, n_obs: int, logger=None) -> dict:
    """Guarantee KL warmup completes within training, and scale it when unset.

    Warmup left unset is derived as `_KL_WARMUP_MAX_FRACTION` of the effective
    max_epochs; one set longer than that cap is clamped with a warning; one already
    within it is left untouched. Step-based warmup (`n_steps_kl_warmup`) and DRVI's
    `"auto"` are passed through. The effective max_epochs is the configured
    `max_epochs` when present, else scvi's own heuristic for `n_obs`.

    Module-level rather than only a `SingleCellIntegrate` method because
    `refmapping._map_query_ref` trains through the same `model.train()` path and
    needs the same guarantee. See docs/decisions/kl-warmup.md.

    Args:
        train_kwargs (dict): kwargs for `model.train()`, optionally carrying a
            `plan_kwargs` block.
        n_obs (int): Cell count, for scvi's max-epochs heuristic.
        logger (logging.Logger, optional): Defaults to 'SingleCellMappingPipeline'.

    Returns:
        dict: A copy of `train_kwargs`, with `plan_kwargs['n_epochs_kl_warmup']`
            set or clamped where needed.

    Raises:
        ValueError: If `n_epochs_kl_warmup` is a string other than "auto".
    """
    log = logger if logger is not None else logging.getLogger('SingleCellMappingPipeline')
    kwargs = dict(train_kwargs)
    plan = dict(kwargs.get('plan_kwargs') or {})

    # Step-based warmup wins in scvi regardless of the epoch count.
    if plan.get('n_steps_kl_warmup') is not None:
        return kwargs

    # DRVI's own "ramp across the whole run" sentinel. Left to the model rather
    # than second-guessed, and returned before the int() coercion below.
    if isinstance(plan.get('n_epochs_kl_warmup'), str):
        requested_str = plan['n_epochs_kl_warmup']
        if requested_str != 'auto':
            raise ValueError(
                "[INTEGRATION] n_epochs_kl_warmup={!r} is not a number and the only string "
                "accepted is 'auto' (DRVI's own default, which ramps KL weight across the "
                "full max_epochs).".format(requested_str))
        log.info(
            '[INTEGRATION] n_epochs_kl_warmup="auto": leaving the KL schedule to the model. '
            'DRVI ramps it across the full max_epochs, which is its tuned default and '
            'deliberately not this pipeline\'s "finish by half of training" rule.')
        return kwargs

    configured_max = kwargs.get('max_epochs')
    effective_max = int(configured_max) if configured_max is not None else int(
        get_max_epochs_heuristic(n_obs))
    cap = max(1, int(round(_KL_WARMUP_MAX_FRACTION * effective_max)))

    requested = plan.get('n_epochs_kl_warmup')
    if requested is None:
        plan['n_epochs_kl_warmup'] = cap
        log.info(
            '[INTEGRATION] KL warmup unset; scaling to {} epochs ({:.0%} of the effective '
            'max_epochs {}{}) so the latent regularizes before training ends.'.format(
                cap, _KL_WARMUP_MAX_FRACTION, effective_max,
                '' if configured_max is not None else ', scvi heuristic'))
    elif int(requested) > cap:
        log.warning(
            '[INTEGRATION] n_epochs_kl_warmup={} exceeds {} epochs ({:.0%} of the effective '
            'max_epochs {}), so KL warmup would not complete and the latent would stay '
            'under-regularized (scvi-tools #507). Clamping warmup to {}.'.format(
                int(requested), cap, _KL_WARMUP_MAX_FRACTION, effective_max, cap))
        plan['n_epochs_kl_warmup'] = cap
    else:
        return kwargs                            # already fits; leave untouched

    kwargs['plan_kwargs'] = plan
    return kwargs


class SingleCellIntegrate:
    """Train (or load) one integration target's scVI-family model."""
    def __init__(self, adata, target, do_tune, do_hvg, output_dir, covariates, logger,
                 model_class: str = "SCVI", n_trials: int = None, tune_metric: str = None,
                 tune_mode: str = None, model_args: dict = None, train_args: dict = None,
                 tune_solver: str = 'randomized',
                 registry: CacheUtils = None, search_space: dict = None,
                 overwrite: bool = False, persist: bool = True, out_dir=None,
                 hvg_batch_key: str = None, exclude_genes=None):
        self.adata = adata
        self.target = target
        self.tune = do_tune
        self.hvg = do_hvg
        self.output_dir = output_dir
        self.covariates = covariates  # {'categorical': [...], 'continuous': [...]}

        self.logger = logger
        self.registry = registry or CacheUtils()
        self.overwrite = overwrite

        # None means "use this model class's default space as written", which is
        # what every non-tuning caller and every target without a `search_space:`
        # block gets. Merged in _build_search_space.
        self.search_space_cfg = search_space

        # `persist=False` makes this a disposable model: nothing is written to
        # disk and nothing touches scvi_tune_registry.json. Used by per-sample
        # doublet detection, which trains a throwaway model purely to read two
        # columns off it -- its parameters go into .uns['qc']['solo'] instead of
        # a registry that exists to locate persisted artifacts.
        self.persist = persist

        self.model_class = model_class
        self.model_cls = resolve_model_class(model_class)
        self.n_trials = n_trials or 200
        self.tune_metric = tune_metric or 'elbo_validation'
        # Derived from the metric rather than defaulted to a constant: getting
        # it wrong is silent. `elbo_validation` is a loss (lower is better)
        # while every scib metric is a score, so tuning on "Total" with
        # mode='min' would select the WORST trial and report no error at all.
        self.tune_mode = tune_mode or ('min' if self.tune_metric == 'elbo_validation' else 'max')
        self.model_args = model_args
        # SVD SOLVER FOR THE scib PATH, and 'randomized' rather than scvi's
        # 'arpack' default because arpack demonstrably fails on this data.
        #
        # A scib-scored tune builds a PCA per trial. scvi passes this straight
        # to scib_metrics.Benchmarker, whose own docstring says the choice is
        # there because it "can help stability issues" -- and on job 42083220,
        # tuning SCVI against `Total` on the 147k-cell nuclei reference, ALL 12
        # trials died with:
        #
        #     ArpackError: ARPACK error -9: Starting vector is zero
        #
        # arpack is an iterative Lanczos solver and fails outright when its
        # start vector lands in a null space, which a latent with a collapsed or
        # rank-deficient dimension will produce. 'randomized' is a direct
        # randomized SVD with no such failure mode. Only reachable when
        # tune_metric selects the scib path; the elbo path never builds a PCA.
        self.tune_solver = tune_solver
        self.train_args = dict(train_args or {})
        # Normalized once, up front, so the search space built below and the
        # final fit inherit one warmup that completes.
        self.train_args = self._normalize_kl_warmup(self.train_args, self.adata.n_obs)

        # Single source of truth for this target's output location: model.pt,
        # h5ad, HVG file and UMAP figures all land here. Injectable so a
        # disposable run does not write into the integration results tree;
        # orchestrator._dispatch_reduction reads the default back via the registry.
        self.out_dir = Path(out_dir) if out_dir is not None else (
            find_root_dir() / "results/integration" / model_output_folder(self.model_class) / self.target)

        # Only built when it will be used -- building it eagerly would log a
        # line per construction, which on the doublet path is once per sample
        # for a space no trial ever samples.
        self.search_space = self._build_search_space() if self.tune else None

        # What the model was ACTUALLY constructed with, filled in by
        # _integrate_scvi. Provenance must read this rather than the parameters
        # passed in: with tune=True the caller's `model_args` are never
        # consulted. Stays None on the registry-hit path, which loads a model
        # from disk without resolving a config here.
        self.resolved_model_params = None

        # Label-space inputs (docs/decisions/two-layer-label-spaces.md). The
        # HVG aggregation key defaults to per-`batch`, as select_hvg always did;
        # the exclusion list is applied after concatenation and before HVG.
        self.hvg_batch_key = hvg_batch_key or 'batch'
        self.exclude_genes = set(exclude_genes) if exclude_genes else None
        # Filled by _stack_training_layers on the two-layer path: the gene-space
        # object the latent is written back onto, and the derivation record.
        self._gene_space_adata = None
        self._two_layer = None

    def _build_search_space(self):
        """The ray search space for this target: model_params plus train_params.

        Trials are given the same training regime the final model will be fit in, so
        the tuner selects hyperparameters for the regime that model actually trains in.
        `check_val_every_n_epoch` is dropped because the tuner forces its own (=1) to
        report the validation metric, and `lr` is added because the documented scvi
        space tunes it; a config-pinned `lr` in `plan_kwargs` still wins.

        Returns:
            dict: {'model_params': {...}, 'train_params': {...}}.
        """
        base = resolve_search_space(self.model_class, self.search_space_cfg, self.logger)

        space = {"model_params": base}

        # scvi's autotune _trainable reads these from the "train_params" key and
        # merges them into model.train().
        train_params = {k: v for k, v in self.train_args.items() if k != 'check_val_every_n_epoch'}
        plan_kwargs = dict(train_params.get('plan_kwargs') or {})
        plan_kwargs.setdefault('lr', tune.loguniform(1e-4, 1e-2))
        train_params['plan_kwargs'] = plan_kwargs
        space["train_params"] = train_params
        return space

    def _normalize_kl_warmup(self, train_kwargs: dict, n_obs: int) -> dict:
        """Instance wrapper over the module-level `normalize_kl_warmup`.

        Args:
            train_kwargs (dict): kwargs for `model.train()`.
            n_obs (int): Cell count.

        Returns:
            dict: The normalized kwargs.
        """
        if self.model_class in _NO_KL_WARMUP_CLASSES:
            kwargs = dict(train_kwargs or {})
            plan = dict(kwargs.get('plan_kwargs') or {})
            dropped = {k: plan.pop(k) for k in ('n_epochs_kl_warmup', 'n_steps_kl_warmup')
                       if plan.get(k)}
            if dropped:
                # STRIPPED, not merely warned about. scvi would reset these to 0
                # anyway, but they would still be recorded in the registry's
                # train_params -- provenance claiming a warmup the fit never had.
                # In practice they arrive from the base config's `defaults:`
                # rather than from the target, so there is nothing for the target
                # author to remove.
                self.logger.warning(
                    '[INTEGRATION] target %r: %s.train() resets KL warmup to 0, so %s '
                    '(inherited unless set on the target) has been REMOVED from '
                    'plan_kwargs rather than passed on to be discarded and then '
                    'recorded as if it applied.',
                    self.target, self.model_class,
                    ', '.join('{}={}'.format(k, v) for k, v in sorted(dropped.items())))
            else:
                self.logger.info(
                    '[INTEGRATION] target %r: skipping KL-warmup normalization -- %s '
                    'does not accept warmup.', self.target, self.model_class)
            kwargs['plan_kwargs'] = plan
            return kwargs
        return normalize_kl_warmup(train_kwargs, n_obs, self.logger)

    def _stack_training_layers(self, layers: list) -> str:
        """Swap `self.adata` for its two-layer feature object; return the layer to train on.

        Args:
            layers (list[str]): The two layer names, in (spliced, unspliced) order.

        Returns:
            str: `TWO_LAYER_INPUT`, the layer `_prep_scvi` copies into `.X`.
        """
        self._gene_space_adata = self.adata
        self.adata = stack_layers(self.adata, layers)
        self._two_layer = dict(self.adata.uns['labelspace'])
        self.logger.info(
            '[INTEGRATION] target %r: training on two layers %s as separate features -- '
            '%d genes -> %d features (suffixes %s); the latent is written back onto the '
            'gene-space object.', self.target, layers, self._two_layer['n_genes'],
            self.adata.n_vars, '/'.join(FEATURE_SUFFIXES))
        return TWO_LAYER_INPUT

    def _ensure_gaussian_input(self, counts_layer: str) -> str:
        """The layer a Gaussian-likelihood model should be registered on.

        Returns `counts_layer` untouched when it already holds non-integral
        values -- i.e. a target that named `lognorm_cp10k` itself gets what it
        asked for. Otherwise CP10K-normalizes and log1p's into
        `_GAUSSIAN_INPUT_LAYER` and returns that, so a SysVI target can name the
        same `layer: corrected_counts` every other target names and still be fit
        on the scale its likelihood assumes.

        Called after HVG selection, so the derived layer carries the final var
        set rather than being subset afterwards.

        Args:
            counts_layer (str): The layer the target named, or None for `.X`.

        Returns:
            str: The layer name to hand `setup_anndata`, or None for `.X`.
        """
        source = self.adata.layers[counts_layer] if counts_layer is not None else self.adata.X
        if not _looks_like_counts(source):
            self.logger.info(
                '[INTEGRATION] target %r: %s needs log-normalized input and layer %r is '
                'already non-integral -- using it as-is, no transform applied.',
                self.target, self.model_class, counts_layer or 'X')
            return counts_layer

        work = anndata.AnnData(X=source.copy())
        sc.pp.normalize_total(work, target_sum=1e4)
        sc.pp.log1p(work)
        self.adata.layers[_GAUSSIAN_INPUT_LAYER] = work.X
        self.logger.info(
            '[INTEGRATION] target %r: %s has a Gaussian reconstruction likelihood, so '
            'layer %r (integral) was CP10K-normalized and log1p-transformed into %r, '
            'which is what the model is registered on.',
            self.target, self.model_class, counts_layer or 'X', _GAUSSIAN_INPUT_LAYER)
        return _GAUSSIAN_INPUT_LAYER

    def integrate(self, batch_key: str='batch', layer='corrected_counts',
                  labels_key: str=None):
        """Train (or load) this target's model and return the annotated AnnData.

        Args:
            batch_key (str, optional): `.obs` column holding the batch. An argument
                rather than a constant so a single sample -- no meaningful batch
                structure, and possibly no `.obs['batch']` at all -- can pass None.
            layer (str | list[str], optional): Counts layer to train on, or a list
                of two layers trained as separate features (2 x genes) -- see
                docs/decisions/two-layer-label-spaces.md.
            labels_key (str, optional): `.obs` column to register as labels. Needed by
                refinement (`SCANVI.from_scvi_model` inherits the source model's setup
                args) and by scib autotuning (the benchmark reads `label_key` off the
                model's registered labels). See docs/decisions/integration-required-keys.md.

        Returns:
            AnnData: `self.adata`, with the latent in `.obsm`.
        """
        if isinstance(layer, (list, tuple)) and len(layer) == 1:
            layer = layer[0]
        self.model = self._integrate_scvi(
            batch_key=batch_key, integration_name=self.target, layer=layer,
            do_hvg=self.hvg, tune=self.tune, model_args=self.model_args,
            save=self.persist, labels_key=labels_key)
        return self.adata

    def _prep_scvi(self, batch_key: str, counts_layer: str=None, categorical_covariates: list=None,
                   continuous_covariates: list=None, hvg:bool=False, labels_key: str=None):
        """Select HVGs if asked, then register this target's fields with the model class.

        Every declared key must be present in `.obs` -- a missing one raises rather
        than being dropped. See docs/decisions/integration-required-keys.md.

        Args:
            batch_key (str): `.obs` batch column, or None.
            counts_layer (str, optional): Layer copied into `.X` before setup.
            categorical_covariates (list, optional): `.obs` column names.
            continuous_covariates (list, optional): `.obs` column names.
            hvg (bool): Whether to subset to HVGs and register the file.
            labels_key (str, optional): `.obs` labels column.

        Raises:
            KeyError: If any declared key is absent from `.obs`.
        """
        categorical_covariates = [] if categorical_covariates is None else categorical_covariates
        continuous_covariates = [] if continuous_covariates is None else continuous_covariates

        # Two layers -> one stacked (cells x 2*genes) training object; the
        # gene-space object is kept aside for the latent write-back.
        if isinstance(counts_layer, (list, tuple)):
            counts_layer = self._stack_training_layers(list(counts_layer))
        if counts_layer is not None:
            self.adata.X = self.adata.layers[counts_layer].copy()

        # A key named in the config but absent from .obs is an ERROR, not a
        # thing to log and carry on from -- dropping one silently trains a model
        # without the correction the config asks for. batch_key is included for
        # the same reason. See docs/decisions/integration-required-keys.md.
        declared = ([batch_key] if batch_key is not None else []) + \
                   list(categorical_covariates) + list(continuous_covariates)
        missing = [k for k in declared if k not in self.adata.obs]
        if missing:
            raise KeyError(
                '[INTEGRATION] target {!r}: key(s) {} named in the config are not in .obs. '
                'Present columns: {}. These are required rather than optional -- dropping '
                'one silently trains a model without the correction the config asks for, and '
                'the saved setup_args would not record that it happened. Fix the name in '
                'integration_config.yaml, or remove it from the target. Run '
                '`python -m mapping.validate --icfg ... --scfg ...` to catch this '
                'before a GPU job starts.'.format(
                    self.target, ', '.join(repr(k) for k in sorted(set(missing))),
                    ', '.join(sorted(self.adata.obs.columns))))

        # Ensure everything is categorical
        for k in ([batch_key] if batch_key is not None else []) + list(categorical_covariates):
            self.adata.obs[k] = self.adata.obs[k].astype("category")

        # Kept so the SAVED h5ad can carry the full gene set while the model
        # trains on a subset -- see _update_save_anndata. Neither an exclusion
        # list nor HVG selection may decide which genes a downstream stage is
        # allowed to look at; both are modelling decisions. Snapshotted BEFORE
        # either narrows the feature space.
        self._full_gene_adata = self.adata if (hvg or self.exclude_genes) else None
        if self.exclude_genes:
            self.adata, n_dropped = exclude_genes_from(
                self.adata, self.exclude_genes,
                feature_gene_column='gene' if self._two_layer is not None else None)
            self.logger.info(
                '[INTEGRATION] target %r: exclude_genes removed %d feature(s) (%d ids listed) '
                'before HVG selection; %d remain. The saved h5ad keeps them, flagged in '
                '.var["excluded"].', self.target, n_dropped, len(self.exclude_genes),
                self.adata.n_vars)
        if hvg:
            hvg_path = self.out_dir / '{}_HVG.tsv'.format(self.target)
            # The aggregation key for the per-batch rankings -- per library by
            # default, exactly as before; the model's own batch_key is
            # registered further down and is a separate choice.
            self.adata = HVGSelector(self.adata, hvg_path).select_hvg(batch_key=self.hvg_batch_key)
            self._update_registry(self.target, {'hvg_path': str(hvg_path)})

        # Likewise not downgraded to None: both callers that pass one silently
        # produce something wrong rather than failing if the labels go missing.
        if labels_key is not None and labels_key not in self.adata.obs:
            raise KeyError(
                '[INTEGRATION] labels_key {!r} is not in .obs. It is required rather than '
                'optional: a labels-free model trains fine and then either forces scANVI to '
                'be given labels separately or makes every scib bio-conservation metric a '
                'one-class computation.'.format(labels_key))
        if labels_key is not None:
            self.adata.obs[labels_key] = self.adata.obs[labels_key].astype('category')

        setup_layer = counts_layer
        if self.model_class in _GAUSSIAN_INPUT_CLASSES:
            setup_layer = self._ensure_gaussian_input(counts_layer)

        # SysVI.setup_anndata takes no labels_key and would swallow one through
        # **kwargs -- registering nothing and reporting nothing. Refuse instead:
        # a target asking for labels wants either scib bio-conservation metrics
        # or a scANVI-ready model, and would silently get neither.
        if self.model_class in _GAUSSIAN_INPUT_CLASSES and labels_key is not None:
            raise ValueError(
                '[INTEGRATION] target {!r}: model_class {!r} has no labels_key -- its '
                'setup_anndata does not accept one and would discard it through **kwargs. '
                'It is an unsupervised model; drop labels_key from the target, or use '
                'SCVI/JointEmbeddingSCVI if the labels are needed downstream.'.format(
                    self.target, self.model_class))

        # SysVI's batch_key is its SYSTEM covariate and is a required positional
        # argument, not an optional one -- there is no single-sample SysVI fit.
        # Caught here rather than as a TypeError out of scvi.
        if self.model_class in _GAUSSIAN_INPUT_CLASSES and batch_key is None:
            raise ValueError(
                '[INTEGRATION] target {!r}: model_class {!r} requires a batch_key. It is '
                'the SYSTEM the model corrects across -- the substantial, structured '
                'difference (here: nuclei vs. protoplast), with within-system batches '
                'going to `covariates:` as categorical instead. A target with no batch '
                'structure has nothing for this model to do; use SCVI.'.format(
                    self.target, self.model_class))

        # Passed through unfiltered: the raise above has already guaranteed
        # every declared key is present, so what the config declares is what
        # gets registered, or the run stopped.
        setup_kwargs = dict(
            batch_key=batch_key,
            categorical_covariate_keys=list(categorical_covariates),
            continuous_covariate_keys=list(continuous_covariates),
            layer=None if setup_layer is None else setup_layer)
        if self.model_class not in _GAUSSIAN_INPUT_CLASSES:
            setup_kwargs['labels_key'] = labels_key
        self.model_cls.setup_anndata(self.adata, **setup_kwargs)

    def _integrate_scvi(self, batch_key:str, integration_name:str, layer:str=None, model_args:dict=None, do_hvg:bool=False, tune:bool=True, save:bool=True, labels_key:str=None):
        """Resolve hyperparameters, train, and optionally save and register the model.

        Returns a registry-cached model without training when one exists and
        `overwrite` is False. A disposable run (`persist=False`) never consults or
        writes the registry.

        Args:
            batch_key (str): `.obs` batch column, or None.
            integration_name (str): Registry key to look this target up under.
            layer (str, optional): Counts layer.
            model_args (dict, optional): Fixed hyperparameters, used only when
                `tune` is False.
            do_hvg (bool): Whether to subset to HVGs.
            tune (bool): Whether to run autotune instead of using `model_args`.
            save (bool): Whether to write the model, h5ad and registry entry.
            labels_key (str, optional): `.obs` labels column.

        Returns:
            scvi.model.base.BaseModelClass: The trained or loaded model.
        """
        # A disposable run must not resolve to some previously-registered model
        # that happens to share this name.
        entry = None if not self.persist else self.registry.find_entry(integration_name)
        if not self.overwrite and entry and "model_path" in entry:
            model_path = Path(entry["model_path"])
            adata_path = Path(entry["anndata_path"])
            if model_path.exists():
                self.logger.info(f"[INTEGRATION] Found existing {self.model_class} model at {model_path}, loading...")
                self.adata = sc.read_h5ad(adata_path)
                model = self.model_cls.load(model_path, adata=self.adata)
                self.logger.info(f"[WARNING][INTEGRATION] Found existing anndata object for model {self.target}, overwriting provided anndata...")
                return model

        self._prep_scvi(batch_key = batch_key, counts_layer = layer, hvg = do_hvg,
                         categorical_covariates = self.covariates['categorical'],
                         continuous_covariates = self.covariates['continuous'],
                         labels_key = labels_key)
        if tune:
            # Raises if the tuner selects nothing, rather than falling through to
            # the fixed hyperparameters below. See _tune_scvi.
            cfg = self._tune_scvi(search_space = self.search_space, n_trials = self.n_trials,
                                  metric = self.tune_metric, mode = self.tune_mode)
        else:
            cfg = model_args if model_args is not None else {"model_params": {}}
        # Published before training so provenance records the tuned values
        # rather than whatever the caller proposed. A tune=False run with no
        # model_args resolves to {}, i.e. scvi's own constructor defaults.
        self.resolved_model_params = dict(cfg["model_params"])
        self.logger.info('Creating {} model with parameters: {}'.format(self.model_class, cfg["model_params"]))
        model = self.model_cls(self.adata, **cfg["model_params"])
        # The tuned config carries train_params too; without this the tuned lr
        # is silently dropped and the model trains at a different learning rate
        # than the one that won the search.
        final_train = self._final_train_args(cfg.get("train_params"))
        self.logger.info('Training {} with train args: {}'.format(self.model_class, final_train))
        model.train(**final_train)

        if save:
            model.save(self.out_dir, overwrite=True)
            self._update_save_anndata(model, save_dir = self.out_dir)
            # 'samples' is recorded here and not only via _save_opt_config (which
            # runs only when tuning executes) so a tune=False run is still
            # resolvable as a scArches --reference. tool_versions/timestamp are
            # stamped at training completion, not on every incidental metadata
            # patch to this entry -- e.g. the hvg_path-only write in _prep_scvi.
            self._update_registry(self.target, {'model_path': str(self.out_dir), 'anndata_path' : str(self.out_dir / f"{self.target}.h5ad"),
                                                 'model_class': self.model_class,
                                                 'samples': self.adata.obs["sample"].unique().tolist(),
                                                 'tool_versions': integration_tool_versions(),
                                                 'timestamp': datetime.datetime.now().isoformat()})
        return model

    def _final_train_args(self, tuned_train_params: dict) -> dict:
        """Merge the tuner's chosen train_params with the config's own train_args.

        The config wins for anything it sets (max_epochs, warmup,
        check_val_every_n_epoch), while values the tuner picked and the config did not
        -- `lr` above all -- survive. `plan_kwargs` is deep-merged so a tuned `lr` and
        a configured `n_epochs_kl_warmup` coexist rather than one clobbering the other.
        On a tune=False run this collapses to exactly `self.train_args`.

        Args:
            tuned_train_params (dict): `train_params` from the winning trial's config,
                or None.

        Returns:
            dict: kwargs for `model.train()`.
        """
        merged = dict(tuned_train_params or {})
        for key, value in self.train_args.items():
            if key == 'plan_kwargs' and isinstance(value, dict) and isinstance(merged.get('plan_kwargs'), dict):
                plan = dict(merged['plan_kwargs'])
                plan.update(value)
                merged['plan_kwargs'] = plan
            else:
                merged[key] = value
        return merged

    def _tune_scvi(self, search_space: dict, n_trials: int=200, metric:str='elbo_validation', mode:str=None, seed:int=42):
        """Run scvi's AutotuneExperiment and return the winning config.

        Returns a previously-tuned config from the registry without searching when one
        exists and `overwrite` is False. Raises rather than falling back to fixed
        hyperparameters when no trial reports the metric; see
        docs/decisions/autotune-failure-diagnosis.md and autotune-ray-runtime.md.

        Args:
            search_space (dict): As built by `_build_search_space`.
            n_trials (int): ray `num_samples`.
            metric (str): Metric to select on.
            mode (str, optional): 'min' or 'max'. Defaults to `self.tune_mode`.
            seed (int): Searcher seed.

        Returns:
            dict: The winning trial's config, with 'model_params' and 'train_params'.

        Raises:
            RuntimeError: If no trial reported `metric`.
        """
        mode = mode or self.tune_mode
        opt_config = None if self.overwrite else self._load_tuned_model(model_name = self.target)
        if opt_config:
            self.logger.info('Found config: {}'.format(opt_config))
            return opt_config
        else:
            self.logger.info('[INTEGRATION] Starting autotune for model {}'.format(self.target))
            cluster = self._cluster_resources()
            resources = self._get_allocated_resources()
            if not torch.cuda.is_available():
                self.logger.info('[INTEGRATION] No GPU detected, tuning on CPU')

            # is_initialized guards the per-sample case: doublet tuning
            # constructs one of these per sample in a single process, and a
            # second ray.init would raise.
            if not ray.is_initialized():
                self.logger.info('[INTEGRATION] Initialising ray with {} cpu / {} gpu'.format(
                    cluster['cpu'], cluster['gpu']))
                ray.init(num_cpus=cluster['cpu'], num_gpus=cluster['gpu'],
                         include_dashboard=False, ignore_reinit_error=True)
            self.logger.info('[INTEGRATION] Using {} cpus and {} gpus per trial'.format(
                resources['cpu'], resources['gpu']))

            # Ray's trial logs churn too much for Lustre, so stage them through
            # node-local scratch, scoped by target. See
            # docs/decisions/autotune-ray-runtime.md.
            scratch_root = os.environ.get('TMPDIR') or tempfile.gettempdir()
            tmp_tune_dir = Path(scratch_root) / 'scvi_tune' / self.target
            tmp_tune_dir.mkdir(parents=True, exist_ok=True)
            # Scratch is wiped when the job ends, so a failed run's logs are
            # copied here to survive. Only written on the failure path below.
            fail_record_dir = self.out_dir / '_tmp_tune'
            # ASHA's per-trial epoch budget, aligned with the actual max_epochs
            # so the top successive-halving rung is reachable.
            max_t = int(self.train_args.get('max_epochs') or 400)
            experiment = AutotuneExperiment(model_cls = self.model_cls, data = self.adata, metrics=metric, mode=mode, search_space=search_space, num_samples=n_trials, searcher="hyperopt", seed=seed, resources=resources, scheduler="asha", scheduler_kwargs={"max_t": max_t, "grace_period": 20, "reduction_factor": 3}, logging_dir=str(tmp_tune_dir), solver=self.tune_solver)
            tuner = experiment.get_tuner()
            results = tuner.fit()
            try:
                opt_mdl = results.get_best_result(metric=metric, mode=mode)
            except RuntimeError as e:
                # Ray's message says only that no trial reported the metric,
                # which is the symptom; the cause is in the individual trials.
                self._log_tuning_failure(results, metric)
                ray.shutdown()
                # The trial logs are the record of what actually happened, and
                # scratch is wiped when the job ends. Best-effort: a copy failure
                # must not mask the real tuning failure below.
                record_location = tmp_tune_dir
                try:
                    if fail_record_dir.exists():
                        shutil.rmtree(fail_record_dir, ignore_errors=True)
                    shutil.copytree(tmp_tune_dir, fail_record_dir)
                    record_location = fail_record_dir
                except Exception as copy_error:
                    self.logger.error(
                        '[INTEGRATION] Could not copy tuning trial logs from scratch ({} -> {}): '
                        '{}. They remain on node-local scratch until the job ends.'.format(
                            tmp_tune_dir, fail_record_dir, copy_error))
                raise RuntimeError(
                    '[INTEGRATION] Autotune for "{}" finished {} trial(s) but none reported '
                    '"{}", so no model could be selected. This is not fixed by raising n_trials '
                    '-- if no trial reported the metric, more of them will not either. See the '
                    'per-trial diagnosis logged above, and ray\'s own trial logs left in {}. '
                    'Original error: {}'.format(
                        self.target, len(results), metric, record_location, e)) from e
            opt_config = opt_mdl.config
            self._save_opt_config(cfg = opt_config)
            ray.shutdown()
            # Only reached once the config is safely in the registry. A clean
            # success leaves nothing behind on scratch; the failure path above
            # is the one that preserves the logs.
            shutil.rmtree(tmp_tune_dir, ignore_errors=True)
            return opt_config

    def _log_tuning_failure(self, results, metric: str) -> None:
        """Log why the trials produced nothing, since ray only reports that they did.

        Separates the three faults ray's single message covers -- trials that errored,
        trials that never reported the metric, and trials that reported it as NaN --
        and logs one representative of each with a pointer at the likely cause.
        Best-effort: it runs on the failure path and must not raise. See
        docs/decisions/autotune-failure-diagnosis.md.

        Args:
            results (ray.tune.ResultGrid): The finished trials.
            metric (str): The metric that was being selected on.
        """
        try:
            errored, no_metric, nan_metric = [], [], []
            for i, r in enumerate(results):
                if getattr(r, 'error', None) is not None:
                    errored.append((i, r.error))
                    continue
                metrics = getattr(r, 'metrics', None) or {}
                if metric not in metrics:
                    no_metric.append((i, sorted(metrics)))
                elif not np.isfinite(float(metrics[metric])):
                    nan_metric.append((i, metrics[metric]))

            self.logger.error(
                '[INTEGRATION] Autotune for "{}" selected no model. Of {} trial(s): {} errored, '
                '{} ran without ever reporting "{}", {} reported it as NaN/inf.'.format(
                    self.target, len(results), len(errored), len(no_metric), metric,
                    len(nan_metric)))

            # One representative of each class: 20 tracebacks would bury the
            # summary line above.
            if errored:
                self.logger.error('[INTEGRATION]   trial {} errored: {}'.format(
                    errored[0][0], str(errored[0][1]).strip().splitlines()[-1]
                    if str(errored[0][1]).strip() else '(empty error)'))
                self.logger.error(
                    '[INTEGRATION]   -> every trial crashing points at the environment or the '
                    'data, not the search space: check GPU memory against the per-trial '
                    'resources, and that the input matrix is counts.')
            if no_metric:
                idx, keys = no_metric[0]
                self.logger.error(
                    '[INTEGRATION]   trial {} finished but reported only: {}'.format(
                        idx, ', '.join(keys) or '(nothing)'))
                self.logger.error(
                    '[INTEGRATION]   -> "{}" is a VALIDATION metric; it is only logged when a '
                    'validation split exists. scvi-tools splits at train_size=0.9 '
                    '(n_val = n - ceil(0.9n)), so n <= 9 gives an empty one. If n is comfortably '
                    'above that, the metric name or the training config is the suspect, not the '
                    'cell count.'.format(metric))
            if nan_metric:
                self.logger.error(
                    '[INTEGRATION]   trial {} reported {}={}'.format(
                        nan_metric[0][0], metric, nan_metric[0][1]))
                self.logger.error(
                    '[INTEGRATION]   -> training diverged rather than failed. Suspect the search '
                    'space (an over-large learning rate is the usual cause) or degenerate input.')
        except Exception as diag_error:
            # The real failure is the caller's; this one must not replace it.
            self.logger.error(
                '[INTEGRATION] Could not inspect the failed tuning results for "{}" ({}). The '
                'trial logs are still on disk.'.format(self.target, diag_error))

    def _cluster_resources(self):
        """What this job actually has, for `ray.init`.

        GPUs come from `torch.cuda.device_count()`, which already respects
        CUDA_VISIBLE_DEVICES, so it cannot disagree with what ray sees. SLURM's own
        variables are consulted only for CPUs, where no equivalent runtime probe
        exists -- `os.cpu_count()` reports the whole node. See
        docs/decisions/autotune-ray-runtime.md.

        Returns:
            dict: {'cpu': int, 'gpu': int} for the whole allocation.
        """
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 1)
        return {"cpu": int(cpus), "gpu": int(torch.cuda.device_count())}

    def _get_allocated_resources(self):
        """Per-trial slice of what `_cluster_resources` reports.

        Returns:
            dict: {'cpu': int, 'gpu': int} per trial.
        """
        cluster = self._cluster_resources()
        cpus, gpus = cluster["cpu"], cluster["gpu"]
        trial_cpus = max(1, cpus // gpus) if gpus > 0 else max(1, cpus // 2)
        trial_gpus = 1 if gpus > 0 else 0
        return {"cpu": trial_cpus, "gpu": trial_gpus}

    def _save_opt_config(self, cfg: dict):
        """Record the tuned parameters in the registry, unless this model is disposable.

        This is the registry write that would otherwise fire even with
        save=False, since tuning records its optimum independently of the model
        save. See docs/decisions/autotune-failure-diagnosis.md.

        Args:
            cfg (dict): The winning trial's config.
        """
        if not self.persist:
            self.logger.info('[INTEGRATION] persist=False: not recording tuned parameters in the registry')
            return
        # Upsert, not add_entry: a blind append here was the root cause of
        # duplicate registry entries.
        self._update_registry(self.target, {"samples": self.adata.obs["sample"].unique().tolist(), "optimal_model": cfg})

    def _load_tuned_model(self, model_name: str):
        """The tuned config recorded for `model_name`, or False if there is none.

        Args:
            model_name (str): Registry key.

        Returns:
            dict or bool: The stored `optimal_model` config, else False.
        """
        entry = self.registry.find_entry(model_name)
        if entry:
            self.logger.info(f'Found entry for model: {model_name}, Attempting to load tuned parameters...')
            try:
                return entry["optimal_model"]
            except KeyError:
                self.logger.info(f'Failed to load tuned parameters for model: {model_name}, tuning....')
                return False
        return False

    def _update_registry(self, integration_name: str, updates: dict):
        """Upsert this target's registry entry.

        Args:
            integration_name (str): Registry key.
            updates (dict): Fields to merge into the entry.
        """
        self.registry.upsert_entry(integration_name, updates)

    def _update_save_anndata(self, model, save_dir: str, method = 'scVI'):
        """Write the latent into `.obsm` and save the AnnData beside the model.

        Args:
            model: The trained model.
            save_dir (Path): Directory to write `{target}.h5ad` into.
            method (str): Names the obsm key, `X_{method}`.
        """
        latent = model.get_latent_representation()
        save_dir.mkdir(parents=True, exist_ok=True)

        # Obsm key stays 'X_scVI' whichever concrete class trained the model:
        # model_class selects the training/loading class, never the obsm
        # namespace that _dispatch_reduction and DimReducer key off.
        key = f"X_{method}"
        save_path = save_dir / f"{self.target}.h5ad"

        self.adata.obsm[key] = latent
        # ...but a DRVI latent also gets a self-describing alias: its individual
        # DIMENSIONS are the deliverable, and a consumer reading 'X_scVI' has no
        # way to know it holds a disentangled latent. An alias rather than a
        # rename so existing readers keep working; the duplication is a few MB.
        if self.model_class == 'DRVI':
            self.adata.obsm['X_DRVI'] = latent

        # THE SAVED OBJECT CARRIES EVERY GENE, even when the model trained on
        # HVGs. Until 2026-08-18 do_hvg replaced self.adata with the 2,000-gene
        # subset and that subset was what got written, so markerid -- which
        # reads this file -- could only ever nominate a marker from those 2,000.
        # Measured on MtSZ_sysvi_labels: 3.8% of Medicago's gene space reachable,
        # and any canonical marker outside the HVG set unreachable regardless of
        # how diagnostic it was.
        #
        # The two questions are separate and now answered separately: HVGs are
        # what the LATENT is learned from, the full set is what genes are SCORED
        # on. markerid applies its own detection floor (min_cells=10,
        # markerid/markers.py) so nothing undetected leaks through.
        #
        # `hvg` is recorded in .var so a consumer can recover the training
        # subset, and the latent is per-cell so it transfers unchanged -- the
        # cells and their order are identical by construction.
        out = self.adata
        # The two-layer path writes back onto the GENE-SPACE object (the stacked
        # feature object is training input only); otherwise onto the pre-HVG /
        # pre-exclusion snapshot, exactly as before.
        full = self._gene_space_adata if self._gene_space_adata is not None else \
            getattr(self, '_full_gene_adata', None)
        if full is not None and full.n_obs == self.adata.n_obs:
            trained_on = self.adata.var_names
            full = full[:, :].copy()
            for k, v in self.adata.obsm.items():
                full.obsm[k] = v
            for k in self.adata.obs.columns:
                if k not in full.obs.columns:
                    full.obs[k] = self.adata.obs[k].values
            mark_trained_features(full, trained_on, two_layer=self._two_layer,
                                  excluded=self.exclude_genes)
            out = full
            self.adata = full
            self.logger.info(
                '[INTEGRATION] target %r: model trained on %d feature(s); saving all %d genes '
                'with .var["hvg"] marking the training subset, so marker calling is not '
                'confined to it.', self.target, len(trained_on), full.n_vars)
        out.write_h5ad(save_path)
        self.logger.info('Saved anndata at path: {}'.format(save_path))
