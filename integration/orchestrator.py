#!/usr/bin/env python3

import logging
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp

from pathlib import Path
from shared.utils import load_config
from shared.samples import load_samples
from shared.targets import (resolve_run_spec, resolve_batch_assignments,
                            load_integration_config, BATCH_KEY_SEP)
from .integrationutils import CacheUtils, resolve_model_class
from .integrate import SingleCellIntegrate
from .refmapping import scArchesIntegrator
from .contrastive import ContrastiveIntegrator
from shared.dimred import DimReducer, UMAPPlotter
from .diagnostics import run_diagnostics
from .sample_discovery import (load_sample_manifest, resolve_sample_artifact, resolve_batch,
                              check_artifact_availability, DEFAULT_ARTIFACT_KEY)
from .report_embedding import embed_into_mapping_report


# Suffixes marking an .obsm key as a *per-sample* embedding, computed by the
# mapping stage on that one sample alone (schema 1.2 onward).
_PER_SAMPLE_EMBEDDING_SUFFIXES = ('_qc', '_scVI')


def _drop_per_sample_embeddings(adata, sample: str, logger) -> None:
    """Remove per-sample embeddings from `adata.obsm`, in place.

    `ad.concat(join="outer")` keeps any key present on every input, and each of
    these is a coordinate space fit to one sample alone -- concatenated, it would
    look like a valid cohort embedding and mean nothing. The `leiden_*` labels
    beside them are left alone. See docs/decisions/concatenation-hygiene.md.

    Args:
        adata (AnnData): One sample's object, modified in place.
        sample (str): Sample name, for the log line.
        logger (logging.Logger): Where the debug lines go.
    """
    for key in [k for k in list(adata.obsm)
                if k.endswith(_PER_SAMPLE_EMBEDDING_SUFFIXES)]:
        del adata.obsm[key]
        logger.debug(
            f"[Integrate] Dropped per-sample embedding '{key}' from '{sample}' before concatenation.")


def _drop_unused_layers(adata, sample: str, keep, logger) -> None:
    """Drop count layers the target will not use, in place, before concatenation.

    A mapping artifact carries eight layers plus `X`, and `join="outer"` keeps
    every one -- ~90% of a 69 GB target that is never read, and the usual cause of
    an OOM here. See docs/decisions/concatenation-hygiene.md.

    Args:
        adata (AnnData): One sample's object, modified in place.
        sample (str): Sample name, for the log lines.
        keep (list[str] or None): Layers to keep. None keeps everything, which is
            the default and what every target predating `keep_layers` relies on.
        logger (logging.Logger): Where the warning and debug lines go.
    """
    if keep is None:
        return
    keep = set(keep)
    missing = keep - set(adata.layers)
    if missing:
        logger.warning(
            f"[Integrate] keep_layers names layer(s) absent from '{sample}': "
            f"{', '.join(sorted(missing))}. Present: {', '.join(sorted(adata.layers))}.")
    for key in [k for k in list(adata.layers) if k not in keep]:
        del adata.layers[key]
        logger.debug(
            f"[Integrate] Dropped unused layer '{key}' from '{sample}' before concatenation.")


class IntegrationManager:
    """Resolve one integration target and run it end to end."""
    def __init__(self, integration_config_path: Path, sample_metadata_path: Path, target: str, dimreduction: str, plot: bool,
                 overwrite: bool = False, force: bool = False,
                 artifact_fallback: bool = False):
        self.target = target
        # Overridden per target in run(); None = keep every layer.
        self.keep_layers = None
        # Overridden per target in run(); None = filter no genes.
        self.min_cells = None
        self.config = load_integration_config(Path(integration_config_path))
        self.sample_metadata_path = sample_metadata_path
        # Same resolver the mapping stage uses, so `batch` (and anything else
        # read off an entry here) reflects the sheet's `defaults:` block rather
        # than only what was written out per sample.
        self.metadata = load_samples(Path(sample_metadata_path))
        self.dimreduction = dimreduction
        self.plot = plot
        self.overwrite = overwrite
        self.force = force
        # Which mapping artifact to integrate: the automatic 'filtered_h5ad' or
        # a hand-reviewed 'filtered_manual_h5ad'. File-level default; run()
        # narrows it to the target's own `artifact_key:` if it states one. No
        # CLI override on purpose -- see docs/decisions/concatenation-hygiene.md.
        self.artifact_key = self.config.get('artifact_key') or DEFAULT_ARTIFACT_KEY
        self.artifact_fallback = artifact_fallback
        self.cache = CacheUtils()

        # No hardcoded fallback on purpose: both are site-specific data
        # locations, not something a fresh clone should silently inherit from
        # whoever wrote the previous default.
        missing = [k for k in ('input_dir', 'output_dir') if not self.config.get(k)]
        if missing:
            raise ValueError(
                'integration_config.yaml is missing required key(s): {}. Both '
                'are absolute paths on this machine and have no default -- '
                'set them explicitly.'.format(', '.join(missing)))
        self.input_dir = Path(self.config['input_dir'])
        self.output_dir = Path(self.config['output_dir'])

        # Fetched, not configured: setup_logger runs once in main.py, and an
        # IntegrationManager is constructed several times per invocation, so
        # configuring here would re-clear and re-attach handlers each time.
        self.logger = logging.getLogger('IntegrateLogging')

    def _get_type(self, target: str):
        """This target's method and its fully-resolved run arguments.

        Everything except the model-class lookup comes from
        `shared/targets.py::resolve_run_spec` -- the one place that answers what a
        target resolves to, shared with the validator so the two cannot disagree about a
        default. `resolve_model_class` stays here because `integrationutils` imports
        scvi at module scope and `resolve_run_spec` must stay importable outside the
        integration env.

        Args:
            target (str): Target name.

        Returns:
            tuple: (method (str), args (dict)).
        """
        args = resolve_run_spec(self.config, target)
        if args["method"] == "scVI":
            resolve_model_class(args["model_class"])  # fail fast on a typo, before any GPU work
        return args["method"], args

    def _concatenate_adata(self, samples: list | None = None, reference: dict | None = None, query: list | None = None):
        """Load this target's samples and concatenate them into one AnnData.

        Per-sample embeddings and unused layers are dropped, sample-sheet design
        columns are stamped onto `.obs`, barcodes are prefixed with the sample name,
        and dtypes are aligned -- see docs/decisions/concatenation-hygiene.md and
        obs-sanitization.md. Pass either `samples`, or both `reference` and `query`.

        Args:
            samples (list[str], optional): Samples to concatenate.
            reference (dict, optional): {sample: {'adata_path': ...}} for the scArches
                reference side.
            query (list[str], optional): scArches query samples.

        Returns:
            AnnData: The concatenated object, gene-filtered if `min_cells` is set.

        Raises:
            ValueError: If neither form of the sample arguments was given.
        """
        adatas = []
        if reference is not None and query is not None:
            all_samples = list(reference.keys()) + query
        elif samples is not None:
            ref_samples, all_samples = [], samples
        else:
            raise ValueError("Either 'samples' or both 'reference' and 'query' must be provided.")

        # Pre-flighted before loading anything, so a 40-sample run reports all
        # the missing artifacts at once. Reference samples loaded from an
        # explicit adata_path are exempt: they never go through the manifest.
        via_manifest = [s for s in all_samples if not (
            reference and s in reference and Path((reference[s] or {}).get('adata_path', '')).is_file())]
        used = check_artifact_availability(via_manifest, self.input_dir, self.artifact_key,
                                           fallback=self.artifact_fallback)
        mixed = sorted({k for k in used.values()})
        if len(mixed) > 1:
            self.logger.warning(
                "[Integrate] This run mixes artifact kinds across samples: {}. That is what "
                "--artifact-fallback asks for, but the resulting model is trained on samples "
                "filtered to different standards.".format(', '.join(mixed)))
        self.logger.info("[Integrate] Loading samples via manifest key(s): {}".format(
            ', '.join(mixed) or self.artifact_key))

        def _load_adata(s):
            """One sample's AnnData and its manifest, preferring an explicit
            reference adata_path over the manifest-resolved artifact."""
            if reference and s in reference:
                ref_info = reference[s]
                adata_path = Path(ref_info.get("adata_path", ""))
                if adata_path.is_file():
                    self.logger.info(f"[scArches] Loading reference AnnData for '{s}' from {adata_path}")
                    return sc.read_h5ad(adata_path), None
                self.logger.warning(f"[scArches] No adata_path for '{s}', falling back to QC output")
            manifest = load_sample_manifest(s, self.input_dir)
            path = resolve_sample_artifact(s, self.input_dir, self.metadata, manifest=manifest,
                                           artifact_key=self.artifact_key,
                                           fallback=self.artifact_fallback)
            return sc.read_h5ad(path), manifest

        for s in all_samples:
            adata, manifest = _load_adata(s)
            _drop_per_sample_embeddings(adata, s, self.logger)
            _drop_unused_layers(adata, s, self.keep_layers, self.logger)
            if "sample" not in adata.obs:
                adata.obs["sample"] = s
            if "batch" not in adata.obs:
                if manifest is None:
                    manifest = load_sample_manifest(s, self.input_dir)
                adata.obs["batch"] = resolve_batch(s, manifest, self.metadata)
            # Sample-sheet facts that describe the cell but never reached .obs.
            # Stamped for every target, not just the ones that use them: a
            # confound belongs in the data as a column, and `batch_key: 'tech'`
            # cannot be written at all until the column is here. What each means
            # is documented in configs/sample_metadata.yaml; which are safe to
            # correct on, in docs/decisions/batch-keys-and-confounds.md. Never
            # overwrites a column the mapping artifact already carries.
            entry = self.metadata.get(s) or {}
            for column in ('tech', 'sample_type', 'prep', 'treatment', 'time_h'):
                if column not in adata.obs and entry.get(column) is not None:
                    adata.obs[column] = entry[column]
            # Unconditional, and separate from the "sample" backfill above: the
            # 10x barcode whitelist is shared and finite, so two independently
            # run samples can produce the same barcode by coincidence. See
            # docs/decisions/concatenation-hygiene.md.
            adata.obs_names = [f"{s}.{bc}" for bc in adata.obs_names]
            adatas.append(adata)

        adatas = self._sanitize_obs(adatas)
        combined = ad.concat(adatas, join="outer", label="concat_sample", keys=samples)
        combined.obs["sample"] = combined.obs["sample"].astype("category")
        combined.obs["batch"] = combined.obs["batch"].astype("category")
        self._filter_genes(combined)
        return combined

    def _filter_genes(self, adata):
        """Drop genes detected in fewer than `self.min_cells` cells, in place.

        Exists for the `do_hvg: False` case, which otherwise has no gene floor at all.
        Counts on the layer the model trains on, and on positive values rather than
        stored ones. See docs/decisions/concatenation-hygiene.md.

        Args:
            adata (AnnData): The concatenated object, subset in place.

        Raises:
            ValueError: If the floor removes every gene.
        """
        if not self.min_cells:
            return

        # The layer the model trains on, not .X: filtering on one matrix and
        # training on another selects a gene set that does not describe the data
        # the model sees.
        layer = (self.keep_layers or [None])[0]
        matrix = adata.layers[layer] if layer and layer in adata.layers else adata.X

        # `(X > 0)`, never getnnz: corrected_counts is CellBender output and can
        # carry explicitly stored zeros, so a gene stored in 10,000 cells but
        # positive in 3 would otherwise pass.
        if sp.issparse(matrix):
            n_cells = np.asarray((matrix > 0).sum(axis=0)).ravel()
        else:
            n_cells = (matrix > 0).sum(axis=0)

        keep = n_cells >= self.min_cells
        n_dropped = int((~keep).sum())
        self.logger.info(
            "[Integrate] Gene floor min_cells={} on layer '{}': keeping {} of {} genes "
            "({} dropped)".format(self.min_cells, layer or 'X', int(keep.sum()),
                                  adata.n_vars, n_dropped))
        if not keep.any():
            raise ValueError(
                "min_cells={} removed every gene from target '{}'. Check that the layer "
                "counted ('{}') is the one carrying counts.".format(
                    self.min_cells, self.target, layer or 'X'))
        adata._inplace_subset_var(keep)

    def _sanitize_obs(self, adatas:list, fill_value='Unknown'):
        """Align `.obs` columns across samples so `ad.concat` can join them.

        A column keeps its real dtype only if it is present on every sample and all of
        them agree on a non-categorical, non-object dtype; everything else is filled
        and stringified. Both halves of that rule exist for a specific serialization
        failure -- see docs/decisions/obs-sanitization.md before relaxing either.

        Args:
            adatas (list[AnnData]): One per sample.
            fill_value (str): Value written into a column a sample lacks.

        Returns:
            list[AnnData]: The same objects, with `.obs` replaced.
        """
        def _dtype_kind(series):
            """The dtype to compare across samples, or None if never safe to leave alone."""
            if pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
                return None  # never "safe to leave alone"
            return series.dtype

        all_columns = set().union(*(a.obs.columns for a in adatas))
        consistent = set()
        for col in all_columns:
            kinds = set()
            for a in adatas:
                if col not in a.obs.columns:
                    kinds = None
                    break
                kinds.add(_dtype_kind(a.obs[col]))
            if kinds and len(kinds) == 1 and next(iter(kinds)) is not None:
                consistent.add(col)

        sanitized = []
        for a in adatas:
            obs = a.obs.copy()
            for col in all_columns:
                if col not in obs.columns:
                    obs[col] = fill_value
            for col in all_columns:
                if col in consistent:
                    continue
                if pd.api.types.is_categorical_dtype(obs[col]):
                    obs[col] = obs[col].astype(object)
                obs[col] = obs[col].fillna(fill_value).astype(str)
            a.obs = obs
            sanitized.append(a)
        return sanitized

    def _ensure_reference(self, ref_model_name):
        """Train the named scArches reference target if the registry has no entry.

        Args:
            ref_model_name (str): Target name of the reference model.
        """
        entry = self.cache.find_entry(ref_model_name)
        if entry is None:
            self.logger.info(f"Reference model '{ref_model_name}' not found. Training...")
            original_target = self.target
            self.target = ref_model_name
            self.run()
            self.target = original_target

    def _target_save_dir(self, method):
        """Where this target's model, h5ad, figures and diagnostics all land.

        Read back from the registry, which is the single source of truth for that path,
        rather than recomputed from a formula that would have to stay in sync with
        integrate.py's and refmapping.py's own folder naming. Falls back to a
        method-based path only when no registry entry exists yet.

        Args:
            method (str): Integration method, used only by the fallback path.

        Returns:
            Path: The target's directory.
        """
        entry = self.cache.find_entry(self.target)
        model_path = entry.get('model_path') if entry else None
        if model_path:
            return Path(model_path)
        save_dir = self.output_dir / method / self.target
        self.logger.warning(
            f"[DimReduc] No model_path recorded for '{self.target}'; falling back to {save_dir}.")
        return save_dir

    def _dispatch_reduction(self, adata, method):
        """Compute UMAP + leiden on each of this method's latents, and plot if asked.

        Args:
            adata (AnnData): Object carrying the latent in `.obsm`.
            method (str): Names the latents to reduce, `X_{method}*`.

        Returns:
            AnnData: The object, with UMAP and leiden keys added.
        """
        latent_keys = [k for k in adata.obsm.keys() if k.startswith(f"X_{method}")]
        if not latent_keys:
            self.logger.warning(f"[DimReduc] No latent keys found for method '{method}' in adata.obsm.")
            return adata

        save_dir = self._target_save_dir(method)

        for key in latent_keys:
            if self.dimreduction != 'umap':
                continue
            result = DimReducer(adata, self.logger).compute(
                use_rep=key, metric='cosine', do_leiden=True, leiden_resolution=0.8)
            adata = result['adata']
            if self.plot:
                color_by = ["batch"]
                if "solo_doublet" in adata.obs.columns:
                    color_by.append("solo_doublet")
                UMAPPlotter(adata, save_dir=save_dir, logger=self.logger).plot(
                    umap_key=result['umap_key'], leiden_key=result['leiden_key'],
                    extra_color_by=color_by, title_prefix=f"UMAP ({key})")
        return adata

    def _apply_batch_key(self, adata, spec):
        """Materialise the `.obs` column named by `spec` and return its name.

        `none`/`single` return the existing column unchanged. `combined`/`mapped` build
        a column that does not exist yet -- after concatenation, so every sample's
        values are present, and before the integrator sees the data. Missing values are
        never filled: a component key absent from `.obs` raises instead. See
        docs/decisions/batch-keys-and-confounds.md.

        Args:
            adata (AnnData): The concatenated object, modified in place.
            spec (dict): Batch-key spec from `resolve_run_spec`, with 'mode', 'column'
                and (for the derived modes) 'components'.

        Returns:
            str or None: The `.obs` column to correct on.

        Raises:
            KeyError: If a component key is not an `.obs` column.
            ValueError: If the derived column has missing values.
        """
        mode = spec['mode']
        if mode in ('none', 'single'):
            return spec['column']

        missing = [k for k in spec['components'] if k not in adata.obs]
        if missing:
            raise KeyError(
                '[INTEGRATION] batch_key component(s) {} are not .obs columns, so the derived '
                'column {!r} cannot be built. Available: {}'.format(
                    ', '.join(repr(m) for m in missing), spec['column'],
                    ', '.join(sorted(adata.obs.columns))))

        if mode == 'combined':
            parts = [adata.obs[k].astype(str) for k in spec['components']]
        else:
            # Mapped: each sample contributes the keys named for it, so the
            # column is built row-wise per sample rather than column-wise. The
            # KEY NAMES go into the value ('sample=p1', not 'p1') so two samples
            # resolved through different keys cannot collide on one string.
            assignments = resolve_batch_assignments(spec, list(adata.obs['sample'].unique()))
            sample_col = adata.obs['sample'].astype(str)
            values = pd.Series(index=adata.obs.index, dtype=object)
            for sample, keys in assignments.items():
                rows = sample_col == sample
                if not rows.any():
                    continue
                values.loc[rows] = [
                    BATCH_KEY_SEP.join('{}={}'.format(k, v) for k, v in zip(keys, vals))
                    for vals in zip(*(adata.obs.loc[rows, k].astype(str) for k in keys))]
            parts = [values.astype(str)]

        if len(parts) == 1:
            derived = parts[0]
        else:
            derived = parts[0].str.cat(parts[1:], sep=BATCH_KEY_SEP)

        if derived.isna().any():
            raise ValueError(
                '[INTEGRATION] derived batch column {!r} has missing values for {} cells.'.format(
                    spec['column'], int(derived.isna().sum())))

        adata.obs[spec['column']] = pd.Categorical(derived)
        self.logger.info(
            '[INTEGRATION] batch_key %r derived from %s -> %d level(s): %s',
            spec['column'], ' + '.join(spec['components']),
            adata.obs[spec['column']].nunique(),
            ', '.join(map(str, sorted(adata.obs[spec['column']].unique())[:12])))
        return spec['column']

    def _will_train_contrast(self) -> bool:
        """Whether the contrastiveVI path is about to train rather than reload.

        Mirrors the reload condition in `ContrastiveIntegrator.integrate()` so a sweep
        is not run for a target that will simply load its cached model. Deliberately in
        step with that check: if one changes, this must.

        Returns:
            bool: True if training will happen.
        """
        if self.overwrite:
            return True
        entry = self.cache.find_entry(self.target)
        cached = bool(entry and 'model_path' in entry and Path(entry['model_path']).exists())
        if cached:
            self.logger.info(
                '[MAIN] %s has a cached model and overwrite is off, so no sweep will run -- '
                'the existing model is returned as-is. Re-run with --overwrite to sweep and '
                'retrain.', self.target)
        return not cached

    def _sweep_contrast(self, adata, args, batch_key):
        """Pick this contrast's `wasserstein_penalty` / `n_salient_latent` by grid.

        What `tune: true` would do if contrastiveVI could be autotuned -- it cannot,
        because its objective rewards exactly the leak the penalty exists to prevent,
        so every criterion here is computed outside the objective. See
        docs/decisions/contrastive-sweep.md.

        Args:
            adata (AnnData): The concatenated object. Swept on a copy.
            args (dict): Resolved run spec for this target.
            batch_key (str or None): Column to correct on.

        Returns:
            dict: The `model_args` to train with. On a grid where every point
                collapsed, the target's own model_params are returned unchanged and the
                run continues.
        """
        from .contrastive import (select_contrast_grid, best_grid_point, METHOD_NAME as _CVI)
        from .cvi_sweep import build_grid, DEFAULT_PENALTIES, DEFAULT_SALIENT

        grid_cfg = args.get('sweep_grid') or {}
        grid = build_grid(grid_cfg.get('wasserstein_penalty', DEFAULT_PENALTIES),
                          grid_cfg.get('n_salient_latent', DEFAULT_SALIENT))
        out_dir = self.output_dir / _CVI / self.target
        self.logger.info('[MAIN] sweep: %d grid point(s) for %s', len(grid), self.target)

        # Swept on a COPY: select_contrast_grid copies per point, but the
        # caller's adata must also survive untouched for the final fit, since
        # _prep overwrites .X and stamps _scvi_* columns.
        #
        # base_model_params IS NOT OPTIONAL despite its default -- without it the
        # grid ranks penalties at contrastive._MODEL_DEFAULTS' n_layers=1 while
        # the winner trains at the target's own depth.
        base_params = (args.get('model_params') or {}).get('model_params') or {}
        df = select_contrast_grid(
            adata.copy(), background_samples=args['background'],
            target_samples=args['target_samples'], target=self.target, grid=grid,
            covariates=args['covariates'], batch_key=batch_key,
            layer=(args.get('keep_layers') or ['corrected_counts'])[0],
            train_args=args.get('train_args'), base_model_params=base_params,
            out_dir=out_dir, logger=self.logger)

        chosen = best_grid_point(df)
        if not chosen:
            self.logger.warning(
                '[MAIN] sweep for %s selected NOTHING -- every grid point either failed or '
                'collapsed (salient space reduced to a group indicator). Training at the '
                "target's stated model_params instead. The grid is in %s; widen or lower "
                'sweep_grid.wasserstein_penalty and re-run.', self.target, out_dir)
            return args.get('model_params')

        self.logger.info('[MAIN] sweep for %s chose %s', self.target, chosen)
        merged = {**((args.get('model_params') or {}).get('model_params') or {}), **chosen}
        # Recorded so an automatic choice stays auditable: which point won, out
        # of what, and where the full table is.
        self.cache.upsert_entry(self.target, {
            'sweep_selected': chosen,
            'sweep_n_points': int(len(df)),
            'sweep_grid_path': str(out_dir / '{}_contrast_grid.tsv'.format(self.target))})
        return {'model_params': merged}

    def run(self):
        """Run this target end to end and return the annotated AnnData.

        Dispatches on the resolved method (scVI / contrastiveVI / scArches), then
        reduces, diagnoses and -- for a single-sample target -- embeds the result back
        into that sample's mapping report. Diagnostics and figures are gated on
        `--plot`.

        Returns:
            AnnData: The integrated object.

        Raises:
            Exception: Whatever the integrator raised, logged and re-raised.
        """
        self.logger.info('[MAIN] Loaded configs, starting integration')
        method, args = self._get_type(self.target)

        # A target may name its own artifact_key, overriding the file-level one.
        if args.get("artifact_key"):
            self.artifact_key = args["artifact_key"]

        # Read before the first _concatenate_adata call, since that is where it
        # takes effect. None = keep every layer (the historical behaviour).
        self.keep_layers = args.get("keep_layers")
        # Same timing, same reason. None = no gene filtering, which every target
        # written before this key existed relies on.
        self.min_cells = args.get("min_cells")

        # Kept so diagnostics can read .history for the training-loss curves.
        # None on the scArches path and on any failure before training.
        model = None
        # The .obs column this run actually corrected on, as materialised by
        # _apply_batch_key -- the batch-mixing diagnostics below are keyed on it.
        # None means "no correction requested", or the scArches path.
        batch_key = None
        try:
            if method == "scVI":
                adata = self._concatenate_adata(samples=args["samples"])
                batch_key = self._apply_batch_key(adata, args["batch_key_spec"])
                integrator = SingleCellIntegrate(adata, do_hvg=args["do_hvg"], do_tune=args["tune"], output_dir=self.output_dir,
                                                  covariates = args["covariates"], logger=self.logger, target = self.target,
                                                  model_class=args["model_class"], n_trials=args["n_trials"],
                                                  tune_metric=args["tune_metric"], tune_mode=args["tune_mode"],
                                                  model_args=args["model_params"], train_args=args["train_args"],
                                                  search_space=args.get("search_space"),
                                                  overwrite=self.overwrite)
                # batch_key comes from the target rather than integrate()'s own
                # default, so a treated cohort is not silently corrected on
                # `batch`. See docs/decisions/batch-keys-and-confounds.md.
                adata = integrator.integrate(batch_key=batch_key)
                model = integrator.model
            elif method == "contrastiveVI":
                adata = self._concatenate_adata(samples=args["samples"])
                batch_key = self._apply_batch_key(adata, args["batch_key_spec"])
                # Gated on actually being about to train: sweeping before a
                # cache hit would burn a grid and throw the result away.
                if args.get("sweep") and self._will_train_contrast():
                    args["model_params"] = self._sweep_contrast(adata, args, batch_key)
                integrator = ContrastiveIntegrator(
                    adata, background_samples=args["background"],
                    target_samples=args["target_samples"], target=self.target,
                    covariates=args["covariates"], logger=self.logger,
                    batch_key=batch_key, do_hvg=args["do_hvg"],
                    model_args=args["model_params"], train_args=args["train_args"],
                    overwrite=self.overwrite)
                adata = integrator.integrate()
                model = integrator.model
            elif method == "scArches":
                self._ensure_reference(args["reference_model"])
                ref_entry = self.cache.find_entry(args["reference_model"])
                # Not restricted to SCVI references: every class in
                # MODEL_CLASSES subclasses SCVI and inherits ArchesMixin's
                # load_query_data, so refmapping dispatches on the reference's
                # own class. This only rejects a class the build cannot
                # construct, before any GPU work.
                ref_model_class = ref_entry.get("model_class", "SCVI")
                resolve_model_class(ref_model_class)
                ref_model_path = ref_entry["model_path"]
                ref_anndata_path = ref_entry["anndata_path"]
                self.logger.info(f"Using {ref_model_class} reference model: {ref_model_path}")
                adata = self._concatenate_adata(reference = {args['reference'][0]:{'adata_path' : ref_anndata_path}}, query = args['query'])
                hvg_path = ref_entry.get('hvg_path', None)
                refmapper = scArchesIntegrator(adata = adata, ref_samples = args['reference'], ref_model_path = ref_model_path, ref_genes_path = hvg_path, query_samples = args['query'], output_dir=self.output_dir, logger=self.logger, name = self.target, model_class = ref_model_class, train_args = args.get('train_args'), ref_anndata_path = ref_anndata_path)
                adata = refmapper.MapReference(overwrite=self.overwrite)
                self.logger.info(adata)
            self.logger.info(f"[IntegrationManager] Completed integration for target '{self.target}'")
        except Exception as e:
            self.logger.error(f"[IntegrationManager] Integration failed for target '{self.target}': {e}")
            raise
        # Doublet detection is no longer run here; it is a per-sample mapping
        # step now (docs/decisions/doublet-detection.md). `.obs['solo_doublet']`
        # arrives with the mapping artifact and is consumed just below by
        # _dispatch_reduction's colouring.
        adata = self._dispatch_reduction(adata, method)

        # Training-loss curves plus label-free batch-mixing metrics, stamped
        # into .uns so they travel with the h5ad main.py writes next. Keyed on
        # what the model ACTUALLY corrected on -- the `or 'batch'` fallback is
        # deliberate, not just a None-guard. See
        # docs/decisions/batch-keys-and-confounds.md.
        if self.plot:
            # Only the background space genuinely should mix; the salient space
            # is designed to separate, so scoring it would mislead.
            diag_latents = (['X_contrastiveVI_background']
                            if method == 'contrastiveVI' else None)
            run_diagnostics(adata, method, self._target_save_dir(method), self.target,
                            model=model, batch_key=batch_key or 'batch',
                            latent_keys=diag_latents,
                            logger=self.logger, save_plot=True)

        # A one-sample run has a natural mapping-stage report to embed the
        # UMAP+summary into. scArches keys off the query alone; a contrast is
        # >= 2 samples by construction, so it never reaches the embed branch.
        resolved_samples = args["query"] if method == "scArches" else args["samples"]
        if len(resolved_samples) == 1:
            sample_name = resolved_samples[0]
            mapper = (self.metadata.get(sample_name) or {}).get('mapper')
            if mapper:
                embed_into_mapping_report(
                    adata, sample_name, self.input_dir, self.sample_metadata_path,
                    mapper=mapper, method=method,
                    model_class=args.get("model_class", "SCVI") if method == "scVI" else "SCVI",
                    tuned=bool(args.get("tune", False)) if method == "scVI" else False,
                    logger=self.logger, force=self.force)
            else:
                self.logger.warning(
                    f"[REPORT-EMBED] No mapper recorded for '{sample_name}' in sample_metadata.yaml; "
                    f"skipping report embedding.")
        return adata
