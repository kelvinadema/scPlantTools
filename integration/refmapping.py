#!/usr/bin/env python3

import datetime
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

from pathlib import Path
from shared.utils import find_root_dir
from .integrationutils import CacheUtils, integration_tool_versions, resolve_model_class
from .integrate import normalize_kl_warmup

class scArchesIntegrator:
    """Map query samples onto a trained reference model by scArches surgery.

    Not SCVI-only: every class in MODEL_CLASSES carries ArchesMixin (SysVI
    without subclassing SCVI; see integrationutils.MODEL_CLASSES), so the surgery
    dispatches on the reference's own class and the fine-tuned output is of that
    same class.
    """
    def __init__(self, adata, ref_samples: list, ref_model_path: str, ref_genes_path: list, query_samples: list, output_dir: str, logger, name: str, model_class: str = 'SCVI', top_n_genes = 2000, train_args: dict = None, ref_anndata_path: str = None):
        self.adata = adata
        self.ref_samples = ref_samples
        self.ref_model_path = Path(ref_model_path)

        # The reference's own untouched AnnData. `self.adata`'s reference rows
        # went through ad.concat(join="outer") against the query, padding them up
        # to the union of ref+query genes; the model needs its exact registered
        # gene set, not that union.
        self.ref_anndata_path = Path(ref_anndata_path) if ref_anndata_path else None

        self.query_samples = query_samples
        self.output_dir = output_dir
        self.name = name

        # From the target's own train_args; without this the surgery would run
        # on scvi's defaults.
        self.train_args = dict(train_args or {})

        # From the reference's registry entry: the surgery dispatches on the
        # reference's actual class, and the fine-tuned output is that same class,
        # which the reload path and the registry entry must agree on.
        self.model_class = model_class
        self.model_cls = resolve_model_class(model_class)

        self.logger = logger
        self.registry = CacheUtils()

        self.top_n = top_n_genes
        self.ref_genes = self._load_hvg_list(ref_genes_path) if ref_genes_path else None

    def _split_reference_query(self):
        """Split the object into (reference, query) by sample name.

        Raises ValueError if a named sample is absent from the loaded object.
        """
        self.logger.info("[scArches] Assigning reference/query labels based on samples")
        missing_ref = set(self.ref_samples) - set(self.adata.obs["sample"])
        missing_query = set(self.query_samples) - set(self.adata.obs["sample"])
        if missing_ref or missing_query:
            raise ValueError(f"Reference/query samples not found in AnnData: {missing_ref | missing_query}")

        ref_adata = self.adata[self.adata.obs["sample"].isin(self.ref_samples)].copy()
        query_adata = self.adata[self.adata.obs["sample"].isin(self.query_samples)].copy()

        if self.ref_genes is not None:
            # Subset to the reference's gene set. Genes the object lacks are NOT
            # padded here: prepare_query_anndata (in _map_query_ref) reorders the
            # query to the reference registry and zero-pads any missing gene --
            # the canonical scArches alignment. The reference AnnData already
            # carries the trained genes, so its intersection is the full list.
            def subset_present(adata, genes):
                present = [g for g in genes if g in adata.var_names]
                return adata[:, present].copy()

            ref_adata = subset_present(ref_adata, self.ref_genes)
            query_adata = subset_present(query_adata, self.ref_genes)
        self.logger.info(f"[scArches] Reference cells: {ref_adata.n_obs}, Query cells: {query_adata.n_obs}")

        return ref_adata, query_adata

    def _load_hvg_list(self, path: str):
        """The top `top_n` genes of the reference's `_HVG.tsv` by consensus_rank.

        Raises ValueError if the file lacks the `gene`/`consensus_rank` columns.
        """
        df = pd.read_csv(path, sep='\t')
        if 'gene' not in df.columns or 'consensus_rank' not in df.columns:
            raise ValueError("HVG file must contain 'gene' and 'consensus_rank' columns")
        df_sorted = df.sort_values('consensus_rank')
        top_genes = df_sorted['gene'].head(self.top_n).tolist()
        self.logger.info(f"[scArches] Loaded {len(top_genes)} HVGs from {path} (top {self.top_n} by consensus_rank)")
        return top_genes

    def _load_reference_model(self, adata):
        """
        Reload the trained reference model.

        scvi's `<Model>.load(dir_path, adata=...)` requires `adata` to carry
        *exactly* the genes the model was registered against. The passed-in
        `ref_adata` has been outer-joined against the query in
        `_concatenate_adata`, which pads it to the ref+query gene union whenever
        the two were mapped to different indices ("Number of vars in
        `adata_target` not the same as source"). So the reference's own
        untouched AnnData is reloaded from disk; the passed-in `adata` is only
        the fallback when no such path is configured. Keeping the query's extra
        genes in the output is `_map_query_ref`'s job.
        """
        self.logger.info(f"[scArches] Loading {self.model_class} reference model from {self.ref_model_path}")
        if self.ref_anndata_path is not None and self.ref_anndata_path.is_file():
            adata = sc.read_h5ad(self.ref_anndata_path)
            if self.ref_genes is not None:
                present = [g for g in self.ref_genes if g in adata.var_names]
                adata = adata[:, present].copy()
        return self.model_cls.load(str(self.ref_model_path), adata = adata)

    def _blank_query_labels(self, query_adata, ref_model):
        """Reset the query's labels column to the reference's unlabeled category.

        Only touches the surgery copy, never the object that gets saved, so the
        query's own `.obs` survives into the output unchanged.

        WHY THIS IS NOT OPTIONAL, and why it is not scANVI-only. A reference
        trained with a real `labels_key` registers that column's categories, and
        `transfer_field` re-runs the query through the SAME CategoricalDtype with
        `extend_categories=False` -- for SCANVI explicitly (it drops the kwarg and
        hardcodes False, "don't extend labels for query data"), for a plain
        CategoricalObsField by default. `_make_column_categorical` then raises
        ValueError on any value outside that mapping, because unmatched rows come
        back as code -1.

        The query reliably HAS such a value. Orchestrator's `_sanitize_obs` fills
        a column a sample lacks with the literal 'Unknown', and the query samples'
        mapping artifacts carry no labels column at all -- so every query cell
        arrives labelled 'Unknown', which is in no reference mapping. scvi's own
        `allow_missing_labels` path does not fire either: it only helps when the
        column is ABSENT, and sanitization guarantees it is present.

        Both refine-derived references here are affected (each registers
        `labels_key: seed_combined`), which is why this is keyed on the model's
        registry rather than on `model_class == 'SCANVI'`. A reference trained
        with no labels_key registers the synthetic '_scvi_labels' and is skipped.
        `query_adata` is modified in place.
        """
        from scvi import REGISTRY_KEYS

        try:
            state = ref_model.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
        except (KeyError, AttributeError):
            return

        labels_key = getattr(state, 'original_key', None)
        # A reference set up with labels_key=None registers the synthetic
        # '_scvi_labels' constant instead; there is no user column to blank and
        # transfer_field builds it itself.
        if not labels_key or str(labels_key).startswith('_scvi'):
            return

        # `list(...)` around it, never `x or []` -- categorical_mapping is a numpy
        # array and its truth value raises on more than one element.
        raw_mapping = getattr(state, 'categorical_mapping', None)
        mapping = [] if raw_mapping is None else [str(m) for m in raw_mapping]
        unlabeled = getattr(state, 'unlabeled_category', None)
        if unlabeled is None:
            # A plain SCVI reference has no unlabeled_category, but the query is
            # genuinely unlabelled and still has to land on a REGISTERED level.
            # The refine cohorts carry scANVI's own 'Unassigned' as a real class,
            # so prefer it and fall back to the first level only to stay running
            # on a reference that has no such convention.
            unlabeled = 'Unassigned' if 'Unassigned' in mapping else (mapping[0] if mapping else None)
        if unlabeled is None:
            return

        present = set(query_adata.obs[labels_key].astype(str).unique()) if labels_key in query_adata.obs else set()
        foreign = sorted(present - set(map(str, mapping)))
        self.logger.info(
            "[scArches] Blanking query labels '{}' to '{}' for surgery ({} cells){}".format(
                labels_key, unlabeled, query_adata.n_obs,
                "; unregistered values replaced: {}".format(foreign) if foreign else ""))
        query_adata.obs[labels_key] = pd.Categorical(
            [unlabeled] * query_adata.n_obs, categories=mapping or [unlabeled])

    def _map_query_ref(self, ref_model, ref_adata, query_adata, save: bool=True):
        """Run the surgery: align the query, load_query_data, then fine-tune.

        `prepare_query_anndata` is destructive, so it runs on a copy -- genes the query
        has beyond the reference's survive into the saved output rather than being
        silently deleted. `weight_decay=0.0` is the canonical scArches setting, merged
        into the target's own plan_kwargs so the KL warmup survives.

        `save` writes the model, the h5ad and the registry entry. Returns
        (fine-tuned model, full outer-joined AnnData).
        """
        query_aligned = query_adata.copy()
        self._blank_query_labels(query_aligned, ref_model)
        self.model_cls.prepare_query_anndata(query_aligned, ref_model)
        model = self.model_cls.load_query_data(query_aligned, ref_model)
        # weight_decay=0.0 is the canonical scArches setting: it freezes the
        # reference-cell representation during surgery. Merged into the target's
        # plan_kwargs rather than replacing them, so the KL warmup survives.
        # See docs/decisions/kl-warmup.md.
        train_kwargs = normalize_kl_warmup(self.train_args, query_adata.n_obs, self.logger)
        plan_kwargs = dict(train_kwargs.pop('plan_kwargs', None) or {})
        plan_kwargs.setdefault('weight_decay', 0.0)
        self.logger.info(
            "[scArches] Fine-tuning query with train_args={}, plan_kwargs={}".format(
                train_kwargs, plan_kwargs))
        model.train(plan_kwargs=plan_kwargs, **train_kwargs)
        if save:
            # scArches fine-tunes a copy of the reference's own model, whatever
            # class it is, so the saved model is of that class -- recorded in the
            # entry below and read back by the reload path so SCVI.load isn't
            # handed a JointEmbeddingSCVI (scvi's own load() raises on that
            # mismatch). The output folder is the fixed 'scArches' name
            # regardless of class, matching the results/integration/scArches/
            # convention.
            out_dir = find_root_dir() / "results/integration/scArches" / self.name
            model.save(out_dir, overwrite=True)
            # Latents are computed on the gene-matched objects (`ref_model`'s
            # own registered adata; `query_aligned`, whose row/obs order
            # `.copy()` preserved from `query_adata`) and then attached onto
            # the full, ungated `ref_adata`/`query_adata` -- outer-joined so
            # every gene either side carries survives, rather than the
            # `join="inner"` default silently reducing back down to the
            # reference-only intersection. Row order matches by construction:
            # `prepare_query_anndata`/`_inplace_subset_var` only ever touch
            # the var axis, never obs order.
            ref_latent = ref_model.get_latent_representation()
            query_latent = model.get_latent_representation(query_aligned)
            full_adata = ad.concat([ref_adata, query_adata], join="outer")
            latent = np.concatenate([ref_latent, query_latent], axis=0)
            self._update_save_anndata(full_adata, model, save_dir = out_dir, latent = latent)
            # The registry entry is written last, once the h5ad write has
            # completed, so a run killed mid-write (walltime, OOM) cannot leave
            # an entry pointing at a truncated file. Upsert, so an --overwrite
            # rerun merges into the existing entry instead of adding a second.
            self.registry.upsert_entry(self.name, {
                'reference_samples': self.ref_samples, 'query_samples': self.query_samples,
                'reference_model': str(self.ref_model_path), 'model_path': str(out_dir),
                'model_class': self.model_class,
                'anndata_path': str(out_dir / f"{self.name}.h5ad"),
                'tool_versions': integration_tool_versions(),
                'timestamp': datetime.datetime.now().isoformat()})
        return model, full_adata

    def _update_save_anndata(self, adata, model, save_dir: str, method = 'scArches', latent=None):
        """Attach the latent as `.obsm['X_{method}']` and write the AnnData beside the model."""
        adata.obsm[f"X_{method}"] = latent if latent is not None else model.get_latent_representation(adata)
        # Temp path + same-filesystem `os.replace` (atomic): a run killed
        # mid-write leaves no file at the trusted path rather than a truncated
        # one the next run's reload would trust.
        save_dir = Path(save_dir)
        final_path = save_dir / f"{self.name}.h5ad"
        tmp_path = save_dir / f".{self.name}.h5ad.partial-{os.getpid()}"
        adata.write_h5ad(tmp_path)
        os.replace(tmp_path, final_path)
        self.adata = adata

    def MapReference(self, overwrite: bool = False):
        """Map the query onto the reference, reloading a cached fit when one exists.

        `overwrite` retrains even when a cached model exists. Returns the mapped
        AnnData.
        """
        entry = self.registry.find_entry(self.name)
        if not overwrite and entry and "model_path" in entry and Path(entry["model_path"]).exists():
            model_path = Path(entry["model_path"])
            adata_path = Path(entry.get("anndata_path", ""))
            self.logger.info(f"[scArches] Found existing {self.model_class} model at {model_path}, loading...")
            try:
                self.adata = sc.read_h5ad(adata_path)
                # The saved object legitimately carries more genes than the
                # model was registered on, and scvi's load() needs an exact
                # match -- so align a COPY for the reload only. A pure subset,
                # not a re-pad: the model's genes are a subset of these.
                aligned = self.model_cls.prepare_query_anndata(
                    self.adata, str(model_path), inplace=False)
                # Same class as the fine-tuned model that was saved here (see
                # the upsert in _map_query_ref) -- scvi's load() would raise
                # if this were SCVI against a JointEmbeddingSCVI checkpoint.
                model = self.model_cls.load(model_path, adata=aligned)
                # Write back to where the model was loaded from, not a
                # recomputed path.
                self._update_save_anndata(self.adata, model, save_dir = model_path,
                                          latent = model.get_latent_representation(aligned))
                return self.adata
            except Exception as e:
                # A cache entry that cannot be reloaded (e.g. a truncated h5ad
                # from an older run) falls through to retraining, like a
                # missing model_path, instead of leaving the target stuck.
                self.logger.warning(
                    "[scArches] Cached model/anndata for '{}' at {} could not be reloaded "
                    "({}); retraining from scratch.".format(self.name, model_path, e))

        # Also reached when a registry entry's model_path no longer exists on
        # disk, or the cached reload above failed.
        ref_adata, query_adata = self._split_reference_query()
        ref_model = self._load_reference_model(ref_adata)
        model, self.adata = self._map_query_ref(ref_model, ref_adata, query_adata)
        return self.adata
