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

    Not SCVI-only: every class in MODEL_CLASSES subclasses SCVI and inherits
    ArchesMixin unchanged, so the surgery dispatches on the reference's own class
    and the fine-tuned output is of that same class.
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
        """Label each cell reference or query from its sample name.

        Returns:
            tuple: (reference AnnData, query AnnData).

        Raises:
            ValueError: If a named sample is absent from the loaded object.
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
            # (The previous inline version added missing genes as .obs columns
            # rather than vars, then indexed [:, genes], which KeyErrors on the
            # still-absent genes.)
            def subset_present(adata, genes):
                present = [g for g in genes if g in adata.var_names]
                return adata[:, present].copy()

            ref_adata = subset_present(ref_adata, self.ref_genes)
            query_adata = subset_present(query_adata, self.ref_genes)
        self.logger.info(f"[scArches] Reference cells: {ref_adata.n_obs}, Query cells: {query_adata.n_obs}")

        return ref_adata, query_adata

    def _load_hvg_list(self, path: str):
        """Read the reference's HVG list, so the query is subset to the same genes.

        Args:
            path (str | Path): The reference's `_HVG.tsv`.

        Returns:
            list[str] or None: Gene names, or None when the file is unusable.
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
        *exactly* the genes the model was registered against at training
        time. The `adata` this used to be called with (a subset of this
        run's own concatenated AnnData, i.e. `ref_adata` in `MapReference`)
        has already been outer-joined against the query's gene set in
        `_concatenate_adata` -- whenever the query was mapped to a different
        index than the reference, that pads the reference's own var space up
        to the ref+query union, which is not what the model was trained on
        and raises "Number of vars in `adata_target` not the same as
        source". Reload the reference's own untouched AnnData straight from
        disk instead -- the query's extra genes never need to touch the
        reference side at all. (This alone does not preserve those genes in
        the final output; `_map_query_ref` handles that half, since scArches's
        own `prepare_query_anndata` is destructive on the query side too.)
        Falls back to the (potentially padded) passed-in `adata` when no
        pristine path is available, so this stays working for any caller
        that does not supply one.
        """
        self.logger.info(f"[scArches] Loading {self.model_class} reference model from {self.ref_model_path}")
        if self.ref_anndata_path is not None and self.ref_anndata_path.is_file():
            adata = sc.read_h5ad(self.ref_anndata_path)
            if self.ref_genes is not None:
                present = [g for g in self.ref_genes if g in adata.var_names]
                adata = adata[:, present].copy()
        return self.model_cls.load(str(self.ref_model_path), adata = adata)

    def _map_query_ref(self, ref_model, ref_adata, query_adata, save: bool=True):
        """Run the surgery: align the query, load_query_data, then fine-tune.

        `prepare_query_anndata` is destructive, so it runs on a copy -- genes the query
        has beyond the reference's survive into the saved output rather than being
        silently deleted. `weight_decay=0.0` is the canonical scArches setting, merged
        into the target's own plan_kwargs so the KL warmup survives.

        Args:
            ref_model: The loaded reference model.
            ref_adata (AnnData): The reference's own registered object.
            query_adata (AnnData): The query cells.
            save (bool): Whether to write the model, h5ad and registry entry.

        Returns:
            The fine-tuned model.
        """
        # prepare_query_anndata is DESTRUCTIVE -- it subsets its argument down
        # to the reference's registered genes in place. On a copy, so genes the
        # query has beyond the reference's survive into the saved output.
        query_aligned = query_adata.copy()
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
            # The registry entry is written last, only once the anndata write
            # above has actually completed -- writing it earlier (as this used
            # to, right after model.save()) let a run killed mid-write (SLURM
            # walltime, OOM) leave a registry entry pointing at a truncated
            # .h5ad forever, since nothing re-validated it on the next run.
            # `MapReference`'s reload branch below now also tolerates this if
            # it happens anyway (e.g. from before this fix), but the ordering
            # here is what stops a fresh occurrence. Upsert, not add_entry --
            # a blind append here was the same duplicate-entry bug as
            # integrate.py's autotune save: an --overwrite rerun of an
            # existing name would otherwise create a second, orphaned entry
            # instead of merging into the first.
            self.registry.upsert_entry(self.name, {
                'reference_samples': self.ref_samples, 'query_samples': self.query_samples,
                'reference_model': str(self.ref_model_path), 'model_path': str(out_dir),
                'model_class': self.model_class,
                'anndata_path': str(out_dir / f"{self.name}.h5ad"),
                'tool_versions': integration_tool_versions(),
                'timestamp': datetime.datetime.now().isoformat()})
        return model, full_adata

    def _update_save_anndata(self, adata, model, save_dir: str, method = 'scArches', latent=None):
        """Attach the latent and write the AnnData beside the model.

        Args:
            adata (AnnData): Object to write.
            model: The fine-tuned model.
            save_dir (Path): Where to write.
            method (str): Names the obsm key.
            latent (np.ndarray, optional): Precomputed latent.
        """
        adata.obsm[f"X_{method}"] = latent if latent is not None else model.get_latent_representation(adata)
        # Write to a same-directory temp path and rename into place rather
        # than writing the final filename directly: a run killed mid-write
        # must leave no file at the trusted path, not a truncated one that
        # the next run's reload silently trusts (the same invariant
        # IndexBuilder's node-local-scratch staging follows for index
        # builds -- `os.replace` on the same filesystem is atomic).
        save_dir = Path(save_dir)
        final_path = save_dir / f"{self.name}.h5ad"
        tmp_path = save_dir / f".{self.name}.h5ad.partial-{os.getpid()}"
        adata.write_h5ad(tmp_path)
        os.replace(tmp_path, final_path)
        self.adata = adata

    def MapReference(self, overwrite: bool = False):
        """Map the query onto the reference, reloading a cached fit when one exists.

        The saved object legitimately carries more genes than the model was registered
        on, so a copy is aligned for the reload and predict calls only.

        Args:
            overwrite (bool): Retrain even when a cached model exists.

        Returns:
            AnnData: The mapped object.
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
                # Reuse the already-resolved model_path rather than
                # recomputing it -- recomputing risked silently diverging
                # from where the model was actually just loaded from if the
                # folder-naming formula ever changes.
                self._update_save_anndata(self.adata, model, save_dir = model_path,
                                          latent = model.get_latent_representation(aligned))
                return self.adata
            except Exception as e:
                # A registry entry is now only ever written once the anndata
                # write it points at has actually completed (see
                # _map_query_ref), but that has not always been true -- a run
                # killed mid-write (SLURM walltime, OOM) used to leave a
                # registry entry pointing at a truncated .h5ad forever, since
                # nothing here re-validated it before trusting it. Rather than
                # leaving this target permanently stuck on a corrupt cache
                # entry, fall through and retrain from scratch, exactly as
                # the "model_path missing" case below already does.
                self.logger.warning(
                    "[scArches] Cached model/anndata for '{}' at {} could not be reloaded "
                    "({}); retraining from scratch.".format(self.name, model_path, e))

        # Also reached when a registry entry exists but its model_path no longer
        # exists on disk (e.g. deleted after cleanup), or the cached reload
        # above failed -- previously this fell through both branches and
        # silently returned the untouched input adata without ever training.
        ref_adata, query_adata = self._split_reference_query()
        ref_model = self._load_reference_model(ref_adata)
        model, self.adata = self._map_query_ref(ref_model, ref_adata, query_adata)
        return self.adata
