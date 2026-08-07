#!/usr/bin/env python3

import os
import json
import fcntl
import logging
import importlib.metadata

from contextlib import contextmanager
from shared.utils import find_root_dir
from scvi.model import SCVI
from scvi.external import JointEmbeddingSCVI, DRVI
import anndata

# Mapping-stage h5ads carry pandas nullable StringArray obs columns (e.g.
# 'barcode'); anndata >=0.12 (this env, upgraded alongside scvi-tools to 1.5.0)
# refuses to write those unless explicitly opted in -- otherwise every
# write_h5ad call in this package (model save, single-sample output, R export
# manifests) raises RuntimeError. This module is imported by every entrypoint
# in integration, so set it once here rather than at each write site.
anndata.settings.allow_write_nullable_strings = True

# Every class here must subclass scvi.model.SCVI: SingleCellIntegrate drives them
# all through one `setup_anndata` -> `cls(adata, **model_params)` -> `train(**kw)`
# path, and refmapping relies on the inherited ArchesMixin for query mapping.
#
# DRVI qualifies (its MRO is SCVI -> EmbeddingMixin -> RNASeqMixin; it swaps only
# the decoder for the additive split decoder). scvi.external.ContrastiveVI does
# NOT and is deliberately absent -- it subclasses BaseModelClass and its train()
# takes two required positional index lists, so it is wired in as a *method*
# (integration/contrastive.py), not as a model_class here.
MODEL_CLASSES = {"SCVI": SCVI, "JointEmbeddingSCVI": JointEmbeddingSCVI, "DRVI": DRVI}


def resolve_model_class(name: str):
    """The scvi model class a `model_class` name refers to.

    Args:
        name (str): A key of MODEL_CLASSES.

    Returns:
        type: The model class.

    Raises:
        ValueError: If the name is unknown.
    """
    try:
        return MODEL_CLASSES[name]
    except KeyError:
        raise ValueError(f"Unknown model_class '{name}'. Must be one of {sorted(MODEL_CLASSES)}")


def model_output_folder(model_class: str) -> str:
    """
    Per-target output directory name for a scVI-method run:
    results/integration/<model_output_folder(model_class)>/<target>/, holding
    everything for that target (model.pt, h5ad, HVG file if generated, UMAP
    figures) in one place. 'SCVI' maps to the existing 'scVI' display
    convention; any other model_class (e.g. 'JointEmbeddingSCVI') uses its own
    name verbatim, since there's no existing folder convention to match.
    scArches targets aren't a model_class in this sense and use the literal
    'scArches' folder name directly at their own call site.
    """
    return 'scVI' if model_class == 'SCVI' else model_class


# validate_solo_covariates was removed along with `run_solo`. The constraint it
# enforced is real and still applies -- SOLO.from_scvi_model raises on a source
# model carrying any registered categorical or continuous covariate -- but it is
# no longer a config combination anyone can express here: doublet detection
# moved to the mapping stage, where its model is trained with no covariates by
# construction (mapping/doublets.py). Integration targets are now free
# to use covariates without interacting with doublet calling at all.


_INTEGRATION_VERSION_PACKAGES = ('scvi-tools', 'scanpy', 'anndata', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'ray', 'torch')
_INTEGRATION_VERSION_CACHE = None


def integration_tool_versions() -> dict:
    """
    Record the package versions that can change a trained model's behavior even
    under identical config -- a scvi-tools/ray/torch upgrade changes results,
    same reasoning as shared/io.py::tool_versions() for the mapping stage's
    simpleaf/piscem/alevin-fry. Not the same function: those are external
    binaries this stage never touches, and this stage's packages (scvi-tools,
    ray, torch, ...) aren't ones mapping cares about.

    Memoized (the environment cannot change mid-run) and never raises -- a
    provenance stamp must not be able to fail a training run.
    """
    global _INTEGRATION_VERSION_CACHE
    if _INTEGRATION_VERSION_CACHE is not None:
        return dict(_INTEGRATION_VERSION_CACHE)

    import platform
    versions = {'python': platform.python_version()}
    missing = []
    for package in _INTEGRATION_VERSION_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except Exception:
            versions[package] = ''
            missing.append(package)

    if missing:
        logging.getLogger('IntegrateLogging').warning(
            '[INTEGRATION] Could not determine version for: {}'.format(', '.join(missing)))

    _INTEGRATION_VERSION_CACHE = versions
    return dict(versions)


# `parse_covariates` moved to shared/targets.py::parse_covariates. It is pure
# config resolution with no scvi dependency, and this module imports scvi at
# module scope -- which would have dragged the whole integration env into
# shared/validate.py the moment the validator needed to resolve a target.


class CacheUtils:
    """Read/write the integration registry, under a file lock."""
    def __init__(self, registry_name: str="scvi_tune_registry.json", subdir: str=""):
        # No wrapper folder by default -- results/integration/scvi_tune_registry.json
        # directly. (Path joins with '' are a no-op, so subdir="" resolves to the
        # parent directory itself.) Previously defaulted to "scvi_models", which
        # existed only to hold per-target model output; that output now lives
        # under results/integration/<model_class or 'scArches'>/<target>/ instead,
        # so the registry no longer needs a subdir of its own.
        self.registry_path = find_root_dir() / "results" / "integration" / subdir / registry_name
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        # The lock is a SEPARATE file, never the registry itself. save_registry
        # replaces the registry by atomic rename, so a lock held on the registry
        # inode would be held on a file that is about to stop being the
        # registry -- the next writer would open the new inode and lock nothing.
        # A sidecar file is never replaced, so every process locks one inode.
        self.lock_path = self.registry_path.with_suffix(self.registry_path.suffix + ".lock")

    @contextmanager
    def _locked(self):
        """
        Hold an exclusive advisory lock on the registry for the duration of a
        read-modify-write.

        This has to wrap the WHOLE upsert, not the individual load/save calls:
        the race is between one process's read and its own write, so locking
        load_registry and save_registry separately would leave the window it is
        meant to close exactly as wide as before.

        fcntl.flock is advisory and cooperative -- it excludes other processes
        that take the same lock, and nothing else. /lustre here is mounted with
        the `flock` option (checked: `mount | grep lustre`), so the lock is
        coherent across compute nodes, which is what makes it usable for
        concurrent array tasks at all rather than only for threads on one node.
        Note this is unrelated to HDF5_USE_FILE_LOCKING=FALSE in the batch
        scripts: that disables HDF5's own locking of data files, not flock.

        The lock is released by close() on the way out even if the body raises,
        so a crashed writer cannot wedge the registry for later runs.
        """
        with open(self.lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def load_registry(self):
        """Read the registry, returning an empty list when it does not exist yet.

        Returns:
            list[dict]: The entries.
        """
        if not self.registry_path.exists() or self.registry_path.stat().st_size == 0:
            return []
        try:
            with open(self.registry_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def save_registry(self, registry):
        """Write the registry atomically.

        Args:
            registry (list[dict]): The entries to write.
        """
        # Write-then-rename rather than truncate-in-place. os.replace is atomic
        # within a filesystem, so a reader sees either the whole old registry or
        # the whole new one, never a half-written file. That matters more here
        # than usual because load_registry swallows JSONDecodeError and returns
        # [] -- a truncated write would not raise, it would silently present an
        # EMPTY registry, and the next save would then persist that emptiness.
        # A crash mid-write now leaves the stale-but-valid registry instead.
        tmp_path = self.registry_path.with_suffix(
            self.registry_path.suffix + f".partial-{os.getpid()}"
        )
        try:
            with open(tmp_path, "w") as f:
                json.dump(registry, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.registry_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def add_entry(self, entry: dict):
        """Append an entry, taking the lock.

        Args:
            entry (dict): The entry.
        """
        with self._locked():
            self._add_entry_unlocked(entry)

    def _add_entry_unlocked(self, entry: dict):
        """Append an entry. The caller must already hold the lock.

        Args:
            entry (dict): The entry.
        """
        registry = self.load_registry()
        registry.append(entry)
        self.save_registry(registry)

    def update_entry(self, dataset_id: str, updates: dict):
        """Merge updates into one entry, taking the lock.

        Args:
            dataset_id (str): Entry key.
            updates (dict): Fields to merge.
        """
        with self._locked():
            return self._update_entry_unlocked(dataset_id, updates)

    def _update_entry_unlocked(self, dataset_id: str, updates: dict):
        """Merge updates into one entry. The caller must already hold the lock.

        Args:
            dataset_id (str): Entry key.
            updates (dict): Fields to merge.
        """
        registry = self.load_registry()
        updated = False
        for entry in registry:
            if entry["dataset_id"] == dataset_id:
                entry.update(updates)
                updated = True
                break
        if updated:
            self.save_registry(registry)
        return updated

    def find_entry(self, dataset_id: str):
        """The entry for `dataset_id`, or None.

        Args:
            dataset_id (str): Entry key.

        Returns:
            dict or None: The entry.
        """
        # Read-only, so it takes the lock only to avoid reading a registry
        # mid-update. It cannot observe a torn file (see save_registry), but it
        # can still observe a stale one without this.
        with self._locked():
            registry = self.load_registry()
        for entry in registry:
            if entry["dataset_id"] == dataset_id:
                return entry
        return None

    def upsert_entry(self, dataset_id: str, updates: dict):
        """
        Update the existing entry for dataset_id if one exists, else create it.

        The one write path every registry mutation should go through -- a bare
        add_entry() call for a dataset_id that might already exist creates a
        second, orphaned entry instead of merging into the first (the bug this
        method replaces: integrate.py's autotune save and refmapping.py's
        scArches save both used to call add_entry directly).

        The lock is taken ONCE here and the unlocked helpers are called beneath
        it. flock is per (process, file description) rather than reentrant per
        acquire, so calling the public update_entry/add_entry here would have
        the inner `with` release the lock on exit and leave the rest of this
        method running unprotected -- which is exactly the create-then-update
        window a second process must not interleave with.
        """
        with self._locked():
            if not self._update_entry_unlocked(dataset_id, updates):
                self._add_entry_unlocked({'dataset_id': dataset_id})
                self._update_entry_unlocked(dataset_id, updates)


 