#!/usr/bin/env python3

"""
Standardized artifact IO for the pipeline.

Every terminal artifact a stage writes is stamped with a provenance record so
downstream stages can identify it without reading the producing stage's config.
The stamp lives in `.uns['pipeline']`; see SCHEMA_VERSION for the contract.
"""

import datetime
import hashlib
import importlib.metadata
import json
import logging

from .utils import active_logger
import os
import platform
import subprocess
import warnings

from pathlib import Path
from typing import Optional, Union

# Bump on any breaking change to the .uns['pipeline'] payload or the manifest.
# What each bump meant is in docs/decisions/artifact-schema.md. Note 1.3 made
# 'X_umap_qc'/'leiden_qc' CONDITIONAL: consumers must check for the key.
SCHEMA_VERSION = '1.4'

PROVENANCE_KEY = 'pipeline'

# External binaries the mapping stage shells out to. Rust tools, so `--version`
# is cheap; CellBender is read from package metadata instead because its CLI
# imports torch and would cost seconds per call.
_VERSION_BINARIES = ('simpleaf', 'piscem', 'alevin-fry')
_VERSION_PACKAGES = ('scanpy', 'anndata', 'numpy', 'pandas',
                     'scipy', 'scikit-learn', 'umap-learn')

# Distribution name -> import name, for the packages where they differ. Used to
# cross-check package metadata against what actually imports; see
# _package_version.
_IMPORT_NAMES = {
    'scikit-learn': 'sklearn',
    'umap-learn': 'umap',
    'scvi-tools': 'scvi',
}

_TOOL_VERSION_CACHE = None


def _module_version(dist: str) -> str:
    """
    `dist`'s version as reported by the imported module, or '' if unavailable.

    Warnings are suppressed because reading `__version__` is itself deprecated
    in scanpy and anndata, which emit a FutureWarning telling the caller to use
    importlib.metadata -- exactly the source being cross-checked here. The
    deprecation is not actionable in this context and would otherwise fire into
    every run's log.
    """
    module_name = _IMPORT_NAMES.get(dist, dist.replace('-', '_'))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            module = importlib.import_module(module_name)
            return str(getattr(module, '__version__', '') or '')
    except Exception:
        return ''


def _package_version(dist: str) -> str:
    """
    Version of an installed package, cross-checked against what actually imports.

    `importlib.metadata` reads `*.dist-info` off disk, which can disagree with
    the module the interpreter really loads: an in-place upgrade that leaves the
    old dist-info behind is enough, and this cluster has a live example --
    scIntegrationV1.12's metadata reports numpy 1.26.4 while `numpy.__version__`
    is 2.2.0. Recording the metadata value there would put a version in the
    provenance record that never ran, which is worse than recording nothing
    because it looks authoritative.

    Metadata is the sanctioned API so it stays primary, but when the two
    disagree the imported module wins -- it is what the code actually ran
    against -- and the drift is logged, since stale dist-info is an environment
    fault worth repairing rather than silently routing around.
    """
    try:
        meta = importlib.metadata.version(dist)
    except Exception:
        meta = ''
    module = _module_version(dist)

    if meta and module and meta != module:
        logging.getLogger('SingleCellMappingPipeline').warning(
            '[IO] {dist}: package metadata reports {meta} but the imported module reports '
            '{module}; recording {module}. This environment has stale *.dist-info -- the '
            'package was likely upgraded in place.'.format(dist=dist, meta=meta, module=module))
        return module
    return meta or module


def _env_package_version(env: Optional[str], dist: str) -> str:
    """
    Version of `dist` as installed in another conda env, via its interpreter.

    Shared by CellBender and scvi-tools, both of which run as subprocesses from
    their own environments (see the doublets/cellbender blocks in
    pipeline_config.yaml) and so are invisible to this interpreter's metadata.
    Asks that env's python rather than running the tool's own CLI, which for
    both of these imports torch and costs seconds per call.

    Applies the same prefer-the-module rule as _package_version.
    """
    if not env:
        return ''
    python = Path(env) / 'bin' / 'python'
    if not python.exists():
        return ''
    module_name = _IMPORT_NAMES.get(dist, dist.replace('-', '_'))
    probe = (
        'import importlib, importlib.metadata as m, warnings\n'
        'meta = mod = ""\n'
        'try:\n'
        '    meta = m.version({dist!r})\n'
        'except Exception:\n'
        '    pass\n'
        'try:\n'
        '    with warnings.catch_warnings():\n'
        '        warnings.simplefilter("ignore")\n'
        '        mod = str(getattr(importlib.import_module({mod!r}), "__version__", "") or "")\n'
        'except Exception:\n'
        '    pass\n'
        'print(mod if (meta and mod and meta != mod) else (meta or mod))\n'
    ).format(mod=module_name, dist=dist)
    try:
        proc = subprocess.run([str(python), '-c', probe],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return ''
    if proc.returncode != 0:
        return ''
    return proc.stdout.decode('utf-8', errors='replace').strip()


def _cellbender_version() -> str:
    """
    CellBender's version, read from its own environment.

    CellBender is not installed alongside the mapping stage (it pins torch and
    will not install on 3.12), so importlib.metadata here would report nothing --
    silently dropping the version of a tool that changes counts. Ask its
    interpreter instead. `cellbender --version` would also work but imports
    torch and costs seconds.
    """
    env = os.environ.get('CELLBENDER_ENV')
    if not env:
        # No isolated env: it may share this interpreter (the legacy layout).
        return _package_version('cellbender')
    return _env_package_version(env, 'cellbender')


def _scvi_version() -> str:
    """
    scvi-tools' version, read from the doublet-detection environment.

    Same situation as CellBender and for the same reason: scvi-tools is not
    installed alongside the mapping stage (the mapping env is on pandas 3.x,
    which scvi-tools does not support), so doublet calling runs as a subprocess
    under its own env and this interpreter's metadata knows nothing about it.
    A scvi-tools upgrade changes doublet calls, so the version belongs in the
    provenance record next to simpleaf's and piscem's.
    """
    env = os.environ.get('DOUBLET_ENV')
    if not env:
        return _package_version('scvi-tools')
    return _env_package_version(env, 'scvi-tools')


def _binary_version(binary: str) -> str:
    """Return `binary --version` output, or '' if it cannot be determined."""
    try:
        proc = subprocess.run([binary, '--version'], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ''
    if proc.returncode != 0:
        return ''
    out = proc.stdout.decode('utf-8', errors='replace').strip().splitlines()
    return out[0].strip() if out else ''


def tool_versions() -> dict:
    """
    Record the versions of everything that can change mapping output.

    Config hashes pin the parameters but not the tools, and a simpleaf or piscem
    upgrade changes results. Memoized: the environment cannot change mid-run, and
    this would otherwise re-shell on every artifact write.

    Never raises and never yields None — a provenance stamp must not be able to
    fail a run, and anndata silently drops None-valued .uns keys.
    """
    global _TOOL_VERSION_CACHE
    if _TOOL_VERSION_CACHE is not None:
        return dict(_TOOL_VERSION_CACHE)

    versions = {'python': platform.python_version()}
    for binary in _VERSION_BINARIES:
        versions[binary] = _binary_version(binary)
    for package in _VERSION_PACKAGES:
        versions[package] = _package_version(package)
    # Both live in their own environments; see _cellbender_version/_scvi_version.
    versions['cellbender'] = _cellbender_version()
    versions['scvi-tools'] = _scvi_version()

    missing = [k for k, v in versions.items() if not v]
    if missing:
        logging.getLogger('SingleCellMappingPipeline').warning(
            '[IO] Could not determine version for: {}'.format(', '.join(missing)))

    _TOOL_VERSION_CACHE = versions
    return dict(versions)


def config_hash(config_paths: Optional[list]) -> Optional[str]:
    """
    Hash the resolved config files a run was driven by, for reproducibility.

    Files are hashed in sorted order so the digest is stable regardless of the
    order the caller happens to pass them in. Returns None if no paths are
    given, and also (after logging) if any path is unreadable -- rather than
    raising, since a provenance stamp must never be the thing that fails a run.
    """
    if not config_paths:
        return None

    logger = logging.getLogger('SingleCellMappingPipeline')
    digest = hashlib.sha256()
    for path in sorted(str(p) for p in config_paths):
        try:
            with open(path, 'rb') as f:
                digest.update(f.read())
        except OSError as e:
            logger.warning('[IO] Could not hash config {}: {}'.format(path, e))
            return None
    return digest.hexdigest()


def stamp_provenance(adata, sample: str, mapper: str, stage: str, config_paths: Optional[list]=None, extra: Optional[dict]=None):
    """
    Write the provenance record into `adata` in place and return it.

    Must be called after any step that replaces `.obs` wholesale — CellBender
    reintegration at preprocess.py does exactly that, so a stamp applied
    earlier would be silently discarded.
    """
    # anndata silently drops None-valued .uns keys on write, which would leave the
    # contract a key short. Empty string keeps the key present and the type stable.
    record = {
        'schema_version': SCHEMA_VERSION,
        'sample': sample,
        'mapper': mapper,
        'stage': stage,
        'timestamp': datetime.datetime.now().isoformat(),
        'config_hash': config_hash(config_paths) or '',
        'source_configs': [str(p) for p in config_paths] if config_paths else [''],
        'tool_versions': tool_versions()}

    if extra:
        # Same None rule as the contract fields above, applied to caller-supplied
        # keys. Without this the coercion covers exactly the fields that were
        # never going to be None anyway, and the one path a caller controls can
        # still put a key into the record that vanishes on write.
        record.update({k: ('' if v is None else v) for k, v in extra.items()})

    adata.uns[PROVENANCE_KEY] = record
    adata.obs['sample'] = sample
    return adata


def write_h5ad(adata, path: Union[str, Path], sample: str, mapper: str, stage: str, config_paths: Optional[list]=None, extra: Optional[dict]=None):
    """
    Stamp provenance onto `adata`, then write it to `path`.

    Use for artifacts a later stage consumes. Purely transient scratch files
    (CellBender's temp input, raw per-mapper matrices) can keep using
    `adata.write_h5ad` directly.
    """
    # active_logger, not a hardcoded name: the calling stage may have configured
    # a different logger (see shared/utils.py::active_logger).
    logger = active_logger()
    stamp_provenance(adata, sample=sample, mapper=mapper, stage=stage, config_paths=config_paths, extra=extra)
    # Terminal writes land at the end of multi-hour jobs; failing here because a
    # parent directory is absent loses the whole run's output. Callers normally
    # go through get_sample_output_paths(create=True), so this is a backstop.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)
    logger.info('[IO] Wrote "{}" (stage={}, schema={})'.format(path, stage, SCHEMA_VERSION))
    return adata


def _json_safe(value):
    """
    Coerce a value read out of `.uns` into something json can write.

    anndata round-trips lists as numpy arrays and numbers as numpy scalars, so
    anything lifted from `.uns` into the manifest arrives in a form
    `json.dump` refuses -- "Object of type ndarray is not JSON serializable",
    raised at the very end of a sample after every stage has succeeded.
    Applied to the whole manifest, where the serialization happens.
    """
    import numpy as _np
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, _np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, _np.generic):
        return value.item()
    return value


def write_manifest(path: Union[str, Path], sample: str, mapper: str, artifacts: dict, base_dir: Union[str, Path], batch: Optional[str]=None, config_paths: Optional[list]=None, status: str='complete', extra: Optional[dict]=None) -> dict:
    """
    Write a per-sample manifest describing what the run produced.

    Sits above the mapper directory, so a consumer can discover a sample's
    outputs by globbing `{output_dir}/*/manifest.json` without knowing which
    mapper ran. Artifact paths are stored relative to `base_dir` to keep the
    results tree relocatable. Only written on success — its presence means the
    sample completed.
    """
    logger = logging.getLogger('SingleCellMappingPipeline')
    base = Path(base_dir)

    relative = {}
    for name, artifact_path in artifacts.items():
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            logger.warning('[IO] Manifest artifact "{}" missing at {}, omitting'.format(name, artifact_path))
            continue
        try:
            relative[name] = str(artifact_path.relative_to(base).as_posix())
        except ValueError:
            relative[name] = str(artifact_path.as_posix())

    manifest = {
        'schema_version': SCHEMA_VERSION,
        'sample': sample,
        'mapper': mapper,
        'batch': batch,
        'status': status,
        'timestamp': datetime.datetime.now().isoformat(),
        'config_hash': config_hash(config_paths),
        'tool_versions': tool_versions(),
        'artifacts': relative}

    # Anything a stage wants discoverable without opening the h5ad. Currently
    # the doublet model's parameters: knowing whether a sample's calls came from
    # a tuned or fixed model, and with which latent size, should not require
    # loading a multi-GB artifact to read .uns.
    if extra:
        manifest.update({k: v for k, v in extra.items() if k not in manifest})

    manifest = _json_safe(manifest)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info('[IO] Wrote manifest for sample "{}" to "{}"'.format(sample, path))
    return manifest
