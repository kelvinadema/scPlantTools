#!/usr/bin/env python3

"""
Manifest-based discovery of mapping-stage sample outputs.

Single source of truth for resolving a mapped sample's filtered h5ad, used by
`orchestrator.py::_concatenate_adata` regardless of whether the run was reached
via --target or an ad hoc --sample/--reference invocation. Reads the per-sample
`manifest.json` mapping writes only on success (see shared/io.py::write_manifest)
instead of hand-reconstructing paths, so a failed or partial mapping run raises
clearly rather than silently loading whatever happens to be on disk.
"""

import json

from pathlib import Path


class SampleNotMappedError(RuntimeError):
    """Raised when a sample has no mapping-stage manifest to read."""
    pass


def load_sample_manifest(sample: str, input_dir) -> dict:
    """Read one sample's mapping manifest.

    Args:
        sample (str): Sample name.
        input_dir (Path): Mapping results root.

    Returns:
        dict: The manifest.

    Raises:
        SampleNotMappedError: If no manifest exists.
    """
    manifest_path = Path(input_dir) / sample / 'manifest.json'
    if not manifest_path.exists():
        raise SampleNotMappedError(
            f"No manifest.json for sample '{sample}' at {manifest_path}; "
            f"mapping has not run (or not completed) for this sample.")
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest.get('status') != 'complete':
        raise SampleNotMappedError(
            f"Sample '{sample}' manifest at {manifest_path} has status="
            f"'{manifest.get('status')}', not 'complete'; mapping did not finish successfully.")
    return manifest


DEFAULT_ARTIFACT_KEY = 'filtered_h5ad'
# Written by mapping.refilter after a human has reviewed the QC report
# and adjusted the filtering by hand.
MANUAL_ARTIFACT_KEY = 'filtered_manual_h5ad'


def resolve_sample_artifact(sample: str, input_dir, sample_metadata: dict, manifest: dict = None,
                            artifact_key: str = DEFAULT_ARTIFACT_KEY,
                            fallback: bool = False) -> Path:
    """
    Locate one sample's input artifact by manifest key.

    `artifact_key` selects *which* artifact: the automatically-filtered
    'filtered_h5ad' by default, or 'filtered_manual_h5ad' for a sample whose
    filtering a human adjusted in the interactive QC report.

    `fallback` is off by default on purpose. Silently falling back to the
    automatic artifact for samples that have no manual one would produce a run
    quietly mixing manually- and auto-filtered samples -- it would succeed, and
    the difference would never appear anywhere. Callers that genuinely want
    "manual where available" have to say so.
    """
    manifest = manifest or load_sample_manifest(sample, input_dir)
    artifacts = manifest.get('artifacts') or {}

    rel_path = artifacts.get(artifact_key)
    if not rel_path and fallback and artifact_key != DEFAULT_ARTIFACT_KEY:
        rel_path = artifacts.get(DEFAULT_ARTIFACT_KEY)

    if not rel_path:
        if artifact_key != DEFAULT_ARTIFACT_KEY:
            raise SampleNotMappedError(
                f"Sample '{sample}' manifest has no '{artifact_key}' artifact recorded. "
                f"Available: {sorted(artifacts) or 'none'}. Run "
                f"`python -m mapping.refilter --sample {sample}` to produce a manually "
                f"filtered artifact, or pass --artifact-fallback to use the automatic one.")
        raise SampleNotMappedError(
            f"Sample '{sample}' manifest has no '{DEFAULT_ARTIFACT_KEY}' artifact recorded "
            f"(mapping may have failed after the manifest was written, or QC did not complete).")

    sample_dir = Path(input_dir) / sample
    path = (sample_dir / rel_path).resolve()
    if not path.exists():
        raise SampleNotMappedError(
            f"Manifest for sample '{sample}' points at {path}, which does not exist on disk "
            f"(results tree may have moved without the manifest being regenerated).")
    return path


def check_artifact_availability(samples, input_dir, artifact_key: str, fallback: bool = False) -> dict:
    """
    Pre-flight every sample's artifact key, reporting all failures at once.

    Called before any GPU work so a 40-sample run needs one round trip to fix
    rather than failing on the first sample missing a manual filter. Returns
    {sample: key_actually_used} for the provenance record -- with fallback on,
    that map is the only place a mixed run is visible, so it is worth keeping.
    """
    resolved, missing = {}, []
    for sample in samples:
        try:
            artifacts = (load_sample_manifest(sample, input_dir).get('artifacts') or {})
        except SampleNotMappedError as e:
            missing.append('{}: {}'.format(sample, e))
            continue
        if artifacts.get(artifact_key):
            resolved[sample] = artifact_key
        elif fallback and artifacts.get(DEFAULT_ARTIFACT_KEY):
            resolved[sample] = DEFAULT_ARTIFACT_KEY
        else:
            missing.append('{} (has: {})'.format(sample, ', '.join(sorted(artifacts)) or 'nothing'))

    if missing:
        raise SampleNotMappedError(
            "No '{key}' artifact for {n} sample(s):\n  {rows}\n\n"
            "Produce them with `python -m mapping.refilter --sample <name>`, or pass "
            "--artifact-fallback to use the automatically-filtered artifact where no manual one "
            "exists. Fallback is not the default because it would silently mix manually- and "
            "automatically-filtered samples in one integration run.".format(
                key=artifact_key, n=len(missing), rows='\n  '.join(missing)))
    return resolved


def resolve_batch(sample: str, manifest: dict, sample_metadata: dict) -> str:
    """The batch a sample belongs to, from its manifest or the sample sheet.

    Args:
        sample (str): Sample name.
        manifest (dict): The sample's manifest.
        sample_metadata (dict): Resolved sample sheet.

    Returns:
        str: The batch label.
    """
    return manifest.get('batch') or (sample_metadata.get(sample, {}) or {}).get('batch', sample)
