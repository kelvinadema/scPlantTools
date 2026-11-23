#!/usr/bin/env python3
"""
Bridge from the annotation stage to marker identification.

The dependency runs one way: `markerid` knows nothing about annotation,
and annotation reaches it only through this module. Marker files can equally be
produced by hand or by running `python -m markerid.main` directly -- this
just removes the manual step when a target sample has no marker file yet.

`markerid` lives in its own conda env (envs/scMarkersV3.12.yml, which
carries hotspot and cosg), so it is invoked as a subprocess against that env's
prefix rather than imported -- the same arrangement CellBender and the doublet
worker use. With no env configured it is imported in-process, which works when
annotation itself runs under scMarkersV3.12.
"""

import logging
import os
import subprocess

from pathlib import Path

logger = logging.getLogger('AnnotationLogging')

ENV_VAR = 'MARKER_ENV'


def resolve_env(config_env=None):
    """Conda *prefix* for the marker environment, or None to import in-process."""
    return config_env or os.environ.get(ENV_VAR)


def marker_file(marker_dir, sample):
    """`{marker_dir}/{sample}.tsv`: one sample's marker TSV for a method."""
    return Path(marker_dir) / f'{sample}.tsv'


def ensure_marker_file(sample, marker_dir, adata_path, method='wilcoxon',
                       cluster_key=None, latent_key=None,
                       counts_layer='corrected_counts', n_genes=50,
                       env=None, force=False, repo_root=None):
    """
    Return the path to `{marker_dir}/{sample}.tsv`, generating it if absent.

    Raises if the file is missing and no `adata_path` is configured for the
    sample -- silently annotating without markers would produce empty OMGs and
    a confidently meaningless result.
    """
    out_path = marker_file(marker_dir, sample)
    if out_path.exists() and not force:
        logger.info('[ANNOTATION] Using existing marker file: %s', out_path)
        return out_path
    if not adata_path:
        raise FileNotFoundError(
            f'No marker file at {out_path} and no AnnData configured for '
            f'{sample!r}. Add it under markers.generate.samples in the '
            f'annotation config, or run markerid.main yourself.')

    env = resolve_env(env)
    logger.info('[ANNOTATION] Generating markers for %s (method=%s)', sample, method)
    if env:
        _run_subprocess(env, sample, marker_dir, adata_path, method, cluster_key,
                        latent_key, counts_layer, n_genes, repo_root)
    else:
        from markerid.markers import identify_markers
        identify_markers(adata_path=adata_path, sample=sample, method=method,
                         cluster_key=cluster_key, latent_key=latent_key,
                         counts_layer=counts_layer, out_dir=marker_dir,
                         n_genes=n_genes)

    if not out_path.exists():
        raise FileNotFoundError(
            f'Marker generation for {sample!r} reported success but {out_path} '
            f'does not exist')
    return out_path


def ensure_marker_files(sample, marker_root, methods, adata_path, cluster_key=None,
                        latent_key=None, counts_layer='corrected_counts', n_genes=50,
                        env=None, force=False, repo_root=None):
    """
    Run several marker methods for one sample, returning
    `({method -> path}, [failure strings])`.

    Each method writes `{marker_root}/{method}/{sample}.tsv`, which is the
    layout `markerid.main` already produces -- so the annotation loop
    can treat a method exactly like the single-method case it replaces.

    **One method failing must not lose the others**, which is why failures are
    returned rather than raised. The six methods are independent and cost
    minutes to hours each; a failure in `cfmf` is no reason to discard a
    finished `wilcoxon`, and raising here would throw away work the caller can
    still annotate. The caller is responsible for surfacing the failures and
    exiting non-zero -- same contract as the mapping stage's per-sample
    handling, where a bad sample is collected and the run still fails at the
    end. Only "no method produced anything" raises, since there is then nothing
    to carry on with.

    The cluster-dependent methods need `cluster_key`; the cluster-free ones
    need a real latent space in `latent_key` (see markers.py::_resolve_latent
    -- with no key it computes PCA on the normalised layer, which is the
    intended fallback, but a 2-D UMAP must never be passed here).
    """
    marker_root = Path(marker_root)
    produced, failures = {}, []
    for method in methods:
        try:
            produced[method] = ensure_marker_file(
                sample=sample, marker_dir=marker_root / method,
                adata_path=adata_path, method=method,
                cluster_key=cluster_key, latent_key=latent_key,
                counts_layer=counts_layer, n_genes=n_genes,
                env=env, force=force, repo_root=repo_root)
        except Exception as exc:                      # noqa: BLE001 - reported to caller
            logger.error('[ANNOTATION] Marker method %r failed for %s: %s',
                         method, sample, exc)
            failures.append(f'{sample}/{method}: {type(exc).__name__}: {exc}')

    if not produced:
        raise RuntimeError(
            f'Every marker method failed for {sample!r}; nothing to annotate:\n  '
            + '\n  '.join(failures))
    return produced, failures


def _run_subprocess(env, sample, marker_dir, adata_path, method, cluster_key,
                    latent_key, counts_layer, n_genes, repo_root):
    """Run the markerID stage in its own conda env, for one sample and method.

    `env` is the conda prefix; `repo_root` is the subprocess working directory.
    Raises CalledProcessError on a non-zero exit.
    """
    python = Path(env) / 'bin' / 'python'
    if not python.exists():
        raise FileNotFoundError(
            f'{python} not found. markers.env must be the conda *prefix* of the '
            f'marker environment (the directory containing bin/), not the binary.')

    # main.py writes to <outdir>/<method>/, so hand it the parent.
    out_root = Path(marker_dir).parent if Path(marker_dir).name == method \
        else Path(marker_dir)
    cmd = [str(python), '-m', 'markerid.main',
           '--anndata', str(adata_path), '--name', sample,
           '--method', method, '--countskey', counts_layer,
           '--ngenes', str(n_genes), '--outdir', str(out_root)]
    if cluster_key:
        cmd += ['--clusterkey', cluster_key]
    if latent_key:
        cmd += ['--latentkey', latent_key]

    # `python -m` and the logs/ directory are both cwd-relative.
    cwd = str(repo_root) if repo_root else None
    logger.info('[ANNOTATION] Running: %s', ' '.join(cmd))
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error('[ANNOTATION] Marker generation failed for %s\nSTDOUT:\n%s\nSTDERR:\n%s',
                     sample, result.stdout, result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)
