#!/usr/bin/env python3
"""Register an already-trained model as a scArches reference.

WHY THIS EXISTS. `Reference_model:` in an integration target is resolved through
the integration registry, not through a path -- orchestrator looks the name up
with `CacheUtils.find_entry` and reads `model_path`, `anndata_path` and
`model_class` off the entry. Every entry there is written by a stage that TRAINED
the model it describes (integrate.py, refmapping.py), so a model produced
somewhere else is unreachable as a reference no matter where it sits on disk.

The refine stage is exactly that case. `refine/refine.py` fits an SCVI and then a
SCANVI over a cohort and saves both under `results/refine/<target>/<run>/`, but it
passes `persist=False` precisely so it does not claim integration-target names it
does not own. Those two models are the best references the project has -- they
carry the refine label set -- and nothing could point a target at them.

This does not train, copy or validate a fit; it writes the one registry row that
makes an existing directory addressable by name, and is idempotent (`upsert_entry`),
so re-running after a refine re-fit just refreshes the timestamp.

`--reference-samples` matters more than it looks: refmapping splits reference from
query by `adata.obs['sample'].isin(...)`, so this must list every sample inside the
reference AnnData or the reference side silently comes back short.

Usage:
    python -m integration.register_reference \\
        --name       MtPartA_TipSZmock_refine_scanvi \\
        --model-dir  results/refine/MtPartA_TipSZmock_DRVI/combined/scanvi \\
        --anndata    results/refine/MtPartA_TipSZmock_DRVI/combined/MtPartA_TipSZmock_DRVI_combined.h5ad \\
        --model-class SCANVI \\
        --reference-samples p1 p2 p3 rep1 rep2 rep3 rep4 BDTip2mm BDTip10mm pMock BDMock SMu008

    python -m integration.register_reference --list
"""

import argparse
import datetime
import sys

from pathlib import Path

from shared.utils import find_root_dir
from .integrationutils import CacheUtils, integration_tool_versions, resolve_model_class


def _resolve(path: str) -> Path:
    """Absolute path, resolving a relative one against the repo root rather than cwd."""
    p = Path(path)
    return p if p.is_absolute() else (find_root_dir() / p)


def register(name: str, model_dir: str, anndata: str, model_class: str,
             reference_samples: list, hvg: str = None, force: bool = False) -> dict:
    """Write (or refresh) the registry entry that makes `model_dir` usable as a reference.

    Args:
        name (str): Registry key, i.e. what a target's `Reference_model:` names.
        model_dir (str): Directory holding `model.pt`.
        anndata (str): The AnnData the model was registered against at fit time.
        model_class (str): A key of integrationutils.MODEL_CLASSES.
        reference_samples (list[str]): Every sample inside `anndata`.
        hvg (str, optional): The reference's `_HVG.tsv`, for a gene-restricted fit.
        force (bool): Overwrite an entry that already points somewhere else.

    Returns:
        dict: The entry written.

    Raises:
        ValueError: On an unknown model_class, a missing file, or a conflicting entry.
    """
    # Fails on a typo before anything is written, and -- more usefully -- before a
    # GPU job starts and dies at load time three hours into a queue.
    resolve_model_class(model_class)

    model_path = _resolve(model_dir)
    adata_path = _resolve(anndata)
    if not (model_path / "model.pt").is_file():
        raise ValueError(f"No model.pt under {model_path}")
    if not adata_path.is_file():
        raise ValueError(f"Reference AnnData not found: {adata_path}")
    if hvg is not None and not _resolve(hvg).is_file():
        raise ValueError(f"HVG file not found: {_resolve(hvg)}")
    if not reference_samples:
        raise ValueError(
            "--reference-samples is required: refmapping splits reference from query by "
            "obs['sample'].isin(...), so an incomplete list silently shrinks the reference.")

    registry = CacheUtils()
    existing = registry.find_entry(name)
    if existing and not force:
        # A name already pointing at a DIFFERENT model is the dangerous case: a
        # target would keep resolving to whatever was there first, and nothing
        # downstream re-checks. Refreshing the same path is fine and stays silent.
        if str(existing.get("model_path")) != str(model_path):
            raise ValueError(
                "'{}' is already registered against {}. Pass --force to repoint it, or "
                "choose another name -- silently repointing a reference would change what "
                "every target using it maps onto.".format(name, existing.get("model_path")))

    entry = {
        'model_path': str(model_path),
        'anndata_path': str(adata_path),
        'model_class': model_class,
        'reference_samples': list(reference_samples),
        'samples': list(reference_samples),
        # Recorded so a reader can tell a hand-registered reference from one a
        # training stage wrote, without diffing paths.
        'registered_by': 'integration.register_reference',
        'tool_versions': integration_tool_versions(),
        'timestamp': datetime.datetime.now().isoformat()}
    if hvg is not None:
        entry['hvg_path'] = str(_resolve(hvg))

    registry.upsert_entry(name, entry)
    return entry


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Register an already-trained model as a scArches reference.")
    parser.add_argument('--name', help="Registry key a target's Reference_model: will name.")
    parser.add_argument('--model-dir', help="Directory holding model.pt.")
    parser.add_argument('--anndata', help="AnnData the model was registered against.")
    parser.add_argument('--model-class', default='SCVI',
                        help="A key of integrationutils.MODEL_CLASSES (default: SCVI).")
    parser.add_argument('--reference-samples', nargs='+', default=[],
                        help="Every sample inside the reference AnnData.")
    parser.add_argument('--hvg', default=None,
                        help="The reference's _HVG.tsv, if it was fit on a gene subset.")
    parser.add_argument('--force', action='store_true',
                        help="Repoint a name that already resolves to another model.")
    parser.add_argument('--list', action='store_true',
                        help="Print the registry and exit.")
    args = parser.parse_args(argv)

    if args.list:
        for e in CacheUtils().load_registry():
            print("{:<46} {:<20} {}".format(
                str(e.get('dataset_id')), str(e.get('model_class')), str(e.get('model_path'))))
        return 0

    missing = [f for f in ('name', 'model_dir', 'anndata') if not getattr(args, f.replace('-', '_'))]
    if missing:
        parser.error("--{} is required".format(', --'.join(m.replace('_', '-') for m in missing)))

    entry = register(args.name, args.model_dir, args.anndata, args.model_class,
                     args.reference_samples, hvg=args.hvg, force=args.force)
    print("Registered '{}' -> {} ({}, {} reference samples)".format(
        args.name, entry['model_path'], entry['model_class'], len(entry['reference_samples'])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
