"""
Read back what a run actually produced.

WHY THIS EXISTS. The pipeline already writes rich state: per-step `.done` flags,
per-sample `manifest.json` carrying schema version, config hash, tool versions
and the artifact map, and the integration registry keyed by target. Until this
module there was no way to *read* any of it -- answering "what stage is SMu006
at?" or "which targets are trained?" meant `ls`-ing around `results/`.

Nothing here computes anything. It reads files the stages already write, which
is why it needs no pipeline dependencies and runs in any environment.

COVERAGE IS UNEVEN, AND SAYS SO. Mapping routes every terminal write through
`shared/io.py`, so its state is complete: step flags plus a manifest. Integration
and annotation still do their IO ad hoc (see CLAUDE.md), so this reports what it
can see on disk -- a target directory and a registry entry -- and marks the rest
`unknown` rather than implying absence. A missing manifest means "this stage
does not write one yet", not "this did not run", and the two must not look alike.
"""

from __future__ import annotations

import json
from pathlib import Path


# Order matters: this is the order the mapping stage runs them, so a row reads
# left to right as progress and the first gap is where a sample stopped.
MAPPING_STEPS = ['mapping', 'preprocessing', 'mappingstats', 'qcmetrics',
                 'qcfilter', 'qcplot', 'doublets', 'report']

DONE = 'done'
MISSING = '-'


def _read_json(path: Path):
    """Read a JSON file, returning {} when it is absent or unreadable.

    Args:
        path (Path): The file.

    Returns:
        dict: The parsed contents.
    """
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def mapping_status(output_dir) -> dict:
    """
    {sample: {'steps': {step: bool}, 'manifest': {...}|None, 'mapper': str|None}}

    Step flags are searched one level below the sample directory because they are
    written per mapper (`{sample}/{mapper}/{step}.done`), and the mapper in use
    is a per-sample config choice rather than a fixed path segment.
    """
    root = Path(output_dir)
    out = {}
    if not root.is_dir():
        return out
    for sample_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        manifest = _read_json(sample_dir / 'manifest.json')
        flags = {s: False for s in MAPPING_STEPS}
        mapper = manifest.get('mapper') if manifest else None
        for done in sample_dir.rglob('*.done'):
            step = done.stem
            if step in flags:
                flags[step] = True
            if mapper is None:
                # {sample}/{mapper}/{step}.done -- recover the mapper from the path
                rel = done.relative_to(sample_dir).parts
                if len(rel) > 1:
                    mapper = rel[0]
        out[sample_dir.name] = {'steps': flags, 'manifest': manifest, 'mapper': mapper}
    return out


def integration_status(output_dir) -> dict:
    """
    {target: {'model_class': str, 'has_h5ad': bool, 'has_model': bool,
              'registry': {...}|None}}

    Built from two independent sources on purpose. The on-disk directory says
    what exists; the registry says what the pipeline believes exists. When they
    disagree that is worth seeing -- a registry entry whose `model_path` is gone
    is exactly the state that makes a rerun retrain, and an h5ad with no entry is
    a disposable (`persist: False`) run or a stale directory.
    """
    root = Path(output_dir)
    out = {}
    if not root.is_dir():
        return out

    registry = _read_json(root / 'scvi_tune_registry.json') or []
    if isinstance(registry, dict):
        registry = registry.get('entries', []) or []
    by_target = {}
    for entry in registry:
        if isinstance(entry, dict):
            # `dataset_id` is the key CacheUtils writes and looks up by
            # (integrationutils.py::find_entry); the others are tolerated in case an
            # older registry is still around.
            name = (entry.get('dataset_id') or entry.get('target')
                    or entry.get('name') or entry.get('integration_name'))
            if name:
                by_target[name] = entry

    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for target_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            name = target_dir.name
            out[name] = {
                'model_class': model_dir.name,
                'has_h5ad': (target_dir / '{}.h5ad'.format(name)).exists()
                            or any(target_dir.glob('*.h5ad')),
                'has_model': (target_dir / 'model.pt').exists()
                             or any(target_dir.rglob('model.pt')),
                'registry': by_target.get(name),
            }
    for name, entry in by_target.items():
        if name not in out:
            # Registered but no directory -- the case that silently makes a
            # rerun retrain, so it must be visible rather than omitted.
            out[name] = {'model_class': entry.get('model_class', '?'),
                         'has_h5ad': False, 'has_model': False, 'registry': entry}
    return out


def simple_stage_status(root) -> dict:
    """
    {name: {'files': int, 'h5ad': int}} for stages that write a directory per
    target but no manifest -- annotation, refine, markers.

    Deliberately shallow. Reporting a file count is honest about the fact that
    these stages do not declare completion; inventing a done/not-done verdict
    from directory contents would be a guess wearing a status column's clothes.
    """
    root = Path(root)
    out = {}
    if not root.is_dir():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        files = [f for f in d.rglob('*') if f.is_file()]
        out[d.name] = {'files': len(files),
                       'h5ad': sum(1 for f in files if f.suffix == '.h5ad')}
    return out
