#!/usr/bin/env python
"""Write an annotation-sized copy of a scArches target.

Stage 1 annotation reads these objects **fully into memory** -- `sc.read_h5ad`,
not backed, because geneset scoring needs `.layers` (annotation/
main.py:229-231). MtInduced carries nine matrices totalling 60.7 GB resident,
of which annotation and marker identification read exactly one:

    X                   9.02 GB   kept (identical nnz to corrected_counts)
    corrected_counts    9.02 GB   kept -- the only layer anything reads
    ambiguous           8.90 GB   dropped
    counts              9.94 GB   dropped
    raw_counts          9.94 GB   dropped
    rspliced            8.00 GB   dropped
    runspliced          3.56 GB   dropped
    spliced             1.59 GB   dropped
    unspliced           0.69 GB   dropped
    obsp                3.56 GB   dropped -- neighbours graph; every
                                  cluster-free method builds its own from
                                  `latent_key` (markers.py::_resolve_latent)

That is 64 GB -> 18 GB, and it is read once per marker-method task. Under the
parallel layout (batch_scripts/run_markerid_induced.sh) twelve tasks load one
of these at once, so the saving is both memory and a great deal of concurrent
Lustre traffic.

`corrected_counts` is what `markers.generate.counts_layer` names, what
`ensure_score_layer` builds `data_norm` from, and what the review page plots.
`X` is kept because scanpy's gene filter operates on it and it costs nothing
extra -- it is the same matrix.

Copied at the **HDF5 group level**, not through AnnData, so this needs no more
memory than a file copy and preserves each dataset's compression and encoding
attributes exactly.

Usage (from the repo root)::

    python integration/slim_for_annotation.py --dry-run
    python integration/slim_for_annotation.py --write
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from shared.utils import find_root_dir

import h5py


ROOT = find_root_dir()
SCARCHES = ROOT / 'results/integration/scArches'
TARGETS = ['MtInduced_10x', 'MtInduced_bd', 'MtInduced']

# Top-level groups copied wholesale.
KEEP_WHOLE = ['obs', 'var', 'obsm', 'uns']
# Layers kept. Everything else in /layers is dropped.
KEEP_LAYERS = ['corrected_counts']
# Kept as empty groups so AnnData still sees a well-formed file.
EMPTY = ['varm', 'varp', 'obsp']


def src_path(name: str) -> Path:
    """Path to a target's full integration h5ad."""
    return SCARCHES / name / f'{name}.h5ad'


def dst_path(name: str) -> Path:
    """Path to a target's slimmed h5ad."""
    return SCARCHES / name / f'{name}_annot.h5ad'


def _gb(n: int) -> float:
    """Bytes as a gigabyte figure."""
    return n / 1024 ** 3


def matrix_bytes(g) -> int:
    """Resident size of a sparse (or dense) matrix group/dataset."""
    if isinstance(g, h5py.Group) and 'data' in g:
        return (g['data'].shape[0] * g['data'].dtype.itemsize
                + g['indices'].shape[0] * g['indices'].dtype.itemsize
                + g['indptr'].shape[0] * g['indptr'].dtype.itemsize)
    if isinstance(g, h5py.Dataset):
        return g.size * g.dtype.itemsize
    return 0


def report(name: str) -> None:
    """Print what slimming one target would save."""
    src = src_path(name)
    with h5py.File(src, 'r') as f:
        kept = matrix_bytes(f['X'])
        dropped = 0
        rows = [('X', matrix_bytes(f['X']), 'keep')]
        for k in f['layers']:
            b = matrix_bytes(f['layers'][k])
            keep = k in KEEP_LAYERS
            kept += b if keep else 0
            dropped += 0 if keep else b
            rows.append((f'layers/{k}', b, 'keep' if keep else 'drop'))
        for k in EMPTY:
            if k in f:
                b = sum(matrix_bytes(f[k][x]) for x in f[k])
                dropped += b
                if b:
                    rows.append((k, b, 'drop'))
        for k in KEEP_WHOLE:
            if k in f:
                b = 0

                def _v(_n, o):
                    nonlocal b
                    if isinstance(o, h5py.Dataset):
                        b += o.size * o.dtype.itemsize
                f[k].visititems(_v)
                kept += b
                rows.append((k, b, 'keep'))
    print(f"\n=== {name}")
    for label, b, verdict in rows:
        print(f"   {verdict:5s} {label:22s} {_gb(b):7.2f} GB")
    print(f"   {'':5s} {'RESIDENT after':22s} {_gb(kept):7.2f} GB "
          f"(was {_gb(kept + dropped):.2f} GB)")


def slim(name: str) -> None:
    """Write a target's h5ad with only what the annotation stage reads."""
    src, dst = src_path(name), dst_path(name)
    tmp = dst.with_suffix('.h5ad.partial')
    if tmp.exists():
        tmp.unlink()
    with h5py.File(src, 'r') as s, h5py.File(tmp, 'w') as d:
        for k, v in s.attrs.items():
            d.attrs[k] = v
        s.copy('X', d, name='X')
        for grp in KEEP_WHOLE:
            if grp in s:
                s.copy(grp, d, name=grp)
        layers = d.create_group('layers')
        for k, v in s['layers'].attrs.items():
            layers.attrs[k] = v
        for k in KEEP_LAYERS:
            if k not in s['layers']:
                raise SystemExit(f'{name}: layers/{k} not in source')
            s.copy(f'layers/{k}', layers, name=k)
        for grp in EMPTY:
            g = d.create_group(grp)
            if grp in s:
                for k, v in s[grp].attrs.items():
                    g.attrs[k] = v
    tmp.replace(dst)
    print(f"   -> {dst}  ({_gb(dst.stat().st_size):.2f} GB on disk, "
          f"source {_gb(src.stat().st_size):.2f} GB)")


def main() -> None:
    """CLI entry point: report on or slim the named targets."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--write', action='store_true')
    ap.add_argument('--target', action='append', choices=TARGETS)
    args = ap.parse_args()

    names = args.target or TARGETS
    free = shutil.disk_usage(SCARCHES).free
    print(f"free space at {SCARCHES}: {_gb(free):.0f} GB")
    for name in names:
        if not src_path(name).exists():
            print(f"{name}: MISSING {src_path(name)} -- skipped")
            continue
        report(name)
        if args.write:
            slim(name)


if __name__ == '__main__':
    main()
