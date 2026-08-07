#!/usr/bin/env python
"""Rewrite a scArches target's .obs index to composite ``sample.barcode``.

**This is a backfill, not a new convention.** ``orchestrator.py::_concatenate_adata``
already sets ``adata.obs_names = [f"{s}.{bc}" ...]``, but that line used to be
nested inside an ``if "sample" not in adata.obs`` that was always False, so it
never ran -- see the comment block at ``orchestrator.py:249-261``. The objects
under ``results/integration/scArches/`` were written before that was un-nested
and so carry the bare 10x barcode as their index. A re-run of the integration
stage today would produce exactly what this script writes.

Why it matters: the 10x whitelist is finite and shared, so independently run
samples collide by coincidence. ``MtInduced`` has 299,913 cells under 298,825
distinct barcodes. A non-unique index makes ``pandas.reindex`` refuse outright
("cannot reindex on an axis with duplicate labels") and, worse, makes any
positional join silently wrong. ``sample.barcode`` is unique in every target.

The edit is in place and surgical: only the ``/obs`` index dataset is replaced
(plus an optional ``barcode_raw`` column and the ``column-order`` attribute).
Row order, ``X``, ``layers``, ``obsm``, ``varm`` and ``obsp`` are never touched,
so this costs seconds rather than a 69 GB rewrite.

It is idempotent -- an index already carrying a valid ``sample.`` prefix is left
alone -- and it refuses to write if the result would not be unique.

Usage (from the repo root)::

    python integration/reindex_composite_barcodes.py --dry-run
    python integration/reindex_composite_barcodes.py --commit
    python integration/reindex_composite_barcodes.py --commit --target MtInduced_10x
"""

from __future__ import annotations

import argparse
from pathlib import Path

from shared.utils import find_root_dir

import h5py
import numpy as np


ROOT = find_root_dir()

TARGETS = {
    'MtInduced': ROOT / 'results/integration/scArches/MtInduced/MtInduced.h5ad',
    'MtInduced_10x': ROOT / 'results/integration/scArches/MtInduced_10x/MtInduced_10x.h5ad',
    'MtInduced_bd': ROOT / 'results/integration/scArches/MtInduced_bd/MtInduced_bd.h5ad',
}

RAW_COL = 'barcode_raw'


def _dec(x):
    """Decode a possibly-bytes value to str."""
    return x.decode() if isinstance(x, bytes) else x


def read_index(obs: h5py.Group) -> tuple[str, np.ndarray]:
    """The obs_names index from an open HDF5 obs group.

    Args:
        obs: The h5py obs group.

    Returns:
        list[str]: Barcodes.
    """
    key = _dec(obs.attrs['_index'])
    g = obs[key]
    if isinstance(g, h5py.Group):  # nullable-string-array
        vals = np.array([_dec(x) for x in g['values'][:]], dtype=object)
    else:
        vals = np.array([_dec(x) for x in g[:]], dtype=object)
    return key, vals.astype(str)


def read_categorical(obs: h5py.Group, key: str) -> np.ndarray:
    """Read one categorical obs column back to plain strings.

    Args:
        obs: The h5py obs group.
        key (str): Column name.

    Returns:
        list[str]: Values.
    """
    g = obs[key]
    cats = np.array([_dec(x) for x in g['categories'][:]], dtype=object)
    return cats[g['codes'][:]].astype(str)


def already_composite(index: np.ndarray, samples: np.ndarray) -> bool:
    """True if every value already carries its own sample as a ``sample.`` prefix."""
    prefixes = np.char.add(samples, '.')
    return bool(np.char.startswith(index.astype(str), prefixes).all())


def _write_string_index(obs: h5py.Group, key: str, values: np.ndarray) -> None:
    """Replace the index with a plain string-array, anndata's normal encoding."""
    del obs[key]
    d = obs.create_dataset(key, data=np.array(values, dtype=object),
                           dtype=h5py.string_dtype(encoding='utf-8'))
    d.attrs['encoding-type'] = 'string-array'
    d.attrs['encoding-version'] = '0.2.0'


def _write_raw_column(obs: h5py.Group, values: np.ndarray) -> None:
    """Write an obs column as a plain string array.

    Args:
        obs: The h5py obs group.
        values (list[str]): Values to write.
    """
    if RAW_COL in obs:
        del obs[RAW_COL]
    d = obs.create_dataset(RAW_COL, data=np.array(values, dtype=object),
                           dtype=h5py.string_dtype(encoding='utf-8'))
    d.attrs['encoding-type'] = 'string-array'
    d.attrs['encoding-version'] = '0.2.0'
    order = [_dec(c) for c in obs.attrs['column-order']]
    if RAW_COL not in order:
        obs.attrs['column-order'] = np.array(order + [RAW_COL], dtype=object)


def obs_aligned_frames(f: h5py.File) -> list[str]:
    """Paths of every obs-aligned DataFrame that carries its own index copy.

    **This is the part that is easy to miss and fatal to skip.** `.obsm` may
    hold a DataFrame -- scvi-tools writes `_scvi_extra_continuous_covs` there
    for a model trained with continuous covariates -- and h5ad stores that with
    its *own* index dataset, duplicating `obs`'s. Rewriting only `obs`'s index
    leaves the two disagreeing, and AnnData validates them on load:

        ValueError: value.index does not match parent's obs names

    which makes the file unreadable by `read_h5ad` from that point on, while
    every h5py-level inspection still looks perfectly healthy.
    """
    found = []
    for container in ('obsm', 'obs'):
        if container not in f:
            continue
        for key in f[container]:
            g = f[container][key]
            if isinstance(g, h5py.Group) and \
                    _dec(g.attrs.get('encoding-type', '')) == 'dataframe':
                found.append(f'{container}/{key}')
    return found


def _rewrite_frame_index(g: h5py.Group, composite: np.ndarray) -> str:
    """Rewrite a DataFrame group's index as a plain ``string-array``.

    The encoding has to match `obs`'s index, not merely the values. A
    ``nullable-string-array`` reads back as pandas ``StringDtype`` while a
    ``string-array`` reads back as ``object``, and `Index.equals` is False
    across that difference even when every value is identical -- so AnnData
    still rejects the file with the same "does not match parent's obs names".
    `_write_string_index` writes `obs` as a plain string-array, so these must
    be plain string-arrays too.
    """
    key = _dec(g.attrs['_index'])
    del g[key]  # drops the nullable mask/values group wholesale, if that is what it was
    d = g.create_dataset(key, data=np.array(composite, dtype=object),
                         dtype=h5py.string_dtype(encoding='utf-8'))
    d.attrs['encoding-type'] = 'string-array'
    d.attrs['encoding-version'] = '0.2.0'
    return key


def process(name: str, path: Path, commit: bool, keep_raw: bool) -> None:
    """Rewrite one h5ad's barcodes to `{sample}.{barcode}` form.

    Args:
        name (str): Target name, for the log lines.
        path (Path): The h5ad.
        commit (bool): Write the change; otherwise report only.
        keep_raw (bool): Keep the original barcodes in a new column.

    Returns:
        dict: What was found and what was written.
    """
    mode = 'r+' if commit else 'r'
    with h5py.File(path, mode) as f:
        obs = f['obs']
        key, index = read_index(obs)
        if 'sample' not in obs:
            print(f"{name}: no 'sample' column -- skipped")
            return
        samples = read_categorical(obs, 'sample')

        print(f"{name}: {len(index)} cells, index dataset '{key}'")
        obs_done = already_composite(index, samples)

        # The obs index may already be composite while an obs-aligned frame is
        # still bare -- that is exactly the broken state a partial earlier run
        # leaves behind, and it must still be repairable.
        # A frame is stale on either count: wrong values, or the right values
        # under an encoding that reads back as a different pandas dtype than
        # obs's index (see _rewrite_frame_index).
        stale = []
        for frame in obs_aligned_frames(f):
            fi = f[frame]
            fnode = fi[_dec(fi.attrs['_index'])]
            nullable = isinstance(fnode, h5py.Group)
            vals = (np.array([_dec(x) for x in fnode['values'][:]]) if nullable
                    else np.array([_dec(x) for x in fnode[:]])).astype(str)
            if not already_composite(vals, samples) or nullable:
                stale.append(frame)

        if obs_done and not stale:
            print("  already composite -- nothing to do\n")
            return

        base = index if not obs_done else np.char.partition(index, '.')[:, 2]
        composite = np.char.add(np.char.add(samples, '.'), base)
        n_dup_after = len(composite) - len(set(composite.tolist()))
        print(f"  obs index:                    "
              f"{'already composite' if obs_done else 'bare -> rewriting'}")
        print(f"  bare barcode duplicates:      {len(base) - len(set(base.tolist()))}")
        print(f"  sample.barcode duplicates:    {n_dup_after}")
        print(f"  obs-aligned frames needing repair: {stale or 'none'}")
        print(f"  example: {base[0]}  ->  {composite[0]}")

        if n_dup_after:
            print("  REFUSING: composite key is still not unique\n")
            return
        if not commit:
            print("  (dry run -- not written)\n")
            return

        for frame in stale:
            fkey = _rewrite_frame_index(f[frame], composite)
            print(f"  -> repaired index of {frame} (dataset '{fkey}')")

        if obs_done:
            print()
            return

        if keep_raw:
            _write_raw_column(obs, base)
        _write_string_index(obs, key, composite)
        print(f"  -> rewrote index in place"
              f"{f', kept bare barcode as {RAW_COL}' if keep_raw else ''}\n")


def main() -> None:
    """CLI entry point: reindex composite barcodes in the named h5ads.

    Returns:
        int: Process exit status.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--commit', action='store_true')
    ap.add_argument('--target', action='append', choices=sorted(TARGETS))
    ap.add_argument('--no-keep-raw', action='store_true',
                    help=f'do not preserve the bare barcode as {RAW_COL} '
                         f'(matches what a fresh integration run produces)')
    ap.add_argument('--path', type=Path,
                    help='operate on an arbitrary h5ad instead of the known targets')
    args = ap.parse_args()

    if args.path:
        process(args.path.stem, args.path, args.commit, not args.no_keep_raw)
        return
    for name in (args.target or sorted(TARGETS)):
        path = TARGETS[name]
        if not path.exists():
            print(f"{name}: MISSING {path} -- skipped")
            continue
        process(name, path, args.commit, not args.no_keep_raw)


if __name__ == '__main__':
    main()
