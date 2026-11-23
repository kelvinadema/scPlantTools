#!/usr/bin/env python
"""Inject stage-3 refine annotations into scArches targets as new .obs columns.

The refine runs (``results/refine/{MtTip,MtSZ}/combined/*_labels.tsv``) annotate
a subset of the cells that the scArches targets hold. This carries those calls
across and leaves every other cell explicitly ``not_annotated``.

**The join key is ``sample`` + ``'.'`` + barcode, never the barcode alone.**
The scArches objects index ``.obs`` on the bare barcode and that index is *not
unique* -- 1088 barcodes collide across samples in MtInduced. The refine objects
are already indexed on the composite ``p1.CATCGTCTCCGGTTCT`` form, which is what
makes the two joinable at all. Joining on ``.obs_names`` would silently
mis-assign the colliding cells rather than fail.

Writes are **in place** with h5py: new datasets are added under ``/obs`` and the
``column-order`` attribute is extended. Nothing else in the file is touched, so
this costs seconds rather than a 69 GB rewrite. Re-running replaces the columns
it owns, so it is idempotent (HDF5 does not reclaim the space of a deleted
dataset, but these columns are a few MB).

Two label columns are emitted, and the difference matters for scANVI:

``refine_annotation``
    Faithful copy of refine's ``annotation``. Keeps refine's own ``Unassigned``
    (scANVI looked at the cell and declined) distinct from ``not_annotated``
    (the cell was never in the refine run).

``refine_scanvi_labels``
    Training-ready. Collapses ``Unassigned`` *and* anything flagged
    ``low_confidence`` into ``not_annotated``, so a downstream
    ``SCANVI.from_scvi_model(..., unlabeled_category='not_annotated')`` is not
    taught to reproduce refine's own hedges as if they were a cell type.

**Sources are selected, not unioned.** ``--source`` defaults to the original
pair (``MtTip`` + ``MtSZ``) because those two refine runs cover disjoint cells.
``MtPartA_TipSZmock_DRVI`` overlaps BOTH of them -- it shares the tip libraries
with ``MtTip`` and ``pMock`` with ``MtSZ`` -- so unioning it in trips the
uniqueness guard in ``load_labels``. Name it explicitly instead.

Usage (from the repo root)::

    python integration/inject_refine_labels.py --dry-run
    python integration/inject_refine_labels.py --commit
    python integration/inject_refine_labels.py --commit --target MtInduced_10x
    python integration/inject_refine_labels.py --commit \\
        --target MtInduced_TipSZmock --source MtPartA_TipSZmock_DRVI
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from shared.utils import find_root_dir

import h5py
import numpy as np
import pandas as pd

ROOT = find_root_dir()

NOT_ANNOTATED = 'not_annotated'

# Refine runs to read, and the columns carried over from each.
SOURCES = {
    'MtTip': ROOT / 'results/refine/MtTip/combined/MtTip_combined_labels.tsv',
    'MtSZ': ROOT / 'results/refine/MtSZ/combined/MtSZ_combined_labels.tsv',
    # Chapter 5 Part B's reference, 2026-08-12. 144,502 cells: the nine tip
    # libraries plus pMock/BDMock/SMu008. It OVERLAPS both entries above, so it
    # is never unioned with them -- see --source in the module docstring.
    'MtPartA_TipSZmock_DRVI': ROOT / ('results/refine/MtPartA_TipSZmock_DRVI/combined/'
                                      'MtPartA_TipSZmock_DRVI_combined_labels.tsv'),
}

#: What `--source` uses when not given. The original pair, whose cells are
#: disjoint; adding a third here would break every existing invocation.
DEFAULT_SOURCES = ('MtTip', 'MtSZ')

TARGETS = {
    'MtInduced': ROOT / 'results/integration/scArches/MtInduced/MtInduced.h5ad',
    'MtInduced_10x': ROOT / 'results/integration/scArches/MtInduced_10x/MtInduced_10x.h5ad',
    'MtInduced_bd': ROOT / 'results/integration/scArches/MtInduced_bd/MtInduced_bd.h5ad',
    'MtInduced_TipSZmock': ROOT / ('results/integration/scArches/MtInduced_TipSZmock/'
                                   'MtInduced_TipSZmock.h5ad'),
}

# (source column, output column, kind). 'cat' -> categorical, 'num' -> float64.
CARRIED = [
    ('annotation', 'refine_annotation', 'cat'),
    ('prediction', 'refine_prediction', 'cat'),
    ('seed', 'refine_seed', 'cat'),
    ('flag', 'refine_flag', 'cat'),
    ('confidence', 'refine_confidence', 'num'),
    ('margin', 'refine_margin', 'num'),
    ('novelty_ratio', 'refine_novelty_ratio', 'num'),
    ('seed_support', 'refine_seed_support', 'num'),
]
OWNED = ([out for _, out, _ in CARRIED]
         + ['refine_scanvi_labels', 'refine_source'])


def _dec(x):
    """Decode a possibly-bytes value to str."""
    return x.decode() if isinstance(x, bytes) else x


def _read_obs_col(obs: h5py.Group, key: str) -> np.ndarray:
    """Read one .obs column out of an h5ad, whatever anndata encoded it as."""
    g = obs[key]
    if isinstance(g, h5py.Group):
        enc = _dec(g.attrs.get('encoding-type', ''))
        if enc == 'categorical':
            cats = np.array([_dec(x) for x in g['categories'][:]], dtype=object)
            codes = g['codes'][:]
            return np.where(codes >= 0, cats[np.clip(codes, 0, None)], None)
        if 'values' in g:  # nullable-string-array / nullable-integer
            return np.array([_dec(x) for x in g['values'][:]], dtype=object)
    return g[:]


def target_keys(path: Path) -> pd.Index:
    """Composite ``sample.barcode`` key for every cell in a scArches target.

    Decided on the *values*, never on the index's name. The name stays
    ``barcode`` after ``reindex_composite_barcodes.py`` has run (and after a
    fresh integration run, which sets ``obs_names`` without renaming the
    index), so keying off the name would re-prefix an already-composite index
    into ``p1.p1.CATCGT...`` and match nothing.
    """
    with h5py.File(path, 'r') as f:
        obs = f['obs']
        index_key = _dec(obs.attrs['_index'])
        index = _read_obs_col(obs, index_key).astype(str)
        if 'sample' not in obs:
            return pd.Index(index)
        samples = _read_obs_col(obs, 'sample').astype(str)
        index = _strip_double_prefix(index, samples)
        if np.char.startswith(index, np.char.add(samples, '.')).all():
            return pd.Index(index)  # already composite
        return pd.Index(np.char.add(np.char.add(samples, '.'), index))


def _strip_double_prefix(index: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Undo the extra ``<ref0>.`` the scArches concat puts on reference cells.

    Building a scArches target prefixes the whole REFERENCE block with the
    FIRST reference library's name, so those cells arrive as
    ``<ref0>.<sample>.<barcode>`` -- ``pMock.pNOD.AGCC...`` on MtInduced,
    ``p1.p1.CATCGT...`` on MtInduced_TipSZmock -- while every query cell is the
    normal ``<sample>.<barcode>``.

    That block is **exactly the labelled one**, so without this the join
    silently matches zero refine rows and the run looks like a label transfer
    that covered nothing. The bare composite check above cannot catch it:
    ``p1.p1.CATCGT`` does start with ``p1.``, so it passes as "already
    composite" and is returned unrepaired.

    The rule drops the first field only when the SECOND field is the cell's own
    ``sample``, which is the signature of the doubling and leaves a barcode
    that merely contains dots alone. It is a no-op once the concat is fixed
    upstream, so it is safe to leave in place.
    """
    def _fix(barcode: str, sample: str) -> str:
        parts = barcode.split('.')
        if len(parts) >= 3 and parts[1] == sample:
            return barcode.split('.', 1)[1]
        return barcode

    out = np.array([_fix(bc, s) for bc, s in zip(index, samples)], dtype=object)
    n = int((out != index).sum())
    if n:
        print(f"  repaired {n} double-prefixed reference barcode(s)")
    return out.astype(str)


def load_labels(sources=None) -> pd.DataFrame:
    """Concatenate the selected refine label tables, tagging each row's source.

    `sources` is a list of `SOURCES` keys, defaulting to `DEFAULT_SOURCES`.
    Selecting rather than unioning everything is deliberate: refine runs may
    cover overlapping cells, and the uniqueness guard below is a stop sign, not
    something to work around by deduplicating -- two runs disagreeing about a
    cell is a question for a human, not a merge rule.
    """
    frames = []
    for name in (sources or DEFAULT_SOURCES):
        path = SOURCES[name]
        if not path.exists():
            sys.exit(f"missing refine labels: {path}")
        df = pd.read_csv(path, sep='\t', index_col=0)
        df['refine_source'] = name
        frames.append(df)
    labels = pd.concat(frames)
    if not labels.index.is_unique:
        dupes = labels.index[labels.index.duplicated()].unique()[:5]
        sys.exit(f"refine runs share {labels.index.duplicated().sum()} cells, "
                 f"e.g. {list(dupes)} -- resolve before injecting")
    return labels


def build_columns(keys: pd.Index, labels: pd.DataFrame) -> dict[str, np.ndarray]:
    """Align refine labels onto a target's cells; uncovered cells stay unset."""
    aligned = labels.reindex(keys)
    covered = aligned['refine_source'].notna()

    out: dict[str, np.ndarray] = {}
    for src, dst, kind in CARRIED:
        col = aligned[src]
        if kind == 'num':
            out[dst] = col.astype('float64').to_numpy()
        else:
            # refine writes an empty `flag` as NaN; that means "no flag", not
            # "no annotation", so only genuinely uncovered cells get the
            # sentinel and covered-but-blank becomes an explicit 'none'.
            vals = col.astype(object).where(col.notna(), 'none')
            out[dst] = np.where(covered, vals, NOT_ANNOTATED).astype(str)

    out['refine_source'] = np.where(
        covered, aligned['refine_source'].astype(object), NOT_ANNOTATED).astype(str)

    # Training-ready labels: refine's own hedges are not a cell type.
    flag = aligned['flag'].astype(object).where(aligned['flag'].notna(), '')
    hedged = (aligned['annotation'] == 'Unassigned') | flag.str.contains('low_confidence')
    out['refine_scanvi_labels'] = np.where(
        covered & ~hedged, aligned['annotation'].astype(object), NOT_ANNOTATED).astype(str)
    return out


def _write_categorical(obs: h5py.Group, name: str, values: np.ndarray) -> None:
    """Write a categorical obs column into an open h5ad.

    Args:
        obs: The h5py obs group.
        name (str): Column name.
        values (list[str]): Values.
    """
    cats = pd.Index(sorted(set(values.tolist())))
    codes = cats.get_indexer(values)
    dtype = np.int8 if len(cats) <= 127 else np.int32
    g = obs.create_group(name)
    g.attrs['encoding-type'] = 'categorical'
    g.attrs['encoding-version'] = '0.2.0'
    g.attrs['ordered'] = np.False_
    d = g.create_dataset('categories', data=np.array(cats, dtype=object),
                         dtype=h5py.string_dtype(encoding='utf-8'))
    d.attrs['encoding-type'] = 'string-array'
    d.attrs['encoding-version'] = '0.2.0'
    d = g.create_dataset('codes', data=codes.astype(dtype))
    d.attrs['encoding-type'] = 'array'
    d.attrs['encoding-version'] = '0.2.0'


def _write_numeric(obs: h5py.Group, name: str, values: np.ndarray) -> None:
    """Write a numeric obs column into an open h5ad.

    Args:
        obs: The h5py obs group.
        name (str): Column name.
        values: Numeric values.
    """
    d = obs.create_dataset(name, data=values.astype('float64'))
    d.attrs['encoding-type'] = 'array'
    d.attrs['encoding-version'] = '0.2.0'


def inject(path: Path, columns: dict[str, np.ndarray]) -> None:
    """Copy a refine run's label columns into an existing h5ad.

    Args:
        path (Path): The h5ad to write into.
        columns (dict): Column name to values.

    Returns:
        list[str]: The columns written.
    """
    numeric = {dst for _, dst, kind in CARRIED if kind == 'num'}
    with h5py.File(path, 'r+') as f:
        obs = f['obs']
        n_obs = len(_read_obs_col(obs, _dec(obs.attrs['_index'])))
        for name, values in columns.items():
            if len(values) != n_obs:
                raise RuntimeError(f"{name}: {len(values)} values for {n_obs} cells")
            if name in obs:
                del obs[name]  # idempotent re-run
            if name in numeric:
                _write_numeric(obs, name, values)
            else:
                _write_categorical(obs, name, values)

        order = [_dec(c) for c in obs.attrs['column-order']]
        order = [c for c in order if c not in columns] + list(columns)
        obs.attrs['column-order'] = np.array(order, dtype=object)


def main() -> None:
    """CLI entry point: inject refine labels into a target's h5ad.

    Returns:
        int: Process exit status.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true',
                      help='report coverage and write per-target TSVs, touch no h5ad')
    mode.add_argument('--commit', action='store_true',
                      help='add the columns to the h5ad files in place')
    ap.add_argument('--target', action='append', choices=sorted(TARGETS),
                    help='restrict to one target (repeatable); default all')
    ap.add_argument('--source', action='append', choices=sorted(SOURCES),
                    help='refine run to read labels from (repeatable); default '
                         f'{" + ".join(DEFAULT_SOURCES)}. Sources that overlap '
                         'cannot be combined -- see the module docstring.')
    ap.add_argument('--tsv-dir', type=Path, default=None,
                    help='where --dry-run writes its TSVs (default: alongside each h5ad)')
    args = ap.parse_args()

    labels = load_labels(args.source)
    print(f"refine labels: {len(labels)} cells "
          f"({labels['refine_source'].value_counts().to_dict()})\n")

    for name in (args.target or sorted(TARGETS)):
        path = TARGETS[name]
        if not path.exists():
            print(f"{name}: MISSING {path} -- skipped")
            continue

        keys = target_keys(path)
        if not keys.is_unique:
            sys.exit(f"{name}: composite sample.barcode key is not unique -- "
                     f"cannot align safely")
        columns = build_columns(keys, labels)

        covered = columns['refine_source'] != NOT_ANNOTATED
        trainable = columns['refine_scanvi_labels'] != NOT_ANNOTATED
        n = len(keys)
        print(f"{name}: {n} cells")
        print(f"  annotated      {covered.sum():>7} ({100 * covered.mean():5.1f}%)")
        print(f"  scanvi-usable  {trainable.sum():>7} ({100 * trainable.mean():5.1f}%)")
        by_source = pd.Series(columns['refine_source']).value_counts()
        for src, cnt in by_source.items():
            if src != NOT_ANNOTATED:
                print(f"    from {src:<8} {cnt:>7}")
        classes = sorted(set(columns['refine_scanvi_labels']) - {NOT_ANNOTATED})
        print(f"  classes ({len(classes)}): {', '.join(classes)}")

        if args.commit:
            inject(path, columns)
            print(f"  -> injected {len(columns)} columns in place")
        else:
            out_dir = args.tsv_dir or path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f'{name}_refine_labels.tsv'
            pd.DataFrame(columns, index=keys).to_csv(out, sep='\t')
            print(f"  -> {out}")
        print()


if __name__ == '__main__':
    main()
