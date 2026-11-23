"""
The output contract -- `shared/io.py`.

This is the interface between stages. Integration reads what mapping writes,
annotation reads what integration writes, and `sc status` reads the manifests.
Nothing downstream re-derives any of it, so a silent change here is a silent
change everywhere.

Smallest surface in the repo with the widest blast radius, which is why it is
tested first.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from fixtures import synthetic          # noqa: E402

from shared import io as sio            # noqa: E402


@pytest.fixture
def adata():
    return synthetic.clean_normal(n_cells=50)


# ------------------------------------------------------------- stamp_provenance

def test_provenance_records_every_contract_field(adata):
    sio.stamp_provenance(adata, sample='S1', mapper='alevin', stage='qc')
    rec = adata.uns[sio.PROVENANCE_KEY]
    for field in ('schema_version', 'sample', 'mapper', 'stage', 'timestamp'):
        assert field in rec, 'provenance is missing {!r}'.format(field)
    assert rec['schema_version'] == sio.SCHEMA_VERSION
    assert rec['sample'] == 'S1' and rec['mapper'] == 'alevin' and rec['stage'] == 'qc'


def test_provenance_never_stores_none(adata):
    """
    anndata SILENTLY DROPS None-valued keys on write, so a contract field set to
    None would vanish from the file rather than raise -- and the reader would see
    a missing key, indistinguishable from an old artifact. `io.py` coerces to ''
    for exactly this reason.
    """
    sio.stamp_provenance(adata, sample='S1', mapper='alevin', stage='qc',
                         config_paths=None, extra={'deliberate_none': None})
    def _walk(d, path=''):
        for k, v in d.items():
            here = '{}.{}'.format(path, k) if path else k
            assert v is not None, 'None at uns.{} -- anndata will drop it on write'.format(here)
            if isinstance(v, dict):
                _walk(v, here)
    _walk(adata.uns[sio.PROVENANCE_KEY])


def test_provenance_survives_a_write_read_roundtrip(adata, tmp_path):
    """The real test of the None rule: write it out, read it back, compare."""
    out = tmp_path / 'x.h5ad'
    sio.write_h5ad(adata, out, sample='S1', mapper='alevin', stage='qc')
    import anndata as ad
    back = ad.read_h5ad(out)
    before = adata.uns[sio.PROVENANCE_KEY]
    after = back.uns[sio.PROVENANCE_KEY]
    missing = set(before) - set(after)
    assert not missing, 'keys lost on the round trip: {}'.format(sorted(missing))


def test_extra_fields_are_merged_not_replaced(adata):
    sio.stamp_provenance(adata, sample='S1', mapper='alevin', stage='qc',
                         extra={'custom_key': 'custom_value'})
    rec = adata.uns[sio.PROVENANCE_KEY]
    assert rec.get('custom_key') == 'custom_value'
    assert rec['sample'] == 'S1', 'extra= clobbered the contract fields'


# -------------------------------------------------------------------- write_h5ad

def test_write_h5ad_creates_a_readable_file(adata, tmp_path):
    out = tmp_path / 'nested' / 'x.h5ad'
    sio.write_h5ad(adata, out, sample='S1', mapper='alevin', stage='qc')
    assert out.exists(), 'write_h5ad did not create parent directories'
    import anndata as ad
    back = ad.read_h5ad(out)
    assert back.n_obs == adata.n_obs and back.n_vars == adata.n_vars


def test_written_artifact_is_self_describing(adata, tmp_path):
    """
    A reader with only the file must be able to tell what produced it. That is
    the whole point of stamping rather than relying on directory layout.
    """
    out = tmp_path / 'x.h5ad'
    sio.write_h5ad(adata, out, sample='S1', mapper='alevin', stage='qc')
    import anndata as ad
    rec = ad.read_h5ad(out).uns[sio.PROVENANCE_KEY]
    assert rec['sample'] == 'S1'
    assert rec['mapper'] == 'alevin'
    assert rec['schema_version'] == sio.SCHEMA_VERSION


# ----------------------------------------------------------------- write_manifest

def test_manifest_is_valid_json_with_the_declared_shape(tmp_path):
    base = tmp_path / 'S1'
    (base / 'qc').mkdir(parents=True)
    (base / 'qc' / 'filtered.h5ad').write_bytes(b'x')
    path = base / 'manifest.json'
    sio.write_manifest(path, sample='S1', mapper='alevin',
                       artifacts={'filtered_h5ad': base / 'qc' / 'filtered.h5ad'},
                       base_dir=base)
    m = json.loads(path.read_text())
    for field in ('schema_version', 'sample', 'mapper', 'status', 'artifacts'):
        assert field in m, 'manifest is missing {!r}'.format(field)
    assert m['sample'] == 'S1'


def test_manifest_artifact_paths_are_relative_and_resolve(tmp_path):
    """
    `sc status` joins these onto the sample directory. An absolute path would
    resolve on the machine that wrote it and nowhere else -- which is exactly
    the failure the two-tier lustre layout invites.
    """
    base = tmp_path / 'S1'
    (base / 'qc').mkdir(parents=True)
    art = base / 'qc' / 'filtered.h5ad'
    art.write_bytes(b'x')
    path = base / 'manifest.json'
    sio.write_manifest(path, sample='S1', mapper='alevin',
                       artifacts={'filtered_h5ad': art}, base_dir=base)
    m = json.loads(path.read_text())
    for key, rel in m['artifacts'].items():
        assert not pathlib.Path(rel).is_absolute(), '{} is absolute: {}'.format(key, rel)
        assert (base / rel).exists(), '{} -> {} does not resolve under base_dir'.format(key, rel)


def test_manifest_status_is_recorded(tmp_path):
    base = tmp_path / 'S1'; base.mkdir()
    path = base / 'manifest.json'
    sio.write_manifest(path, sample='S1', mapper='alevin', artifacts={},
                       base_dir=base, status='partial')
    assert json.loads(path.read_text())['status'] == 'partial'


def test_manifest_is_json_serialisable_for_numpy_values(tmp_path):
    """
    Stage code passes numpy scalars around freely. `json.dump` raises on those,
    which would fail the manifest write at the very end of a multi-hour run --
    `_json_safe` exists to prevent that.
    """
    base = tmp_path / 'S1'; base.mkdir()
    path = base / 'manifest.json'
    sio.write_manifest(path, sample='S1', mapper='alevin', artifacts={},
                       base_dir=base,
                       extra={'n_cells': np.int64(1234), 'frac': np.float32(0.5)})
    m = json.loads(path.read_text())
    assert m['n_cells'] == 1234


# --------------------------------------------------------------------- versions

def test_tool_versions_are_strings_never_none():
    """Same None rule -- these go into `.uns` and into the manifest."""
    for name, ver in sio.tool_versions().items():
        assert ver is not None, '{} reported None'.format(name)
        assert isinstance(ver, str)


def test_config_hash_is_stable_and_sensitive(tmp_path):
    c = tmp_path / 'c.yaml'
    c.write_text('a: 1\n')
    first = sio.config_hash([c])
    assert first == sio.config_hash([c]), 'hash is not stable across calls'
    c.write_text('a: 2\n')
    assert sio.config_hash([c]) != first, 'hash did not change when the config did'


def test_config_hash_tolerates_missing_files(tmp_path):
    """Preflight and ad-hoc runs call this before every config exists."""
    sio.config_hash([tmp_path / 'nope.yaml'])
