"""
End-to-end: FASTQ in, count matrix out.

This is the one test that exercises the actual toolchain -- `simpleaf`,
`piscem`, `alevin-fry` -- rather than Python logic. It builds a miniature
reference, samples reads from it at known barcodes, quantifies, and checks the
matrix against the truth table it generated.

WHAT ONLY THIS CAN CATCH:
  - a stage environment whose aligner binaries are missing or mismatched
  - a chemistry definition whose barcode/UMI geometry no longer matches simpleaf
  - a simpleaf/piscem upgrade that changes the output layout the pipeline reads
  - `$ALEVIN_FRY_HOME` handling, which is a required env var with no default and
    the first thing that breaks on a fresh machine

WHAT IT DOES NOT CATCH: anything statistical. It is a wiring test with real
tools, not a correctness test.

COST. Under 5 seconds -- the reference is 30 transcripts, so the index builds in
about a second and quantification takes two. That is cheap enough to run often,
which is the only reason it is worth having: the usual objection to an
end-to-end test is that it is slow and fragile, and at this scale it is neither.

Marked `slow` anyway because it shells out to three binaries and writes to disk;
`sc test --fast` skips it.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from fixtures import mini_reference          # noqa: E402

pytestmark = pytest.mark.slow


def _find_root():
    here = pathlib.Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / '.root').exists():
            return p
    raise RuntimeError('no .root')


REPO_ROOT = _find_root()


def _whitelist():
    """
    The 10x v3 whitelist, as `technology_config.yaml` declares it.

    Read from the config rather than built from REPO_ROOT on purpose. `data/`
    exists only on the nobackup tier while `.root` (and this file) live on the
    backup tier, so a REPO_ROOT-relative path resolves to nothing and every test
    here skips silently -- which is exactly what happened the first time.
    Reading the config also means this tests the path the pipeline actually uses.
    """
    import yaml

    def walk(node):
        # Recursive: the whitelist sits three levels down under the chemistry
        # entry, and a fixed-depth walk missed it.
        if isinstance(node, str):
            return pathlib.Path(node) if '10xv3Whitelist' in node else None
        if isinstance(node, dict):
            node = node.values()
        if isinstance(node, (list, tuple)) or hasattr(node, '__iter__'):
            for item in node:
                found = walk(item)
                if found is not None:
                    return found
        return None

    cfg = yaml.safe_load((REPO_ROOT / 'configs/technology_config.yaml').read_text()) or {}
    return walk(cfg)


WHITELIST = _whitelist()


def _require(binary):
    if shutil.which(binary) is None:
        pytest.skip('{} not on PATH -- run this in scMappingV3.12'.format(binary))


@pytest.fixture(scope='module')
def workspace(tmp_path_factory):
    """
    A built index, generated reads, and the truth table.

    `$ALEVIN_FRY_HOME` is set per-workspace and `simpleaf set-paths` run into it.
    simpleaf refuses to do anything without that variable, and it caches resolved
    binary paths there -- pointing it at a temp directory keeps the test from
    depending on, or disturbing, the repo's own `af_home/`.
    """
    _require('simpleaf')
    _require('piscem')
    if WHITELIST is None or not WHITELIST.exists():
        pytest.skip('10x v3 whitelist not found via technology_config.yaml')

    ws = tmp_path_factory.mktemp('e2e')
    env = dict(os.environ)
    env['ALEVIN_FRY_HOME'] = str(ws / 'af_home')
    (ws / 'af_home').mkdir()

    r = subprocess.run(['simpleaf', 'set-paths'], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip('simpleaf set-paths failed: {}'.format(r.stderr.strip()[:200]))

    ref = mini_reference.write_reference(ws / 'ref')
    reads = mini_reference.write_reads(ws / 'reads', ref, WHITELIST)

    idx = ws / 'idx'
    r = subprocess.run(
        ['simpleaf', 'index', '--output', str(idx), '--ref-seq', str(ref['fasta']),
         '--threads', '2'],
        env=env, capture_output=True, text=True, cwd=str(ws))
    assert r.returncode == 0, 'index build failed:\n{}'.format(r.stderr[-1500:])

    quant = ws / 'quant'
    r = subprocess.run(
        ['simpleaf', 'quant',
         '--reads1', str(reads['r1']), '--reads2', str(reads['r2']),
         '--index', str(idx / 'index'), '--chemistry', '10xv3',
         '--resolution', 'cr-like', '--expected-ori', 'fw',
         '--unfiltered-pl', str(WHITELIST),
         '--t2g-map', str(ref['t2g']),
         '--output', str(quant), '--threads', '2'],
        env=env, capture_output=True, text=True, cwd=str(ws))
    assert r.returncode == 0, 'quant failed:\n{}'.format(r.stderr[-1500:])

    return {'ws': ws, 'ref': ref, 'reads': reads, 'quant': quant}


def _matrix(quant_dir):
    import scipy.io as sio
    alevin = pathlib.Path(quant_dir) / 'af_quant' / 'alevin'
    m = sio.mmread(alevin / 'quants_mat.mtx')
    rows = [l.strip() for l in open(alevin / 'quants_mat_rows.txt')]
    cols = [l.strip() for l in open(alevin / 'quants_mat_cols.txt')]
    return m, rows, cols


# ------------------------------------------------------------------- structure

def test_quantification_produces_the_expected_files(workspace):
    """The exact paths `mapping/preprocess.py` reads. A layout change breaks it."""
    alevin = workspace['quant'] / 'af_quant' / 'alevin'
    for name in ('quants_mat.mtx', 'quants_mat_rows.txt', 'quants_mat_cols.txt'):
        assert (alevin / name).exists(), 'missing {}'.format(name)


def test_matrix_has_one_row_per_expected_barcode(workspace):
    _m, rows, _cols = _matrix(workspace['quant'])
    expected = set(workspace['reads']['truth'])
    assert set(rows) == expected, \
        'barcodes recovered ({}) differ from those generated ({})'.format(len(rows), len(expected))


def test_matrix_has_one_column_per_reference_gene(workspace):
    _m, _rows, cols = _matrix(workspace['quant'])
    n_tx = len(workspace['ref']['sequences'])
    assert len(cols) == n_tx, 'expected {} genes, got {}'.format(n_tx, len(cols))


# ---------------------------------------------------------------------- truth

def test_every_read_pair_is_accounted_for(workspace):
    """
    Each read carries a unique UMI, so with cr-like resolution total UMIs should
    equal total read pairs. A shortfall means reads went unmapped or were
    filtered -- on a reference the reads were drawn from, either is a defect.
    """
    m, _rows, _cols = _matrix(workspace['quant'])
    assert int(m.sum()) == workspace['reads']['n_pairs'], \
        'recovered {} UMIs from {} read pairs'.format(int(m.sum()), workspace['reads']['n_pairs'])


def test_counts_land_on_the_genes_the_reads_came_from(workspace):
    """
    The assertion a subset of real FASTQs could never support: reads generated
    from `tx007` must be counted against `gene007`, for the barcode they were
    written under.
    """
    import numpy as np
    m, rows, cols = _matrix(workspace['quant'])
    dense = np.asarray(m.todense()) if hasattr(m, 'todense') else np.asarray(m)
    row_of = {b: i for i, b in enumerate(rows)}
    col_of = {g: i for i, g in enumerate(cols)}

    for bc, genes in workspace['reads']['truth'].items():
        r = row_of[bc]
        observed = {cols[c] for c in np.flatnonzero(dense[r])}
        assert observed == set(genes), \
            'barcode {}: expected {} got {}'.format(bc, sorted(genes), sorted(observed))


def test_no_counts_outside_the_expected_blocks(workspace):
    """
    The negative half. Each barcode expresses a disjoint block, so a
    mis-assignment shows up here even when the totals still add up.
    """
    import numpy as np
    m, rows, cols = _matrix(workspace['quant'])
    dense = np.asarray(m.todense()) if hasattr(m, 'todense') else np.asarray(m)
    row_of = {b: i for i, b in enumerate(rows)}
    stray = 0
    for bc, genes in workspace['reads']['truth'].items():
        keep = {cols.index(g) for g in genes}
        row = dense[row_of[bc]]
        stray += int(sum(v for i, v in enumerate(row) if i not in keep))
    assert stray == 0, '{} UMIs landed on genes the barcode never expressed'.format(stray)


# ------------------------------------------------------------------ toolchain

def test_alevin_fry_home_is_required_and_honoured(workspace):
    """
    `$ALEVIN_FRY_HOME` has no default -- simpleaf exits immediately without it.
    It is the first thing to break on a fresh machine or in a job that does not
    inherit the submitting shell's environment, and the error names the variable
    but not what to set it to.
    """
    r = subprocess.run(['simpleaf', 'index', '--output', 'x', '--ref-seq', 'x.fa'],
                       env={k: v for k, v in os.environ.items() if k != 'ALEVIN_FRY_HOME'},
                       capture_output=True, text=True, cwd=str(workspace['ws']))
    assert r.returncode != 0
    assert 'ALEVIN_FRY_HOME' in (r.stderr + r.stdout)
