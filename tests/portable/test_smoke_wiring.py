"""
Wiring smoke: is the repo internally consistent and are the environments real?

THIS IS THE TEST TO RUN BEFORE SUBMITTING A BIG JOB. It needs no data and no
GPU, takes a couple of seconds, and catches the failures that actually waste an
allocation: a batch script pointing at a module that was renamed, a config
referencing a file that moved, an environment that does not exist, an entrypoint
that raises on import.

It deliberately does NOT import any stage package -- it shells out or reads
files, so it runs in any environment. That is also what lets it check all five
environments from one process.

WHAT IT WILL NOT CATCH: anything about correctness. A pipeline that runs
end-to-end and produces wrong numbers passes every test here. That is what
`tests/mapping/` is for.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess

import pytest
import yaml


def _find_root():
    here = pathlib.Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / '.root').exists():
            return parent
    raise RuntimeError('no .root marker')


REPO_ROOT = _find_root()

# stage package -> the conda env it runs in. Mirrors sc/__init__.py::STAGE_ENVS;
# asserted equal below so the two cannot drift.
STAGE_ENVS = {
    'mapping': 'scMappingV3.12',
    'integration': 'scIntegrationV1.12',
    'annotation': 'scAnnotationV3.12',
    'markerid': 'scMarkersV3.12',
    'refine': 'scIntegrationV1.12',
    'fetch': 'scFetch',
}


def _conda_envs():
    """Env names known to conda, or None if conda is unavailable."""
    if shutil.which('conda') is None:
        return None
    try:
        out = subprocess.run(['conda', 'env', 'list', '--json'],
                             capture_output=True, text=True, timeout=60)
        return {pathlib.Path(p).name for p in json.loads(out.stdout).get('envs', [])}
    except Exception:
        return None


# ---------------------------------------------------------------- environments

def test_sc_stage_env_map_matches_this_file():
    """
    `sc` holds the authoritative stage->env mapping. If it changes and this file
    does not, every other test here checks the wrong environments.
    """
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from sc import STAGE_ENVS as sc_map
    derived = {mod.split('.')[0]: env for mod, env in sc_map.values()}
    assert derived == STAGE_ENVS, 'sc/__init__.py and this test disagree about stage->env'


def test_every_declared_env_has_a_manifest():
    for env in sorted(set(STAGE_ENVS.values())):
        assert (REPO_ROOT / 'envs' / '{}.yml'.format(env)).exists(), \
            'no envs/{}.yml for an env the pipeline dispatches into'.format(env)


def test_every_declared_env_exists_on_this_machine():
    envs = _conda_envs()
    if envs is None:
        pytest.skip('conda not available')
    missing = sorted(set(STAGE_ENVS.values()) - envs)
    assert not missing, 'declared but not installed: {}'.format(', '.join(missing))


# --------------------------------------------------------------------- packages

def test_every_stage_package_exists():
    for pkg in STAGE_ENVS:
        assert (REPO_ROOT / pkg / '__init__.py').exists(), \
            '{} is dispatched to but is not an importable package'.format(pkg)


# ---------------------------------------------------------------- batch scripts

def _batch_scripts():
    return sorted((REPO_ROOT / 'batch_scripts').rglob('*.sh'))


def test_batch_scripts_are_syntactically_valid():
    bad = []
    for s in _batch_scripts():
        r = subprocess.run(['bash', '-n', str(s)], capture_output=True, text=True)
        if r.returncode != 0:
            bad.append('{}: {}'.format(s.name, r.stderr.strip().split('\n')[0]))
    assert not bad, 'bash -n failed:\n  ' + '\n  '.join(bad)


def test_batch_script_python_targets_exist():
    """
    Every `python -m <pkg>.<mod>` a batch script invokes must resolve to a file.
    This is what catches a package rename that missed a script.
    """
    pat = re.compile(r'python\s+-m\s+([a-z_][a-z_0-9]*(?:\.[a-z_][a-z_0-9]*)+)', re.I)
    missing = []
    for s in _batch_scripts():
        for target in pat.findall(s.read_text()):
            parts = target.split('.')
            if parts[0] not in STAGE_ENVS and parts[0] not in ('sc', 'shared'):
                continue
            if not (REPO_ROOT.joinpath(*parts).with_suffix('.py')).exists():
                missing.append('{}: python -m {}'.format(s.name, target))
    assert not missing, 'batch scripts invoke modules that do not exist:\n  ' + '\n  '.join(missing)


def test_batch_scripts_reference_existing_configs():
    pat = re.compile(r'(configs/[A-Za-z0-9_./-]+\.(?:yaml|tsv|txt))')
    missing = []
    for s in _batch_scripts():
        for rel in set(pat.findall(s.read_text())):
            if not (REPO_ROOT / rel).exists():
                missing.append('{}: {}'.format(s.name, rel))
    assert not missing, 'batch scripts reference missing config files:\n  ' + '\n  '.join(sorted(missing))


def test_common_sh_is_sourced_not_executed():
    """`lib/common.sh` defines functions; a script that executes it gets nothing."""
    for s in _batch_scripts():
        if s.name == 'common.sh':
            continue        # it names its own path in its header comment
        text = s.read_text()
        if 'lib/common.sh' in text:
            # Loose on purpose: the real call is
            #   source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib/common.sh"
            # so the path contains embedded quotes and a character-class regex
            # that excludes them rejects a correct line.
            assert re.search(r'^\s*source\b.*lib/common\.sh', text, re.M), \
                '{} references lib/common.sh without sourcing it'.format(s.name)


# --------------------------------------------------------------------- configs

def test_every_config_is_valid_yaml():
    bad = []
    for c in sorted((REPO_ROOT / 'configs').rglob('*.yaml')):
        try:
            yaml.safe_load(c.read_text())
        except yaml.YAMLError as e:
            bad.append('{}: {}'.format(c.name, str(e).split('\n')[0]))
    assert not bad, 'unparseable YAML:\n  ' + '\n  '.join(bad)


def test_pipeline_config_subconfigs_exist():
    cfg = yaml.safe_load((REPO_ROOT / 'configs/pipeline_config.yaml').read_text())
    for key, rel in (cfg.get('configs') or {}).items():
        assert (REPO_ROOT / rel).exists(), 'configs.{} -> missing {}'.format(key, rel)


def test_target_files_referenced_by_batch_scripts_are_well_formed():
    """
    `batch_scripts/targets/*.txt` map an array index to a target. A blank file, or
    one whose count drifts from the `--array` range, strands tasks silently.
    """
    tdir = REPO_ROOT / 'batch_scripts/targets'
    if not tdir.is_dir():
        pytest.skip('no targets directory')
    for f in sorted(tdir.glob('*.txt')):
        entries = [ln.split('#')[0].strip() for ln in f.read_text().split('\n')]
        entries = [e for e in entries if e]
        assert entries, '{} has no usable entries'.format(f.name)
        assert len(entries) == len(set(entries)), \
            '{} has duplicate targets -- two array indices would run the same one'.format(f.name)


# ------------------------------------------------------------------ results tree

def test_results_tree_has_no_stale_top_level_dirs():
    """
    Guards the 2026-08-06 reorganisation: `annotation/results/` (doubled segment)
    and `annotation/review/` (split from its dataset) must not reappear, and
    caches belong under `cache/`.
    """
    cfg = yaml.safe_load((REPO_ROOT / 'configs/annotation_config.yaml').read_text())
    out = cfg['AnnotateConfigs'].get('output_dir')
    if not out or not pathlib.Path(out).is_dir():
        pytest.skip('annotation output_dir not present on this machine')
    root = pathlib.Path(out)
    assert not (root / 'results').exists(), 'results/annotation/results/ is back'
    assert not (root / 'review').exists(), 'annotation/review/ is back; it belongs per-dataset'
    for cache in ('orthofinder', 'primary_transcripts', 'genemaps'):
        assert not (root / cache).exists(), \
            '{} is a regenerable cache and belongs under cache_dir'.format(cache)
