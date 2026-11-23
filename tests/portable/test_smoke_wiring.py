"""
Portable source/configuration wiring. Local environment/results checks live in tests/site/.

These checks inspect source and YAML without importing stage packages. They do
not depend on the developer's installed conda environments or live results.

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




# --------------------------------------------------------------------- packages

def test_every_stage_package_exists():
    for pkg in STAGE_ENVS:
        assert (REPO_ROOT / pkg / '__init__.py').exists(), \
            '{} is dispatched to but is not an importable package'.format(pkg)


# ---------------------------------------------------------------- batch scripts

def _batch_scripts():
    scripts = sorted((REPO_ROOT / 'batch_scripts').rglob('*.sh'))
    assert scripts, 'public launchers are missing from the checkout'
    return scripts


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




# ------------------------------------------------------------------ results tree
