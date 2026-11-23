"""Opt-in checks of the installed environments and the live research workspace.

Run `python -m sc test site`. Failures diagnose local state; nothing is changed.
"""
import json
import pathlib
import shutil
import subprocess
import pytest
import yaml
from sc import STAGE_ENVS as _DISPATCH
from shared.cohorts import resolve_cohort

REPO_ROOT = next(p for p in pathlib.Path(__file__).resolve().parents if (p / '.root').exists())
STAGE_ENVS = {module.split('.')[0]: env for module, env in _DISPATCH.values()}

def _load(rel):
    return yaml.safe_load((REPO_ROOT / rel).read_text())

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


def test_every_declared_env_exists_on_this_machine():
    envs = _conda_envs()
    if envs is None:
        pytest.skip('conda not available')
    missing = sorted(set(STAGE_ENVS.values()) - envs)
    assert not missing, 'declared but not installed: {}'.format(', '.join(missing))


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


def test_real_pipeline_config_declares_working_cohorts():
    cfg = _load('configs/pipeline_config.yaml')
    for name in cfg.get('cohorts', {}):
        sheet = resolve_cohort(cfg, name)['samples']
        assert (REPO_ROOT / sheet).exists(), '{} -> missing sheet {}'.format(name, sheet)

