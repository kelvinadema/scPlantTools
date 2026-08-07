"""Config overlays -- `shared/targets.py::load_integration_config`.

A thesis chapter owns its integration targets without owning a copy of the
pipeline's cohort definitions: its config `extends:` the pipeline's and adds
targets. That makes the merge rules load-bearing, because a chapter overlay and
the base are edited by different people at different times.

The rule that matters most is that a name defined in **both** files raises. An
overlay cannot see what it would be shadowing, and silent precedence here means
a run trains on a target nobody chose -- the same failure class as the
`batch_key` precedence rules in `docs/decisions/batch-keys-and-confounds.md`.

Runs in the portable suite: pure config resolution, no scvi imports.
"""

from __future__ import annotations

import os

import pytest
import yaml

from shared.targets import (TargetConfigError, load_integration_config,
                            resolve_run_spec, resolve_target)


def _write(path, doc):
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


@pytest.fixture
def base(tmp_path):
    return _write(tmp_path / 'base.yaml', {
        'input_dir': '/data/mapping',
        'output_dir': '/data/integration',
        'defaults': {'method': 'scVI', 'tune': False, 'do_hvg': True},
        'groups': {'tip': ['p1', 'p2'], 'sz': ['s1', 's2']},
        'IntegrateConfigs': {'MtTip': {'Samples': ['tip']}},
    })


# ---------------------------------------------------------------- no overlay

def test_a_config_without_extends_loads_unchanged(base):
    cfg = load_integration_config(base)
    assert set(cfg['IntegrateConfigs']) == {'MtTip'}
    assert cfg['input_dir'] == '/data/mapping'


# ------------------------------------------------------------------- merging

def test_an_overlay_adds_its_targets_to_the_base_s(base, tmp_path):
    overlay = _write(tmp_path / 'ch5.yaml', {
        'extends': str(base),
        'IntegrateConfigs': {'MtSZ_DRVI': {'Samples': ['sz']}}})
    cfg = load_integration_config(overlay)
    assert set(cfg['IntegrateConfigs']) == {'MtTip', 'MtSZ_DRVI'}


def test_an_overlay_inherits_the_base_s_groups_and_site_paths(base, tmp_path):
    overlay = _write(tmp_path / 'ch5.yaml', {
        'extends': str(base),
        'IntegrateConfigs': {'MtSZ_DRVI': {'Samples': ['sz']}}})
    cfg = load_integration_config(overlay)
    assert cfg['groups']['tip'] == ['p1', 'p2']
    assert cfg['output_dir'] == '/data/integration'


def test_an_overlay_may_add_groups(base, tmp_path):
    overlay = _write(tmp_path / 'ch5.yaml', {
        'extends': str(base),
        'groups': {'sz_10x': ['s1']},
        'IntegrateConfigs': {'T': {'Samples': ['sz_10x']}}})
    cfg = load_integration_config(overlay)
    assert set(cfg['groups']) == {'tip', 'sz', 'sz_10x'}


def test_a_group_from_the_base_resolves_inside_an_overlay_target(base, tmp_path):
    overlay = _write(tmp_path / 'ch5.yaml', {
        'extends': str(base),
        'IntegrateConfigs': {'T': {'method': 'scVI', 'Samples': ['tip']}}})
    cfg = load_integration_config(overlay)
    assert resolve_target(cfg, 'T')['Samples'] == ['p1', 'p2']


def test_defaults_merge_per_key_with_the_overlay_winning(base, tmp_path):
    overlay = _write(tmp_path / 'ch5.yaml', {
        'extends': str(base),
        'defaults': {'tune': True},
        'IntegrateConfigs': {'T': {'Samples': ['sz']}}})
    cfg = load_integration_config(overlay)
    assert cfg['defaults']['tune'] is True       # overridden
    assert cfg['defaults']['do_hvg'] is True     # inherited
    assert cfg['defaults']['method'] == 'scVI'


def test_a_scalar_the_overlay_states_replaces_the_base_s(base, tmp_path):
    overlay = _write(tmp_path / 'ch5.yaml', {
        'extends': str(base),
        'output_dir': '/elsewhere',
        'IntegrateConfigs': {'T': {'Samples': ['sz']}}})
    assert load_integration_config(overlay)['output_dir'] == '/elsewhere'


def test_the_merged_config_carries_no_extends_key(base, tmp_path):
    # Downstream consumers must not be able to tell an overlay from a
    # standalone file.
    overlay = _write(tmp_path / 'ch5.yaml', {
        'extends': str(base),
        'IntegrateConfigs': {'T': {'Samples': ['sz']}}})
    assert 'extends' not in load_integration_config(overlay)


# ---------------------------------------------------------------- collisions

def test_a_target_defined_in_both_files_raises(base, tmp_path):
    overlay = _write(tmp_path / 'ch5.yaml', {
        'extends': str(base),
        'IntegrateConfigs': {'MtTip': {'Samples': ['sz']}}})
    with pytest.raises(TargetConfigError, match='MtTip'):
        load_integration_config(overlay)


def test_a_group_defined_in_both_files_raises(base, tmp_path):
    overlay = _write(tmp_path / 'ch5.yaml', {
        'extends': str(base),
        'groups': {'tip': ['other']},
        'IntegrateConfigs': {'T': {'Samples': ['tip']}}})
    with pytest.raises(TargetConfigError, match='tip'):
        load_integration_config(overlay)


def test_the_collision_message_names_both_files(base, tmp_path):
    overlay = _write(tmp_path / 'ch5.yaml', {
        'extends': str(base),
        'IntegrateConfigs': {'MtTip': {}}})
    with pytest.raises(TargetConfigError) as excinfo:
        load_integration_config(overlay)
    assert 'ch5.yaml' in str(excinfo.value)
    assert 'base.yaml' in str(excinfo.value)


# --------------------------------------------------------------- path forms

def test_a_relative_extends_resolves_against_the_overlay_not_the_cwd(base, tmp_path, monkeypatch):
    sub = tmp_path / 'chapter'
    sub.mkdir()
    overlay = _write(sub / 'ch5.yaml', {
        'extends': '../base.yaml',
        'IntegrateConfigs': {'T': {'Samples': ['sz']}}})
    monkeypatch.chdir(tmp_path.parent)          # a job starts wherever it starts
    assert 'MtTip' in load_integration_config(overlay)['IntegrateConfigs']


def test_an_env_var_in_extends_is_expanded(base, tmp_path, monkeypatch):
    monkeypatch.setenv('TEST_PIPELINE_ROOT', str(tmp_path))
    overlay = _write(tmp_path / 'ch5.yaml', {
        'extends': '${TEST_PIPELINE_ROOT}/base.yaml',
        'IntegrateConfigs': {'T': {'Samples': ['sz']}}})
    assert 'MtTip' in load_integration_config(overlay)['IntegrateConfigs']


def test_an_unset_env_var_raises_rather_than_reading_a_literal_path(tmp_path, monkeypatch):
    monkeypatch.delenv('TEST_PIPELINE_ROOT', raising=False)
    overlay = _write(tmp_path / 'ch5.yaml', {
        'extends': '${TEST_PIPELINE_ROOT}/base.yaml',
        'IntegrateConfigs': {'T': {}}})
    with pytest.raises(TargetConfigError, match='unexpanded'):
        load_integration_config(overlay)


def test_a_missing_base_names_what_it_resolved_to(tmp_path):
    overlay = _write(tmp_path / 'ch5.yaml', {
        'extends': str(tmp_path / 'nope.yaml'),
        'IntegrateConfigs': {'T': {}}})
    with pytest.raises(TargetConfigError, match='not a file'):
        load_integration_config(overlay)


# ------------------------------------------------------------------- chains

def test_overlays_may_be_chained(base, tmp_path):
    mid = _write(tmp_path / 'mid.yaml', {
        'extends': str(base),
        'IntegrateConfigs': {'Mid': {'Samples': ['sz']}}})
    top = _write(tmp_path / 'top.yaml', {
        'extends': str(mid),
        'IntegrateConfigs': {'Top': {'Samples': ['tip']}}})
    cfg = load_integration_config(top)
    assert set(cfg['IntegrateConfigs']) == {'MtTip', 'Mid', 'Top'}


def test_a_cycle_raises_rather_than_recursing_forever(tmp_path):
    a = tmp_path / 'a.yaml'
    b = tmp_path / 'b.yaml'
    _write(a, {'extends': str(b), 'IntegrateConfigs': {'A': {}}})
    _write(b, {'extends': str(a), 'IntegrateConfigs': {'B': {}}})
    with pytest.raises(TargetConfigError, match='cycle'):
        load_integration_config(a)


# ------------------------------------------------------- end-to-end resolution

def test_an_overlay_target_resolves_to_a_full_run_spec(base, tmp_path):
    overlay = _write(tmp_path / 'ch5.yaml', {
        'extends': str(base),
        'IntegrateConfigs': {'MtSZ': {'method': 'scVI', 'Samples': ['sz'],
                                      'batch_key': 'tech'}}})
    spec = resolve_run_spec(load_integration_config(overlay), 'MtSZ')
    assert spec['method'] == 'scVI'
    assert spec['samples'] == ['s1', 's2']       # base group, overlay target
    assert spec['batch_key'] == 'tech'
    assert spec['do_hvg'] is True                # base defaults
