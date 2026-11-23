"""
Label-space target keys -- `layer`, `hvg_batch_key`, `exclude_genes`,
`derive_corrected_layers` (shared/config_spec.py, shared/targets.py).

See docs/decisions/two-layer-label-spaces.md. The invariant that matters most is
the first test: a target that states none of them resolves exactly as it did
before the keys existed, so nothing already fitted changes.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from shared import config_spec as spec
from shared.targets import (parse_layer, expand_env_path, resolve_run_spec,
                            TargetConfigError)
from shared.validate import _structural, Report


def _find_root():
    here = pathlib.Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / '.root').exists():
            return p
    raise RuntimeError('no .root')


REPO_ROOT = _find_root()
NEW_KEYS = ('derive_corrected_layers', 'layer', 'hvg_batch_key', 'exclude_genes')


def _cfg(**target):
    return {'IntegrateConfigs': {'T': {'method': 'scVI', 'Samples': ['s1'], **target}},
            'input_dir': '/in', 'output_dir': '/out'}


# ------------------------------------------------------------- the spec

def test_the_four_keys_are_target_keys_and_in_the_emitted_schema():
    for key in NEW_KEYS:
        assert key in spec.INTEGRATION_TARGET.properties
    schema = json.loads((REPO_ROOT / 'configs/schemas/integration_config.schema.json').read_text())
    props = schema['properties']['IntegrateConfigs']['additionalProperties']['properties']
    for key in NEW_KEYS:
        assert key in props, '{} missing from the emitted schema -- rerun --emit-schemas'.format(key)


def test_a_target_stating_none_of_them_resolves_as_before():
    run = resolve_run_spec(_cfg(), 'T')
    assert run['layer'] is None
    assert run['hvg_batch_key'] is None
    assert run['exclude_genes'] is None
    assert run['derive_corrected_layers'] is False


def test_structural_validation_accepts_the_keys_and_rejects_a_typo():
    ok = Report()
    _structural(spec.INTEGRATION_CONFIG,
                _cfg(layer=['corrected_rspliced', 'corrected_runspliced'],
                     hvg_batch_key='sample_type', exclude_genes='/x/genes.txt',
                     derive_corrected_layers=True), 'x.yaml', ok)
    assert ok.ok, ok.render()
    bad = Report()
    _structural(spec.INTEGRATION_CONFIG, _cfg(exclude_gene='/x'), 'x.yaml', bad)
    assert not bad.ok


# ----------------------------------------------------------- parse_layer

def test_a_string_or_one_element_list_is_one_layer():
    assert parse_layer('corrected_counts') == ['corrected_counts']
    assert parse_layer(['rspliced']) == ['rspliced']


def test_two_layers_are_kept_in_order():
    assert parse_layer(['a', 'b']) == ['a', 'b']


@pytest.mark.parametrize('raw', ['', [], ['a', 'b', 'c'], ['a', 'a'], ['a', 3], 7])
def test_malformed_layer_values_are_rejected(raw):
    with pytest.raises(TargetConfigError):
        parse_layer(raw)


def test_layer_flows_through_resolve_run_spec():
    run = resolve_run_spec(_cfg(layer=['s', 'u'], hvg_batch_key='sample_type',
                                derive_corrected_layers=True), 'T')
    assert run['layer'] == ['s', 'u']
    assert run['hvg_batch_key'] == 'sample_type'
    assert run['derive_corrected_layers'] is True


# ------------------------------------------------------- expand_env_path

def test_a_variable_is_expanded_from_the_environment(monkeypatch):
    monkeypatch.setenv('C11_TEST_DIR', '/some/where')
    assert expand_env_path('${C11_TEST_DIR}/genes.txt', 'exclude_genes') == '/some/where/genes.txt'


def test_an_unexpanded_variable_is_an_error_not_a_path(monkeypatch):
    monkeypatch.delenv('C11_NOPE', raising=False)
    with pytest.raises(TargetConfigError, match='unexpanded'):
        expand_env_path('${C11_NOPE}/genes.txt', 'exclude_genes')


def test_none_stays_none():
    assert expand_env_path(None) is None
