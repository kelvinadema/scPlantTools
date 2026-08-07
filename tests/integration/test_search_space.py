"""Autotune search space resolution -- `integration/integrate.py`.

`resolve_search_space` turns the built-in defaults and a target's own
`search_space:` block into ray samplers. It is the only converter, deliberately,
so the two cannot drift into meaning different things -- which makes the value
grammar it implements worth pinning.

The claims here were prose in `integrate.py` until 2026-08-06 (see
`docs/decisions/autotune-search-spaces.md`); they are assertions now because
prose cannot fail when the code regresses.
"""

from __future__ import annotations

import pytest

from integration.integrate import (_BASE_SPACE, _SAMPLERS,
                                   _SEARCH_SPACE_DEFAULTS, _to_sampler,
                                   resolve_search_space)


# ------------------------------------------------------------------ grammar

def test_a_list_becomes_a_choice_over_exactly_those_values():
    sampler = _to_sampler('n_layers', [1, 2, 3])
    assert sorted(sampler.categories) == [1, 2, 3]


def test_a_one_element_list_pins_the_parameter():
    sampler = _to_sampler('dispersion', ['gene-cell'])
    assert list(sampler.categories) == ['gene-cell']


def test_a_single_key_mapping_names_a_sampler():
    sampler = _to_sampler('lr', {'loguniform': [1e-4, 1e-2]})
    assert sampler.lower == pytest.approx(1e-4)
    assert sampler.upper == pytest.approx(1e-2)


def test_an_empty_list_is_rejected_rather_than_silently_dropping_the_key():
    with pytest.raises(ValueError, match='empty list'):
        _to_sampler('n_latent', [])


def test_a_sampler_outside_the_whitelist_is_rejected():
    # The search space is executed, so an arbitrary attribute name out of a
    # YAML file must not reach getattr(tune, name).
    with pytest.raises(ValueError, match='unknown sampler'):
        _to_sampler('n_latent', {'__class__': [1]})


def test_the_whitelist_only_names_real_ray_samplers():
    from ray import tune
    for name in _SAMPLERS:
        assert callable(getattr(tune, name))


def test_a_bare_scalar_is_rejected():
    # Not silently promoted to a 1-element choice: `n_latent: 30` and
    # `n_latent: [30]` should not be spelled two ways.
    with pytest.raises(ValueError):
        _to_sampler('n_latent', 30)


# ------------------------------------------------------------------ merging

def test_an_override_replaces_only_its_own_key():
    base = resolve_search_space('SCVI')
    merged = resolve_search_space('SCVI', {'n_layers': [7]})
    assert list(merged['n_layers'].categories) == [7]
    assert set(merged) == set(base)


def test_null_drops_the_key_entirely_leaving_scvi_its_own_default():
    merged = resolve_search_space('SCVI', {'dropout_rate': None})
    assert 'dropout_rate' not in merged
    assert 'dropout_rate' in resolve_search_space('SCVI')


def test_dropping_a_key_that_is_not_there_is_not_an_error():
    merged = resolve_search_space('SCVI', {'not_a_parameter': None})
    assert 'not_a_parameter' not in merged


def test_an_override_may_introduce_a_key_the_default_does_not_have():
    merged = resolve_search_space('SCVI', {'joint_embedding_weight': [1.0]})
    assert list(merged['joint_embedding_weight'].categories) == [1.0]


def test_an_unknown_model_class_raises_and_names_the_known_ones():
    with pytest.raises(ValueError, match='No default search space'):
        resolve_search_space('NotAModel')


def test_resolving_does_not_mutate_the_module_level_defaults():
    before = dict(_SEARCH_SPACE_DEFAULTS['SCVI'])
    resolve_search_space('SCVI', {'n_layers': [7], 'dropout_rate': None})
    assert _SEARCH_SPACE_DEFAULTS['SCVI'] == before


# ------------------------------------------------- the defaults themselves

def test_every_default_space_resolves():
    for model_class in _SEARCH_SPACE_DEFAULTS:
        assert resolve_search_space(model_class)


def test_drvi_widens_the_base_space_rather_than_replacing_it():
    # Verified against scvi-tools 1.5.0.post1: DRVI inherits SCVI's __init__
    # and DRVIModule takes **kwargs, so it accepts every base-space key.
    assert set(_BASE_SPACE) <= set(_SEARCH_SPACE_DEFAULTS['DRVI'])


def test_scvi_is_exactly_the_base_space():
    # Unrevised on purpose: the tune registry holds no plain-SCVI fit, so there
    # is no evidence to narrow it with.
    assert _SEARCH_SPACE_DEFAULTS['SCVI'] == _BASE_SPACE


def test_the_unanimous_drvi_choices_stay_pinned():
    # dispersion was chosen 6/6 across the registry's DRVI fits; a value chosen
    # unanimously carries no information and costs a factor in the grid.
    assert _SEARCH_SPACE_DEFAULTS['DRVI']['dispersion'] == ['gene-cell']


@pytest.mark.parametrize('key', ['use_layer_norm', 'use_batch_norm',
                                 'encode_covariates'])
def test_the_named_module_defaults_are_pinned_not_searched(key):
    # Single-element on purpose: named so the constructor receives them, but
    # there is nothing to search.
    assert len(_BASE_SPACE[key]) == 1
