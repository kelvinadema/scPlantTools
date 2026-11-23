"""Autotune search space resolution -- `integration/integrate.py`.

`resolve_search_space` turns the built-in defaults and a target's own
`search_space:` block into ray samplers. It is the only converter, deliberately,
so the two cannot drift into meaning different things -- which makes the value
grammar it implements worth pinning.

The rationale is in `docs/decisions/autotune-search-spaces.md`; the claims are
assertions here because prose cannot fail when the code regresses.
"""

from __future__ import annotations

import pytest

from integration.integrate import (_BASE_SPACE, _DRVI_NORM_PINS,
                                   _GAUSSIAN_INPUT_CLASSES,
                                   _NO_KL_WARMUP_CLASSES, _SAMPLERS,
                                   _SCVI_NORM_PINS, _SEARCH_SPACE_DEFAULTS,
                                   _looks_like_counts, _to_sampler,
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


def test_scvi_is_the_base_space_plus_its_own_normalization_pins():
    # Unrevised on purpose: the tune registry holds no plain-SCVI fit, so there
    # is no evidence to narrow the searched keys with. The three pins are
    # separate from _BASE_SPACE -- they are per-class, and sharing them would
    # impose DRVI's convention on the other classes.
    assert _SEARCH_SPACE_DEFAULTS['SCVI'] == {**_BASE_SPACE, **_SCVI_NORM_PINS}


def test_the_unanimous_drvi_choices_stay_pinned():
    # dispersion was chosen 6/6 across the registry's DRVI fits; a value chosen
    # unanimously carries no information and costs a factor in the grid.
    assert _SEARCH_SPACE_DEFAULTS['DRVI']['dispersion'] == ['gene-cell']


@pytest.mark.parametrize('pins', [_SCVI_NORM_PINS, _DRVI_NORM_PINS])
@pytest.mark.parametrize('key', ['use_layer_norm', 'use_batch_norm',
                                 'encode_covariates'])
def test_the_named_module_defaults_are_pinned_not_searched(key, pins):
    # Single-element on purpose: named so the constructor receives them, but
    # there is nothing to search.
    assert len(pins[key]) == 1


def test_the_normalization_pins_are_per_class_not_shared():
    # They sat in _BASE_SPACE until 2026-08-17, which meant plain SCVI was
    # built with scvi's own normalization inverted -- three modelling decisions
    # never chosen for it, and pinned, so no search could revisit them.
    assert not set(_SCVI_NORM_PINS) & set(_BASE_SPACE)
    assert _SCVI_NORM_PINS['use_batch_norm'] != _DRVI_NORM_PINS['use_batch_norm']


# ------------------------------------------------------------------- SysVI

def test_sysvi_replaces_the_base_space_rather_than_widening_it():
    # SysVAE takes neither count-likelihood key: its decoder emits a Normal and
    # the reconstruction term is GaussianNLLLoss. Inheriting _BASE_SPACE would
    # search two parameters the constructor swallows through **kwargs.
    space = _SEARCH_SPACE_DEFAULTS['SysVI']
    assert 'dispersion' not in space
    assert 'gene_likelihood' not in space
    assert not set(_SCVI_NORM_PINS) & set(space)


@pytest.mark.parametrize('key', ['prior', 'n_prior_components', 'out_var_mode'])
def test_the_sysvi_cycle_and_prior_controls_are_pinned(key):
    # elbo_validation cannot see them: SysVAE.loss puts the cycle term in
    # extra_metrics['cycle_loss'], so the metric is rec_loss + kl_local and a
    # search over the cycle/prior controls ranks them by their drag on
    # likelihood alone. Same defect as joint_embedding_weight.
    assert len(_SEARCH_SPACE_DEFAULTS['SysVI'][key]) == 1


def test_sysvi_still_searches_capacity():
    # The pins above are about what the criterion is blind to, not a decision
    # to stop searching -- architecture is exactly what ELBO ranks well.
    space = _SEARCH_SPACE_DEFAULTS['SysVI']
    for key in ('n_layers', 'n_latent', 'n_hidden', 'dropout_rate'):
        assert len(space[key]) > 1


def _warmup_probe(model_class, plan_kwargs):
    """_normalize_kl_warmup alone, with no adata and no training."""
    import logging
    from integration.integrate import SingleCellIntegrate

    probe = SingleCellIntegrate.__new__(SingleCellIntegrate)
    probe.model_class = model_class
    probe.target = 'probe'
    probe.logger = logging.getLogger('probe')
    return probe._normalize_kl_warmup(
        {'max_epochs': 200, 'plan_kwargs': dict(plan_kwargs)}, 24000)


@pytest.mark.parametrize('key', ['n_epochs_kl_warmup', 'n_steps_kl_warmup'])
def test_kl_warmup_is_stripped_for_sysvi_not_merely_warned_about(key):
    # scvi resets it to 0 regardless, but an unstripped value is still recorded
    # in the registry's train_params -- provenance claiming a warmup the fit
    # never had. It arrives from the base config's `defaults:`, so no target
    # author can remove it at the config level.
    out = _warmup_probe('SysVI', {key: 80, 'lr': 1e-3})
    assert key not in out['plan_kwargs']


def test_stripping_warmup_keeps_the_rest_of_plan_kwargs():
    # Above all the tuned lr: dropping it silently trains at a different rate
    # than the one that won the search.
    out = _warmup_probe('SysVI', {'n_epochs_kl_warmup': 80, 'lr': 1e-3})
    assert out['plan_kwargs'] == {'lr': 1e-3}
    assert out['max_epochs'] == 200


def test_warmup_normalization_still_applies_to_the_scvi_shaped_classes():
    out = _warmup_probe('JointEmbeddingSCVI', {'n_epochs_kl_warmup': 80, 'lr': 1e-3})
    assert out['plan_kwargs']['n_epochs_kl_warmup'] == 80


def test_sysvi_is_declared_in_both_departure_sets():
    # Each departure is silent if unhandled: counts into a Gaussian likelihood
    # converge to nothing meaningful, and a configured KL warmup is reset to 0
    # by SysVI.train while the config and registry still claim it.
    assert 'SysVI' in _GAUSSIAN_INPUT_CLASSES
    assert 'SysVI' in _NO_KL_WARMUP_CLASSES


# ------------------------------------------- the counts / lognorm detector

@pytest.mark.parametrize('values, expected', [
    ([0, 1, 5, 300], True),          # raw counts
    ([0.0, 1.0, 2.0], True),         # float dtype, still integral
    ([0.0, 0.6931, 2.3025], False),  # log1p output
    ([], True),                      # empty -> nothing to disprove
])
def test_looks_like_counts_tests_the_data_not_the_layer_name(values, expected):
    import numpy as np
    assert _looks_like_counts(np.array(values, dtype=float)) is expected


def test_looks_like_counts_reads_stored_values_of_a_sparse_matrix():
    import numpy as np
    import scipy.sparse as sp
    assert _looks_like_counts(sp.csr_matrix(np.array([[0., 3.], [1., 0.]])))
    assert not _looks_like_counts(sp.csr_matrix(np.array([[0., 1.6], [0.4, 0.]])))


def test_negative_values_are_not_counts():
    # A scaled/centered layer is integral nowhere near as often as it is
    # negative; either way it must not be re-normalized.
    import numpy as np
    assert not _looks_like_counts(np.array([-1.0, 0.0, 2.0]))


# --------------------------------------------- HVG trains, full set is saved

def test_hvg_selection_stashes_the_full_gene_object():
    """`do_hvg` must not decide which genes downstream stages may look at.

    Until 2026-08-18 the HVG subset replaced self.adata and was what got
    written, so markerid -- which reads that file -- could only nominate a
    marker from the 2,000. Measured on MtSZ_sysvi_labels that was 3.8% of the
    Medicago gene space.
    """
    import inspect
    from integration.integrate import SingleCellIntegrate
    src = inspect.getsource(SingleCellIntegrate._prep_scvi)
    assert '_full_gene_adata' in src, 'HVG branch no longer stashes the full object'
    save = inspect.getsource(SingleCellIntegrate._update_save_anndata)
    assert '_full_gene_adata' in save, 'save path no longer restores the full gene set'
    # The save path must call mark_trained_features, and it must write .var['hvg'].
    from integration.integrate import mark_trained_features
    assert 'mark_trained_features' in save, 'save path no longer marks the trained genes'
    assert "var['hvg']" in inspect.getsource(mark_trained_features), \
        'saved object must mark which genes were trained on'
