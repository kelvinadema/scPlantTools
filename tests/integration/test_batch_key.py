"""`batch_key` resolution -- `shared/targets.py::parse_batch_key`.

Which column a model corrects on decides whether a treated cohort keeps its
treatment effect or has it removed (see
`docs/decisions/batch-keys-and-confounds.md`), so every rule here is one where a
silent fallback would produce a plausible model trained on the wrong thing.

Pure config logic with no scvi imports, by design -- preflight resolves these
outside the integration env.
"""

from __future__ import annotations

import pytest

from shared.targets import (BATCH_KEY_SEP, NULLABLE_KEYS, TargetConfigError,
                            parse_batch_key, resolve_batch_assignments,
                            resolve_target)


# ------------------------------------------------------------- simple forms

def test_null_means_no_correction_not_an_omission():
    spec = parse_batch_key(None)
    assert spec['mode'] == 'none'
    assert spec['column'] is None
    assert spec['components'] == ()


def test_a_string_names_an_existing_column():
    spec = parse_batch_key('tech')
    assert spec['mode'] == 'single'
    assert spec['column'] == 'tech'
    assert spec['components'] == ('tech',)


def test_an_empty_string_is_rejected_rather_than_read_as_null():
    with pytest.raises(TargetConfigError, match='empty string'):
        parse_batch_key('')


def test_batch_key_is_nullable_so_an_explicit_null_overrides_the_default():
    # Reading `batch_key: null` as "unstated" would hand batch correction to a
    # target that explicitly refused it.
    assert 'batch_key' in NULLABLE_KEYS
    resolved = resolve_target({'defaults': {'batch_key': 'batch'},
                               'IntegrateConfigs': {'T': {'batch_key': None}}}, 'T')
    assert resolved['batch_key'] is None


# ---------------------------------------------------------------- combined

def test_two_keys_derive_one_joined_column():
    spec = parse_batch_key(['tech', 'prep'])
    assert spec['mode'] == 'combined'
    assert spec['column'] == BATCH_KEY_SEP.join(['tech', 'prep'])
    assert spec['components'] == ('tech', 'prep')


def test_a_one_element_list_collapses_to_the_bare_key():
    # So a target can gain and lose a second key without its .obs column name
    # changing under it.
    assert parse_batch_key(['tech']) == parse_batch_key('tech')


def test_an_empty_list_is_rejected():
    with pytest.raises(TargetConfigError, match='empty list'):
        parse_batch_key([])


def test_a_repeated_key_is_rejected():
    # Repeating one duplicates its values inside every level without changing
    # the grouping.
    with pytest.raises(TargetConfigError, match='repeated'):
        parse_batch_key(['tech', 'tech'])


def test_a_non_string_key_is_rejected():
    with pytest.raises(TargetConfigError, match='not a column name'):
        parse_batch_key(['tech', 3])


# ------------------------------------------------------------------ mapped

def test_a_mapping_resolves_keys_per_sample():
    spec = parse_batch_key({'s1': 'sample', 's2': ['tech', 'prep']})
    assert spec['mode'] == 'mapped'
    assert spec['assignments']['s1'] == ('sample',)
    assert spec['assignments']['s2'] == ('tech', 'prep')


def test_mapping_keys_are_expanded_through_groups():
    spec = parse_batch_key({'tip': 'sample'}, groups={'tip': ['a', 'b']})
    assert set(spec['assignments']) == {'a', 'b'}


def test_every_referenced_key_is_reported_for_validation():
    spec = parse_batch_key({'s1': 'sample', 's2': ['tech', 'sample']})
    assert set(spec['components']) == {'sample', 'tech'}


def test_a_sample_claimed_by_two_entries_is_an_error_not_a_precedence_rule():
    # Silent precedence is how a cohort ends up corrected on something nobody
    # chose.
    with pytest.raises(TargetConfigError):
        parse_batch_key({'g1': 'sample', 'g2': 'tech'},
                        groups={'g1': ['a', 'b'], 'g2': ['b', 'c']})


def test_an_empty_mapping_is_rejected():
    with pytest.raises(TargetConfigError, match='empty mapping'):
        parse_batch_key({})


# -------------------------------------------------------------- coverage

def test_assignments_are_returned_in_the_target_s_sample_order():
    spec = parse_batch_key({'a': 'sample', 'b': 'tech'})
    assert list(resolve_batch_assignments(spec, ['b', 'a'])) == ['b', 'a']


def test_an_uncovered_sample_raises_rather_than_inheriting_a_default():
    spec = parse_batch_key({'a': 'sample'})
    with pytest.raises(TargetConfigError, match='does not cover'):
        resolve_batch_assignments(spec, ['a', 'b'])


def test_the_non_mapped_modes_have_no_assignments():
    for raw in (None, 'tech', ['tech', 'prep']):
        assert resolve_batch_assignments(parse_batch_key(raw), ['a']) is None


# ------------------------------------------------------------------ shape

@pytest.mark.parametrize('raw', [None, 'tech', ['tech', 'prep'],
                                 {'a': 'sample', 'b': 'tech'}])
def test_every_form_returns_the_same_four_keys(raw):
    assert set(parse_batch_key(raw)) == {'mode', 'column', 'components', 'assignments'}


def test_an_unsupported_type_is_rejected():
    with pytest.raises(TargetConfigError):
        parse_batch_key(3)
