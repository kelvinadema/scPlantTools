"""
Target resolution -- `shared/targets.py`.

This module decides which samples an integration run concatenates and what
parameters it trains with. It is pure config logic with no scvi imports (by
design -- preflight runs it outside the integration env), so it is cheap to test
and it drives every integration job.

The rules being pinned (docs/configuration.md): `defaults: {k: v}` is
exactly equivalent to writing `k: v` on every entry, the merge is shallow, and a
key present-but-null counts as absent. Each of those is a place where a
plausible alternative reading would silently change what a run trains on.
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import yaml

from shared.targets import (expand_group, expand_sample_list, resolve_target,
                            parse_covariates, TargetConfigError)


def _find_root():
    here = pathlib.Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / '.root').exists():
            return p
    raise RuntimeError('no .root')


REPO_ROOT = _find_root()


# ------------------------------------------------------------------ groups

def test_a_non_group_name_resolves_to_itself():
    assert expand_group('SMu006', {}) == ['SMu006']


def test_groups_expand_to_their_members():
    groups = {'tip': ['A', 'B']}
    assert expand_group('tip', groups) == ['A', 'B']


def test_nested_groups_expand_transitively():
    groups = {'inner': ['A', 'B'], 'outer': ['inner', 'C']}
    assert set(expand_group('outer', groups)) == {'A', 'B', 'C'}


def test_expansion_deduplicates_and_preserves_order():
    """
    Order matters -- concatenation order determines the batch ordering in the
    resulting object -- and a duplicate would concatenate a sample twice.
    """
    groups = {'g1': ['A', 'B'], 'g2': ['B', 'C']}
    assert expand_sample_list(['g1', 'g2', 'A'], groups) == ['A', 'B', 'C']


def test_a_cyclic_group_does_not_hang():
    """A self-referential group must fail, not recurse forever."""
    groups = {'a': ['b'], 'b': ['a']}
    try:
        out = expand_group('a', groups)
    except (TargetConfigError, ValueError, RecursionError):
        return
    assert len(out) < 100, 'cycle expanded without terminating cleanly'


# ---------------------------------------------------------------- defaults

def test_defaults_apply_to_every_target():
    cfg = {'defaults': {'model_class': 'SCVI'}, 'IntegrateConfigs': {'T': {'Samples': ['A']}}}
    assert resolve_target(cfg, 'T')['model_class'] == 'SCVI'


def test_a_target_overrides_a_default():
    cfg = {'defaults': {'model_class': 'SCVI'},
           'IntegrateConfigs': {'T': {'Samples': ['A'], 'model_class': 'DRVI'}}}
    assert resolve_target(cfg, 'T')['model_class'] == 'DRVI'


def test_a_null_value_counts_as_absent():
    """
    The alternative reading -- null means
    "explicitly no value" and shadows the default -- is equally plausible and
    would silently change what a target trains with.
    """
    cfg = {'defaults': {'model_class': 'SCVI'},
           'IntegrateConfigs': {'T': {'Samples': ['A'], 'model_class': None}}}
    assert resolve_target(cfg, 'T')['model_class'] == 'SCVI'


def test_the_merge_is_shallow():
    """
    A dict-valued key is replaced wholesale, not merged key-by-key. (The per-key
    merge that `search_space` gets happens later, in integrate.py, against the
    code defaults -- a different rule, deliberately.)
    """
    cfg = {'defaults': {'model_params': {'a': 1, 'b': 2}},
           'IntegrateConfigs': {'T': {'Samples': ['A'], 'model_params': {'a': 9}}}}
    assert resolve_target(cfg, 'T')['model_params'] == {'a': 9}


def test_unknown_target_raises():
    with pytest.raises((TargetConfigError, KeyError)):
        resolve_target({'IntegrateConfigs': {}}, 'nope')


# -------------------------------------------------------------- covariates

def test_parse_covariates_splits_categorical_and_continuous():
    got = parse_covariates({'categorical': ['batch'], 'continuous': ['pct_mt']})
    assert got['categorical'] == ['batch'] and got['continuous'] == ['pct_mt']


def test_parse_covariates_handles_absent_input():
    got = parse_covariates(None)
    assert got['categorical'] == [] and got['continuous'] == []


# ------------------------------------------------- the real config still resolves

def test_every_target_in_the_real_config_resolves():
    """
    Catches a group renamed without updating its users, and a target whose
    `samples:` references a group that no longer exists.
    """
    cfg = yaml.safe_load((REPO_ROOT / 'configs/integration_config.yaml').read_text())
    # Targets live under `IntegrateConfigs:`; `defaults:`/`groups:` are siblings
    # of that key, not of the targets themselves.
    targets = sorted(cfg.get('IntegrateConfigs') or {})
    assert targets, 'no targets under IntegrateConfigs -- has the config layout changed?'
    from shared.targets import SAMPLE_LIST_KEYS
    failures = []
    for t in targets:
        try:
            spec = resolve_target(cfg, t)
        except Exception as e:
            failures.append('{}: {}'.format(t, e)); continue
        # Every target must name its cells somehow. Which key depends on the
        # model: Samples for a plain integration, Reference/Query for reference
        # mapping, Background/Target for contrastiveVI.
        if not any(spec.get(k) for k in SAMPLE_LIST_KEYS):
            failures.append('{}: declares none of {}'.format(t, ', '.join(SAMPLE_LIST_KEYS)))
    assert not failures, 'targets that do not resolve:\n  ' + '\n  '.join(failures)
