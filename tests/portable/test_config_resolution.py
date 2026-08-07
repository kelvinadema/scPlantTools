"""
Cohort and species-profile resolution.

Both mechanisms exist to make a run say what it operates on. Both are also
places where getting it wrong is silent and expensive: the wrong cohort maps the
wrong dataset, and the wrong species profile sends OrthoFinder at the wrong
proteome set. Neither failure raises -- it just produces confident output about
the wrong thing. Hence tests.

No pipeline dependencies: these parse YAML and call two `shared/` resolvers.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

from shared.cohorts import resolve_cohort, CohortConfigError
from shared.species_profiles import resolve_species, SpeciesProfileError

def _find_root():
    """Nearest ancestor with a `.root` marker -- depth-independent, so this file
    can move between tests/ and tests/<subdir>/ without resolving one level short."""
    here = pathlib.Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / '.root').exists():
            return parent
    raise RuntimeError('no .root marker above {}'.format(here))


REPO_ROOT = _find_root()


def _load(rel):
    return yaml.safe_load((REPO_ROOT / rel).read_text())


# ---------------------------------------------------------------- cohorts

def test_no_cohorts_block_is_the_old_behaviour():
    cfg = {'configs': {'samples': 'sheet.yaml'}, 'output_dir': '/out'}
    got = resolve_cohort(cfg)
    assert got['samples'] == 'sheet.yaml' and got['name'] is None


def test_cohort_flag_against_a_config_with_no_cohorts_is_an_error():
    """Silently ignoring --cohort would be worse than rejecting it: the user
    asked for a specific dataset and would get whatever the pointer says."""
    with pytest.raises(CohortConfigError):
        resolve_cohort({'configs': {'samples': 'sheet.yaml'}}, 'medicago')


def test_declared_cohort_overrides_configs_samples():
    cfg = {'configs': {'samples': 'fallback.yaml'},
           'cohorts': {'a': {'samples': 'a.yaml'}, 'b': {'samples': 'b.yaml'}},
           'default_cohort': 'a'}
    assert resolve_cohort(cfg)['samples'] == 'a.yaml'
    assert resolve_cohort(cfg, 'b')['samples'] == 'b.yaml'


def test_cohorts_without_a_default_require_the_flag():
    cfg = {'configs': {}, 'cohorts': {'a': {'samples': 'a.yaml'}}}
    with pytest.raises(CohortConfigError):
        resolve_cohort(cfg)


def test_unknown_cohort_names_the_valid_ones():
    cfg = {'configs': {}, 'cohorts': {'a': {'samples': 'a.yaml'}}, 'default_cohort': 'a'}
    with pytest.raises(CohortConfigError) as e:
        resolve_cohort(cfg, 'nope')
    assert 'a' in str(e.value)


def test_real_pipeline_config_declares_working_cohorts():
    cfg = _load('configs/pipeline_config.yaml')
    for name in cfg.get('cohorts', {}):
        sheet = resolve_cohort(cfg, name)['samples']
        assert (REPO_ROOT / sheet).exists(), '{} -> missing sheet {}'.format(name, sheet)


# ---------------------------------------------------------- species profiles

def test_no_profiles_block_is_the_identity():
    cfg = {'AnnotateConfigs': {'target_species': 'X', 'reference_species': {'Y': {}}}}
    assert resolve_species(cfg) == cfg


def test_target_is_removed_from_the_reference_pool():
    """
    A species is never its own reference. The three original configs each applied
    this by hand, which is why adding a species to the pool could be -- and was --
    forgotten in some of them.
    """
    cfg = {'AnnotateConfigs': {
        'reference_species': {'A': {'p': 1}, 'B': {'p': 2}, 'C': {'p': 3}},
        'default_species': 'b',
        'species_profiles': {'b': {'target_species': 'B', 'target_species_files': {'f': 1}}}}}
    got = resolve_species(cfg)['AnnotateConfigs']
    assert got['target_species'] == 'B'
    assert set(got['reference_species']) == {'A', 'C'}
    assert 'species_profiles' not in got, 'the profile block must not leak downstream'


def test_unknown_species_names_the_valid_ones():
    cfg = {'AnnotateConfigs': {'species_profiles': {'a': {'target_species': 'A'}},
                               'default_species': 'a'}}
    with pytest.raises(SpeciesProfileError) as e:
        resolve_species(cfg, 'nope')
    assert 'a' in str(e.value)


@pytest.mark.parametrize('species', ['medicago', 'arabidopsis', 'zea_mays'])
def test_every_declared_profile_resolves_completely(species):
    """
    Each profile must yield a usable config: a target, a non-empty reference pool
    that excludes the target, and the target's own file block.

    This replaced an equality check against the three original per-species files
    (`annotation_config{,_arabidopsis,_zea_mays}.yaml`), which proved the merge
    faithful at the time it was made -- all three resolved byte-identically --
    but could not survive their deletion. What is checked now is the invariant
    that made the merge correct in the first place.
    """
    cfg = _load('configs/annotation_config.yaml')
    got = resolve_species(cfg, species)['AnnotateConfigs']
    target = got['target_species']
    assert target, 'profile {} resolved no target_species'.format(species)
    assert got['target_species_files'].get('proteome_file'), 'no proteome for ' + target
    assert target not in got['reference_species'], target + ' is its own reference'
    assert len(got['reference_species']) >= 8, 'reference pool collapsed'
