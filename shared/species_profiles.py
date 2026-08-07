"""
One annotation config, several target species.

WHY THIS EXISTS. There were three files -- `annotation_config.yaml`,
`annotation_config_arabidopsis.yaml`, `annotation_config_zea_mays.yaml` --
totalling 1,130 lines, of which **73 of 117 keys were identical across all
three**. Every change to a shared parameter had to be made three times, and
nothing checked that it was. The four things that genuinely differ are:

    target_species          which species is being annotated
    target_species_files    its proteome / annotation / marker paths + gene_id rule
    markers.generate.samples  the samples to generate markers for
    reference_species       the same pool minus the target (you do not reference
                            yourself), which is derivable rather than declared

So the three files are one config plus a small per-species profile. This module
resolves `merged config + species name` back into exactly the dict the old
per-species file produced -- `tests/test_species_profiles.py` asserts that
equality against the originals, which is what makes the merge safe to trust.

Kept free of stage imports: `shared/validate.py` uses it during preflight.
"""

from __future__ import annotations

import copy


class SpeciesProfileError(ValueError):
    """Unknown species profile, or a malformed `species_profiles:` block."""


def profile_names(acfg: dict) -> list:
    """Declared profile names, in file order. Empty if the config declares none."""
    root = acfg.get('AnnotateConfigs') or {}
    return list((root.get('species_profiles') or {}).keys())


def resolve_species(acfg: dict, species: str = None) -> dict:
    """
    Return a config with one species' profile folded in, shaped exactly like the
    old single-species files.

    With no `species_profiles:` block this is the identity, so a config in the
    old shape still works. With one, `species` (or `default_species:`) selects a
    profile; omitting both is an error rather than a guess -- picking the wrong
    target species silently would send OrthoFinder at the wrong proteome set and
    produce annotations against the wrong reference pool.

    `reference_species` is the declared pool **minus the resolved target**. That
    is the rule the three original files followed by hand, and deriving it here
    means adding a species to the pool cannot be forgotten in two of three files.
    """
    root = acfg.get('AnnotateConfigs') or {}
    profiles = root.get('species_profiles') or {}
    if not profiles:
        if species:
            raise SpeciesProfileError(
                "--species {!r} was given but this annotation config declares no "
                "`species_profiles:` block.".format(species))
        return acfg

    name = species or root.get('default_species')
    if not name:
        raise SpeciesProfileError(
            "this annotation config declares species_profiles but no "
            "`default_species:`, so one must be named. Use --species with one of: "
            "{}.".format(', '.join(sorted(profiles))))
    if name not in profiles:
        raise SpeciesProfileError(
            "unknown species profile {!r}. Declared profiles are: {}.".format(
                name, ', '.join(sorted(profiles))))

    profile = profiles[name] or {}
    target = profile.get('target_species')
    if not target:
        raise SpeciesProfileError(
            "species profile {!r} does not set `target_species:`.".format(name))

    out = copy.deepcopy(acfg)
    cfg = out['AnnotateConfigs']

    cfg['target_species'] = target
    cfg['target_species_files'] = copy.deepcopy(profile.get('target_species_files') or {})

    # The reference pool is everything declared except the target itself.
    pool = copy.deepcopy(cfg.get('reference_species') or {})
    pool.pop(target, None)
    cfg['reference_species'] = pool

    samples = profile.get('samples')
    if samples is not None:
        cfg.setdefault('markers', {}).setdefault('generate', {})
        cfg['markers']['generate']['samples'] = copy.deepcopy(samples)

    # The profile block itself is not part of the resolved config -- downstream
    # code should never see it, so a stray read cannot depend on it.
    cfg.pop('species_profiles', None)
    cfg.pop('default_species', None)
    out['_resolved_species'] = name
    return out
