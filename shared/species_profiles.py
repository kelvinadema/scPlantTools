"""
One annotation config, several target species.

Most annotation settings are shared across target species. The four things
that genuinely differ per species are:

    target_species          which species is being annotated
    target_species_files    its proteome / annotation / marker paths + gene_id rule
    markers.generate.samples  the samples to generate markers for
    reference_species       the same pool minus the target (you do not reference
                            yourself), which is derivable rather than declared

So the config holds one shared block plus a small per-species profile, and
this module resolves `config + species name` into the single-species dict the
annotation stage reads.

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
    Return a config with one species' profile folded in, in single-species shape.

    With no `species_profiles:` block this is the identity, so a single-species
    config still works. With one, `species` (or `default_species:`) selects a
    profile; omitting both is an error rather than a guess -- picking the wrong
    target species silently would send OrthoFinder at the wrong proteome set and
    produce annotations against the wrong reference pool.

    `reference_species` is the declared pool **minus the resolved target**
    (you do not reference yourself), derived here rather than declared per
    species.
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
