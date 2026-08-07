#!/usr/bin/env python3
"""
Phylogenetic weighting of the reference-species vote.

The annotation vote pools evidence from several reference species, and the
ranking score rewards agreement across them (`n_species` in the score at
`annotate.py`). But the reference panel is not phylogenetically balanced: for a
*Medicago* (legume) target, four of nine references are grasses (Poaceae) on
the far side of the ~160-My monocot/eudicot split, casting 44% of the ballots
from one clade. Counting them as four independent votes over-weights whatever
that one clade happens to agree on.

This is the species-axis analogue of the method-axis family weighting the
refinement stage already does (three cluster-dependent marker methods read one
leiden partition, so they are resolved as one family, not three votes). The
correction here is the same idea: **redundant, same-clade evidence should not
count as independent evidence.**

The measure, per candidate cell type:

    weighted_species = C + delta * (N - C)

where `N` is the raw number of supporting species and `C` is the number of
distinct clades among them. Its properties are what make it safe:

  - a lone species is never penalised: its clade contributes its first,
    full vote (`C` counts it once), so single-species calls -- the bulk of
    candidates -- keep `weighted == 1`.
  - genuinely independent support is untouched: four species from four clades
    give `C = N = 4`, so `weighted == 4`.
  - a redundant clade is down-weighted: four grasses give `C = 1, N = 4`, so
    `weighted = 1 + 3*delta` (1.75 at delta=0.25) rather than 4.
  - `delta` is the one knob, and it has a clean meaning -- the fraction of a
    vote a second (third, ...) member of an already-represented clade adds.
    **delta = 1 reproduces the old `n_species` exactly**, which is the revert.

`delta` is a starting point (0.25), not a measured optimum: there is no
gold-standard annotation to tune it against, which is the thing the pipeline
produces. It only ever *lowers* a multi-species score relative to the old one,
never raises one, so it re-ranks without inventing evidence.

**This changes the ranking score only.** Significance is decided upstream by the
Fisher test + BH-FDR and is untouched; the consensus/single/ambiguous
classification still keys off the raw species count. So the set of candidates,
and their q-values, are exactly as before -- only the order (and the top call
pre-filled in the decision template) can move.

Clades come from a config map (`phylogeny.clades`), not from the orthogroup
data: gene-content similarity confirms the coarse structure (the eudicot/grass
split is stark) but does not cleanly separate the legume subclade from the
other eudicots at any single threshold, and Fabaceae is exactly the group that
matters for a legume target. Taxonomy encodes that; a similarity cut blurs it.
"""

import logging

logger = logging.getLogger('AnnotationLogging')

# delta = 1 is a no-op (weighted == n_species); 0 counts only distinct clades.
DEFAULT_DELTA = 0.25

# Species with no clade assignment are treated as their own singleton clade --
# a conservative default (full weight, no redundancy assumed) rather than
# silently lumping them together.
UNASSIGNED_CLADE = None


class CladeWeights:
    """
    A clade assignment plus the weighting it induces. Cheap and immutable; one
    is built per run and passed to the annotator and the review report so both
    describe the vote the same way.
    """

    def __init__(self, clade_of: dict, delta: float = DEFAULT_DELTA):
        self.clade_of = dict(clade_of or {})
        self.delta = float(delta)

    def clade(self, species: str) -> str:
        """The clade of a species; its own name if unassigned (singleton)."""
        return self.clade_of.get(species) or species

    def clades(self, species) -> list:
        """Distinct clades among a set of supporting species, order-stable."""
        seen, out = set(), []
        for s in sorted(species):
            c = self.clade(s)
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def weighted_species(self, species) -> float:
        """
        Effective independent support: C + delta*(N - C).

        N and C both count *distinct* species/clades, so a species listed twice
        (which should not happen, but is cheap to guard) cannot inflate N.
        """
        species = set(species)
        n = len(species)
        if n == 0:
            return 0.0
        c = len(self.clades(species))
        return c + self.delta * (n - c)

    def profile(self, species) -> dict:
        """
        The clade breakdown for one candidate, for the review report.

        `{clade -> [species]}`, plus `n_clades`/`n_species`, so the page can say
        'backed by 2 legumes + Arabidopsis' and flag a call resting on a single
        clade. Advisory only -- nothing here feeds the score.
        """
        by_clade = {}
        for s in sorted(species):
            by_clade.setdefault(self.clade(s), []).append(s)
        return {
            'by_clade': by_clade,
            'n_clades': len(by_clade),
            'n_species': len(set(species)),
            'weighted': round(self.weighted_species(species), 3),
            'single_clade': len(by_clade) <= 1,
        }

    def as_record(self) -> dict:
        """Serialisable summary for provenance / the report payload."""
        return {'delta': self.delta,
                'clades': {c: sorted(s for s in self.clade_of if self.clade_of[s] == c)
                           for c in sorted(set(self.clade_of.values()))}}


def build(config: dict, species=None, logger_=None) -> CladeWeights:
    """
    Build `CladeWeights` from the `phylogeny:` config block.

    `species` (optional) is the list of species actually in the run; any that
    the config does not place in a clade is warned about and treated as its own
    singleton, so a newly-added species degrades to 'full weight, no redundancy'
    rather than silently joining someone else's clade.
    """
    log = logger_ or logger
    cfg = config or {}
    delta = cfg.get('delta', DEFAULT_DELTA)

    # Config is written clade -> [species] (readable); invert to species -> clade.
    clade_of = {}
    for clade, members in (cfg.get('clades') or {}).items():
        for sp in (members or []):
            if sp in clade_of and clade_of[sp] != clade:
                log.warning('[PHYLO] %s assigned to two clades (%s, %s); keeping %s',
                            sp, clade_of[sp], clade, clade)
                continue
            clade_of[sp] = clade

    if species is not None:
        missing = [s for s in species if s not in clade_of]
        if missing:
            log.warning('[PHYLO] %d species have no clade and count as singletons '
                        '(full weight): %s', len(missing), ', '.join(sorted(missing)))

    weights = CladeWeights(clade_of, delta=delta)
    if species is not None:
        # One line so the run log records exactly how the vote was weighted.
        by_clade = {}
        for s in species:
            by_clade.setdefault(weights.clade(s), []).append(s)
        log.info('[PHYLO] %d species in %d clade(s), delta=%.2f: %s', len(species),
                 len(by_clade), delta,
                 '; '.join('%s=%s' % (c, ','.join(sorted(m))) for c, m in sorted(by_clade.items())))
    return weights
