#!/usr/bin/env python3
"""
Plant Ontology traversal.

One place to answer "what is this term part of, or a kind of", because the
answer has to be the same everywhere: the rollup that decides which
orthogroups a marker contributes to (standardize.py), the parent/child
collapse that decides which candidates survive (annotate.py), and the
specificity measure behind `too_general` all have to agree, or a term can be
simultaneously a parent for one purpose and unrelated for another.

**`is_a` alone is not enough for PO.** Anatomical containment is modelled with
`part_of`, and pronto's `superclasses()`/`subclasses()` follow only `is_a`, so
following just those makes the ontology look far flatter than it is. In
`po_final.obo`, 717 terms carry a `part_of` relationship:

    root cortex      is_a      cortex           <- superclasses() sees this
    root endodermis  part_of   root cortex      <- it does not
    pericycle cell   part_of   pericycle        <- nor this

Missing the second kind meant markers for `pericycle cell` never contributed
to `pericycle` or `root stele`, so those terms were tested on less evidence
than the data actually supports.

`DEFAULT_RELATIONS` is deliberately just `part_of`, alongside `is_a` which is
always followed. The other relations in the file (`develops_from`,
`adjacent_to`, `participates_in`, ...) are not containment and must not be
walked -- `develops_from` in particular would make a mature cell type inherit
its precursor's markers.
"""

DEFAULT_RELATIONS = ('part_of',)

# The traversal is a fixed-point walk rather than a recursion, so a cyclic
# part_of chain cannot blow the stack. PO should be acyclic; this does not
# assume it.
_MAX_DEPTH = 32


def _direct_parents(term, relations):
    """Immediate is_a parents plus the targets of each containment relation."""
    out = set()
    for parent in term.superclasses(with_self=False, distance=1):
        if parent.id != term.id:
            out.add(parent.id)
    for relation, targets in term.relationships.items():
        if relation.id in relations:
            out.update(t.id for t in targets)
    return out


def po_ancestors(term_id, ontology, relations=DEFAULT_RELATIONS):
    """
    Every term `term_id` is a kind of, or a part of, transitively.

    Returns pronto terms (not IDs) so callers can read `.name` without a second
    lookup. `term_id` itself is never included.
    """
    if term_id not in ontology:
        return []
    seen, frontier = set(), {term_id}
    for _ in range(_MAX_DEPTH):
        nxt = set()
        for tid in frontier:
            if tid in ontology:
                nxt |= _direct_parents(ontology[tid], relations)
        nxt -= seen | {term_id}
        if not nxt:
            break
        seen |= nxt
        frontier = nxt
    return [ontology[t] for t in sorted(seen) if t in ontology]


def po_descendants(term_id, ontology, relations=DEFAULT_RELATIONS):
    """
    Inverse of `po_ancestors`: every term that is a kind of, or a part of,
    `term_id`.

    Needed because pronto indexes relationships only in the forward direction,
    so `part_of` descendants cannot be read off a term the way `subclasses()`
    reads `is_a` ones. Callers that ask for this repeatedly should use
    `descendant_index` instead, which pays the one full pass once.
    """
    return [t for t in ontology.terms()
            if term_id in {a.id for a in po_ancestors(t.id, ontology, relations)}]


def descendant_index(ontology, relations=DEFAULT_RELATIONS) -> dict:
    """
    {term id -> set of descendant ids}, built in one pass over the ontology.

    `po_descendants` is O(terms) per call; the annotator asks for descendants
    of every candidate of every cluster, so doing it per query is quadratic
    over a 1791-term ontology. This inverts the ancestor map once instead.
    """
    index = {}
    for term in ontology.terms():
        for ancestor in po_ancestors(term.id, ontology, relations):
            index.setdefault(ancestor.id, set()).add(term.id)
    return index

