"""
Plant Ontology traversal -- `annotation/ontology.py`.

The one rule worth pinning: **traversal must follow `part_of`, not just `is_a`.**
717 of the terms in `po_final.obo` carry a `part_of` relationship, and anatomy is
overwhelmingly expressed that way -- "root cortex *part_of* root", not "root
cortex *is_a* root". A traversal that walked only `is_a` would return a
plausible-looking but much smaller ancestor set, and every enrichment computed
against it would be quietly wrong rather than obviously broken.

Requires `scAnnotationV3.12` (pronto).
"""

from __future__ import annotations

import pathlib

import pytest

from annotation.ontology import po_ancestors, po_descendants, descendant_index


def _find_root():
    here = pathlib.Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / '.root').exists():
            return p
    raise RuntimeError('no .root')


REPO_ROOT = _find_root()
ONTOLOGY = REPO_ROOT / 'annotation' / 'po_final.obo'


pytestmark = pytest.mark.slow      # loading po_final.obo takes ~40 s


@pytest.fixture(scope='module')
def ontology():
    if not ONTOLOGY.exists():
        pytest.skip('po_final.obo not present')
    import pronto
    return pronto.Ontology(str(ONTOLOGY))


def _ids(terms):
    """
    `po_ancestors` / `po_descendants` yield pronto Term objects, while
    `descendant_index` maps id-string to a set of id-strings. Normalise before
    comparing, or every comparison is str-vs-Term and silently False.
    """
    return {getattr(t, 'id', t) for t in terms}


def test_ontology_loads_offline(ontology):
    """
    The whole point of the `po_final.obo` stripping chain is that pronto can load
    it without resolving external references. If this starts reaching the
    network, the chain has been undone -- see resources/ontology/attic/README.md.
    """
    assert len(ontology.terms()) > 1000


def test_traversal_follows_part_of_not_only_is_a(ontology):
    """
    The load-bearing test. Find a term whose ancestor set differs between
    is_a-only and is_a+part_of, and assert the default traversal includes the
    part_of ancestors.
    """
    both, is_a_only = None, None
    for term in ontology.terms():
        tid = term.id
        if not tid.startswith('PO:'):
            continue
        a_both = _ids(po_ancestors(tid, ontology, relations=('is_a', 'part_of')))
        a_isa = _ids(po_ancestors(tid, ontology, relations=('is_a',)))
        if a_both - a_isa:
            both, is_a_only = a_both, a_isa
            break
    if both is None:
        pytest.skip('no term in this ontology distinguishes the two traversals')
    assert both > is_a_only, 'part_of contributed no ancestors anywhere'


def test_ancestors_and_descendants_are_inverse(ontology):
    """If X is an ancestor of Y then Y is a descendant of X. Cheap consistency."""
    checked = 0
    for term in ontology.terms():
        tid = term.id
        if not tid.startswith('PO:'):
            continue
        ancestors = list(po_ancestors(tid, ontology))
        if not ancestors:
            continue
        parent = getattr(ancestors[0], 'id', ancestors[0])
        assert tid in _ids(po_descendants(parent, ontology)), \
            '{} lists {} as an ancestor but is not among its descendants'.format(tid, parent)
        checked += 1
        if checked >= 25:
            break
    assert checked, 'no term had any ancestors -- traversal returned nothing'


def test_a_term_is_not_its_own_ancestor(ontology):
    """A self-referential ancestor set makes every enrichment trivially true."""
    n = 0
    for term in ontology.terms():
        tid = term.id
        if not tid.startswith('PO:'):
            continue
        assert tid not in _ids(po_ancestors(tid, ontology)), '{} is its own ancestor'.format(tid)
        n += 1
        if n >= 50:
            break


def test_unknown_term_does_not_raise(ontology):
    """Marker tables carry stale PO ids; an unknown one must not kill the run."""
    assert _ids(po_ancestors('PO:9999999', ontology)) == set()


def test_descendant_index_agrees_with_direct_traversal(ontology):
    """
    `descendant_index` is the precomputed form used in the hot loop. If it drifts
    from `po_descendants`, enrichment silently uses a different term set than the
    one the code appears to describe.
    """
    index = descendant_index(ontology)
    checked = 0
    for tid, descendants in index.items():
        assert set(descendants) == _ids(po_descendants(tid, ontology)), \
            'index and direct traversal disagree for {}'.format(tid)
        checked += 1
        if checked >= 20:
            break
    assert checked, 'descendant_index was empty'
