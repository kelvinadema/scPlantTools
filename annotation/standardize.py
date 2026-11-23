import io
import re

import pandas as pd
import pronto

from difflib import get_close_matches, SequenceMatcher

from shared.utils import merged_obo_bytes
from .ontology import po_ancestors


class CellTypeStandardizer:
    """Map free-text cell-type labels onto Plant Ontology terms."""

    def __init__(self, obo_file: str, extensions=()):
        # Extensions are merged as OBO text before loading, so a grafted term's
        # is_a/part_of resolve against real PO terms in the same document and
        # every operation below -- the name lookup built here, the DAG collapse
        # and too_general count in annotate.py (which read this same self.po) --
        # treats them as native. This is the single injection point for the
        # nodule vocabulary; nothing downstream special-cases it.
        self.po = pronto.Ontology(io.BytesIO(merged_obo_bytes(obo_file, extensions)))
        self.lookup, self.token_lookup = self._build_lookup()
        # Materialised once: get_close_matches needs a sequence, and rebuilding
        # it per call dominates match()'s cost.
        self._labels = list(self.lookup)

    def _build_lookup(self):
        """Index every ontology term's name and synonyms for matching."""
        lookup = {}
        token_lookup = {}
        for term in self.po.terms():
            if term.obsolete:
                continue
            labels = [term.name.lower()] + [syn.description.lower() for syn in term.synonyms]
            for lbl in labels:
                lookup[lbl] = {"id": term.id, "name": term.name}
                token_lookup[lbl] = self._tokenize(lbl)
        return lookup, token_lookup

    def _tokenize(self, text: str):
        """Split a cell-type string into a set of lowercase word tokens."""
        return set(re.findall(r'\b\w+\b', text.lower()))

    def _break_tie(self, label, tied_labels):
        """
        Resolve a token-overlap tie, or return None if it genuinely cannot be.

        Rejecting every tie would drop labels whose right answer sits in the
        tied set. Two structurally different cases, distinguished by whether the
        query is broader or narrower than its candidates:

        **The query is broader** -- its tokens are a subset of the candidates'.
        'Epidermis' ties with `plant epidermis`, `leaf epidermis`,
        `root epidermis`, `petiole epidermis`, ... all at 0.5. That is not real
        ambiguity: `plant epidermis` is the **common ancestor** of the others,
        and a bare 'epidermis' means exactly that generic term. So if one tied
        candidate is an ancestor of every other, it wins.

        **The query is narrower** -- the candidates' tokens are a subset of
        its own. 'Root exodermis' ties `exodermis` with `root`. Here the
        ancestor rule would pick the *wrong* one (exodermis is part_of root, so
        `root` is the ancestor), yet 'root exodermis' plainly means the
        exodermis. English anatomical compounds are head-final, so the last
        token of the query names what the thing *is*: prefer the candidate
        containing it.

        Anything else still returns None. An unmapped label is recoverable and
        loudly reported; a confidently wrong term is neither.
        """
        ordered = re.findall(r'\b\w+\b', label.lower())
        label_tokens = set(ordered)
        by_id = {}
        for lbl in tied_labels:
            by_id.setdefault(self.lookup[lbl]["id"], lbl)

        # Head noun FIRST: for 'root exodermis' the ancestor rule would return
        # `root`, the container rather than the thing (see the docstring).
        head = ordered[-1] if ordered else None
        if head:
            contenders = [lbl for lbl in by_id.values()
                          if self.token_lookup[lbl] <= label_tokens
                          and head in self.token_lookup[lbl]]
            if len({self.lookup[c]["id"] for c in contenders}) == 1:
                return contenders[0]

        # Query broader than its candidates -> the generic ancestor among them.
        ancestors = {tid: {a.id for a in po_ancestors(tid, self.po)} for tid in by_id}
        for tid in by_id:
            others = set(by_id) - {tid}
            if others and all(tid in ancestors[other] for other in others):
                return by_id[tid]
        return None

    def _best_token_match(self, label: str, threshold: float = 0.5):
        """
        Closest ontology label by Jaccard overlap of tokens, or None.

        Jaccard (intersection over *union*) rather than intersection over the
        query's own tokens: the latter is asymmetric, so a short query sharing
        one generic token with a long candidate scores 0.5 and passes --
        'Myrosinase cell', 'Photosynthetic cell' and 'Proliferating cell' all
        resolve to 'cultured plant cell' that way, sharing only 'cell'.

        A tie on the best score between terms with *different* PO ids is
        rejected rather than broken arbitrarily: 'Epidermis' overlaps 'bract
        epidermis' and 'leaf epidermis' equally, and picking one silently
        invents a specificity the label does not have (a strict '>' would keep
        whichever came first in ontology file order). Returning None leaves the
        label unmapped, which is recoverable; a confidently wrong term is not.

        `_standardized.csv` marker tables are cached and reused (see
        OMGBuilder._load_markers), so a cache written under an earlier matching
        rule keeps that rule's mappings until it is deleted.
        """
        label_tokens = self._tokenize(label)
        if not label_tokens:
            return None

        scored = []
        for candidate, candidate_tokens in self.token_lookup.items():
            shared = label_tokens & candidate_tokens
            if not shared:
                continue
            score = len(shared) / len(label_tokens | candidate_tokens)
            if score >= threshold:
                scored.append((-score, len(candidate_tokens), candidate))
        if not scored:
            return None

        scored.sort()
        best_score, best_size, best = scored[0]
        tied_labels = [c for s, size, c in scored
                       if s == best_score and size == best_size]
        tied = {self.lookup[c]["id"] for c in tied_labels}
        if len(tied) > 1:
            return self._break_tie(label, tied_labels)
        return best

    def _make_entry(self, matched_label, rollup):
        """
        Roll a matched cell type up to the terms it generalises to.

        **`is_a` only -- deliberately not `part_of`, unlike everywhere else in
        this stage.** Marker inheritance and anatomical containment are
        different relations and only one of them propagates markers:

            root hair cell IS_A   epidermal cell  -> its markers do mark
                                                     epidermal cells, weakly
            root hair cell PART_OF root           -> its markers do NOT mark
                                                     "root"; a root contains
                                                     dozens of other cell types

        Rolling up `part_of` was tried and reverted. It registered every root
        cell type's markers against `root`, `plant organ`, `plant axis` and
        `portion of plant tissue` (16 ancestors for `root hair cell` instead of
        6, 19 instead of 5 for `pericycle cell`), which both invents evidence
        for container terms that no gene set actually marks and adds a Fisher
        test per invented label -- inflating the BH-FDR correction until
        marginal real calls drop out. `hotspot_M16` lost significance entirely
        that way.

        `part_of` *is* correct for deciding which candidates are redundant and
        how general a term is; see annotate.py, which uses ontology.py's
        default relation set.
        """
        entry = self.lookup[matched_label].copy()
        term = self.po[entry["id"]]
        entry["match"] = matched_label
        entry["parents"] = (
            [(t.id, t.name) for t in po_ancestors(entry["id"], self.po, relations=())]
            if rollup else []
        )
        return entry

    def match(self, celltype: str, fuzzy: bool = True, token_threshold: float = 0.5,
              cutoff: float = 0.8, rollup: bool = True):
        """Resolve one free-text cell type to a PO term: exact name/synonym, then
        token (Jaccard >= `token_threshold`), then difflib fuzzy (>= `cutoff`).

        Returns a dict with `id`, `name`, `match` (the ontology label matched)
        and `parents` (its is_a ancestors as (id, name) pairs, when `rollup`);
        all empty/None when nothing matches.
        """
        label = celltype.strip().lower()
        if label in self.lookup:
            return self._make_entry(label, rollup)

        token_candidate = self._best_token_match(label, threshold=token_threshold)
        if token_candidate:
            return self._make_entry(token_candidate, rollup)

        if fuzzy:
            matches = get_close_matches(label, self._labels, n=3, cutoff=cutoff)
            if matches:
                best = matches[0]
                # Accept a single candidate, or a clear winner over the runner-up.
                if len(matches) == 1 or (
                    SequenceMatcher(None, label, best).ratio()
                    - SequenceMatcher(None, label, matches[1]).ratio() > 0.1
                ):
                    return self._make_entry(best, rollup)

        return {"id": None, "name": None, "match": None, "parents": []}

    def standardize_file(self, filepath, column, outpath=None):
        """Add `<column>_po_id/_po_name/_po_match/_po_parents` columns to a marker
        table, writing it to `outpath` as CSV when given. Returns (table, outpath).
        """
        df = pd.read_csv(filepath, sep=None, engine="python")
        if column not in df.columns:
            raise KeyError(
                f"{filepath}: no column {column!r}; found {list(df.columns)}")
        results = []
        for val in df[column].fillna(""):
            std = self.match(val, rollup=True)
            results.append({
                column: val,
                column + "_po_id": std["id"],
                column + "_po_name": std["name"],
                column + "_po_match": std["match"],
                column + "_po_parents": ";".join(p[0] for p in std["parents"])})
        std_df = pd.DataFrame(results)
        merged = pd.concat([df, std_df.drop(columns=[column])], axis=1)
        if outpath is not None:
            merged.to_csv(outpath, index=False)
        return merged, outpath
