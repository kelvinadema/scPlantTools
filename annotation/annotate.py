import logging
import math

from collections import defaultdict, namedtuple
from pathlib import Path

import pandas as pd

from .ontology import descendant_index
from .proteome import normalize_tissue

from .plot import AnnotationPlotter

# Positional tuples were indexed as cand[0]..cand[4] across four functions.
# `weighted_species` is the phylogenetically-corrected support that drives the
# score; `n_species` stays the raw count the classification and reports use.
Candidate = namedtuple('Candidate', 'label n_species weighted_species total_X best_q score')


class scAnnotator:
    """
    Turns per-cluster enrichment results into a cell-type call.

    Each (reference species, reference cell type) hit for a target cluster is
    aggregated by cell type: how many species support it, the summed overlap,
    and the best q-value. Candidates are ranked, then parent/child redundancy
    in the Plant Ontology is collapsed.
    """

    def __init__(self, omg, out_dir, outfn, standardizer, logger=None, clade_weights=None):
        self.omg = omg
        self.standardizer = standardizer
        self.out_dir = Path(out_dir)
        self.outfn = outfn
        self.logger = logger or logging.getLogger('AnnotationLogging')
        # Phylogenetic weighting of the species vote (annotation/phylo.py).
        # None => every species weighs 1, i.e. the score is exactly n_species as
        # before, so the feature is fully opt-in and the un-configured path is a
        # no-op rather than a different number.
        self.clade_weights = clade_weights

        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Enrichment is deliberately not run here: a constructor that spends
        # minutes on a Fisher sweep cannot be constructed to inspect.
        self._enrichment = None

    @property
    def enrichment(self):
        """Test every (geneset, reference cell type) pair for orthogroup enrichment.

        Returns:
            pandas.DataFrame: One row per tested pair, with counts, p and q.
        """
        if self._enrichment is None:
            self._enrichment = self.omg.enrich_fisher()
        return self._enrichment

    def _collect_hits(self):
        """Group the significant tests by geneset into candidate cell types.

        Returns:
            dict: {geneset: list of candidate records}.
        """
        per_target = defaultdict(list)
        for sp, ref_map in self.enrichment.items():
            for ref_ct, tgt_map in ref_map.items():
                for tgt_ct, stats in tgt_map.items():
                    per_target[tgt_ct].append({
                        'species': sp,
                        'ref_cluster': ref_ct,
                        'qval': float(stats.get('qval', 1.0)),
                        'X': int(stats.get('X', 0))})
        return per_target

    def _aggregate_hits(self, hits, q_thresh, X_min_filter):
        """Collapse hits for one target cluster by reference cell type."""
        agg = {}
        for hit in hits:
            if hit['qval'] > q_thresh or hit['X'] < X_min_filter:
                continue
            entry = agg.setdefault(hit['ref_cluster'],
                                   {'species': set(), 'total_X': 0, 'best_q': hit['qval']})
            entry['species'].add(hit['species'])
            entry['total_X'] += hit['X']
            entry['best_q'] = min(entry['best_q'], hit['qval'])
        return agg

    def _score_candidates(self, agg):
        """
        Rank cell types by -log10(best q) * sqrt(total overlap) * weighted species.

        Rewards agreement across independent reference species over a single
        very strong hit, which is the point of using several references. The
        species term is phylogenetically weighted (phylo.py): four grasses count
        for less than four different families, because same-clade agreement is
        not four independent observations. With no `clade_weights` configured the
        weight is the raw count and this is the original score exactly.

        Significance is *not* affected -- the candidates here already passed the
        Fisher/BH-FDR filter upstream, and `best_q` is unchanged. Only the order,
        and hence the top call pre-filled for the reviewer, can move.
        """
        candidates = []
        for lbl, info in agg.items():
            n_species = len(info['species'])
            weighted = (self.clade_weights.weighted_species(info['species'])
                        if self.clade_weights is not None else float(n_species))
            candidates.append(Candidate(
                label=lbl, n_species=n_species, weighted_species=weighted,
                total_X=info['total_X'], best_q=info['best_q'],
                score=-math.log10(info['best_q'] + 1e-300)
                * (info['total_X'] ** 0.5) * weighted))
        return sorted(candidates,
                      key=lambda c: (-c.score, -c.total_X, -c.weighted_species, c.best_q))

    def _classify_candidate(self, cand, S_min, X_total_min, q_single_strict, X_single_min):
        """Decide whether one candidate is a call, a weak call, or rejected.

        Args:
            cand (dict): The candidate record.
            S_min (float): Minimum weighted species support.
            X_total_min (int): Minimum total orthogroup overlap.
            q_single_strict (float): q a single-species call must clear.
            X_single_min (int): Overlap a single-species call must clear.

        Returns:
            str: The verdict.
        """
        if cand.n_species >= S_min and cand.total_X >= X_total_min:
            return 'consensus'
        if cand.best_q <= q_single_strict and cand.total_X >= X_single_min:
            return 'single'
        return 'ambiguous'

    def _get_dataset_children(self, label_id, ontology, dataset_terms):
        """Ontology children of `label_id` that this dataset actually has terms for.

        Args:
            label_id (str): Parent PO id.
            ontology: The loaded ontology.
            dataset_terms (set[str]): PO ids present in this dataset.

        Returns:
            set[str]: Child PO ids.
        """
        # Follows part_of as well as is_a (see ontology.py). Using only
        # subclasses() undercounts how much a term subsumes -- and that count
        # is what `too_general` thresholds on.
        if label_id not in ontology:
            return []
        return sorted(self._descendants.get(label_id, set()) & dataset_terms)

    def _po_name(self, po_id, ontology):
        """Readable name for a PO id, falling back to the id itself.

        Args:
            po_id (str): The ontology id.
            ontology: The loaded ontology.

        Returns:
            str: Name or id.
        """
        if po_id and po_id in ontology:
            return ontology[po_id].name
        return None

    def _significant_children(self, child_terms, candidates, q_thresh, X_min_filter):
        """Which of a term's children are themselves significantly enriched.

        Used to prefer a specific call over its parent when the evidence supports it.

        Args:
            child_terms (set[str]): Candidate child PO ids.
            candidates (list[dict]): This geneset's candidate records.
            q_thresh (float): Significance threshold.
            X_min_filter (int): Minimum overlap.

        Returns:
            list[dict]: The significant children.
        """
        strong = {c.label for c in candidates
                  if c.best_q <= q_thresh and c.total_X >= X_min_filter}
        return [child for child in child_terms if child in strong]

    def _tissue_match(self, po_id, wanted):
        """
        Was this term ever observed in the tissue(s) of interest?

        Applied to candidates, never to the markers that produced them -- see
        OMGBuilder._record_tissues for the measurement showing why filtering
        markers weakens every test instead of just narrowing the answer.

        A term with no recorded tissue at all is kept: absence of evidence
        here means the term came from a marker file with no tissue column, not
        that it belongs to the wrong tissue.
        """
        if not wanted:
            return True
        seen = self.omg.term_tissues.get(po_id)
        if not seen:
            return True
        return bool(seen & wanted)

    def consensus_annotation(self, q_thresh=0.05, X_single_min=8, X_min_filter=3, S_min=2,
                             q_single_strict=1e-3, X_total_min=6, return_top_n=10,
                             tissue=None, max_dataset_children=15, save=True):
        """
        `tissue` and `max_dataset_children` annotate the candidates rather than
        removing them: every row is written with `tissue_match` and
        `too_general` flags, and the decision template picks the best row that
        passes both. Keeping the rejected rows means the evidence behind a
        judgement stays inspectable instead of vanishing from the output.
        """
        ontology = self.standardizer.po
        # Built once: the ancestor walk is O(terms) per query and every
        # candidate of every cluster asks for it.
        self._descendants = descendant_index(ontology)
        dataset_terms = {ct for sp in self.omg.get_ref_omgs().values() for ct in sp}
        wanted_tissues = {normalize_tissue(t) for t in
                          ([tissue] if isinstance(tissue, str) else (tissue or []))}

        rows = []
        for tgt_ct, hits in self._collect_hits().items():
            agg = self._aggregate_hits(hits, q_thresh, X_min_filter)
            candidates = self._score_candidates(agg)

            for cand in candidates[:return_top_n]:
                dataset_children = self._get_dataset_children(cand.label, ontology, dataset_terms)
                rows.append({
                    'cluster': tgt_ct,
                    'po_id': cand.label,
                    'po_name': self._po_name(cand.label, ontology),
                    'support_species': cand.n_species,
                    'support_species_list': ';'.join(sorted(agg[cand.label]['species'])),
                    # Phylogenetically-corrected support that drove the score;
                    # equals support_species when no clade weighting is configured,
                    # so an old-vs-new diff is visible in the table itself.
                    'weighted_species': round(cand.weighted_species, 3),
                    'n_clades': (self.clade_weights.profile(agg[cand.label]['species'])['n_clades']
                                 if self.clade_weights is not None else cand.n_species),
                    'total_X': cand.total_X,
                    'best_q': cand.best_q,
                    'score': cand.score,
                    'decision': self._classify_candidate(
                        cand, S_min, X_total_min, q_single_strict, X_single_min),
                    'is_parent': cand.label in ontology
                    and any(ontology[cand.label].subclasses(with_self=False)),
                    'dataset_children': ';'.join(dataset_children),
                    'significant_children': ';'.join(self._significant_children(
                        dataset_children, candidates, q_thresh, X_min_filter)),
                    # How many of the cell types we actually have evidence for
                    # this term subsumes. A term covering most of them says
                    # almost nothing -- 'portion of plant tissue' subsumes 34.
                    'n_dataset_children': len(dataset_children),
                    'too_general': len(dataset_children) > max_dataset_children,
                    'tissues': ';'.join(sorted(self.omg.term_tissues.get(cand.label, ()))),
                    'tissue_match': self._tissue_match(cand.label, wanted_tissues)})

        df = self._collapse_terms(pd.DataFrame(rows), ontology)

        if save and self.out_dir:
            outpath = self.out_dir / '{}_significant_annotations.tsv'.format(self.outfn)
            df.to_csv(outpath, sep='\t', index=False)
            self.logger.info('[ANNOTATION] Full significant hits written to %s', outpath)

        if df.empty or 'cluster' not in df.columns:
            self.logger.warning('[ANNOTATION] No significant annotations for %s', self.outfn)
            return df

        AnnotationPlotter(df, ontology, self.out_dir, self.logger).plotAnnotation()
        return df

    def _collapse_terms(self, df, ontology, cluster_col='cluster', po_col='po_id',
                        support_col='support_species'):
        """
        Resolve parent/child redundancy within a cluster, keeping one term.

        Two things were wrong here and both showed up in real output.

        **It only followed `is_a`.** `ontology[term].subclasses()` ignores
        `part_of`, which is how PO models anatomical containment, so
        `root endodermis part_of root cortex` was invisible and both survived.
        It now walks the same relation set as the marker rollup (ontology.py),
        so a term and something it is part of can no longer both be reported.

        **It kept both whenever both were supported.** The old rule collapsed
        only when exactly one of the pair cleared `min_support`, on the
        reasoning that two well-evidenced levels are each informative. In
        practice that produced exactly the redundancy a reader objects to --
        `cortex` beside `root cortex`, `endodermis` beside `root endodermis`,
        both with the same support and near-identical scores, for the same
        module.

        The rule now always resolves the pair. The **child wins ties**, because
        a child implies its parent -- reporting `root endodermis` already tells
        you it is endodermis, while the reverse is not true, so the specific
        term is strictly more informative at equal evidence. The parent only
        wins when it is supported by *more* species, which is the one case
        where the general call rests on evidence the specific one lacks.

        The losing term is not lost: it stays in the per-cluster candidate list
        that the decision template surfaces as `alternatives`.
        """
        if df.empty or po_col not in df.columns:
            return df

        keep_frames = []
        for _cluster, subdf in df.groupby(cluster_col):
            keep = set(subdf[po_col].unique())
            support = dict(zip(subdf[po_col], subdf[support_col]))

            for term in list(keep):
                if term not in keep or term not in ontology:
                    continue
                for child in self._descendants.get(term, set()) & keep:
                    if child not in keep:
                        continue
                    # Child wins on >=; the parent needs strictly more species.
                    if support.get(child, 0) >= support.get(term, 0):
                        keep.discard(term)
                        break
                    keep.discard(child)

            keep_frames.append(subdf[subdf[po_col].isin(keep)])

        return pd.concat(keep_frames, ignore_index=True) if keep_frames else df
