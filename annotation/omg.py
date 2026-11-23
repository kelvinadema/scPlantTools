#!/usr/bin/env python3

import csv
import logging

from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from .proteome import normalize_gene_id, normalize_tissue

# Marker tables come from two sources with different layouts: the reference
# tables ('<Species>_high_confidence.txt') and this pipeline's own per-sample
# output from scanpy's rank_genes_groups.
REFERENCE_COLUMNS = {'cluster': 'Celltypes', 'gene': 'Genes'}
TARGET_COLUMNS = {'cluster': 'clusterName', 'gene': 'gene'}


class OMGBuilder:
    """
    Projects marker genes into orthogroup space (OMGs) and tests each target
    cluster against each reference cell type for enrichment.

    Follows Chau et al. 2025 (doi:10.1038/s41467-024-55755-0): marker gene
    names are mapped to their orthogroup, and a one-sided Fisher exact test on
    [[X, Y], [Z, W]] asks whether two clusters' OMG sets overlap more than
    chance, with BH-FDR across the tests.
    """

    def __init__(self, gene_to_og, og_to_gene, marker_dict, target_marker_file,
                 standardizer, target_species='Medicago_truncatula',
                 reference_columns=None, target_columns=None, q_threshold=0.05,
                 marker_resolver=None, tissue_column='Tissues'):
        self.gene_to_og = gene_to_og
        self.og_to_gene = og_to_gene
        self.marker_files = marker_dict
        self.target_species = target_species
        self.standardizer = standardizer
        self.reference_columns = reference_columns or REFERENCE_COLUMNS
        self.target_columns = target_columns or TARGET_COLUMNS
        self.q_threshold = q_threshold
        # OrthoManager.resolve_marker_tokens; see _resolve_genes. Optional:
        # without it every token maps to itself.
        self.marker_resolver = marker_resolver
        # {po_id -> {tissues it was observed in}}, built while the reference
        # tables are read and consumed by annotate.py's post-hoc candidate
        # filter. Marker rows themselves are never filtered -- see
        # _record_tissues for why.
        self.tissue_column = tissue_column
        self.term_tissues = {}
        self.logger = logging.getLogger('AnnotationLogging')

        self.ref_markers = {}
        self.ref_omgs = {}
        self.target_omgs = {}
        self.target_markers = (
            self._load_markers(target_marker_file, self.target_columns,
                               standardize=False, species=target_species)
            if target_marker_file else {}
        )

    # ------------------------------------------------------------------ markers

    def _read_rows(self, path, delimiter):
        """csv.DictReader rather than str.split -- cell-type names and
        multi-alias gene cells contain commas, which are quoted on write and
        would be split back apart by a naive split."""
        with open(path, newline='') as handle:
            yield from csv.DictReader(handle, delimiter=delimiter)

    def _resolve_genes(self, species, tokens):
        """
        Marker tokens -> the gene IDs OrthoFinder emitted, as {token: [genes]}.

        Without a resolver a token maps to itself: correct for a marker file
        naming locus IDs, and a silent drop for one naming gene symbols. The resolver consults the species' annotation and
        recovers the symbol case -- 283 markers across the eight reference
        species here, 263 of them Arabidopsis.

        A symbol occasionally names more than one locus (ACO2 -> AT1G62380,
        AT4G26970). Both are kept: markers are being projected into orthogroup
        space, which is set-valued anyway, and refusing the marker outright
        would discard real evidence to avoid a choice that does not have to be
        made. Measured at 2 markers across every reference species, so the
        policy costs almost nothing either way.
        """
        if self.marker_resolver is None:
            return {t: [t] for t in tokens}
        return self.marker_resolver(species, tokens)

    def _record_tissues(self, rows, cluster_column):
        """
        Record which tissues each PO term was observed in, for the *post-hoc*
        candidate filter in annotate.py.

        Marker rows are **not** filtered by tissue. Doing so shrinks the
        orthogroup universe `_species_universe` computes, and Fisher's expected
        overlap scales as 1/N, so it weakens every test -- including for cell
        types whose own markers were untouched. Measured on `MtTip_noHVG`:
        identical overlap went from q=2.1e-04 to 1.2e-02 (root epidermis) and
        3.3e-15 to 4.5e-11 (root meristem) purely from filtering, and 11 of 12
        shared calls got worse. So every marker votes, and tissue relevance is
        applied to the resulting candidates instead.

        A term is recorded against a tissue if it, or any term it subsumes, was
        seen there -- the same `[po_id] + parents` expansion the markers get,
        so `plant epidermis` counts as root-relevant because `root epidermis` is.
        """
        for row in rows:
            tissue = normalize_tissue(row.get(self.tissue_column))
            if not tissue:
                continue
            po_id = (row.get(f'{cluster_column}_po_id') or '').strip()
            parents = (row.get(f'{cluster_column}_po_parents') or '').strip()
            for label in [po_id] + [p.strip() for p in parents.split(';')]:
                if label:
                    self.term_tissues.setdefault(label, set()).add(tissue)

    def _load_markers(self, file_path, columns, standardize=True, species=None):
        """
        Read a marker table into {cluster label -> set of orthogroup-ready gene IDs}.

        Reference tables are standardized to Plant Ontology terms first and a
        gene is registered against its term *and* that term's ancestors, so a
        marker for a specific cell type also supports the broader types it
        belongs to. Target clusters are used as-is -- they are this pipeline's
        own cluster names, not ontology terms.
        """
        cluster_column, gene_column = columns['cluster'], columns['gene']

        if standardize:
            use_path = Path(file_path).with_name(Path(file_path).stem + '_standardized.csv')
            if use_path.exists():
                self.logger.info('[ANNOTATION] Reusing standardized markers: %s', use_path)
            else:
                self.logger.info('[ANNOTATION] Standardizing markers -> %s', use_path)
                self.standardizer.standardize_file(
                    filepath=file_path, column=cluster_column, outpath=use_path)
            delimiter = ','
        else:
            use_path, delimiter = file_path, '\t'

        # Materialized so every token can be resolved in one call rather than
        # one per row; marker tables are a few thousand rows.
        rows = list(self._read_rows(use_path, delimiter))
        # Reference tables only: `standardize` is what distinguishes them from
        # the target's own marker table, which carries no tissue column.
        if standardize:
            self._record_tissues(rows, cluster_column)
        # Gene IDs are normalized here because gene_to_og is keyed on the
        # normalized form written into the OrthoFinder input (see
        # proteome.normalize_gene_id). A raw marker ID never matches.
        tokens = {normalize_gene_id(g)
                  for row in rows
                  for g in (row.get(gene_column) or '').split(',') if g.strip()}
        resolved = self._resolve_genes(species, tokens)

        n_alias = sum(1 for t, v in resolved.items() if v != [t])
        if n_alias:
            self.logger.info(
                '[ANNOTATION] %s: %d marker(s) resolved to a different gene ID via the '
                'annotation index', species, n_alias)
        dropped = len(tokens) - len(resolved)
        if dropped:
            self.logger.info(
                '[ANNOTATION] %s: %d of %d marker token(s) map to no proteome gene',
                species, dropped, len(tokens))

        markers = defaultdict(set)
        unmapped = Counter()
        for row in rows:
            raw = (row.get(gene_column) or '').strip()
            if not raw:
                continue
            genes = set()
            for token in raw.split(','):
                if token.strip():
                    genes.update(resolved.get(normalize_gene_id(token), ()))
            if not genes:
                continue

            if standardize:
                po_id = (row.get(f'{cluster_column}_po_id') or '').strip()
                parents = (row.get(f'{cluster_column}_po_parents') or '').strip()
                if not po_id:
                    # A cell type the standardizer could not place in the
                    # ontology is dropped, not passed through as raw text: a raw
                    # label ('Phloem parenchyma') as `po_id` has no parents to
                    # collapse against, no descendants to judge generality by,
                    # and fails validation in a decision file. (Older outputs
                    # could carry such raw labels as candidates.)
                    unmapped[(row.get(cluster_column) or '').strip()] += 1
                    continue
                labels = [po_id]
                labels += [p.strip() for p in parents.split(';') if p.strip()]
            else:
                labels = [(row.get(cluster_column) or '').strip()]

            for label in labels:
                if label:
                    markers[label] |= genes

        if unmapped:
            self.logger.warning(
                '[ANNOTATION] %s: %d marker row(s) dropped -- cell type not in the '
                'ontology: %s', species, sum(unmapped.values()),
                ', '.join(f'{k} (x{v})' for k, v in sorted(unmapped.items())[:6]))
        return dict(markers)

    def get_reference_markers(self):
        """Load every reference marker table: {species: {po_id: set of gene ids}}."""
        for species, file_path in self.marker_files.items():
            self.ref_markers[species] = self._load_markers(
                file_path, self.reference_columns, standardize=True, species=species)
        self.logger.info('[ANNOTATION] Loaded reference markers for %d species',
                         len(self.ref_markers))
        return self.ref_markers

    # --------------------------------------------------------------------- OMGs

    def _to_omgs(self, markers):
        """Marker genes -> the set of orthogroups they belong to. A set, not a
        list: several markers routinely share an orthogroup, and the Fisher
        test counts distinct OMGs."""
        return {
            cluster: {og for og in (self.gene_to_og.get(g) for g in genes) if og}
            for cluster, genes in markers.items()
        }

    def build_omgs(self):
        """Project every species' markers into orthogroup space (into
        `ref_omgs` and `target_omgs`; returns nothing)."""
        if not self.ref_markers:
            self.get_reference_markers()

        self.ref_omgs = {sp: self._to_omgs(clusters) for sp, clusters in self.ref_markers.items()}
        self.target_omgs = self._to_omgs(self.target_markers)

        n_ref_types = len({ct for sp in self.ref_omgs.values() for ct in sp})
        self.logger.info('[ANNOTATION] Built reference OMGs for %d cell types', n_ref_types)
        self.logger.info('[ANNOTATION] Built target OMGs for %d clusters', len(self.target_omgs))

        empty = [sp for sp, clusters in self.ref_omgs.items()
                 if not any(clusters.values())]
        if empty:
            self.logger.warning(
                '[ANNOTATION] No marker gene mapped to an orthogroup for: %s -- '
                'check that marker IDs match the proteome gene IDs', ', '.join(empty))
        if not any(self.target_omgs.values()):
            self.logger.warning('[ANNOTATION] No target marker mapped to an orthogroup')

    def get_target_omgs(self):
        """The target's orthogroup set per geneset."""
        return self.target_omgs

    def get_ref_omgs(self):
        """The reference orthogroup sets: {species: {po_id: set of orthogroup ids}}."""
        return self.ref_omgs

    def get_ref_genes(self):
        """Target-species genes in each reference cell type's orthogroups:
        {species: {po_id: sorted gene ids}}."""
        return {sp: {ct: self._map_og_to_target_genes(ogs) for ct, ogs in clusters.items()}
                for sp, clusters in self.ref_omgs.items()}

    def get_target_genes(self):
        """Target-species genes in each geneset's orthogroups: {geneset: sorted gene ids}."""
        return {ct: self._map_og_to_target_genes(ogs) for ct, ogs in self.target_omgs.items()}

    def _map_og_to_target_genes(self, ogs):
        """The target-species genes in the given orthogroups, sorted."""
        genes = set()
        for og in ogs:
            genes.update(self.og_to_gene.get(og, {}).get(self.target_species, []))
        return sorted(genes)

    # --------------------------------------------------------------- enrichment

    @staticmethod
    def _contingency(ref_ogs, tgt_ogs, universe_size):
        """The 2x2 table for one reference/target orthogroup overlap within a
        universe of `universe_size` orthogroups; returns (table, X, Y, Z, W) =
        (table, overlap, reference-only, target-only, neither)."""
        X = len(ref_ogs & tgt_ogs)
        Y = len(ref_ogs) - X
        Z = len(tgt_ogs) - X
        # Non-negative by construction: the universe contains both sets.
        W = universe_size - X - Y - Z
        return [[X, Y], [Z, W]], X, Y, Z, W

    def _species_universe(self, species):
        """
        Number of OMGs in play for one target-vs-reference-species comparison.

        Pairwise, per the paper, which tests one species' cluster against
        another's. Pooling every reference species into a single universe would
        make W -- and therefore every p-value -- depend on how many species
        happened to be included in the run.
        """
        ref = set().union(*self.ref_omgs[species].values()) if self.ref_omgs[species] else set()
        tgt = set().union(*self.target_omgs.values()) if self.target_omgs else set()
        return len(ref | tgt)

    def enrich_fisher(self, qval_thresh=None):
        """
        One-sided Fisher exact test per (reference species, reference cell type,
        target cluster), BH-FDR corrected across all tests.

        The correction is applied over the whole set rather than per species
        pair -- more conservative, and it keeps `significant` comparable across
        species within one sample.
        """
        qval_thresh = self.q_threshold if qval_thresh is None else qval_thresh
        if not self.ref_omgs or not self.target_omgs:
            self.build_omgs()

        results = defaultdict(lambda: defaultdict(dict))
        pvals = []
        # Target sets are reused across every reference cell type, so build the
        # list once instead of per test.
        target_items = list(self.target_omgs.items())

        for species, celltypes in self.ref_omgs.items():
            universe_size = self._species_universe(species)
            for ref_ct, ref_ogs in celltypes.items():
                for tgt_ct, tgt_ogs in target_items:
                    table, X, Y, Z, W = self._contingency(ref_ogs, tgt_ogs, universe_size)
                    p = fisher_exact(table, alternative='greater')[1]
                    results[species][ref_ct][tgt_ct] = {
                        'table': table, 'X': X, 'Y': Y, 'Z': Z, 'W': W, 'pval': p}
                    pvals.append(p)

        if pvals:
            reject, qvals, _, _ = multipletests(pvals, method='fdr_bh')
            idx = 0
            for ref_map in results.values():
                for tgt_map in ref_map.values():
                    for stats in tgt_map.values():
                        stats['qval'] = float(qvals[idx])
                        stats['significant'] = bool(reject[idx] and qvals[idx] < qval_thresh)
                        idx += 1
        self.logger.info('[ANNOTATION] Ran %d Fisher tests (q threshold %.3g)',
                         len(pvals), qval_thresh)
        return {sp: {ct: dict(tgt) for ct, tgt in ref.items()} for sp, ref in results.items()}
