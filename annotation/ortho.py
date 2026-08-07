#!/usr/bin/env python3

import csv
import json
import logging
import os

from collections import defaultdict
from pathlib import Path

from shared.utils import run_cmd
from .genemap import build_all as build_genemaps
from .proteome import (
    cache_key,
    crosscheck_annotation,
    normalize_gene_id,
    prepare_all,
    validate_all,
)


class OrthoManager:
    """
    Runs OrthoFinder over the reference proteomes and exposes the resulting
    orthogroups as gene<->OG maps.

    The run is content-addressed: proteomes are first reduced to primary
    transcripts (see proteome.py), and the sha256 of that set of files is the
    run's identity. A run for the same species/proteome-version combination is
    therefore found again rather than repeated, and changing any proteome
    yields a new key rather than silently reusing the old orthogroups.

    Gene IDs are normalized when the primary-transcript FASTAs are written, so
    OrthoFinder emits normalized gene IDs and Orthogroups.tsv needs no cleaning
    here. Marker IDs must be passed through proteome.normalize_gene_id before
    being looked up.
    """

    def __init__(self, proteome_dict: dict, out_dir, marker_dict: dict = None,
                 gene_id_rules: dict = None, force_rerun: bool = False,
                 threads: int = 16, analysis_threads: int = 8,
                 stop_after_orthogroups: bool = True,
                 min_marker_coverage: float = 0.8,
                 min_marker_coverage_overrides: dict = None,
                 marker_gene_column: str = 'Genes',
                 annotation_dict: dict = None,
                 alias_attributes: dict = None,
                 processing: dict = None,
                 crosscheck: dict = None):
        self.processing = processing or {}
        self.crosscheck_cfg = crosscheck or {}
        self.proteome_dict = proteome_dict
        self.marker_dict = marker_dict or {}
        self.gene_id_rules = gene_id_rules or {}
        self.annotation_dict = annotation_dict or {}
        self.alias_attributes = alias_attributes or {}
        self.out_dir = Path(out_dir)
        self.force_rerun = force_rerun
        self.threads = threads
        self.analysis_threads = analysis_threads
        self.stop_after_orthogroups = stop_after_orthogroups
        self.min_marker_coverage = min_marker_coverage
        self.min_marker_coverage_overrides = min_marker_coverage_overrides or {}
        self.marker_gene_column = marker_gene_column
        self.logger = logging.getLogger('AnnotationLogging')

        self.prepared = None
        self.key = None
        self.validation = None
        self.genemaps = {}
        self.crosscheck = None
        self.gene_to_og = {}
        self.og_to_genes = {}

    # ------------------------------------------------------------------ setup

    def prepare(self):
        """
        Build primary-transcript proteomes, validate marker coverage, and
        compute this combination's cache key.

        Runs before any OrthoFinder invocation on purpose: both steps are
        cheap, and a bad proteome path or an unresolvable marker file should
        fail in seconds rather than after hours of compute.
        """
        if self.prepared is not None:
            return self.prepared

        self.prepared = prepare_all(
            self.proteome_dict,
            self.out_dir / 'primary_transcripts',
            gene_id_rules=self.gene_id_rules,
            processing=self.processing,
            logger=self.logger)

        self.key = cache_key(self.prepared)
        self.logger.info('[ANNOTATION] Proteome set key: %s', self.key[:12])

        # Annotation indexes are cached on their own source digest rather than
        # folded into the proteome cache key: swapping a GTF changes how
        # markers resolve, but not the OrthoFinder input, so the (expensive)
        # orthogroup run stays valid while the (cheap) marker mapping is redone.
        if self.annotation_dict:
            self.genemaps = build_genemaps(
                self.annotation_dict,
                self.out_dir / 'genemaps',
                alias_attributes=self.alias_attributes,
                logger=self.logger)

        # Covers every species with an annotation file, including the
        # proteome-only references and the target -- which validate_all below
        # skips entirely, since it only looks at species that have markers.
        if self.genemaps:
            self.crosscheck = crosscheck_annotation(
                self.prepared, self.genemaps,
                self.out_dir / 'validation' / self.key[:12],
                min_recall=self.crosscheck_cfg.get('min_recall', 0.5),
                warn_recall=self.crosscheck_cfg.get('warn_recall', 0.99),
                logger=self.logger)

        if self.marker_dict:
            self.validation = validate_all(
                self.marker_dict, self.prepared,
                self.out_dir / 'validation' / self.key[:12],
                min_coverage=self.min_marker_coverage,
                min_coverage_overrides=self.min_marker_coverage_overrides,
                gene_column=self.marker_gene_column,
                genemaps=self.genemaps,
                logger=self.logger)
        else:
            self.logger.warning('[ANNOTATION] No marker files supplied; skipping validation')

        return self.prepared

    def resolve_marker_tokens(self, species: str, tokens) -> dict:
        """
        Map marker tokens to the gene IDs written into the OrthoFinder input,
        for one species. Returns {token -> [gene IDs]}, omitting tokens that
        resolve to nothing.

        This is the bridge between a marker table and orthogroup space, and it
        is deliberately the *same* call `validate_markers` makes. A rescue that
        validation counted but enrichment did not apply would raise the
        reported coverage while changing no annotation at all -- the number
        would improve and the result would not.

        Passed to `OMGBuilder` as a callable so the target species resolves
        through it too: its per-sample marker tables are read there, not here,
        and are not known at validation time.
        """
        self.prepare()
        if species not in self.prepared:
            # Every marker would resolve to nothing and the species would just
            # cast no votes -- a silent, total failure that looks like a
            # species with weak markers. Name it instead.
            raise KeyError(
                f"{species} has no prepared proteome; cannot resolve its marker IDs. "
                f"Prepared: {sorted(self.prepared)}")
        genes = self.prepared[species]['genes']
        genemap = self.genemaps.get(species)
        resolved = {}
        for token in tokens:
            if token in genes:
                resolved[token] = [token]
            elif genemap is not None:
                status, hits = genemap.resolve(token, genes)
                if status == 'alias':
                    resolved[token] = sorted(hits)
        return resolved

    @property
    def run_dir(self):
        """Content-addressed directory holding this proteome combination's run."""
        if self.key is None:
            self.prepare()
        return self.out_dir / 'orthofinder' / self.key[:12]

    # ------------------------------------------------------------- orthofinder

    def run_orthofinder(self):
        """Prepare the proteomes, run OrthoFinder if the cache misses, and parse the result.

        Returns:
            Path: The OrthoFinder results directory.
        """
        self.prepare()
        run_dir = self.run_dir

        existing = self._find_cached_result(run_dir)
        if existing and not self.force_rerun:
            self.logger.info(
                '[ANNOTATION] Reusing OrthoFinder results for this proteome set: %s', existing)
            return existing
        if existing and self.force_rerun:
            self.logger.info('[ANNOTATION] --force given; ignoring cached results at %s', existing)

        indir = run_dir / 'input'
        indir.mkdir(parents=True, exist_ok=True)
        for species, stats in self.prepared.items():
            link = indir / f'{species}.fa'
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(Path(stats['primary_transcripts']).resolve(), link)

        # OrthoFinder refuses to write into an existing -o directory.
        outdir = run_dir / 'run'
        if outdir.exists():
            suffix = 1
            while (run_dir / f'run.{suffix}').exists():
                suffix += 1
            outdir = run_dir / f'run.{suffix}'

        cmd = ['orthofinder', '-f', str(indir), '-o', str(outdir),
               '-t', str(self.threads), '-a', str(self.analysis_threads)]
        if self.stop_after_orthogroups:
            # Only Orthogroups.tsv is consumed downstream; -og skips the gene
            # tree / species tree inference, which is the bulk of the runtime.
            cmd.append('-og')

        self.logger.info('[ANNOTATION] Running: %s', ' '.join(cmd))
        run_cmd(cmd, logfile='AnnotationLogging')

        self._write_cache_key(run_dir)
        results = self._find_cached_result(run_dir)
        if results is None:
            raise FileNotFoundError(
                f'OrthoFinder reported success but no Orthogroups.tsv was found under {outdir}')
        self.logger.info('[ANNOTATION] OrthoFinder results: %s', results)
        return results

    def _write_cache_key(self, run_dir):
        """Record what this run was built from, so a cached hit is auditable."""
        payload = {
            'key': self.key,
            'species': {
                sp: {
                    'source': st['source'],
                    'n_proteins': st['n_proteins'],
                    'n_genes': st['n_genes'],
                }
                for sp, st in sorted(self.prepared.items())
            },
            'gene_id_rules': self.gene_id_rules,
            'processing': self.processing,
            'stop_after_orthogroups': self.stop_after_orthogroups,
        }
        (run_dir / 'cache_key.json').write_text(json.dumps(payload, indent=2, default=str))

    def _find_cached_result(self, run_dir):
        """
        Newest completed OrthoFinder result under this key's run directory.

        A result counts only if Orthogroups.tsv exists -- a killed run leaves
        the Results_* directory behind without it, and treating that as a hit
        would skip the rerun forever.
        """
        run_dir = Path(run_dir)
        if not run_dir.exists():
            return None
        candidates = [d for d in run_dir.glob('run*/Results_*')
                      if (d / 'Orthogroups' / 'Orthogroups.tsv').exists()]
        if not candidates:
            return None
        return max(candidates, key=os.path.getmtime)

    # ------------------------------------------------------------------ parsing

    def _resolve_results_dir(self, results_dir):
        """Find the actual results directory inside an OrthoFinder output tree.

        Args:
            results_dir (Path): The directory OrthoFinder was pointed at.

        Returns:
            Path: The directory holding Orthogroups.tsv.
        """
        if results_dir is not None:
            return Path(results_dir)
        found = self._find_cached_result(self.run_dir)
        if found is None:
            raise FileNotFoundError('No OrthoFinder results found. Run run_orthofinder() first.')
        self.logger.info('[ANNOTATION] Parsing results from run: %s', found)
        return found

    def _iter_orthogroups(self, results_dir):
        """Yield (orthogroup id, {species: [gene ids]}) from Orthogroups.tsv.

        Args:
            results_dir (Path): The resolved results directory.

        Yields:
            tuple: (og_id (str), members (dict)).
        """
        og_file = Path(results_dir) / 'Orthogroups' / 'Orthogroups.tsv'
        with og_file.open() as handle:
            for row in csv.DictReader(handle, delimiter='\t'):
                og = row['Orthogroup']
                for species, genes in row.items():
                    if species == 'Orthogroup' or not genes or not genes.strip():
                        continue
                    # Species name is the input filename stem, which prepare()
                    # writes as the species key itself -- no remapping needed.
                    yield og, species, [g.strip() for g in genes.split(',') if g.strip()]

    def parse_gene_to_og(self, results_dir=None):
        """Build {species: {gene id: orthogroup id}}.

        Args:
            results_dir (Path): The resolved results directory.

        Returns:
            dict: The gene-to-orthogroup index.
        """
        results_dir = self._resolve_results_dir(results_dir)
        gene_to_og = {}
        for og, _species, genes in self._iter_orthogroups(results_dir):
            for gene in genes:
                gene_to_og[gene] = og
        self.gene_to_og = gene_to_og
        self.logger.info('[ANNOTATION] Parsed %d gene-to-OG mappings', len(gene_to_og))
        return gene_to_og

    def parse_og_to_genes(self, results_dir=None):
        """Build {orthogroup id: {species: set[gene id]}}.

        Args:
            results_dir (Path): The resolved results directory.

        Returns:
            dict: The orthogroup-to-gene index.
        """
        results_dir = self._resolve_results_dir(results_dir)
        og_to_genes = defaultdict(lambda: defaultdict(list))
        for og, species, genes in self._iter_orthogroups(results_dir):
            og_to_genes[og][species].extend(genes)
        self.og_to_genes = {og: dict(sp) for og, sp in og_to_genes.items()}
        self.logger.info('[ANNOTATION] Parsed %d OG entries', len(self.og_to_genes))
        return self.og_to_genes

    def get_og_for_gene(self, gene):
        """Orthogroup for a gene ID. The ID is normalized here, so callers may
        pass a marker file's raw spelling."""
        if not self.gene_to_og:
            raise RuntimeError('gene_to_og is empty. Run parse_gene_to_og() first.')
        return self.gene_to_og.get(normalize_gene_id(gene))

    def get_genes_for_og(self, og, species=None):
        """The genes one species contributes to one orthogroup.

        Args:
            og (str): Orthogroup id.
            species (str): Species name.

        Returns:
            set[str]: Gene ids.
        """
        if not self.og_to_genes:
            raise RuntimeError('og_to_genes is empty. Run parse_og_to_genes() first.')
        if og not in self.og_to_genes:
            return None
        if species:
            return self.og_to_genes[og].get(species, [])
        return self.og_to_genes[og]
