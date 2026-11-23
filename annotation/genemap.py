#!/usr/bin/env python3
"""
Gene alias and biotype index, built from a species' GTF/GFF3 annotation.

Marker tables and proteome FASTAs disagree about how to name a gene. Marker
files routinely write a *symbol* (AAP4, WRKY36, CYP73A11) while the proteome
header carries only a locus ID (AT1G77380), so a direct comparison drops the
marker even though the gene is present. The annotation file is the only place
that carries both, so it is the bridge between them.

It also answers the other half of the question. A marker naming an rRNA, a
lincRNA or a mitochondrial gene is *expected* to be absent from a proteome,
which without biotypes is indistinguishable from a genuine mismatch. The
`gene_biotype` attribute separates the two, which is what lets marker coverage
be measured against the genes that could have matched.

Both jobs are served by one pass over the annotation, producing three maps:

    canonical  {normalized gene ID}                      -- genes the annotation knows
    aliases    {normalized alias -> {normalized gene ID}} -- symbol/synonym -> locus
    biotypes   {normalized gene ID -> biotype}            -- 'protein_coding', 'lincRNA', ...

Nothing here assumes a species has any of it. Two of the annotation files in
this project (Oryza's transcript-only IRGSP GTF, Lotus's bare-`ID=` PGSB GFF3)
carry neither names nor biotypes; they yield an index with an empty alias map
and resolution degrades to direct comparison.
"""

import gzip
import json
import logging
import re

from pathlib import Path

from .proteome import CODING_BIOTYPES, file_digest, normalize_gene_id

# Bumped when parsing changes in a way that would alter a cached index, so a
# stale cache is rebuilt rather than silently reused. Paired with the source
# file's digest in the cache header.
PARSER_VERSION = 1

# Attributes consulted for an alias, in order. `locus_tag` and `Name` are
# frequently just the gene ID again (Gossypium, Manihot) -- those collapse to
# nothing once normalized and are dropped, so listing them costs nothing.
DEFAULT_ALIAS_ATTRIBUTES = ('gene_name', 'Name', 'gene_synonym', 'locus_tag', 'Alias')

# Attributes naming the gene itself, in order of preference.
_GENE_ID_ATTRIBUTES = ('gene_id', 'ID')
# On a transcript-level line the gene is the *parent*, not the record's own ID.
_TRANSCRIPT_PARENT_ATTRIBUTES = ('gene_id', 'Parent')

_TRANSCRIPT_FEATURES = frozenset({'transcript', 'mRNA'})

# GFF3 IDs are namespaced ('ID=gene:Zm00001d027230', 'Parent=gene:X').
_ID_PREFIX = re.compile(r'^(?:gene|transcript|mRNA):', re.IGNORECASE)

# One GTF `key "value"` pair. `[^"]*` spans any ';' inside the value, which is
# the whole reason this is a regex and not a split -- see _parse_attributes.
_GTF_ATTRIBUTE = re.compile(r'(\S+)\s+"([^"]*)"')

# Ensembl GFF3 types the non-coding gene rather than tagging it: a lincRNA is
# feature type 'lincRNA_gene' with no biotype attribute at all. Maize's AGPv4
# GFF3 is exactly this, and it is the difference between 37 of its markers
# being classified as expected-missing and being reported as unexplained.
_TYPED_GENE_FEATURE = re.compile(r'^(?P<biotype>.+)_gene$')


class GeneMapError(ValueError):
    """Raised for an unreadable or unusable annotation file."""


def _open(path: Path):
    """Annotation files are large and sometimes shipped gzipped."""
    if str(path).endswith('.gz'):
        return gzip.open(path, 'rt')
    return open(path)


def _parse_attributes(field: str) -> dict:
    """
    Parse a GTF or GFF3 attribute column into a plain dict.

    The two formats differ in how a value is delimited (`gene_id "X";` vs
    `gene_id=X;`), and the parser dispatches on the presence of quotes rather
    than on the file extension -- which lies often enough (several '.gtf'
    files in this project are GFF3-flavoured, and vice versa) not to be worth
    trusting.

    GTF is matched with a regex rather than split on ';' because a quoted
    value may itself contain one. Arabidopsis has 7 marker genes whose symbol
    does (CDKB1;1, PIP2;8, SULTR2;1, ...); splitting on ';' truncates the name
    to 'CDKB1' and silently loses the alias. GFF3 needs no such care -- the
    spec requires a literal ';' in a value to be percent-encoded.
    """
    field = field.strip()
    attrs = {}
    if '"' in field:
        for match in _GTF_ATTRIBUTE.finditer(field):
            key = match.group(1).lstrip(';').strip()
            value = match.group(2).strip()
            # Gossypium's GTF writes `transcript_id ""` on gene lines; an
            # empty value is absence, not a value.
            if key and value:
                attrs[key] = value
        return attrs

    for chunk in field.rstrip(';').split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, _, value = chunk.partition('=') if '=' in chunk else chunk.partition(' ')
        key, value = key.strip(), value.strip()
        if key and value:
            attrs[key] = value
    return attrs


def _strip_id_prefix(value: str) -> str:
    """Drop a `gene:`-style prefix an annotation may write on its ids."""
    return _ID_PREFIX.sub('', value)


class GeneMap:
    """
    One species' alias/biotype index.

    Construct with `GeneMap.build()` (parses the annotation) or
    `GeneMap.load()` / `GeneMap.cached()` (reads a previously written JSON).
    """

    def __init__(self, species, canonical=None, aliases=None, biotypes=None, source=None):
        self.species = species
        self.canonical = canonical or set()
        self.aliases = aliases or {}
        self.biotypes = biotypes or {}
        self.source = source

    # ------------------------------------------------------------------ build

    @classmethod
    def build(cls, species: str, annotation_path, alias_attributes=None, logger=None):
        """Parse an annotation file into an index. One pass, no index built on disk."""
        logger = logger or logging.getLogger('AnnotationLogging')
        annotation_path = Path(annotation_path)
        if not annotation_path.exists():
            raise GeneMapError(f"{species}: annotation file not found: {annotation_path}")

        alias_attributes = tuple(alias_attributes or DEFAULT_ALIAS_ATTRIBUTES)
        canonical, aliases, biotypes = set(), {}, {}
        n_lines = n_gene_records = 0

        with _open(annotation_path) as handle:
            for line in handle:
                if not line or line[0] == '#':
                    continue
                parts = line.split('\t')
                if len(parts) < 9:
                    continue
                n_lines += 1
                feature = parts[2]
                typed = _TYPED_GENE_FEATURE.match(feature)
                is_gene = feature == 'gene' or typed is not None
                is_transcript = feature in _TRANSCRIPT_FEATURES
                if not (is_gene or is_transcript):
                    continue

                attrs = _parse_attributes(parts[8])
                keys = _GENE_ID_ATTRIBUTES if is_gene else _TRANSCRIPT_PARENT_ATTRIBUTES
                raw_gene = next((attrs[k] for k in keys if k in attrs), None)
                if not raw_gene:
                    continue
                gene = normalize_gene_id(_strip_id_prefix(raw_gene))
                if not gene:
                    continue
                canonical.add(gene)

                if is_gene:
                    n_gene_records += 1
                    for attr in alias_attributes:
                        value = attrs.get(attr)
                        if not value:
                            continue
                        # 'Alias' is comma-separated by GFF3 convention; the
                        # others are single-valued but splitting is harmless.
                        for token in value.split(','):
                            alias = normalize_gene_id(_strip_id_prefix(token))
                            # An alias equal to the gene ID carries no
                            # information -- Gossypium's locus_tag and
                            # Manihot's Name are both exactly this.
                            if alias and alias != gene:
                                aliases.setdefault(alias, set()).add(gene)

                    biotype = attrs.get('gene_biotype') or attrs.get('biotype')
                    if not biotype and typed:
                        biotype = typed.group('biotype')
                    # A gene-level record wins over anything a transcript line
                    # implied, but the first gene record for an ID is kept.
                    if biotype and gene not in biotypes:
                        biotypes[gene] = biotype

        if not canonical:
            raise GeneMapError(
                f"{species}: no gene records found in {annotation_path.name}. "
                f"Checked {n_lines} feature lines for 'gene'/'*_gene'/'transcript'/'mRNA'.")

        logger.info(
            '[ANNOTATION] %s: annotation indexed -- %d genes, %d aliases, %d biotypes (%s)',
            species, len(canonical), len(aliases), len(biotypes), annotation_path.name)
        if not aliases:
            logger.info(
                '[ANNOTATION] %s: annotation carries no gene names/synonyms; '
                'symbol-based marker rescue is unavailable for this species', species)
        if not biotypes:
            logger.info(
                '[ANNOTATION] %s: annotation carries no biotypes; non-coding markers '
                'cannot be distinguished from genuine mismatches for this species', species)

        return cls(species, canonical=canonical, aliases=aliases,
                   biotypes=biotypes, source=str(annotation_path))

    # ------------------------------------------------------------------ cache

    def to_dict(self, digest=None) -> dict:
        """JSON-serializable form, stamped with the parser version and the source
        file's `digest`."""
        return {
            'parser_version': PARSER_VERSION,
            'species': self.species,
            'source': self.source,
            'source_digest': digest,
            'canonical': sorted(self.canonical),
            'aliases': {a: sorted(g) for a, g in sorted(self.aliases.items())},
            'biotypes': dict(sorted(self.biotypes.items())),
        }

    @classmethod
    def load(cls, path):
        """Read a GeneMap back from its cached JSON."""
        payload = json.loads(Path(path).read_text())
        return cls(
            payload['species'],
            canonical=set(payload['canonical']),
            aliases={a: set(g) for a, g in payload['aliases'].items()},
            biotypes=payload['biotypes'],
            source=payload.get('source'),
        )

    @classmethod
    def cached(cls, species, annotation_path, cache_dir, alias_attributes=None, logger=None):
        """
        Build the index, or reuse a cached one whose source file is unchanged.

        Parsing every annotation in this project is a couple of GB of text --
        worth doing once per annotation-file version, not once per run. The
        cache is keyed on the source digest and the parser version, so either
        changing invalidates it.
        """
        logger = logger or logging.getLogger('AnnotationLogging')
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f'{species}.json'
        digest = file_digest(annotation_path)

        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text())
                if (payload.get('source_digest') == digest
                        and payload.get('parser_version') == PARSER_VERSION):
                    logger.info('[ANNOTATION] %s: reusing cached annotation index (%s)',
                                species, cache_path.name)
                    return cls.load(cache_path)
            except (json.JSONDecodeError, KeyError):
                logger.warning('[ANNOTATION] %s: cached index unreadable, rebuilding', species)

        genemap = cls.build(species, annotation_path,
                            alias_attributes=alias_attributes, logger=logger)
        cache_path.write_text(json.dumps(genemap.to_dict(digest=digest)))
        return genemap

    # --------------------------------------------------------------- resolve

    def resolve(self, token: str, proteome_genes: set):
        """
        Map one normalized marker token to the gene IDs written into the
        OrthoFinder input.

        Returns `(status, genes)`. `genes` is non-empty only for a resolution;
        every other status is a reason the marker cannot vote:

            direct      already a proteome gene ID
            alias       a symbol/synonym for one or more proteome genes
            noncoding   a gene the annotation calls non-protein-coding, so its
                        absence from a proteome is correct, not a defect
            no_protein  a gene the annotation knows and does not call
                        non-coding, but which has no protein in the proteome
            unknown     absent from the annotation entirely (a retired locus
                        ID, or an ID from a different annotation release)
        """
        if token in proteome_genes:
            return 'direct', {token}

        hits = self.aliases.get(token)
        if hits:
            resolved = hits & proteome_genes
            if resolved:
                return 'alias', set(resolved)

        biotype = self.biotypes.get(token)
        if biotype and biotype not in CODING_BIOTYPES:
            return 'noncoding', set()
        if token in self.canonical:
            return 'no_protein', set()
        return 'unknown', set()

    def alias_table(self, proteome_genes: set, tokens) -> dict:
        """
        The resolved-alias subset of this index, as {token -> [gene IDs]}: only
        the tokens a marker file uses that resolve to a proteome gene by alias
        (283 across all species here). Not called by the pipeline:
        OrthoManager.resolve_marker_tokens calls `resolve` directly.
        """
        table = {}
        for token in tokens:
            status, genes = self.resolve(token, proteome_genes)
            if status == 'alias':
                table[token] = sorted(genes)
        return table


def build_all(annotation_dict: dict, cache_dir, alias_attributes=None, logger=None) -> dict:
    """
    Index every species that declares an annotation file.

    A species with no `annotation_file:` is simply absent from the result --
    marker resolution then falls back to direct comparison.
    """
    logger = logger or logging.getLogger('AnnotationLogging')
    alias_attributes = alias_attributes or {}
    genemaps = {}
    for species in sorted(annotation_dict):
        path = annotation_dict[species]
        if not path:
            continue
        genemaps[species] = GeneMap.cached(
            species, path, cache_dir,
            alias_attributes=alias_attributes.get(species), logger=logger)
    return genemaps
