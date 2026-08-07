#!/usr/bin/env python3
"""
Proteome preparation and marker validation for the annotation stage.

Two jobs, both prerequisites of the OrthoFinder run:

1. Reduce each proteome to one protein per gene -- the longest isoform --
   and rewrite the FASTA headers to a normalized gene ID. OrthoFinder then
   emits gene IDs directly, so no post-hoc cleaning of Orthogroups.tsv is
   needed and the marker lookup is an exact dict hit.

2. Check that the genes named in each species' marker file actually exist in
   that species' proteome, before hours of OrthoFinder compute are spent.

The gene ID of a protein is *not* recoverable by a single rule across these
proteomes -- for Sorghum and Gossypium it is not in the FASTA ID at all, only
in the description -- so the rule is declared per species in the config. See
GENE_ID_SOURCES for the three forms.
"""

import hashlib
import logging
import re

from pathlib import Path
from urllib.parse import unquote

GENE_ID_SOURCES = ('id', 'description', 'attribute')

# Stripped during normalization: an annotation-release suffix carried by some
# ID schemes (Manes.01G001400.v6.1). Marker files and proteomes routinely
# disagree on it while naming the same gene.
_ANNOT_VERSION = re.compile(r'\.v\d+(?:\.\d+)*$', re.IGNORECASE)

# Residues an aligner will accept: the 20 standard amino acids plus the IUPAC
# ambiguity and rare codes (B=Asx, Z=Glx, J=Xle, U=selenocysteine,
# O=pyrrolysine, X=unknown). Anything else is masked -- see sanitize_sequence.
VALID_RESIDUES = frozenset('ACDEFGHIKLMNPQRSTVWY' 'BZJUOX')

# A *set*, because the annotations disagree on the word: Ensembl GTFs say
# 'protein_coding', Medicago's EuGene GTF says 'mRNA'. Defined here rather than
# in genemap.py (which re-exports it) only to avoid an import cycle. See
# docs/decisions/proteome-preparation.md.
CODING_BIOTYPES = frozenset({'protein_coding', 'mRNA'})

# Defaults for the sequence hygiene pass. `min_length` is deliberately low --
# it removes obvious junk, not borderline genes; the measurements are in
# docs/decisions/proteome-preparation.md and pinned in tests/annotation/.
DEFAULT_PROCESSING = {
    'min_length': 10,
    'strip_trailing_stop': True,
    'mask_char': 'X',
    'line_width': 60,
}


class ProteomeError(ValueError):
    """Raised for a malformed gene_id rule or an unusable proteome."""


def normalize_gene_id(gene: str) -> str:
    """Canonical form for every gene identifier in this stage.

    Applied to proteome headers, marker files and orthogroup members alike, so only
    normalized IDs are ever compared. Unifies the separator and case disagreements
    between marker files and proteomes and strips any annotation-release suffix.
    See docs/decisions/proteome-preparation.md.

    Args:
        gene (str): Identifier as written.

    Returns:
        str: Normalized identifier.
    """
    gene = unquote(gene.strip())
    gene = _ANNOT_VERSION.sub('', gene)
    return gene.upper().replace('-', '_').replace('.', '_')


def _compile_rule(species: str, rule) -> dict:
    """Validate and compile one species' gene_id rule from the config.

    Args:
        species (str): Species name, for the error messages.
        rule (dict or None): The `gene_id:` block. None means 'from: id'.

    Returns:
        dict: Compiled rule with 'source', 'pattern', 'template' and 'exclude'.

    Raises:
        ProteomeError: If the rule is malformed or its regex will not compile.
    """
    if rule is None:
        rule = {'from': 'id'}
    if not isinstance(rule, dict):
        raise ProteomeError(f"{species}: gene_id must be a mapping, got {type(rule).__name__}")

    source = rule.get('from', 'id')
    if source not in GENE_ID_SOURCES:
        raise ProteomeError(
            f"{species}: gene_id.from must be one of {GENE_ID_SOURCES}, got {source!r}")

    compiled = {'source': source, 'template': rule.get('template')}

    # A regex marking proteins to skip ON PURPOSE. Without it the only way to
    # drop a subset is a rule that fails to match them, which is
    # indistinguishable from a broken rule.
    exclude = rule.get('exclude')
    if exclude:
        try:
            compiled['exclude'] = re.compile(exclude)
        except re.error as exc:
            raise ProteomeError(
                f"{species}: invalid gene_id.exclude {exclude!r}: {exc}") from exc
    else:
        compiled['exclude'] = None

    if source == 'attribute':
        key = rule.get('key')
        if not key:
            raise ProteomeError(f"{species}: gene_id.from='attribute' requires 'key' (e.g. 'gene:')")
        # key is written with its own separator ('gene:', 'locus=') so the
        # config reads like the header it matches.
        compiled['pattern'] = re.compile(re.escape(key) + r'(\S+)')
    else:
        pattern = rule.get('pattern')
        if not pattern:
            if source == 'description':
                raise ProteomeError(f"{species}: gene_id.from='description' requires 'pattern'")
            compiled['pattern'] = None  # whole first token
        else:
            try:
                compiled['pattern'] = re.compile(pattern)
            except re.error as exc:
                raise ProteomeError(f"{species}: invalid gene_id.pattern {pattern!r}: {exc}") from exc
    return compiled


def _gene_from_header(header: str, rule: dict):
    """Extract the raw (un-normalized) gene ID from one FASTA header line.

    Args:
        header (str): Header text, without the leading '>'.
        rule (dict): A compiled rule from `_compile_rule`.

    Returns:
        str or None: The gene ID, or None when the rule does not match.
    """
    ident, _, description = header.partition(' ')
    source, pattern, template = rule['source'], rule['pattern'], rule['template']

    if source == 'id' and pattern is None:
        return ident

    target = ident if source == 'id' else description
    match = pattern.search(target)
    if not match:
        return None
    groups = match.groups() or (match.group(0),)
    if template:
        return template.format(*groups)
    return groups[0]


def sanitize_sequence(sequence: str, strip_trailing_stop=True, mask_char='X'):
    """Clean one protein sequence for OrthoFinder.

    Strips a trailing stop codon and masks anything outside VALID_RESIDUES.
    Non-standard characters are masked to X rather than dropped: deleting a residue
    would shift the rest of the sequence out of frame relative to its true
    alignment. See docs/decisions/proteome-preparation.md.

    Args:
        sequence (str): Raw sequence.
        strip_trailing_stop (bool): Whether to remove a terminal '*'.
        mask_char (str): Replacement for a non-standard residue.

    Returns:
        tuple: (clean (str), n_masked (int), had_trailing_stop (bool)).
    """
    clean = sequence.strip().upper()
    had_trailing_stop = False
    if strip_trailing_stop:
        stripped = clean.rstrip('*')
        had_trailing_stop = len(stripped) != len(clean)
        clean = stripped

    if all(c in VALID_RESIDUES for c in clean):
        return clean, 0, had_trailing_stop

    masked = [c if c in VALID_RESIDUES else mask_char for c in clean]
    n_masked = sum(1 for a, b in zip(clean, masked) if a != b)
    return ''.join(masked), n_masked, had_trailing_stop


def _wrap(sequence: str, width: int):
    """FASTA line wrapping. `width` <= 0 writes the sequence on one line.

    Args:
        sequence (str): Sequence to wrap.
        width (int): Characters per line.

    Returns:
        str: The wrapped sequence.
    """
    if width and width > 0:
        return '\n'.join(sequence[i:i + width] for i in range(0, len(sequence), width))
    return sequence


def _iter_fasta(path: Path):
    """Yield (header, sequence) pairs, concatenating each record's lines.

    Args:
        path (Path): FASTA file.

    Yields:
        tuple: (header (str), sequence (str)).
    """
    header, chunks = None, []
    with open(path) as handle:
        for line in handle:
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(chunks)
                header, chunks = line[1:].strip(), []
            elif header is not None:
                chunks.append(line.strip())
    if header is not None:
        yield header, ''.join(chunks)


def prepare_proteome(species: str, proteome_path, out_path, gene_id_rule=None,
                     processing=None) -> dict:
    """Reduce one proteome to its longest isoform per gene and normalize the headers.

    OrthoFinder then emits gene IDs directly, so `Orthogroups.tsv` needs no
    post-hoc cleaning and the marker lookup is an exact dict hit. See
    docs/decisions/proteome-preparation.md.

    Args:
        species (str): Species name.
        proteome_path (Path): Source FASTA.
        out_path (Path): Where the primary-transcript FASTA is written.
        gene_id_rule (dict, optional): This species' `gene_id:` block.
        processing (dict, optional): Sequence-hygiene settings, over
            DEFAULT_PROCESSING.

    Returns:
        dict: Counts and paths describing what was written.

    Raises:
        ProteomeError: If the rule is malformed or no sequence survives.
    """
    proteome_path, out_path = Path(proteome_path), Path(out_path)
    if not proteome_path.exists():
        raise ProteomeError(f"{species}: proteome not found: {proteome_path}")
    rule = _compile_rule(species, gene_id_rule)

    opts = {**DEFAULT_PROCESSING, **(processing or {})}
    min_length = opts['min_length']
    mask_char = opts['mask_char']
    if mask_char not in VALID_RESIDUES:
        raise ProteomeError(
            f"{species}: proteome_processing.mask_char must be a valid residue, got {mask_char!r}")

    best = {}          # normalized gene -> (seq_len, protein_id, sequence)
    n_proteins = 0
    unresolved = []
    n_excluded = 0
    excluded_genes = set()
    n_trailing_stop = n_masked_seqs = n_masked_residues = 0

    for header, sequence in _iter_fasta(proteome_path):
        n_proteins += 1
        if rule['exclude'] is not None and rule['exclude'].search(header):
            n_excluded += 1
            # Record *which* genes were excluded, not just how many. The
            # proteome/annotation cross-check subtracts them from its
            # denominator: a deliberately excluded gene is still in the
            # annotation, so counting it as missing would penalise the
            # exclusion and blunt the check. Lotus reads 72% with them counted
            # and ~100% without, and only the latter still detects a genuinely
            # broken rule. Best-effort -- maize's provisional headers use a
            # different key than its gene_id rule, so they yield nothing here,
            # which is harmless since they are absent from its annotation too.
            excluded_raw = _gene_from_header(header, rule)
            if excluded_raw:
                excluded_genes.add(normalize_gene_id(excluded_raw))
            continue
        raw_gene = _gene_from_header(header, rule)
        if not raw_gene:
            unresolved.append(header.split()[0] if header else '')
            continue

        clean, n_masked, had_stop = sanitize_sequence(
            sequence, strip_trailing_stop=opts['strip_trailing_stop'], mask_char=mask_char)
        n_trailing_stop += had_stop
        if n_masked:
            n_masked_seqs += 1
            n_masked_residues += n_masked

        gene = normalize_gene_id(raw_gene)
        protein_id = header.split()[0]
        candidate = (len(clean), protein_id, clean)
        current = best.get(gene)
        # Longest wins; on an exact length tie the lexicographically smaller
        # protein ID wins, so isoform choice does not depend on file order.
        if current is None or (candidate[0], current[1]) > (current[0], candidate[1]):
            best[gene] = candidate

    if not best:
        raise ProteomeError(
            f"{species}: gene_id rule matched no header in {proteome_path.name}. "
            f"First unresolved IDs: {unresolved[:3]}")

    # Applied to the *selected* isoform, so a gene is dropped only when its
    # longest protein is still too short to align meaningfully. An empty
    # sequence is dropped whatever the floor -- OrthoFinder rejects it.
    too_short = sorted(g for g, v in best.items() if len(v[2]) < max(min_length, 1))
    for gene in too_short:
        del best[gene]
    if not best:
        raise ProteomeError(
            f"{species}: every sequence in {proteome_path.name} was shorter than "
            f"min_length={min_length}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as handle:
        for gene in sorted(best):
            handle.write(f'>{gene}\n{_wrap(best[gene][2], opts["line_width"])}\n')

    return {
        'species': species,
        'source': str(proteome_path),
        'primary_transcripts': str(out_path),
        'n_proteins': n_proteins,
        'n_genes': len(best),
        'n_isoforms_dropped': (n_proteins - len(best) - len(unresolved)
                               - len(too_short) - n_excluded),
        'n_unresolved': len(unresolved),
        'unresolved_examples': unresolved[:5],
        'n_excluded': n_excluded,
        'excluded_genes': excluded_genes,
        'n_trailing_stop_stripped': n_trailing_stop,
        'n_sequences_masked': n_masked_seqs,
        'n_residues_masked': n_masked_residues,
        'n_too_short': len(too_short),
        'too_short_examples': too_short[:5],
        'genes': set(best),
    }


def normalize_tissue(value: str) -> str:
    """Tissue names are compared case- and whitespace-insensitively.

    Args:
        value (str): Tissue as written.

    Returns:
        str: Normalized tissue name.
    """
    return str(value or '').strip().lower()


def read_marker_genes(marker_file, gene_column='Genes') -> dict:
    """Read a marker table as {normalized gene: raw ID as written}.

    Tolerates the '<Species>_high_confidence.txt' layout and falls back to the last
    column when `gene_column` is absent. Multi-alias cells are split, since either
    alias may be the resolvable one. **No tissue filtering happens here** -- it
    would shrink the orthogroup universe and weaken every enrichment test. See
    docs/decisions/proteome-preparation.md.

    Args:
        marker_file (Path): The marker table.
        gene_column (str): Column holding gene IDs.

    Returns:
        dict: {normalized gene: raw ID}.

    Raises:
        ProteomeError: If the file does not exist.
    """
    marker_file = Path(marker_file)
    if not marker_file.exists():
        raise ProteomeError(f"marker file not found: {marker_file}")

    genes = {}
    with open(marker_file) as handle:
        header = handle.readline().rstrip('\n').split('\t')
        try:
            idx = header.index(gene_column)
        except ValueError:
            idx = len(header) - 1
        for line in handle:
            fields = line.rstrip('\n').split('\t')
            if len(fields) <= idx:
                continue
            for token in fields[idx].split(','):
                token = token.strip()
                if token:
                    genes.setdefault(normalize_gene_id(token), token)
    return genes


def validate_markers(species: str, marker_file, proteome_genes: set,
                     gene_column='Genes', genemap=None) -> dict:
    """Check every marker gene against the prepared proteome's gene set.

    With a `genemap`, unmatched markers are classified as alias / noncoding /
    no_protein / unknown rather than merely counted -- only `unknown` means
    something is wrong. See docs/decisions/proteome-preparation.md.

    Args:
        species (str): Species name.
        marker_file (Path): The marker table.
        proteome_genes (set[str]): Normalized gene IDs in the prepared proteome.
        gene_column (str): Column holding gene IDs.
        genemap (GeneMap, optional): Enables the classification.

    Returns:
        dict: Counts, the four classified lists, and three coverage figures.
            `coverage` is raw matched/total; `coverage_adjusted` divides by the
            markers that could have matched and is what the floor applies to.
    """
    markers = read_marker_genes(marker_file, gene_column=gene_column)
    total = len(markers)

    direct = {g for g in markers if g in proteome_genes}
    alias, noncoding, no_protein, unknown = {}, [], [], []

    for gene in markers:
        if gene in direct:
            continue
        if genemap is None:
            unknown.append(gene)
            continue
        status, resolved = genemap.resolve(gene, proteome_genes)
        if status == 'alias':
            alias[gene] = sorted(resolved)
        elif status == 'noncoding':
            noncoding.append(gene)
        elif status == 'no_protein':
            no_protein.append(gene)
        else:
            unknown.append(gene)

    n_resolved = len(direct) + len(alias)
    # Non-coding markers are excluded from the denominator; a marker naming an
    # rRNA is not evidence of a broken marker file. `no_protein` is *not*
    # excluded -- it is unexplained often enough to be worth keeping visible.
    denominator = total - len(noncoding)
    return {
        'species': species,
        'marker_file': str(marker_file),
        'n_markers': total,
        'n_matched': len(direct),
        'n_alias': len(alias),
        'n_resolved': n_resolved,
        'n_noncoding': len(noncoding),
        'n_no_protein': len(no_protein),
        'n_unknown': len(unknown),
        'n_missing': total - n_resolved,
        'coverage': (len(direct) / total) if total else 0.0,
        'coverage_resolved': (n_resolved / total) if total else 0.0,
        'coverage_adjusted': (n_resolved / denominator) if denominator else 0.0,
        'alias': {markers[g]: v for g, v in alias.items()},
        'noncoding': sorted(markers[g] for g in noncoding),
        'no_protein': sorted(markers[g] for g in no_protein),
        'unknown': sorted(markers[g] for g in unknown),
        'missing': sorted(markers[g] for g in list(noncoding) + no_protein + unknown),
    }


def crosscheck_annotation(prepared: dict, genemaps: dict, out_dir, min_recall=0.5,
                          warn_recall=0.99, logger=None) -> dict:
    """Check each prepared proteome against its own annotation, and fail on a mismatch.

    `validate_all` only covers species that have a marker file, so a proteome-only
    reference and the target species would otherwise go unchecked. Measured as
    annotation recall -- the fraction of the annotation's coding genes with a
    protein -- because the inverse does not catch a proteome that is a strict
    subset of its annotation. See docs/decisions/proteome-preparation.md.

    Args:
        prepared (dict): Output of `prepare_all`.
        genemaps (dict): {species: GeneMap}.
        out_dir (Path): Where the report TSV is written.
        min_recall (float): Below this, the species is an error.
        warn_recall (float): Below this, the species is a warning.
        logger (logging.Logger, optional): Defaults to 'AnnotationLogging'.

    Returns:
        dict: Per-species recall and the overall verdict.
    """
    logger = logger or logging.getLogger('AnnotationLogging')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports, failures = {}, []
    for species in sorted(prepared):
        genemap = genemaps.get(species)
        if genemap is None:
            continue
        proteome_genes = prepared[species]['genes']
        coding = {g for g, b in genemap.biotypes.items() if b in CODING_BIOTYPES}
        # No biotypes -> every gene is a candidate, ncRNA included, so recall
        # is legitimately below 1 and only the hard floor is meaningful.
        expected = coding or genemap.canonical
        has_biotypes = bool(coding)
        # Genes dropped on purpose by gene_id.exclude are still in the
        # annotation, so leaving them in the denominator would score the
        # exclusion as a defect and mask a real one behind it.
        expected = expected - prepared[species].get('excluded_genes', set())
        covered = len(proteome_genes & expected)
        recall = covered / len(expected) if expected else 0.0

        report = {
            'species': species,
            'n_expected': len(expected),
            'n_covered': covered,
            'recall': recall,
            'basis': 'protein_coding' if has_biotypes else 'all_genes',
            'passed': recall >= min_recall,
        }
        reports[species] = report

        if not report['passed']:
            failures.append(
                f"{species}: {100 * recall:.1f}% of {len(expected)} annotation genes have a "
                f"protein (floor {100 * min_recall:.0f}%)")
        elif has_biotypes and recall < warn_recall:
            logger.warning(
                '[ANNOTATION] %s: only %.2f%% of the annotation\'s %d protein-coding genes '
                'have a protein in the prepared proteome -- check the gene_id rule is '
                'capturing every header, and that proteome and annotation are the same '
                'annotation release', species, 100 * recall, len(expected))
        else:
            logger.info(
                '[ANNOTATION] %s: %d/%d annotation genes have a protein (%.2f%%, %s)',
                species, covered, len(expected), 100 * recall, report['basis'])

    summary = out_dir / 'annotation_crosscheck.tsv'
    with open(summary, 'w') as handle:
        handle.write('species\tn_expected\tn_covered\trecall\tbasis\tpassed\n')
        for species in sorted(reports):
            r = reports[species]
            handle.write(f"{species}\t{r['n_expected']}\t{r['n_covered']}\t{r['recall']:.4f}\t"
                         f"{r['basis']}\t{r['passed']}\n")

    if failures:
        raise ProteomeError(
            'Proteome/annotation cross-check failed for {} species:\n  {}\n'
            'That fraction is far below anything a real proteome produces, so the two '
            'files almost certainly do not describe the same thing -- check that '
            'proteome_file and annotation_file point at the same species and the same '
            'annotation release, and that the gene_id rule matches this proteome\'s '
            'headers.'.format(len(failures), '\n  '.join(failures)))
    return reports


def file_digest(path, chunk_size=1 << 20) -> str:
    """sha256 of a file, read in chunks -- proteomes are hundreds of MB.

    Args:
        path (Path): File to hash.
        chunk_size (int): Bytes per read.

    Returns:
        str: Hex digest.
    """
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(prepared: dict) -> str:
    """Identity of an OrthoFinder run: the species set and each prepared FASTA.

    Hashing the *prepared* proteome folds the proteome version, the gene_id rule and
    the isoform choice into one value, so any change yields a new key and any
    combination already computed is found again.

    Args:
        prepared (dict): Output of `prepare_all`.

    Returns:
        str: Hex digest.
    """
    digest = hashlib.sha256()
    for species in sorted(prepared):
        digest.update(species.encode())
        digest.update(file_digest(prepared[species]['primary_transcripts']).encode())
    return digest.hexdigest()


def prepare_all(proteome_dict: dict, out_dir, gene_id_rules=None, processing=None,
                logger=None) -> dict:
    """Prepare every species' primary-transcript proteome into `out_dir`.

    Args:
        proteome_dict (dict): {species: source FASTA path}.
        out_dir (Path): Where the prepared FASTAs go.
        gene_id_rules (dict, optional): {species: gene_id block}.
        processing (dict, optional): Shared sequence-hygiene config; a species may
            override any key via `processing['overrides'][species]`.
        logger (logging.Logger, optional): Defaults to 'AnnotationLogging'.

    Returns:
        dict: {species: the dict `prepare_proteome` returned}.
    """
    logger = logger or logging.getLogger('AnnotationLogging')
    gene_id_rules = gene_id_rules or {}
    processing = processing or {}
    overrides = processing.get('overrides') or {}
    shared = {k: v for k, v in processing.items() if k != 'overrides'}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prepared = {}
    for species in sorted(proteome_dict):
        out_path = out_dir / f'{species}.fa'
        stats = prepare_proteome(species, proteome_dict[species], out_path,
                                 gene_id_rules.get(species),
                                 processing={**shared, **(overrides.get(species) or {})})
        prepared[species] = stats
        logger.info(
            '[ANNOTATION] %s: %d proteins -> %d genes (%d isoforms dropped, %d headers unresolved)',
            species, stats['n_proteins'], stats['n_genes'],
            stats['n_isoforms_dropped'], stats['n_unresolved'])
        if stats['n_excluded']:
            logger.info(
                '[ANNOTATION] %s: %d protein(s) excluded by gene_id.exclude',
                species, stats['n_excluded'])
        if stats['n_trailing_stop_stripped'] or stats['n_sequences_masked'] or stats['n_too_short']:
            logger.info(
                '[ANNOTATION] %s: cleaned -- %d trailing stop codons stripped, '
                '%d sequences masked (%d residues), %d genes dropped below min_length%s',
                species, stats['n_trailing_stop_stripped'], stats['n_sequences_masked'],
                stats['n_residues_masked'], stats['n_too_short'],
                f" (e.g. {', '.join(stats['too_short_examples'])})"
                if stats['too_short_examples'] else '')
        if stats['n_unresolved']:
            logger.warning(
                '[ANNOTATION] %s: gene_id rule failed on %d header(s), e.g. %s -- '
                'those proteins are excluded from the OrthoFinder input',
                species, stats['n_unresolved'], stats['unresolved_examples'])
    return prepared


def validate_all(marker_dict: dict, prepared: dict, out_dir, min_coverage=0.8,
                 min_coverage_overrides=None, gene_column='Genes', genemaps=None,
                 logger=None) -> dict:
    """Validate every species' marker file against its prepared proteome.

    Args:
        prepared (dict): Output of `prepare_all`.
        marker_files (dict): {species: marker table path}.
        out_dir (Path): Where the coverage report is written.
        min_coverage (float): Adjusted-coverage floor below which a species fails.
        gene_column (str): Column holding gene IDs.
        genemaps (dict, optional): {species: GeneMap}, enabling classification.
        logger (logging.Logger, optional): Defaults to 'AnnotationLogging'.

    Returns:
        dict: {species: the dict `validate_markers` returned}.
    """
    logger = logger or logging.getLogger('AnnotationLogging')
    overrides = min_coverage_overrides or {}
    genemaps = genemaps or {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports, failures = {}, []
    for species in sorted(marker_dict):
        if species not in prepared:
            raise ProteomeError(
                f"{species} has a marker file but no proteome -- markers cannot be validated")
        report = validate_markers(species, marker_dict[species],
                                  prepared[species]['genes'], gene_column=gene_column,
                                  genemap=genemaps.get(species))
        reports[species] = report

        floor = overrides.get(species, min_coverage)
        report['min_coverage'] = floor
        report['passed'] = report['coverage_adjusted'] >= floor

        level = logger.info if report['passed'] else logger.error
        level('[ANNOTATION] %s: %d/%d markers resolved (%.1f%% raw, %.1f%% adjusted, '
              'floor %.0f%%) -- %d direct, %d via alias, %d non-coding, %d no protein, '
              '%d unknown',
              species, report['n_resolved'], report['n_markers'],
              100 * report['coverage_resolved'], 100 * report['coverage_adjusted'],
              100 * floor, report['n_matched'], report['n_alias'],
              report['n_noncoding'], report['n_no_protein'], report['n_unknown'])

        if species not in genemaps:
            logger.warning(
                '[ANNOTATION] %s: no annotation_file configured -- markers naming gene '
                'symbols cannot be recovered and non-coding markers count against '
                'coverage', species)

        # One file per class. Lumping them together was the old behaviour and
        # made the actionable IDs (unknown) impossible to find among the ones
        # that need no action at all (noncoding).
        for kind in ('noncoding', 'no_protein', 'unknown'):
            if report[kind]:
                path = out_dir / f'{species}_{kind}_markers.txt'
                path.write_text('\n'.join(report[kind]) + '\n')
        if report['alias']:
            path = out_dir / f'{species}_alias_markers.tsv'
            with open(path, 'w') as handle:
                handle.write('marker\tresolved_genes\n')
                for raw, genes in sorted(report['alias'].items()):
                    handle.write(f"{raw}\t{','.join(genes)}\n")
            logger.info('[ANNOTATION] %s: %d markers recovered via annotation aliases -> %s',
                        species, report['n_alias'], path.name)
        if report['n_unknown']:
            logger.info('[ANNOTATION] %s: %d unresolved marker IDs written to %s',
                        species, report['n_unknown'],
                        out_dir / f'{species}_unknown_markers.txt')
        if not report['passed']:
            failures.append(
                f"{species}: {100 * report['coverage_adjusted']:.1f}% < {100 * floor:.0f}%")

    summary = out_dir / 'marker_coverage.tsv'
    with open(summary, 'w') as handle:
        handle.write('species\tn_markers\tn_direct\tn_alias\tn_resolved\tn_noncoding\t'
                     'n_no_protein\tn_unknown\tcoverage\tcoverage_resolved\t'
                     'coverage_adjusted\tmin_coverage\tpassed\n')
        for species in sorted(reports):
            r = reports[species]
            handle.write(
                f"{species}\t{r['n_markers']}\t{r['n_matched']}\t{r['n_alias']}\t"
                f"{r['n_resolved']}\t{r['n_noncoding']}\t{r['n_no_protein']}\t"
                f"{r['n_unknown']}\t{r['coverage']:.4f}\t{r['coverage_resolved']:.4f}\t"
                f"{r['coverage_adjusted']:.4f}\t{r['min_coverage']:.4f}\t{r['passed']}\n")
    logger.info('[ANNOTATION] Marker coverage summary written to %s', summary)

    if failures:
        raise ProteomeError(
            'Marker validation failed for {} species:\n  {}\n'
            'Inspect the *_unknown_markers.txt files in {} -- those are the IDs the '
            'annotation could not explain, and usually mean the proteome and the '
            'marker file come from different annotation releases. Non-coding markers '
            'are already excluded from this figure. If the coverage is expected, set '
            'a per-species floor under markers.min_coverage_overrides in the config.'
            .format(len(failures), '\n  '.join(failures), out_dir))
    return reports
