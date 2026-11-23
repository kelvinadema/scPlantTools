# Preparing proteomes for OrthoFinder

Rationale extracted from `annotation/proteome.py`, 2026-08-06. The measured
claims are asserted in `tests/annotation/test_proteome.py`.

## Gene IDs are normalized, and the rule is per species

The gene ID of a protein is **not recoverable by a single rule** across these
proteomes — for Sorghum and Gossypium it is not in the FASTA ID at all, only in
the description — so the rule is declared per species in the config
(`GENE_ID_SOURCES`: `id`, `description`, `attribute`).

Reducing each proteome to one protein per gene and rewriting the headers to the
normalized ID means OrthoFinder emits gene IDs directly: no post-hoc cleaning of
`Orthogroups.tsv`, and the marker lookup is an exact dict hit.

Normalization unifies separators and case because marker files write locus IDs
with `-` where the proteome uses `_`:

| Marker file | Proteome |
|---|---|
| `GLYMA-03G105000` | `Glyma.07G273800` |
| `SORBI-3001G000400` | `SORBI_3008G134800` |
| `PVK06-000059` | `PVK06_050008` |

Free text is URL-encoded in some headers, so it is unquoted first. An
annotation-release suffix (`Manes.01G001400.v6.1`) is stripped: marker files and
proteomes routinely disagree on it while naming the same gene.

`gene_id.exclude` exists so a species can drop a subset of proteins **on
purpose**. Without it the only way is a rule that fails to match them, which is
indistinguishable from a broken rule — it warns, and the count looks like a
defect rather than a choice.

## Three things are wrong with these proteomes as shipped

All of them reach DIAMOND unchanged unless `sanitize_sequence` handles them:

| Problem | Where | Scale |
|---|---|---|
| Trailing stop codon | Cassava, rice, maize | 32,729 / 33,135 / 7,879 sequences |
| Internal stop codons | Maize | 1,694 sequences, up to 10 in one 206-residue model |
| Gap characters (`.`) | Lotus | 488 sequences, all `_LC` low-confidence models |

Every maize sequence with an internal stop is a **provisional** gene model —
verified, not assumed. These are broken translations, and an aligner scoring
them as real residues is comparing noise.

**Non-standard characters are masked to `X`, never dropped.** Deleting a residue
silently shifts the rest of the sequence out of frame relative to its true
alignment, while `X` is what the substitution matrix already means by "unknown".
The sequence keeps its length and its gene keeps its place in an orthogroup.

`min_length` is deliberately low. Measured against every reference marker file,
a floor of **10 drops no marker gene at all**, while 30 would drop 12 and 50
would drop 62. It exists to remove obvious junk — Arabidopsis ships a 1-residue
"protein", `AT1G64633` — not to judge what is a real gene.

## `CODING_BIOTYPES` is a set because the annotations disagree

Ensembl-derived GTFs say `protein_coding`; Medicago's EuGene GTF writes
`gene_biotype "mRNA"` for all 44,594 of its coding genes.

Treating only `protein_coding` as coding made every Medicago gene look
non-coding, which mis-scored its proteome/annotation cross-check as **84.64%**
when its true recall against the mRNA set is exactly **1.0000**, and would have
classified any Medicago marker as `noncoding`.

It is defined in `proteome.py` rather than `genemap.py`, where it conceptually
belongs, because `genemap` imports this module — putting it there and importing
it back would be a cycle. `genemap` re-exports it.

## Missing markers are classified, not just counted

Without a genemap a marker either is a proteome gene ID or it is missing, with
no way to say why. That conflates three situations whose fixes differ, so every
unmatched marker goes through `GeneMap.resolve`:

| Outcome | Meaning |
|---|---|
| `alias` | the marker names a gene *symbol* the proteome writes as a locus ID. Recovered, and the recovered locus is what votes. |
| `noncoding` | an rRNA/lincRNA/miRNA. Absent from a proteome by definition, so counting it against coverage measures the annotation rather than the marker file. |
| `no_protein` | the annotation knows the gene and does not call it non-coding, yet it has no protein. Usually a non-coding locus in an annotation that omits biotypes (Oryza's IRGSP GTF). |
| `unknown` | absent from the annotation entirely — a retired locus ID, or an ID from a different release than the proteome. **This is the one that means something is wrong.** |

`coverage` stays the raw matched/total so it is comparable with previous runs.
`coverage_adjusted` divides by the markers that *could* have matched and is what
the floor is applied to. Non-coding markers are excluded from that denominator;
`no_protein` is **not** — it is unexplained often enough to be worth keeping
visible.

## No tissue filtering when reading markers

Restricting the reference markers to one tissue shrinks the orthogroup universe
`OMGBuilder._species_universe` computes, and Fisher's expected overlap scales as
1/N — so it weakens every test, including for cell types whose own markers were
untouched.

Measured: identical overlaps (X=9, X=33) went from q=2.1e-04 to 1.2e-02, and
3.3e-15 to 4.5e-11, simply by filtering. The tissue filter is applied to the
*candidates* instead (see `annotate.py`).

## The annotation cross-check

`validate_all` only covers species that have a marker file, so a proteome-only
reference (Lotus) and the target species get no coverage check at all. A wrong
path, a broken `gene_id` rule, or a proteome from a different annotation release
than its GTF would pass silently and simply produce worse orthogroups.

The measure is **annotation recall** — the fraction of the annotation's
protein-coding genes that have a protein in the prepared proteome. The obvious
inverse (are the proteome's genes in the annotation?) does not catch a proteome
that is a strict subset of its annotation, which is exactly the failure mode.

## The cache key hashes the prepared proteome

Not the source file. That folds the proteome version, the `gene_id` rule and the
isoform choice into one value, so any change to any of them yields a new key,
and any combination already computed is found again.
