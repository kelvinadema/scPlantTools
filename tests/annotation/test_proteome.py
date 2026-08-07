"""Proteome preparation -- `annotation/proteome.py`.

These are the pure functions every gene identifier in the annotation stage
passes through. Their rules were prose in the module docstrings until
2026-08-06 (see `docs/decisions/proteome-preparation.md`); the ones that name a
concrete input and output are assertions here.

The normalization cases are taken from the real disagreements between this
project's marker files and proteomes, so a "simplification" that breaks any of
them breaks marker lookup for that species silently -- an unmatched marker looks
exactly like a marker that legitimately is not in the proteome.
"""

from __future__ import annotations

import pytest

from annotation.proteome import (CODING_BIOTYPES, DEFAULT_PROCESSING,
                                 GENE_ID_SOURCES, ProteomeError,
                                 VALID_RESIDUES, _compile_rule,
                                 _gene_from_header, _iter_fasta, _wrap,
                                 normalize_gene_id, normalize_tissue,
                                 read_marker_genes, sanitize_sequence)


# ------------------------------------------------------------- normalization

@pytest.mark.parametrize('marker, proteome', [
    ('GLYMA-03G105000', 'Glyma.03G105000'),
    ('SORBI-3001G000400', 'SORBI_3001G000400'),
    ('PVK06-000059', 'PVK06_000059'),
])
def test_the_separator_disagreements_normalize_to_one_id(marker, proteome):
    assert normalize_gene_id(marker) == normalize_gene_id(proteome)


def test_an_annotation_release_suffix_is_stripped():
    assert normalize_gene_id('Manes.01G001400.v6.1') == normalize_gene_id('Manes.01G001400')


def test_a_url_encoded_id_is_unquoted_before_normalizing():
    assert normalize_gene_id('AT1G01010%2E1') == normalize_gene_id('AT1G01010.1')


def test_normalization_is_idempotent():
    once = normalize_gene_id('Manes.01G001400.v6.1')
    assert normalize_gene_id(once) == once


def test_surrounding_whitespace_does_not_change_the_id():
    assert normalize_gene_id('  AT1G01010  ') == normalize_gene_id('AT1G01010')


def test_tissue_names_compare_case_and_whitespace_insensitively():
    assert normalize_tissue('Root ') == normalize_tissue('root')
    assert normalize_tissue(None) == ''


# ---------------------------------------------------------------- sanitizing

def test_a_trailing_stop_is_stripped_and_reported():
    clean, n_masked, had_stop = sanitize_sequence('MKV*')
    assert clean == 'MKV'
    assert had_stop is True
    assert n_masked == 0


def test_a_trailing_stop_is_kept_when_not_asked_for():
    clean, _, had_stop = sanitize_sequence('MKV*', strip_trailing_stop=False)
    assert clean.startswith('MKV')
    assert had_stop is False


def test_an_internal_stop_is_masked_not_deleted():
    # Deleting would shift the rest of the sequence out of frame relative to
    # its true alignment; X is what the substitution matrix means by "unknown".
    clean, n_masked, _ = sanitize_sequence('MK*VA')
    assert clean == 'MKXVA'
    assert len(clean) == len('MK*VA')
    assert n_masked == 1


def test_a_gap_character_is_masked_not_deleted():
    clean, n_masked, _ = sanitize_sequence('MK.VA')
    assert clean == 'MKXVA'
    assert n_masked == 1


def test_every_masked_residue_is_counted():
    clean, n_masked, _ = sanitize_sequence('M*K.V@A')
    assert clean == 'MXKXVXA'
    assert n_masked == 3


def test_a_clean_sequence_is_returned_unchanged_and_uppercased():
    clean, n_masked, had_stop = sanitize_sequence('mkvay')
    assert clean == 'MKVAY'
    assert (n_masked, had_stop) == (0, False)


@pytest.mark.parametrize('residue', sorted(VALID_RESIDUES))
def test_no_accepted_residue_is_ever_masked(residue):
    # The IUPAC ambiguity codes (B, Z, J, U, O, X) are real residues an aligner
    # accepts, not junk to mask.
    clean, n_masked, _ = sanitize_sequence('MK' + residue + 'VA')
    assert n_masked == 0
    assert clean == 'MK' + residue + 'VA'


def test_a_sequence_of_only_stops_collapses_to_empty():
    clean, _, had_stop = sanitize_sequence('***')
    assert clean == ''
    assert had_stop is True


# ------------------------------------------------------------------ constants

def test_medicago_s_mrna_biotype_counts_as_coding():
    # Treating only 'protein_coding' as coding made every Medicago gene look
    # non-coding and mis-scored its cross-check as 84.64% against a true 1.0.
    assert 'mRNA' in CODING_BIOTYPES
    assert 'protein_coding' in CODING_BIOTYPES


def test_the_length_floor_stays_low_enough_to_keep_every_marker():
    # Measured: 10 drops no marker gene, 30 drops 12, 50 drops 62.
    assert DEFAULT_PROCESSING['min_length'] <= 10


# --------------------------------------------------------------- gene_id rules

def test_the_default_rule_takes_the_whole_first_token():
    rule = _compile_rule('Test', None)
    assert _gene_from_header('AT1G01010.1 some description', rule) == 'AT1G01010.1'


def test_an_attribute_rule_reads_the_key_from_the_description():
    rule = _compile_rule('Test', {'from': 'attribute', 'key': 'gene:'})
    assert _gene_from_header('PROT1 gene:SORBI_3001G000400 more', rule) == 'SORBI_3001G000400'


def test_a_description_rule_needs_a_pattern():
    with pytest.raises(ProteomeError, match='requires'):
        _compile_rule('Test', {'from': 'description'})


def test_an_attribute_rule_needs_a_key():
    with pytest.raises(ProteomeError, match='requires'):
        _compile_rule('Test', {'from': 'attribute'})


def test_an_unknown_source_is_rejected():
    with pytest.raises(ProteomeError, match='gene_id.from'):
        _compile_rule('Test', {'from': 'nowhere'})


def test_a_non_mapping_rule_is_rejected():
    with pytest.raises(ProteomeError, match='must be a mapping'):
        _compile_rule('Test', ['id'])


def test_an_invalid_regex_is_reported_against_its_species():
    with pytest.raises(ProteomeError, match='Test'):
        _compile_rule('Test', {'from': 'description', 'pattern': '([unclosed'})


def test_a_header_the_rule_cannot_match_returns_none_rather_than_guessing():
    rule = _compile_rule('Test', {'from': 'attribute', 'key': 'gene:'})
    assert _gene_from_header('PROT1 no attribute here', rule) is None


def test_every_declared_source_compiles_with_its_required_keys():
    ok = {'id': None,
          'description': {'from': 'description', 'pattern': r'gene=(\S+)'},
          'attribute': {'from': 'attribute', 'key': 'gene:'}}
    for source in GENE_ID_SOURCES:
        assert _compile_rule('Test', ok[source])['source'] == source


# ---------------------------------------------------------------------- fasta

def test_wrapping_splits_at_the_given_width():
    assert _wrap('ABCDEFG', 3) == 'ABC\nDEF\nG'


def test_a_non_positive_width_writes_one_line():
    assert _wrap('ABCDEFG', 0) == 'ABCDEFG'


def test_fasta_records_are_concatenated_across_lines(tmp_path):
    fa = tmp_path / 'p.fa'
    fa.write_text('>one desc\nMKV\nAY\n>two\nMM\n')
    assert list(_iter_fasta(fa)) == [('one desc', 'MKVAY'), ('two', 'MM')]


def test_an_empty_fasta_yields_nothing(tmp_path):
    fa = tmp_path / 'empty.fa'
    fa.write_text('')
    assert list(_iter_fasta(fa)) == []


# -------------------------------------------------------------- marker tables

def test_markers_are_read_from_the_named_column(tmp_path):
    mf = tmp_path / 'm.txt'
    mf.write_text('Species\tTissues\tCelltypes\tGenes\n'
                  'At\troot\tepidermis\tAT1G01010\n')
    genes = read_marker_genes(mf)
    assert genes[normalize_gene_id('AT1G01010')] == 'AT1G01010'


def test_a_multi_alias_cell_is_split_since_either_may_resolve(tmp_path):
    mf = tmp_path / 'm.txt'
    mf.write_text('Species\tTissues\tCelltypes\tGenes\n'
                  'Mt\troot\tcortex\tAtPR-1-like,MtCAPE12\n')
    genes = read_marker_genes(mf)
    assert set(genes.values()) == {'AtPR-1-like', 'MtCAPE12'}


def test_a_missing_gene_column_falls_back_to_the_last_one(tmp_path):
    mf = tmp_path / 'm.txt'
    mf.write_text('Species\tTissues\tCelltypes\tSomethingElse\n'
                  'At\troot\tepidermis\tAT1G01010\n')
    genes = read_marker_genes(mf, gene_column='Genes')
    assert normalize_gene_id('AT1G01010') in genes


def test_the_raw_id_as_written_is_preserved_alongside_the_normalized_key(tmp_path):
    mf = tmp_path / 'm.txt'
    mf.write_text('Species\tTissues\tCelltypes\tGenes\n'
                  'Gm\troot\tnodule\tGLYMA-03G105000\n')
    genes = read_marker_genes(mf)
    assert genes[normalize_gene_id('Glyma.03G105000')] == 'GLYMA-03G105000'


def test_a_missing_marker_file_raises_proteome_error(tmp_path):
    with pytest.raises(ProteomeError, match='not found'):
        read_marker_genes(tmp_path / 'nope.txt')
