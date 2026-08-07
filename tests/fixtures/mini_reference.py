"""
A self-contained miniature scRNA-seq dataset: reference, reads, and truth.

WHY GENERATED RATHER THAN SUBSET FROM A REAL SAMPLE. A subset of real FASTQs
gives you data but no ground truth -- you cannot assert which barcode should
carry which gene, only that *something* came out. Generating the reference and
then sampling reads from it means the expected count matrix is known exactly, so
the end-to-end test can assert content rather than merely non-emptiness.

It is also small (a few hundred KB, built in about a second), deterministic, and
depends on no data that might be archived or moved.

The only external dependency is the 10x v3 barcode whitelist, because simpleaf
filters against it -- inventing barcodes would make every read unassigned.
"""

from __future__ import annotations

import gzip
import pathlib

import numpy as np

# Read geometry for 10x v3: 16 bp cell barcode + 12 bp UMI on R1, cDNA on R2.
BARCODE_LEN = 16
UMI_LEN = 12
READ_LEN = 91


def write_reference(out_dir, n_transcripts=30, tx_len=800, seed=0):
    """
    A FASTA of random transcripts plus the transcript->gene map simpleaf needs.

    One gene per transcript keeps the expected matrix trivial to state: a read
    from `tx007` must land on `gene007`.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    fasta = out_dir / 'ref.fa'
    t2g = out_dir / 't2g.tsv'
    seqs = {}
    with open(fasta, 'w') as fa, open(t2g, 'w') as g:
        for i in range(n_transcripts):
            name = 'tx{:03d}'.format(i)
            seq = ''.join(rng.choice(list('ACGT'), tx_len))
            seqs[name] = seq
            fa.write('>{}\n{}\n'.format(name, seq))
            g.write('{}\tgene{:03d}\n'.format(name, i))
    return {'fasta': fasta, 't2g': t2g, 'sequences': seqs}


def write_reads(out_dir, reference, whitelist, n_barcodes=20,
                tx_per_cell=3, reads_per_tx=40, seed=1):
    """
    Paired FASTQs plus the truth table.

    Each barcode expresses a disjoint block of transcripts, so the expected
    matrix is block-structured and a mis-assignment shows up as counts in the
    wrong block rather than as a change in the total.

    Returns {'r1', 'r2', 'truth': {barcode: {gene: n_reads}}, 'n_pairs'}.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    barcodes = [ln.strip() for ln in open(whitelist)][:n_barcodes]
    seqs = reference['sequences']
    tx_names = sorted(seqs)

    r1, r2, truth = [], [], {}
    for bi, bc in enumerate(barcodes):
        start = (bi * tx_per_cell) % len(tx_names)
        block = (tx_names + tx_names)[start:start + tx_per_cell]
        truth[bc] = {}
        for tx in block:
            gene = 'gene' + tx[2:]
            truth[bc][gene] = truth[bc].get(gene, 0) + reads_per_tx
            for _ in range(reads_per_tx):
                umi = ''.join(rng.choice(list('ACGT'), UMI_LEN))
                pos = int(rng.integers(0, len(seqs[tx]) - READ_LEN))
                r1.append(bc + umi)
                r2.append(seqs[tx][pos:pos + READ_LEN])

    paths = {}
    for key, reads, qlen in (('r1', r1, BARCODE_LEN + UMI_LEN), ('r2', r2, READ_LEN)):
        p = out_dir / '{}.fq.gz'.format(key.upper())
        with gzip.open(p, 'wt') as f:
            for i, seq in enumerate(reads):
                f.write('@read{}\n{}\n+\n{}\n'.format(i, seq, 'I' * qlen))
        paths[key] = p
    paths['truth'] = truth
    paths['n_pairs'] = len(r1)
    return paths
