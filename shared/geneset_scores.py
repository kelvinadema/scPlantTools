#!/usr/bin/env python3
"""
Per-cell geneset scoring, shared by the annotation and refinement stages.

One scorer for all six marker methods, deliberately. The methods differ in how
they *find* genesets; if they also differed in how those genesets are turned
into per-cell scores, a difference in the final annotation could not be
attributed to either. Only `hotspot` emits a native per-cell module score
anyway (`run_hotspot` in markerid/methods.py) -- `lmd` and `cfmf` do not, despite
also being cluster-free -- so a uniform scorer is less code than special-casing,
not more.

Two consumers, from the same scores:

- **annotation** associates each geneset with the cluster its top-scoring cells
  fall in, which is what makes a cluster-free module and a leiden cluster
  comparable at all
- **refinement** picks seed cells per geneset and hands the rest to scANVI

`scanpy` is the only heavy import and is present in both environments
(1.11.4 in each), so this can live in `shared/`.
"""

import logging

import numpy as np
import pandas as pd
import scanpy as sc

logger = logging.getLogger('AnnotationLogging')

# Minimum genes a geneset needs (in the marker file, and present in the object)
# to be scored at all; a score built on two observed genes is noise wearing a number.
DEFAULT_MIN_GENES = 3


def read_genesets(marker_file, cluster_column='clusterName', gene_column='gene',
                  min_genes=DEFAULT_MIN_GENES) -> dict:
    """
    `{geneset name -> [genes]}` from a marker table, preserving file order.

    Genesets smaller than `min_genes` are dropped: `sc.tl.score_genes`
    subtracts a binned control mean, and with one or two genes the result is
    dominated by which control bin they landed in rather than by the geneset.
    """
    df = pd.read_csv(marker_file, sep='\t', dtype={cluster_column: str})
    out = {}
    for name, group in df.groupby(cluster_column, sort=False):
        genes = list(dict.fromkeys(group[gene_column].dropna().astype(str)))
        if len(genes) >= min_genes:
            out[str(name)] = genes
    dropped = df[cluster_column].nunique() - len(out)
    if dropped:
        logger.info('[SCORES] %d geneset(s) below min_genes=%d were not scored',
                    dropped, min_genes)
    return out


def ensure_score_layer(adata, layer='data_norm', source_layer='corrected_counts'):
    """
    Build `layer` (normalised, log1p-transformed) from `source_layer` if it is
    not already present; a no-op otherwise.

    `score_genesets` requires `layer` to already hold normalised log
    expression, but nothing upstream of annotation ever writes one: mapping
    and integration outputs carry only raw/corrected counts, and
    `markerid.markers.MarkerRun._filter_and_normalize` builds its own
    normalised layer (`norm_log`, by default) on an in-memory, gene-filtered
    copy that is local to the marker-generation subprocess and never written
    back to the h5ad annotation later reads. This is the annotation side's own
    equivalent -- built once per sample, in-memory only, on every gene (no
    filtering: a dropped marker gene would silently under-score its geneset).
    """
    if layer is None or layer in adata.layers:
        return
    if source_layer != 'X' and source_layer not in adata.layers:
        raise KeyError(
            f'score_source_layer {source_layer!r} not in .layers '
            f'({list(adata.layers)})')
    source = adata.X if source_layer == 'X' else adata.layers[source_layer]
    adata.layers[layer] = source.copy()
    sc.pp.normalize_total(adata, target_sum=1e4, layer=layer)
    adata.uns.pop('log1p', None)  # see MarkerRun._filter_and_normalize
    sc.pp.log1p(adata, layer=layer)
    logger.info('[SCORES] Built %r from %r (normalize_total + log1p)',
               layer, source_layer)


def score_genesets(adata, genesets: dict, layer='data_norm', use_raw=False,
                   random_state=0) -> pd.DataFrame:
    """
    Score every geneset in every cell -> DataFrame (cells x genesets).

    `layer` must be **normalised, log-transformed expression**, not raw counts:
    `score_genes` compares a geneset's mean against a control set binned on
    average expression, and on raw counts those bins track library size, so
    every score becomes a proxy for sequencing depth.

    Genes absent from `adata.var_names` are dropped per geneset and counted, so
    a marker table built against a different gene annotation is visible rather
    than silently producing weak scores.
    """
    if layer is not None and layer not in adata.layers:
        raise KeyError(
            f'layer {layer!r} not in .layers ({list(adata.layers)}). Scoring needs '
            f'normalised log expression; pass layer=None to use .X if that is what '
            f'holds it.')

    var_names = set(adata.var_names)
    scores, missing_total = {}, 0
    for name, genes in genesets.items():
        present = [g for g in genes if g in var_names]
        missing_total += len(genes) - len(present)
        if len(present) < DEFAULT_MIN_GENES:
            logger.warning('[SCORES] geneset %r: only %d/%d genes present; not scored',
                           name, len(present), len(genes))
            continue
        key = f'_score_{name}'
        sc.tl.score_genes(adata, gene_list=present, score_name=key, layer=layer,
                          use_raw=use_raw, random_state=random_state)
        scores[name] = np.asarray(adata.obs.pop(key), dtype=float)

    if missing_total:
        logger.info('[SCORES] %d geneset gene(s) absent from the AnnData and skipped',
                    missing_total)
    if not scores:
        raise ValueError('No geneset could be scored; check gene naming and min_genes')
    return pd.DataFrame(scores, index=adata.obs_names)


def zscore(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise each geneset's scores across cells.

    Raw `score_genes` values are **not comparable between genesets** -- their
    scale depends on how many genes the set has and how highly expressed they
    are. Any operation that compares one geneset against another for the same
    cell (the argmax in `assign_genesets`, the margin rule) has to happen on a
    per-geneset standardised scale, or the widest-scaled geneset simply wins
    everywhere.
    """
    std = scores.std(axis=0).replace(0, np.nan)
    return (scores - scores.mean(axis=0)) / std


def assign_genesets(scores: pd.DataFrame, margin=0.25, min_z=0.0) -> pd.Series:
    """
    Best geneset per cell, or NaN where the call is not clear-cut.

    Comparison is on z-scores (see `zscore`). A cell is assigned only when its
    best geneset beats the runner-up by `margin` standard deviations *and*
    clears `min_z`. Everything else is left unassigned for scANVI to resolve:
    seeds exist to be confident, and an ambiguous cell handed to the model as a
    guess is worse than one handed over as a question.
    """
    if scores.shape[1] == 1:
        z = zscore(scores)
        best = z.columns[0]
        return pd.Series(np.where(z[best] >= min_z, best, None), index=z.index)

    z = zscore(scores)
    values = z.to_numpy()
    order = np.argsort(-values, axis=1)
    top_idx = order[:, 0]
    top = values[np.arange(len(values)), top_idx]
    second = values[np.arange(len(values)), order[:, 1]]

    clear = (top - second >= margin) & (top >= min_z)
    names = np.asarray(z.columns)
    return pd.Series(np.where(clear, names[top_idx], None), index=z.index)


def seed_cells(scores: pd.DataFrame, quantile=0.9, margin=0.25,
               min_z=0.0) -> pd.Series:
    """
    The confident core of each geneset: assigned cells in its top `quantile`.

    A quantile per geneset rather than one absolute cut, because scores are not
    comparable across genesets -- an absolute threshold would hand every seed
    to whichever geneset happens to score highest overall.
    """
    assigned = assign_genesets(scores, margin=margin, min_z=min_z)
    out = pd.Series(None, index=scores.index, dtype=object)
    for name in scores.columns:
        mask = assigned == name
        if not mask.any():
            continue
        cutoff = scores.loc[mask, name].quantile(quantile)
        out[mask & (scores[name] >= cutoff)] = name
    return out


def associate_with_clusters(scores: pd.DataFrame, clusters: pd.Series,
                            quantile=0.9, margin=0.25) -> pd.DataFrame:
    """
    Which cluster does each geneset actually describe?

    Returns one row per geneset: the cluster holding most of its seed cells,
    what fraction that is, and the seed count.

    This is what makes the cluster-free methods comparable with the
    cluster-dependent ones. A `hotspot` module and a leiden cluster are not the
    same kind of object, so "does hotspot agree with wilcoxon?" is otherwise
    unanswerable -- both have to be expressed against the same partition first.

    The cluster-dependent methods double as a positive control: their genesets
    were *derived* from these very clusters, so `wilcoxon`'s geneset for
    cluster 3 must associate back to cluster 3. If it does not, the scoring is
    wrong and nothing downstream of it should be trusted.
    """
    seeds = seed_cells(scores, quantile=quantile, margin=margin)
    clusters = clusters.astype(str).reindex(scores.index)

    rows = []
    for name in scores.columns:
        members = clusters[seeds == name]
        if len(members) == 0:
            rows.append({'geneset': name, 'assoc_cluster': '', 'assoc_frac': np.nan,
                         'n_seed_cells': 0})
            continue
        counts = members.value_counts()
        rows.append({'geneset': name,
                     'assoc_cluster': counts.index[0],
                     'assoc_frac': round(float(counts.iloc[0] / counts.sum()), 3),
                     'n_seed_cells': int(counts.sum())})
    return pd.DataFrame(rows).set_index('geneset')
