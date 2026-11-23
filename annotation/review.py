#!/usr/bin/env python3
"""
Stage 2 -- where a person decides what the cell types are.

The six marker methods largely rediscover the same populations (counts for the
test target are in shared/groups.py), so the reviewer's object is not a
geneset, it is a **group of genesets spanning methods** -- one final cell type,
and the gene sets that describe it. That group is what seeds
the single refinement run.

The page is organised around that, and around the three questions a review
actually turns on:

  **Is this a real, coherent population?**  Select a group and the cells its
  member genesets confidently score for light up on the UMAP, coloured by how
  many methods independently found them. A group whose cells scatter across
  four clusters is not a cell type, whatever its enrichment says.

  **Do the methods agree?**  A group backed by six methods and one backed by
  one are different propositions, and the panel says which is which without
  needing a separate matrix.

  **What is the evidence, and can I overrule it?**  Every geneset keeps its
  full candidate list -- including the ones `too_general` or tissue filtering
  set aside -- and can be re-labelled, moved to another group, given a group of
  its own, or dropped entirely.

**Every number here was computed by stage 1.** Nothing is estimated. In
particular a group's cell count is the union of its members' stage-1 seed
cells, which is a real union of real sets but is *not* the final seed count:
the seeds refinement trains on are assigned at group level, which needs the
score matrices and cannot happen in a browser. `refine.main --dry-run`
prints that number, exactly, in seconds. The page says so where it shows the
count rather than leaving it to be misread.

Self-contained: stdlib templating, vanilla JS, no CDN, so it survives being
copied off the cluster -- which is how anyone actually reads one.

    python -m annotation.review --acfg configs/annotation_config.yaml \\
        --sample MtTip_noHVG \\
        --adata results/integration/scVI/MtTip_noHVG/MtTip_noHVG.h5ad \\
        --cluster-key leiden_scVI
"""

import argparse
import html
import json
import logging

from pathlib import Path

import numpy as np
import pandas as pd

from shared.groups import DROP, UNGROUPED, file_hash, propose
from shared.geneset_scores import seed_cells
from shared.html import _CSS, _PAGE, _esc, _fmt_int
from shared.species_profiles import resolve_species
from shared.utils import find_root_dir, load_config

logger = logging.getLogger('AnnotationLogging')

DEFAULT_MAX_POINTS = 60000

# Carried per candidate: everything bearing on the decision, nothing else.
# `_significant_annotations.tsv` also holds per-gene lists that would multiply
# the payload for no gain.
_CANDIDATE_COLUMNS = ('po_id', 'po_name', 'score', 'best_q', 'total_X',
                      'support_species', 'support_species_list', 'weighted_species',
                      'n_clades', 'decision', 'too_general', 'tissue_match')


def _text(row, key) -> str:
    """
    A template cell as a string, with an empty cell staying empty.

    `str(row.get(k, '') or '')` is the obvious form and is wrong: pandas reads
    a blank TSV cell as `float('nan')`, which is **truthy**, so `or ''` never
    fires and the value becomes the literal string `'nan'`. Not cosmetic --
    `top_po_id` is blank for exactly the genesets nothing could name, so every
    one of them would be exported as a confident call to a cell type that does
    not exist.
    """
    value = row.get(key)
    return '' if value is None or pd.isna(value) else str(value)


def _count(row, key) -> int:
    """`row[key]` as an int; 0 when absent, blank or not numeric."""
    value = row.get(key)
    try:
        return 0 if value is None or pd.isna(value) else int(value)
    except (TypeError, ValueError):
        return 0


def _round(values, digits=2) -> list:
    """Round a numeric sequence to `digits` places for JSON (NaN/inf are not converted)."""
    return [round(float(v), digits) for v in values]


def load_report_gene_groups(value, species=None, root=None) -> list:
    """
    `[(celltype, [gene, ...]), ...]`, celltype in first-seen order.

    Two file shapes are accepted:

    - the original one-gene-per-line list (optionally with extra tab-separated
      columns, ignored) -- comes back as a single unlabelled group `('', genes)`,
      same as before this existed.
    - a table with `GeneID`/`Species`/`Celltype` columns, e.g. a marker list
      spanning several reference species. `species` then filters to one
      organism's rows before grouping by `Celltype` -- necessary once one file
      names multiple species' genes, since every other species' IDs are
      guaranteed absent from this target's `.var_names` and would otherwise
      just be reported as "missing" noise. Comparison is underscore/space
      insensitive ('Medicago truncatula' == 'Medicago_truncatula'), matching
      this repo's own species-name convention (`target_species` in
      annotation_config.yaml uses underscores; a hand-curated marker table
      tends to use the binomial with a space).

    An inline list (`markers.report_genes: [gene1, gene2]`) also comes back as
    a single unlabelled group -- there is no celltype to group by there either.
    """
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        genes = [str(v).strip() for v in value if str(v).strip()]
        return [('', genes)] if genes else []

    path = Path(value)
    if not path.is_absolute() and root is not None:
        path = Path(root) / path
    if not path.exists():
        raise FileNotFoundError('markers.report_genes points at {}, which does not '
                                'exist'.format(path))

    rows = [line.split('\t') for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return []
    header = [c.strip() for c in rows[0]]
    if 'GeneID' in header and 'Celltype' in header:
        gi, ci = header.index('GeneID'), header.index('Celltype')
        si = header.index('Species') if 'Species' in header else None
        want = (species or '').strip().replace(' ', '_').lower()
        groups, seen = {}, set()
        for cols in rows[1:]:
            if len(cols) <= gi or len(cols) <= ci:
                continue
            gene = cols[gi].strip()
            if not gene or gene in seen:
                continue
            if si is not None and want:
                row_species = cols[si].strip().replace(' ', '_').lower() if len(cols) > si else ''
                if row_species != want:
                    continue
            seen.add(gene)
            groups.setdefault(cols[ci].strip() or '(unlabelled)', []).append(gene)
        result = [(ct, genes) for ct, genes in groups.items()]
        logger.info('[REVIEW] %d gene(s) across %d cell type group(s) to plot, from %s '
                    '(species=%s)', sum(len(g) for _, g in result), len(result), path,
                    species or 'any')
        return result

    # Original format: first column only, no header handling beyond "won't
    # match .var_names and gets dropped" (a column name is not worth a config
    # key to special-case).
    genes, seen = [], set()
    for cols in rows:
        gene = cols[0].strip()
        if gene and gene not in seen:
            seen.add(gene)
            genes.append(gene)
    logger.info('[REVIEW] %d gene(s) to plot, from %s', len(genes), path)
    return [('', genes)] if genes else []


def gene_expression(adata, genes, keep) -> dict:
    """
    `{gene -> [values]}` on the `corrected_counts` layer, normalised and logged.

    Exactly `sc.pp.normalize_total` followed by `sc.pp.log1p`, but evaluated for
    the requested genes only. **The library size has to come from every gene**,
    which is the whole reason this is not just a column slice: `normalize_total`
    divides each cell by its total count across the full matrix, so subsetting
    genes first would normalise by the wrong denominator and produce numbers
    that look plausible and are not comparable between cells.

    The layer is 62,750 x 52,689 with 123M non-zeros (~1.5 GB on disk), so it is
    read in row chunks straight out of the CSR arrays: one pass yields both the
    per-cell totals and the requested columns, and nothing larger than a chunk
    is ever resident. The equivalent scanpy call would materialise the whole
    normalised matrix to read 28 columns off it.

    `target_sum` follows scanpy's default: the median of the non-zero per-cell
    totals, computed over **all** cells and not just the drawn subset.
    """
    import h5py

    layer = 'corrected_counts'
    var_names = list(adata.var_names)
    position = {g: i for i, g in enumerate(var_names)}
    wanted = [g for g in genes if g in position]
    missing = [g for g in genes if g not in position]
    if missing:
        logger.warning('[REVIEW] %d of %d gene(s) are not in .var_names and will not be '
                       'plotted: %s', len(missing), len(genes), ', '.join(missing[:8]) +
                       (' ...' if len(missing) > 8 else ''))
    if not wanted:
        return {}

    path = getattr(adata, 'filename', None)
    if path is None:
        return _gene_expression_memory(adata, layer, wanted, keep)

    columns = np.array([position[g] for g in wanted])
    # Reverse lookup as a dense array: one indexing operation per chunk instead
    # of a python-level membership test per non-zero.
    col_to_slot = np.full(len(var_names), -1, dtype=np.int32)
    col_to_slot[columns] = np.arange(len(columns))

    n_obs = adata.n_obs
    totals = np.zeros(n_obs, dtype=np.float64)
    values = np.zeros((n_obs, len(columns)), dtype=np.float64)

    with h5py.File(path, 'r') as handle:
        group = handle['layers/{}'.format(layer)]
        if not isinstance(group, h5py.Group):
            raise TypeError('{} is dense; this path expects CSR'.format(layer))
        indptr = group['indptr'][:]
        chunk = 4096
        for start in range(0, n_obs, chunk):
            stop = min(start + chunk, n_obs)
            lo, hi = int(indptr[start]), int(indptr[stop])
            if hi == lo:
                continue
            data = group['data'][lo:hi]
            indices = group['indices'][lo:hi]
            counts = np.diff(indptr[start:stop + 1])
            rows = np.repeat(np.arange(start, stop), counts)
            # bincount rather than reduceat: a cell with no counts at all gives
            # an empty CSR row, which reduceat mishandles and bincount does not.
            totals[start:stop] = np.bincount(rows - start, weights=data,
                                             minlength=stop - start)
            slots = col_to_slot[indices]
            hit = slots >= 0
            if hit.any():
                # CSR holds one entry per (row, column), so assignment is safe.
                values[rows[hit], slots[hit]] = data[hit]

    return _normalise(values, totals, wanted, keep)


def _gene_expression_memory(adata, layer, genes, keep) -> dict:
    """Same computation for an AnnData already in memory."""
    matrix = adata.layers[layer]
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    columns = [list(adata.var_names).index(g) for g in genes]
    block = matrix[:, columns]
    values = np.asarray(block.todense()) if hasattr(block, 'todense') else np.asarray(block)
    return _normalise(values, totals, genes, keep)


def _normalise(values, totals, genes, keep) -> dict:
    """`normalize_total` + `log1p`, given the per-gene counts and per-cell totals."""
    positive = totals[totals > 0]
    if not len(positive):
        raise ValueError('every cell has zero counts in corrected_counts')
    target = float(np.median(positive))
    # Cells with no counts are left at zero rather than dividing by zero --
    # scanpy does the same.
    scale = np.divide(target, totals, out=np.zeros_like(totals), where=totals > 0)
    scaled = np.log1p(values * scale[:, None])
    logger.info('[REVIEW] %d gene(s) normalised to target_sum=%.1f (median per-cell total) '
                'and log1p-transformed', len(genes), target)

    # Sparse, because 95% of the entries are zero: a marker gene is detected in
    # 1-25% of cells here. Dense arrays cost ~7 MB of page for these 28 genes
    # and the zeros carry no information; index/value pairs cost under 1 MB and
    # the JS rebuilds a dense array on first use. Values are rounded to 4 dp,
    # which is well below anything a colour ramp resolves and is the precision
    # the equivalence against scanpy was checked at.
    out = {}
    for i, gene in enumerate(genes):
        column = scaled[keep, i]
        hit = np.nonzero(column)[0]
        out[gene] = {'i': [int(j) for j in hit], 'v': _round(column[hit], 4)}
    return out


def _resolve_umap_key(obsm_keys, preferred=None) -> str:
    """
    Which embedding to draw. Never derived from a latent key -- a latent space
    is not 2-D.
    """
    keys = list(obsm_keys)
    if preferred:
        if preferred not in keys:
            raise KeyError('umap key {!r} not in .obsm ({})'.format(preferred, ', '.join(keys)))
        return preferred
    umaps = [k for k in keys if k.startswith('X_umap')]
    if not umaps:
        raise KeyError('No X_umap* in .obsm ({})'.format(', '.join(keys)))
    for candidate in ('X_umap_scVI', 'X_umap_scANVI', 'X_umap_qc'):
        if candidate in umaps:
            return candidate
    return umaps[0]


def _read_table(path: Path, **kwargs) -> pd.DataFrame:
    """Read one of stage 1's TSVs; an empty frame if it is absent or unreadable."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep='\t', **kwargs)
    except Exception as exc:                          # noqa: BLE001 - a report is not worth a run
        logger.warning('[REVIEW] could not read %s (%s)', path, exc)
        return pd.DataFrame()


def _candidates(annotations: pd.DataFrame, cluster: str) -> list:
    """Every candidate for one geneset, best score first, set-aside ones included."""
    if not len(annotations) or 'cluster' not in annotations:
        return []
    sub = annotations[annotations['cluster'].astype(str) == str(cluster)]
    if not len(sub):
        return []
    if 'score' in sub:
        sub = sub.sort_values('score', ascending=False)
    rows = []
    for _, r in sub.iterrows():
        row = {}
        for col in _CANDIDATE_COLUMNS:
            if col not in sub.columns:
                continue
            value = r[col]
            if isinstance(value, (np.bool_, bool)):
                row[col] = bool(value)
            elif isinstance(value, (np.integer, np.floating, int, float)):
                row[col] = None if pd.isna(value) else float(value)
            else:
                row[col] = '' if pd.isna(value) else str(value)
        rows.append(row)
    return rows


def _seed_index(scores_path: Path, index, keep, quantile, margin) -> dict:
    """
    `{geneset -> [positions in the drawn subset]}` -- the cells stage 1 seeded.

    Sparse on purpose: about 150 cells per geneset, against 60,000 booleans if
    stored densely, and the union of two sparse sets is just as exact.

    Recomputed here rather than cached, because it is derived (`seed_cells` is
    the same function `associate_with_clusters` calls) and a cached copy would
    silently answer for whichever seed parameters were in force when it was
    written.
    """
    if not scores_path.exists():
        return {}
    try:
        scores = pd.read_csv(scores_path, sep='\t', index_col=0)
        scores.columns = [str(c) for c in scores.columns]
        seeds = seed_cells(scores, quantile=quantile, margin=margin).reindex(index)
    except Exception as exc:                          # noqa: BLE001
        logger.warning('[REVIEW] no seed overlay from %s (%s)', scores_path, exc)
        return {}
    values = seeds.to_numpy()
    # Position within the drawn subset, so the JS can index the scatter arrays
    # directly without carrying a second lookup table.
    position = {int(orig): i for i, orig in enumerate(keep)}
    out = {}
    for name in scores.columns:
        hits = np.where(values == name)[0]
        out[str(name)] = [position[int(h)] for h in hits if int(h) in position]
    return out


def build_payload(sample, adata, annotation_dir, methods, cluster_key,
                  umap_key=None, gene_groups=None, max_points=DEFAULT_MAX_POINTS,
                  seed_quantile=0.9, seed_margin=0.25, clade_weights=None) -> dict:
    """Everything the page needs, in one JSON-serialisable dict."""
    annotation_dir = Path(annotation_dir)
    sample_dir = annotation_dir / sample
    umap_key = _resolve_umap_key(adata.obsm.keys(), umap_key)
    logger.info('[REVIEW] Drawing on %s, coloured by %s', umap_key, cluster_key)

    coords = np.asarray(adata.obsm[umap_key])[:, :2]
    clusters = adata.obs[cluster_key].astype(str)
    n_obs = len(clusters)

    # Stratified by cluster rather than uniform: a rare population is exactly
    # what the reviewer is looking for, and uniform sampling is likeliest to
    # drop it.
    keep = np.arange(n_obs)
    if n_obs > max_points:
        rng = np.random.default_rng(0)
        take = []
        for name in clusters.unique():
            idx = np.where(clusters.to_numpy() == name)[0]
            quota = max(50, int(round(max_points * len(idx) / n_obs)))
            take.append(rng.choice(idx, min(quota, len(idx)), replace=False))
        keep = np.sort(np.concatenate(take))
        logger.info('[REVIEW] Subsampled %d of %d cells (stratified by %s)',
                    len(keep), n_obs, cluster_key)

    payload = {
        'sample': sample, 'cluster_key': cluster_key, 'umap_key': umap_key,
        'n_obs': int(n_obs), 'n_vars': int(adata.n_vars), 'n_drawn': int(len(keep)),
        'x': _round(coords[keep, 0]), 'y': _round(coords[keep, 1]),
        'clusters': [str(c) for c in clusters.to_numpy()[keep]],
        'drop_sentinel': DROP, 'ungrouped_sentinel': UNGROUPED,
        'genesets': {}, 'methods': [], 'genes': {}, 'gene_groups': [],
    }
    # Phylogenetic clade map (advisory): {species -> clade}, so the JS
    # can render which clades back each candidate and flag a call resting on a
    # single clade. Empty when no weighting is configured -- the page then shows
    # raw species and no clade column.
    if clade_weights is not None:
        payload['phylo'] = {
            'clade_of': {sp: clade_weights.clade(sp) for sp in clade_weights.clade_of},
            'delta': clade_weights.delta,
        }

    templates, annotation_paths = {}, {}
    for method in methods:
        method_dir = sample_dir / method
        template = _read_table(method_dir / '{}_decisions.tsv'.format(sample),
                               dtype={'cluster': str})
        if not len(template):
            logger.warning('[REVIEW] no decision template for %s/%s; skipping', sample, method)
            continue
        templates[method] = template
        annotation_paths[method] = method_dir / '{}_significant_annotations.tsv'.format(sample)
        annotations = _read_table(annotation_paths[method], dtype={'cluster': str})
        payload['methods'].append(method)

        overlay = _seed_index(method_dir / '{}_geneset_scores.tsv'.format(sample),
                              clusters.index, keep, seed_quantile, seed_margin)

        for _, row in template.iterrows():
            geneset = str(row['cluster'])
            key = '{}|{}'.format(method, geneset)
            record = {
                'method': method, 'geneset': geneset,
                'decision': _text(row, 'decision'),
                'top_po_id': _text(row, 'top_po_id'),
                'top_po_name': _text(row, 'top_po_name'),
                'n_candidates': _count(row, 'n_candidates'),
                'rejected': _text(row, 'rejected'),
                'candidates': _candidates(annotations, geneset),
                'seed_idx': overlay.get(geneset, []),
            }
            record['n_seed_cells'] = _count(row, 'n_seed_cells') or len(record['seed_idx'])
            for col, cast in (('assoc_cluster', str), ('assoc_frac', float)):
                if col in template.columns and not pd.isna(row.get(col)):
                    try:
                        record[col] = cast(row[col])
                    except (TypeError, ValueError):
                        record[col] = ''
            payload['genesets'][key] = record

    # The starting grouping, from stage 1's own top-1 calls: a proposal the
    # reviewer edits, merging genesets that share a top call across methods.
    document = propose(sample, templates, annotations=annotation_paths)
    payload['proposed'] = {
        'groups': {gid: {'po_id': g['po_id'], 'po_name': g['po_name'],
                         'members': ['{}|{}'.format(*m) for m in sorted(g['members'])]}
                   for gid, g in sorted(document.groups.items())},
        'ungrouped': sorted('{}|{}'.format(*m) for m in document.ungrouped),
        'dropped': [],
    }
    payload['proposed_hash'] = document.hash
    payload['parent_hashes'] = {m: file_hash(p) for m, p in annotation_paths.items()
                                if Path(p).exists()}

    # Every ontology term any method proposed anywhere, so a group can be
    # renamed to something its own top-1 call never suggested.
    terms = {}
    for record in payload['genesets'].values():
        for cand in record['candidates']:
            if cand.get('po_id'):
                terms[cand['po_id']] = cand.get('po_name', '')
    payload['terms'] = [{'po_id': k, 'po_name': v} for k, v in sorted(terms.items())]

    consensus = _read_table(sample_dir / '{}_consensus.tsv'.format(sample))
    if len(consensus):
        payload['consensus'] = json.loads(consensus.to_json(orient='records'))

    if gene_groups:
        flat = [g for _, genes in gene_groups for g in genes]
        try:
            payload['genes'] = gene_expression(adata, flat, keep)
            # Groups carried through as (celltype, [genes]), trimmed to genes
            # that actually made it into payload['genes'] (present in
            # .var_names) -- an empty group after trimming is dropped rather
            # than rendered as an empty <optgroup>.
            payload['gene_groups'] = [
                (ct, [g for g in genes if g in payload['genes']])
                for ct, genes in gene_groups]
            payload['gene_groups'] = [(ct, gs) for ct, gs in payload['gene_groups'] if gs]
        except Exception:                             # noqa: BLE001 - genes are not the deliverable
            logger.exception('[REVIEW] gene expression failed; the page will have no gene panel')
    return payload


# --------------------------------------------------------------------- render

_REVIEW_CSS = """
.rv-wrap { border: 1px solid #e5e5e5; border-radius: 8px; padding: 1rem; margin: 1rem 0; }
.rv-controls { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; margin-bottom: .6rem; }
.rv-controls select, .rv-controls input, .rv-controls button {
  font: inherit; font-size: .85rem; padding: .3rem .5rem; border: 1px solid #ccc;
  border-radius: 5px; background: #fff; color: inherit; }
.rv-controls button { cursor: pointer; }
.rv-controls button.primary { background: #2c5aa0; color: #fff; border-color: #2c5aa0; }
.rv-controls label { font-size: .8rem; color: #666; }
.rv-canvas { width: 100%; height: 470px; display: block; border: 1px solid #e5e5e5;
             border-radius: 6px; touch-action: none; }
.rv-status { font-size: .85rem; color: #555; margin: .5rem 0; min-height: 1.4em; }
.rv-legend { font-size: .78rem; color: #666; display: flex; gap: .8rem; flex-wrap: wrap; margin: .5rem 0; }
.sw { display: inline-block; width: .7rem; height: .7rem; border-radius: 2px;
      margin-right: .25rem; vertical-align: -1px; }
.rv-split { display: grid; grid-template-columns: 1fr; gap: 1rem; }
@media (min-width: 980px) { .rv-split { grid-template-columns: 1fr 1fr; } }
.rv-panel { font-size: .85rem; max-height: 620px; overflow-y: auto; border: 1px solid #e5e5e5;
            border-radius: 6px; padding: .4rem .6rem; }
.rv-grp { border-bottom: 1px solid #ededed; padding: .4rem .3rem; }
.rv-grp:last-child { border-bottom: none; }
.rv-grp.sel { background: #eef3fb; border-radius: 5px; }
.rv-grp .hd { cursor: pointer; display: flex; align-items: baseline; gap: .4rem; }
.rv-grp .nm { font-weight: 600; }
.rv-grp .meta, .rv-mem .meta { color: #777; font-size: .78rem; }
.rv-grp .cnt { margin-left: auto; font-variant-numeric: tabular-nums; color: #555; }
.rv-mem { padding: .3rem .3rem .3rem 1rem; border-left: 2px solid #e5e5e5; margin: .25rem 0 .25rem .3rem; }
.rv-mem.off { opacity: .5; }
.rv-mem .who { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .8rem; }
.rv-acts { display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .3rem; }
.rv-acts button, .rv-acts select { font-size: .75rem; padding: .15rem .4rem; border: 1px solid #ccc;
  border-radius: 4px; background: #fff; color: inherit; cursor: pointer; font-family: inherit; }
.rv-cands table { margin: .25rem 0; font-size: .77rem; }
.rv-bar { display: inline-block; height: .5rem; background: #4a7ebb; border-radius: 2px;
          vertical-align: 1px; margin-right: .3rem; min-width: 1px; }
.rv-pill { display: inline-block; padding: .05rem .4rem; border-radius: 999px; font-size: .7rem;
           font-weight: 600; background: #e3f2e3; color: #256029; }
.rv-pill.weak { background: #fdecea; color: #9b1c1c; }
.rv-pill.mid  { background: #fff6e5; color: #7a4a00; }
.rv-dispute td.hot { background: #fff6e5; font-weight: 600; }
.rv-empty { color: #999; font-style: italic; }
.rv-tray { margin-top: .6rem; }
@media (prefers-color-scheme: dark) {
  .rv-wrap, .rv-canvas, .rv-panel { border-color: #2e3238; }
  .rv-controls select, .rv-controls input, .rv-controls button,
  .rv-acts button, .rv-acts select { background: #1e2126; border-color: #3a3f46; }
  .rv-controls button.primary { background: #3f6fb5; border-color: #3f6fb5; }
  .rv-status, .rv-legend, .rv-controls label, .rv-grp .meta, .rv-mem .meta { color: #9aa0a6; }
  .rv-grp { border-bottom-color: #262a30; }
  .rv-grp.sel { background: #1e2a3d; }
  .rv-mem { border-left-color: #2e3238; }
  .rv-dispute td.hot { background: #2b2210; }
  .rv-pill { background: #16301b; color: #7fca8b; }
  .rv-pill.weak { background: #3a1a1a; color: #e79a9a; }
  .rv-pill.mid { background: #2b2210; color: #e8c777; }
}
"""

_REVIEW_JS = r"""
(function () {
  var D = window.__REVIEW__;
  if (!D) return;

  var cv = document.getElementById('rv-canvas'), ctx = cv.getContext('2d');
  var elColor = document.getElementById('rv-color');
  var elGene = document.getElementById('rv-gene');
  var elPanel = document.getElementById('rv-panel');
  var elStatus = document.getElementById('rv-status');
  var elLegend = document.getElementById('rv-legend');

  // ---------------------------------------------------------------- state
  // A flat assignment map, not a nested tree: every geneset is in exactly one
  // place, so "move it" is one assignment and no state can describe a geneset
  // being in two groups at once.
  var assign = {};                 // "method|geneset" -> groupId | DROP | UNGROUPED
  var meta = {};                   // groupId -> {po_id, po_name, note}
  var sel = null;                  // selected groupId
  var open = {};                   // expanded member keys
  // Which members are drawn. A VIEW concern, deliberately separate from
  // `assign`: hiding a geneset to look at its neighbours must never change
  // what gets exported, and conflating the two would make "I hid it to see
  // past it" and "I removed it from this cell type" the same gesture.
  var hidden = {};                 // member key -> true when not drawn

  Object.keys(D.proposed.groups).forEach(function (gid) {
    var g = D.proposed.groups[gid];
    meta[gid] = { po_id: g.po_id, po_name: g.po_name, note: '' };
    g.members.forEach(function (k) { assign[k] = gid; });
  });
  D.proposed.ungrouped.forEach(function (k) { assign[k] = D.ungrouped_sentinel; });
  Object.keys(D.genesets).forEach(function (k) {
    if (!(k in assign)) assign[k] = D.ungrouped_sentinel;
  });

  function groupIds() {
    var seen = {};
    Object.keys(assign).forEach(function (k) {
      var g = assign[k];
      if (g !== D.drop_sentinel && g !== D.ungrouped_sentinel) seen[g] = 1;
    });
    return Object.keys(seen).sort(function (a, b) {
      var na = membersOf(b).length - membersOf(a).length;
      return na !== 0 ? na : (a < b ? -1 : 1);
    });
  }
  function membersOf(gid) {
    return Object.keys(assign).filter(function (k) { return assign[k] === gid; }).sort();
  }
  function methodsOf(gid) {
    var seen = {};
    membersOf(gid).forEach(function (k) { seen[D.genesets[k].method] = 1; });
    return Object.keys(seen).sort();
  }

  // Union of the members' stage-1 seed cells. A real union of real sets --
  // every index came from `seed_cells` in stage 1. NOT the final seed count:
  // refinement assigns at group level, which needs the score matrices.
  function unionSeeds(gid, visibleOnly) {
    var hit = {}, n = 0;
    (visibleOnly ? visibleOf(gid) : membersOf(gid)).forEach(function (k) {
      (D.genesets[k].seed_idx || []).forEach(function (i) {
        if (!(i in hit)) { hit[i] = 0; n++; }
        hit[i] += 1;
      });
    });
    return { set: hit, size: n };
  }
  function visibleOf(gid) {
    return membersOf(gid).filter(function (k) { return !hidden[k]; });
  }

  // How many distinct METHODS seeded each cell, over the VISIBLE members only,
  // which is the quantity that means something -- two hotspot modules agreeing
  // is correlated modules, not independent evidence.
  function methodSupport(gid) {
    var per = {};
    visibleOf(gid).forEach(function (k) {
      var m = D.genesets[k].method;
      (D.genesets[k].seed_idx || []).forEach(function (i) {
        (per[i] = per[i] || {})[m] = 1;
      });
    });
    var out = {};
    Object.keys(per).forEach(function (i) { out[i] = Object.keys(per[i]).length; });
    return out;
  }

  // How scattered a geneset's seed cells are across the clustering. This is
  // the "is this set very broad?" question, answered from data stage 1 already
  // computed -- the seed cells and the cluster labels -- rather than by eye.
  // A set concentrated in one cluster is describing something; one spread over
  // eight is describing the tissue.
  // Do a group's genesets describe the SAME population? Each member is
  // summarised by the cluster it concentrates in, and the group is coherent
  // when they agree. This catches something no per-member figure can: on this
  // data `root_endodermis` pools nine genesets that are each >=92%
  // concentrated -- individually tight, so nothing looks broad -- but six of
  // them sit in cluster 9 and three in cluster 2. That is two cell types
  // sharing one automatic call, and it should be split rather than trained as
  // one class.
  function coherence(gid) {
    var counts = {};
    membersOf(gid).forEach(function (k) {
      var b = breadth(k);
      if (b.n) counts[b.top] = (counts[b.top] || 0) + 1;
    });
    var names = Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a]; });
    return { clusters: names, counts: counts, split: names.length > 1 };
  }

  // Genes arrive as index/value pairs -- 95% of entries are zero, so the dense
  // form would be most of the page. Rebuilt densely once per gene, on demand.
  var geneCache = {};
  function geneValues(name) {
    if (name in geneCache) return geneCache[name];
    var g = D.genes[name];
    if (!g) return null;
    var arr = new Float64Array(D.x.length);
    for (var k = 0; k < g.i.length; k++) arr[g.i[k]] = g.v[k];
    geneCache[name] = arr;
    return arr;
  }

  var breadthCache = {};
  function breadth(key) {
    if (key in breadthCache) return breadthCache[key];
    var idx = D.genesets[key].seed_idx || [], counts = {}, n = idx.length;
    idx.forEach(function (i) {
      var c = D.clusters[i];
      counts[c] = (counts[c] || 0) + 1;
    });
    var names = Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a]; });
    var out = n ? { n: n, k: names.length, top: names[0], frac: counts[names[0]] / n }
                : { n: 0, k: 0, top: '', frac: 0 };
    breadthCache[key] = out;
    return out;
  }

  // ---------------------------------------------------------------- colour
  function catColor(i, alpha) {
    if (i < 0) return 'rgba(170,170,170,.25)';
    return 'hsla(' + ((i * 137.508) % 360).toFixed(1) + ',65%,52%,' + (alpha || .8) + ')';
  }
  function seqColor(t) {
    t = Math.max(0, Math.min(1, t));
    return 'hsla(215,' + (25 + 50 * t).toFixed(0) + '%,' + (88 - 48 * t).toFixed(0) + '%,.9)';
  }
  var uniqClusters = (function () {
    var seen = {}, out = [];
    for (var i = 0; i < D.clusters.length; i++) {
      if (!(D.clusters[i] in seen)) { seen[D.clusters[i]] = 1; out.push(D.clusters[i]); }
    }
    out.sort(function (a, b) {
      var na = parseFloat(a), nb = parseFloat(b);
      if (!isNaN(na) && !isNaN(nb)) return na - nb;
      return a < b ? -1 : (a > b ? 1 : 0);
    });
    return out;
  })();
  var clusterIdx = {};
  uniqClusters.forEach(function (c, i) { clusterIdx[c] = i; });

  var view = null;
  function fit() {
    var dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    cv.width = w * dpr; cv.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    var xmin = Math.min.apply(null, D.x), xmax = Math.max.apply(null, D.x);
    var ymin = Math.min.apply(null, D.y), ymax = Math.max.apply(null, D.y);
    var pad = 12, s = Math.min((w - 2 * pad) / ((xmax - xmin) || 1),
                               (h - 2 * pad) / ((ymax - ymin) || 1));
    view = { s: s, xmin: xmin, ymax: ymax, pad: pad,
             ox: (w - 2 * pad - s * (xmax - xmin)) / 2,
             oy: (h - 2 * pad - s * (ymax - ymin)) / 2 };
  }
  function px(i) { return view.pad + view.ox + (D.x[i] - view.xmin) * view.s; }
  function py(i) { return view.pad + view.oy + (view.ymax - D.y[i]) * view.s; }

  var SUPPORT_COLORS = ['rgba(170,170,170,.25)', 'hsla(210,55%,62%,.85)',
                        'hsla(190,60%,45%,.9)', 'hsla(150,60%,38%,.92)',
                        'hsla(90,60%,36%,.94)', 'hsla(40,80%,45%,.95)',
                        'hsla(8,78%,48%,.97)'];

  function currentColors() {
    var n = D.x.length, cols = new Array(n), i;
    var mode = elColor.value;
    if (mode === 'gene') {
      var g = geneValues(elGene.value);
      if (!g) { for (i = 0; i < n; i++) cols[i] = 'rgba(170,170,170,.3)'; return cols; }
      var lo = Infinity, hi = -Infinity;
      for (i = 0; i < n; i++) { if (g[i] < lo) lo = g[i]; if (g[i] > hi) hi = g[i]; }
      // Undetected cells stay grey rather than taking the bottom of the ramp:
      // "no counts here" and "the lowest non-zero level" are different claims,
      // and on a gene detected in 2% of cells the ramp's low end would
      // otherwise be the entire background.
      for (i = 0; i < n; i++) {
        cols[i] = g[i] > 0 ? seqColor((g[i] - lo) / ((hi - lo) || 1))
                           : 'rgba(170,170,170,.16)';
      }
      return cols;
    }
    if (mode === 'group') {
      for (i = 0; i < n; i++) cols[i] = 'rgba(170,170,170,.16)';
      if (sel) {
        var sup = methodSupport(sel);
        Object.keys(sup).forEach(function (i) {
          cols[i] = SUPPORT_COLORS[Math.min(sup[i], SUPPORT_COLORS.length - 1)];
        });
      }
      return cols;
    }
    if (mode === 'members') {
      // One hue per visible geneset, so several can be compared at once. A cell
      // seeded by more than one takes the colour of the last drawn -- stated in
      // the legend rather than silently blended, since a blend would invent a
      // colour that means nothing.
      for (i = 0; i < n; i++) cols[i] = 'rgba(170,170,170,.16)';
      if (sel) {
        visibleOf(sel).forEach(function (k, mi) {
          var c = catColor(mi, .85);
          (D.genesets[k].seed_idx || []).forEach(function (i) { cols[i] = c; });
        });
      }
      return cols;
    }
    if (mode === 'allgroups') {
      for (i = 0; i < n; i++) cols[i] = 'rgba(170,170,170,.16)';
      groupIds().forEach(function (gid, gi) {
        var u = unionSeeds(gid);
        Object.keys(u.set).forEach(function (i) { cols[i] = catColor(gi, .8); });
      });
      return cols;
    }
    for (i = 0; i < n; i++) cols[i] = catColor(clusterIdx[D.clusters[i]], .72);
    return cols;
  }

  function draw() {
    if (!view) fit();
    ctx.clearRect(0, 0, cv.clientWidth, cv.clientHeight);
    var cols = currentColors(), n = D.x.length;
    var r = n > 40000 ? 1.1 : (n > 15000 ? 1.6 : 2.4);
    // Faint points first so a selection is never buried under its context.
    var faint = [], front = [];
    for (var i = 0; i < n; i++) (cols[i].indexOf('170,170,170') >= 0 ? faint : front).push(i);
    [faint, front].forEach(function (group) {
      for (var k = 0; k < group.length; k++) {
        var i = group[k];
        ctx.fillStyle = cols[i];
        ctx.beginPath(); ctx.arc(px(i), py(i), r, 0, 6.283); ctx.fill();
      }
    });
    drawLegend();
  }

  function drawLegend() {
    var mode = elColor.value, out = [];
    if (mode === 'cluster') {
      uniqClusters.forEach(function (c, i) {
        out.push('<span><span class="sw" style="background:' + catColor(i, .9) + '"></span>' + c + '</span>');
      });
    } else if (mode === 'group') {
      if (!sel) {
        out.push('<span class="rv-empty">select a cell type on the right</span>');
      } else {
        var vis = visibleOf(sel), seen = {};
        vis.forEach(function (k) { seen[D.genesets[k].method] = 1; });
        var nm = Object.keys(seen).length;
        out.push('<span class="rv-empty">methods independently seeding each cell:</span>');
        for (var k = 1; k <= Math.max(1, Math.min(nm, 6)); k++) {
          out.push('<span><span class="sw" style="background:' + SUPPORT_COLORS[k] + '"></span>' + k + '</span>');
        }
        if (vis.length < membersOf(sel).length) {
          out.push('<span class="rv-empty">(' + (membersOf(sel).length - vis.length) + ' geneset(s) hidden)</span>');
        }
      }
    } else if (mode === 'members') {
      if (!sel) {
        out.push('<span class="rv-empty">select a cell type on the right</span>');
      } else {
        visibleOf(sel).forEach(function (key, mi) {
          var gs = D.genesets[key];
          out.push('<span><span class="sw" style="background:' + catColor(mi, .9) + '"></span>' +
            esc(gs.method) + ':' + esc(gs.geneset) + '</span>');
        });
        if (!visibleOf(sel).length) out.push('<span class="rv-empty">every geneset hidden</span>');
        else out.push('<span class="rv-empty">overlaps take the last colour drawn</span>');
      }
    } else if (mode === 'allgroups') {
      groupIds().forEach(function (gid, gi) {
        out.push('<span><span class="sw" style="background:' + catColor(gi, .9) + '"></span>' +
                 esc(meta[gid] && meta[gid].po_name || gid) + '</span>');
      });
    } else if (mode === 'gene') {
      var gv = geneValues(elGene.value);
      var det = 0, mx = 0;
      if (gv) for (var gi = 0; gi < gv.length; gi++) { if (gv[gi] > 0) { det++; if (gv[gi] > mx) mx = gv[gi]; } }
      out.push('<span>low <span class="sw" style="background:' + seqColor(0) + '"></span>' +
               '<span class="sw" style="background:' + seqColor(.5) + '"></span>' +
               '<span class="sw" style="background:' + seqColor(1) + '"></span> ' +
               mx.toFixed(2) + '</span>');
      out.push('<span><span class="sw" style="background:rgba(170,170,170,.4)"></span>not detected</span>');
      out.push('<span class="rv-empty">detected in ' +
               (100 * det / (gv ? gv.length : 1)).toFixed(1) + '% of drawn cells &middot; ' +
               'normalize_total + log1p on corrected_counts</span>');
    }
    elLegend.innerHTML = out.join('');
  }

  function esc(s) {
    return String(s === null || s === undefined ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function num(v, d) {
    return (v === null || v === undefined) ? '&mdash;' : Number(v).toFixed(d === undefined ? 1 : d);
  }

  // ---------------------------------------------------------------- panel
  // Clades backing a candidate, from its support_species_list and the payload's
  // {species -> clade} map. Advisory (Rung 1): shows whether support is broad
  // (several clades) or rests on one clade, and names them so a grass-only call
  // is visible. Nothing here changes the score -- that correction already
  // happened server-side and is in `weighted_species`.
  function cladeProfile(c) {
    var map = (D.phylo && D.phylo.clade_of) || null;
    var list = (c.support_species_list || '').split(';').filter(function (s) { return s; });
    if (!map || !list.length) return null;
    var by = {};
    list.forEach(function (sp) { var cl = map[sp] || sp; (by[cl] = by[cl] || []).push(sp); });
    var clades = Object.keys(by).sort();
    return { clades: clades, by: by, single: clades.length <= 1, n: list.length };
  }
  function cladeCell(c) {
    var p = cladeProfile(c);
    if (!p) return '<td class="num">' + num(c.support_species, 0) + '</td>';
    var label = p.clades.map(function (cl) {
      var k = p.by[cl].length; return esc(cl) + (k > 1 ? ' ×' + k : '');
    }).join(', ');
    var cls = p.single ? 'rv-pill weak' : 'rv-pill';
    var w = (c.weighted_species === undefined || c.weighted_species === null)
      ? '' : ' <span class="meta">w=' + Number(c.weighted_species).toFixed(2) + '</span>';
    return '<td><span class="' + cls + '">' + p.clades.length + ' clade' +
      (p.clades.length === 1 ? '' : 's') + '</span> ' + label + w + '</td>';
  }

  function candidateTable(gs) {
    if (!gs.candidates || !gs.candidates.length) {
      return '<p class="rv-empty">No candidate cleared the significance threshold.</p>';
    }
    var hasPhylo = !!(D.phylo && D.phylo.clade_of);
    var rows = gs.candidates.map(function (c) {
      var flags = [];
      if (c.too_general) flags.push('too general');
      if (c.tissue_match === false) flags.push('wrong tissue');
      return '<tr><td>' + esc(c.po_name) + '<br><span class="meta">' + esc(c.po_id) + '</span></td>' +
        '<td class="num">' + num(c.score) + '</td>' +
        '<td class="num">' + (c.best_q === null || c.best_q === undefined ? '&mdash;' : Number(c.best_q).toExponential(1)) + '</td>' +
        '<td class="num">' + num(c.total_X, 0) + '</td>' +
        cladeCell(c) +
        '<td>' + (flags.length ? '<span class="meta">' + flags.join('; ') + '</span>' : '') + '</td>' +
        '<td><button data-use="' + esc(c.po_id) + '" data-usename="' + esc(c.po_name) + '">use</button></td></tr>';
    });
    return '<div class="rv-cands"><table><thead><tr><th>candidate</th><th class="num">score</th>' +
      '<th class="num">q</th><th class="num">X</th><th>' + (hasPhylo ? 'clade support' : 'spp') +
      '</th><th>set aside</th><th></th>' +
      '</tr></thead><tbody>' + rows.join('') + '</tbody></table></div>';
  }

  function memberHtml(key, gid, mi) {
    var gs = D.genesets[key], isOpen = !!open[key], isHidden = !!hidden[key];
    var b = breadth(key);
    var inGroup = gid !== D.drop_sentinel && gid !== D.ungrouped_sentinel;

    // Spread across the clustering, from the seeds and cluster labels the
    // payload already carries. A set concentrated in one cluster is describing
    // something; one spread over eight is describing the tissue. The `broad`
    // marker is a display hint at an arbitrary cut, not a rule -- the numbers
    // beside it are what to judge on.
    var spread = b.n
      ? b.k + ' cluster' + (b.k === 1 ? '' : 's') + ' &middot; ' +
        Math.round(b.frac * 100) + '% in ' + esc(b.top) +
        (b.frac < 0.5 || b.k > 6 ? ' <span class="rv-pill weak">broad</span>' : '')
      : '<span class="rv-empty">no seed cells</span>';

    var swatch = (mi !== undefined && !isHidden)
      ? '<span class="sw" style="background:' + catColor(mi, .9) + '"></span>' : '';

    var target = '<select data-move="' + esc(key) + '"><option value="">move to&hellip;</option>' +
      groupIds().filter(function (g) { return g !== gid; }).map(function (g) {
        return '<option value="' + esc(g) + '">' + esc(meta[g] && meta[g].po_name || g) + '</option>';
      }).join('') + '</select>';

    return '<div class="rv-mem' + (isHidden ? ' off' : '') + '" data-mem="' + esc(key) + '">' +
      swatch + '<span class="who">' + esc(gs.method) + ':' + esc(gs.geneset) + '</span> ' +
      // Stage 1's own seed count, over the whole dataset. The spread figures
      // beside it come from the drawn subset, which is ~96% of cells here --
      // the percentages are stable at that rate, the raw count would not be.
      '<span class="meta">' + _fmtInt(gs.n_seed_cells || b.n) + ' seeds &middot; ' + spread + '</span>' +
      '<div class="rv-acts">' +
        '<button data-vis="' + esc(key) + '">' + (isHidden ? 'show' : 'hide') + '</button>' +
        '<button data-solo="' + esc(key) + '">only this</button>' +
        '<button data-toggle="' + esc(key) + '">' + (isOpen ? 'less' : 'evidence') +
          ' (' + gs.n_candidates + ')</button>' +
        target +
        '<button data-own="' + esc(key) + '">own group</button>' +
        (inGroup ? '<button data-ungroup="' + esc(key) + '">remove from group</button>' : '') +
        '<button data-drop="' + esc(key) + '">drop</button>' +
      '</div>' +
      (isOpen ? candidateTable(gs) : '') + '</div>';
  }

  function _fmtInt(v) { return Number(v || 0).toLocaleString(); }

  function renderPanel() {
    var out = [], ids = groupIds();
    out.push('<div class="rv-grp"><div class="meta"><strong>' + ids.length +
      ' cell type(s)</strong> from ' + Object.keys(D.genesets).length + ' genesets across ' +
      D.methods.length + ' methods</div></div>');

    ids.forEach(function (gid) {
      var members = membersOf(gid), ms = methodsOf(gid), u = unionSeeds(gid);
      var isSel = sel === gid;
      var pill = ms.length >= 4 ? 'rv-pill' : (ms.length >= 2 ? 'rv-pill mid' : 'rv-pill weak');
      var coh = coherence(gid);
      var head = '<div class="hd" data-grp="' + esc(gid) + '">' +
        '<span class="nm">' + esc(meta[gid].po_name || gid) + '</span>' +
        '<span class="meta">' + esc(meta[gid].po_id || 'unnamed') + '</span>' +
        '<span class="' + pill + '">' + ms.length + '/' + D.methods.length + '</span>' +
        (coh.split ? '<span class="rv-pill weak" title="its genesets concentrate in ' +
          coh.clusters.length + ' different clusters">split?</span>' : '') +
        '<span class="cnt">' + _fmtInt(u.size) + '</span></div>';
      var body = '';
      if (isSel) {
        var vis = visibleOf(gid), nHidden = members.length - vis.length;
        var visU = unionSeeds(gid, true);
        var where = coh.clusters.map(function (c) {
          return coh.counts[c] + ' in ' + esc(c); }).join(', ');
        body = '<div class="meta">' + esc(ms.join(', ')) + ' &middot; ' + members.length +
          ' geneset(s) &middot; union of stage-1 seeds: ' + _fmtInt(u.size) + ' cells ' +
          '<em>(not the final seed count &mdash; run --dry-run)</em>' +
          (nHidden ? ' &middot; showing ' + vis.length + '/' + members.length +
                     ' (' + _fmtInt(visU.size) + ' cells)' : '') + '</div>' +
          '<div class="meta">genesets concentrate in: ' + where +
          (coh.split ? ' &mdash; <strong>these may be different populations sharing one call. '
                     + 'Consider splitting.</strong>' : '') + '</div>' +
          '<div class="rv-acts">' +
            '<select data-rename="' + esc(gid) + '"><option value="">rename&hellip;</option>' +
              D.terms.map(function (t) {
                return '<option value="' + esc(t.po_id) + '|' + esc(t.po_name) + '">' +
                  esc(t.po_name) + ' (' + esc(t.po_id) + ')</option>'; }).join('') +
              '<option value="__NONE__">leave unnamed (unknown_ class)</option>' +
            '</select>' +
            (nHidden ? '<button data-showall="' + esc(gid) + '">show all ' +
                       members.length + '</button>' : '') +
            '<button data-dropgrp="' + esc(gid) + '">drop whole group</button>' +
          '</div>' +
          members.map(function (k) {
            // Index among the VISIBLE members, so the swatch matches the hue
            // the canvas uses in "each geneset separately" mode.
            return memberHtml(k, gid, vis.indexOf(k) >= 0 ? vis.indexOf(k) : undefined);
          }).join('');
      }
      out.push('<div class="rv-grp' + (isSel ? ' sel' : '') + '">' + head + body + '</div>');
    });

    ['__UNGROUPED__', '__DROP__'].forEach(function (bucket) {
      var sentinel = bucket === '__DROP__' ? D.drop_sentinel : D.ungrouped_sentinel;
      var keys = Object.keys(assign).filter(function (k) { return assign[k] === sentinel; }).sort();
      var isSel = sel === sentinel;
      var title = bucket === '__DROP__'
        ? 'DROPPED &mdash; contribute no seed cells at all'
        : 'UNGROUPED &mdash; each becomes its own unknown_ class';
      out.push('<div class="rv-grp rv-tray' + (isSel ? ' sel' : '') + '">' +
        '<div class="hd" data-grp="' + sentinel + '"><span class="nm">' + title + '</span>' +
        '<span class="cnt">' + keys.length + '</span></div>' +
        (isSel ? keys.map(function (k) { return memberHtml(k, sentinel); }).join('')
                 || '<p class="rv-empty">nothing here</p>' : '') + '</div>');
    });

    elPanel.innerHTML = out.join('');
    wire();
    status();
  }

  function wire() {
    var q = function (s) { return Array.prototype.slice.call(elPanel.querySelectorAll(s)); };
    q('[data-grp]').forEach(function (n) {
      n.addEventListener('click', function () {
        var g = n.getAttribute('data-grp');
        sel = (sel === g) ? null : g;
        if (sel && elColor.value === 'cluster') elColor.value = 'group';
        renderPanel(); draw();
      });
    });
    q('[data-toggle]').forEach(function (b) {
      b.addEventListener('click', function (e) {
        e.stopPropagation();
        var k = b.getAttribute('data-toggle');
        open[k] = !open[k];
        renderPanel();
      });
    });
    // Visibility. Never touches `assign` -- looking past a geneset is not the
    // same act as removing it, and the export must not be able to tell.
    q('[data-vis]').forEach(function (b) {
      b.addEventListener('click', function (e) {
        e.stopPropagation();
        var k = b.getAttribute('data-vis');
        if (hidden[k]) delete hidden[k]; else hidden[k] = true;
        renderPanel(); draw();
      });
    });
    q('[data-solo]').forEach(function (b) {
      b.addEventListener('click', function (e) {
        e.stopPropagation();
        var k = b.getAttribute('data-solo'), gid = assign[k];
        membersOf(gid).forEach(function (o) { hidden[o] = (o !== k); });
        delete hidden[k];
        if (elColor.value !== 'members') elColor.value = 'members';
        renderPanel(); draw();
      });
    });
    q('[data-showall]').forEach(function (b) {
      b.addEventListener('click', function (e) {
        e.stopPropagation();
        membersOf(b.getAttribute('data-showall')).forEach(function (k) { delete hidden[k]; });
        renderPanel(); draw();
      });
    });
    // Removing from a group is NOT dropping: the geneset goes back to the
    // ungrouped tray, where it still gets its own unknown_ class. Dropping
    // says it is junk and contributes no seed cells at all.
    q('[data-ungroup]').forEach(function (b) {
      b.addEventListener('click', function (e) {
        e.stopPropagation();
        var k = b.getAttribute('data-ungroup');
        assign[k] = D.ungrouped_sentinel;
        delete hidden[k];
        renderPanel(); draw();
      });
    });
    q('[data-drop]').forEach(function (b) {
      b.addEventListener('click', function (e) {
        e.stopPropagation();
        var k = b.getAttribute('data-drop');
        assign[k] = D.drop_sentinel;
        delete hidden[k];
        renderPanel(); draw();
      });
    });
    q('[data-own]').forEach(function (b) {
      b.addEventListener('click', function (e) {
        e.stopPropagation();
        var k = b.getAttribute('data-own'), gs = D.genesets[k];
        var gid = (gs.method + '_' + gs.geneset).replace(/[^A-Za-z0-9]+/g, '_').toLowerCase();
        while (gid in meta) gid += '_';
        meta[gid] = { po_id: gs.top_po_id || '', po_name: gs.top_po_name || '', note: '' };
        assign[k] = gid; sel = gid;
        renderPanel(); draw();
      });
    });
    q('[data-move]').forEach(function (s) {
      s.addEventListener('change', function (e) {
        e.stopPropagation();
        if (!s.value) return;
        assign[s.getAttribute('data-move')] = s.value;
        renderPanel(); draw();
      });
    });
    q('[data-rename]').forEach(function (s) {
      s.addEventListener('change', function (e) {
        e.stopPropagation();
        if (!s.value) return;
        var gid = s.getAttribute('data-rename');
        if (s.value === '__NONE__') { meta[gid].po_id = ''; meta[gid].po_name = ''; }
        else { var p = s.value.split('|'); meta[gid].po_id = p[0]; meta[gid].po_name = p[1] || p[0]; }
        renderPanel(); draw();
      });
    });
    q('[data-dropgrp]').forEach(function (b) {
      b.addEventListener('click', function (e) {
        e.stopPropagation();
        var gid = b.getAttribute('data-dropgrp');
        membersOf(gid).forEach(function (k) { assign[k] = D.drop_sentinel; });
        sel = null; renderPanel(); draw();
      });
    });
    q('[data-use]').forEach(function (b) {
      b.addEventListener('click', function (e) {
        e.stopPropagation();
        // Adopting a candidate renames the group this geneset is in -- that is
        // what "this module is actually X" means when the module is grouped.
        var mem = b.closest('.rv-mem'), key = mem && mem.getAttribute('data-mem');
        var gid = key && assign[key];
        if (!gid || gid === D.drop_sentinel || gid === D.ungrouped_sentinel) {
          // Ungrouped: adopting a term promotes it to a group of its own.
          var gs = D.genesets[key];
          gid = (gs.method + '_' + gs.geneset).replace(/[^A-Za-z0-9]+/g, '_').toLowerCase();
          while (gid in meta) gid += '_';
          meta[gid] = { po_id: '', po_name: '', note: '' };
          assign[key] = gid; sel = gid;
        }
        meta[gid].po_id = b.getAttribute('data-use');
        meta[gid].po_name = b.getAttribute('data-usename');
        renderPanel(); draw();
      });
    });
  }

  function status() {
    var ids = groupIds();
    var named = ids.filter(function (g) { return meta[g].po_id; }).length;
    var ung = Object.keys(assign).filter(function (k) { return assign[k] === D.ungrouped_sentinel; }).length;
    var drp = Object.keys(assign).filter(function (k) { return assign[k] === D.drop_sentinel; }).length;
    elStatus.innerHTML = D.n_drawn.toLocaleString() + ' of ' + D.n_obs.toLocaleString() +
      ' cells drawn &middot; ' + ids.length + ' cell type(s) (' + named + ' named, ' +
      (ids.length - named) + ' unnamed) &middot; ' + ung + ' ungrouped &middot; ' + drp + ' dropped' +
      (sel && meta[sel] ? ' &middot; selected <strong>' + esc(meta[sel].po_name || sel) + '</strong>' : '');
  }

  // ---------------------------------------------------------------- export
  function exportGroups() {
    var groups = {};
    groupIds().forEach(function (gid) {
      groups[gid] = {
        po_id: meta[gid].po_id || '', po_name: meta[gid].po_name || '',
        note: meta[gid].note || '',
        members: membersOf(gid).map(function (k) { return k.split('|'); })
      };
    });
    var bucket = function (s) {
      return Object.keys(assign).filter(function (k) { return assign[k] === s; })
        .sort().map(function (k) { return k.split('|'); });
    };
    var payload = {
      schema: 'groups/1', sample: D.sample,
      // Recorded so `check_parents` can refuse a grouping derived from
      // annotation tables that have since been regenerated.
      parent_hashes: D.parent_hashes || {},
      groups: groups,
      dropped: bucket(D.drop_sentinel),
      ungrouped: bucket(D.ungrouped_sentinel)
    };
    var blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = D.sample + '_groups.json';
    a.click();
    URL.revokeObjectURL(a.href);
    elStatus.innerHTML = 'Exported <code>' + D.sample + '_groups.json</code> &mdash; put it in ' +
      '<code>results/annotation/results/' + D.sample + '/</code>, then run ' +
      '<code>refine.main --dry-run</code>';
  }

  elColor.addEventListener('change', function () {
    elGene.style.display = elColor.value === 'gene' ? '' : 'none';
    draw();
  });
  elGene.addEventListener('change', draw);
  document.getElementById('rv-export').addEventListener('click', exportGroups);
  document.getElementById('rv-reset').addEventListener('click', function () {
    if (!confirm('Discard all grouping edits and return to the proposal?')) return;
    assign = {}; meta = {}; sel = null; open = {};
    Object.keys(D.proposed.groups).forEach(function (gid) {
      var g = D.proposed.groups[gid];
      meta[gid] = { po_id: g.po_id, po_name: g.po_name, note: '' };
      g.members.forEach(function (k) { assign[k] = gid; });
    });
    Object.keys(D.genesets).forEach(function (k) {
      if (!(k in assign)) assign[k] = D.ungrouped_sentinel;
    });
    renderPanel(); draw();
  });
  window.addEventListener('resize', function () { fit(); draw(); });

  Array.prototype.forEach.call(document.querySelectorAll('[data-jump]'), function (row) {
    row.style.cursor = 'pointer';
    row.addEventListener('click', function () {
      var target = row.getAttribute('data-jump'), hit = null;
      groupIds().forEach(function (gid) {
        membersOf(gid).forEach(function (k) {
          if (!hit && String(D.genesets[k].assoc_cluster) === target) hit = gid;
        });
      });
      if (!hit) { elStatus.textContent = 'No group is associated with cluster ' + target; return; }
      sel = hit; elColor.value = 'group'; elGene.style.display = 'none';
      renderPanel(); draw();
      cv.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
  });

  elGene.style.display = 'none';
  renderPanel();
  fit(); draw();
})();
"""


def _consensus_section(payload: dict) -> str:
    """Where the methods disagree, most-disputed first."""
    rows = payload.get('consensus') or []
    if not rows:
        return ''
    method_cols = [k for k in rows[0] if k not in ('cluster', 'n_methods', 'n_distinct')]
    rows = sorted(rows, key=lambda r: (-(r.get('n_distinct') or 0), str(r.get('cluster'))))
    disputed = sum(1 for r in rows if (r.get('n_distinct') or 0) > 1)

    head = ''.join('<th>{}</th>'.format(_esc(c)) for c in method_cols)
    body = []
    for r in rows:
        n_distinct = r.get('n_distinct') or 0
        cells = ''.join('<td>{}</td>'.format(html.escape(str(r.get(c) or '')) or '&mdash;')
                        for c in method_cols)
        body.append(
            '<tr data-jump="{cl}"><td><strong>{cl}</strong></td><td class="num">{nm}</td>'
            '<td class="num {hot}">{nd}</td>{cells}</tr>'.format(
                cl=html.escape(str(r.get('cluster'))), nm=_fmt_int(r.get('n_methods')),
                nd=_fmt_int(n_distinct), hot='hot' if n_distinct > 1 else '', cells=cells))

    return """
<h2>Where the methods disagree</h2>
<p class="sub">{disputed} of {total} clusters carry more than one distinct call. Sorted by how
contested they are; click a row to select the cell type grouped onto that cluster.</p>
<div class="rv-wrap"><table class="rv-dispute">
<thead><tr><th>cluster</th><th class="num">methods</th><th class="num">distinct</th>{head}</tr></thead>
<tbody>{body}</tbody></table></div>
""".format(disputed=disputed, total=len(rows), head=head, body=''.join(body))


def render(payload: dict) -> str:
    """The complete review page HTML for a `build_payload` payload."""
    if not payload.get('genesets'):
        raise ValueError('No stage 1 output to review')

    gene_groups = payload.get('gene_groups') or []
    if len(gene_groups) == 1 and gene_groups[0][0] == '':
        # Inline list / headerless file: nothing to group by, so a flat list
        # of <option>s reads better than a single unlabelled <optgroup>.
        gene_options = ''.join(
            '<option value="{g}">{g}</option>'.format(g=html.escape(g))
            for g in sorted(gene_groups[0][1]))
    else:
        gene_options = ''.join(
            '<optgroup label="{ct}">{opts}</optgroup>'.format(
                ct=html.escape(celltype),
                opts=''.join('<option value="{g}">{g}</option>'.format(g=html.escape(g))
                             for g in sorted(genes)))
            for celltype, genes in sorted(gene_groups, key=lambda g: g[0]))
    gene_control = ('<select id="rv-gene">{}</select>'.format(gene_options) if gene_options
                    else '<select id="rv-gene" disabled></select>')
    if gene_options:
        gene_note = (
            '<p class="note"><strong>Gene expression</strong> is the <code>corrected_counts</code>'
            ' layer, normalised per cell with <code>sc.pp.normalize_total</code> and'
            ' <code>log1p</code>-transformed &mdash; the same values a standard scanpy workflow'
            ' plots. The per-cell total is taken over <em>all</em> {v} genes, not just the {n}'
            ' shown, so the colours are comparable between cells; normalising within a gene subset'
            ' would divide by the wrong denominator. Only these {n} columns are embedded in the'
            ' page: the full matrix is {c} x {v} and cannot travel with it.</p>'.format(
                n=len(payload['genes']), c=_fmt_int(payload['n_obs']),
                v=_fmt_int(payload.get('n_vars', 0))))
    else:
        gene_note = (
            '<p class="note">No genes are configured, so the expression view is empty. Point '
            '<code>markers.report_genes</code> in <code>annotation_config.yaml</code> at a '
            'one-gene-per-line file &mdash; full expression cannot be inlined ({n} cells x tens '
            'of thousands of genes), so only the listed columns travel with the page.</p>'.format(
                n=_fmt_int(payload['n_obs'])))

    phylo = payload.get('phylo')
    if phylo:
        clades = sorted(set(phylo['clade_of'].values()))
        phylo_note = (
            '<p class="note"><strong>Clade support</strong> in each candidate table shows which '
            'phylogenetic clades back a call, not just how many species &mdash; a call resting on '
            'one clade (flagged) is weaker evidence than the same species count spread across '
            'several, because same-clade agreement is not independent. The reference panel spans '
            '{nc} clades ({cl}). This already shaped the ranking: the score uses a clade-weighted '
            'species count (&delta;={d}), so four grasses count for less than four families &mdash; '
            '<code>w=</code> is that weighted value. <strong>Significance is unchanged</strong>: '
            'the q-values and which candidates appear are exactly as before, only the order '
            'moved.</p>'.format(nc=len(clades), cl=', '.join(html.escape(c) for c in clades),
                                d=phylo.get('delta')))
    else:
        phylo_note = ''

    body = """
<h1>Annotation review &mdash; {sample}</h1>
<p class="sub">{n_gs} genesets from {n_m} marker methods, grouped into cell types. Each group is
one final cell type and the gene sets that describe it; the groups you export here are what seeds
the refinement run. Drawn on <code>{umap}</code>, clustered by <code>{cluster}</code>.</p>

<div class="rv-wrap">
  <div class="rv-controls">
    <label>colour by</label>
    <select id="rv-color">
      <option value="cluster">cluster</option>
      <option value="group">selected cell type (method support)</option>
      <option value="members">selected cell type (each geneset separately)</option>
      <option value="allgroups">all cell types</option>
      <option value="gene">gene expression</option>
    </select>
    {gene_control}
    <button class="primary" id="rv-export">export groups.json</button>
    <button id="rv-reset">reset to proposal</button>
  </div>
  <div class="rv-status" id="rv-status"></div>
  <div class="rv-split">
    <div>
      <canvas class="rv-canvas" id="rv-canvas"></canvas>
      <div class="rv-legend" id="rv-legend"></div>
    </div>
    <div class="rv-panel" id="rv-panel"></div>
  </div>
</div>

<p class="note"><strong>Every number on this page was computed by stage 1.</strong> A group's cell
count is the union of its members' stage-1 seed cells &mdash; a real union of real sets, but
<em>not</em> the final seed count. Refinement assigns cells at group level, which needs the score
matrices and cannot happen in a browser. <code>python -m refine.main --cfg
configs/refine_config.yaml --dry-run</code> prints the exact figure in seconds; run it after every
editing session.</p>

<p class="note">Selecting a cell type colours each cell by <strong>how many methods independently
seeded it</strong>. That is the point of grouping: a cell six methods agree on and a cell one
method found are both seeds, and only this distinction tells them apart. Cells whose methods
disagree are left for scANVI rather than seeded on a coin-flip.</p>

<p class="note"><strong>Inspecting the genesets inside a cell type.</strong> Each member has
<em>hide</em> and <em>only this</em>; switch the colouring to <em>each geneset separately</em> to
see several at once in their own colours. Each member also reports how its seed cells are spread
across the clustering &mdash; <code>4 clusters &middot; 78% in 3</code> &mdash; so a set that is
really describing the tissue rather than a cell type shows up as a number rather than an
impression. The <em>broad</em> marker is a display hint at an arbitrary cut; judge on the figures
beside it.</p>

<p class="note"><strong>Hiding is not removing.</strong> Visibility is a view setting and never
reaches the export &mdash; looking past a geneset to see its neighbours is not a decision about
it. The three decisions are <em>remove from group</em> (back to the ungrouped tray, keeps its own
<code>unknown_</code> class), <em>move to&hellip;</em> (into a different cell type), and
<em>drop</em> (junk, seeds nothing).</p>

<p class="note"><strong>A <code>split?</code> marker means the group's genesets do not agree with
each other.</strong> Each member is summarised by the cluster it concentrates in; when those
differ, one automatic call is covering more than one population. This catches what no per-member
figure can &mdash; the members can each be tightly concentrated, and still be concentrated in
<em>different places</em>. Fix it by selecting the odd members and using <em>own group</em> or
<em>move to&hellip;</em>, then renaming.</p>
{phylo_note}
{gene_note}
{consensus}

<h2>What the three buckets mean</h2>
<p class="sub">They are three different statements and refinement treats them differently.</p>
<table>
<thead><tr><th>bucket</th><th>meaning</th><th>what refinement does</th></tr></thead>
<tbody>
<tr><td><strong>grouped</strong></td><td>part of a cell type, named or not</td>
    <td>seeds that class; a named group is its PO term, an unnamed one is <code>unknown_&lt;group&gt;</code></td></tr>
<tr><td><strong>ungrouped</strong></td><td>a real population you could not place</td>
    <td>gets its <em>own</em> <code>unknown_</code> class &mdash; not absorbed into a neighbour</td></tr>
<tr><td><strong>dropped</strong></td><td>junk</td>
    <td>contributes no seed cells at all</td></tr>
</tbody></table>
<p class="note">Ungrouped is not a milder form of dropped. scANVI is a closed-world classifier: a
cell type missing from the label set does not come back as &ldquo;uncertain&rdquo;, it comes back
as the nearest class it was shown, often confidently. Leaving a population ungrouped keeps an
honest option open for it; dropping it removes that option.</p>

<script>window.__REVIEW__ = {json};</script>
<script>{js}</script>
""".format(sample=html.escape(payload['sample']), umap=html.escape(payload['umap_key']),
           cluster=html.escape(payload['cluster_key']),
           n_gs=len(payload['genesets']), n_m=len(payload['methods']),
           gene_control=gene_control, gene_note=gene_note, phylo_note=phylo_note,
           consensus=_consensus_section(payload),
           json=json.dumps(payload, separators=(',', ':')), js=_REVIEW_JS)

    return _PAGE.format(title='Annotation review - {}'.format(payload['sample']),
                        css=_CSS + _REVIEW_CSS, body=body)


def write_review(sample, adata, annotation_dir, methods, cluster_key, out_path,
                 umap_key=None, gene_groups=None, max_points=DEFAULT_MAX_POINTS,
                 seed_quantile=0.9, seed_margin=0.25, clade_weights=None) -> Path:
    """Assemble and write one sample's review page; returns its path.

    `gene_groups` is the `[(celltype, [gene, ...]), ...]` list from
    `load_report_gene_groups`; `clade_weights` is a `phylo.CladeWeights` or None.
    """
    payload = build_payload(sample=sample, adata=adata, annotation_dir=annotation_dir,
                            methods=methods, cluster_key=cluster_key, umap_key=umap_key,
                            gene_groups=gene_groups, max_points=max_points,
                            seed_quantile=seed_quantile, seed_margin=seed_margin,
                            clade_weights=clade_weights)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(payload))
    logger.info('[REVIEW] %s (%d genesets, %d proposed groups, %.1f MB)', out_path,
                len(payload['genesets']), len(payload['proposed']['groups']),
                out_path.stat().st_size / 1e6)
    return out_path


def main(annotation_config_path, sample, species=None, adata_path=None, cluster_key=None,
         umap_key=None, methods=None, out=None) -> int:
    """CLI entry point: build the review page for one sample; returns 0.

    Arguments mirror the CLI flags; unset ones fall back to the annotation config.
    """
    import scanpy as sc

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    # One config, several target species. resolve_species folds the selected
    # profile in and strips the target from `reference_species` -- see
    # shared/species_profiles.py. A config in the old single-species shape is
    # returned unchanged, so nothing here depends on the merge having happened.
    config = resolve_species(load_config(Path(annotation_config_path)),
                             species)['AnnotateConfigs']
    marker_cfg = config.get('markers', {}) or {}
    gen_cfg = marker_cfg.get('generate', {}) or {}

    root = find_root_dir()
    # Config-driven rather than cwd-relative -- see the matching comment in
    # annotation/main.py. Must resolve the same way this stage's own
    # run did, or a standalone rebuild silently looks in the wrong place.
    stage_out = Path(config.get('output_dir') or (root / 'results/annotation'))
    annotation_dir = stage_out / 'results'
    methods = list(methods or marker_cfg.get('methods') or [])
    cluster_key = cluster_key or gen_cfg.get('cluster_key')
    adata_path = adata_path or (gen_cfg.get('samples') or {}).get(sample)
    if not adata_path:
        raise ValueError('--adata is required (no entry for {!r} under '
                         'markers.generate.samples)'.format(sample))
    if not cluster_key:
        raise ValueError('--cluster-key is required (none in markers.generate.cluster_key)')

    # Same clade weighting the annotation stage used, so a standalone rebuild
    # shows the identical clade profile. Absent config -> None -> no clade column.
    clade_weights = None
    if config.get('phylogeny'):
        from .phylo import build as phylo_build
        clade_weights = phylo_build(config['phylogeny'])

    adata = sc.read_h5ad(adata_path, backed='r')
    # Beside the dataset's decisions. Pages built before 2026-08-06 are at
    # `stage_out/review/{sample}_review.html`.
    out = Path(out) if out else (stage_out / sample / '{}_review.html'.format(sample))
    write_review(sample=sample, adata=adata, annotation_dir=annotation_dir, methods=methods,
                 cluster_key=cluster_key, out_path=out, umap_key=umap_key,
                 gene_groups=load_report_gene_groups(marker_cfg.get('report_genes'), root=root,
                                                     species=config.get('target_species')),
                 seed_quantile=marker_cfg.get('seed_quantile', 0.9),
                 seed_margin=marker_cfg.get('seed_margin', 0.25),
                 clade_weights=clade_weights)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the stage 2 annotation review page')
    parser.add_argument('--acfg', required=True)
    parser.add_argument('--species',
                        help='Which target species profile to use (see '
                             '`species_profiles:` in the annotation config). '
                             'Omit to use `default_species:`.')
    parser.add_argument('--sample', required=True)
    parser.add_argument('--adata', dest='adata_path', required=False)
    parser.add_argument('--cluster-key', required=False)
    parser.add_argument('--umap-key', required=False,
                        help='.obsm key to draw. Default: X_umap_scVI, else any X_umap*.')
    parser.add_argument('--methods', nargs='+', required=False)
    parser.add_argument('--out', required=False)
    args = parser.parse_args()
    raise SystemExit(main(annotation_config_path=args.acfg, sample=args.sample, species=args.species,
                          adata_path=args.adata_path, cluster_key=args.cluster_key,
                          umap_key=args.umap_key, methods=args.methods, out=args.out))
