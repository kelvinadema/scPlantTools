#!/usr/bin/env python3
"""
The hand-off between the annotation stage and refinement: one chosen cell type
per cluster or gene module.

Stage 1 writes a **template** per (sample, method) pre-filled with the best
automatic call; a human edits it; stage 3 reads it back and turns it into seed
labels. Both a TSV (hand-editable, diffable, works headless) and a JSON (what
the review report exports) are accepted, with JSON winning when both exist.

This module lives in `shared/` because the two ends run under different
interpreters -- annotation under `scAnnotationV3.12`, refinement under
`scIntegrationV1.12` -- so it must import nothing either side lacks. In
particular **it does not import `pronto`**, which is absent from the
integration env: ontology validation is injected by the caller that has it
(see `valid_po_ids` on `read_decisions`).
"""

import hashlib
import json

from pathlib import Path

import pandas as pd

# Written by build_template; `chosen_po_id` and `note` are the human's columns
# and everything else is evidence for the choice.
TEMPLATE_COLUMNS = (
    'cluster', 'n_candidates', 'n_usable', 'decision',
    'top_po_id', 'top_po_name', 'score', 'support_species',
    'alternatives', 'rejected', 'chosen_po_id', 'note',
)

# `chosen_po_id` sentinel: this cluster has no usable annotation and its cells
# should stay Unknown. Distinct from an empty cell, which means "accept the
# automatic top call" -- the difference between a considered decision and one
# not yet made.
DROP = '__DROP__'

# Excluded from the decision hash. Editing a comment must not invalidate a
# refinement run; changing a call must.
_UNHASHED = ('note',)


class DecisionError(ValueError):
    """Raised for a malformed, inconsistent or stale decision file."""


# --------------------------------------------------------------------- build

def _fmt_alternatives(rows, limit=3):
    """Format candidate rows for a "did you mean" message, cut at 600 characters.

    `limit` is accepted but not applied.
    """
    return ' | '.join(
        f"{r.po_name} ({r.po_id}; {r.score:.1f})" for r in rows.itertuples())[:600] \
        if len(rows) else ''


def _fmt_rejected(rows):
    """Why candidates were set aside, so the human can second-guess it."""
    parts = []
    n_general = int(rows['too_general'].sum()) if 'too_general' in rows else 0
    n_tissue = int((~rows['tissue_match']).sum()) if 'tissue_match' in rows else 0
    if n_general:
        parts.append(f'{n_general} too_general')
    if n_tissue:
        parts.append(f'{n_tissue} wrong_tissue')
    return '; '.join(parts)


def build_template(annotations: pd.DataFrame, genesets, cluster_col='cluster',
                   association: pd.DataFrame = None) -> pd.DataFrame:
    """
    One row per geneset, with the best automatic call pre-filled.

    `genesets` is the **full** list of clusters/modules the marker method
    produced, not just those with a significant hit. That distinction is
    load-bearing: on `MtTip_noHVG`, `cfmf` finds hits for 6 of 16 modules and
    `hotspot` for 16 of 30. Building the template from the annotation table
    alone would silently drop the rest, and their cells exist whether or not a
    cell type was found for them -- they must appear, marked `none`, so the
    gap is visible and can be acted on.

    The top call is the highest-scoring candidate that is neither `too_general`
    nor tissue-mismatched (see annotate.py). Rejected candidates are not
    discarded -- they are summarised in `rejected` and remain in the
    annotations TSV.
    """
    annotations = annotations if annotations is not None else pd.DataFrame()
    rows = []
    for cluster in genesets:
        sub = annotations[annotations[cluster_col].astype(str) == str(cluster)] \
            if len(annotations) else annotations
        usable = sub
        if len(sub):
            mask = pd.Series(True, index=sub.index)
            if 'too_general' in sub:
                mask &= ~sub['too_general'].astype(bool)
            if 'tissue_match' in sub:
                mask &= sub['tissue_match'].astype(bool)
            usable = sub[mask]
        usable = usable.sort_values('score', ascending=False) if len(usable) else usable

        top = usable.iloc[0] if len(usable) else None
        rows.append({
            'cluster': cluster,
            'n_candidates': len(sub),
            'n_usable': len(usable),
            'decision': top['decision'] if top is not None else 'none',
            'top_po_id': top['po_id'] if top is not None else '',
            'top_po_name': top['po_name'] if top is not None else '',
            'score': round(float(top['score']), 3) if top is not None else '',
            'support_species': top['support_species'] if top is not None else '',
            'alternatives': _fmt_alternatives(usable.iloc[1:]) if len(usable) > 1 else '',
            'rejected': _fmt_rejected(sub) if len(sub) else '',
            'chosen_po_id': '',
            'note': '',
        })
        # Which cluster this geneset's confident cells actually sit in. For a
        # cluster-free module this is the only thing that makes it comparable
        # with a leiden cluster; for a cluster-dependent one it should be the
        # cluster the geneset came from, which is the built-in sanity check.
        if association is not None and cluster in association.index:
            a = association.loc[cluster]
            rows[-1].update({'assoc_cluster': a['assoc_cluster'],
                             'assoc_frac': a['assoc_frac'],
                             'n_seed_cells': int(a['n_seed_cells'])})
    columns = list(TEMPLATE_COLUMNS)
    if association is not None:
        # Inserted before the human's columns so they stay at the right edge.
        for extra in ('assoc_cluster', 'assoc_frac', 'n_seed_cells'):
            columns.insert(columns.index('chosen_po_id'), extra)
    return pd.DataFrame(rows).reindex(columns=columns)


def is_edited(path) -> bool:
    """Has a human touched this template? Any chosen_po_id or note counts."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, sep='\t', dtype=str).fillna('')
    except Exception:                                   # noqa: BLE001 - unreadable == precious
        return True
    return any(df.get(c, pd.Series(dtype=str)).str.strip().any()
               for c in ('chosen_po_id', 'note'))


def write_template(path, template: pd.DataFrame, force=False) -> Path:
    """
    Write a template, refusing to overwrite one that has been edited.

    Losing a session of curation to an incidental rerun is the only
    unrecoverable failure in this stage, so the default is to refuse. `force`
    overwrites deliberately.
    """
    path = Path(path)
    if is_edited(path) and not force:
        raise DecisionError(
            f'{path} has edits (chosen_po_id/note) and would be overwritten. '
            f'Move it aside, or pass force=True to discard those edits.')
    path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(path, sep='\t', index=False)
    return path


# ---------------------------------------------------------------------- read

def decision_hash(mapping: dict) -> str:
    """
    Identity of a set of decisions: the resolved {cluster -> po_id} map only.

    Notes are excluded (see `_UNHASHED`), so editing a comment is a no-op while
    changing a call re-runs. `None` is encoded explicitly rather than dropped,
    since "this cluster is deliberately Unknown" is a decision that must
    invalidate a cached run just as any other does.
    """
    payload = json.dumps({str(k): (v if v is not None else '__NONE__')
                          for k, v in sorted(mapping.items())}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_hash(path) -> str:
    """sha256 of a file, used as the parent hash of the annotations table."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


class Decisions:
    """Resolved decisions plus the provenance needed to detect staleness."""

    def __init__(self, mapping, source, parent_hash=None, notes=None, dropped=None):
        self.mapping = mapping          # {cluster (str) -> po_id or None}
        self.source = str(source)
        self.parent_hash = parent_hash
        self.notes = notes or {}
        # Genesets marked __DROP__ explicitly, as distinct from those that
        # simply have no call. Both resolve to None, but they mean opposite
        # things downstream: "this geneset is junk, do not seed from it" versus
        # "this geneset is real and we could not name it" -- and the second is
        # exactly what gets an `unknown_<geneset>` class of its own rather than
        # being absorbed into a neighbouring cell type.
        self.dropped = set(dropped or ())

    def unnamed(self):
        """Genesets that are real but have no cell-type call."""
        return {k for k, v in self.mapping.items() if not v and k not in self.dropped}

    @property
    def hash(self):
        """Content hash of the decision mapping, for staleness checks."""
        return decision_hash(self.mapping)

    @property
    def labelled(self):
        """The entries that carry a label, i.e. the ones that seed anything."""
        return {k: v for k, v in self.mapping.items() if v}

    def __len__(self):
        """Number of decision entries."""
        return len(self.mapping)

    def __repr__(self):
        """Short representation: labelled and total entry counts and the source file name."""
        return (f'<Decisions {len(self.labelled)}/{len(self.mapping)} clusters labelled '
                f'from {Path(self.source).name}>')


def _resolve_row(cluster, chosen, top):
    """Blank chosen == accept the automatic top call; DROP == deliberate Unknown."""
    chosen = (chosen or '').strip()
    if chosen == DROP:
        return None
    if chosen:
        return chosen
    top = (top or '').strip()
    return top or None


def read_decisions(path, genesets=None, valid_po_ids=None,
                   candidates=None, logger=None, method=None) -> Decisions:
    """
    Read a decision file (TSV or JSON) and resolve it to {cluster -> po_id}.

    `path` may be a directory, in which case `*_decisions.json` is preferred
    over `*_decisions.tsv` -- the JSON is what the review report exports, so if
    a human produced one it supersedes the template they started from.

    Validation, and why each severity was chosen:

    - a `po_id` absent from `valid_po_ids` **raises**. A typo must not silently
      become an unlabelled cluster; that is the same rule `refilter` applies to
      an unrecognised decision `kind`. Skipped when the caller cannot supply
      the ontology (the integration env has no `pronto`).
    - a cluster not in `genesets` **raises** -- it is a typo or a stale file.
    - a geneset missing from the file **raises**, naming every one at once
      rather than one per run.
    - a `method` disagreeing with the caller's **raises**. This one is not
      redundant with the geneset check, which is what makes it worth its own
      rule: `wilcoxon`, `t-test` and `cosg` all read the same leiden partition,
      so all three name their genesets `0`..`14` and a file exported for one of
      them validates perfectly against either of the others. The review report
      exports `{sample}_decisions.json` -- a name carrying no method -- so
      dropping it in the wrong sibling directory is one mis-click, and the
      result would be a plausible, wrong, silent annotation. Only checked when
      the file records a method; a hand-written TSV does not.
    - a chosen term that was not among that cluster's candidates only
      **warns**: the human may be applying outside knowledge the enrichment
      never had, which is a legitimate use of this file.
    """
    path = Path(path)
    if path.is_dir():
        js = sorted(path.glob('*_decisions.json'))
        tsv = sorted(path.glob('*_decisions.tsv'))
        if not js and not tsv:
            raise DecisionError(f'No *_decisions.json or *_decisions.tsv in {path}')
        path = js[0] if js else tsv[0]

    if path.suffix == '.json':
        payload = json.loads(path.read_text())
        raw = payload.get('decisions', {})
        mapping = {str(k): _resolve_row(k, v.get('po_id'), v.get('po_id'))
                   for k, v in raw.items()}
        notes = {str(k): v.get('note', '') for k, v in raw.items()}
        dropped = {str(k) for k, v in raw.items()
                   if str(v.get('po_id', '')).strip() == DROP}
        parent_hash = payload.get('parent_hash')
        file_method = payload.get('method')
        if method is not None and file_method and str(file_method) != str(method):
            raise DecisionError(
                f'{path} was exported for method {file_method!r} but is being read as '
                f'{method!r}. The cluster-dependent methods share cluster names, so this '
                f'would otherwise validate and produce a wrong annotation silently. Move '
                f'the file to the {file_method!r} directory.')
    else:
        df = pd.read_csv(path, sep='\t', dtype=str).fillna('')
        missing = [c for c in ('cluster', 'top_po_id', 'chosen_po_id') if c not in df]
        if missing:
            raise DecisionError(f'{path}: missing column(s) {missing}')
        mapping = {str(r['cluster']): _resolve_row(r['cluster'], r['chosen_po_id'],
                                                   r['top_po_id'])
                   for _, r in df.iterrows()}
        notes = {str(r['cluster']): r.get('note', '') for _, r in df.iterrows()}
        dropped = {str(r['cluster']) for _, r in df.iterrows()
                   if str(r['chosen_po_id']).strip() == DROP}
        parent_hash = None

    if genesets is not None:
        known = {str(g) for g in genesets}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise DecisionError(
                f'{path}: cluster(s) {unknown} are not in this method\'s marker table. '
                f'A stale decision file, or a typo.')
        absent = sorted(known - set(mapping))
        if absent:
            raise DecisionError(
                f'{path}: no decision for cluster(s) {absent}. Every cluster needs a '
                f'row -- use {DROP} in chosen_po_id to leave one deliberately Unknown.')

    if valid_po_ids is not None:
        bad = sorted({v for v in mapping.values() if v and v not in valid_po_ids})
        if bad:
            raise DecisionError(
                f'{path}: {bad} are not Plant Ontology terms in the configured '
                f'ontology. Check for a typo.')

    if candidates is not None and logger is not None:
        for cluster, po_id in mapping.items():
            if po_id and po_id not in candidates.get(str(cluster), set()):
                logger.warning(
                    '[DECISIONS] %s: chosen term %s was not among the automatic '
                    'candidates for this cluster -- accepted, but not supported by '
                    'the enrichment', cluster, po_id)

    return Decisions(mapping, source=path, parent_hash=parent_hash, notes=notes,
                     dropped=dropped)


def build_consensus(per_method: dict, clusters=None) -> pd.DataFrame:
    """
    One row per cluster, one column per method: what each method called it.

    `per_method` is `{method -> decision template}`, each carrying the
    `assoc_cluster` column. This is the artifact that answers "where do the
    methods disagree", which nothing else can: a `hotspot` module and a leiden
    cluster are different objects, so the six annotations are otherwise six
    parallel universes with no common index. Association gives them one.

    A method may associate several genesets with the same cluster -- three
    hotspot modules can all describe cluster 2. They are joined with '|' rather
    than reduced, because which of them is right is exactly the judgement the
    reviewer is here to make.
    """
    frames = {}
    for method, template in per_method.items():
        if 'assoc_cluster' not in template:
            continue
        rows = template[(template['assoc_cluster'].astype(str) != '')
                        & (template['top_po_name'].astype(str) != '')]
        calls = {}
        for _, r in rows.iterrows():
            calls.setdefault(str(r['assoc_cluster']), []).append(str(r['top_po_name']))
        frames[method] = {k: ' | '.join(dict.fromkeys(v)) for k, v in calls.items()}

    index = sorted({c for m in frames.values() for c in m},
                   key=lambda x: (len(x), x)) if clusters is None else [str(c) for c in clusters]
    out = pd.DataFrame(index=index)
    out.index.name = 'cluster'
    for method in sorted(frames):
        out[method] = [frames[method].get(c, '') for c in out.index]

    # How many methods produced a call at all, and how many distinct calls --
    # the second is the disagreement measure worth sorting on.
    filled = out.apply(lambda r: [v for v in r if v], axis=1)
    out.insert(0, 'n_methods', filled.apply(len))
    out.insert(1, 'n_distinct', filled.apply(lambda vs: len({x for v in vs for x in v.split(' | ')})))
    return out


def candidates_from_annotations(annotations: pd.DataFrame, cluster_col='cluster',
                                po_col='po_id') -> dict:
    """
    {cluster -> {po_id}} for `read_decisions(candidates=...)`.

    Every candidate the enrichment produced, including ones flagged
    `too_general` or tissue-mismatched: choosing one of those is a judgement
    the human is entitled to make, and warning about it would be noise. The
    warning is for a term the enrichment never proposed at all.
    """
    if annotations is None or not len(annotations):
        return {}
    return {str(c): set(g[po_col].dropna())
            for c, g in annotations.groupby(annotations[cluster_col].astype(str))}


def check_parent(decisions: Decisions, annotations_path, allow_stale=False):
    """
    Refuse decisions derived from an annotation table that has since changed.

    Same guard as `refilter`'s `parent_config_hash`: re-running stage 1 can
    change which candidates exist, and a decision naming a term that is no
    longer a candidate -- or a cluster that no longer exists -- is not
    something to silently carry forward.
    """
    if decisions.parent_hash is None:
        return
    current = file_hash(annotations_path)
    if current != decisions.parent_hash and not allow_stale:
        raise DecisionError(
            f'{decisions.source} was written against a different version of '
            f'{Path(annotations_path).name} (parent hash {decisions.parent_hash[:12]}, '
            f'current {current[:12]}). Re-generate the template and redo the review, '
            f'or pass allow_stale=True if the change is known to be irrelevant.')
