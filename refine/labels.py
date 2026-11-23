#!/usr/bin/env python3
"""
Turn decisions + geneset scores into the seed label vector scANVI trains on.

The interesting part is what happens to a geneset nobody could name.

scANVI is a **closed-world classifier**: its softmax sums to 1 over the classes
it was shown, so a cell type absent from the seed set does not come back as
"uncertain" -- it comes back as whichever known class is nearest, often with
high confidence. On `MtTip_noHVG` that exposure is real: clusters 8 and 11
(5,647 cells, 9.0%) got no call from any of the six methods, and would be
silently absorbed into their neighbours.

So an unnamed geneset gets **its own named class**, `unknown_<geneset>`, rather
than being left unlabelled. Each unnamed geneset is a separate class, not one
pooled "unassigned" bucket -- pooling them would build a class out of several
unrelated cell types, and a heterogeneous class becomes an attractor that
swallows cells the model could otherwise have placed. Kept separate, each stays
coherent, the model has a real "none of the named types, but like this module"
option, and the label says which geneset to go and annotate next.

`__DROP__` is the one case that is *not* given a class: it is the reviewer
saying the geneset is junk, which is different from saying it is unnamed.
"""

import logging

import numpy as np
import pandas as pd

from shared.geneset_scores import seed_cells, zscore

logger = logging.getLogger('AnnotationLogging')

# scANVI's `unlabeled_category`. Deliberately not the string 'Unknown', which
# would read as a cell-type call and collide with the `unknown_<geneset>`
# classes, which are the opposite thing -- a real class the model must learn.
UNASSIGNED = 'Unassigned'

UNKNOWN_PREFIX = 'unknown_'


def build_seed_labels(scores: pd.DataFrame, decisions, quantile=0.9, margin=0.25,
                      min_seed_cells=10, name_map=None) -> pd.Series:
    """
    Per-cell seed labels: a PO term, an `unknown_<geneset>` class, or UNASSIGNED.

    `decisions` is a `shared.decisions.Decisions`. `name_map` optionally
    renames PO ids to readable names for the final `.obs` column; the labels
    themselves stay ids so two genesets resolving to the same term merge, which
    is intended -- two modules describing root hair cells are one class.

    A class with fewer than `min_seed_cells` seeds is dropped back to
    UNASSIGNED: scANVI cannot learn a class from a handful of examples, and a
    label resting on three cells reads exactly like one resting on four hundred
    once it is in `.obs`.
    """
    seeds = seed_cells(scores, quantile=quantile, margin=margin)
    labels = pd.Series(UNASSIGNED, index=scores.index, dtype=object)

    for geneset in scores.columns:
        key = str(geneset)
        mask = seeds == geneset
        if not mask.any():
            continue
        if key in decisions.dropped:
            continue                      # reviewer marked it junk; no class
        po_id = decisions.mapping.get(key)
        label = po_id if po_id else f'{UNKNOWN_PREFIX}{key}'
        labels[mask] = label

    counts = labels[labels != UNASSIGNED].value_counts()
    too_small = counts[counts < min_seed_cells]
    if len(too_small):
        labels[labels.isin(too_small.index)] = UNASSIGNED
        logger.warning(
            '[REFINE] %d class(es) had fewer than %d seed cells and were returned to '
            '%s: %s', len(too_small), min_seed_cells, UNASSIGNED,
            ', '.join(f'{k} (n={v})' for k, v in too_small.items()))

    if name_map:
        labels = labels.map(lambda v: name_map.get(v, v))

    final = labels[labels != UNASSIGNED].value_counts()
    logger.info('[REFINE] %d seed class(es), %d/%d cells seeded (%.1f%%); '
                '%d named, %d unknown_*',
                len(final), int((labels != UNASSIGNED).sum()), len(labels),
                100 * (labels != UNASSIGNED).mean(),
                sum(not str(k).startswith(UNKNOWN_PREFIX) for k in final.index),
                sum(str(k).startswith(UNKNOWN_PREFIX) for k in final.index))
    return labels


def summarise(labels: pd.Series) -> pd.DataFrame:
    """Seed count per class, for the run record and the report."""
    counts = labels.value_counts()
    return pd.DataFrame({
        'label': counts.index,
        'n_seed_cells': counts.to_numpy(),
        'kind': [('unassigned' if k == UNASSIGNED
                  else 'unknown' if str(k).startswith(UNKNOWN_PREFIX) else 'named')
                 for k in counts.index],
    }).reset_index(drop=True)


# --------------------------------------------------------------------- groups

def _group_assignment(scores: pd.DataFrame, membership: dict, quantile, margin,
                      min_z=0.0, quantile_scope='group', min_cells_per_geneset=10):
    """
    One method's verdict: which group each cell is a confident seed for.

    `membership` is `{group_id: [geneset, ...]}` restricted to this method's
    genesets. The rule is the per-geneset one from `shared.geneset_scores`
    lifted one level up:

        group score  = max z-score over that group's member genesets
        assignment   = argmax over GROUPS, and only if it beats the runner-up
                       group by `margin`
        seeds        = top `quantile` of the assigned cells, per group
                       (`quantile_scope='group'`), or per WINNING MEMBER
                       GENESET within the group (`quantile_scope='geneset'`)

    **Why the scope option exists.** A
    group that unites two clusters -- the reviewer saying "these are one cell
    type" -- unites two genesets whose z-scores live on different scales, and
    a single decile over the pooled assigned cells hands every seed to the
    higher-scoring cluster. On `At_tip_ls_H` the pericycle group held a nuclei
    cluster (median group z 3.3) and a protoplast cluster (3.1); the pooled
    cutoff was 3.9 and 1 of 185 seeds was a protoplast, so the class stayed
    nuclei-only by construction and the novelty safeguard then demoted the
    protoplasts scANVI predicted into it. `geneset` scope takes the decile
    among the cells each member geneset wins, which is what `seed_cells` does
    per geneset before grouping -- exclusivity is unchanged (a cell must still
    win its group by `margin`), only the cutoff is local. A geneset that wins
    fewer than `min_cells_per_geneset` assigned cells keeps the group cutoff:
    a decile of five cells is not a core.

    Caveat of the local cutoff: a member geneset that describes no real
    population still wins the scattered cells on which its noise happens to be
    highest, and under `geneset` scope those cells get a decile of their own.
    `min_cells_per_geneset` is the only guard, so under this scope a reviewer
    should DROP a junk geneset rather than park it in a group.

    **Exclusivity has to stay, and it has to be at the group level.** Dropping
    it entirely was measured and is unusable: scoring each geneset's own top
    decile independently makes 88% of the dataset a seed for something, with a
    typical cell "supporting" four different hotspot modules -- correlated
    modules overlapping, not evidence. Keeping it at the *geneset* level is
    what the old per-method path did, and that is what threw away 3,316
    wilcoxon cells for being torn between clusters 3 and 4 after the reviewer
    had already declared both to be root hair cells. At the group level the
    specificity survives and that particular ambiguity cannot arise, because
    the two are no longer competing.

    Returns `{group_id -> boolean mask}`.
    """
    groups = [g for g, sets in membership.items()
              if any(s in scores.columns for s in sets)]
    if not groups:
        return {}

    z = zscore(scores)
    # Column order is fixed by `groups`, not by set iteration, so the argmax
    # below is reproducible between processes. Python randomises string
    # hashing, and a per-process column order would silently move the ties.
    matrix = np.column_stack([
        z[[s for s in membership[g] if s in scores.columns]].to_numpy().max(axis=1)
        for g in groups])

    if matrix.shape[1] == 1:
        clear = matrix[:, 0] >= min_z
        top_idx = np.zeros(len(matrix), dtype=int)
    else:
        order = np.argsort(-matrix, axis=1)
        rows = np.arange(len(matrix))
        top_idx = order[:, 0]
        top = matrix[rows, top_idx]
        second = matrix[rows, order[:, 1]]
        clear = (top - second >= margin) & (top >= min_z)

    if quantile_scope not in ('group', 'geneset'):
        raise ValueError("quantile_scope must be 'group' or 'geneset', got {!r}".format(quantile_scope))
    out = {}
    for i, group in enumerate(groups):
        assigned = clear & (top_idx == i)
        if not assigned.any():
            out[group] = assigned
            continue
        # Quantile on the group's own score among its assigned cells, exactly
        # as `seed_cells` does per geneset: an absolute cut would hand every
        # seed to whichever group happens to score highest overall.
        cutoff = np.quantile(matrix[assigned, i], quantile)
        seeds = assigned & (matrix[:, i] >= cutoff)
        if quantile_scope == 'geneset':
            members = [s for s in membership[group] if s in scores.columns]
            if len(members) > 1:
                sub = z[members].to_numpy()
                winner = sub.argmax(axis=1)
                seeds = np.zeros(len(matrix), dtype=bool)
                for j in range(len(members)):
                    won = assigned & (winner == j)
                    if not won.any():
                        continue
                    local = (np.quantile(matrix[won, i], quantile)
                             if won.sum() >= min_cells_per_geneset else cutoff)
                    seeds |= won & (matrix[:, i] >= local)
        out[group] = seeds
    return out


def build_group_seed_labels(scores_by_method: dict, document, quantile=0.9, margin=0.25,
                            min_seed_cells=10, name_map=None, index=None,
                            quantile_scope='group'):
    """
    Seed labels for the combined run, from a cross-method grouping.

    Every method votes independently: each is scored, assigned and seeded on
    its own, then the votes are combined. A cell's label is the group the most
    methods seeded it for.

    **Ties go to UNASSIGNED, deliberately.** Two methods saying `cortex` and
    two saying `endodermis` is not a cell type, it is a disagreement, and
    handing scANVI a coin-flip as a confident seed is worse than handing it the
    cell as a question -- the model propagates from seeds, so a wrong seed
    spreads.

    `quantile_scope` ('group', the original rule, or 'geneset') is passed to
    `_group_assignment`; see there for why it exists.

    Returns `(labels, support)` where `support` is how many methods backed each
    cell's final label. That number is the whole point of combining: a cell
    with 6/6 backing and one with 1/6 both become seeds, and only this column
    can tell them apart afterwards.
    """
    index = index if index is not None else next(iter(scores_by_method.values())).index

    # {group -> {method -> [geneset]}}, plus the ungrouped genesets, which each
    # become a group of one so they are seeded on exactly the same footing.
    per_group = {}
    for gid, group in document.groups.items():
        for method, geneset in group['members']:
            per_group.setdefault(gid, {}).setdefault(method, []).append(geneset)
    for method, geneset in document.ungrouped:
        per_group.setdefault('{}{}_{}'.format(UNKNOWN_PREFIX, method, geneset),
                             {}).setdefault(method, []).append(geneset)
    # Dropped genesets are simply absent: they compete for nothing and seed
    # nothing, which is what the reviewer asked for.

    votes = pd.DataFrame(0, index=index, columns=sorted(per_group), dtype=np.int16)
    for method, scores in sorted(scores_by_method.items()):
        membership = {gid: sets[method] for gid, sets in per_group.items()
                      if method in sets}
        if not membership:
            continue
        scores = scores.reindex(index)
        masks = _group_assignment(scores, membership, quantile=quantile, margin=margin,
                                  quantile_scope=quantile_scope)
        for gid, mask in masks.items():
            votes.loc[mask, gid] += 1
        logger.info('[REFINE] %s seeded %d cell(s) across %d group(s)',
                    method, int(sum(m.sum() for m in masks.values())), len(masks))

    counts = votes.to_numpy()
    best = counts.argmax(axis=1)
    best_n = counts[np.arange(len(counts)), best]
    # A tie for the top is a disagreement, not a call.
    tied = (counts == best_n[:, None]).sum(axis=1) > 1
    names = np.asarray(votes.columns)

    labels = pd.Series(UNASSIGNED, index=index, dtype=object)
    decided = (best_n > 0) & ~tied
    labels[decided] = names[best[decided]]
    support = pd.Series(0, index=index, dtype=int)
    support[decided] = best_n[decided]

    # A named group carries its PO id as the label so it is the same class
    # wherever it came from; an unnamed one keeps its own name.
    rename = {gid: g['po_id'] for gid, g in document.groups.items() if g.get('po_id')}
    if rename:
        labels = labels.map(lambda v: rename.get(v, v))

    class_counts = labels[labels != UNASSIGNED].value_counts()
    too_small = class_counts[class_counts < min_seed_cells]
    if len(too_small):
        drop = labels.isin(too_small.index)
        labels[drop] = UNASSIGNED
        support[drop] = 0
        logger.warning(
            '[REFINE] %d class(es) had fewer than %d seed cells and were returned to %s: %s',
            len(too_small), min_seed_cells, UNASSIGNED,
            ', '.join('{} (n={})'.format(k, v) for k, v in too_small.items()))

    if name_map:
        labels = labels.map(lambda v: name_map.get(v, v))

    final = labels[labels != UNASSIGNED].value_counts()
    seeded = int((labels != UNASSIGNED).sum())
    logger.info('[REFINE] %d seed class(es), %d/%d cells seeded (%.1f%%); %d named, '
                '%d unknown_*', len(final), seeded, len(labels),
                100 * seeded / max(len(labels), 1),
                sum(not str(k).startswith(UNKNOWN_PREFIX) for k in final.index),
                sum(str(k).startswith(UNKNOWN_PREFIX) for k in final.index))
    if seeded:
        hist = support[labels != UNASSIGNED].value_counts().sort_index()
        logger.info('[REFINE] method support of seeded cells: %s',
                    ', '.join('{}x: {}'.format(k, v) for k, v in hist.items()))
    return labels, support
