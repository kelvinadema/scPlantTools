"""
Seed quantile scope for the combined refine run --
`refine.labels._group_assignment(..., quantile_scope=)` (added 2026-09-14 for
chapter 3's C2 addendum).

The case that motivated it: a review group unites two genesets (two clusters)
whose z-scores live on different scales. Under the original `group` scope the
decile is taken over the pooled assigned cells and every seed goes to the
higher-scoring cluster; under `geneset` scope each winning member geneset gets
its own decile, so both clusters seed. What has to hold:

  * `group` reproduces the old behaviour exactly (seeds only from the
    high-scale cluster);
  * `geneset` seeds both clusters, ~10 % of each, and never a cell the group
    did not win by the margin;
  * a geneset winning fewer than `min_cells_per_geneset` cells falls back to
    the group cutoff (with the guard above every winner count, `geneset` scope
    reproduces `group` scope exactly);
  * single-geneset groups are identical under both scopes;
  * an unknown scope is an error, not a silent default.

Requires `scIntegrationV1.12` (shared.geneset_scores imports scanpy).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from refine.labels import _group_assignment  # noqa: E402


def _two_cluster_scores(n_high=100, n_low=800, n_other=400, seed=0):
    """Raw scores for three genesets: A and B mark two clusters of one review
    group, C marks a third cluster in another group. A marks a small cluster
    with a large shift, B a large cluster with a modest one, so after the
    per-geneset z-scoring `_group_assignment` applies, A's cluster sits at
    z ~ 3.5 and B's at z ~ 0.8 -- the two-scale situation the option exists for
    (on `At_tip_ls_H` the pericycle group's two clusters sat at median group z
    3.3 and 3.1 against a pooled cutoff of 3.9)."""
    rng = np.random.default_rng(seed)
    n = n_high + n_low + n_other
    a = rng.normal(0, 0.2, n); b = rng.normal(0, 0.2, n); c = rng.normal(0, 0.3, n)
    hi = slice(0, n_high); lo = slice(n_high, n_high + n_low); ot = slice(n_high + n_low, n)
    a[hi] += rng.normal(8.0, 0.5, n_high)      # small, sharp cluster, geneset A
    b[lo] += rng.normal(2.0, 0.3, n_low)       # large, modest cluster, geneset B
    c[ot] += rng.normal(4.0, 0.8, n_other)     # the other group's cluster
    idx = [f'cell{i}' for i in range(n)]
    scores = pd.DataFrame({'A': a, 'B': b, 'C': c}, index=idx)
    cluster = pd.Series(['hi'] * n_high + ['lo'] * n_low + ['other'] * n_other, index=idx)
    return scores, cluster


MEMBERSHIP = {'stele': ['A', 'B'], 'cortex': ['C']}


def test_group_scope_seeds_only_the_high_scale_cluster():
    scores, cluster = _two_cluster_scores()
    masks = _group_assignment(scores, MEMBERSHIP, quantile=0.9, margin=0.25, quantile_scope='group')
    seeded = cluster[masks['stele']]
    assert (seeded == 'hi').all(), 'the pooled decile must land entirely on the high-scale cluster'
    assert (seeded == 'lo').sum() == 0


def test_geneset_scope_seeds_both_clusters_about_a_decile_each():
    scores, cluster = _two_cluster_scores()
    masks = _group_assignment(scores, MEMBERSHIP, quantile=0.9, margin=0.25, quantile_scope='geneset')
    seeded = cluster[masks['stele']].value_counts()
    # roughly a decile of each cluster's assigned cells, not of the pool
    assert 5 <= seeded.get('hi', 0) <= 20 and 50 <= seeded.get('lo', 0) <= 120, seeded.to_dict()
    assert seeded.get('other', 0) == 0


def test_geneset_scope_never_seeds_a_cell_the_group_did_not_win():
    scores, _ = _two_cluster_scores()
    grp = _group_assignment(scores, MEMBERSHIP, quantile=0.0, margin=0.25, quantile_scope='group')
    gen = _group_assignment(scores, MEMBERSHIP, quantile=0.9, margin=0.25, quantile_scope='geneset')
    # quantile=0 under group scope == every assigned cell; geneset seeds must be a subset
    assert not (gen['stele'] & ~grp['stele']).any()
    assert not (gen['cortex'] & ~grp['cortex']).any()


def test_below_min_cells_per_geneset_falls_back_to_the_group_cutoff():
    # With the guard above any geneset's winner count, every member falls back
    # to the group cutoff and `geneset` scope must reproduce `group` scope
    # exactly; with the default guard the low-scale cluster seeds.
    scores, cluster = _two_cluster_scores()
    grp = _group_assignment(scores, MEMBERSHIP, quantile=0.9, margin=0.25, quantile_scope='group')
    held = _group_assignment(scores, MEMBERSHIP, quantile=0.9, margin=0.25,
                             quantile_scope='geneset', min_cells_per_geneset=10 ** 6)
    assert np.array_equal(grp['stele'], held['stele'])
    own = _group_assignment(scores, MEMBERSHIP, quantile=0.9, margin=0.25,
                            quantile_scope='geneset', min_cells_per_geneset=10)
    assert (cluster[own['stele']] == 'lo').sum() >= 50


def test_single_geneset_group_is_identical_under_both_scopes():
    scores, _ = _two_cluster_scores()
    a = _group_assignment(scores, MEMBERSHIP, quantile=0.9, margin=0.25, quantile_scope='group')
    b = _group_assignment(scores, MEMBERSHIP, quantile=0.9, margin=0.25, quantile_scope='geneset')
    assert np.array_equal(a['cortex'], b['cortex'])


def test_unknown_scope_is_an_error():
    scores, _ = _two_cluster_scores()
    with pytest.raises(ValueError):
        _group_assignment(scores, MEMBERSHIP, quantile=0.9, margin=0.25, quantile_scope='modality')


def test_seeds_for_lays_the_target_block_over_the_global_one():
    from refine.main import seeds_for
    cfg = {'seeds': {'quantile': 0.9, 'margin': 0.25, 'quantile_scope': 'geneset'}}
    assert seeds_for(cfg, {'adata': 'x'}) == cfg['seeds']
    merged = seeds_for(cfg, {'adata': 'x', 'seeds': {'quantile_scope': 'group'}})
    assert merged == {'quantile': 0.9, 'margin': 0.25, 'quantile_scope': 'group'}
    assert cfg['seeds']['quantile_scope'] == 'geneset', 'the global block is not modified'
    assert seeds_for({}, {}) == {}
