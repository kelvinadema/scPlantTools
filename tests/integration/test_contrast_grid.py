"""Contrast grid ranking -- `integration/contrastive.py`.

contrastiveVI cannot be autotuned (see `docs/decisions/contrastive-sweep.md`),
so these three functions ARE the model-selection criterion. Each encodes a rule
that was learned from a sweep that went wrong, and each rule is the kind that a
plausible-looking refactor would quietly undo:

  * a collapsed fit maximises `separation` by construction, so it must sort last
    rather than winning;
  * replicates of one point disagree enough to matter, so the median decides,
    not the best row;
  * a grid where everything collapsed must return nothing, not its least-bad row.

The numbers behind those rules are in the decisions doc; the rules themselves
are here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from integration.contrastive import (best_grid_point, rank_contrast_grid,
                                     summarise_contrast_grid)


def _row(penalty, salient, separation, leakage, collapsed=False, **extra):
    return {'wasserstein_penalty': penalty, 'n_salient_latent': salient,
            'separation': separation, 'leakage_ratio': leakage,
            'collapsed': collapsed, **extra}


# ------------------------------------------------------------------ ranking

def test_quality_is_separation_minus_leakage():
    df = rank_contrast_grid(pd.DataFrame([_row(0.0, 10, 0.5, 0.2)]))
    assert df['quality'].iloc[0] == pytest.approx(0.3)


def test_a_collapsed_point_sorts_last_even_with_the_best_separation():
    # The whole point: a collapsed fit has salient iLISI 0, which MAXIMISES
    # separation. Ranking on separation alone would crown it.
    df = rank_contrast_grid(pd.DataFrame([
        _row(10.0, 10, 0.99, 0.0, collapsed=True),
        _row(0.1, 10, 0.30, 0.1, collapsed=False)]))
    assert bool(df['collapsed'].iloc[0]) is False
    assert df['wasserstein_penalty'].iloc[0] == 0.1


def test_collapsed_points_are_kept_not_dropped():
    # They are the evidence that the penalty range was too high.
    df = rank_contrast_grid(pd.DataFrame([
        _row(10.0, 10, 0.99, 0.0, collapsed=True),
        _row(0.1, 10, 0.30, 0.1)]))
    assert len(df) == 2


def test_a_missing_collapsed_column_is_treated_as_not_collapsed():
    df = rank_contrast_grid(pd.DataFrame([
        {'wasserstein_penalty': 0.0, 'n_salient_latent': 10,
         'separation': 0.4, 'leakage_ratio': 0.1}]))
    assert bool(df['collapsed'].iloc[0]) is False


def test_a_failed_point_keeps_its_row_and_sorts_to_the_bottom():
    df = rank_contrast_grid(pd.DataFrame([
        {'wasserstein_penalty': 1.0, 'n_salient_latent': 10, 'error': 'boom'},
        _row(0.1, 10, 0.30, 0.1)]))
    assert len(df) == 2
    assert df['error'].iloc[-1] == 'boom'


def test_ranking_an_empty_grid_returns_it_unchanged():
    assert rank_contrast_grid(pd.DataFrame()).empty


def test_ranking_does_not_mutate_the_caller_s_frame():
    original = pd.DataFrame([_row(0.0, 10, 0.5, 0.2)])
    before = original.copy()
    rank_contrast_grid(original)
    pd.testing.assert_frame_equal(original, before)


# ------------------------------------------------------------------ choosing

def test_the_best_point_is_returned_as_typed_model_params():
    chosen = best_grid_point(pd.DataFrame([
        _row(0.1, 20, 0.60, 0.10),
        _row(1.0, 10, 0.20, 0.15)]))
    assert chosen == {'wasserstein_penalty': 0.1, 'n_salient_latent': 20}
    assert isinstance(chosen['n_salient_latent'], int)
    assert isinstance(chosen['wasserstein_penalty'], float)


def test_replicates_are_aggregated_by_median_not_by_best_row():
    # Point A has one lucky replicate (quality 0.9) but a poor median (0.3).
    # Point B is consistently 0.5. The median must pick B.
    df = pd.DataFrame([
        _row(0.1, 10, 0.95, 0.05, seed=0),   # quality 0.90
        _row(0.1, 10, 0.35, 0.05, seed=1),   # quality 0.30
        _row(0.1, 10, 0.25, 0.05, seed=2),   # quality 0.20
        _row(0.5, 20, 0.55, 0.05, seed=0),   # quality 0.50
        _row(0.5, 20, 0.55, 0.05, seed=1),   # quality 0.50
        _row(0.5, 20, 0.55, 0.05, seed=2)])  # quality 0.50
    assert best_grid_point(df) == {'wasserstein_penalty': 0.5,
                                   'n_salient_latent': 20}


def test_a_point_collapsed_in_the_majority_of_replicates_is_rejected():
    df = pd.DataFrame([
        _row(10.0, 10, 0.99, 0.0, collapsed=True, seed=0),
        _row(10.0, 10, 0.99, 0.0, collapsed=True, seed=1),
        _row(10.0, 10, 0.99, 0.0, collapsed=False, seed=2),
        _row(0.1, 20, 0.30, 0.10, seed=0)])
    assert best_grid_point(df) == {'wasserstein_penalty': 0.1,
                                   'n_salient_latent': 20}


def test_a_point_collapsed_in_a_minority_of_replicates_survives():
    df = pd.DataFrame([
        _row(0.1, 10, 0.80, 0.10, collapsed=True, seed=0),
        _row(0.1, 10, 0.80, 0.10, collapsed=False, seed=1),
        _row(0.1, 10, 0.80, 0.10, collapsed=False, seed=2)])
    assert best_grid_point(df) == {'wasserstein_penalty': 0.1,
                                   'n_salient_latent': 10}


def test_an_all_collapsed_grid_returns_none_rather_than_the_least_bad_point():
    df = pd.DataFrame([
        _row(10.0, 10, 0.99, 0.0, collapsed=True),
        _row(20.0, 10, 0.98, 0.0, collapsed=True)])
    assert best_grid_point(df) is None


def test_an_empty_or_missing_grid_returns_none():
    assert best_grid_point(None) is None
    assert best_grid_point(pd.DataFrame()) is None


def test_failed_rows_are_excluded_from_the_choice():
    df = pd.DataFrame([
        {'wasserstein_penalty': 0.1, 'n_salient_latent': 10, 'error': 'boom'},
        _row(0.5, 20, 0.30, 0.10, error=None)])
    assert best_grid_point(df) == {'wasserstein_penalty': 0.5,
                                   'n_salient_latent': 20}


# ---------------------------------------------------------------- summarising

def test_the_summary_is_one_row_per_parameter_combination():
    df = pd.DataFrame([
        _row(0.1, 10, 0.5, 0.1, salient_eff_dim=4.0, seed=0),
        _row(0.1, 10, 0.3, 0.2, salient_eff_dim=3.0, seed=1),
        _row(0.5, 20, 0.4, 0.1, salient_eff_dim=5.0, seed=0)])
    out = summarise_contrast_grid(df)
    assert len(out) == 2
    assert set(out['n']) == {2, 1}


def test_quality_range_exposes_a_plateau():
    # Two points whose medians differ by less than either one's own spread.
    df = pd.DataFrame([
        _row(0.1, 10, 0.90, 0.10, salient_eff_dim=4.0, seed=0),
        _row(0.1, 10, 0.30, 0.10, salient_eff_dim=4.0, seed=1),
        _row(0.5, 20, 0.62, 0.10, salient_eff_dim=4.0, seed=0),
        _row(0.5, 20, 0.58, 0.10, salient_eff_dim=4.0, seed=1)])
    out = summarise_contrast_grid(df).set_index('wasserstein_penalty')
    gap = abs(out.loc[0.1, 'quality_median'] - out.loc[0.5, 'quality_median'])
    assert out.loc[0.1, 'quality_range'] > gap
