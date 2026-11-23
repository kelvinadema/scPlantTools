#!/usr/bin/env python3

"""
Rank a `wasserstein_penalty` x `n_salient_latent` grid for one contrastiveVI target.

    python -m integration.cvi_sweep \
        --icfg configs/integration_config.yaml \
        --scfg configs/sample_metadata.yaml \
        --target MtSZ_cVI_10x_BAP

WHY THIS IS A SEPARATE ENTRY POINT and not `--tune` on the target: contrastiveVI's
objective is a sum of a target ELBO and a background ELBO with no term rewarding
salient/background separation. Raising `n_salient_latent` strictly improves target
reconstruction and any nonzero `wasserstein_penalty` strictly worsens the objective,
so an elbo_validation search selects maximum salient capacity and zero penalty --
the degenerate fit. validate.py refuses `tune: true` here for that reason. Every
criterion this script ranks on is computed OUTSIDE the objective.

WHAT IT WRITES. `{target}_contrast_grid.tsv` under the target's results dir, one
row per grid point. Nothing else: each fit is disposable (`persist=False`), so no
model, no h5ad, and the target's registry entry is never touched.

The table ACCUMULATES across runs by default, so extending a ladder does not
re-pay for the points already measured, and re-running a point stores it as a
replicate rather than overwriting. Replication matters: `leakage_ratio` carries
fit-to-fit variance that a single observation hides, so `best_grid_point` takes
the MEDIAN across replicates (docs/decisions/contrastive-sweep.md). Pass
`--fresh` to start a clean table.

This script only measures. To have a target choose its own parameters and train
the winner, set `sweep: true` on it in integration_config.yaml -- the
orchestrator then runs this same grid and trains the selected point, recording
the choice in the registry.

READING THE TABLE. Four columns, deliberately NOT collapsed into one score,
because they trade against each other and the balance is a judgement call:

  separation      background iLISI minus salient iLISI. Higher = the two latents
                  did their separate jobs. <=0 is degenerate. This is a screen,
                  not evidence the salient axis is biological.
  leakage_ratio   identity NMI in the salient space over the same in the
                  background space, measured on BACKGROUND CELLS ONLY -- they
                  had no perturbation, so any structure there is leakage by
                  construction. Lower is better; near 1 means the salient space
                  is a cell-type map that happens to split the contrast.
  qc_max_abs_rho  largest |Spearman| between a salient dimension and a QC
                  fraction. Matters more here than anywhere else in the pipeline
                  because ContrastiveVI CANNOT take covariates (scvi-tools
                  registers them and its module never reads them -- see
                  contrastive.module_supports_covariates), so this variation is
                  uncorrectable and can only be measured.
  responder_frac  fraction of target cells above the background 95th percentile
                  of salient norm. Present so a penalty that suppresses leakage
                  can be seen suppressing real signal too.

A good point has high `separation`, low `leakage_ratio`, low `qc_max_abs_rho`,
and a `responder_frac` that has not collapsed relative to penalty=0.

COST, AND HOW THE GRID SIZES ITSELF. Each point is one full fit per seed, run
serially, and per-fit cost scales with cell count -- roughly 33 min at 95k cells
down to ~1.5 min at 5.6k. The default grid is therefore not a constant:
`default_penalties(n_obs)` narrows the ladder for larger cohorts so one
contrast's grid stays inside one job, and the narrowed ladders are NESTED inside
the full one so a large contrast is still measured at rungs its (differently
sized) null contrast also has. Both entry points -- this CLI and the `sweep:
true` orchestrator path -- go through that one function.

Override per target with `sweep_grid:` in the integration config, or per run with
`--penalties` / `--salient`. Sizing lives here rather than in a chapter's config
because it follows from cell count and measured cost, not from what the contrast
is about; a chapter that hand-classifies its own targets is re-deriving this rule
and may disagree with it silently.
"""

import argparse
import logging
import sys
from pathlib import Path

from shared.targets import resolve_run_spec
from .orchestrator import IntegrationManager
from .contrastive import select_contrast_grid, summarise_contrast_grid, METHOD_NAME

# Default axes. The ladder has been cut off below its turnover twice; the
# measurements and what they showed are in docs/decisions/contrastive-sweep.md.
# The sub-1 rungs are kept because that is where leakage_ratio moves fastest,
# even though the minimum lies higher.
DEFAULT_PENALTIES = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)

# Narrower ladders for cohorts where the full one does not fit in a job. NESTED
# INSIDE DEFAULT_PENALTIES, and that is the constraint that shapes them, not
# taste: a treated contrast's `separation` and `responder_frac` mean nothing
# until there is a SAME-PENALTY number from a null contrast where no treatment
# happened, and a null is usually a different size from the contrast it
# calibrates. Non-nested ladders would put the two on rungs that never meet.
#
#   _PENALTIES_LARGE  subset of _PENALTIES_MID  subset of DEFAULT_PENALTIES
#
# What is dropped, and why in this order. 5/10/20 go first: every point at
# penalty >= 1 collapsed on all five measured contrasts, so the high end holds
# the least information -- but 1.0 and 2.0 are KEPT in every ladder, because
# that measurement was made at n_layers=1 (before base_model_params was passed)
# and the collapse boundary at the trained depth is still unmeasured. A ladder
# that stops below the turnover has been the mistake twice. 0.05 goes next as
# the finest sub-1 rung, then 0.1 and 0.5.
_PENALTIES_MID = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
_PENALTIES_LARGE = (0.0, 0.25, 1.0, 2.0)

# Cell counts at which the ladder narrows. Set from measured fit time -- roughly
# 33 min per fit at 95k cells, scaling with cell count -- so that one contrast's
# grid stays inside a single job at DEFAULT_SALIENT x DEFAULT_SEEDS. They are
# deliberately round: the cost is continuous and the boundary is a judgement,
# so a cohort that lands just the wrong side of one loses two rungs, not a
# result.
_MID_CELLS = 20_000
_LARGE_CELLS = 80_000

# Both axes are swept: capacity and penalty interact directly (more salient
# capacity gives shared variation more room to leak, which is what the penalty
# resists), so a penalty chosen at one capacity does not transfer to another.
# Three rungs around the published default is the smallest ladder that can
# show whether it matters.
DEFAULT_SALIENT = (5, 10, 20)

# Three fits per point. Enough to see whether two points differ by more than the
# noise, which one fit cannot show at all; more than three multiplies cost
# against a spread that is already visible at three.
DEFAULT_SEEDS = (0, 1, 2)


def default_penalties(n_obs: int = None) -> tuple:
    """The penalty ladder for a cohort of `n_obs` cells, nested within `DEFAULT_PENALTIES`.

    Pipeline logic rather than a per-target choice (see the module docstring); a
    target's `sweep_grid: {wasserstein_penalty: [...]}` still wins. `None` returns
    the full ladder, the right answer when the size is not yet known.
    """
    if n_obs is None or n_obs < _MID_CELLS:
        return DEFAULT_PENALTIES
    if n_obs < _LARGE_CELLS:
        return _PENALTIES_MID
    return _PENALTIES_LARGE


def build_grid(penalties, salient):
    """Grid points, penalty-major so the cheap baseline (penalty 0) runs first."""
    return [{'wasserstein_penalty': float(p), 'n_salient_latent': int(n)}
            for p in penalties for n in salient]


def main(argv=None):
    """CLI entry point: run a contrastiveVI grid for one target."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--icfg', required=True, help='Path to integration_config.yaml')
    parser.add_argument('--scfg', required=True, help='Path to sample_metadata.yaml')
    parser.add_argument('--target', required=True, help='A contrastiveVI target name')
    # Resolved after the AnnData is loaded, not here: the default ladder narrows
    # with cohort size and the cell count is not known at parse time.
    parser.add_argument('--penalties', type=float, nargs='+', default=None,
                        help='Penalty ladder (default: sized from the cohort by '
                             'default_penalties)')
    parser.add_argument('--salient', type=int, nargs='+', default=list(DEFAULT_SALIENT))
    parser.add_argument('--identity-key', default=None,
                        help='.obs column standing in for cell type in the leakage test '
                             '(default: contrastive.DEFAULT_IDENTITY_KEY)')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS),
                        help='Fit each grid point once per seed. Replication is what turns an '
                             'ordering into a claim: leakage_ratio carries fit-to-fit variance '
                             'large enough to reorder the top points, and scvi.settings.seed '
                             'defaults to None so unseeded runs were never reproducible.')
    parser.add_argument('--fresh', action='store_true',
                        help='Start a clean grid table instead of appending to the existing '
                             'one. Appending is the default so extending a ladder keeps what '
                             'is already measured, and repeated points accumulate as '
                             'replicates.')
    parser.add_argument('--out-dir', default=None,
                        help='Where to write the TSV (default: the target results dir)')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    logger = logging.getLogger('cvi_sweep')

    manager = IntegrationManager(args.icfg, args.scfg, args.target,
                                 dimreduction='umap', plot=False)
    spec = resolve_run_spec(manager.config, args.target)
    if spec['method'] != 'contrastiveVI':
        parser.error("target '{}' has method '{}'; this sweep is only for contrastiveVI "
                     'targets.'.format(args.target, spec['method']))

    # Same artifact resolution and layer dropping as the real run. Unlike the
    # orchestrator path, min_cells, derive_corrected_layers and batch-key
    # derivation are not applied here.
    if spec.get('artifact_key'):
        manager.artifact_key = spec['artifact_key']
    manager.keep_layers = spec.get('keep_layers')
    adata = manager._concatenate_adata(samples=spec['samples'])

    # Same rule the `sweep: true` orchestrator path uses, so the two entry points
    # measure the same ladder for the same contrast.
    penalties = args.penalties or default_penalties(adata.n_obs)
    grid = build_grid(penalties, args.salient)
    logger.info('[cvi_sweep] %s: %d grid point(s) x %d seed(s) over %d cells '
                '(penalties: %s)', args.target, len(grid), len(args.seeds),
                adata.n_obs, ', '.join(str(p) for p in penalties))

    out_dir = Path(args.out_dir) if args.out_dir else (
        manager.output_dir / METHOD_NAME / args.target)
    kwargs = {}
    if args.identity_key:
        kwargs['identity_key'] = args.identity_key

    # The target's own architecture under every grid point: depth and
    # wasserstein_penalty both govern how much shared variation reaches the
    # salient space, so a penalty selected at another depth is not comparable.
    # Grids written before 2026-08-04 were fitted at contrastive._MODEL_DEFAULTS'
    # n_layers=1; re-run rather than reuse. See docs/decisions/contrastive-sweep.md.
    base_params = (spec.get('model_params') or {}).get('model_params') or {}

    df = select_contrast_grid(
        adata,
        background_samples=spec['background'],
        target_samples=spec['target_samples'],
        target=args.target,
        grid=grid,
        covariates=spec['covariates'],
        batch_key=spec['batch_key'],
        layer=(spec.get('keep_layers') or ['corrected_counts'])[0],
        train_args=spec.get('train_args'),
        base_model_params=base_params,
        out_dir=out_dir,
        logger=logger,
        append=not args.fresh,
        seeds=args.seeds,
        **kwargs)

    summary = summarise_contrast_grid(df)
    cols = [c for c in ('wasserstein_penalty', 'n_salient_latent', 'n', 'quality_median',
                        'quality_range', 'separation_median', 'leakage_median',
                        'leakage_range', 'eff_dim_median', 'collapsed_frac')
            if c in summary.columns]
    print('\n=== {} (median over seeds; read quality_range before trusting the order) ===\n{}'
          .format(args.target, summary[cols].to_string(index=False)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
