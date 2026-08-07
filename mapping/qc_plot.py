#!/usr/bin/env python3

import logging

import numpy as np
import pandas as pd
import scanpy as sc

from dataclasses import dataclass, field
from mizani.transforms import trans
from pathlib import Path
from sklearn.decomposition import PCA
from pandas.api.types import is_numeric_dtype
from plotnine import *
from shared.utils import write_done_flag, get_sample_output_paths
from shared.samples import load_samples
from shared.io import write_h5ad
from shared.dimred import DimReducer, UMAPPlotter
from .qc_filter import (compute_dod_distances, collect_removal_flags, feature_transform,
                        invert_transform, LOGIT_CLIP, MAX_DOD_CELLS, NON_FILTER_METHODS)

# This stage's own figure prefixes, kept in sync with report.py's
# _METHOD_FIGURE_PREFIX. Used to clear stale figures before regenerating, so an
# earlier stage's figures must NOT be listed here. See
# docs/decisions/qc-terminal-umap.md.
_FIGURE_PREFIXES = ('bcrank', 'MAD', 'absoluteCut', 'miqc', 'regression', 'gmm', 'iF', 'DoD', 'solo', 'scrublet', 'consensus', 'filtered')

# Methods whose plotter handles a skipped metadata dict itself. Every other
# method's metadata omits the keys a real plot needs once skipped, so run()
# guards those generically.
_PLOTS_ON_SKIP = {'miqc'}

# Fewest surviving cells the terminal UMAP will attempt. A technical floor, not
# a quality threshold; samples under it still get every artifact except the
# embedding. See docs/decisions/qc-terminal-umap.md.
_MIN_CELLS_FOR_EMBEDDING = 20

# Fraction values a logit axis is worth labelling, in raw units. Fixed because
# the useful ones are not evenly spaced.
_FRACTION_BREAKS = np.array([0.0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02,
                             0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])


def _logit(x):
    """Logit of `x`, clipped to LOGIT_CLIP so exact 0 and 1 stay finite."""
    x = np.clip(np.asarray(x, dtype=float), LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(x / (1.0 - x))


def _thin_breaks(candidates, forward, limits, n: int = 6):
    """Up to `n` of `candidates` inside `limits`, spread along the `forward` axis.

    Greedy with a minimum gap rather than "nearest candidate to each of n evenly
    spaced targets": the latter picks candidates that are adjacent on the
    transformed axis, which on a logit axis renders as `0.00050010` -- two labels
    printed on top of each other.

    Args:
        candidates (np.ndarray): Ascending candidate break values, in raw units.
        forward (callable): The axis transform, applied to measure spacing.
        limits (tuple): (lo, hi) of the axis, in raw units.
        n (int): Maximum number of breaks.

    Returns:
        np.ndarray: The selected breaks.
    """
    lo, hi = min(limits), max(limits)
    sel = candidates[(candidates >= lo) & (candidates <= hi)]
    if len(sel) < 2:
        return np.array([lo, hi])
    if len(sel) <= n:
        return sel
    t = forward(sel)
    gap = (t[-1] - t[0]) / n
    keep = [0]
    for i in range(1, len(sel)):
        if t[i] - t[keep[-1]] >= gap:
            keep.append(i)
    return sel[keep]


def _fraction_breaks(limits, n: int = 6):
    """Fraction breaks for a logit axis, from the fixed `_FRACTION_BREAKS` list.

    Fixed rather than computed: the action is at 0.1% and 1%, and a generic break
    algorithm over logit space returns labels like 0.00734. See
    docs/decisions/qc-plot-axes.md.

    Args:
        limits (tuple): (lo, hi) of the axis, in raw units.
        n (int): Maximum number of breaks.

    Returns:
        np.ndarray: The selected breaks.
    """
    return _thin_breaks(_FRACTION_BREAKS, _logit, limits, n)


def _count_breaks(limits, n: int = 6):
    """1/2/5 x 10^k decades covering `limits`.

    mizani's `breaks_extended` works in data space and on a 200-2400 gene count
    returns just 1000 and 2000 -- two labels, both in the right-hand third. See
    docs/decisions/qc-plot-axes.md.

    Args:
        limits (tuple): (lo, hi) of the axis, in raw units.
        n (int): Maximum number of breaks.

    Returns:
        np.ndarray: The selected breaks.
    """
    hi = max(limits)
    decades = [0.0]
    m = 1.0
    while m <= max(hi, 1) * 10 and len(decades) < 40:
        decades += [m, 2 * m, 5 * m]
        m *= 10
    return _thin_breaks(np.array(decades), np.log1p, limits, n)


@dataclass(kw_only=True)
class logit_trans(trans):
    """A logit x-axis carrying **raw** fraction labels.

    Hand-rolled because mizani's own `logit_trans` transforms with the logistic CDF
    and never sets `breaks_func`. See docs/decisions/qc-plot-axes.md.
    """
    domain = (LOGIT_CLIP, 1 - LOGIT_CLIP)
    breaks_func: object = field(default=_fraction_breaks)

    def transform(self, x):
        """Raw fractions to logit space."""
        return _logit(x)

    def inverse(self, x):
        """Logit space back to raw fractions."""
        return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


@dataclass(kw_only=True)
class log1p_axis_trans(trans):
    """mizani's `log1p_trans` with break values a person recognises."""
    domain = (0.0, np.inf)
    breaks_func: object = field(default=_count_breaks)

    def transform(self, x):
        """Raw counts to log1p space, clamping negatives to 0."""
        return np.log1p(np.maximum(np.asarray(x, dtype=float), 0.0))

    def inverse(self, x):
        """log1p space back to raw counts."""
        return np.expm1(np.asarray(x, dtype=float))


# The x-axis each transform gets in the MAD figures, keyed by what
# `feature_transform` reports.
_MAD_AXIS_TRANS = {'log1p': log1p_axis_trans, 'logit': logit_trans}


class QCPlotter():
    """Draw one sample's QC diagnostics and write its filtered artifact."""
    def __init__(self, sample_name: str, sample_base_dir: Path, sample_config: Path):
        self.logger = logging.getLogger('SingleCellMappingPipeline')

        self.sample_name = sample_name
        self.sample_cfg = load_samples(sample_config)
        self.mapper = self.sample_cfg[self.sample_name]['mapper'].lower()
        self.config_paths = [sample_config]

        self.paths = get_sample_output_paths(Path(sample_base_dir) / self.sample_name, mapper = self.mapper)
        self.raw_dir = self.paths['raw']
        self.out_dir = self.paths['processed']
        self.qc_dir = self.paths['qc']

        self.adata = sc.read_h5ad(self.out_dir / '{}_qcfilters.h5ad'.format(self.sample_name))

        self.plot_caller = {
            'mad': self._plot_mad,
            # The fixed-cap 'absolute' method reuses _plot_mad; its entries carry
            # mode='absolute', which _plot_mad branches on for title and prefix.
            'absolute': self._plot_mad,
            'gmm': self._plot_gmm,
            'bcrank': self._plot_bcrank,
            'iforest': self._plot_iforest,
            'dod': self._plot_dod,
            'regression': self._plot_regression,
            'miqc': self._plot_miqc,
            'solo': self._plot_solo,
            'scrublet': self._plot_scrublet,
            'doublet_consensus': self._plot_consensus,
        }

    def _parse_qc_diagnostics(self):
        """Yield (method, metadata) for every plottable `.uns['qc']` entry.

        Flattens the per-feature entries, stamping the feature name into each, and skips
        the sample-level verdicts in NON_FILTER_METHODS -- those are not filters and
        report.py renders them as their own banner.

        Yields:
            tuple: (method (str), metadata (dict)).
        """
        diag = self.adata.uns.get('qc', {})
        for method, entry in diag.items():
            if method in NON_FILTER_METHODS:
                # A sample-level verdict, not a filter -- nothing to plot, and
                # report.py renders it as its own banner instead of a table row.
                continue
            if isinstance(entry, dict) and all(isinstance(v, dict) for v in entry.values()):
                for subkey, metadata in entry.items():
                    metadata = metadata.copy()
                    metadata["feature"] = subkey
                    yield method, metadata

            elif isinstance(entry, dict):
                yield method, entry.copy()

    def _get_plotting_view(self, run_context: dict):
        """The AnnData view the given QC step actually saw.

        Each cascade step is seeded with the previous steps' flags, so a figure drawn
        over the full object would show cells that step never assessed.

        Args:
            run_context (dict): The step's recorded context, carrying
                'filtered_on_flags'.

        Returns:
            AnnData: `self.adata`, or a copy subset to the cells that step saw.
        """
        filter_flags = list(run_context.get("filtered_on_flags", []))

        if not filter_flags:
            return self.adata
        valid = np.ones(self.adata.n_obs, dtype=bool)
        for flag in filter_flags:
            if flag not in self.adata.obs:
                self.logger.warning(f"[QCPlot] Flag column '{flag}' not found in .obs, skipping.")
                continue
            valid &= ~self.adata.obs[flag].values
        return self.adata[valid].copy()

    def _save_plot(self, plot, name, prefix="plot", dpi=600, width=6, height=4):
        """Write a figure to `{qc_dir}/{prefix}_{name}.png`.

        Args:
            plot (plotnine.ggplot): The figure.
            name (str): Filename stem after the prefix.
            prefix (str): Groups the figure in the report; see _FIGURE_PREFIXES.
            dpi (int): Output resolution.
            width (float): Inches.
            height (float): Inches.
        """
        filename = f"{self.qc_dir}/{prefix}_{name}.png"
        plot.save(filename=filename, dpi=dpi, width=width, height=height)


    def _plot_bcrank(self, metadata: dict, save: bool=True):
        """Barcode-rank curve with the knee threshold marked.

        Args:
            metadata (dict): This method's `.uns['qc']` entry.
            save (bool): Whether to write the PNG.

        Returns:
            plotnine.ggplot: The figure.
        """
        input_col = metadata.get('feature')
        ranks = metadata.get("ranks")
        threshold = metadata.get("threshold")
        clips = metadata.get("clipped_range", [None, None])
        start_idx, end_idx = clips
        smoothed_vals = metadata.get("smoothed_vals")
        flag_col = metadata.get('flag_col')

        adata_view = self._get_plotting_view(metadata.get("run_context", {}))
        counts = adata_view.obs[input_col].dropna().values
        counts = counts[counts > 0]
        counts_sorted = np.sort(counts)[::-1]

        if len(ranks) != len(counts_sorted):
            self.logger.warning("[QCPlot] Length mismatch between stored ranks and data-derived counts — skipping plot.")
            return None

        # Raw line
        df_raw = pd.DataFrame({
            "rank": ranks,
            "count": counts_sorted,
            "source": "Raw"
        })

        # Smoothed line (if available)
        if smoothed_vals is not None:
            smoothed_curve = np.expm1(smoothed_vals[:len(ranks)])
            df_sm = pd.DataFrame({
                "rank": ranks,
                "count": smoothed_curve,
                "source": "Smoothed"
            })
            df_plot = pd.concat([df_raw, df_sm], ignore_index=True)
        else:
            df_plot = df_raw

        p = (
            ggplot(df_plot, aes(x="rank", y="count"))
            + geom_line(color="black", alpha=0.8)
            + scale_x_log10()
            + scale_y_log10()
            + facet_wrap("~source")
            + labs(
                title=f"Barcode Rank Plot ({input_col})",
                x="Barcode Rank (log10)",
                y="Counts (log10)"
            )
            + theme_bw()
        )

        if threshold is not None:
            p += geom_hline(yintercept=threshold, linetype="dashed", color="red")
            threshold_idx = next((r for c, r in zip(counts_sorted, ranks) if c < threshold), None)
            if threshold_idx:
                p += geom_vline(xintercept=threshold_idx, linetype="dashed", color="red")

        ymin, ymax = df_plot["count"].min() / 10, df_plot["count"].max() * 10
        if start_idx is not None and start_idx > 0:
            p += geom_rect(
                aes(xmin=min(ranks), xmax=ranks[start_idx], ymin=ymin, ymax=ymax),
                fill="lightgrey", alpha=0.08, inherit_aes=False
            )
        if end_idx is not None and end_idx < len(ranks):
            p += geom_rect(
                aes(xmin=ranks[end_idx], xmax=max(ranks), ymin=ymin, ymax=ymax),
                fill="lightgrey", alpha=0.08, inherit_aes=False
            )

        if save:
            self._save_plot(p, name=flag_col, prefix='bcrank')
        return p
        
    def _plot_mad(self, metadata: dict, save: bool=True):
        """MAD's distribution and cut-offs, drawn in the feature's **raw** units.

        The values plotted are raw but the axis carries the transform MAD fit on, so the
        distribution keeps its shape while the ticks stay in units a person can type
        back into the interactive panel. Also serves the fixed-cap 'absolute' method,
        which it branches on for title and prefix. See docs/decisions/qc-plot-axes.md.

        Args:
            metadata (dict): This method's `.uns['qc']` entry.
            save (bool): Whether to write the PNG.

        Returns:
            plotnine.ggplot: The figure.
        """
        # `feature` is the column MAD worked on (transformed for the adaptive
        # entries); `source_feature` is the raw column behind it.
        input_col = metadata.get('feature')
        raw_col = metadata.get('source_feature') or input_col
        # Only claim a transform when a different raw column was resolved: a
        # log1p axis over already-log1p values would misplace every cell.
        kind = feature_transform(input_col) if raw_col != input_col else ''
        # The adaptive rule vs a flat max_value/min_value cap. A fraction feature
        # carries one entry of each, and they read identically without this.
        mode = 'absolute' if metadata.get('mode') == 'absolute' else 'adaptive'
        thresholds = metadata.get('thresholds', {})
        lower = invert_transform(thresholds.get('lower', None), kind)
        upper = invert_transform(thresholds.get('upper', None), kind)
        flag_col = metadata.get('flag_col')

        adata_view = self._get_plotting_view(metadata.get("run_context", {}))
        x = adata_view.obs[raw_col].dropna()
        flagged = adata_view.obs.get(flag_col, pd.Series(False, index=adata_view.obs.index))

        if kind == 'logit':
            # logit(0) is -inf and plotnine drops it silently, so without this
            # the figure omits whole populations the filter did assess.
            x = x.clip(LOGIT_CLIP, 1 - LOGIT_CLIP)

        # Prepare plotting data
        df = pd.DataFrame({
            'feature': x,
            "flagged": flagged.loc[x.index].astype(bool)
        })
        # Build plot
        p = (
            ggplot(df, aes(x='feature', fill="flagged"))
            + geom_histogram(bins=50, alpha=0.6, position="identity")
            + scale_fill_manual(values={True: "red", False: "grey"})
            + labs(
                title=("Absolute cutoff: {}".format(raw_col) if mode == 'absolute'
                       else "MAD Filtering (adaptive): {}".format(raw_col)),
                x=raw_col + ({'logit': ' (logit scale)', 'log1p': ' (log1p scale)'}.get(kind, '')),
                y="Cell Count"
            )
            + theme_bw()
            + theme(legend_title=element_text(text="Flagged"))
        )
        axis_trans = _MAD_AXIS_TRANS.get(kind)
        if axis_trans is not None:
            p = p + scale_x_continuous(trans=axis_trans)
        if lower is not None:
            p = p + geom_vline(xintercept=lower, linetype="dashed", color="blue")
        if upper is not None:
            p = p + geom_vline(xintercept=upper, linetype="dashed", color="blue")

        if save:
            # Separate prefixes so report.py groups them under their own
            # headings. Wider than default: the feature names are long.
            if mode == 'absolute':
                self._save_plot(p, name=raw_col, prefix="absoluteCut", width=8)
            else:
                self._save_plot(p, name='{}_{}'.format(mode, raw_col), prefix="MAD", width=8)
        return p

    def _plot_regression(self, metadata: dict, save: bool=True):
        """Residual scatter for the regression filter, with its threshold band.

        Args:
            metadata (dict): This method's `.uns['qc']` entry.
            save (bool): Whether to write the PNG.

        Returns:
            plotnine.ggplot: The figure.
        """
        adata_view = self._get_plotting_view(metadata.get("run_context", {}))

        x_key, y_key = metadata["features"]
        flag_col = metadata["flag_col"]
        threshold_low = metadata['thresholds']['lower']
        threshold_high = metadata['thresholds']['upper']
        coef = float(metadata["coef"])
        intercept = float(metadata["intercept"])

        # Get data
        df = pd.DataFrame({
            x_key: adata_view.obs[x_key],
            y_key: adata_view.obs[y_key],
            "flagged": adata_view.obs[flag_col]
        }, index=adata_view.obs_names).replace([np.inf, -np.inf], np.nan).dropna()

        p = (ggplot(df, aes(x=x_key, y=y_key, color="flagged")) + geom_point(alpha=0.6, size=1.5) + geom_abline(intercept=intercept, slope=coef, color="blue", linetype="solid", size=1) +
        scale_color_manual(values={True: "red", False: "black"}) + 
        labs(title=f"Regression Filter: {y_key} ~ {x_key}", x=x_key, y=y_key, color="Flagged")
        + theme_bw()
        + theme(figure_size=(6, 5), axis_text=element_text(size=10), title=element_text(size=12)))

        if threshold_low is not None:
            p += geom_abline(intercept=intercept + threshold_low, slope=coef, color="lightgrey", linetype="dashed", size=1)

        if threshold_high is not None:
            p += geom_abline(intercept=intercept + threshold_high, slope=coef, color="lightgrey", linetype="dashed", size=1)

        if save:
            self._save_plot(p, name=f"{x_key}_{y_key}", prefix='regression')
        return p

    def _plot_miqc(self, metadata: dict, save: bool=True):
        """miQC's (complexity, fraction) plane with the intact-component trend line.

        Handles a skipped entry itself: for miqc a skip is the common case, not an edge
        case. See docs/decisions/qc-terminal-umap.md.

        Args:
            metadata (dict): This method's `.uns['qc']` entry.
            save (bool): Whether to write the PNG.

        Returns:
            plotnine.ggplot: The figure.
        """
        feature = metadata['feature']
        complexity_key = metadata['complexity_key']
        adata_view = self._get_plotting_view(metadata.get('run_context', {}))

        if metadata.get('skipped'):
            df = pd.DataFrame({
                complexity_key: adata_view.obs[complexity_key],
                feature: adata_view.obs[feature],
            }, index=adata_view.obs_names).replace([np.inf, -np.inf], np.nan).dropna()

            p = (ggplot(df, aes(x=complexity_key, y=feature))
                + geom_point(alpha=0.5, size=1.2, color='black')
                + labs(title='miQC (not filtered): {} ~ {}'.format(feature, complexity_key),
                       subtitle=str(metadata.get('skip_reason', '')),
                       x=complexity_key, y=feature)
                + theme_bw())

            if save:
                self._save_plot(p, name=feature, prefix='miqc')
            return p

        flag_col = metadata['flag_col']
        posterior_col = metadata['posterior_col']
        posterior_threshold = metadata['posterior_threshold']
        slope = metadata['intact_line']['slope']
        intercept = metadata['intact_line']['intercept']

        df = pd.DataFrame({
            complexity_key: adata_view.obs[complexity_key],
            feature: adata_view.obs[feature],
            'flagged': adata_view.obs[flag_col]
        }, index=adata_view.obs_names).replace([np.inf, -np.inf], np.nan).dropna()

        p = (ggplot(df, aes(x=complexity_key, y=feature, color='flagged'))
            + geom_point(alpha=0.6, size=1.5)
            + geom_abline(intercept=intercept, slope=slope, color='blue', linetype='solid', size=1)
            + scale_color_manual(values={True: 'red', False: 'black'})
            + labs(title=f'miQC Filter: {feature} ~ {complexity_key}',
                   x=complexity_key, y=feature, color='Flagged')
            + theme_bw())

        plot_posterior = self._plot_score_histogram(
            adata_view.obs, score_col=posterior_col, threshold=posterior_threshold,
            method_name='miQC posterior ({})'.format(feature))

        if save:
            self._save_plot(p, name=feature, prefix='miqc')
            self._save_plot(plot_posterior, name=f'{feature}_posterior', prefix='miqc')
        return p, plot_posterior

    def _plot_solo(self, metadata: dict, save: bool=True):
        """SOLO doublet-score distribution against its threshold.

        Args:
            metadata (dict): This method's `.uns['qc']` entry.
            save (bool): Whether to write the PNG.

        Returns:
            plotnine.ggplot: The figure.
        """
        score_col = metadata['score_column']
        flag_col = metadata['flag_col']
        threshold = metadata.get('threshold', 0.5)

        adata_view = self._get_plotting_view(metadata.get('run_context', {}))
        assessed = adata_view.obs[score_col].notna().values
        sub = adata_view[assessed]
        if sub.n_obs == 0:
            self.logger.warning('[QCPlot] No cells carry a SOLO score; skipping doublet plots.')
            return None

        plot_score = self._plot_score_histogram(
            sub.obs, score_col=score_col, threshold=threshold, method_name='SOLO doublet')

        plot_umap = None
        if 'X_umap_scVI' in sub.obsm:
            coords = np.asarray(sub.obsm['X_umap_scVI'], dtype=float)
            keep = ~np.isnan(coords).any(axis=1)
            df = pd.DataFrame({
                'UMAP1': coords[keep, 0],
                'UMAP2': coords[keep, 1],
                'score': sub.obs[score_col].values[keep],
                'doublet': sub.obs[flag_col].astype(bool).values[keep]})
            plot_umap = (
                ggplot(df, aes(x='UMAP1', y='UMAP2', color='score'))
                + geom_point(size=0.7, alpha=0.7)
                + scale_color_gradient(low='lightgrey', high='red', name='doublet score')
                + labs(title='scVI UMAP coloured by SOLO doublet score',
                       subtitle='{} of {} assessed cells called doublet at score > {}'.format(
                           int(df['doublet'].sum()), len(df), threshold))
                + theme_bw())

        if save:
            for plot, name in [(plot_score, 'score'), (plot_umap, 'umap')]:
                if plot is not None:
                    self._save_plot(plot, name=name, prefix='solo')
        return plot_score, plot_umap

    def _plot_scrublet(self, metadata: dict, save: bool=True):
        """Scrublet doublet-score distribution against its threshold.

        Args:
            metadata (dict): This method's `.uns['qc']` entry.
            save (bool): Whether to write the PNG.

        Returns:
            plotnine.ggplot: The figure.
        """
        score_col = metadata['score_column']
        flag_col = metadata['flag_col']
        threshold = metadata.get('threshold', 0.5)

        adata_view = self._get_plotting_view(metadata.get('run_context', {}))
        assessed = adata_view.obs[score_col].notna().values
        sub = adata_view[assessed]
        if sub.n_obs == 0:
            self.logger.warning('[QCPlot] No cells carry a Scrublet score; skipping scrublet plots.')
            return None

        plot_score = self._plot_score_histogram(
            sub.obs, score_col=score_col, threshold=threshold, method_name='Scrublet doublet')

        plot_umap = None
        if 'X_umap_scVI' in sub.obsm:
            coords = np.asarray(sub.obsm['X_umap_scVI'], dtype=float)
            keep = ~np.isnan(coords).any(axis=1)
            df = pd.DataFrame({
                'UMAP1': coords[keep, 0],
                'UMAP2': coords[keep, 1],
                'score': sub.obs[score_col].values[keep],
                'doublet': sub.obs[flag_col].astype(bool).values[keep]})
            plot_umap = (
                ggplot(df, aes(x='UMAP1', y='UMAP2', color='score'))
                + geom_point(size=0.7, alpha=0.7)
                + scale_color_gradient(low='lightgrey', high='red', name='doublet score')
                + labs(title='scVI UMAP coloured by Scrublet doublet score',
                       subtitle='{} of {} assessed cells called doublet at score > {}'.format(
                           int(df['doublet'].sum()), len(df), threshold))
                + theme_bw())

        if save:
            for plot, name in [(plot_score, 'score'), (plot_umap, 'umap')]:
                if plot is not None:
                    self._save_plot(plot, name=name, prefix='scrublet')
        return plot_score, plot_umap

    def _plot_consensus(self, metadata: dict, save: bool=True):
        """Agreement between the doublet callers that ran.

        Args:
            metadata (dict): This method's `.uns['qc']` entry.
            save (bool): Whether to write the PNG.

        Returns:
            plotnine.ggplot: The figure.
        """
        if 'X_umap_scVI' not in self.adata.obsm:
            return None
        for col in ('flag_solo', 'flag_scrublet', 'solo_doublet_score', 'scrublet_doublet_score'):
            if col not in self.adata.obs:
                return None

        adata_view = self._get_plotting_view(metadata.get('run_context', {}))
        assessed = (adata_view.obs['solo_doublet_score'].notna().values
                    & adata_view.obs['scrublet_doublet_score'].notna().values)
        sub = adata_view[assessed]
        if sub.n_obs == 0:
            return None

        coords = np.asarray(sub.obsm['X_umap_scVI'], dtype=float)
        keep = ~np.isnan(coords).any(axis=1)
        solo = sub.obs['flag_solo'].astype(bool).values[keep]
        scr = sub.obs['flag_scrublet'].astype(bool).values[keep]
        agreement = np.where(solo & scr, 'both',
                    np.where(solo & ~scr, 'SOLO only',
                    np.where(~solo & scr, 'Scrublet only', 'neither')))
        df = pd.DataFrame({'UMAP1': coords[keep, 0], 'UMAP2': coords[keep, 1],
                           'agreement': pd.Categorical(
                               agreement, categories=['both', 'SOLO only', 'Scrublet only', 'neither'])})
        plot_umap = (
            ggplot(df, aes(x='UMAP1', y='UMAP2', color='agreement'))
            + geom_point(size=0.7, alpha=0.7)
            + scale_color_manual(values={'both': '#d62728', 'SOLO only': '#1f77b4',
                                         'Scrublet only': '#ff7f0e', 'neither': 'lightgrey'},
                                 name='called by')
            + labs(title='scVI UMAP coloured by SOLO/Scrublet agreement',
                   subtitle='{} consensus doublets (called by both) of {} jointly-assessed cells'.format(
                       int((solo & scr).sum()), len(df)))
            + theme_bw())

        if save:
            self._save_plot(plot_umap, name='umap', prefix='consensus')
        return plot_umap

    def _reduce_to_pca(self, adata, features: list, n_components: int = 2):
        """Reduce `.obs` features to a few principal components.

        Args:
            adata (AnnData): Object to read features from.
            features (list[str]): `.obs` columns to reduce.
            n_components (int): Components to compute.

        Returns:
            tuple or None: (DataFrame of PCs joined to `.obs`, fitted PCA), or None if
                fewer than three cells have finite values.
        """
        df = adata.obs[features].replace([np.inf, -np.inf], np.nan).dropna()
        if df.empty or df.shape[0] < 3:
            self.logger.warning('[QC filter] Too few valid cells or NaN values for PCA reduction')
            return None

        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(df.values)

        pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)], index=df.index)

        # Merge with metadata for those cells only
        merged_df = pca_df.join(adata.obs, how="left")
        return merged_df, pca

    def _plot_2D_projection(self, adata, features: list, color_by: str):
        """Scatter of the first two PCs of `features`, coloured by `color_by`.

        Args:
            adata (AnnData): Object to read from.
            features (list[str]): `.obs` columns to reduce.
            color_by (str): `.obs` column to colour by.

        Returns:
            plotnine.ggplot or None: None if too few cells survive the PCA.

        Raises:
            ValueError: If `color_by` is not an `.obs` column.
        """
        pca_df, pca = self._reduce_to_pca(adata, features)
        if pca_df is None or pca_df.shape[0] < 3:
            self.logger.warning('[QC filter] Too few valid cells for PCA plotting')
            return None

        if color_by not in adata.obs.columns:
            raise ValueError(f"Color column '{color_by}' not found in adata.obs")
        pca_df[color_by] = adata.obs.loc[pca_df.index, color_by]

        p = (
            ggplot(pca_df, aes(x="PC1", y="PC2", color=color_by)) +
            geom_point(size=1.5, alpha=0.7) +
            labs(title="2D PCA Projection", x="PC1", y="PC2", color=color_by) +
            theme_bw() +
            theme(axis_text_x=element_text(rotation=45, hjust=1))
        )
        return p

    def _plot_score_histogram(self, df, score_col: str, threshold: float, method_name: str = "QC Score"):
        """Histogram of a score column with its threshold marked.

        Args:
            df (pandas.DataFrame): Frame carrying `score_col`.
            score_col (str): Column to plot.
            threshold (float): Where to draw the cut-off line.
            method_name (str): Title prefix.

        Returns:
            plotnine.ggplot: The figure.

        Raises:
            ValueError: If `score_col` is not a column of `df`.
        """
        if score_col not in df.columns:
            raise ValueError(f"Score column '{score_col}' not found in dataframe.")

        return (
            ggplot(df, aes(score_col)) +
            geom_histogram(bins=30, fill="steelblue", alpha=0.8) +
            geom_vline(xintercept=threshold, linetype="dashed", color="red") +
            labs(
                title=f"{method_name} Score Distribution",
                x=score_col,
                y="Cell Count"
            ) +
            theme_bw()
        )

    def _plot_parallel(self, adata, features: list, outlier_col: str = None, cluster_col: str = None, score_col: str = None, threshold: float = None, color_by: str = None, title: str = None):
        """Parallel-coordinates plot of several QC features at once.

        Args:
            adata (AnnData): Object to read from.
            features (list[str]): `.obs` columns forming the axes.
            outlier_col (str, optional): Boolean `.obs` column to highlight.
            cluster_col (str, optional): `.obs` column to group lines by.
            score_col (str, optional): `.obs` column carrying a per-cell score.
            threshold (float, optional): Cut-off on `score_col`.
            color_by (str, optional): `.obs` column to colour by.
            title (str, optional): Figure title.

        Returns:
            plotnine.ggplot or None: None if no feature is usable.
        """
        all_features = list(features)
        if score_col and score_col not in all_features:
            all_features.append(score_col)

        # Ensure all features exist
        missing = [f for f in all_features if f not in adata.obs.columns]
        if missing:
            raise ValueError(f"[QC Plot] Missing features in adata.obs: {missing}")
        if outlier_col is not None and outlier_col not in adata.obs.columns:
            raise ValueError(f"[QC Plot] Outlier column '{outlier_col}' not found in adata.obs")

        df = adata.obs[all_features].copy()
        # All-False when no outlier column was given: the outlier overlay below
        # then draws nothing and the median lines are taken over every cell.
        df["outlier"] = adata.obs[outlier_col] if outlier_col is not None else False

        if cluster_col:
            if cluster_col not in adata.obs.columns:
                raise ValueError(f"[QC Plot] Cluster column '{cluster_col}' not found")
            df["group"] = adata.obs[cluster_col].astype(str)
        else:
            df["group"] = "All"

        if color_by:
            if color_by not in adata.obs.columns:
                raise ValueError(f"[QC Plot] Color column '{color_by}' not found")
            df["color"] = adata.obs[color_by]

        df["cell_id"] = df.index

        # Melt to long format
        id_vars = ["cell_id", "group", "outlier"] + (["color"] if color_by else [])
        df_long = df.reset_index().melt(
            id_vars=id_vars,
            value_vars=all_features,
            var_name="feature",
            value_name="value"
        )

        # Median lines only for inliers
        inlier_df = df_long[df_long["outlier"] == False]
        median_df = inlier_df.groupby(["group", "feature"])["value"].median().reset_index()

        # Outlier lines
        outlier_df = df_long[df_long["outlier"]]

        p = ggplot()

        # Draw individual red lines for outliers
        if not outlier_df.empty:
            p += geom_line(
                data=outlier_df,
                mapping=aes(x="feature", y="value", group="cell_id"),
                color="red", alpha=0.3
            )

        # Draw median lines per group (either per cluster, or single group 'All')
        p += geom_line(
            data=median_df,
            mapping=aes(x="feature", y="value", group="group"),
            color="black", size=1.2
        )

        # Draw points
        point_df = df_long
        if color_by:
            color_vals = adata.obs[color_by]
            if is_numeric_dtype(color_vals):
                p += geom_point(
                    data=point_df,
                    mapping=aes(x="feature", y="value", color="color"),
                    size=0.6, alpha=0.6
                )
                p += scale_color_gradient(low="blue", high="red", name=color_by)
            else:
                p += geom_point(
                    data=point_df,
                    mapping=aes(x="feature", y="value", color="color"),
                    size=0.6, alpha=0.6
                )
        else:
            p += geom_point(
                data=point_df,
                mapping=aes(x="feature", y="value"),
                color="lightgray", size=0.5, alpha=0.4
            )

        # Optional threshold line (for score_col only)
        if score_col and threshold is not None and score_col in all_features:
            p += geom_hline(yintercept=threshold, linetype="dashed", color="black")

        p += theme_bw() + theme(
            axis_text_x=element_text(rotation=45, hjust=1),
            figure_size=(10, 5))

        if title:
            p += ggtitle(title)

        return p + facet_wrap("~group")


    def _plot_gmm(self, metadata, save: bool=True):
        """GMM component fit with the flagged cells marked.

        Args:
            metadata (dict): This method's `.uns['qc']` entry.
            save (bool): Whether to write the PNG.

        Returns:
            plotnine.ggplot: The figure.
        """
        mahal_key = metadata["distance_col"]
        cluster_key = metadata["cluster_col"]
        flag_key = metadata["flag_col"]
        threshold = metadata["thresholds"]
        features = metadata["features"]
        run_context = metadata.get("run_context", {})

        adata_view = self._get_plotting_view(run_context)

        plot_parallel = self._plot_parallel(adata_view, features = features, outlier_col = flag_key, cluster_col = cluster_key, score_col = mahal_key, threshold = threshold, color_by = mahal_key, title = 'GMM features')
        plot_mahal = self._plot_score_histogram(adata_view.obs, score_col = mahal_key, threshold = threshold, method_name = 'gmm')

        if len(features) < 2:
            self.logger.warning('[QCPlot] detected fewer then 2 features used for GMM, skipping 2D projections')
            plot_flags = None
            plot_cluster = None
        else:
            plot_flags = self._plot_2D_projection(adata_view, features, color_by = flag_key)
            plot_cluster = self._plot_2D_projection(adata_view, features, color_by = cluster_key)

        if save:
            for plot, name in [(plot_cluster, "cluster"), (plot_flags, "flags"), (plot_mahal, "mahal"), (plot_parallel, "parallel_coordinates")]:
                if plot:
                    self._save_plot(plot, name=name, prefix='gmm')
        return plot_cluster, plot_flags, plot_mahal, plot_parallel

    def _plot_iforest(self, metadata, save: bool=True):
        """Isolation-forest score distribution against its contamination quota.

        Args:
            metadata (dict): This method's `.uns['qc']` entry.
            save (bool): Whether to write the PNG.

        Returns:
            plotnine.ggplot: The figure.
        """
        adata_view = self._get_plotting_view(metadata.get("run_context", {}))
        score_col = metadata['score_column']
        threshold = metadata['threshold']
        features = metadata['features']

        plot_score = self._plot_score_histogram(
            adata_view.obs, score_col=score_col, threshold=threshold, method_name='iF')
        # Parallel coordinates coloured by the continuous score, not by the flag.
        plot_parallel = self._plot_parallel(
            adata_view, features, outlier_col=None, score_col=score_col,
            threshold=threshold, color_by=score_col, title='iF features')

        if save:
            for plot, name in [(plot_parallel, "parallel"), (plot_score, "score")]:
                if plot:
                    self._save_plot(plot, name=name, prefix='iF')
        return plot_parallel

    def _plot_dod(self, metadata, save: bool=True):
        """Distance-of-distance diagnostics: change, neighbourhood overlap, and their joint.

        Args:
            metadata (dict): This method's `.uns['qc']` entry.
            save (bool): Whether to write the PNG.

        Returns:
            plotnine.ggplot: The figure.
        """
        adata_view = self._get_plotting_view(metadata.get("run_context", {}))

        flag_col = metadata["flag_col"]
        change_col = metadata.get("change_column", "dod_change")
        overlap_col = metadata.get("overlap_column", "dod_overlap")
        features = metadata.get("features", [])

        # Restrict to the cells the filter actually ran on: _zscore_features drops
        # NaN/inf rows, so the DoD matrices were built on a subset of this view.
        # (The previous `obs_names.isin(self.adata.obs_names)` mask was always all
        # True, since a view is by definition a subset, and so never realigned
        # anything.)
        usable = adata_view.obs[features].replace([np.inf, -np.inf], np.nan).dropna().index
        sub = adata_view[usable]

        flags = sub.obs[flag_col].values
        dod_change = sub.obs[change_col].values
        overlap = sub.obs[overlap_col].values

        # Recomputed rather than read back from .uns: these are n x n, and
        # persisting them wrote ~400 MB of QC metadata into every h5ad. The
        # transform is vectorised, so this is ~1s at the 5000-cell cap.
        X = sub.obs[features].to_numpy(dtype=float)
        d_raw, d_dod = compute_dod_distances(
            X, k=metadata.get('k', 15), logger=self.logger,
            max_cells=metadata.get('max_cells', MAX_DOD_CELLS))

        # Raw-distance and DoD-transformed UMAPs: no leiden (never requested at
        # these two call sites, only a color-by-flag scatter on the full cell
        # set DoD ran on). Fresh .copy() per call, matching the old _plot_umap's
        # own copy-before-mutating behaviour -- `sub` itself is a view.
        sub_raw = sub.copy()
        keys_raw = DimReducer(sub_raw, self.logger).compute(
            distances=d_raw, key='dod_raw', do_leiden=False, metric='euclidean')
        plot_umap_raw = UMAPPlotter(sub_raw, self.qc_dir, self.logger).plot(
            umap_key=keys_raw['umap_key'], leiden_key=None, color=flags,
            title_prefix="UMAP (Raw Distance)", save=False)[0]

        sub_dod = sub.copy()
        keys_dod = DimReducer(sub_dod, self.logger).compute(
            distances=d_dod, key='dod_dod', do_leiden=False, metric='euclidean')
        plot_umap_dod = UMAPPlotter(sub_dod, self.qc_dir, self.logger).plot(
            umap_key=keys_dod['umap_key'], leiden_key=None, color=flags,
            title_prefix="UMAP (distance of distance)", save=False)[0]

        # Filtered (post-flag-removal) UMAPs: cluster the surviving cells, once
        # on the same z-scored QC-feature space DoD itself used, and once on
        # corrected_counts gene expression -- matching the old two raw_mtx=
        # calls, both of which excluded flagged cells before embedding.
        sub_filt = sub[~flags].copy()
        sub_filt.obsm['X_qc_features'] = sub_filt.obs[features].to_numpy()
        keys_filt = DimReducer(sub_filt, self.logger).compute(
            use_rep='X_qc_features', key='filt', metric='cosine', do_leiden=True)
        plot_umap_filt = UMAPPlotter(sub_filt, self.qc_dir, self.logger).plot(
            umap_key=keys_filt['umap_key'], leiden_key=keys_filt['leiden_key'],
            title_prefix="UMAP (Filtered)", save=False)[0]

        # No gene-expression UMAP here. There used to be one -- corrected_counts
        # on the DoD-surviving cells -- but it is not a DoD diagnostic: it is a
        # second copy of the final UMAP that run() already produces, and it was
        # the reason consensus HVG selection (eight ranking methods, the slowest
        # step in this path) ran twice per sample. The three plots below are the
        # ones that actually say something about DoD: the embedding under raw
        # distances, the same under the DoD transform, and the QC-feature space
        # after its flags are applied.
        plots = [
            self._plot_dod_change(dod_change, flags = flags),
            self._plot_dod_neighbourhood(overlap, flags = flags),
            self._plot_dod_vs_overlap(dod_change, overlap, flags = flags),
            plot_umap_raw,
            plot_umap_dod,
            plot_umap_filt]
        if save:
            for p, n in zip(plots, ['change', 'overlap', 'change_overlap', 'umap_raw', 'umap_dod', 'umap_filt']):
                if p:
                    self._save_plot(p, name=n, prefix='DoD')

    def _plot_dod_change(self, delta: list, flags: list = None):
        """Histogram of per-cell distance-of-distance change.

        Args:
            delta (list[float]): Per-cell change.
            flags (list[bool], optional): Which cells were flagged.

        Returns:
            plotnine.ggplot: The figure.
        """
        df = pd.DataFrame({'DoD_change': delta})
        if flags is not None:
            df["flagged"] = np.array(flags)
            p = (
                ggplot(df, aes(x='DoD_change', fill='flagged'))
                + geom_histogram(bins=100, alpha=0.8)
                + scale_fill_manual(values={False: "steelblue", True: "red"})
            )
        else:
            p = (
                ggplot(df, aes(x='DoD_change'))
                + geom_histogram(bins=100, fill='steelblue', alpha=0.8)
            )
        p += labs(title="DoD Change Histogram", x="DoD - Original Distance", y="Count")
        p += theme_bw() + theme(title=element_text(size=12), axis_text=element_text(size=10))
        return p

    def _plot_dod_neighbourhood(self, overlaps: list, flags: list = None):
        """Histogram of per-cell neighbourhood overlap before and after the transform.

        Args:
            overlaps (list[float]): Per-cell overlap fraction.
            flags (list[bool], optional): Which cells were flagged.

        Returns:
            plotnine.ggplot: The figure.
        """
        df = pd.DataFrame({'overlap': overlaps})
        if flags is not None:
            df["flagged"] = np.array(flags)
            p = (
                ggplot(df, aes(x='overlap', fill='flagged'))
                + geom_histogram(bins=100, alpha=0.8)
                + scale_fill_manual(values={False: "steelblue", True: "red"})
            )
        else:
            p = (
                ggplot(df, aes(x='overlap'))
                + geom_histogram(bins=100, fill='steelblue', alpha=0.8)
            )
        p += labs(title="Neighborhood Overlap", x="Shared Neighbors", y="Count")
        p += theme_bw() + theme(title=element_text(size=12), axis_text=element_text(size=10))
        return p

    def _plot_dod_vs_overlap(self, dod_change, overlap, flags: list = None):
        """Distance-of-distance change against neighbourhood overlap, per cell.

        Args:
            dod_change (list[float]): Per-cell change.
            overlap (list[float]): Per-cell overlap fraction.
            flags (list[bool], optional): Which cells were flagged.

        Returns:
            plotnine.ggplot: The figure.
        """
        df = pd.DataFrame({'DoD_change': dod_change, 'overlap': overlap})
        if flags is not None:
            df["flagged"] = np.array(flags)
            p = (
                ggplot(df, aes(x='overlap', y='DoD_change', color='flagged'))
                + geom_point(alpha=0.4, size=0.7)
                + scale_color_manual(values={False: "steelblue", True: "red"})
            )
        else:
            p = (
                ggplot(df, aes(x='overlap', y='DoD_change'))
                + geom_point(alpha=0.4, size=0.7, color='steelblue')
            )
        p += labs(title="DoD Change vs. Neighborhood Overlap", x="Shared Neighbors", y="DoD Change")
        p += theme_bw() + theme(title=element_text(size=12), axis_text=element_text(size=10))
        return p

    def run(self, force: bool=False):
        """Draw every diagnostic, then write the filtered artifact and final UMAP.

        Only methods whose effective mode is 'filter' contribute to the removal mask;
        advisory methods have recorded scores and a flag column for the report but
        remove nothing. The embedding is best-effort -- the artifact, metadata and
        manifest are the contract. See docs/decisions/qc-terminal-umap.md.

        Args:
            force (bool): Re-run even if `qcplot.done` exists.
        """
        self.logger.info('[QC Plot] Plotting QC diagnostics for sample: {}'.format(self.sample_name))
        done_path = self.paths['done'] / 'qcplot.done'
        if done_path.exists() and not force:
            self.logger.info('[QCPlot] skipping sample "{}"; qcplot.done exists (use --force-plotting to override)'.format(self.sample_name))
            return

        # report.py discovers figures by globbing, so a stale PNG from an
        # earlier run would keep appearing in every report.
        for prefix in _FIGURE_PREFIXES:
            for stale in self.qc_dir.glob('{}_*.png'.format(prefix)):
                stale.unlink()

        for method, metadata in self._parse_qc_diagnostics():
            if metadata.get('skipped') and method not in _PLOTS_ON_SKIP:
                # The filter declined to threshold this feature, so there is no
                # flag column or cut-off to draw. Guarded here rather than in
                # each plotter, which would all hard-index the missing keys.
                self.logger.info("[QCPlot] Nothing to plot for {} on '{}': {}".format(
                    method, metadata.get('feature', ''), metadata.get('skip_reason', 'skipped')))
                continue
            plot_func = self.plot_caller.get(method)
            if plot_func is None:
                self.logger.warning(f"[QCPlot] No plotter implemented for method '{method}'")
                continue
            try:
                plot_func(metadata)
            except Exception as e:
                self.logger.error(f"[QCPlot] Failed to plot {method} → {e}")

        # Write filtered object. Only methods that ran in exec_mode 'filter'
        # contribute; advisory methods have recorded scores and a flag column
        # for the report to render, but must not remove anything.
        keep_mask = np.ones(self.adata.n_obs, dtype=bool)
        flag_cols = collect_removal_flags(self.adata.uns.get("qc", {}))
        for flag in flag_cols:
            if flag not in self.adata.obs:
                # Recorded in .uns but never written to .obs. Removing nothing is
                # right, but say so: silently skipping is how a disabled filter
                # passes for a clean one.
                self.logger.warning(
                    "[QCPlot] Flag column '{}' is recorded in .uns['qc'] but missing from .obs; "
                    "not applying it.".format(flag))
                continue
            mask = self.adata.obs[flag].astype(bool).values
            keep_mask &= ~mask
        self.logger.info('[QCPlot] Removal mask built from {} filtering flag column(s): {}'.format(
            len(flag_cols), ', '.join(flag_cols) if flag_cols else 'none'))
        adata_filtered = self.adata[keep_mask].copy()
        
        # Computed on a throwaway copy so the neighbour graph stays out of the
        # terminal artifact; the embedding and leiden labels are copied back,
        # since the interactive report selects cells on them.
        #
        # Everything from here to the write is BEST-EFFORT -- the artifact, its
        # metadata and the manifest are the stage's contract, and a sample that
        # cannot be embedded must still produce all three. The n_obs floor is
        # checked first so the common case gives a clear reason rather than an
        # opaque error from inside scanpy. See docs/decisions/qc-terminal-umap.md.
        skip_reason = ''
        if adata_filtered.n_obs < _MIN_CELLS_FOR_EMBEDDING:
            skip_reason = (
                'only {} cells survived the QC cascade, below the {}-cell floor needed to build '
                'an HVG/PCA/neighbour graph'.format(adata_filtered.n_obs, _MIN_CELLS_FOR_EMBEDDING))
            self.logger.warning(
                '[QCPlot] Sample "{}": skipping the final UMAP -- {}. The filtered artifact, its '
                'metadata and the HTML report are still written; the report will show the reason '
                'in place of the embedding. No X_umap_qc/leiden_qc for this sample, so the '
                'interactive report cannot lasso or cluster-select on it.'.format(
                    self.sample_name, skip_reason))
        else:
            adata_umap = adata_filtered.copy()
            try:
                keys = DimReducer(adata_umap, self.logger).compute(
                    layer='corrected_counts', metric='cosine', do_leiden=True,
                    hvg_outfn=self.qc_dir / f"{self.sample_name}_hvg_consensus.tsv")
                plots = UMAPPlotter(adata_umap, self.qc_dir, self.logger).plot(
                    umap_key=keys['umap_key'], leiden_key=keys['leiden_key'], title_prefix="UMAP", save=False)
                self._save_plot(plots[0], 'final_umap', prefix="filtered", dpi=600, width=6, height=4)

                # The '_qc' suffix is load-bearing: it is what orchestrator.py
                # keys off to drop per-sample embeddings before concatenation.
                # See docs/decisions/concatenation-hygiene.md.
                if keys.get('umap_key') and keys['umap_key'] in adata_umap.obsm:
                    adata_filtered.obsm['X_umap_qc'] = np.asarray(adata_umap.obsm[keys['umap_key']])
                if keys.get('leiden_key') and keys['leiden_key'] in adata_umap.obs:
                    adata_filtered.obs['leiden_qc'] = adata_umap.obs[keys['leiden_key']].values

                # So the report can describe how this embedding was built.
                # Merged, not overwritten: doublets.py may already have recorded
                # 'X_umap_scVI' here, carried over by the .copy() above.
                if keys.get('params'):
                    umap_params = dict(adata_filtered.uns.get('umap_params', {}))
                    umap_params['X_umap_qc'] = {k: ('' if v is None else v) for k, v in keys['params'].items()}
                    adata_filtered.uns['umap_params'] = umap_params
            except Exception as e:
                skip_reason = 'the embedding failed: {}'.format(e)
                self.logger.error(
                    '[QCPlot] Sample "{}": final UMAP failed on {} cells; continuing without it so '
                    'the filtered artifact and report are still produced. Reason: {}'.format(
                        self.sample_name, adata_filtered.n_obs, e))

        # Recorded whether or not it fired, so the report can distinguish "no
        # embedding, and here is why" from "this artifact predates the field".
        adata_filtered.uns['umap_qc_status'] = {
            'computed': not skip_reason,
            'skip_reason': skip_reason,
            'n_cells': int(adata_filtered.n_obs)}

        self.logger.info('[QCPlot] Finished QC diagnostic plotting for sample "{}", writing done flag to "{}"'.format(self.sample_name, done_path))
        write_h5ad(
            adata_filtered,
            self.qc_dir / f"{self.sample_name}_filtered.h5ad",
            sample=self.sample_name,
            mapper=self.mapper,
            stage='qc_plot',
            config_paths=self.config_paths,
            # n_cells_input is the only record of the pre-filter count: the filter
            # stage flags without subsetting, so self.adata still holds every
            # barcode that entered QC, while adata_filtered holds only survivors.
            extra={'n_cells': int(adata_filtered.n_obs),
                   'n_genes': int(adata_filtered.n_vars),
                   'n_cells_input': int(self.adata.n_obs)})
        adata_filtered.obs.to_csv("{}/metadata.tsv".format(self.qc_dir), sep="\t", index=True)
        self.logger.info('[QCPlot] Writing filtered object {} to "{}"'.format(f"{self.sample_name}_filtered.h5ad", self.qc_dir))
        self.logger.info('[QCPlot] Writing metadata file "metadata.tsv" to "{}"'.format(self.qc_dir))
        write_done_flag(done_path)










