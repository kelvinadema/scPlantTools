#!/usr/bin/env python3

import numpy as np
import pandas as pd
import scanpy as sc
import logging

from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from scipy.stats import chi2, shapiro, skew, kurtosis, gaussian_kde
from scipy.special import logit
from scipy.signal import savgol_filter, find_peaks

from collections.abc import Iterable

from pathlib import Path

from shared.utils import load_config, log_exceptions, write_done_flag, get_sample_output_paths
from shared.samples import load_samples
from shared.io import write_h5ad
from shared.config_spec import FILTER_MODES


# DoD builds several dense n x n matrices, so both time and memory are quadratic
# in cells; the paper's own guidance is that it suits smaller N.
MAX_DOD_CELLS = 5000

# Downgraded to 'advisory' on a flagged sample.
DISTRIBUTION_DEPENDENT_FILTERS = frozenset(
    {'mad', 'miqc', 'regression', 'gmm', 'iforest', 'dod', 'solo'})

# Methods bounding one feature on one axis; can be superseded by a hand-set
# threshold in the interactive report.
SINGLE_BOUND_METHODS = frozenset({'bcrank', 'mad', 'absolute'})

# .uns['qc'] method name -> filter_config.yaml key. Identical for every method
# except barcode rank, which is 'bcrank' in .uns and 'barcode_rank' in config.
UNS_METHOD_TO_CONFIG_KEY = {
    'bcrank': 'barcode_rank',
    'mad': 'mad',
    # The fixed max_value/min_value caps are recorded and reported as their own
    # 'absolute' method, but they live in and follow the 'mad' config block's mode.
    'absolute': 'mad',
    'miqc': 'miqc',
    'regression': 'regression',
    'gmm': 'gmm',
    'iforest': 'iforest',
    'dod': 'dod',
    'solo': 'solo',
    # Not a filter but a gate over the others; 'advisory' means "record the
    # verdict, do not skip anything". It records no flag_col, so it can never
    # reach a removal mask regardless of mode.
    'sample_quality': 'sample_quality',
}


# Clamp bounds for the logit transform; keeps 0 and 1 finite.
LOGIT_CLIP = 1e-6


def feature_transform(feature: str) -> str:
    """Which transform `_transform_features` applied to produce `feature`.

    Args:
        feature (str): Possibly-transformed feature name.

    Returns:
        str or None: Transform kind, or None if untransformed.
    """
    if not feature:
        return ''
    if feature.startswith('logit_'):
        return 'logit'
    if feature.startswith('log_'):
        return 'log1p'
    return ''


def invert_transform(value, kind: str):
    """Map a value in transformed space back to the raw feature scale.

    Args:
        value (float): Value in transformed space.
        kind (str): Transform kind.

    Returns:
        float: Value on the raw scale.
    """
    if value is None:
        return None
    value = float(value)
    if kind == 'logit':
        return float(1.0 / (1.0 + np.exp(-value)))
    if kind == 'log1p':
        return float(np.expm1(value))
    return value


def _is_per_feature(entry) -> bool:
    """True when a `.uns['qc'][method]` entry nests one payload per feature.

    Args:
        entry (dict): Recorded method entry.

    Returns:
        bool: Whether the entry is per-feature.
    """
    return bool(entry) and isinstance(entry, dict) and all(
        isinstance(v, dict) for v in entry.values())


# Sample-level verdicts, not filters: skipped by the flag collectors.
NON_FILTER_METHODS = frozenset({'sample_quality', 'post_cascade'})


def collect_removal_flags(uns_qc: dict) -> list:
    """The `.obs` flag columns that should actually remove cells.

    Union of the flags of every method whose effective mode is 'filter'. Skip
    records carry no flag column and pass over harmlessly.

    Args:
        uns_qc (dict): The `.uns['qc']` mapping.

    Returns:
        list[str]: Flag column names.
    """
    flags = []
    for method, entry in (uns_qc or {}).items():
        if method in NON_FILTER_METHODS:
            # A sample-level verdict, not a filter: no flag column, nothing to
            # remove. Skipped explicitly so it can never be walked into.
            continue
        if not isinstance(entry, dict):
            continue
        payloads = entry.values() if _is_per_feature(entry) else [entry]
        for meta in payloads:
            if not isinstance(meta, dict) or meta.get('skipped'):
                continue
            if meta.get('exec_mode', 'filter') != 'filter':
                continue
            flag_col = meta.get('flag_col')
            if flag_col:
                flags.append(flag_col)
    return flags


def compute_dod_distances(X, k: int=15, metric='euclidean', logger=None, max_cells: int=MAX_DOD_CELLS):
    """Distance-of-distance transform (doi:10.1371/journal.pcbi.1010764, Eq. 1).

    Quadratic in cell count; `max_cells` bounds it. See
    docs/decisions/qc-cascade.md.

    Args:
        X (np.ndarray): Cells x features.
        k (int): Neighbours used by the transform.
        metric (str): Distance metric.
        logger (logging.Logger, optional): Logger.
        max_cells (int, optional): Subsample above this many cells.

    Returns:
        tuple: (D_raw (np.ndarray), D_dod (np.ndarray), index (np.ndarray)).
    """
    log = logger or logging.getLogger('SingleCellMappingPipeline')
    n = X.shape[0]

    if n > max_cells:
        raise ValueError(
            '[QC Filter] DoD needs {n}x{n} dense matrices ({gb:.1f} GB) for {n} cells, '
            'above the {cap} cell cap. The transform is quadratic in cells and the '
            'paper targets smaller N. Raise dod.max_cells in filter_config.yaml '
            'deliberately (mind available RAM: 3*n^2*8 bytes), or filter on '
            'another method.'.format(n=n, cap=max_cells, gb=3 * n * n * 8 / 1e9))
    if k >= n:
        raise ValueError('[QC Filter] DoD neighbourhood K={} must be smaller than n={}'.format(k, n))
    if k > n / 10:
        log.warning(
            '[QC Filter] DoD K={} is large relative to n={}; the transform degrades as K '
            'approaches N, and K should stay below the smallest cluster size.'.format(k, n))

    log.info('[QC filter] Computing DoD transform (n={}, K={})'.format(n, k))

    D = pairwise_distances(X, metric=metric)

    # kneighbors() with no argument excludes each point from its own neighbours.
    nn = NearestNeighbors(n_neighbors=k, metric=metric).fit(X)
    neighbors = nn.kneighbors(return_distance=False)

    # T[i, j] = sum_{n in I_i} |d_ni - d_nj|; symmetrised below.
    T = np.empty((n, n), dtype=float)
    for i in range(n):
        nb = neighbors[i]                       # (K,) neighbours of i
        # D[nb, :] is (K, n): each neighbour's distance to every point.
        # D[nb, i] is (K,):   each neighbour's distance to i.
        T[i] = np.abs(D[nb, :] - D[nb, i][:, None]).sum(axis=0)

    F = (T + T.T) / (2.0 * k)
    del T
    np.fill_diagonal(F, 0.0)
    return D, F

class FilterQCMetrics():
    """Run the QC filter cascade over one sample and record every decision.

    Each stage is seeded with the previous stages' flags, so a filter only ever sees
    the cells its upstream stages passed. See docs/decisions/qc-cascade.md.
    """
    def __init__(self, sample_name: str, sample_base_dir: Path, sample_config: Path, filter_config_path: Path):
        self.logger = logging.getLogger('SingleCellMappingPipeline')

        self.sample_name = sample_name
        self.sample_cfg = load_samples(sample_config)
        self.mapper = self.sample_cfg[self.sample_name]['mapper'].lower()
        # Same key/default as preprocess.py -- drives max_value/min_value
        # resolution for sample-type-keyed MAD entries, see
        # _resolve_sample_type_value.
        self.sample_type = self.sample_cfg[self.sample_name].get('sample_type', 'protoplast')
        self.config_paths = [sample_config, filter_config_path]

        self.filter_profile = self.sample_cfg[self.sample_name].get('qc_filter', 'default')

        all_filter_configs = load_config(Path(filter_config_path)).get('QCfilterconfigs', {})
        if self.filter_profile not in all_filter_configs:
            self.logger.warning(
                '[QC Filter] QC filter profile "{}" not found in filter_config.yaml for sample "{}"; falling back to "default"'.format(
                    self.filter_profile, self.sample_name))
            self.filter_profile = 'default'
        self.filter_cfg = all_filter_configs[self.filter_profile]

        self.paths = get_sample_output_paths(Path(sample_base_dir) / self.sample_name, mapper = self.mapper)
        self.raw_dir = self.paths['raw']
        self.out_dir = self.paths['processed']

        self.adata = sc.read_h5ad(self.out_dir / '{}_qcfilters.h5ad'.format(self.sample_name))

        # Key are writting during metric computation, store this to simplify filter config specification
        self.QC_KEY_REGISTRY = {
            "mitochondrial_fraction": lambda layer: f"mitochondrial_fraction_{layer}",
            "chloroplast_fraction": lambda layer: f"chloroplast_fraction_{layer}",
            "ribosomal_rna_fraction": lambda layer: f"ribosomal_rna_fraction_{layer}",
            "ribosomal_protein_fraction": lambda layer: f"ribosomal_protein_fraction_{layer}",
            "ambient_fraction": lambda layer: f"{layer}_fraction",
            "total_counts": lambda layer: "total_counts" if layer == "X" else f"{layer}_total_counts",
            "n_genes": lambda layer: "n_genes" if layer == "X" else f"{layer}_n_genes",
            "entropy": lambda layer: f"entropy_{layer}",
            "sparsity": lambda layer: f"cell_sparsity_{layer}",
            "complexity": lambda layer: f"cell_complexity_{layer}",
            "top_n_genes": self._resolve_top_n_genes}

    def _resolve_top_n_genes(self, layer: str) -> str:
        """Resolve the top-N-genes feature name for a layer.

        Args:
            layer (str): Layer name.

        Returns:
            str: Feature name.
        """
        prefix = "fraction_top" if layer == "X" else f"{layer}_fraction_top"
        matches = sorted(c for c in self.adata.obs.columns
                         if c.startswith(prefix) and c[len(prefix):].isdigit())

        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(
                "[QC Filter] Filter requests 'top_n_genes' for layer '{}', but no '{}<n>' column "
                "exists in .obs. Enable 'fraction_top_n_genes' for this layer in qc_config.yaml.".format(
                    layer, prefix))
        raise KeyError(
            "[QC Filter] Filter requests 'top_n_genes' for layer '{}', but several candidates exist "
            "in .obs ({}). Reference the column by its exact name in filter_config.yaml.".format(
                layer, ", ".join(matches)))

    def _resolve_keys(self, metrics: list, layer: str = "X"):
        """Resolve configured metric names to `.obs` column names.

        Unrecognised names pass through as literal column names -- filter_config.yaml
        legitimately uses columns that are not registry aliases.

        Args:
            metrics (list[str]): Names as written in the config.
            layer (str, optional): Layer whose per-layer columns to resolve against.

        Returns:
            list[str]: `.obs` column names.
        """
        resolved = []
        for key in metrics:
            if key in self.QC_KEY_REGISTRY:
                resolved.append(self.QC_KEY_REGISTRY[key](layer))
            else:
                # Not a registry alias; treat it as a literal .obs column name.
                # _iterate_features raises if it turns out not to exist.
                resolved.append(key)
        return resolved

    def _filter_mode(self, key):
        """Configured execution mode for one filter block.

        Raises on an unrecognised value rather than defaulting to 'off', which would
        silently disable a filter. See docs/decisions/qc-cascade.md.

        Args:
            key (str): Filter block name.

        Returns:
            str: One of 'off', 'advisory', 'filter'.

        Raises:
            ValueError: If the configured mode is not recognised.
        """
        mode = self.filter_cfg.get("filters", {}).get(key, {}).get("mode", "off")
        if mode not in FILTER_MODES:
            raise ValueError(
                '[QC Filter] filter "{}" in profile "{}" has mode {!r}; expected one of {}. '
                '(This key replaced the old boolean `enabled:` -- "filter" is the equivalent '
                'of `enabled: true`, "off" of `enabled: false`.)'.format(
                    key, self.filter_profile, mode, ', '.join(repr(m) for m in FILTER_MODES)))
        return mode

    def _effective_mode(self, key, distribution_dependent=None) -> str:
        """The mode a filter actually runs in, after the sample_quality verdict.

        A flagged sample downgrades distribution-dependent filters from 'filter' to
        'advisory'. See docs/decisions/qc-cascade.md.

        Args:
            key (str): Filter block name.
            distribution_dependent (bool, optional): Override the per-method default.
                Needed because mad's adaptive and absolute entries differ.

        Returns:
            str: Effective mode.
        """
        configured = self._filter_mode(key)
        if distribution_dependent is None:
            distribution_dependent = key in DISTRIBUTION_DEPENDENT_FILTERS
        # getattr: barcode rank runs before _assess_sample_quality has set this.
        if (configured == 'filter' and distribution_dependent
                and getattr(self, 'low_quality_sample', False)):
            return 'advisory'
        return configured

    def _source_feature(self, feature):
        """The raw `.obs` column a possibly-transformed feature came from.

        Args:
            feature (str): Feature name.

        Returns:
            str: Source column name.
        """
        if not feature:
            return None
        inverse = {v: k for k, v in getattr(self, 'transformed_features', {}).items()}
        return inverse.get(feature, feature)

    def _record_qc_metadata(self, method: str, metadata: dict, feature: str=None, ):
        """Store QC metadata in `adata.uns['qc'][method]`.

        Args:
            method (str): Filter method name.
            metadata (dict): Payload to record.
            feature (str, optional): Feature name, when the entry is per-feature.
        """
        if "qc" not in self.adata.uns:
            self.adata.uns["qc"] = {}

        if method not in self.adata.uns["qc"]:
            self.adata.uns["qc"][method] = {}

        config_key = UNS_METHOD_TO_CONFIG_KEY.get(method)
        if config_key is not None:
            metadata = dict(metadata)
            # `absolute` records under its own method name so the report can tell a fixed
            # cap from an adaptive cut.
            distribution_dependent = None
            if method == 'absolute':
                distribution_dependent = False

            # Raw column this entry bounds, for the interactive override.
            if method in SINGLE_BOUND_METHODS:
                source = self._source_feature(metadata.get('feature') or feature)
                if source:
                    metadata['source_feature'] = source

            configured = self._filter_mode(config_key)
            effective = self._effective_mode(config_key, distribution_dependent)
            metadata['exec_mode'] = effective
            if effective != configured:
                metadata['downgraded_from'] = configured
                metadata['downgrade_reason'] = (
                    'sample_quality flagged this sample, so this threshold was fit from a '
                    'distribution too small or too shallow to trust; it was computed and is '
                    'shown for manual review, but removed no cells')

        if feature is not None:
            self.adata.uns["qc"][method][feature] = metadata
        else:
            self.adata.uns["qc"][method] = metadata


    def _iterate_features(self, features, callback):
        """Run `callback` for each configured feature, raising on one that is absent.

        A filter naming a feature absent from `.obs` raises rather than warning: silently
        disabling a filter lets the run report success on unfiltered data.

        Args:
            features (list): Feature config entries.
            callback (callable): Called with (key, values).

        Raises:
            KeyError: If a named feature is not an `.obs` column.
        """
        for key in features:
            if key not in self.adata_active.obs.columns:
                # Skipping here would silently disable the filter and let the run
                # report success on unfiltered data. A configured filter whose
                # feature does not exist is a config error, so say so.
                raise KeyError(
                    '[QC Filter] Feature "{}" requested by filter profile "{}" is not present in .obs. '
                    'Check that the corresponding metric is enabled in qc_config.yaml for this layer. '
                    'Available columns: {}'.format(
                        key, self.filter_profile, ', '.join(sorted(self.adata_active.obs.columns))))

            x = self.adata_active.obs[key].values
            if np.all(np.isnan(x)):
                self.logger.warning('[QC Filter] Feature: "{}" is all NaNs, skipping'.format(key))
                continue
            callback(key, x)

    def _get_filtered_view(self, filter_flags: list = None):
        """The cells that survived the given upstream flags.

        Args:
            filter_flags (list[str]): Flag columns to exclude on.

        Returns:
            AnnData: A view of the survivors.
        """
        self._adata_view_flags = []

        if not filter_flags:
            return self.adata

        valid = np.ones(self.adata.n_obs, dtype=bool)
        used_flags = []

        for flag in filter_flags:
            col = flag if flag.startswith("flag_") else f"flag_{flag}"
            if col not in self.adata.obs:
                self.logger.warning(f"[QC Filter] Flag column '{col}' not found in adata.obs. Skipping.")
                continue
            valid &= ~self.adata.obs[col].values
            used_flags.append(col)

        self._adata_view_flags = used_flags
        if not used_flags:
            # Nothing resolved, so the "view" is every cell: return the object
            # rather than copying the whole matrix to say so.
            return self.adata
        return self.adata[valid].copy()

    def _get_run_context(self) -> dict:
        """What this stage saw, recorded into its `.uns['qc']` entry.

        Returns:
            dict: The context, carrying 'filtered_on_flags'.
        """
        return {
            "adata_view_used": self.adata_view is not None,
            "filtered_on_flags": list(self._adata_view_flags) if hasattr(self, "_adata_view_flags") else []}

    # Fraction of its own input a single feature may flag before we treat the
    # result as a broken threshold rather than a biological finding.
    MAX_PLAUSIBLE_FLAG_FRACTION = 0.5

    def _check_flag_fraction(self, method: str, key: str, flags) -> float:
        """Warn when one feature flags most of the cells it saw.

        Warns rather than acts: a genuinely poor sample may legitimately lose most of
        its cells.

        Args:
            method (str): Filter method name.
            key (str): Feature name.
            flags (np.ndarray): Boolean flags.

        Returns:
            float: Fraction flagged.
        """
        flags = np.asarray(flags, dtype=bool)
        if flags.size == 0:
            return 0.0
        fraction = float(flags.mean())
        if fraction > self.MAX_PLAUSIBLE_FLAG_FRACTION:
            self.logger.warning(
                # One decimal place: 99.6% and 100% mean different things here.
                '[QC Filter] {} flagged {:.1%} of cells on "{}" ({}/{}). A single feature '
                'removing most of its input usually means a degenerate threshold, not '
                'biology -- inspect this before trusting the filtered object.'.format(
                    method, fraction, key, int(flags.sum()), flags.size))
        return fraction

    def _assess_sample_quality(self, filter_flags=None):
        """Decide whether this sample can be trusted to fit its own thresholds.

        Runs once, right after barcode rank. Sets `low_quality_sample`, which
        `_effective_mode` reads. See docs/decisions/qc-cascade.md.

        Args:
            filter_flags (list, optional): Flags recorded so far.
        """
        cfg = self.filter_cfg.get("filters", {}).get("sample_quality", {})
        self.low_quality_sample = False
        # 'advisory' still records scores and figures for review.
        mode = self._filter_mode('sample_quality')
        if mode == 'off':
            return

        layer = cfg.get('layer', 'corrected_counts')
        min_cells = cfg.get('min_cells', 500)
        min_median_depth = cfg.get('min_median_depth', 1000)
        depth_key = self._resolve_keys(['total_counts'], layer=layer)[0]

        self.adata_view = self._get_filtered_view(filter_flags)
        ad = self.adata_active

        n_cells = int(ad.n_obs)
        median_depth = float(np.nanmedian(ad.obs[depth_key].values)) if n_cells else 0.0
        run_context = self._get_run_context()
        self.adata_view = None

        reasons = []
        if n_cells < min_cells:
            reasons.append('only {} cells survived barcode rank (below min_cells={})'.format(
                n_cells, min_cells))
        if median_depth < min_median_depth:
            reasons.append('median depth {:.0f} among surviving cells is below min_median_depth={}'.format(
                median_depth, min_median_depth))

        flagged = bool(reasons)
        # Only 'filter' arms the gate. In 'advisory' the verdict is still
        # recorded and reported, but every downstream filter runs as if the
        # sample were fine.
        self.low_quality_sample = flagged and mode == 'filter'
        if flagged and mode == 'filter':
            self.logger.warning(
                '[QC Filter] Sample "{}" flagged for manual review: {}. Every filter fit from '
                'this sample\'s own distribution (mad\'s adaptive entries, miqc, regression, gmm, '
                'iforest, dod, solo) is downgraded to advisory: it still runs and records its '
                'scores and thresholds for review, but removes no cells. Barcode rank and mad\'s '
                'absolute-bound entries are unaffected and still filter.'.format(
                    self.sample_name, '; '.join(reasons)))
        elif flagged:
            self.logger.warning(
                '[QC Filter] Sample "{}" would be flagged for manual review: {}. sample_quality '
                'is set to "advisory", so the verdict is recorded but no filter is skipped -- '
                'every threshold below is fit from this sample\'s own distribution '
                'regardless.'.format(self.sample_name, '; '.join(reasons)))

        self._record_qc_metadata(method='sample_quality', metadata={
            'n_cells': n_cells,
            'median_depth': median_depth,
            'min_cells': min_cells,
            'min_median_depth': min_median_depth,
            # Whether the sample met the thresholds, independent of whether the
            # gate acted on it -- the report's banner keys off this, and it must
            # still appear when the gate is advisory.
            'flagged_for_manual_review': flagged,
            'gate_applied': self.low_quality_sample,
            'reasons': reasons,
            'run_context': run_context})

    def _record_post_cascade_viability(self):
        """Count and record what survives once every filter has run."""
        keep = np.ones(self.adata.n_obs, dtype=bool)
        applied = collect_removal_flags(self.adata.uns.get('qc', {}))
        for flag in applied:
            if flag not in self.adata.obs:
                continue
            keep &= ~self.adata.obs[flag].astype(bool).values

        n_surviving = int(keep.sum())
        n_input = int(self.adata.n_obs)
        # One floor, quoted to the human, rather than per-consumer.
        min_viable = int(self.filter_cfg.get('filters', {})
                             .get('sample_quality', {}).get('min_cells_post_cascade', 200))
        viable = n_surviving >= min_viable

        if n_surviving == 0:
            self.logger.warning(
                '[QC Filter] Sample "{}": the QC cascade removed every cell ({}/{}). Nothing '
                'downstream can use this sample -- no embedding, no doublet calls, and the '
                'filtered artifact will be empty. The filters that did the removing are recorded '
                'in .uns["qc"] and shown in the HTML report; check the absolute max_value caps in '
                'filter_config.yaml against this sample\'s fraction features before assuming a '
                'bug.'.format(self.sample_name, n_input, n_input))
        elif not viable:
            self.logger.warning(
                '[QC Filter] Sample "{}": only {} of {} cells survived the full QC cascade, below '
                'the {}-cell viability floor. Doublet detection will skip it and the final UMAP '
                'may be omitted; the filtered artifact and HTML report are still produced. This '
                'is a very small population to carry into integration -- review the report before '
                'using it.'.format(self.sample_name, n_surviving, n_input, min_viable))
        else:
            self.logger.info('[QC Filter] Sample "{}": {} of {} cells survived the full QC '
                             'cascade.'.format(self.sample_name, n_surviving, n_input))

        self._record_qc_metadata(method='post_cascade', metadata={
            'n_cells_input': n_input,
            'n_cells_surviving': n_surviving,
            'min_cells_post_cascade': min_viable,
            'viable': viable,
            'flags_applied': applied})

    # A flagged sample downgrades rather than skips, so the report still has
    # evidence to show.

    def _degenerate_scale(self, x):
        """Report why a MAD threshold cannot be computed, or None if it can.

        Args:
            x (np.ndarray): Feature values.

        Returns:
            tuple or None: (reason (str), share (float)) when the scale estimate is
                unusable, else None.
        """
        x = np.asarray(x, dtype=float)
        finite = x[np.isfinite(x)]
        if finite.size == 0:
            return 'no finite values', 0.0

        mdev = self._mad(finite)
        if not np.isfinite(mdev) or mdev <= 0:
            _, counts = np.unique(finite, return_counts=True)
            tie_fraction = float(counts.max() / finite.size)
            return ('scale estimate is zero: {:.1%} of cells share a single value, '
                    'so median +/- k*MAD cannot separate outliers'.format(tie_fraction),
                    tie_fraction)
        return None

    @property
    def adata_active(self):
        """The filtered view if one is set, else the full object."""
        view = getattr(self, "adata_view", None)
        return self.adata if view is None else view
        
    def _write_obs_column(self, key: str, series: pd.Series):
        """Write one derived column into `.obs`, aligned to the full cell axis.

        Args:
            key (str): Column name.
            series (pandas.Series): Values for the cells this stage saw.
        """
        self.adata.obs[key] = series
        if hasattr(self, "adata_view") and self.adata_view is not None:
            self.adata_view.obs[key] = series.loc[self.adata_view.obs.index]


    @log_exceptions()
    def _transform_features(self, features: list):
        """Apply the configured transform to each feature, caching the result.

        Args:
            features (list[str]): Feature names.
        """
        if not hasattr(self, 'transformed_features'):
            self.transformed_features = {}

        ad = self.adata_active

        def transform(key, x):
            """Apply this feature's configured transform to its values."""
            if key in self.transformed_features:
                self.logger.debug(f'[QC Filter] Feature "{key}" already transformed, skipping.')
                return

            series = ad.obs[key]
            full_index = self.adata.obs.index
            view_index = ad.obs.index

            transformed = pd.Series(np.nan, index=full_index)

            mask_valid = ~series.isna()
            x_valid = series[mask_valid].values

            if np.all((x_valid >= 0) & (x_valid <= 1)):
                self.logger.debug(f'[QC Filter] Transforming fraction metric "{key}" using logit')
                transformed_vals = logit(np.clip(x_valid, LOGIT_CLIP, 1 - LOGIT_CLIP))
                new_key = f'logit_{key}'
            else:
                self.logger.debug(f'[QC Filter] Transforming metric "{key}" using log1p')
                transformed_vals = np.log1p(x_valid)
                new_key = f'log_{key}'

            # Set transformed values only for the filtered cells
            transformed.loc[series[mask_valid].index] = transformed_vals
            self._write_obs_column(new_key, transformed)

            self.transformed_features[key] = new_key
            self.logger.info(
                f'[QC Filter] Transformed "{key}" → "{new_key}" | Valid: {len(x_valid)} | NaNs: {transformed.isna().sum()}'
            )

        self._iterate_features(features, transform)

    def _zscore_features(self, features: list):
        """Z-score features on the active view.

        Args:
            features (list[str]): Feature names.

        Returns:
            np.ndarray: Z-scored matrix.
        """
        ad = self.adata_active
        full_index = self.adata.obs.index

        existing = [f for f in features if f in ad.obs.columns]
        missing = list(set(features) - set(existing))
        if missing:
            self.logger.warning(f"[QC Filter] Skipping missing features: {', '.join(missing)}")

        if not existing:
            self.logger.warning(f"[QC Filter] No valid features to scale")
            return None, None

        df = ad.obs[existing].copy()
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        valid_idx = df.index

        if len(valid_idx) == 0:
            self.logger.warning("[QC Filter] No valid rows to scale after dropping NaNs/Infs.")
            return None, None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)

        scaled_cols = [f'z_{col}' for col in existing]
        for i, col in enumerate(scaled_cols):
            # Start with full-length NaN vector
            result = pd.Series(np.nan, index=full_index)
            result.loc[valid_idx] = X_scaled[:, i]
            self.adata.obs[col] = result

            # Also write to adata_view if active
            if hasattr(self, 'adata_view') and self.adata_view is not None:
                self.adata_view.obs[col] = result.loc[self.adata_view.obs.index]

        self.logger.info(f"[QC Filter] Z-scored features: {', '.join(existing)} | Valid cells: {len(valid_idx)}")

        return scaled_cols, valid_idx


    @log_exceptions()
    def _bcrank_filter(self, features: list, min_total_counts: int = 500, clip_percent=(5, 95), smoothing=True, polyorder=3, flag_name: str = 'flag_bcrank', threshold_override=None):
        """Barcode rank thresholding with clipping and optional smoothing.

        Args:
            features (list[str]): Feature names.
            min_total_counts (int): Lower bound on total counts.
            clip_percent (tuple): Percentile clip applied before the knee search.
            smoothing (bool): Savitzky-Golay smoothing of the rank curve.
            polyorder (int): Smoothing polynomial order.
            flag_name (str): Output flag column.
            threshold_override (float, optional): Use this cut instead of the knee.

        Returns:
            list[str] or None: Flag columns written, or None if skipped.
        """
        ad = self.adata_active
        flags = []

        def compute_rank(key, x):
            """Rank one feature's values and find its knee."""
            counts = np.array(x)
            counts = counts[~np.isnan(counts)]
            counts = counts[counts > 0]  # remove 0-UMI barcodes

            sorted_counts = np.sort(counts)[::-1]
            ranks = np.log10(np.arange(1, len(sorted_counts) + 1))
            log_counts = np.log1p(sorted_counts)
            n = len(log_counts)

            # Smoothing
            if smoothing:
                window_size = min(101, max(31, (n // 100) | 1))
                self.logger.info(f"[QC Filter] Smoothing window size: {window_size}")
                smoothed = savgol_filter(log_counts, window_size, polyorder)
            else:
                smoothed = log_counts

            # Clip range
            start = int(n * clip_percent[0] / 100)
            end = int(n * clip_percent[1] / 100)
            clipped = smoothed[start:end]
            clipped_ranks = ranks[start:end]

            candidate_indices = {}

                #// slope method
            slope = np.gradient(clipped, clipped_ranks)
            slope_change = np.gradient(slope)
            candidate_indices["slope"] = np.argmax(np.abs(slope_change)) + start

                #// Curvature method
            x_n = (clipped_ranks - clipped_ranks.min()) / (clipped_ranks.max() - clipped_ranks.min())
            y_n = (clipped - clipped.min()) / (clipped.max() - clipped.min())
            line_start = np.array([x_n[0], y_n[0]])
            line_end = np.array([x_n[-1], y_n[-1]])
            line_vec = line_end - line_start
            dist = np.abs(np.cross(line_vec, np.column_stack((x_n, y_n)) - line_start) / np.linalg.norm(line_vec))
            candidate_indices["curvature"] = np.argmax(dist) + start

            #// Knee method
            second_deriv = np.gradient(np.gradient(clipped))
            candidate_indices["knee"] = np.argmax(second_deriv) + start

            #// Inflection method
            first_deriv = np.gradient(clipped)
            candidate_indices["inflection"] = np.argmin(first_deriv) + start
            
            
            candidate_thresholds = {m: float(sorted_counts[i]) for m, i in candidate_indices.items()}
            self.logger.info("[QC Filter] Candidate thresholds: " +", ".join(f"{k}: idx={v}, umi={sorted_counts[v]:.1f}" for k,v in candidate_indices.items()))

            # Largest index == lowest UMI cut-off == most lenient candidate. See
            # the docstring: this is a deliberate policy, not a tie-break.
            selected_method = max(candidate_indices, key=candidate_indices.get)
            idx = candidate_indices[selected_method]

            self.logger.info(f"[QC Filter] Selected method: {selected_method} (threshold {sorted_counts[idx]:.1f}, most lenient of {len(candidate_indices)})")

            if threshold_override is not None:
                # Manual override from filter_config.yaml; bypasses the knee search.
                threshold = float(threshold_override)
                selected_method = 'manual'
                self.logger.info(
                    "[QC Filter] Using hand-set barcode rank threshold {} on '{}' "
                    "(candidate selection and the dynamic clamp are both bypassed; the "
                    "automatic policy would have chosen {:.1f})".format(
                        threshold, key, sorted_counts[idx]))
            else:
                threshold = sorted_counts[idx]
                umi_median = np.median(sorted_counts)
                umi_p95 = np.percentile(sorted_counts, 95)
                min_dynamic = max(min_total_counts, np.percentile(sorted_counts, 2))
                max_dynamic = umi_median * 10 if umi_median > 0 else umi_p95 * 2

                if threshold < min_dynamic or threshold > max_dynamic:
                    self.logger.info(f"[QC Filter] Threshold {threshold} is either < {min_dynamic} or > {max_dynamic} Forcing threshold = {np.clip(threshold, min_dynamic, max_dynamic)}")
                    threshold = np.clip(threshold, min_dynamic, max_dynamic)
            flag_col = f"{flag_name}_{key}"
            rank_col = f"rank_{flag_name}_{key}"

            # Create full-length flag vector
            full_flags = pd.Series(False, index=self.adata.obs.index)
            full_flags.loc[ad.obs.index] = ad.obs[key] < threshold
            self._write_obs_column(flag_col, full_flags)

            # Compute full rank vector
            full_ranks = pd.Series(np.nan, index=self.adata.obs.index)
            sorted_obs_idx = self.adata.obs[key].sort_values(ascending=False).index
            full_ranks.loc[sorted_obs_idx[:len(ranks)]] = ranks
            self._write_obs_column(rank_col, full_ranks)

            retained = np.sum(ad.obs[key] >= threshold)
            self.logger.info(f"[QC Filter] Barcode rank ({selected_method}) on '{key}': threshold = {threshold}, retained = {retained} cells.")
            self._check_flag_fraction('Barcode rank', key, (ad.obs[key] < threshold).values)

            run_context = self._get_run_context()
            self._record_qc_metadata(method = 'bcrank', feature = key, metadata = {
                'flag_col': flag_col,
                'feature': key,
                'threshold': float(threshold),
                # All four candidates, so the leniency policy is auditable per sample
                # rather than only visible as the one that won.
                'candidate_thresholds': candidate_thresholds,
                'selection_policy': 'manual' if threshold_override is not None else 'most_lenient',
                'clipped_range': [start, end],
                'smoothed': smoothing,
                'smoothed_vals': smoothed.tolist(),
                'ranks': ranks.tolist(),
                'n_flagged': int(np.sum(ad.obs[key] < threshold)),
                'run_context': run_context,
                'method': selected_method
            })
            flags.append(flag_col)
        self._iterate_features(features, compute_rank)
        return flags
    

    @log_exceptions()
    def _mad_filter(self, features: list, k: int=3, direction: str='both', transform: bool=True,
                    max_value: float=None, min_value: float=None, record_method: str='mad'):
        """Flag cells outside median +/- k * MAD, or outside explicit bounds.

        `k` is in raw MAD units, not sigma. See docs/decisions/qc-cascade.md.

        Args:
            features (list[str]): Feature names.
            k (float): Multiplier on the MAD.
            direction (str): 'lower', 'upper' or 'both'.
            transform (bool): Fit on the transformed feature.
            max_value (float, optional): Absolute upper cap.
            min_value (float, optional): Absolute lower cap.
            record_method (str): Name recorded in `.uns['qc']`.

        Returns:
            list[str] or None: Flag columns written, or None if skipped.
        """
        if direction not in ('lower', 'upper', 'both'):
            raise ValueError(
                "[QC Filter] `direction` must be 'lower', 'upper' or 'both', got {!r}. "
                "An unrecognised value used to fall through to 'both' silently.".format(direction))

        absolute = max_value is not None or min_value is not None
        if transform and not absolute:
            self._transform_features(features)
            # `.get`, not `[...]`: _iterate_features SKIPS a feature that is
            # all-NaN (it logs "is all NaNs, skipping" and continues), so its
            # transform is never recorded. Indexing unconditionally turned that
            # deliberate skip into `KeyError: '<feature>'` -- a message that reads
            # like a missing .obs column when the column is present and merely
            # unusable. A skipped feature is simply not filtered on.
            features = [self.transformed_features[feature] for feature in features
                        if feature in self.transformed_features]
        flag_cols = []

        if transform and not absolute and not features:
            self.logger.warning(
                '[QC Filter] mad: every requested feature was skipped before '
                'transform (all-NaN or absent); no flags produced.')
            return flag_cols

        def flag_outliers(key, x):
            """Flag cells outside median +/- k*MAD, declining on a degenerate scale."""
            run_context = self._get_run_context()

            if absolute:
                lower, upper = min_value, max_value
                mode = 'absolute'
            else:
                degenerate = self._degenerate_scale(x)
                if degenerate is not None:
                    reason, tie_fraction = degenerate
                    self.logger.warning(
                        '[QC Filter] Skipping MAD filter on "{}": {}. No cells will be '
                        'flagged on this feature. Set max_value/min_value in '
                        'filter_config.yaml to threshold it explicitly.'.format(key, reason))
                    # Recorded, not just logged: a skip that lives only in a log line
                    # is indistinguishable from a filter that silently did nothing.
                    # No flag_col -- a skipped feature must not contribute to the mask.
                    self._record_qc_metadata(method=record_method, feature=key, metadata={
                        'feature': key,
                        'skipped': True,
                        'skip_reason': reason,
                        'tie_fraction': tie_fraction,
                        'k': k,
                        'direction': direction,
                        'transform': transform,
                        'n_flagged': 0,
                        'run_context': run_context})
                    return

                med = np.nanmedian(x)
                mdev = self._mad(x)
                lower = med - k * mdev if direction in ('lower', 'both') else None
                upper = med + k * mdev if direction in ('upper', 'both') else None
                mode = 'mad'

            flags = np.zeros(np.shape(x), dtype=bool)
            if lower is not None:
                flags |= x < lower
            if upper is not None:
                flags |= x > upper

            flag_key = f"flag_{record_method}_{key}"
            full_flags = pd.Series(False, index=self.adata.obs.index)
            full_flags.loc[self.adata_active.obs.index] = flags

            self._write_obs_column(flag_key, full_flags)
            flag_cols.append(flag_key)

            self.logger.info('[QC Filter] {}: flagged {} cells ({} thresholding)'.format(
                key, int(np.sum(flags)), mode))
            self._check_flag_fraction('MAD filter', key, flags)

            self._record_qc_metadata(method = record_method, feature = key, metadata = {
                "flag_col": flag_key,
                "feature": key,
                "k": k,
                "direction": direction,
                "transform": transform and not absolute,
                "mode": mode,
                "thresholds": {
                    # `if lower` would drop a genuine threshold of exactly 0.0.
                    "lower": float(lower) if lower is not None else None,
                    "upper": float(upper) if upper is not None else None
                },
                "n_flagged": int(np.sum(flags)),
                'run_context': run_context
            })
        self._iterate_features(features, flag_outliers)
        return flag_cols

    @log_exceptions()
    def _mad(self, x):
        """Median absolute deviation, unscaled.

        Args:
            x (np.ndarray): Values.

        Returns:
            float: MAD.
        """
        med = np.nanmedian(x)
        return np.nanmedian(np.abs(x - med))

    def _compute_mahalanobis(self, x, mean, cov, cov_type: str='diag'):
        """Mahalanobis distance from each row of `x` to `mean` under `cov`.

        Args:
            x (np.ndarray): Rows to score.
            mean (np.ndarray): Distribution mean.
            cov (np.ndarray): Covariance.
            cov_type (str): Covariance structure.

        Returns:
            np.ndarray: Distances.
        """
        x_centered = np.asarray(x) - np.asarray(mean)

        if cov_type == 'diag':
            return np.sqrt(np.sum((x_centered ** 2) / cov, axis=1))

        if cov_type == 'full':
            # pinv, not inv: a component fitted to a spike of identical values has
            # a singular covariance, and inv would raise or return garbage.
            VI = np.linalg.pinv(np.asarray(cov))
            sq = np.einsum('ij,jk,ik->i', x_centered, VI, x_centered)
            # Clip tiny negatives from floating-point error before the sqrt.
            return np.sqrt(np.maximum(sq, 0.0))

        raise ValueError(
            "[QC Filter] _compute_mahalanobis does not support covariance_type "
            "'{}'; only 'diag' and 'full' are selected by _select_gmm_by_bic.".format(cov_type))


    @log_exceptions()
    def _regression_filter(self, x_key, y_key, k: int=3, transform: bool=True, direction: str='both', flag_name: str='flag_regression'):
        """Fit y ~ x and flag cells with residuals beyond median +/- k * MAD.

        Args:
            x_key (str): Predictor column.
            y_key (str): Response column.
            k (float): Multiplier on the residual MAD.
            transform (bool): Fit on transformed features.
            direction (str): 'lower', 'upper' or 'both'.
            flag_name (str): Output flag column.

        Returns:
            list[str] or None: Flag columns written, or None if skipped.
        """
        ad = self.adata_active

        if transform:
            self._transform_features([x_key, y_key])
            x_key = self.transformed_features.get(x_key)
            y_key = self.transformed_features.get(y_key)

        df = ad.obs[[x_key, y_key]].copy()
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        if df.shape[0] < 10:
            self.logger.info('[QC filter] Too few valid cells for regression filtering')
            return None

        X = df[[x_key]].values
        y = df[y_key].values

        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred

        mad = self._mad(residuals)
        med = np.nanmedian(residuals)

        degenerate = self._degenerate_scale(residuals)
        if degenerate is not None:
            reason, tie_fraction = degenerate
            self.logger.warning(
                '[QC filter] Skipping regression filter on "{}" ~ "{}": residual {}. '
                'No cells flagged.'.format(y_key, x_key, reason))
            self._record_qc_metadata(method='regression', metadata={
                'features': [x_key, y_key],
                'skipped': True,
                'skip_reason': 'residual {}'.format(reason),
                'tie_fraction': tie_fraction,
                'k': k,
                'direction': direction,
                'transform': transform,
                'n_flagged': 0,
                'run_context': self._get_run_context()})
            return None

        lower_thresh = med - k * mad
        upper_thresh = med + k * mad

        if direction == 'lower':
            flags = residuals < lower_thresh
        elif direction == 'upper':
            flags = residuals > upper_thresh
        elif direction == 'both':
            flags = (residuals < lower_thresh) | (residuals > upper_thresh)
        else:
            raise ValueError("`direction` must be one of 'lower', 'upper', or 'both'")
    
        flag_series = pd.Series(False, index=self.adata.obs.index)
        flag_series.loc[df.index] = flags

        self._write_obs_column(flag_name, flag_series)
        self._check_flag_fraction('Regression filter', '{} ~ {}'.format(y_key, x_key), flags)

        run_context = self._get_run_context()
        self._record_qc_metadata(method = 'regression', metadata = {
            "flag_col": flag_name,
            "features": [x_key, y_key],
            "k": k,
            "direction": direction,
            "transform": transform,
            "thresholds": {
                "lower": float(lower_thresh) if direction != "upper" else None,
                "upper": float(upper_thresh) if direction != "lower" else None
                },
            "n_flagged": int(np.sum(flags)),
            'coef': float(model.coef_[0]), 
            'intercept': float(model.intercept_),
            'run_context': run_context
            })

        self.logger.info('[QC filter] Flagged {} cells based gene-count regression filter using features: "{}" and "{}"'.format(np.sum(flags), x_key, y_key))
        return [flag_name]
        

    @log_exceptions()
    def _gmm_filter(self, features, max_components=3, cutoff_quantile=0.99, scope_global=False, flag_name: str='flag_gmm', transform: bool=True):
        """Flag cells by Mahalanobis distance to a BIC-selected mixture.

        Args:
            features (list[str]): Feature names.
            max_components (int): Upper bound on components.
            cutoff_quantile (float): Distance quantile to cut at.
            scope_global (bool): Fit on all cells rather than the active view.
            flag_name (str): Output flag column.
            transform (bool): Fit on transformed features.

        Returns:
            list[str] or None: Flag columns written, or None if skipped.
        """
        ad = self.adata_active
        if transform:
            self._transform_features(features)
            features  = [self.transformed_features[feature] for feature in features if feature in self.transformed_features]

        # Check normality of each feature
        non_gaussian = self._is_approximately_normal(features)
        if non_gaussian:
           self.logger.warning('[QC filter] the following features are non-gaussing, checking for potential multimodality: "{}"'.format(', '.join(non_gaussian)))
           multimodal_features = self._detect_multimodal_features(non_gaussian)
           drop = set(non_gaussian) - set(multimodal_features)
           # List comprehension, not list(set(...)): set iteration order over
           # strings varies per process, so the column order of X -- and hence the
           # covariances and every Mahalanobis distance -- changed between runs.
           # random_state cannot make a fit reproducible if its input does not
           # have a stable column order.
           features = [f for f in features if f not in drop]

           if not features:
                self.logger.warning('[QC filter] None of provided features are approximately guassian, skipping GMM')
                return

        def run_gmm():
            """Fit the mixture and flag cells by Mahalanobis distance."""
            df = ad.obs[features].copy()
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            valid_cells = df.index
            X = df.values

            n_components, cov_type, bics = self._select_gmm_by_bic(df, max_components=max_components)

            if len(X) < n_components * 5:
                self.logger.warning(f"[QC Filter] Too few cells ({len(X)}) to fit GMM with {n_components} components. Skipping.")
                return None

            gmm = GaussianMixture(n_components=n_components, covariance_type=cov_type, random_state=42)
            gmm.fit(X)
            preds = gmm.predict(X)

            mahal_dist = np.zeros(X.shape[0])
            for i in range(gmm.n_components):
                idx = preds == i
                if not idx.any():
                    continue
                mean = gmm.means_[i]
                cov = gmm.covariances_[i]
                mahal_dist[idx] = self._compute_mahalanobis(X[idx], mean, cov, cov_type=cov_type)

            chi2_thresh = chi2.ppf(cutoff_quantile, df=X.shape[1])
            outliers = mahal_dist > chi2_thresh

            # Prepare full-length vectors
            full_index = self.adata.obs.index

            flag_series = pd.Series(False, index=full_index)
            flag_series.loc[valid_cells] = outliers
            self._write_obs_column(flag_name, flag_series)

            mahal_series = pd.Series(np.nan, index=full_index)
            mahal_series.loc[valid_cells] = mahal_dist
            self._write_obs_column("gmm_mahal", mahal_series)

            cluster_series = pd.Series(np.nan, index=full_index)
            cluster_series.loc[valid_cells] = preds
            self._write_obs_column("gmm_cluster", cluster_series)

            self.logger.info(f"[QC Filter] Flagged {np.sum(outliers)} cells using GMM filter with features: {', '.join(features)}")
            self._check_flag_fraction('GMM filter', ', '.join(features), outliers)

            # Store diagnostics for reproducibility
            run_context = self._get_run_context()
            self._record_qc_metadata(method = 'gmm', metadata = {
                "flag_col": flag_name,
                'cluster_col': 'gmm_cluster',
                'distance_col': 'gmm_mahal',
                "features": features,
                "transform": transform,
                "thresholds": float(chi2_thresh),
                'cutoff_quantile': cutoff_quantile,
                # The fit itself, so a run can be reproduced/audited rather than
                # only its verdict.
                'n_components': int(n_components),
                'covariance_type': cov_type,
                'bics': [float(b) for b in bics],
                "n_flagged": int(np.sum(outliers)),
                'run_context': run_context
                })
        if scope_global:
            self.logger.warning('[QC Filter] global scope not yet implemented')
        else:
            run_gmm()
        return [flag_name]

    @log_exceptions()
    def _is_approximately_normal(self, features: list, alpha=0.01, skew_thresh=1, kurt_thresh=1):
        """Whether each feature is close enough to normal for a Gaussian-based rule.

        Args:
            features (list[str]): `.obs` columns to test.
            alpha (float): Significance level for the normality test.
            skew_thresh (float): Absolute skew above which normality is rejected.
            kurt_thresh (float): Absolute excess kurtosis above which it is rejected.

        Returns:
            dict: {feature: bool}.
        """
        failed = []
        def normality(key, x):
            """Test one feature and record its statistics."""
            x = self.adata.obs[key].dropna()
            if len(x) < 20:
                self.logger.warning('[QC filter] Feature "{}" has fewer then 20 observations, cannot assume normality'.format(key))
                failed.append(key)
                return

            s = skew(x)
            skew_ok = abs(s) < skew_thresh

            k = kurtosis(x)
            kurt_ok = abs(k) < kurt_thresh

            if len(x) < 5000:
                stat, p = shapiro(x)
                shapiro_ok = p > alpha
            else:
                self.logger.info('[QC Filter] shapiro test too sensitive for large datasets, detected {} cells'.format(len(x)))
                shapiro_ok = True

            if not (skew_ok and kurt_ok and shapiro_ok):
                self.logger.warning('[QC filter] Feature "{}" is not approximatelly normally distributed'.format(key))
                failed.append(key)
 
        self._iterate_features(features, normality)
        return failed

    @log_exceptions()
    def _detect_multimodal_features(self, features: list, min_peaks: int = 2, prominence: float = 0.05) -> list:
        """Detect likely-multimodal features by KDE peak detection.

        Args:
            features (list[str]): Feature names.
            min_peaks (int): Peaks required to call multimodality.
            prominence (float): Minimum peak prominence.

        Returns:
            list[str]: Features judged multimodal.
        """
        multimodal = []

        def check_multimodal(feature, x):
            """Test one feature for more than one mode."""
            x = self.adata.obs[feature].dropna().values
            if len(x) < 50:
                self.logger.warning(f'[QC Filter] Not enough data to assess modality for {feature}')
                return

            try:
                kde = gaussian_kde(x)
                x_grid = np.linspace(np.min(x), np.max(x), 500)
                kde_vals = kde(x_grid)

                peaks, _ = find_peaks(kde_vals, prominence=prominence)

                if len(peaks) >= min_peaks:
                    self.logger.info(f'[QC Filter] Feature "{feature}" appears multimodal ({len(peaks)} peaks detected).')
                    multimodal.append(feature)
            except Exception as e:
                self.logger.warning(f'[QC Filter] Failed to assess modality for {feature}: {str(e)}')

        self._iterate_features(features, check_multimodal)
        return multimodal

    @log_exceptions()
    def _select_gmm_by_bic(self, X, max_components: int=3, random_state: int=42, min_k_cells: int=15):
        """Choose the number of GMM components by BIC.

        Args:
            X (np.ndarray): Observations.
            max_components (int): Upper bound.
            random_state (int): Seed.
            min_k_cells (int): Cells required per component.

        Returns:
            tuple: (best_k (int), cov_type (str), bics (dict)).
        """
        bics = []
        lowest_bic = np.inf
        best_k = 1

        n_cells, n_features = X.shape

        # Bound by the caller's ceiling as well as by dataset size. Without the
        # caller's bound this reached min(n/15, sqrt(n)) -- 31 components for 1000
        # cells -- and fitted a GMM for every one of them.
        max_components = max(1, min(max_components, n_cells // min_k_cells, int(np.sqrt(n_cells))))
        self.logger.info("[GMM] Starting BIC-based component selection up to {} components".format(max_components))

        # A full covariance has p(p+1)/2 free parameters per component; require
        # roughly 10 samples per parameter before trusting one, else stay diagonal.
        # (This previously compared n_features against a figure derived from
        # n_cells -- always false, so 'full' was unreachable.)
        params_per_component = n_features * (n_features + 1) / 2
        required_full = 10 * params_per_component * max_components
        cov_type = 'full' if n_cells >= required_full else 'diag'
        self.logger.info("[GMM] Setting covariance type to '{}' ({} cells vs {:.0f} required for full)".format(
            cov_type, n_cells, required_full))

        for k in range(1, max_components + 1):
            try:
                gmm = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=random_state)
                gmm.fit(X)
                bic = gmm.bic(X)
                bics.append(bic)

                self.logger.debug(f"[GMM] BIC for {k} components: {bic:.2f}")

                if bic < lowest_bic:
                    lowest_bic = bic
                    best_k = k

            except Exception as e:
                self.logger.warning(f"[GMM] GMM fitting failed for {k} components: {e}")
                bics.append(np.inf)

        self.logger.info(f"[QC Filter] Optimal number of components by BIC: {best_k}")
        return best_k, cov_type, bics


    @log_exceptions()
    def _miqc_filter(self, feature: str, complexity_key: str, posterior_threshold: float=0.75,
                     min_component_separation: float=0.02, min_cells_per_component: int=20,
                     flag_name: str='flag_miqc'):
        """
        miQC-style joint outlier detection (Hippen et al. 2021,
        doi:10.1371/journal.pcbi.1009290): flag cells whose fraction metric rises
        as library complexity falls -- the signature of a compromised/lysed cell
        leaking cytoplasmic RNA while mitochondria-, plastid- or rRNA-enclosed RNA
        is retained.

        Reimplemented on a 2-component full-covariance GaussianMixture over
        (complexity, fraction) rather than the original's mixture of linear
        regressions (`flexmix` in R, `flexmix(subsets_mito_percent ~ detected,
        k=2)` in the package's own `mixtureModel.R`): a full-covariance bivariate
        Gaussian's conditional mean of fraction given complexity *is* a linear
        regression line, so this is the same model under a Gaussian-residual
        assumption, built on the same GMM machinery `_gmm_filter` already uses
        rather than a bespoke EM implementation. Not routed through `_gmm_filter`
        itself: that method thresholds on Mahalanobis distance to a BIC-selected
        number of components, which answers a different question (multivariate
        distance from any cluster centre) than the one here (posterior
        probability of membership in the specific higher-fraction component,
        given a fixed 2-component model).

        **Do not add a transform to either axis.** Both `feature` and
        `complexity_key` are used on their raw scale -- confirmed against the
        package source (`mixtureModel.R`), where
        `flexmix(subsets_mito_percent ~ detected, ...)` uses both variables
        completely untransformed. A logit on the fraction fabricates a
        "compromised" component out of near-zero noise; it was tried, it
        misfired on a real sample, and it was removed. The account is in
        `docs/decisions/qc-miqc-transform.md`; what stops it coming back is
        `tests/test_qc_miqc.py::test_raw_scale_skips_on_pure_noise`.

        Both axes are standardized (zero mean, unit variance) *inside* the fit,
        and every fitted quantity is converted back to raw units immediately
        after. With `covariance_type='full'` and no rescaling, the axis with the
        larger absolute variance dominates which axis the fit actually splits on
        -- complexity spans hundreds-to-thousands of genes while fraction's
        natural spread can be a few thousandths, and standardizing complexity
        alone is not enough because unit variance still dwarfs fraction's.
        `min_component_separation` / `posterior_threshold` / the intact-line
        comparison all keep their raw-proportion-unit meaning; only the fit
        itself operates in standardized space. This is uniform across whichever
        feature the function is called on.

        A cell is flagged only when both hold:
          - its posterior probability of belonging to the higher-fraction
            ("compromised") component exceeds `posterior_threshold`;
          - its fraction exceeds the *intact* component's predicted fraction at
            its complexity (the conditional-mean line).
        The second condition is the paper's own protection against an uncertain
        fit flagging cells whose fraction is already explained by the intact
        trend -- without it, a cell can have high posterior probability under a
        two-component fit while still being unremarkable in absolute terms.

        Call once per fraction feature (mitochondrial/chloroplast/ribosomal_rna),
        same shape as `_mad_filter` iterating one feature at a time -- the model
        only ever contrasts one fraction against complexity, not all three at
        once.

        Parameters
        ----------
        feature : str
            .obs column for the fraction metric (e.g. mitochondrial_fraction_*).
        complexity_key : str
            .obs column for library complexity (e.g. n_genes, corrected_counts_n_genes).
        posterior_threshold : float
            Minimum posterior probability of "compromised" membership to flag.
        min_component_separation : float
            Minimum gap between the two components' mean fraction, in raw
            proportion units (0.02 = 2 percentage points), to trust the split
            as a real compromised/intact distinction rather than an arbitrary
            partition of a single population.
        min_cells_per_component : int
            Lower bound on cells per component before fitting; below this the
            fit is unstable regardless of what it reports.

        Returns
        -------
        list or None
            [flag_col] if the filter ran, None if it was skipped.
        """
        ad = self.adata_active
        run_context = self._get_run_context()

        # Same trap _mad_filter guards against: if most cells share one fraction
        # value (nuclei preps at ~0% mitochondrial), there is no spread for a
        # second component to fit, and the "compromised" line degenerates into
        # noise.
        degenerate = self._degenerate_scale(ad.obs[feature].values)
        if degenerate is not None:
            reason, tie_fraction = degenerate
            self.logger.warning(
                '[QC Filter] Skipping miQC filter on "{}": {}. No cells will be '
                'flagged on this feature.'.format(feature, reason))
            self._record_qc_metadata(method='miqc', feature=feature, metadata={
                'feature': feature, 'complexity_key': complexity_key, 'skipped': True,
                'skip_reason': reason, 'tie_fraction': tie_fraction, 'n_flagged': 0,
                'run_context': run_context})
            return None

        df = ad.obs[[complexity_key, feature]].replace([np.inf, -np.inf], np.nan).dropna()
        valid_idx = df.index
        n_components = 2

        if len(df) < n_components * min_cells_per_component:
            self.logger.warning(
                '[QC Filter] Too few cells ({}) to fit miQC model on "{}", skipping.'.format(
                    len(df), feature))
            self._record_qc_metadata(method='miqc', feature=feature, metadata={
                'feature': feature, 'complexity_key': complexity_key, 'skipped': True,
                'skip_reason': 'too few cells ({}) for {} components'.format(len(df), n_components),
                'n_flagged': 0, 'run_context': run_context})
            return None

        X = df[[complexity_key, feature]].values

        # Standardize BOTH axes for the fit -- see the docstring. Complexity
        # spans hundreds-to-thousands of genes; fraction spans roughly 0-0.05
        # with a natural spread that can be orders of magnitude smaller still
        # (a low-incidence feature's real signal might have a spread of just a
        # few thousandths). Standardizing complexity alone to unit variance
        # does NOT fix this: unit variance is still enormously larger than
        # fraction's own natural variance, so complexity would still dominate
        # covariance_type='full' clustering. Both axes need to be brought to
        # comparable scale, not just complexity brought to an arbitrary one.
        # gmm.means_/covariances_ are unscaled back to raw units immediately
        # after fitting so every downstream computation (separation check,
        # intact_line, recorded metadata) uses the exact same formulas and
        # raw-unit meaning as before this change.
        col_mean = X.mean(axis=0)
        col_std = X.std(axis=0)
        if col_std[0] <= 0:
            reason = 'complexity has no spread (all cells share one value)'
            self.logger.warning('[QC Filter] Skipping miQC filter on "{}": {}.'.format(feature, reason))
            self._record_qc_metadata(method='miqc', feature=feature, metadata={
                'feature': feature, 'complexity_key': complexity_key, 'skipped': True,
                'skip_reason': reason, 'n_flagged': 0, 'run_context': run_context})
            return None
        # col_std[1] (the feature) can't be <=0 here: _degenerate_scale above
        # already gates on exactly this condition for the raw feature values.
        X_fit = (X - col_mean) / col_std

        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(X_fit)
        posterior = gmm.predict_proba(X_fit)

        # Unscale fitted means/covariances back to raw (complexity, fraction)
        # units: Cov(D @ x) = D Cov(x) D for a diagonal scaling D, so the
        # inverse is Cov(x) = diag(std) @ Cov(D@x) @ diag(std), i.e. elementwise
        # multiplication by outer(std, std).
        means_raw = gmm.means_ * col_std + col_mean
        covariances_raw = gmm.covariances_ * np.outer(col_std, col_std)

        # "Compromised" = whichever component sits higher on the fraction axis;
        # nothing else in a 2-component fit identifies which is which.
        fraction_means = means_raw[:, 1]
        compromised_idx = int(np.argmax(fraction_means))
        intact_idx = 1 - compromised_idx
        separation = float(fraction_means[compromised_idx] - fraction_means[intact_idx])

        if separation < min_component_separation:
            self.logger.warning(
                '[QC Filter] Skipping miQC filter on "{}": components are not well separated '
                '(mean fraction gap {:.3f} < {}). No clear compromised population found.'.format(
                    feature, separation, min_component_separation))
            self._record_qc_metadata(method='miqc', feature=feature, metadata={
                'feature': feature, 'complexity_key': complexity_key, 'skipped': True,
                'skip_reason': 'component separation {:.3f} below min_component_separation={}'.format(
                    separation, min_component_separation),
                'n_flagged': 0, 'run_context': run_context})
            return None

        # Conditional mean of a bivariate normal is linear in the conditioning
        # variable, so the intact component's fitted covariance already contains
        # the "boundary line" miQC fits directly -- read off here instead of
        # fitting a second regression. means_raw/covariances_raw are already
        # back in raw units, so this is the same formula as before scaling
        # was introduced; X[:, 0] is the original raw complexity array.
        mean_intact = means_raw[intact_idx]
        cov_intact = covariances_raw[intact_idx]
        var_complexity = cov_intact[0, 0]
        slope = float(cov_intact[1, 0] / var_complexity) if var_complexity > 1e-12 else 0.0
        intercept = float(mean_intact[1] - slope * mean_intact[0])
        intact_line = intercept + slope * X[:, 0]

        posterior_compromised = posterior[:, compromised_idx]
        flags = (posterior_compromised > posterior_threshold) & (X[:, 1] > intact_line)

        flag_col = '{}_{}'.format(flag_name, feature)
        flag_series = pd.Series(False, index=self.adata.obs.index)
        flag_series.loc[valid_idx] = flags
        self._write_obs_column(flag_col, flag_series)

        posterior_col = 'miqc_posterior_{}'.format(feature)
        posterior_series = pd.Series(np.nan, index=self.adata.obs.index)
        posterior_series.loc[valid_idx] = posterior_compromised
        self._write_obs_column(posterior_col, posterior_series)

        self.logger.info(
            '[QC Filter] miQC flagged {} of {} cells on "{}" (posterior > {}, above intact trend)'.format(
                int(np.sum(flags)), len(valid_idx), feature, posterior_threshold))
        self._check_flag_fraction('miQC filter', feature, flags)

        self._record_qc_metadata(method='miqc', feature=feature, metadata={
            'flag_col': flag_col,
            'posterior_col': posterior_col,
            'feature': feature,
            'complexity_key': complexity_key,
            'posterior_threshold': posterior_threshold,
            'compromised_component': compromised_idx,
            'intact_component': intact_idx,
            'component_means': {
                'complexity': means_raw[:, 0].tolist(),
                'fraction': fraction_means.tolist()},
            'component_separation': separation,
            'intact_line': {'slope': slope, 'intercept': intercept},
            'n_flagged': int(np.sum(flags)),
            'run_context': run_context})

        return [flag_col]

    @log_exceptions()
    def _iforest_filter(self, features: list, flag_name: str='flag_iforest', n_estimators: int=100, contamination=0.05, random_state: int=42):
        """Isolation-forest outlier flagging on z-scored features.

        Args:
            features (list[str]): Feature names.
            flag_name (str): Output flag column.
            n_estimators (int): Number of trees.
            contamination (float): Expected outlier fraction.
            random_state (int): Seed.

        Returns:
            list[str] or None: Flag columns written, or None if skipped.
        """
        ad = self.adata_active

        features, valid_idx = self._zscore_features(features)
        if not features or valid_idx is None:
            self.logger.warning('[QC filter] No valid features/cells for isolation forest, skipping')
            return None

        X_scaled = ad.obs.loc[valid_idx, features].values

        if X_scaled.shape[0] < 10:
            self.logger.warning('[QC filter] Too few valid cells to run isolation forest'.format())
            return None

        clf = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state, bootstrap = True)
        preds = clf.fit_predict(X_scaled)

        scores = clf.decision_function(X_scaled)
        flags = preds == -1

        # Index by valid_idx, not ad.obs.index: _zscore_features drops NaN/inf rows,
        # so these arrays are as long as the *surviving* cells. Assigning them to
        # the full view is a length mismatch, and the fraction metrics carry NaN by
        # design (see _layer_fraction's mask_with_zero), so this always fired.
        flag_series = pd.Series(False, index=self.adata.obs.index)
        flag_series.loc[valid_idx] = flags

        self._write_obs_column(flag_name, flag_series)
        score_series = pd.Series(np.nan, index=self.adata.obs.index)
        score_series.loc[valid_idx] = scores
        self._write_obs_column( f"{flag_name}_scores", score_series)

        self.logger.info('[QC filter] Flagged {} of {} cells using Isolation Forest (contamination={}) on features {}'.format(
            int(np.sum(flags)), len(valid_idx), contamination, ', '.join(features)))
        self._check_flag_fraction('Isolation forest', ', '.join(features), flags)

        run_context = self._get_run_context()
        self._record_qc_metadata(method = 'iforest', metadata = {
            "flag_col": flag_name,
            "features": features,
            # offset_ is what predict() actually compares against; the previous
            # np.quantile(scores, contamination) only approximated it and was
            # meaningless once contamination='auto'.
            "threshold": float(clf.offset_),
            "contamination": contamination,
            "n_estimators": n_estimators,
            'score_column': f"{flag_name}_scores",
            "n_flagged": int(np.sum(flags)),
            'run_context': run_context})
        return [flag_name]

    @log_exceptions()
    def _dod_filter(self, features: list, method='overlap', flag_name='flag_dod',
                    k: int=15, overlap_threshold: float=0.65, contamination=0.05,
                    max_cells: int=MAX_DOD_CELLS):
        """Flag cells whose neighbourhood is rearranged by the DoD transform.

        See docs/decisions/qc-cascade.md.

        Args:
            features (list[str]): Feature names.
            method (str): 'overlap' or 'change'.
            flag_name (str): Output flag column.
            k (int): Neighbours compared.
            overlap_threshold (float): Minimum neighbour overlap to pass.
            contamination (float): Expected outlier fraction.
            max_cells (int): Subsample above this many cells.

        Returns:
            list[str] or None: Flag columns written, or None if skipped.
        """
        run_context = self._get_run_context()

        metrics, skip_reason = self._dod_metrics(features, k=k, max_cells=max_cells)
        if metrics is None:
            self.logger.warning("[DoD] Skipping filtering: {}.".format(skip_reason))
            self._record_qc_metadata(method='dod', metadata={
                'features': features, 'skipped': True, 'skip_reason': skip_reason,
                'k': k, 'n_flagged': 0, 'run_context': run_context})
            return None

        df = pd.DataFrame({
            "dod_change": metrics["change"],
            "neighborhood_overlap": metrics["overlap"]
        }).dropna()

        if df.empty:
            reason = 'no cells had both a change score and an overlap score to evaluate'
            self.logger.warning("[DoD] Skipping filtering: {}.".format(reason))
            self._record_qc_metadata(method='dod', metadata={
                'features': metrics["features"], 'skipped': True, 'skip_reason': reason,
                'k': metrics["k"], 'n_flagged': 0, 'run_context': run_context})
            return None

        if method == "overlap":
            # The paper's criterion: noise points retain little of their original
            # neighbourhood once distances are transformed.
            flags = (df["neighborhood_overlap"] < overlap_threshold).values
        elif method == "quantile":
            high_change = df["dod_change"] > df["dod_change"].quantile(0.95)
            low_overlap = df["neighborhood_overlap"] < df["neighborhood_overlap"].quantile(0.05)
            flags = (high_change & low_overlap).values
        elif method == "iforest":
            clf = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
            flags = clf.fit_predict(df.values) == -1
        else:
            raise ValueError(
                "[DoD] Unknown outlier method: {!r}; expected 'overlap', 'quantile' or 'iforest'".format(method))

        flag_series = pd.Series(False, index=self.adata.obs.index)
        flag_series.loc[df.index] = flags
        self._write_obs_column(flag_name, flag_series)

        # _write_obs_column already aligns a partially-indexed series onto the full
        # frame; the previous .loc assignment beforehand was a redundant duplicate.
        self._write_obs_column("dod_change", metrics["change"])
        self._write_obs_column("dod_overlap", metrics["overlap"])

        self.logger.info("[QC filter] DoD flagged {} of {} cells using method '{}'".format(
            int(np.sum(flags)), len(df), method))
        self._check_flag_fraction('DoD filter', ', '.join(metrics["features"]), flags)

        self._record_qc_metadata(method = 'dod', metadata = {
            "flag_col": flag_name,
            "features": metrics["features"],
            'change_column': 'dod_change',
            'overlap_column': 'dod_overlap',
            'selection_method': method,
            'k': metrics["k"],
            'max_cells': max_cells,
            'overlap_threshold': overlap_threshold if method == 'overlap' else None,
            "n_flagged": int(np.sum(flags)),
            'run_context': run_context})
        # The n x n d/dod matrices are deliberately not recorded: they went into
        # .uns and therefore into the h5ad, ~400 MB per 5k-cell sample.
        return [flag_name]

    def _dod_metrics(self, features: list, k: int=15, max_cells: int=MAX_DOD_CELLS):
        """Compute DoD metrics on the z-scored features of the active view.

        Args:
            features (list[str]): Feature names.
            k (int): Neighbours compared.
            max_cells (int): Subsample above this many cells.

        Returns:
            tuple: (overlap (np.ndarray), change (np.ndarray), index (np.ndarray)).
        """
        ad = self.adata_active

        scaled_features, valid_idx = self._zscore_features(features)
        if not scaled_features or valid_idx is None:
            reason = 'no valid features or cells after z-scoring'
            self.logger.warning("[DoD] {}.".format(reason))
            return None, reason

        n = len(valid_idx)
        if n <= k:
            reason = ('only {} cells survived upstream filtering, at or below the K={} '
                      'neighbourhood DoD requires'.format(n, k))
            self.logger.warning("[DoD] {}.".format(reason))
            return None, reason

        X_scaled = ad.obs.loc[valid_idx, scaled_features].values

        d_raw, d_dod = self._compute_dod_distances(X_scaled, k=k, max_cells=max_cells)
        # Both indexed by valid_idx, which is shorter than the view whenever
        # _zscore_features dropped a NaN row.
        overlap = self._compute_neighborhood_overlap(d_raw, d_dod, index=valid_idx, k=k)
        change = self._compute_dod_change_score(d_raw, d_dod, index=valid_idx)

        return {
            "change": change,
            "overlap": overlap,
            "valid_idx": valid_idx,
            "features": scaled_features,
            "k": k,
        }, None

    def _compute_dod_distances(self, X, k: int=15, metric='euclidean', max_cells: int=MAX_DOD_CELLS):
        """Thin wrapper over the module-level `compute_dod_distances`."""
        return compute_dod_distances(X, k=k, metric=metric, logger=self.logger, max_cells=max_cells)

    @log_exceptions()
    def _compute_dod_change_score(self, D_raw, D_dod, index):
        """Mean relative shrinkage of a cell's distances under the DoD transform.

        Args:
            D_raw (np.ndarray): Original distances.
            D_dod (np.ndarray): Transformed distances.
            index (np.ndarray): Cells scored.

        Returns:
            np.ndarray: Per-cell change score.
        """
        if D_raw.shape != D_dod.shape:
            raise ValueError("D_raw and D_dod must have same shape")
        if D_raw.shape[0] != len(index):
            raise ValueError('[QC Filter] DoD matrix ({}) and index ({}) disagree'.format(
                D_raw.shape[0], len(index)))

        mean_raw = np.mean(D_raw, axis=1)
        mean_dod = np.mean(D_dod, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            change = np.where(mean_raw > 0, np.abs(mean_raw - mean_dod) / mean_raw, np.nan)
        return pd.Series(change, index=index, name='dod_change')

    @log_exceptions()
    def _compute_neighborhood_overlap(self, dmat_original, dmat_dod, index, k: int=15):
        """Fraction of a cell's K nearest neighbours preserved under the DoD transform.

        Args:
            dmat_original (np.ndarray): Original distances.
            dmat_dod (np.ndarray): Transformed distances.
            index (np.ndarray): Cells scored.
            k (int): Neighbours compared.

        Returns:
            np.ndarray: Per-cell overlap fraction.
        """
        if dmat_original.shape != dmat_dod.shape:
            raise ValueError("Distance matrices must be the same shape")
        if dmat_original.shape[0] != len(index):
            raise ValueError('[QC Filter] DoD matrix ({}) and index ({}) disagree'.format(
                dmat_original.shape[0], len(index)))

        n = dmat_original.shape[0]
        k = max(1, min(k, n - 1))

        # argpartition, not argsort: only the k nearest matter, and this is O(n)
        # per row rather than O(n log n).
        nn_orig = np.argpartition(dmat_original, kth=k, axis=1)[:, :k + 1]
        nn_dod = np.argpartition(dmat_dod, kth=k, axis=1)[:, :k + 1]

        overlap = np.zeros(n)
        for i in range(n):
            # Drop self, which argpartition leaves somewhere in the slice.
            set_orig = set(nn_orig[i]) - {i}
            set_dod = set(nn_dod[i]) - {i}
            overlap[i] = len(set_orig & set_dod) / float(k)

        return pd.Series(np.clip(overlap, 0.0, 1.0), index=index, name='neighborhood_overlap')


    def _run_bc_filter(self, filter_flags=None):
        """Run the barcode-rank stage.

        Args:
            filter_flags (list[str]): Upstream flag columns this stage is seeded with.

        Returns:
            list[str]: The flag columns after this stage, for the next one.
        """
        bc_cfg = self.filter_cfg.get("filters", {}).get("barcode_rank", {})
        mode = self._effective_mode('barcode_rank')
        if mode == 'off':
            return
        layer = bc_cfg.get('layer', 'corrected_counts')
        features = bc_cfg.get('features', [])
        self.adata_view = self._get_filtered_view(filter_flags)
        flag_col = self._bcrank_filter(
            features=self._resolve_keys(metrics = features, layer = layer),
            min_total_counts=bc_cfg.get('min_total_counts', 500),
            clip_percent=tuple(bc_cfg.get('clip_percent', (5, 95))),
            smoothing=bc_cfg.get('smoothing', True),
            polyorder=bc_cfg.get('polyorder', 3),
            threshold_override=bc_cfg.get('threshold'))
        self.adata_view = None
        # Advisory: the flags were computed and recorded, but must not seed the
        # cascade views below or reach qc_plot's removal mask.
        return flag_col if mode == 'filter' else None

    def _resolve_sample_type_value(self, value, feature_name=''):
        """Resolve a MAD cap that may vary by sample_type.

        Args:
            value: Scalar cap, or a mapping of sample_type to cap.
            feature_name (str): Feature the cap applies to.

        Returns:
            float or None: Cap for this sample, or None if not applicable.
        """
        if not isinstance(value, dict):
            return value
        if self.sample_type not in value:
            raise ValueError(
                '[QC Filter] Sample "{}" has sample_type "{}", but the sample_type-keyed cap '
                'for feature "{}" in filter profile "{}" only defines {}. A cap that does not '
                'cover this sample would leave the absolute backstop unset -- add a "{}" entry '
                'to it in filter_config.yaml, or correct the sample\'s sample_type.'.format(
                    self.sample_name, self.sample_type, feature_name, self.filter_profile,
                    sorted(value), self.sample_type))
        return value[self.sample_type]

    def _run_mad_filter(self, filter_flags=None):
        """Run the adaptive MAD entries of the filter config.

        Args:
            filter_flags (list): Accumulated flag columns, appended to in place.
        """
        mad_cfg = self.filter_cfg.get("filters", {}).get("mad", {})
        mode = self._effective_mode('mad')
        if mode == 'off':
            return
        feature_cfgs = mad_cfg.get('features', [])
        layer = mad_cfg.get('layer', 'corrected_counts')
        default_n_mad = mad_cfg.get("default_n_mad", 3)
        default_direction = mad_cfg.get("default_direction", 'both')
        self.adata_view = self._get_filtered_view(filter_flags)

        adaptive_flags = []
        for feat_cfg in feature_cfgs:
            if isinstance(feat_cfg, str):
                feat_cfg = {"name": feat_cfg}
            metric = feat_cfg["name"]
            # Config-key presence, not the resolved value: a bare max_value/
            # min_value with no `direction` key of its own is the documented
            # degenerate-feature override (absolute-only, adaptive rule not run),
            # and its cap is applied by _run_absolute_filter regardless.
            has_absolute = (feat_cfg.get("max_value") is not None
                            or feat_cfg.get("min_value") is not None)
            if not (('direction' in feat_cfg) or not has_absolute):
                continue
            obs_key = self._resolve_keys([metric], layer=layer)[0]
            n_mad = feat_cfg.get("n_mad", default_n_mad)
            transform = feat_cfg.get("transform", mad_cfg.get("transform", True))
            # extend, not append(...[0]): a feature whose scale is degenerate is
            # skipped and returns no flag column.
            adaptive_flags.extend(self._mad_filter(
                features=[obs_key],
                k=n_mad,
                direction=feat_cfg.get("direction", default_direction),
                transform=transform,
                max_value=None,
                min_value=None))
        self.adata_view = None
        return adaptive_flags if mode == 'filter' else None

    def _run_absolute_filter(self, filter_flags=None):
        """Apply the fixed max_value/min_value caps as a cleanup pass.

        Runs after the adaptive rules. See docs/decisions/qc-cascade.md.

        Args:
            filter_flags (list): Accumulated flag columns, appended to in place.
        """
        mad_cfg = self.filter_cfg.get("filters", {}).get("mad", {})
        # distribution_dependent=False: the caps keep filtering even when
        # sample_quality flags the sample. mad.mode still gates them (off -> skip,
        # advisory -> record only, filter -> remove).
        mode = self._effective_mode('mad', distribution_dependent=False)
        if mode == 'off':
            return
        feature_cfgs = mad_cfg.get('features', [])
        layer = mad_cfg.get('layer', 'corrected_counts')
        default_n_mad = mad_cfg.get("default_n_mad", 3)
        default_direction = mad_cfg.get("default_direction", 'both')
        self.adata_view = self._get_filtered_view(filter_flags)

        absolute_flags = []
        for feat_cfg in feature_cfgs:
            if isinstance(feat_cfg, str):
                feat_cfg = {"name": feat_cfg}
            metric = feat_cfg["name"]
            max_value = self._resolve_sample_type_value(feat_cfg.get("max_value"), feature_name=metric)
            min_value = self._resolve_sample_type_value(feat_cfg.get("min_value"), feature_name=metric)
            if max_value is None and min_value is None:
                continue
            obs_key = self._resolve_keys([metric], layer=layer)[0]
            n_mad = feat_cfg.get("n_mad", default_n_mad)
            transform = feat_cfg.get("transform", mad_cfg.get("transform", True))
            absolute_flags.extend(self._mad_filter(
                features=[obs_key],
                k=n_mad,
                direction=feat_cfg.get("direction", default_direction),
                transform=transform,
                max_value=max_value,
                min_value=min_value,
                record_method='absolute'))
        self.adata_view = None
        return absolute_flags if mode == 'filter' else None

    def _run_miqc_filter(self, filter_flags=None):
        """Run the miQC stage, in parallel with MAD's adaptive rule.

        Args:
            filter_flags (list[str]): Upstream flag columns this stage is seeded with.

        Returns:
            list[str]: The flag columns after this stage, for the next one.
        """
        miqc_cfg = self.filter_cfg.get("filters", {}).get("miqc", {})
        mode = self._effective_mode('miqc')
        if mode == 'off':
            return
        features = miqc_cfg.get('features', [])
        layer = miqc_cfg.get('layer', 'corrected_counts')
        complexity_key = self._resolve_keys([miqc_cfg.get('complexity_metric', 'n_genes')], layer=layer)[0]
        features = self._resolve_keys(features, layer=layer)
        self.adata_view = self._get_filtered_view(filter_flags)

        flags = []
        for feature in features:
            flag_col = self._miqc_filter(
                feature=feature,
                complexity_key=complexity_key,
                posterior_threshold=miqc_cfg.get('posterior_threshold', 0.75),
                min_component_separation=miqc_cfg.get('min_component_separation', 0.02),
                min_cells_per_component=miqc_cfg.get('min_cells_per_component', 20))
            if flag_col:
                flags.extend(flag_col)
        self.adata_view = None
        return flags if mode == 'filter' else None

    def _run_regression_filter(self, filter_flags=None):
        """Run the regression-residual stage.

        Args:
            filter_flags (list[str]): Upstream flag columns this stage is seeded with.

        Returns:
            list[str]: The flag columns after this stage, for the next one.
        """
        reg_cfg = self.filter_cfg.get("filters", {}).get("regression", {})
        mode = self._effective_mode('regression')
        if mode == 'off':
            return
        x = reg_cfg.get('x', 'total_counts')
        y = reg_cfg.get('y', 'n_genes')
        k = reg_cfg.get('k', 3)
        layer = reg_cfg.get('layer', 'corrected_counts')
        x, y = self._resolve_keys([x, y], layer=layer)
        self.adata_view = self._get_filtered_view(filter_flags)
        flag_col = self._regression_filter(
            x, y, k,
            transform=reg_cfg.get('transform', True),
            direction=reg_cfg.get('direction', 'both'))
        self.adata_view = None
        return flag_col if mode == 'filter' else None

    def _run_gmm_filter(self, filter_flags=None):
        """Run the Gaussian-mixture stage.

        Args:
            filter_flags (list[str]): Upstream flag columns this stage is seeded with.

        Returns:
            list[str]: The flag columns after this stage, for the next one.
        """
        gmm_cfg = self.filter_cfg.get("filters", {}).get("gmm", {})
        mode = self._effective_mode('gmm')
        if mode == 'off':
            return
        features = gmm_cfg.get('features', {})
        layer = gmm_cfg.get('layer', 'corrected_counts')
        features = self._resolve_keys(features, layer=layer)
        self.adata_view = self._get_filtered_view(filter_flags)
        flag_col = self._gmm_filter(
            features,
            max_components=gmm_cfg.get('max_components', 3),
            cutoff_quantile=gmm_cfg.get('cutoff_quantile', 0.99),
            transform=gmm_cfg.get('transform', True))
        self.adata_view = None
        return flag_col if mode == 'filter' else None

    def _run_iforest_filter(self, filter_flags=None):
        """Run the isolation-forest stage. Its count is a quota, not a finding.

        Args:
            filter_flags (list[str]): Upstream flag columns this stage is seeded with.

        Returns:
            list[str]: The flag columns after this stage, for the next one.
        """
        if_cfg = self.filter_cfg.get("filters", {}).get("iforest", {})
        mode = self._effective_mode('iforest')
        if mode == 'off':
            return
        features = if_cfg.get('features', {})
        layer = if_cfg.get('layer', 'corrected_counts')
        features = self._resolve_keys(features, layer=layer)
        self.adata_view = self._get_filtered_view(filter_flags)
        flag_col = self._iforest_filter(
            features,
            n_estimators=if_cfg.get('n_estimators', 100),
            contamination=if_cfg.get('contamination', 0.05),
            random_state=if_cfg.get('random_state', 42))
        self.adata_view = None
        return flag_col if mode == 'filter' else None

    def _run_dod_filter(self, filter_flags=None):
        """Run the distance-of-distance stage, subject to its `max_cells` guard.

        Args:
            filter_flags (list[str]): Upstream flag columns this stage is seeded with.

        Returns:
            list[str]: The flag columns after this stage, for the next one.
        """
        dod_cfg = self.filter_cfg.get("filters", {}).get("dod", {})
        mode = self._effective_mode('dod')
        if mode == 'off':
            return
        features = dod_cfg.get("features", {})
        layer = dod_cfg.get('layer', 'corrected_counts')
        method = dod_cfg.get("method", "overlap")
        features = self._resolve_keys(features, layer=layer)
        self.adata_view = self._get_filtered_view(filter_flags)
        flag_col = self._dod_filter(
            features,
            method=method,
            k=dod_cfg.get('k', 15),
            overlap_threshold=dod_cfg.get('overlap_threshold', 0.65),
            contamination=dod_cfg.get('contamination', 0.05),
            max_cells=dod_cfg.get('max_cells', MAX_DOD_CELLS))
        self.adata_view = None
        return flag_col if mode == 'filter' else None

    @log_exceptions()
    def run(self, force: bool=False):
        """Run the whole cascade and write the annotated object.

        Args:
            force (bool): Re-run even if the filter sentinel exists.

        Returns:
            AnnData: The object, with flag columns in `.obs` and diagnostics in
                `.uns['qc']`.
        """
        self.logger.info('[QC filter] Starting QC filtering for sample: {}'.format(self.sample_name))
        done_path = self.paths['done'] / 'qcfilter.done'
        if done_path.exists() and not force:
            self.logger.info('[QC] skipping sample "{}"; qcfilter.done exists (use --force-qc to override)'.format(self.sample_name))
            return

        # Clear any previous run's diagnostics before recording this one's.
        # `_qcfilters.h5ad` is re-read and re-written in place across stages, so
        # .uns['qc'] otherwise accumulates: an entry for a feature that has since
        # been removed from the config, or recorded under a key this version no
        # longer writes, survives forever and keeps appearing in the report with
        # a stale explanation attached. Seen in practice after the sample_quality
        # gate changed from skipping filters to downgrading them -- the old
        # '{feature}_adaptive' skip records lingered alongside the new entries,
        # claiming filters had been skipped when they had actually run.
        #
        # Same reasoning as qc_plot.py clearing its own figure prefixes. Nothing
        # upstream of this stage writes .uns['qc'], and doublets.py writes its
        # own entry after this runs, so clearing the whole key is safe.
        if 'qc' in self.adata.uns:
            self.logger.info('[QC Filter] Clearing {} diagnostic block(s) from a previous run'.format(
                len(self.adata.uns['qc'])))
        self.adata.uns['qc'] = {}

        flag_bc = self._run_bc_filter()

        self._assess_sample_quality(filter_flags = flag_bc)

        # MAD's adaptive rule and miQC run in PARALLEL, both on the barcode-rank
        # survivors: a cell fails the organelle/count check if EITHER flags it.
        # miQC used to be seeded on bc+mad, so it only ever saw cells the adaptive
        # upper-tail rule and the absolute caps had already stripped down -- the
        # compromised high-fraction population it exists to find was gone before it
        # looked. Seeding it on flag_bc gives it the full putative-droplet set.
        flag_mad = self._run_mad_filter(filter_flags = flag_bc)
        flag_miqc = self._run_miqc_filter(filter_flags = flag_bc)

        # The absolute caps are a cleanup pass on whatever the adaptive rule and
        # miQC left. Fixed bounds, so the OR'd removal mask is identical to folding
        # them into MAD; running them here reports them as their own "Absolute
        # cutoff" stage and keeps them filtering even on a flagged sample.
        flag_abs = self._run_absolute_filter(
            filter_flags = self._merge_flag_lists(flag_bc, flag_mad, flag_miqc))

        # Everything downstream sees the union of the organelle/count checks --
        # exactly the set it saw before (bc + all MAD + miQC).
        organelle = self._merge_flag_lists(flag_bc, flag_mad, flag_miqc, flag_abs)

        flag_rgrs = self._run_regression_filter(filter_flags = organelle)

        flag_gmm = self._run_gmm_filter(filter_flags = self._merge_flag_lists(organelle, flag_rgrs))

        flag_if = self._run_iforest_filter(filter_flags = self._merge_flag_lists(organelle, flag_rgrs))

        flag_dod = self._run_dod_filter(filter_flags = self._merge_flag_lists(organelle, flag_rgrs))

        self._record_post_cascade_viability()

        write_done_flag(done_path)
        write_h5ad(
            self.adata,
            self.out_dir / f"{self.sample_name}_qcfilters.h5ad",
            sample=self.sample_name,
            mapper=self.mapper,
            stage='qc_filter',
            config_paths=self.config_paths)
        self.logger.info('[QC] Finished QC filter computation for sample "{}", writing done flag to "{}"'.format(self.sample_name, done_path))
        
    def _merge_flag_lists(self,*args):
        """Combine every stage's flag columns into the sample's removal record.

        Returns:
            list[str]: Flag columns, in cascade order.
        """
        merged = []
        for arg in args:
            if arg is None:
                continue
            elif isinstance(arg, str):
                merged.append(arg)  # treat single string as one flag
            elif isinstance(arg, Iterable):
                merged.extend(arg)
        return list(set(merged))
                