#!/usr/bin/env python3

"""
Self-contained HTML QC reports for the mapping stage.

Pure consumer: everything rendered here is read back from artifacts the earlier
stages already wrote. Nothing is recomputed, re-plotted or re-clustered, so a
report can be regenerated at any time without touching the pipeline.

Templating is stdlib only. jinja2 happens to be importable in the environment as
a CellBender transitive, but depending on that accidentally would make the
environment unreproducible, and the layout here is fixed enough not to need it.
"""

import html
import json
import logging

import numpy as np
import pandas as pd
import scanpy as sc

from pathlib import Path
from shared.utils import log_exceptions, write_done_flag, get_sample_output_paths
from .interactive import build_payload, render_section, interactive_css, as_list
from .qc_filter import NON_FILTER_METHODS
from shared.samples import load_samples


# Cascade stages, in the order qc_filter.py runs them. gmm/iforest/dod are
# siblings at the same stage -- they all filter on bc+mad+miqc+regression flags.
_METHOD_ORDER = ['bcrank', 'mad', 'miqc', 'absolute', 'regression', 'gmm', 'iforest', 'dod',
                 'solo', 'scrublet', 'doublet_consensus']

_METHOD_LABELS = {
    'bcrank': 'Barcode rank',
    'mad': 'Median absolute deviation',
    'absolute': 'Absolute cutoff',
    'miqc': 'miQC (complexity vs. fraction)',
    'regression': 'Regression residual',
    'gmm': 'Gaussian mixture',
    'iforest': 'Isolation forest',
    'dod': 'Distance of distance',
    'solo': 'SOLO (doublet detection)',
    'scrublet': 'Scrublet (doublet detection)',
    'doublet_consensus': 'Doublet consensus (SOLO &amp; Scrublet)'}

# Figure filename prefix per method, as composed by QCPlotter._save_plot
# ({qc_dir}/{prefix}_{name}.png).
_METHOD_FIGURE_PREFIX = {
    'bcrank': 'bcrank_',
    'mad': 'MAD_',
    'absolute': 'absoluteCut_',
    'miqc': 'miqc_',
    'regression': 'regression_',
    'gmm': 'gmm_',
    'iforest': 'iF_',
    'dod': 'DoD_',
    'solo': 'solo_',
    'scrublet': 'scrublet_',
    'doublet_consensus': 'consensus_'}

# Shared with the annotation stage's review report: an unknown value that
# formats differently between two reports is two different claims about the data.
from shared.html import _CSS, _PAGE, _esc, _fmt_int, _fmt_float, _figure



def describe_threshold(method: str, meta: dict) -> str:
    """Render a method's threshold as HTML.

    Each filter stores this differently -- a dict, a bare float, a singular
    'threshold', or nothing -- so there is no uniform accessor to lean on. An em
    dash when there is no threshold.
    """
    if method in ('bcrank', 'iforest'):
        return _fmt_float(meta.get('threshold'))
    if method in ('mad', 'absolute', 'regression'):
        bounds = meta.get('thresholds') or {}
        if not isinstance(bounds, dict):
            return _fmt_float(bounds)
        lower, upper = bounds.get('lower'), bounds.get('upper')
        parts = []
        if lower is not None:
            parts.append('lower {}'.format(_fmt_float(lower)))
        if upper is not None:
            parts.append('upper {}'.format(_fmt_float(upper)))
        return ', '.join(parts) if parts else '&mdash;'
    if method == 'gmm':
        return _fmt_float(meta.get('thresholds'))
    if method == 'miqc':
        return 'posterior &gt; {}'.format(_fmt_float(meta.get('posterior_threshold'), 2))
    if method == 'dod':
        if meta.get('selection_method') == 'overlap':
            return 'neighbourhood overlap &lt; {}'.format(_fmt_float(meta.get('overlap_threshold'), 2))
        return html.escape(str(meta.get('selection_method', ''))) or '&mdash;'
    if method in ('solo', 'scrublet'):
        thr = meta.get('threshold')
        if thr is None:
            return '&mdash;'
        auto = meta.get('threshold_method') == 'gmm_valley'
        return 'score &gt; {}{}'.format(_fmt_float(thr, 2),
                                        ' <small>(auto)</small>' if auto else '')
    if method == 'doublet_consensus':
        return 'called by <strong>both</strong> SOLO and Scrublet'
    return '&mdash;'


def describe_selection(method: str, meta: dict) -> str:
    """Explain how a method reached its threshold, where that is not self-evident.

    A barcode-rank cut-off is one of four candidates chosen by policy, and an
    isolation-forest count is a quota rather than a finding. '' when the number
    speaks for itself.
    """
    if method == 'bcrank':
        candidates = meta.get('candidate_thresholds') or {}
        if not candidates:
            return ''
        chosen = meta.get('method', '')
        parts = []
        for name, value in sorted(dict(candidates).items()):
            label = '{} {}'.format(html.escape(str(name)), _fmt_float(value, 0))
            parts.append('<strong>{}</strong>'.format(label) if name == chosen else label)
        policy = meta.get('selection_policy', '')
        suffix = ' &rarr; selected <strong>{}</strong>{}'.format(
            html.escape(str(chosen)),
            ' (most lenient of {})'.format(len(candidates)) if policy == 'most_lenient' else '')
        return ' &middot; '.join(parts) + suffix

    if method == 'iforest':
        c = meta.get('contamination')
        if isinstance(c, (int, float)):
            return ('contamination={:g} is a <strong>quota</strong>: exactly {:.0%} of cells are '
                    'flagged regardless of sample quality, so this is the number requested, '
                    'not a finding.'.format(c, c))
        return 'contamination={} &mdash; scored against the forest offset rather than a fixed quota.'.format(
            html.escape(str(c)))

    if method in ('solo', 'scrublet') and meta.get('unconverged'):
        # A doublet rate this high describes the model, not the sample. Said
        # here rather than left to the generic "downgraded" pill, because the
        # number in the adjacent cell looks like an ordinary result and would
        # otherwise be read as one.
        return ('<strong>{:.1%} of assessed cells called doublets</strong>, above the {:.0%} '
                'plausibility ceiling &mdash; a per-sample doublet model that flags most of its '
                'input has not converged. These calls removed no cells and should not be trusted; '
                'treat the scores as evidence that the model failed, not as doublet '
                'probabilities.'.format(float(meta.get('doublet_rate', 0.0)),
                                        float(meta.get('max_plausible_rate', 0.5))))

    if method in ('solo', 'scrublet'):
        diag = meta.get('threshold_diagnostics') or {}
        if diag.get('method') == 'gmm_valley':
            return ('cut placed at the <strong>valley</strong> between the singlet and doublet '
                    'modes of this sample&rsquo;s score distribution (2-component fit, mode '
                    'separation {} in probability units) rather than a fixed cut.'.format(
                        _fmt_float(diag.get('separation'), 2)))
        if diag.get('skip_reason'):
            return ('score distribution not clearly bimodal, so the fixed fallback threshold was '
                    'used &mdash; {}.'.format(html.escape(str(diag.get('skip_reason')))))
        return ''

    if method == 'doublet_consensus':
        n_solo = meta.get('n_solo_doublets')
        n_scr = meta.get('n_scrublet_doublets')
        n_both = meta.get('n_flagged')
        n_assessed = meta.get('n_assessed')
        if None in (n_solo, n_scr, n_both):
            return ('intersection of the SOLO and Scrublet calls &mdash; a cell is a consensus '
                    'doublet only when both methods flag it.')
        return ('<strong>intersection</strong> of the two methods on the {} jointly-assessed '
                'cells: SOLO called {}, Scrublet called {}, and <strong>{}</strong> were called by '
                'both. Requiring agreement is deliberately conservative &mdash; it keeps SOLO&rsquo;s '
                'over-calling from driving removal on its own.'.format(
                    _fmt_int(n_assessed), _fmt_int(n_solo), _fmt_int(n_scr), _fmt_int(n_both)))

    if method == 'gmm':
        bits = []
        if meta.get('n_components') is not None:
            bits.append('{} components by BIC'.format(meta['n_components']))
        if meta.get('covariance_type'):
            bits.append('{} covariance'.format(html.escape(str(meta['covariance_type']))))
        return ', '.join(bits)

    if method == 'miqc':
        gap = meta.get('component_separation')
        if gap is None:
            return ''
        return ('2-component fit, mean fraction gap {} between compromised and intact '
                'components &mdash; below the configured minimum is reported as '
                '"not filtered" instead of a threshold.').format(_fmt_float(gap))

    if method == 'dod':
        if meta.get('k') is not None:
            return 'K={} neighbourhood'.format(meta['k'])
    return ''


def parse_diagnostics(uns_qc: dict) -> list:
    """Flatten `.uns['qc']` into one record per (method, feature).

    Mirrors `QCPlotter._parse_qc_diagnostics`. Reads only scalars -- bcrank's
    payload also carries full-length per-cell arrays that must never be rendered --
    and excludes NON_FILTER_METHODS, which are sample-level verdicts rendered as
    banners. Returns (method, feature, meta) tuples in cascade order. See
    docs/decisions/qc-report.md.
    """
    records = []
    for method, entry in (uns_qc or {}).items():
        if method in NON_FILTER_METHODS:
            continue
        if not isinstance(entry, dict):
            continue
        if entry and all(isinstance(v, dict) for v in entry.values()):
            for feature, meta in entry.items():
                records.append((method, feature, meta))
        else:
            feature = ', '.join(entry.get('features', [])) or entry.get('feature', '')
            records.append((method, feature, entry))

    def sort_key(rec):
        """Sort by cascade depth, then method order, then feature name."""
        method, feature, meta = rec
        # Cascade position is implicit: each stage is seeded with every
        # upstream stage's flags, so their count orders the stages.
        depth = len((meta.get('run_context') or {}).get('filtered_on_flags', []))
        order = _METHOD_ORDER.index(method) if method in _METHOD_ORDER else len(_METHOD_ORDER)
        return (depth, order, feature)

    return sorted(records, key=sort_key)


# Figure stem -> (what it shows, what the axis is). Without these the caption is
# the filename with underscores swapped for spaces.
_FIGURE_CAPTIONS = {
    'DoD_umap_raw': (
        'UMAP of the cells DoD assessed, embedded from the <strong>raw</strong> pairwise '
        'distances in z-scored QC-feature space (not gene expression, and not PCA). '
        'Coloured by DoD\u2019s own flag. This is the "before" half of the pair below.'),
    'DoD_umap_dod': (
        'The same cells embedded from the <strong>distance-of-distance transform</strong> of '
        'those distances. Comparing it with the raw-distance UMAP above is the point: cells '
        'whose neighbourhood changes most under the transform are the ones DoD calls noisy.'),
    'DoD_umap_filt': (
        'UMAP of the cells DoD kept, embedded from <strong>z-scored QC features</strong> '
        '(cosine metric), coloured by Leiden clusters computed on that same graph. Still QC '
        'space \u2014 it says how the retained cells sit relative to each other on quality '
        'metrics, not on biology.'),
    'DoD_change': 'Distribution of how far each cell\u2019s distances move under the DoD transform.',
    'DoD_overlap': (
        'Fraction of each cell\u2019s K nearest neighbours retained after the transform. '
        'Low overlap is the paper\u2019s own noise cue.'),
    'DoD_change_overlap': 'The two DoD statistics against each other, coloured by flag.',
    'gmm_mahal': 'Mahalanobis distance to the fitted mixture, per cell, with the cut-off marked.',
    'gmm_parallel_coordinates': 'Each cell\u2019s QC features as a line, coloured by Mahalanobis distance.',
    'iF_score': (
        'Isolation-forest score distribution. The cut-off shown is a <strong>quota</strong> '
        '(<code>contamination</code>), not a detection threshold \u2014 that proportion is '
        'flagged however clean the sample is.'),
    'iF_parallel': 'QC features per cell, coloured by isolation-forest score.',
    'solo_score': 'SOLO doublet score per assessed cell, with the hard-call threshold marked.',
    'solo_umap': (
        'UMAP of the <strong>scVI latent space</strong> trained on this sample\u2019s QC-passed '
        'cells, coloured by SOLO doublet score. This is the best-conditioned embedding the '
        'pipeline produces for this sample.'),
    'scrublet_score': (
        'Scrublet doublet score per assessed cell, with the hard-call threshold marked. '
        'Scrublet runs in-process on the raw counts of the QC-passed cells and simulates '
        'doublets from the observed transcriptomes.'),
    'scrublet_umap': (
        'The scVI latent-space UMAP (shared with SOLO, when SOLO ran), coloured by '
        '<strong>Scrublet</strong> doublet score \u2014 so the two methods\u2019 calls can be '
        'compared on the same embedding.'),
    'consensus_umap': (
        'The scVI latent-space UMAP, coloured by <strong>agreement</strong>: called by both '
        'methods (consensus doublet), by SOLO only, by Scrublet only, or by neither. Only the '
        '"both" cells are consensus doublets.'),
    'filtered_final_umap': (
        'UMAP of the retained cells, embedded from <strong>gene expression</strong> '
        '(corrected_counts \u2192 consensus HVGs \u2192 PCA), coloured by Leiden cluster. '
        'The one figure here in expression space rather than QC space.'),
}


def figure_caption(path, method: str = '') -> str:
    """A caption for a QC figure, keyed on its stem in `_FIGURE_CAPTIONS`, else
    built per method, else the stem with spaces."""
    stem = Path(path).stem
    if stem in _FIGURE_CAPTIONS:
        return _FIGURE_CAPTIONS[stem]
    prefix = _METHOD_FIGURE_PREFIX.get(method, '')
    feature = stem[len(prefix):] if prefix and stem.startswith(prefix) else stem
    if method == 'mad':
        # qc_plot names these `MAD_{adaptive|absolute}_{raw feature}` -- a
        # fraction feature carries one entry of each, and the two read
        # identically without the mode in the name.
        mode, _, feat = feature.partition('_')
        if mode not in ('adaptive', 'absolute'):
            mode, feat = '', feature
        which = {'adaptive': ' The cut-off is fit from this sample’s own spread.',
                 'absolute': ' The cut-off is the flat <code>max_value</code>/<code>min_value</code> '
                             'cap from the config, identical on every sample.'}.get(mode, '')
        return ('Distribution of <code>{}</code>, coloured by whether MAD flagged the cell; '
                'dashed lines are the cut-offs. Values and cut-offs are in the feature’s own '
                'raw units — the axis carries the transform MAD fit on, so the shape is the '
                'fitted one but the numbers are the ones to type into the interactive panel '
                'below.{}'.format(html.escape(feat), which))
    if method == 'absolute':
        return ('Distribution of <code>{}</code>, coloured by whether the absolute cutoff '
                'removed the cell. The cut-off is the flat <code>max_value</code>/'
                '<code>min_value</code> cap from filter_config.yaml, in the feature’s own '
                'raw units and identical on every sample.'.format(html.escape(feature)))
    if method == 'miqc':
        return ('miQC: <code>{}</code> against library complexity, with the intact-population '
                'trend line where one was fitted.'.format(html.escape(feature)))
    if method == 'regression':
        return ('Regression fit between the two features named in the filename, with the '
                'residual bounds that define the cut.')
    if method == 'bcrank':
        return 'Barcode-rank curve with the selected cut-off; the shaded bands are the clipped range.'
    return html.escape(stem.replace('_', ' '))


def _umap_param_note(params: dict, intro: str = 'UMAP') -> str:
    """One-line record of the settings a UMAP figure was built with.

    Formatted from the `.uns['umap_params']` entry persisted by whichever stage
    computed the embedding, so the report stays a pure consumer. Renders nothing
    when the entry is absent, so a missing record is silent rather than a hole.
    """
    if not params:
        return ''

    def _num(v):
        """Format a number compactly, falling back to escaped text."""
        try:
            return '{:g}'.format(float(v))
        except (TypeError, ValueError):
            return html.escape(str(v))

    bits = []
    n_nb = params.get('n_neighbors')
    if n_nb not in (None, ''):
        nb = 'neighbours: k = {}'.format(_num(n_nb))
        rep = params.get('representation')
        if rep not in (None, ''):
            nb += ' on <code>{}</code>'.format(html.escape(str(rep)))
        metric = params.get('metric')
        if metric not in (None, ''):
            nb += ' ({} metric)'.format(html.escape(str(metric)))
        bits.append(nb)
    if params.get('min_dist') not in (None, ''):
        bits.append('UMAP min_dist = {}'.format(_num(params['min_dist'])))
    if params.get('leiden_resolution') not in (None, ''):
        bits.append('Leiden resolution = {}'.format(_num(params['leiden_resolution'])))

    extra = []
    if params.get('n_pcs') not in (None, ''):
        extra.append('{} PCs'.format(_num(params['n_pcs'])))
    if params.get('n_hvg') not in (None, ''):
        extra.append('{} HVGs'.format(_num(params['n_hvg'])))
    tail = ' ({})'.format(', '.join(extra)) if extra else ''

    if not bits:
        return ''
    return '<p class="note">{} settings &mdash; {}{}.</p>'.format(
        html.escape(intro), '; '.join(bits), tail)


def describe_view(meta: dict, n_input=None) -> str:
    """Say how many cells a method's figures actually show.

    Each cascade stage only ever saw the cells its upstream stages passed, and its
    figures are drawn on exactly that subset -- which matters once those figures
    become the surface for selecting cells by hand. Renders nothing when the method
    saw everything, rather than a reassuring "all cells" the data does not support.
    See docs/decisions/qc-report.md.
    """
    # anndata round-trips a .uns list as a numpy array, so `x or []` would raise
    # on its truthiness. Convert first, test after.
    upstream = as_list((meta or {}).get('run_context', {}).get('filtered_on_flags', None))
    if not upstream or n_input in (None, 0):
        return ''
    # Methods, not flag columns: 'mad, miqc' reads better than six mangled
    # column names. Sorted, since the upstream list has no usable order.
    methods = sorted({
        m for m in _METHOD_ORDER
        for flag in upstream
        if flag.startswith('flag_{}'.format('bcrank' if m == 'bcrank' else m))})
    gated = ', '.join(_METHOD_LABELS.get(m, m) for m in methods) or 'upstream filters'
    return ('<p class="sub">Figures below are drawn on the cells that reached this stage, '
            'not on all {total} &mdash; {gated} ran first.</p>').format(
        total=_fmt_int(n_input), gated=html.escape(gated))


def is_advisory(meta: dict) -> bool:
    """True when this entry was recorded by a method that removes no cells.

    Absent exec_mode means 'filter', so artifacts written before execution modes
    existed read exactly as they did before.
    """
    return (meta or {}).get('exec_mode', 'filter') != 'filter'


def cascade_stages(records: list) -> list:
    """Collapse per-feature records into cascade stages.

    `n_flagged` is view-relative, so per-method counts cannot be summed or stacked;
    each stage's removal is reported against the cells that reached it. Advisory
    methods are excluded outright -- they would put non-removing rows in a removal
    waterfall. Returns one dict per stage ('method', 'depth', 'features'). See
    docs/decisions/qc-report.md.
    """
    records = [r for r in records if not is_advisory(r[2])]
    stages = []
    for method, feature, meta in records:
        depth = len((meta.get('run_context') or {}).get('filtered_on_flags', []))
        n_flagged = int(meta.get('n_flagged', 0) or 0)
        skipped = bool(meta.get('skipped'))
        for stage in stages:
            if stage['method'] == method:
                stage['features'].append((feature, n_flagged, skipped))
                break
        else:
            stages.append({'method': method, 'depth': depth,
                           'features': [(feature, n_flagged, skipped)]})
    return stages


class ReportBuilder():
    """Assemble a single-file HTML report for one sample."""

    def __init__(self, sample_name: str, sample_base_dir: Path, sample_config: Path):
        self.logger = logging.getLogger('SingleCellMappingPipeline')

        self.sample_name = sample_name
        self.sample_cfg = load_samples(sample_config)
        self.mapper = self.sample_cfg[self.sample_name]['mapper'].lower()
        self.batch = self.sample_cfg[self.sample_name].get('batch')

        self.paths = get_sample_output_paths(Path(sample_base_dir) / self.sample_name, mapper=self.mapper)
        self.qc_dir = self.paths['qc']
        self.report_path = self.qc_dir / '{}_report.html'.format(self.sample_name)
        # Populated by _load_artifacts; '' means "unknown", which the scVI
        # section treats as "cannot vouch for these figures" rather than "fresh".
        self._config_hash = ''
        # {embedding_key: {n_neighbors, metric, min_dist, ...}}, read off the
        # filtered artifact's .uns so the UMAP figures can state how they were
        # built. Empty for an older artifact that predates this being recorded.
        self._umap_params = {}
        # Set by _load_artifacts. Empty means the artifact predates the field,
        # which is not the same as "skipped" -- see _section_umap.
        self._umap_status = {}
        self._post_cascade = {}
        self._solo = {}

    # -- data ---------------------------------------------------------------

    def _load_artifacts(self):
        """Read back everything the report renders, without recomputing any of it.

        Returns (provenance, diagnostics, stats or None, quality).
        """
        adata = sc.read_h5ad(self.qc_dir / '{}_filtered.h5ad'.format(self.sample_name))
        provenance = dict(adata.uns.get('pipeline', {}))
        # For _section_scvi_integration, which must decide whether figures
        # another stage left in qc/ describe this artifact or an older one.
        self._config_hash = provenance.get('config_hash', '')
        self._umap_params = dict(adata.uns.get('umap_params', {}))
        # Why the final UMAP is missing, when it is. Absent on older artifacts,
        # which _section_umap reads as "no reason on file", not as a skip.
        self._umap_status = dict(adata.uns.get('umap_qc_status', {}))
        uns_qc = adata.uns.get('qc', {})
        diagnostics = parse_diagnostics(uns_qc)
        quality = dict(uns_qc.get('sample_quality') or {})
        self._post_cascade = dict(uns_qc.get('post_cascade') or {})
        self._solo = dict(uns_qc.get('solo') or {})

        stats_path = self.qc_dir / 'MappingStats.tsv'
        stats = pd.read_csv(stats_path, sep='\t') if stats_path.exists() else None
        return provenance, diagnostics, stats, quality

    def _interactive_payload(self, provenance: dict):
        """Build the interactive section's data from the **pre-filter** object.

        The filtered artifact holds only survivors, so a report built from it could
        offer removal but never rescue -- and rescuing cells the cascade got wrong is
        half the point. Read `backed='r'`; the QC UMAP is merged in from the filtered
        object, since removed cells have no coordinates by construction. See
        docs/decisions/qc-report.md. None if the pre-filter object is absent.
        """
        parent = self.paths['processed'] / '{}_qcfilters.h5ad'.format(self.sample_name)
        if not parent.exists():
            self.logger.info('[REPORT] No pre-filter object at "{}"; skipping the interactive '
                             'section for "{}".'.format(parent, self.sample_name))
            return None

        adata = sc.read_h5ad(parent, backed='r')
        try:
            payload = build_payload(adata, self.sample_name,
                                    config_hash=provenance.get('config_hash', ''))
            index = pd.Index([str(b) for b in adata.obs_names])
        finally:
            if getattr(adata, 'file', None) is not None:
                adata.file.close()

        try:
            filtered = sc.read_h5ad(self.qc_dir / '{}_filtered.h5ad'.format(self.sample_name),
                                    backed='r')
            try:
                if 'X_umap_qc' in filtered.obsm:
                    coords = pd.DataFrame(
                        np.asarray(filtered.obsm['X_umap_qc'], dtype=float)[:, :2],
                        index=[str(b) for b in filtered.obs_names],
                        columns=['x', 'y']).reindex(index)
                    payload['embeddings']['X_umap_qc'] = {
                        'label': 'QC UMAP (filtered cells only)',
                        'x': [None if pd.isna(v) else round(float(v), 6) for v in coords['x']],
                        'y': [None if pd.isna(v) else round(float(v), 6) for v in coords['y']]}
                for key in ('leiden_qc',):
                    if key in filtered.obs.columns and key not in payload['clusterings']:
                        labels = pd.Series(filtered.obs[key].astype(str).values,
                                           index=[str(b) for b in filtered.obs_names]).reindex(index)
                        payload['clusterings'][key] = ['' if pd.isna(v) else str(v) for v in labels]
            finally:
                if getattr(filtered, 'file', None) is not None:
                    filtered.file.close()
        except Exception as e:
            self.logger.warning('[REPORT] Could not merge the QC embedding for "{}": {}'.format(
                self.sample_name, e))
        return payload

    # -- sections -----------------------------------------------------------

    def _section_quality_banner(self, quality: dict) -> str:
        """Render the sample_quality verdict as a standalone banner.

        Not a row in the generic filter table: it is the one diagnostic meant to change
        how a human reads everything below it, so it needs to be seen first. '' when
        the sample was not flagged.
        """
        if not quality or not quality.get('flagged_for_manual_review'):
            return ''

        reasons = as_list(quality.get('reasons'))
        reasons_html = ''.join('<li>{}</li>'.format(html.escape(str(r))) for r in reasons)

        # Distinguishes `mode: filter` (verdict reached AND downstream filters
        # skipped) from `mode: advisory` (verdict reached, nothing skipped).
        # Defaults True so pre-exec_mode artifacts read unchanged.
        gate_applied = quality.get('gate_applied', True)
        if gate_applied:
            consequence = """
  <p>Every filter that fits its threshold from this sample's own distribution was
  <strong>downgraded to advisory</strong> as a result &mdash; MAD's adaptive entries, miqc,
  regression, gmm, isolation forest, DoD and SOLO. They all still ran and their scores,
  thresholds and figures are below for you to judge; they simply removed no cells. Barcode
  rank and MAD's absolute (<code>max_value</code>/<code>min_value</code>) entries did filter
  as configured, since those are fixed bounds that do not depend on this sample's
  distribution.</p>
  <p>So the filtered object for this sample reflects barcode rank and the absolute caps only.
  Use the interactive section at the foot of this report to apply the rest by hand, or set
  <code>sample_quality.mode</code> to <code>advisory</code> in filter_config.yaml to let every
  filter act normally on samples like this one.</p>"""
        else:
            consequence = """
  <p><code>sample_quality</code> is set to <code>advisory</code>, so <strong>nothing was
  skipped</strong>: every filter below still fit its thresholds from this sample's own
  distribution. Treat those thresholds with corresponding caution &mdash; on a sample this
  small or this shallow, a filter can find exactly the signal it is looking for in what is
  really counting noise. This banner is the warning; the filtering below proceeded
  anyway.</p>"""

        return """
<div class="banner">
  <strong>Flagged for manual review.</strong> This sample did not meet the configured
  cell count / depth thresholds after barcode rank:
  <ul>{reasons}</ul>{consequence}
  <p>Cells surviving barcode rank: <strong>{n_cells}</strong> (min_cells={min_cells}) &middot;
  median depth: <strong>{depth:.0f}</strong> (min_median_depth={min_depth})</p>
</div>
""".format(reasons=reasons_html or '<li class="missing">no reason recorded</li>',
           consequence=consequence,
           n_cells=_fmt_int(quality.get('n_cells')),
           min_cells=_fmt_int(quality.get('min_cells')),
           depth=float(quality.get('median_depth') or 0.0),
           min_depth=_fmt_int(quality.get('min_median_depth')))

    def _section_header(self, provenance: dict) -> str:
        """Sample title, headline counts, and the provenance disclosure."""
        n_in = provenance.get('n_cells_input')
        n_out = provenance.get('n_cells')
        n_genes = provenance.get('n_genes')

        retained = '&mdash;'
        if n_in and n_out is not None:
            retained = '{:.1f}%'.format(100.0 * float(n_out) / float(n_in))

        cards = [
            ('Cells in', _fmt_int(n_in)),
            ('Cells retained', '{} <small>({})</small>'.format(_fmt_int(n_out), retained)),
            ('Genes', _fmt_int(n_genes)),
            ('Mapper', _esc(self.mapper)),
            ('Batch', _esc(self.batch))]

        cards_html = ''.join(
            '<div class="card"><div class="k">{}</div><div class="v">{}</div></div>'.format(k, v)
            for k, v in cards)

        versions = provenance.get('tool_versions') or {}
        version_rows = ''.join(
            '<tr><td>{}</td><td>{}</td></tr>'.format(html.escape(str(k)), _esc(v))
            for k, v in sorted(dict(versions).items()))

        sources = as_list(provenance.get('source_configs'))
        source_items = ''.join('<li><code>{}</code></li>'.format(html.escape(str(s)))
                               for s in sources if s)

        return """
<h1>{sample}</h1>
<p class="sub">Mapping QC report &middot; generated from pipeline schema {schema} &middot; {ts}</p>
<div class="cards">{cards}</div>
<details>
  <summary>Provenance &amp; tool versions</summary>
  <table><tr><th>Config hash</th><td><code>{cfg_hash}</code></td></tr></table>
  <h3>Source configs</h3>
  <ul>{sources}</ul>
  <h3>Tool versions</h3>
  <table><tr><th>Component</th><th>Version</th></tr>{versions}</table>
</details>
""".format(sample=html.escape(self.sample_name),
           schema=_esc(provenance.get('schema_version')),
           ts=_esc(provenance.get('timestamp')),
           cards=cards_html,
           cfg_hash=html.escape(str(provenance.get('config_hash') or 'unknown')),
           sources=source_items or '<li class="missing">none recorded</li>',
           versions=version_rows or '<tr><td colspan="2" class="missing">not recorded</td></tr>')

    def _section_mapping(self, stats) -> str:
        """Render MappingStats.tsv as a table, or say it is unavailable."""
        if stats is None or stats.empty:
            table = '<p class="missing">MappingStats.tsv not available.</p>'
        else:
            head = ''.join('<th class="{}">{}</th>'.format(
                'num' if c != 'Metric' else '', html.escape(str(c)))
                for c in stats.columns if c != 'Sample')
            rows = []
            for _, row in stats.iterrows():
                cells = []
                for col in stats.columns:
                    if col == 'Sample':
                        continue
                    value = row[col]
                    if pd.isna(value):
                        cells.append('<td class="num missing">&mdash;</td>')
                    elif col == 'Metric':
                        cells.append('<td>{}</td>'.format(html.escape(str(value))))
                    else:
                        cells.append('<td class="num">{}</td>'.format(_fmt_int(value)
                                     if col == 'Value' else html.escape(str(value))))
                rows.append('<tr>{}</tr>'.format(''.join(cells)))
            table = '<table><tr>{}</tr>{}</table>'.format(head, ''.join(rows))

        figures = _figure(
            self.qc_dir / '{}_ReadFlow.png'.format(self.sample_name),
            'Read flow from total reads to corrected, spliced/unspliced/ambiguous counts '
            '(the spliced/unspliced/ambiguous split of the corrected total is a proportional '
            'estimate, not a direct measurement -- see note in mappingutils.py::_plot_sankey)')

        return '<h2>Mapping statistics</h2>{}{}'.format(table, figures)

    def _section_qc(self, provenance: dict, diagnostics: list, quality: dict) -> str:
        """The QC cascade: removal waterfall, per-method detail table, and figures."""
        if not diagnostics:
            return '<h2>Quality control</h2><p class="missing">No QC diagnostics recorded.</p>'

        n_in = provenance.get('n_cells_input')
        n_out = provenance.get('n_cells')

        # Waterfall: what each cascade stage removed from the cells it actually saw.
        stages = cascade_stages(diagnostics)
        rows = []
        for position, stage in enumerate(stages, start=1):
            label = _METHOD_LABELS.get(stage['method'], stage['method'])
            detail = ', '.join(
                '{} ({})'.format(f or '&mdash;', 'not filtered' if skipped else _fmt_int(n))
                for f, n, skipped in stage['features'])
            # Skipped features contribute no count; including their 0 would drag the
            # stage's lower bound down and imply they were tested and found clean.
            counts = [n for _, n, skipped in stage['features'] if not skipped]
            if len(counts) > 1 and max(counts) != sum(counts):
                # The features within a stage flag overlapping cell sets, and the
                # overlap is not recorded anywhere, so the true union is bounded
                # but not known. Showing either endpoint alone would be a guess.
                flagged = '{}&ndash;{}'.format(_fmt_int(max(counts)), _fmt_int(sum(counts)))
            else:
                flagged = _fmt_int(max(counts) if counts else 0)
            rows.append(
                '<tr><td class="num">{pos}</td><td>{label}</td><td class="num">{flagged}</td>'
                '<td>{detail}</td></tr>'.format(
                    pos=position,
                    label=html.escape(label),
                    flagged=flagged,
                    detail=detail))

        removed = None
        if n_in is not None and n_out is not None:
            removed = int(n_in) - int(n_out)

        bar = ''
        if n_in and n_out is not None:
            pct = 100.0 * float(n_out) / float(n_in)
            bar = ('<div class="bar"><span style="width:{pct:.1f}%"></span></div>'
                   '<p class="sub">{out} of {inp} cells retained ({pct:.1f}%); '
                   '{rem} removed.</p>').format(
                pct=pct, out=_fmt_int(n_out), inp=_fmt_int(n_in), rem=_fmt_int(removed))

        # Same shape as `bar` above, but relative to putative droplets (cells
        # that already survived barcode-rank) rather than the raw pre-filter
        # droplet count -- a more meaningful denominator for judging how much
        # the rest of the cascade removed. Absent (not an error) whenever
        # `sample_quality` is disabled, since that's the only place this
        # count is recorded.
        bcrank_bar = ''
        n_bcrank = quality.get('n_cells') if quality else None
        if n_bcrank and n_out is not None:
            pct_bc = 100.0 * float(n_out) / float(n_bcrank)
            removed_bc = int(n_bcrank) - int(n_out)
            bcrank_bar = ('<div class="bar"><span style="width:{pct:.1f}%"></span></div>'
                          '<p class="sub">{out} of {inp} putative droplets (cells surviving '
                          'barcode-rank filtering) retained ({pct:.1f}%); {rem} removed.</p>').format(
                pct=pct_bc, out=_fmt_int(n_out), inp=_fmt_int(n_bcrank), rem=_fmt_int(removed_bc))

        note = ('<p class="note">Filters run as a cascade: each stage only sees cells the '
                'previous stages passed, so these counts are relative to each stage\'s own '
                'input and <strong>do not sum</strong> to the total removed. Where a stage '
                'flags on several features those flags overlap, and the overlap is not '
                'recorded &mdash; such stages show the range the true count lies within '
                '(largest single feature to the sum of all of them).</p>')

        # Per-method detail table.
        detail_rows = []
        any_advisory = False
        for method, feature, meta in diagnostics:
            advisory = is_advisory(meta)
            any_advisory = any_advisory or advisory
            if meta.get('failed'):
                # A fault, not a decision. Rendered distinctly from a skip
                # because the two demand opposite responses: a skip is the
                # pipeline working as designed and needs nothing from anyone, a
                # fault means this step never ran and someone has to look at it.
                threshold = ('<span class="missing"><strong>FAILED</strong> &mdash; {}</span>'.format(
                    html.escape(str(meta.get('failure_reason', 'no reason recorded')))))
                flagged = '<span class="missing">n/a</span>'
            elif meta.get('skipped'):
                threshold = '<span class="missing">not filtered &mdash; {}</span>'.format(
                    html.escape(str(meta.get('skip_reason', 'no reason recorded'))))
                # "0" would read as "nothing was an outlier"; nothing was tested.
                flagged = '<span class="missing">n/a</span>'
            else:
                threshold = describe_threshold(method, meta)
                # meta['mode'] is mad's own adaptive/absolute distinction, not
                # the execution mode -- see meta['exec_mode'] for that. The
                # 'absolute' method already labels its rows "Absolute cutoff", so
                # this only annotates a stray absolute-mode entry under 'mad'.
                if method != 'absolute' and meta.get('mode') == 'absolute':
                    threshold += ' <small>(absolute)</small>'
                flagged = _fmt_int(meta.get('n_flagged', 0))
            # An advisory method's count is cells it *would* have removed. Saying
            # so in the cell matters: the number is otherwise identical in shape
            # to a filtering method's, and the whole point of advisory mode is
            # that those cells are still in the object.
            if advisory and not meta.get('skipped'):
                flagged = '{} <small>(advisory)</small>'.format(flagged)
            # Not `note`: that name holds the cascade explanation below.
            selection = describe_selection(method, meta)
            label = html.escape(_METHOD_LABELS.get(method, method))
            if advisory:
                # A row downgraded by sample_quality reads very differently from
                # one the config deliberately set to advisory: the first is a
                # threshold that exists but should not be trusted, the second is
                # a diagnostic nobody intended to act on.
                label += (' <span class="pill advisory">downgraded</span>'
                          if meta.get('downgraded_from') else
                          ' <span class="pill advisory">advisory</span>')
            detail_rows.append(
                '<tr><td>{m}</td><td><code>{f}</code></td><td>{t}</td>'
                '<td class="num">{n}</td></tr>'.format(
                    m=label,
                    f=html.escape(str(feature or '')),
                    t=threshold,
                    n=flagged))
            if selection:
                detail_rows.append(
                    '<tr><td></td><td colspan="3" class="sub" style="font-size:.8rem">{}</td></tr>'.format(selection))

        advisory_note = ''
        if any_advisory:
            advisory_note = (
                '<p class="note"><strong>Advisory</strong> methods ran and recorded their scores, '
                'thresholds and figures, but removed no cells &mdash; their counts below are what '
                '<em>would</em> have been flagged, and every one of those cells is still present in '
                'the filtered object. They are excluded from the removal waterfall above for that '
                'reason. A method is advisory either because <code>mode: advisory</code> is set for '
                'it in filter_config.yaml, or because <code>sample_quality</code> flagged this '
                'sample and downgraded it &mdash; rows below say which.</p>')

        detail = ('<h3>Filter detail</h3>{note}<table>'
                  '<tr><th>Method</th><th>Feature</th><th>Threshold</th>'
                  '<th class="num">Cells flagged</th></tr>{rows}</table>'.format(
                      note=advisory_note, rows=''.join(detail_rows)))

        reg_note = ''
        if any(m == 'regression' for m, _, _ in diagnostics):
            reg_note = ('<p class="note">Regression thresholds are offsets in residual space, '
                        'not cut-offs on the raw feature.</p>')

        return ('<h2>Quality control</h2>{bar}{bcrank_bar}{note}'
                '<table><tr><th class="num">Stage</th><th>Method</th>'
                '<th class="num">Cells flagged</th>'
                '<th>Per-feature</th></tr>{rows}</table>{detail}{reg}{figs}').format(
            bar=bar, bcrank_bar=bcrank_bar, note=note, rows=''.join(rows), detail=detail, reg=reg_note,
            figs=self._qc_figures(diagnostics, n_input=n_in))

    def _qc_figures(self, diagnostics: list, n_input=None) -> str:
        """Every QC figure present on disk, grouped by method in cascade order."""
        seen, blocks = [], []
        for method, _, meta in diagnostics:
            if method in seen:
                continue
            seen.append(method)
            prefix = _METHOD_FIGURE_PREFIX.get(method)
            if not prefix:
                continue
            figures = sorted(self.qc_dir.glob('{}*.png'.format(prefix)))
            if not figures:
                continue
            label = _METHOD_LABELS.get(method, method)
            if is_advisory(meta):
                label += ' <span class="pill advisory">advisory</span>'
            block = '<h3>{}</h3>{}{}'.format(
                label,
                describe_view(meta, n_input),
                ''.join(_figure(f, figure_caption(f, method)) for f in figures))
            # The doublet figures reuse the scVI UMAP; note how it was built,
            # when one exists (only when SOLO ran).
            if method in ('solo', 'scrublet', 'doublet_consensus') and self._umap_params.get('X_umap_scVI'):
                block += _umap_param_note(self._umap_params.get('X_umap_scVI', {}), intro='scVI UMAP')
            blocks.append(block)
        return ''.join(blocks)

    def _section_umap(self, provenance: dict) -> str:
        """The final UMAP, or the recorded reason it was not computed.

        'computed' is trusted only when the key exists: an artifact written before
        qc_plot recorded this has no entry, and defaulting to False would report every
        older sample as skipped. `provenance` is unused; it keeps the section
        signatures uniform.
        """
        # Say so, rather than rendering an empty figure slot that reads as a
        # missing file.
        if self._umap_status and not self._umap_status.get('computed', True):
            reason = str(self._umap_status.get('skip_reason', '') or 'no reason recorded')
            return (
                '<h2>Final UMAP</h2>'
                '<p class="banner"><strong>Not computed.</strong> {}. Everything else in this '
                'report &mdash; the filter diagnostics, their figures and the filtered object '
                '&mdash; is unaffected, but there is no embedding for this sample, so '
                '<code>X_umap_qc</code> and <code>leiden_qc</code> are absent from the filtered '
                'artifact and the interactive section below cannot lasso or select by cluster. '
                'For a sample this small that is usually the correct outcome rather than a fault: '
                'check the QC cascade banner above for what removed the cells.</p>').format(
                    html.escape(reason))
        caption = 'UMAP of retained cells, coloured by Leiden cluster'
        return '<h2>Final UMAP</h2>{}{}{}'.format(
            _figure(self.qc_dir / 'filtered_final_umap.png', caption),
            '<p class="note">Computed on the filtered matrix after HVG selection and PCA. '
            'The embedding is not persisted &mdash; this figure is the only record.</p>',
            _umap_param_note(self._umap_params.get('X_umap_qc', {}), intro='PCA UMAP'))

    def _section_doublet_failure_banner(self) -> str:
        """Surface a SOLO *fault* at the top of the report, not just as a table row.

        Only fires for `failed`, never for a deliberate skip: a skip is the pipeline
        working as designed and a table row is the right weight for it.
        """
        solo = self._solo
        if not solo.get('failed'):
            return ''
        return (
            '<div class="banner"><p><strong>Doublet detection failed for this sample.</strong> '
            'SOLO did not run to completion, so there are no doublet scores, no calls and no '
            'scVI embedding here. Every other part of this report is unaffected and describes '
            'the sample accurately.</p>'
            '<p>This is a fault rather than a deliberate skip: the run exited non-zero because '
            'of it, <code>doublets.done</code> was not written, and re-running the mapping stage '
            'will retry this sample without <code>--force-doublets</code>.</p>'
            '<p class="note">Reason: <code>{}</code></p></div>').format(
                html.escape(str(solo.get('failure_reason', 'no reason recorded'))))

    def _section_post_cascade_banner(self) -> str:
        """Banner when too few cells survived the whole cascade; '' otherwise."""
        pc = self._post_cascade
        if not pc:
            return ''
        n = pc.get('n_cells_surviving')
        total = pc.get('n_cells_input')
        floor = pc.get('min_cells_post_cascade')
        if n is None or pc.get('viable', True):
            return ''

        if n == 0:
            body = (
                '<strong>Every cell was removed.</strong> All {total} barcodes that entered QC '
                'were flagged by the cascade below. The filtered artifact for this sample is '
                'empty: there is nothing to embed, nothing to call doublets on, and nothing for '
                'integration to use.'.format(total=total))
        else:
            body = (
                '<strong>Only {n} of {total} cells survived the full QC cascade</strong>, below '
                'the {floor}-cell viability floor. Doublet detection was skipped and the final '
                'UMAP may be absent. This is a very small population to carry into '
                'integration.'.format(n=n, total=total, floor=floor))

        return (
            '<div class="banner"><p>{body}</p>'
            '<p>This is reported, not corrected. The per-filter table below shows which filters '
            'did the removing &mdash; if the absolute <code>max_value</code> caps in '
            '<code>filter_config.yaml</code> (mitochondrial, chloroplast and ribosomal RNA '
            'fractions) account for most of it, that is the pipeline judging this a poor prep '
            'rather than a threshold misfiring, and the caps are working as intended. Confirm '
            'against the fraction distributions before overriding anything.</p></div>').format(
                body=body)

    def _section_scvi_integration(self) -> str:
        """Link the single-sample integration figures, when they describe this artifact.

        Guarded on the config hash, since the figures another stage left in `qc/` may
        describe an older run.
        """
        summary_path = self.qc_dir / '{}_scvi_summary.json'.format(self.sample_name)
        summary = {}
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    summary = json.load(f)
            except (OSError, ValueError) as e:
                self.logger.warning('[REPORT] Could not read scVI summary for "{}": {}'.format(self.sample_name, e))

        # X_umap_scVI for a scVI run, X_umap_scArches for a 1-sample scArches
        # query -- both land in this sample's own qc/ dir the same way.
        figures = sorted(self.qc_dir.glob('plot_umap_*_X_umap_scVI.png')) + \
                  sorted(self.qc_dir.glob('plot_umap_*_X_umap_scArches.png')) + \
                  sorted(self.qc_dir.glob('plot_solo_doublet_score_hist.png'))
        if not summary and not figures:
            return ''

        # These files are produced by a *different stage*, with its own done
        # flag, and are discovered by globbing -- so nothing here can tell a
        # figure computed from the current QC output from one an integration run
        # left behind weeks ago. Left unchecked the report presents stale scVI
        # results as current (e.g. a `plot_umap_solo_doublet_*.png` from an older
        # SOLO run passing for this run's doublet calls). Compare the hash
        # report_embedding.py stamps into the sidecar.
        stale = ''
        recorded = str(summary.get('parent_config_hash', '') or '')
        current = str(self._config_hash or '')
        if not summary or not recorded:
            stale = ('<p class="banner">These figures were found in this sample\'s <code>qc/</code> '
                     'directory but carry no record of which mapping output they were computed '
                     'from &mdash; they predate that being tracked, so they may be from an earlier '
                     'run and may not describe the cells in the current filtered object. Re-run '
                     'the single-sample integration step to refresh them.</p>')
        elif current and recorded != current:
            stale = ('<p class="banner"><strong>Stale.</strong> These were computed from a '
                     'different version of this sample\'s mapping output (config hash '
                     '<code>{}</code>, current is <code>{}</code>). QC has been re-run since, so '
                     'the cells shown below are not the cells in the current filtered object. '
                     'Re-run the single-sample integration step.</p>').format(
                         html.escape(recorded[:12]), html.escape(current[:12]))

        cards = []
        if summary:
            doublet_pct = ('{:.1f}%'.format(100 * summary['doublet_fraction'])
                           if summary.get('doublet_fraction') is not None else '&mdash;')
            cards = [
                ('Model', _esc(summary.get('model_class'))),
                ('Latent dims', _fmt_int(summary.get('n_latent'))),
                ('Cells', _fmt_int(summary.get('n_cells'))),
                ('Doublets (SOLO)', '{} <small>({})</small>'.format(
                    _fmt_int(summary.get('n_doublets')), doublet_pct))]
        cards_html = ''.join(
            '<div class="card"><div class="k">{}</div><div class="v">{}</div></div>'.format(k, v)
            for k, v in cards)
        figs_html = (''.join(_figure(f, figure_caption(f)) for f in figures)
                     if figures else '<p class="missing">UMAP not available.</p>')
        return ('<h2>Single-sample scVI integration</h2>{stale}'
                '<div class="cards">{cards}</div>{figs}').format(
            stale=stale, cards=cards_html, figs=figs_html)

    # -- entrypoint ---------------------------------------------------------

    @log_exceptions()
    def run(self, force: bool=False) -> None:
        """Build the report and write it, unless `report.done` exists.

        The interactive block is guarded separately: it reads a second, much larger
        artifact, and losing the static report to a fault in the interactive extra
        would be a poor trade.
        """
        done_path = self.paths['done'] / 'report.done'
        if done_path.exists() and not force:
            self.logger.info('[REPORT] skipping sample "{}"; report.done exists (use --force-report to override)'.format(self.sample_name))
            return

        self.logger.info('[REPORT] Building QC report for sample "{}"'.format(self.sample_name))
        provenance, diagnostics, stats, quality = self._load_artifacts()

        interactive = ''
        try:
            payload = self._interactive_payload(provenance)
            if payload is not None:
                interactive = render_section(payload)
        except Exception as e:
            self.logger.warning('[REPORT] Could not build the interactive section for "{}": {} '
                                '(the rest of the report is unaffected)'.format(self.sample_name, e))

        body = (self._section_header(provenance)
                + self._section_quality_banner(quality)
                + self._section_post_cascade_banner()
                + self._section_doublet_failure_banner()
                + self._section_mapping(stats)
                + self._section_qc(provenance, diagnostics, quality)
                + self._section_umap(provenance)
                + self._section_scvi_integration()
                + interactive)

        page = _PAGE.format(title='{} - mapping QC'.format(html.escape(self.sample_name)),
                            css=_CSS + interactive_css(), body=body)
        self.report_path.write_text(page, encoding='utf-8')

        size_mb = self.report_path.stat().st_size / (1024 * 1024)
        self.logger.info('[REPORT] Wrote "{}" ({:.1f} MB)'.format(self.report_path, size_mb))
        write_done_flag(done_path)


def _read_manifests(out_path: Path) -> dict:
    """Discover completed samples the way any downstream consumer should.

    Globs the per-sample manifests rather than walking the sample config, so the
    index reflects what the run actually produced. A manifest is only written on
    success, so its absence is itself the failure signal. Returns
    {sample: (manifest, sample_dir)}.
    """
    logger = logging.getLogger('SingleCellMappingPipeline')
    manifests = {}
    for path in sorted(Path(out_path).glob('*/manifest.json')):
        try:
            with open(path) as f:
                manifest = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning('[REPORT] Could not read manifest "{}": {}'.format(path, e))
            continue
        manifests[manifest.get('sample', path.parent.name)] = (manifest, path.parent)
    return manifests


def write_run_index(out_path: Path, samples: dict, failures: list) -> None:
    """Write a run-level index linking to every per-sample report.

    Links rather than embeds: at 40 samples an index carrying the figures would be
    far too large to open. `failures` is (sample, error) pairs to mark as failed.
    """
    logger = logging.getLogger('SingleCellMappingPipeline')
    out_path = Path(out_path)
    manifests = _read_manifests(out_path)
    failed = {name for name, _ in (failures or [])}

    rows = []
    for sample_name in samples:
        entry = manifests.get(sample_name)

        if entry is None:
            status = ('<span class="pill bad">failed</span>' if sample_name in failed
                      else '<span class="pill bad">no manifest</span>')
            rows.append('<tr><td>{s}</td><td>{st}</td><td class="num">&mdash;</td>'
                        '<td class="num">&mdash;</td><td class="num">&mdash;</td>'
                        '<td class="missing">no report</td></tr>'.format(
                            s=html.escape(str(sample_name)), st=status))
            continue

        manifest, sample_dir = entry
        mapper = manifest.get('mapper', '')
        n_in = n_out = None

        # Cell counts live in the artifact's provenance, not the manifest; read
        # them back rather than duplicating them into a second source of truth.
        h5ad_rel = (manifest.get('artifacts') or {}).get('filtered_h5ad')
        if h5ad_rel:
            try:
                adata = sc.read_h5ad(sample_dir / h5ad_rel, backed='r')
                prov = dict(adata.uns.get('pipeline', {}))
                n_in, n_out = prov.get('n_cells_input'), prov.get('n_cells')
                if hasattr(adata, 'file') and adata.file is not None:
                    adata.file.close()
            except Exception as e:
                logger.warning('[REPORT] Could not read counts for "{}": {}'.format(sample_name, e))

        retained = '&mdash;'
        if n_in and n_out is not None:
            retained = '{:.1f}%'.format(100.0 * float(n_out) / float(n_in))

        report_rel = (manifest.get('artifacts') or {}).get('qc_report')
        if report_rel:
            link = '<a href="{}">report</a>'.format(
                html.escape('{}/{}'.format(sample_name, report_rel)))
        else:
            link = '<span class="missing">no report</span>'

        rows.append(
            '<tr><td>{s}<br><small>{m}</small></td><td>{st}</td><td class="num">{i}</td>'
            '<td class="num">{o}</td><td class="num">{r}</td><td>{l}</td></tr>'.format(
                s=html.escape(str(sample_name)), m=html.escape(str(mapper)),
                st='<span class="pill ok">{}</span>'.format(
                    html.escape(str(manifest.get('status', 'complete')))),
                i=_fmt_int(n_in), o=_fmt_int(n_out), r=retained, l=link))

    # manifests can hold leftover entries from samples that aren't part of this
    # run at all (anything ever completed under out_path) -- scope the count to
    # this run's own sample list, matching the per-row loop above, rather than
    # reporting a total that isn't comparable to n_total.
    n_ok = sum(1 for sample_name in samples if sample_name in manifests)
    n_total = len(samples)
    body = """
<h1>Mapping run</h1>
<p class="sub">{ok} of {total} samples completed &middot; <code>{path}</code></p>
<table>
  <tr><th>Sample</th><th>Status</th><th class="num">Cells in</th>
      <th class="num">Cells retained</th><th class="num">%</th><th>Report</th></tr>
  {rows}
</table>
<p class="note">A sample appears here as complete only if it wrote a manifest, which
happens solely on success. Cell counts are read from each sample's filtered AnnData.</p>
""".format(ok=n_ok, total=n_total, path=html.escape(str(out_path)), rows=''.join(rows))

    index_path = out_path / 'index.html'
    index_path.write_text(_PAGE.format(title='Mapping run index', css=_CSS, body=body),
                          encoding='utf-8')
    logger.info('[REPORT] Wrote run index "{}" ({}/{} samples complete)'.format(
        index_path, n_ok, n_total))
