#!/usr/bin/env python3

"""
The interactive QC section: select cells by hand, export a decision file.

Renders into the same self-contained HTML report the rest of the mapping stage
produces. Everything is inlined -- no CDN, no server, no build step -- because
the report's whole value is that it can be copied off the cluster and opened
anywhere, and because there is no dependency manifest in this repo to add
plotly or bokeh to (neither is installed in the mapping env in any case).

What the user does here:

  * drag a threshold on any numeric QC feature's histogram
  * lasso a region of the QC or scVI UMAP
  * click a Leiden cluster to select it whole

...and export `{sample}_decisions.json`, which
`python -m mapping.refilter` then applies to produce
`{sample}_filtered_manual.h5ad`.

Two things the payload is careful about:

**Raw units.** The automatic cascade thresholds several features in transformed
space (`log_`/`logit_` columns), but a human reads and reasons in the raw
fraction or count. Only untransformed columns are offered here, and the decision
file records raw values, which is what refilter applies.

**Cells the model never saw.** Advisory methods run on their own cascade view,
so a cell removed upstream has no GMM or SOLO score. Those are NaN in the
payload and are drawn as "not assessed" rather than at zero, which is what a
naive fillna would silently do.
"""

import html
import json

import numpy as np
import pandas as pd

from .qc_filter import (collect_removal_flags, feature_transform, invert_transform,
                        LOGIT_CLIP)

# Columns that exist for the machine, not for a person choosing a cut-off:
# nobody sets a threshold in logit units or standard deviations. The raw column
# is offered instead; the methods' own scores are kept, being what the figures
# show.
_HIDDEN_PREFIXES = ('flag_', 'rank_flag_', 'log_', 'logit_', 'z_')

# .obs columns that are labels rather than measurements.
_LABEL_COLUMNS = ('sample', 'batch', 'barcode', 'leiden_qc', 'leiden_scVI', 'cellbender_analyzed')

_MAX_POINTS = 60000


# Keys under which a .uns['qc'] entry names a per-cell score column it produced.
# These have figures in the report and are offered for thresholding alongside
# the features the filters were configured on.
_SCORE_KEYS = ('score_column', 'distance_col', 'posterior_col', 'change_column', 'overlap_column')


def as_list(value) -> list:
    """Coerce a `.uns` sequence field to a plain list.

    anndata round-trips a list stored in `.uns` as a numpy array, so `value or []`
    raises on its truthiness. Convert first, test after.

    Args:
        value: The `.uns` field.

    Returns:
        list: The values, or [] when absent.
    """
    if value is None:
        return []
    return list(value)


def _qc_features(uns_qc: dict) -> set:
    """The `.obs` columns the QC pipeline actually looked at.

    Everything in `.obs` is available, but most is bookkeeping, and offering a
    threshold control for each buries the handful that correspond to a filter
    someone can reason about. Reads `source_feature` in preference to `feature`, so
    a MAD entry keyed by a transformed column contributes the raw one a person can
    set a bound on.

    Args:
        uns_qc (dict): The `.uns['qc']` mapping.

    Returns:
        set[str]: Column names worth offering.
    """
    wanted = set()
    for method, entry in (uns_qc or {}).items():
        if method == 'sample_quality' or not isinstance(entry, dict):
            continue
        payloads = entry.values() if (entry and all(isinstance(v, dict) for v in entry.values())) else [entry]
        for meta in payloads:
            if not isinstance(meta, dict):
                continue
            wanted.add(meta.get('source_feature') or meta.get('feature'))
            wanted.update(as_list(meta.get('features')))
            if meta.get('complexity_key'):
                wanted.add(meta['complexity_key'])
            for key in _SCORE_KEYS:
                if meta.get(key):
                    wanted.add(meta[key])
    return {w for w in wanted if w}


def _numeric_features(obs: pd.DataFrame, uns_qc: dict = None) -> list:
    """Numeric `.obs` columns worth offering as a manual threshold.

    Restricted to what QC actually used when the diagnostics are available, falling
    back to every numeric column when they are not, so an artifact predating
    `source_feature` still gets a usable panel. Transformed columns are excluded
    either way -- offering both spaces side by side is how someone types a raw
    fraction into a logit-scaled control.

    Args:
        obs (pandas.DataFrame): The `.obs` frame.
        uns_qc (dict, optional): The `.uns['qc']` mapping.

    Returns:
        list[str]: Column names, sorted.
    """
    wanted = _qc_features(uns_qc) if uns_qc else set()
    out = []
    for col in obs.columns:
        if col in _LABEL_COLUMNS or col.startswith(_HIDDEN_PREFIXES):
            continue
        if wanted and col not in wanted:
            continue
        values = pd.to_numeric(obs[col], errors='coerce')
        if values.notna().sum() < 2 or values.nunique(dropna=True) < 2:
            continue
        out.append(col)
    return sorted(out)


def _clean(values) -> list:
    """JSON-safe floats: NaN/inf become null, which the page renders as 'not assessed'.

    Args:
        values: Numeric sequence.

    Returns:
        list: Rounded floats and Nones.
    """
    arr = np.asarray(values, dtype=float)
    return [None if not np.isfinite(v) else round(float(v), 6) for v in arr]


def build_payload(adata, sample: str, config_hash: str = '') -> dict:
    """Assemble everything the page needs. Pure read; nothing is recomputed.

    Args:
        adata (AnnData): The pre-filter object, usually opened backed.
        sample (str): Sample name.
        config_hash (str): Stamped into the payload so a decision file can be tied
            to the config that produced the cells it addresses.

    Returns:
        dict: Features, embeddings, clusterings, bounds and QC plot specs.
    """
    obs = adata.obs
    uns_qc = adata.uns.get('qc', {})
    features = _numeric_features(obs, uns_qc)

    # The plots are built first because they decide what data has to ship.
    # `_HIDDEN_PREFIXES` keeps transformed columns out of any *chooser*, which is
    # right -- nobody sets a cut-off in logit units. But the regression figure is
    # plotted in exactly that transformed space (log counts vs log genes), so
    # excluding those columns from the payload left its canvas blank: the plot
    # was offered and had no data behind it. Anything a plot names is carried,
    # whether or not it is offerable on its own.
    plots = _qc_plots(uns_qc, obs, _automatic_bounds(uns_qc, None))
    for pl in plots:
        for axis in (pl.get('x'), pl.get('y')):
            if axis and axis in obs.columns and axis not in features:
                features.append(axis)
    bcrank_meta = (uns_qc or {}).get('bcrank') or {}
    for meta in (bcrank_meta.values() if isinstance(bcrank_meta, dict) else []):
        col = isinstance(meta, dict) and (meta.get('source_feature') or meta.get('feature'))
        if col and col in obs.columns and col not in features:
            features.append(col)

    payload = {
        'sample': sample,
        'parent_config_hash': config_hash or '',
        # Shipped rather than repeated in the JS: the page's logit axis has to
        # clip exactly where _transform_features clips, or it places a cell
        # somewhere the filter never saw it.
        'logit_clip': float(LOGIT_CLIP),
        'n_cells': int(adata.n_obs),
        'barcodes': [str(b) for b in obs.index],
        'features': {f: _clean(pd.to_numeric(obs[f], errors='coerce').values) for f in features},
        'embeddings': {},
        'clusterings': {},
    }

    # scVI first: it is the better-conditioned embedding for judging whether a
    # blob is a real population, so it should be what the dropdown opens on.
    # Insertion order is preserved through json.dumps and drives the <option>
    # order in render_section. The QC UMAP is not a fallback in the trivial
    # sense -- on a sample whose doublet step skipped (no model, hence no scVI
    # embedding) it is the only one there is.
    for key, label in (('X_umap_scVI', 'scVI UMAP'), ('X_umap_qc', 'QC UMAP')):
        if key in adata.obsm:
            coords = np.asarray(adata.obsm[key], dtype=float)
            if coords.ndim == 2 and coords.shape[1] >= 2:
                payload['embeddings'][key] = {
                    'label': label,
                    'x': _clean(coords[:, 0]),
                    'y': _clean(coords[:, 1])}

    for key in ('leiden_qc', 'leiden_scVI'):
        if key in obs.columns:
            labels = obs[key].astype(str).fillna('').tolist()
            if any(v not in ('', 'nan') for v in labels):
                payload['clusterings'][key] = labels

    # Cells the automatic cascade already removes, so the page can show what a
    # manual decision is being layered on top of rather than implying it starts
    # from every barcode.
    removal_flags = collect_removal_flags(uns_qc)
    removed = np.zeros(adata.n_obs, dtype=bool)
    for flag in removal_flags:
        if flag in obs:
            removed |= obs[flag].astype(bool).values
    payload['auto_removed'] = [bool(v) for v in removed]

    payload['bounds'] = _automatic_bounds(uns_qc, features)
    payload['plots'] = plots

    # Removals the page cannot reconstruct from a feature and a bound: the
    # multivariate filters, and anything whose entry carries no source_feature.
    # The derivable ones are left out on purpose -- the page recomputes those
    # from `bounds`, which is what lets it *project* the effect of replacing one
    # of them with a hand-set cut. Without this split the panel could only ever
    # show the additive result, which for a loosened bound is no change at all.
    derivable = {b['flag_col'] for bl in payload['bounds'].values() for b in bl
                 if b.get('exec_mode') == 'filter' and b.get('flag_col')}
    other = np.zeros(adata.n_obs, dtype=bool)
    for flag in removal_flags:
        if flag in obs and flag not in derivable:
            other |= obs[flag].astype(bool).values
    payload['auto_removed_other'] = [bool(v) for v in other]
    payload['bcrank'] = _bcrank_curve(obs, uns_qc)
    # The SOLO/Scrublet call overlap, when both ran. Positional indices into
    # `barcodes`, turned into an exclude_barcodes decision on click -- so the
    # reviewer can remove the consensus (the lens) or either method's exclusive
    # calls without touching refilter, which already applies exclude_barcodes.
    payload['doublets'] = _doublet_venn(obs, uns_qc)
    if payload['doublets']:
        # Offered as an entry in the same QC-plot dropdown as the histograms and
        # scatters, drawn on the shared canvas -- not a standalone widget. `kind`
        # 'venn' is what drawHist dispatches on; the region data lives in
        # payload['doublets']. Not marked advisory: it is a manual selection
        # tool, not a filter, so the '(advisory)' dropdown suffix would mislead.
        payload['plots'].append({
            'kind': 'venn', 'method': 'doublet_consensus',
            'label': 'Doublet call overlap — SOLO ∩ Scrublet',
            'advisory': False, 'flag_col': None, 'downgraded': False})
    return payload


def _qc_plots(uns_qc: dict, obs, bounds: dict) -> list:
    """One entry per QC figure in the report, in the form that figure takes.

    Mirrors `qc_plot.py`'s plot dispatch -- same axes, same colouring, same
    thresholds -- because a reviewer is looking at *the miQC scatter* or *the
    regression fit* and wants to select on that, not on a generic histogram.
    `kind` drives the interaction: 'hist' gets a draggable cut-off, 'scatter' gets
    lasso selection, since a joint fit has no single axis to threshold on.

    Args:
        uns_qc (dict): The `.uns['qc']` mapping.
        obs (pandas.DataFrame): The `.obs` frame.
        bounds (dict): As returned by `_automatic_bounds`.

    Returns:
        list[dict]: One spec per panel.
    """
    out = []

    def col_ok(*names):
        """True when every named column is present in `.obs`."""
        return all(n and n in obs.columns for n in names)

    for method, entry in (uns_qc or {}).items():
        if method == 'sample_quality' or not isinstance(entry, dict):
            continue
        payloads = (entry.items() if (entry and all(isinstance(v, dict) for v in entry.values()))
                    else [('', entry)])
        for key, meta in payloads:
            if not isinstance(meta, dict) or meta.get('skipped'):
                continue
            flag = meta.get('flag_col') if meta.get('flag_col') in obs.columns else None
            advisory = meta.get('exec_mode', 'filter') != 'filter'
            common = {'method': method, 'flag_col': flag, 'advisory': advisory,
                      'downgraded': bool(meta.get('downgraded_from'))}

            if method in ('mad', 'absolute'):
                feat = meta.get('source_feature')
                if not col_ok(feat):
                    continue
                # One fraction feature produces two histograms -- the adaptive
                # rule and the fixed cap -- and feature_transform yields the right
                # axis for each, so each panel matches its report figure. Values
                # and bounds stay raw; only the spacing is transformed.
                label = ('Absolute cutoff \u2014 {}'.format(feat) if method == 'absolute'
                         else 'MAD (adaptive) \u2014 {}'.format(feat))
                out.append(dict(common, kind='hist', x=feat,
                                scale=feature_transform(meta.get('feature')),
                                label=label,
                                bounds=bounds.get(feat, [])))

            elif method == 'miqc':
                feat, comp = (meta.get('feature') or key), meta.get('complexity_key')
                if not col_ok(feat, comp):
                    continue
                line = meta.get('intact_line') or {}
                out.append(dict(common, kind='scatter', x=comp, y=feat,
                                label='miQC \u2014 {} vs complexity'.format(feat),
                                line={'slope': line.get('slope'), 'intercept': line.get('intercept')}))

            elif method == 'regression':
                feats = as_list(meta.get('features'))
                if len(feats) < 2 or not col_ok(feats[0], feats[1]):
                    continue
                # thresholds are *residual offsets* from the fitted line, not
                # cut-offs on either axis -- the report figure draws them as two
                # lines parallel to the fit, and so does the interactive version.
                band = meta.get('thresholds') if isinstance(meta.get('thresholds'), dict) else {}
                out.append(dict(common, kind='scatter', x=feats[0], y=feats[1],
                                label='Regression \u2014 {} ~ {}'.format(feats[1], feats[0]),
                                line={'slope': meta.get('coef'), 'intercept': meta.get('intercept'),
                                      'lower': band.get('lower'), 'upper': band.get('upper')}))

            elif method == 'gmm':
                score = meta.get('distance_col')
                if not col_ok(score):
                    continue
                cutoff = meta.get('thresholds')
                out.append(dict(common, kind='hist', x=score,
                                label='GMM \u2014 Mahalanobis distance',
                                cutoff=float(cutoff) if isinstance(cutoff, (int, float)) else None,
                                bounds=bounds.get(score, [])))

            elif method == 'iforest':
                score = meta.get('score_column')
                if not col_ok(score):
                    continue
                cutoff = meta.get('threshold')
                out.append(dict(common, kind='hist', x=score,
                                label='Isolation forest \u2014 score',
                                cutoff=float(cutoff) if cutoff is not None else None,
                                bounds=bounds.get(score, [])))

            elif method == 'dod':
                chg, ov = meta.get('change_column', 'dod_change'), meta.get('overlap_column', 'dod_overlap')
                if not col_ok(chg, ov):
                    continue
                out.append(dict(common, kind='hist', x=chg,
                                label='DoD \u2014 change in distance',
                                cutoff=None, bounds=bounds.get(chg, [])))
                out.append(dict(common, kind='hist', x=ov,
                                label='DoD \u2014 neighbourhood overlap',
                                cutoff=meta.get('overlap_threshold'),
                                bounds=bounds.get(ov, [])))
                out.append(dict(common, kind='scatter', x=ov, y=chg,
                                label='DoD \u2014 change vs neighbourhood overlap', line={}))

            elif method in ('solo', 'scrublet'):
                score = meta.get('score_column')
                if not col_ok(score):
                    continue
                cutoff = meta.get('threshold')
                label = ('SOLO \u2014 doublet score' if method == 'solo'
                         else 'Scrublet \u2014 doublet score')
                out.append(dict(common, kind='hist', x=score,
                                label=label,
                                cutoff=float(cutoff) if cutoff is not None else None,
                                bounds=bounds.get(score, [])))
            # doublet_consensus has no single score to threshold (it is the
            # boolean intersection of the other two), so it is intentionally not
            # offered here -- a reviewer adjusts SOLO or Scrublet, not the AND.
    return out


def _automatic_bounds(uns_qc: dict, features=None) -> dict:
    """Per raw feature, the bounds the pipeline currently applies to it.

    Two consumers: the dotted lines in the histograms need the values, and the
    loosening guard needs to know which automatic flag columns a hand-set threshold
    would be arguing with. Only entries carrying `source_feature` appear -- miQC,
    regression, GMM, iForest and DoD are absent by design, being joint fits that a
    cut on one input does not supersede.

    Args:
        uns_qc (dict): The `.uns['qc']` mapping.
        features (set[str], optional): Restrict to these raw features.

    Returns:
        dict: {raw feature: [{'method', 'flag_col', 'exec_mode', 'downgraded',
            'lower', 'upper'}, ...]}.
    """
    out = {}
    for method, entry in (uns_qc or {}).items():
        if method == 'sample_quality' or not isinstance(entry, dict):
            continue
        payloads = entry.values() if (entry and all(isinstance(v, dict) for v in entry.values())) else [entry]
        for meta in payloads:
            if not isinstance(meta, dict) or meta.get('skipped'):
                continue
            source = meta.get('source_feature')
            if not source or (features is not None and source not in features):
                continue
            lower, upper = _raw_bounds(method, meta)
            if lower is None and upper is None:
                continue
            out.setdefault(source, []).append({
                'method': method,
                'flag_col': meta.get('flag_col', ''),
                'exec_mode': meta.get('exec_mode', 'filter'),
                'downgraded': bool(meta.get('downgraded_from')),
                'lower': lower,
                'upper': upper})
    return out


def _raw_bounds(method: str, meta: dict):
    """One method's lower/upper bound for a feature, on the RAW scale.

    MAD's adaptive bounds live in transformed space, so they are inverted back --
    a line drawn at the logit value would sit in completely the wrong place on a
    raw axis.

    Args:
        method (str): Filter name.
        meta (dict): That method's `.uns['qc']` entry.

    Returns:
        tuple: (lower, upper), either possibly None.
    """
    if method == 'bcrank':
        t = meta.get('threshold')
        return (float(t), None) if t is not None else (None, None)

    bounds = meta.get('thresholds')
    if not isinstance(bounds, dict):
        return (None, None)

    kind = feature_transform(str(meta.get('feature') or ''))
    try:
        return (invert_transform(bounds.get('lower'), kind),
                invert_transform(bounds.get('upper'), kind))
    except (TypeError, ValueError, OverflowError):
        return (None, None)


def _bcrank_curve(obs: pd.DataFrame, uns_qc: dict, max_points: int = 4000) -> dict:
    """The barcode-rank curve and its knee, downsampled for the page.

    Args:
        uns_qc (dict): The `.uns['qc']` mapping.

    Returns:
        dict or None: Curve points and threshold, or None when bcrank did not run.
    """
    entry = (uns_qc or {}).get('bcrank') or {}
    metas = [v for v in entry.values() if isinstance(v, dict)] if entry else []
    meta = next((m for m in metas if not m.get('skipped')), None)
    if meta is None:
        return {}
    feature = meta.get('source_feature') or meta.get('feature')
    if not feature or feature not in obs.columns:
        return {}

    counts = pd.to_numeric(obs[feature], errors='coerce').values
    counts = np.sort(counts[np.isfinite(counts) & (counts > 0)])[::-1]
    if counts.size == 0:
        return {}

    n = counts.size
    if n > max_points:
        idx = np.unique(np.logspace(0, np.log10(n), max_points).astype(int) - 1)
        idx = idx[(idx >= 0) & (idx < n)]
    else:
        idx = np.arange(n)

    return {
        'feature': feature,
        'flag_col': meta.get('flag_col', ''),
        'threshold': float(meta['threshold']) if meta.get('threshold') is not None else None,
        'selection_policy': str(meta.get('selection_policy', '')),
        'n_total': int(n),
        'rank': [int(i) + 1 for i in idx],
        'count': [float(counts[i]) for i in idx]}


def _method_advisory(uns_qc: dict, method: str) -> bool:
    """Map each method to whether it removed cells.

    Args:
        uns_qc (dict): The `.uns['qc']` mapping.

    Returns:
        dict: {method: bool}.
    """
    entry = (uns_qc or {}).get(method)
    if not isinstance(entry, dict):
        return False
    payloads = (entry.values() if entry and all(isinstance(v, dict) for v in entry.values())
                else [entry])
    for meta in payloads:
        if isinstance(meta, dict) and 'exec_mode' in meta:
            return meta.get('exec_mode', 'filter') != 'filter'
    return False


def _doublet_venn(obs: pd.DataFrame, uns_qc: dict) -> dict:
    """Overlap between the doublet callers that ran, for the page's Venn panel.

    Args:
        obs (pandas.DataFrame): The `.obs` frame.
        uns_qc (dict): The `.uns['qc']` mapping.

    Returns:
        dict or None: Set sizes and their intersections, or None when fewer than
            two callers produced a flag column.
    """
    needed = ('flag_solo', 'flag_scrublet', 'solo_doublet_score', 'scrublet_doublet_score')
    if not all(c in obs.columns for c in needed):
        return {}

    solo = obs['flag_solo'].fillna(False).astype(bool).values
    scr = obs['flag_scrublet'].fillna(False).astype(bool).values
    assessed = (pd.to_numeric(obs['solo_doublet_score'], errors='coerce').notna().values
                & pd.to_numeric(obs['scrublet_doublet_score'], errors='coerce').notna().values)
    solo = solo & assessed
    scr = scr & assessed

    def idx(mask):
        """Positional indices of the cells a flag column marks."""
        return [int(i) for i in np.where(mask)[0]]

    solo_only, scrublet_only, both = idx(solo & ~scr), idx(~solo & scr), idx(solo & scr)
    if not (solo_only or scrublet_only or both):
        return {}

    return {
        'solo_only': solo_only,
        'scrublet_only': scrublet_only,
        'both': both,
        'n_assessed': int(assessed.sum()),
        'n_neither': int((assessed & ~solo & ~scr).sum()),
        'solo_advisory': _method_advisory(uns_qc, 'solo'),
        'scrublet_advisory': _method_advisory(uns_qc, 'scrublet'),
    }


_CSS = """
.ix-wrap { border: 1px solid #e5e5e5; border-radius: 8px; padding: 1rem; margin: 1rem 0; }
.ix-controls { display: flex; flex-wrap: wrap; gap: .6rem; align-items: center; margin-bottom: .8rem; }
.ix-controls select, .ix-controls button, .ix-controls input {
  font: inherit; font-size: .85rem; padding: .3rem .5rem; border: 1px solid #ccc;
  border-radius: 5px; background: #fff; color: inherit; }
.ix-controls button { cursor: pointer; }
.ix-controls input[type=number] { width: 9em; }
.ix-controls input:disabled, .ix-controls select:disabled { opacity: .45; }
.ix-controls button.primary { background: #2c5aa0; color: #fff; border-color: #2c5aa0; }
.ix-controls button:disabled { opacity: .5; cursor: default; }
.ix-canvas { width: 100%; height: 460px; display: block; border: 1px solid #e5e5e5;
             border-radius: 6px; cursor: crosshair; touch-action: none; }
.ix-hist { width: 100%; height: 200px; display: block; border: 1px solid #e5e5e5;
           border-radius: 6px; cursor: ew-resize; }
.ix-status { font-size: .85rem; color: #555; margin: .5rem 0; min-height: 1.4em; }
.ix-list { font-size: .82rem; margin: .5rem 0; padding: 0; list-style: none; }
.ix-list li { padding: .35rem .5rem; border: 1px solid #e5e5e5; border-radius: 5px;
              margin-bottom: .35rem; display: flex; justify-content: space-between; gap: .5rem; }
.ix-list button { font-size: .75rem; padding: .1rem .4rem; }
.ix-legend { font-size: .78rem; color: #666; display: flex; gap: 1rem; flex-wrap: wrap; margin: .4rem 0; }
.ix-legend span::before { content: '\\25CF'; margin-right: .3rem; }
.ix-warn { font-size: .85rem; background: #fff6e5; border: 1px solid #f0d38a; color: #7a4a00;
           border-radius: 6px; padding: .7rem .9rem; margin: .6rem 0; }
.ix-warn pre { background: rgba(0,0,0,.06); padding: .45rem .6rem; border-radius: 4px;
               margin: .5rem 0 0; overflow-x: auto; font-size: .8rem; }
.ix-kept::before { color: #2c5aa0; }
.ix-cut::before { color: #d33; }
.ix-auto::before { color: #bbb; }
@media (prefers-color-scheme: dark) {
  .ix-wrap, .ix-canvas, .ix-hist, .ix-list li { border-color: #2e3238; }
  .ix-controls select, .ix-controls button, .ix-controls input { background: #1e2126; border-color: #3a3f46; }
  .ix-controls button.primary { background: #3f6fb5; border-color: #3f6fb5; }
  .ix-status, .ix-legend { color: #9aa0a6; }
  .ix-warn { background: #2b2210; border-color: #6b5420; color: #e8c777; }
  .ix-warn pre { background: rgba(255,255,255,.07); }
}
"""

_JS = r"""
(function () {
  var D = window.__IX_DATA__;
  if (!D) return;
  var $ = function (id) { return document.getElementById(id); };

  // Decision state. Barcode sets are kept as objects (not arrays) so repeated
  // lassoing of overlapping regions cannot duplicate entries.
  // No `mode`: the lasso only removes. Every embedding here is computed on
  // cells that passed QC, so cells the cascade removed have no coordinates and
  // are never drawn -- a "keep selected" lasso had nothing it could select.
  // `rescued` stays in the state and the schema because refilter still supports
  // it and it is what makes decisions undoable.
  // `doubletRegions` is a set of selected Venn regions ('solo_only',
  // 'scrublet_only', 'both'). Kept apart from `excluded` so a region toggles
  // cleanly and undoing it does not also wipe a lasso selection sharing that set.
  var state = { thresholds: {}, excluded: {}, rescued: {}, doubletRegions: {} };

  var canvas = $('ix-scatter'), ctx = canvas.getContext('2d');
  var hist = $('ix-hist'), hctx = hist.getContext('2d');
  var embSel = $('ix-emb'), plotSel = $('ix-plot'), clustSel = $('ix-clust');
  var view = null, lasso = null;

  // --- axis scales ----------------------------------------------------------
  // A MAD entry is fit in transformed space (log1p for counts, logit for
  // fractions) and the report's MAD figure is drawn on that axis. Drawing the
  // same feature here on a linear axis made the two figures look like different
  // data. So the *spacing* is transformed to match, while every value the panel
  // shows, stores or exports stays raw -- which is the unit the decision file
  // records, refilter applies, and a person types.
  //
  // These must stay the exact inverses of qc_filter._transform_features:
  // LOGIT_CLIP is shipped in the payload rather than repeated here so the two
  // cannot drift.
  var CLIP = D.logit_clip || 1e-6;

  function fwd(scale, v) {
    if (scale === 'logit') {
      var x = Math.min(Math.max(v, CLIP), 1 - CLIP);
      return Math.log(x / (1 - x));
    }
    if (scale === 'log1p') return Math.log1p(Math.max(v, 0));
    return v;
  }

  function inv(scale, u) {
    if (scale === 'logit') return 1 / (1 + Math.exp(-u));
    if (scale === 'log1p') return Math.expm1(u);
    return u;
  }

  // Fraction values worth labelling on a logit axis, and 1/2/5 x 10^n decades
  // for a log1p one. Fixed lists rather than a generic break algorithm run over
  // transformed space, which returns labels like 0.00734 -- the whole point of
  // the axis is that the action is at 0.1% and 1%, so the ticks have to be
  // numbers someone recognises. Mirrors qc_plot._FRACTION_BREAKS.
  var FRACTION_BREAKS = [0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05,
                         0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1];

  function decadeBreaks(hi) {
    var out = [0], m = 1;
    while (m <= hi * 10 && out.length < 40) {
      out.push(m); out.push(2 * m); out.push(5 * m);
      m *= 10;
    }
    return out;
  }

  // Thin `cand` to at most `want` entries along the transformed axis. Greedy
  // with a minimum gap rather than nearest-to-evenly-spaced-targets, which
  // happily picks two candidates that are adjacent on the axis and prints their
  // labels on top of each other. Mirrors qc_plot._thin_breaks.
  function thin(cand, scale, lo, hi, want) {
    var sel = [], i;
    for (i = 0; i < cand.length; i++) if (cand[i] >= lo && cand[i] <= hi) sel.push(cand[i]);
    if (sel.length < 2) return [lo, hi];
    if (sel.length <= want) return sel;
    var t = sel.map(function (v) { return fwd(scale, v); });
    var gap = (t[t.length - 1] - t[0]) / want, out = [sel[0]], last = t[0];
    for (i = 1; i < sel.length; i++) {
      if (t[i] - last >= gap) { out.push(sel[i]); last = t[i]; }
    }
    return out;
  }

  function axisTicks(scale, lo, hi) {
    if (scale === 'logit') return thin(FRACTION_BREAKS, scale, lo, hi, 6);
    if (scale === 'log1p') return thin(decadeBreaks(hi), scale, lo, hi, 6);
    return niceTicks(lo, hi, 6);
  }

  function dpr(cv) {
    var r = window.devicePixelRatio || 1, b = cv.getBoundingClientRect();
    cv.width = b.width * r; cv.height = b.height * r;
    cv.getContext('2d').setTransform(r, 0, 0, r, 0, 0);
    return { w: b.width, h: b.height };
  }

  // --- current removal state ------------------------------------------------
  // Recomputed from scratch on every change rather than tracked incrementally,
  // so the displayed state can never drift from what an export would produce.
  // `projected=false` is what refilter will actually produce: manual bounds are
  // OR'd onto the automatic ones and can only tighten. `projected=true` instead
  // *replaces* each automatic bound with the hand-set one where given, which is
  // what the config + --force-qc route would produce. Both are shown, because
  // for a loosened bound the applied number does not move at all and a panel
  // that only reported it would look broken.
  function cut(projected) {
    var n = D.n_cells, out = new Uint8Array(n), i, f, t, v, col;
    var base = projected ? D.auto_removed_other : D.auto_removed;
    for (i = 0; i < n; i++) out[i] = base[i] ? 1 : 0;

    if (projected) {
      // Re-derive each automatic single-feature bound, substituting the manual
      // value on whichever side the user set.
      for (f in D.bounds) {
        col = D.features[f];
        if (!col) continue;
        var man = state.thresholds[f];
        for (var bi = 0; bi < D.bounds[f].length; bi++) {
          var b = D.bounds[f][bi];
          if (b.exec_mode !== 'filter') continue;
          var lo = (man && man.op === 'min') ? man.value : b.lower;
          var hi = (man && man.op === 'max') ? man.value : b.upper;
          for (i = 0; i < n; i++) {
            v = col[i];
            if (v === null) continue;
            if (lo !== null && lo !== undefined && v < lo) out[i] = 1;
            if (hi !== null && hi !== undefined && v > hi) out[i] = 1;
          }
        }
      }
    }

    for (f in state.thresholds) {
      // In projected mode a bounded feature was already handled above; applying
      // it again here would double-count and, worse, re-add the automatic bound
      // the projection just replaced.
      if (projected && D.bounds[f]) continue;
      t = state.thresholds[f];
      col = D.features[f];
      if (!col) continue;
      for (i = 0; i < n; i++) {
        v = col[i];
        if (v === null) continue;
        if (t.op === 'max' ? v > t.value : v < t.value) out[i] = 1;
      }
    }
    for (i = 0; i < n; i++) if (state.excluded[D.barcodes[i]]) out[i] = 1;
    // Selected Venn regions: remove every cell in a chosen SOLO/Scrublet region.
    if (D.doublets) {
      for (var rk in state.doubletRegions) {
        if (!state.doubletRegions[rk]) continue;
        var reg = D.doublets[rk] || [];
        for (var ri = 0; ri < reg.length; ri++) out[reg[ri]] = 1;
      }
    }
    // Rescue last: an explicit keep beats every rule above, including automatic.
    for (i = 0; i < n; i++) if (state.rescued[D.barcodes[i]]) out[i] = 0;
    return out;
  }

  // True when any hand-set bound is more permissive than the pipeline's, i.e.
  // the applied and projected numbers will disagree.
  function hasLoosened() {
    for (var f in state.thresholds) {
      var t = state.thresholds[f], bl = D.bounds[f] || [];
      for (var i = 0; i < bl.length; i++) {
        if (bl[i].exec_mode !== 'filter') continue;
        var auto = t.op === 'min' ? bl[i].lower : bl[i].upper;
        if (auto === null || auto === undefined) continue;
        if (t.op === 'min' ? t.value < auto : t.value > auto) return true;
      }
    }
    return false;
  }

  function status() {
    var c = cut(false), kept = 0, i, auto = 0;
    for (i = 0; i < c.length; i++) if (!c[i]) kept++;
    for (i = 0; i < D.n_cells; i++) if (D.auto_removed[i]) auto++;
    var autoKept = D.n_cells - auto, delta = kept - autoKept;

    var msg = kept.toLocaleString() + ' of ' + D.n_cells.toLocaleString() + ' cells kept (' +
      (100 * kept / D.n_cells).toFixed(1) + '%). Automatic filtering alone kept ' +
      autoKept.toLocaleString() + '. Net manual change: ' + (delta >= 0 ? '+' : '') +
      delta.toLocaleString() + ' cells.';

    if (hasLoosened()) {
      var pc = cut(true), pkept = 0;
      for (i = 0; i < pc.length; i++) if (!pc[i]) pkept++;
      msg += '  \u2014  One of your bounds is more permissive than the pipeline\u2019s, which a ' +
             'decision file cannot apply (see below). Set in config and re-run, it would keep ' +
             pkept.toLocaleString() + ' cells.';
    }
    $('ix-status').textContent = msg;
    renderDecisions();
    $('ix-export').disabled = !decisions().length;
  }

  // --- scatter --------------------------------------------------------------
  function drawScatter() {
    var key = embSel.value, emb = D.embeddings[key];
    var size = dpr(canvas);
    ctx.clearRect(0, 0, size.w, size.h);
    if (!emb) { view = null; return; }

    var xs = emb.x, ys = emb.y, n = D.n_cells, i;
    var minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity;
    for (i = 0; i < n; i++) {
      if (xs[i] === null || ys[i] === null) continue;
      if (xs[i] < minx) minx = xs[i]; if (xs[i] > maxx) maxx = xs[i];
      if (ys[i] < miny) miny = ys[i]; if (ys[i] > maxy) maxy = ys[i];
    }
    if (!isFinite(minx)) { view = null; return; }
    var pad = 14, sx = (size.w - 2 * pad) / ((maxx - minx) || 1),
        sy = (size.h - 2 * pad) / ((maxy - miny) || 1);
    view = {
      px: function (x) { return pad + (x - minx) * sx; },
      py: function (y) { return size.h - pad - (y - miny) * sy; }
    };

    var c = cut(false);
    var clustKey = clustSel.value, labels = clustKey ? D.clusterings[clustKey] : null;
    if (clustKey) buildClusterIndex(clustKey);
    for (i = 0; i < n; i++) {
      if (xs[i] === null || ys[i] === null) continue;
      ctx.beginPath();
      ctx.arc(view.px(xs[i]), view.py(ys[i]), 1.8, 0, 6.283);
      if (c[i]) ctx.fillStyle = D.auto_removed[i] && !state.excluded[D.barcodes[i]]
        ? 'rgba(180,180,180,.45)' : 'rgba(221,51,51,.75)';
      else ctx.fillStyle = labels ? clusterColor(labels[i]) : 'rgba(44,90,160,.75)';
      ctx.fill();
    }
    if (lasso && lasso.length > 1) {
      ctx.strokeStyle = '#d33'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(lasso[0][0], lasso[0][1]);
      for (i = 1; i < lasso.length; i++) ctx.lineTo(lasso[i][0], lasso[i][1]);
      ctx.closePath(); ctx.stroke();
    }
  }

  // Hue by the label's position in the sorted unique set, stepped by the golden
  // angle. The previous version hashed the label string, which for leiden's
  // first ten clusters -- single characters '0'..'9' -- reduced to the character
  // code itself: hues 48 through 57, ten shades spanning nine degrees, i.e.
  // visually one colour. Index-based stepping also keeps a cluster's colour
  // stable across redraws and independent of the label vocabulary.
  var clusterIndex = {};
  function buildClusterIndex(key) {
    var labels = D.clusterings[key] || [], seen = {}, uniq = [], i;
    for (i = 0; i < labels.length; i++) {
      if (labels[i] !== '' && !(labels[i] in seen)) { seen[labels[i]] = 1; uniq.push(labels[i]); }
    }
    uniq.sort(function (a, b) {
      var na = parseFloat(a), nb = parseFloat(b);
      if (!isNaN(na) && !isNaN(nb)) return na - nb;   // '2' before '10'
      return a < b ? -1 : (a > b ? 1 : 0);
    });
    clusterIndex = {};
    for (i = 0; i < uniq.length; i++) clusterIndex[uniq[i]] = i;
  }

  function clusterColor(label) {
    if (label === '' || label === undefined) return 'rgba(170,170,170,.5)';
    var i = clusterIndex[label];
    if (i === undefined) return 'rgba(170,170,170,.5)';
    return 'hsla(' + ((i * 137.508) % 360).toFixed(1) + ',65%,52%,.85)';
  }

  // Ray casting; the polygon is in screen space, so points are projected first.
  function inside(poly, x, y) {
    var hit = false;
    for (var i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      var xi = poly[i][0], yi = poly[i][1], xj = poly[j][0], yj = poly[j][1];
      if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / ((yj - yi) || 1e-9) + xi)) hit = !hit;
    }
    return hit;
  }

  function applyLasso() {
    var key = embSel.value, emb = D.embeddings[key];
    if (!emb || !view || !lasso || lasso.length < 3) return;
    var target = state.excluded, n = 0;
    for (var i = 0; i < D.n_cells; i++) {
      if (emb.x[i] === null || emb.y[i] === null) continue;
      if (inside(lasso, view.px(emb.x[i]), view.py(emb.y[i]))) {
        target[D.barcodes[i]] = 1;
        n++;
      }
    }
    lasso = null;
    if (n) { drawScatter(); status(); }
  }

  var drawing = false;
  canvas.addEventListener('pointerdown', function (e) {
    drawing = true; lasso = []; canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', function (e) {
    if (!drawing) return;
    var b = canvas.getBoundingClientRect();
    lasso.push([e.clientX - b.left, e.clientY - b.top]);
    drawScatter();
  });
  canvas.addEventListener('pointerup', function () { drawing = false; applyLasso(); });

  // --- QC diagnostic plots --------------------------------------------------
  // Renders whichever figure the picker names, in that figure's own form:
  // 'hist' is a distribution with a draggable cut-off, 'scatter' is the joint
  // view (miQC, regression, DoD) with lasso selection, because a residual or a
  // two-axis fit has no single value to threshold on.
  var histMeta = null, plotView = null, plotLasso = null;

  function currentPlot() {
    var plots = D.plots || [];
    return plots.length ? plots[parseInt(plotSel.value || '0', 10)] : null;
  }

  // Round tick values covering [lo, hi]: 1/2/5 x 10^n, the usual choice, so
  // labels read as numbers a person recognises rather than raw data extremes.
  function niceTicks(lo, hi, want) {
    if (!isFinite(lo) || !isFinite(hi) || hi <= lo) return [lo];
    var raw = (hi - lo) / Math.max(1, want);
    var mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var norm = raw / mag;
    var step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
    var out = [], t = Math.ceil(lo / step) * step;
    for (; t <= hi + step * 1e-9 && out.length < 20; t += step) out.push(t);
    return out.length ? out : [lo, hi];
  }

  // Significant digits, not fixed decimals. A logit axis legitimately labels
  // 0.0005 and 0.02 on the same row, and rounding to three decimal places turns
  // the first into '0.001' -- a tick that is simply wrong.
  function fmtTick(v) {
    if (v === 0) return '0';
    var a = Math.abs(v);
    if (a < 1e-4 || a >= 1e6) return v.toExponential(1);
    if (a >= 1000) return String(Math.round(v));
    return String(parseFloat(v.toPrecision(3)));
  }

  // A blank canvas is indistinguishable from a broken one, so say why.
  function noData(size, msg) {
    hctx.fillStyle = '#999'; hctx.font = '13px sans-serif'; hctx.textAlign = 'center';
    hctx.fillText(msg, size.w / 2, size.h / 2);
    hctx.textAlign = 'left';
  }

  function drawHist() {
    var size = dpr(hist);
    hctx.clearRect(0, 0, size.w, size.h);
    histMeta = null; plotView = null; vennGeom = null;
    hist.style.cursor = '';   // fall back to the CSS cursor; drawVenn overrides
    var pl = currentPlot();
    if (!pl) return;
    if (pl.kind === 'venn') { drawVenn(size); return; }
    if (pl.kind === 'scatter') { drawPlotScatter(pl, size); return; }

    var col = D.features[pl.x], f = pl.x, i;
    if (!col) { noData(size, 'No data for ' + pl.x); return; }
    // Min/max by loop, not Math.min.apply: apply spreads every element as a
    // separate argument, and these columns run to tens of thousands of values
    // (the payload covers the pre-filter object). That overflows the call stack
    // -- RangeError, drawHist dies mid-draw, and the canvas is left blank with
    // no indication why.
    var vals = [], lo = Infinity, hi = -Infinity, v;
    for (i = 0; i < col.length; i++) {
      v = col[i];
      if (v === null) continue;
      vals.push(v);
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!vals.length) {
      noData(size, 'No cell has a value for ' + pl.x + ' \u2014 this method assessed none.');
      return;
    }
    if (hi === lo) hi = lo + 1;
    // Bin on the axis the filter was fit on, not on raw values: a log1p-scaled
    // count feature binned linearly puts nearly every cell in the first bar.
    // The transform is monotone, so cell membership is identical either way --
    // only the bar widths change, and these are the widths the report figure
    // shows.
    var scale = pl.scale || '';
    var loU = fwd(scale, lo), hiU = fwd(scale, hi);
    if (hiU === loU) hiU = loU + 1;
    var bins = 60, counts = new Array(bins).fill(0);
    for (i = 0; i < vals.length; i++) {
      counts[Math.min(bins - 1, Math.floor((fwd(scale, vals[i]) - loU) / (hiU - loU) * bins))]++;
    }
    var peak = Math.max.apply(null, counts), pad = 34, bot = 26;
    var W = size.w - pad - 14, H = size.h - 10 - bot;
    histMeta = { lo: lo, hi: hi, pad: pad, W: W, feature: f, scale: scale,
                 toVal: function (px) { return inv(scale, loU + (px - pad) / W * (hiU - loU)); },
                 toPx: function (v) { return pad + (fwd(scale, v) - loU) / (hiU - loU) * W; } };

    hctx.fillStyle = 'rgba(44,90,160,.55)';
    for (i = 0; i < bins; i++) {
      var h = counts[i] / peak * H;
      hctx.fillRect(pad + i * W / bins, size.h - bot - h, Math.max(1, W / bins - 1), h);
    }

    // Pipeline bounds in raw units; downgraded ones fainter and labelled.
    (pl.bounds || []).forEach(function (b, bi) {
      [b.lower, b.upper].forEach(function (bv) {
        if (bv === null || bv === undefined || bv < lo || bv > hi) return;
        var bx = histMeta.toPx(bv);
        hctx.strokeStyle = b.downgraded ? 'rgba(136,136,136,.45)' : 'rgba(90,90,90,.9)';
        hctx.lineWidth = 1; hctx.setLineDash([3, 3]);
        hctx.beginPath(); hctx.moveTo(bx, 4); hctx.lineTo(bx, size.h - bot); hctx.stroke();
        hctx.setLineDash([]);
        hctx.fillStyle = b.downgraded ? 'rgba(136,136,136,.8)' : 'rgba(90,90,90,1)';
        hctx.font = '10px sans-serif';
        hctx.fillText(b.method + (b.downgraded ? ' (advisory)' : ''), bx + 3, 13 + bi * 11);
      });
    });

    // The method's own cut-off (GMM's Mahalanobis cut, iForest's quota
    // boundary, SOLO's 0.5, DoD's overlap threshold). Distinct from the manual
    // line below and from the MAD bounds above.
    if (pl.cutoff !== null && pl.cutoff !== undefined) {
      // A cut-off outside the observed range is worth showing, not hiding: it
      // means the method flagged nothing, and silently omitting the line makes
      // that indistinguishable from a plot that simply has no threshold. Pin it
      // to the edge and say which way it lies.
      var off = pl.cutoff < lo ? -1 : (pl.cutoff > hi ? 1 : 0);
      var cx = histMeta.toPx(off ? (off < 0 ? lo : hi) : pl.cutoff);
      hctx.strokeStyle = pl.advisory ? 'rgba(90,90,90,.55)' : 'rgba(90,90,90,.95)';
      hctx.lineWidth = off ? 2 : 1;
      hctx.setLineDash(off ? [2, 2] : [5, 3]);
      hctx.beginPath(); hctx.moveTo(cx, 4); hctx.lineTo(cx, size.h - bot); hctx.stroke();
      hctx.setLineDash([]);
      hctx.fillStyle = hctx.strokeStyle; hctx.font = '10px sans-serif';
      var lbl = pl.method + (pl.advisory ? ' (advisory)' : '') + ' ' + Number(pl.cutoff).toPrecision(3);
      if (off) lbl += off < 0 ? '  \u2190 off scale, nothing flagged'
                              : '  off scale \u2192 nothing flagged';
      hctx.textAlign = off > 0 ? 'right' : 'left';
      hctx.fillText(lbl, cx + (off > 0 ? -3 : 3), 13);
      hctx.textAlign = 'left';
    }

    var t = state.thresholds[f];
    if (t) {
      // A typed threshold is deliberately not clamped to the observed range --
      // 0.5 means 0.5 even on a sample whose scores top out at 0.4, and silently
      // moving it to 0.4 would make the same typed number mean something
      // different per sample, which is the whole reason typing exists. Off-scale
      // values are pinned to the edge and labelled instead of vanishing.
      var toff = t.value < lo ? -1 : (t.value > hi ? 1 : 0);
      var x = histMeta.toPx(toff ? (toff < 0 ? lo : hi) : t.value);
      hctx.strokeStyle = '#d33'; hctx.lineWidth = 2;
      hctx.setLineDash(toff ? [2, 2] : []);
      hctx.beginPath(); hctx.moveTo(x, 4); hctx.lineTo(x, size.h - bot); hctx.stroke();
      hctx.setLineDash([]);
      hctx.fillStyle = 'rgba(221,51,51,.12)';
      if (t.op === 'max') hctx.fillRect(x, 4, pad + W - x, size.h - bot - 4);
      else hctx.fillRect(pad, 4, x - pad, size.h - bot - 4);
      if (toff) {
        hctx.fillStyle = '#d33'; hctx.font = '10px sans-serif';
        hctx.textAlign = toff > 0 ? 'right' : 'left';
        hctx.fillText('yours ' + fmtTick(t.value) +
                      (toff < 0 ? '  ← off scale' : '  off scale →'),
                      x + (toff > 0 ? -3 : 3), size.h - bot - 6);
        hctx.textAlign = 'left';
      }
    }
    // Axis: ticks along x, and the count scale on y so bar heights are readable.
    // x ticks come from axisTicks, which knows the axis is transformed; their
    // labels are raw values either way.
    hctx.strokeStyle = 'rgba(136,136,136,.7)'; hctx.lineWidth = 1;
    hctx.beginPath(); hctx.moveTo(pad, size.h - bot); hctx.lineTo(pad + W, size.h - bot); hctx.stroke();
    hctx.fillStyle = '#888'; hctx.font = '10px sans-serif'; hctx.textAlign = 'center';
    axisTicks(scale, lo, hi).forEach(function (t) {
      var tx = histMeta.toPx(t);
      if (tx < pad - 1 || tx > pad + W + 1) return;
      hctx.beginPath(); hctx.moveTo(tx, size.h - bot); hctx.lineTo(tx, size.h - bot + 3); hctx.stroke();
      hctx.fillText(fmtTick(t), tx, size.h - bot + 14);
    });
    hctx.textAlign = 'right';
    niceTicks(0, peak, 3).forEach(function (t) {
      if (t <= 0) return;
      var ty = size.h - bot - t / peak * H;
      hctx.fillText(fmtTick(t), pad - 3, ty + 3);
    });
    hctx.textAlign = 'left';
  }

  function drawPlotScatter(pl, size) {
    var xs = D.features[pl.x], ys = D.features[pl.y], n = D.n_cells, i;
    if (!xs || !ys) { noData(size, 'No data for ' + pl.x + ' / ' + pl.y); return; }
    var minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity;
    for (i = 0; i < n; i++) {
      if (xs[i] === null || ys[i] === null) continue;
      if (xs[i] < minx) minx = xs[i]; if (xs[i] > maxx) maxx = xs[i];
      if (ys[i] < miny) miny = ys[i]; if (ys[i] > maxy) maxy = ys[i];
    }
    if (!isFinite(minx)) {
      noData(size, 'No cell has both ' + pl.x + ' and ' + pl.y + ' \u2014 this method assessed none.');
      return;
    }
    var pad = 34, W = size.w - pad - 12, H = size.h - pad - 10;
    var px = function (v) { return pad + (v - minx) / ((maxx - minx) || 1) * W; };
    var py = function (v) { return 10 + H - (v - miny) / ((maxy - miny) || 1) * H; };
    plotView = { px: px, py: py, x: pl.x, y: pl.y };

    var c = cut(false);
    for (i = 0; i < n; i++) {
      if (xs[i] === null || ys[i] === null) continue;
      hctx.beginPath(); hctx.arc(px(xs[i]), py(ys[i]), 1.6, 0, 6.283);
      hctx.fillStyle = c[i] ? 'rgba(221,51,51,.6)' : 'rgba(44,90,160,.55)';
      hctx.fill();
    }
    // The method's own fitted line, where it has one (miQC's intact trend,
    // regression's fit). Drawn in the same axes as the report figure.
    if (pl.line && pl.line.slope !== null && pl.line.slope !== undefined) {
      var drawFit = function (offset, style, dash) {
        hctx.strokeStyle = style; hctx.lineWidth = 1.5; hctx.setLineDash(dash);
        hctx.beginPath();
        hctx.moveTo(px(minx), py(pl.line.intercept + offset + pl.line.slope * minx));
        hctx.lineTo(px(maxx), py(pl.line.intercept + offset + pl.line.slope * maxx));
        hctx.stroke(); hctx.setLineDash([]);
      };
      drawFit(0, '#2c5aa0', []);
      // The cut is on the residual, so the bounds are the fit shifted up and
      // down -- exactly what the report's regression figure draws. Without
      // them the plot shows the fit but not where the filter actually cuts.
      var faint = pl.advisory ? 'rgba(136,136,136,.7)' : 'rgba(90,90,90,.95)';
      if (pl.line.lower !== null && pl.line.lower !== undefined) drawFit(pl.line.lower, faint, [5, 3]);
      if (pl.line.upper !== null && pl.line.upper !== undefined) drawFit(pl.line.upper, faint, [5, 3]);
    }
    if (plotLasso && plotLasso.length > 1) {
      hctx.strokeStyle = '#d33'; hctx.lineWidth = 1.5;
      hctx.beginPath(); hctx.moveTo(plotLasso[0][0], plotLasso[0][1]);
      for (i = 1; i < plotLasso.length; i++) hctx.lineTo(plotLasso[i][0], plotLasso[i][1]);
      hctx.closePath(); hctx.stroke();
    }
    // Axes with values on both, not just names -- a scatter of unlabelled
    // residuals tells you nothing about where a selection actually sits.
    hctx.strokeStyle = 'rgba(136,136,136,.7)'; hctx.lineWidth = 1;
    hctx.beginPath();
    hctx.moveTo(pad, 10); hctx.lineTo(pad, 10 + H); hctx.lineTo(pad + W, 10 + H);
    hctx.stroke();
    hctx.fillStyle = '#888'; hctx.font = '10px sans-serif';
    hctx.textAlign = 'center';
    niceTicks(minx, maxx, 6).forEach(function (t) {
      var tx = px(t);
      if (tx < pad - 1 || tx > pad + W + 1) return;
      hctx.beginPath(); hctx.moveTo(tx, 10 + H); hctx.lineTo(tx, 13 + H); hctx.stroke();
      hctx.fillText(fmtTick(t), tx, 24 + H);
    });
    hctx.textAlign = 'right';
    niceTicks(miny, maxy, 5).forEach(function (t) {
      var ty = py(t);
      if (ty < 8 || ty > 12 + H) return;
      hctx.beginPath(); hctx.moveTo(pad - 3, ty); hctx.lineTo(pad, ty); hctx.stroke();
      hctx.fillText(fmtTick(t), pad - 5, ty + 3);
    });
    hctx.textAlign = 'left';
    hctx.font = '11px sans-serif'; hctx.fillStyle = '#666';
    hctx.fillText(pl.x, pad, size.h - 1);
    hctx.save(); hctx.translate(9, 12); hctx.fillText(pl.y, 0, 0); hctx.restore();
  }

  // Six significant figures, trailing zeros dropped: enough to round-trip a
  // dragged value (decisions() rounds to 6 dp anyway) without showing a count
  // threshold as 1584.9999999999998 in a box someone is about to edit.
  function fmtBox(v) {
    if (!isFinite(v)) return '';
    return String(parseFloat(Number(v).toPrecision(6)));
  }

  // Single entry point for both routes. `clamp` is true only for the drag, where
  // the pointer cannot mean a value outside the axis anyway; a typed value is
  // taken literally -- see the off-scale note in drawHist.
  function setThreshold(f, value, clamp) {
    if (!f || !isFinite(value)) return;
    if (clamp && histMeta) value = Math.max(histMeta.lo, Math.min(histMeta.hi, value));
    state.thresholds[f] = { op: $('ix-op').value, value: value };
    syncBox();
    drawHist(); drawScatter(); drawBcrank(); status();
  }

  function setThresholdFromPx(px) {
    if (!histMeta) return;
    setThreshold(histMeta.feature, histMeta.toVal(px), true);
  }

  // Keep the number box and the operator showing the current plot's own
  // threshold. Without this, switching plots leaves the previous feature's value
  // sitting in the box, and pressing Enter would apply it to the new feature.
  function syncBox() {
    var box = $('ix-val'), pl = currentPlot();
    if (!box) return;
    var f = (pl && pl.kind !== 'scatter') ? pl.x : null;
    var t = f ? state.thresholds[f] : null;
    box.disabled = !f;
    $('ix-op').disabled = !f;
    $('ix-set').disabled = !f;
    box.placeholder = f ? 'value in ' + (pl.scale ? 'raw units' : 'feature units') : 'no axis to cut';
    box.value = t ? fmtBox(t.value) : '';
    if (t) $('ix-op').value = t.op;
    // The barcode-rank cut is an entry in the same `state.thresholds` map --
    // keyed by the same feature the MAD count plot uses, so setting it in one
    // place has to show up in the other rather than leaving two boxes claiming
    // different values for one threshold.
    var bcBox = $('ix-bcval');
    if (bcBox && D.bcrank && D.bcrank.feature) {
      var bt = state.thresholds[D.bcrank.feature];
      bcBox.value = bt ? fmtBox(bt.value) : '';
    }
  }

  function applyBox() {
    var pl = currentPlot();
    if (!pl || pl.kind !== 'hist') return;   // only histograms take a typed cut-off
    var raw = $('ix-val').value.trim();
    if (raw === '') {           // cleared, then Set/Enter -- drop the threshold
      delete state.thresholds[pl.x];
      drawHist(); drawScatter(); drawBcrank(); syncBox(); status();
      return;
    }
    var v = parseFloat(raw);
    if (!isFinite(v)) { syncBox(); return; }
    setThreshold(pl.x, v, false);
  }

  var dragging = false, lassoing = false;
  function vennIsCurrent() { var pl = currentPlot(); return pl && pl.kind === 'venn'; }

  hist.addEventListener('pointerdown', function (e) {
    if (vennIsCurrent()) return;   // venn toggles on click, not drag/lasso
    hist.setPointerCapture(e.pointerId);
    var b = hist.getBoundingClientRect();
    if (plotView) { lassoing = true; plotLasso = [[e.clientX - b.left, e.clientY - b.top]]; }
    else { dragging = true; setThresholdFromPx(e.clientX - b.left); }
  });
  hist.addEventListener('pointermove', function (e) {
    var b = hist.getBoundingClientRect();
    if (vennIsCurrent()) {
      var rk = vennRegionAt(e.clientX - b.left, e.clientY - b.top);
      hist.style.cursor = (rk && D.doublets[rk].length) ? 'pointer' : 'default';
      return;
    }
    if (dragging) setThresholdFromPx(e.clientX - b.left);
    else if (lassoing) { plotLasso.push([e.clientX - b.left, e.clientY - b.top]); drawHist(); }
  });
  hist.addEventListener('click', function (e) {
    if (vennIsCurrent()) vennToggleAt(e.clientX, e.clientY);
  });
  hist.addEventListener('pointerup', function () {
    if (vennIsCurrent()) return;
    dragging = false;
    if (lassoing) {
      lassoing = false;
      var pl = currentPlot();
      if (pl && plotView && plotLasso && plotLasso.length > 2) {
        var xs = D.features[plotView.x], ys = D.features[plotView.y], n = 0;
        for (var i = 0; i < D.n_cells; i++) {
          if (xs[i] === null || ys[i] === null) continue;
          if (inside(plotLasso, plotView.px(xs[i]), plotView.py(ys[i]))) {
            state.excluded[D.barcodes[i]] = 1; n++;
          }
        }
        if (n) { plotLasso = null; drawHist(); drawScatter(); status(); return; }
      }
      plotLasso = null; drawHist();
    }
  });

  // --- barcode rank ---------------------------------------------------------
  // Log-log rank/count curve over every barcode, with a draggable cut. The
  // asymmetry is the whole point: dragging right removes more (coherent, applies
  // directly); dragging left admits cells no downstream filter ever assessed,
  // which cannot be expressed as a decision and needs a config change plus a QC
  // re-run instead.
  var bc = $('ix-bcrank'), bctx = bc ? bc.getContext('2d') : null, bcMeta = null;

  function drawBcrank() {
    if (!bc || !D.bcrank || !D.bcrank.rank) return;
    var size = dpr(bc);
    bctx.clearRect(0, 0, size.w, size.h);
    var R = D.bcrank.rank, C = D.bcrank.count, i;
    var padL = 46, padR = 12, padT = 12, padB = 26;
    var W = size.w - padL - padR, H = size.h - padT - padB;
    var lx = function (v) { return Math.log10(Math.max(v, 1)); };
    var minX = lx(R[0]), maxX = lx(R[R.length - 1]);
    var minY = lx(C[C.length - 1]), maxY = lx(C[0]);
    if (maxX === minX) maxX = minX + 1;
    if (maxY === minY) maxY = minY + 1;
    var px = function (r) { return padL + (lx(r) - minX) / (maxX - minX) * W; };
    var py = function (c) { return padT + H - (lx(c) - minY) / (maxY - minY) * H; };
    bcMeta = { py: py, padT: padT, H: H, minY: minY, maxY: maxY,
               toCount: function (y) {
                 return Math.pow(10, minY + (padT + H - y) / H * (maxY - minY));
               } };

    var auto = D.bcrank.threshold;
    var manual = state.thresholds[D.bcrank.feature];
    var cur = manual ? manual.value : auto;

    // Shade the band between the automatic cut and a looser manual one: those
    // are the cells being admitted without ever having been assessed.
    if (auto !== null && cur < auto) {
      bctx.fillStyle = 'rgba(221,51,51,.10)';
      bctx.fillRect(padL, py(auto), W, py(cur) - py(auto));
    }

    bctx.strokeStyle = '#2c5aa0'; bctx.lineWidth = 1.6;
    bctx.beginPath();
    for (i = 0; i < R.length; i++) {
      var X = px(R[i]), Y = py(C[i]);
      if (i === 0) bctx.moveTo(X, Y); else bctx.lineTo(X, Y);
    }
    bctx.stroke();

    if (auto !== null) {
      bctx.strokeStyle = '#888'; bctx.lineWidth = 1; bctx.setLineDash([4, 3]);
      bctx.beginPath(); bctx.moveTo(padL, py(auto)); bctx.lineTo(padL + W, py(auto)); bctx.stroke();
      bctx.setLineDash([]);
      bctx.fillStyle = '#888'; bctx.font = '11px sans-serif';
      bctx.fillText('pipeline ' + auto.toPrecision(4), padL + 4, py(auto) - 3);
    }
    if (manual) {
      bctx.strokeStyle = '#d33'; bctx.lineWidth = 2;
      bctx.beginPath(); bctx.moveTo(padL, py(cur)); bctx.lineTo(padL + W, py(cur)); bctx.stroke();
      bctx.fillStyle = '#d33'; bctx.font = '11px sans-serif';
      bctx.fillText('yours ' + cur.toPrecision(4), padL + 4, py(cur) + 12);
    }
    bctx.fillStyle = '#888'; bctx.font = '11px sans-serif';
    bctx.fillText('barcode rank (log)', padL, size.h - 8);
    bcWarn(cur, auto);
  }

  function bcWarn(cur, auto) {
    var box = $('ix-bcrank-warn');
    if (!box) return;
    if (auto === null || cur === undefined || cur >= auto) { box.innerHTML = ''; return; }
    var col = D.features[D.bcrank.feature], admitted = 0, i;
    for (i = 0; i < col.length; i++) {
      if (col[i] !== null && col[i] >= cur && col[i] < auto) admitted++;
    }
    box.innerHTML =
      '<div class="ix-warn"><strong>This admits ' + admitted.toLocaleString() +
      ' cells the pipeline removed.</strong> They were never in MAD\'s, miQC\'s or ' +
      'regression\'s view, so their flags are <code>False</code> by default &mdash; not tested, ' +
      'not passed. Exporting this will not apply it. To use this cut-off, set it in ' +
      '<code>configs/filter_config.yaml</code> and re-run QC so every downstream filter re-fits ' +
      'on the population you actually want:' +
      '<pre>barcode_rank:\n  threshold: ' + Number(cur.toPrecision(4)) + '</pre>' +
      '<pre>python -m mapping.main -c configs/pipeline_config.yaml --force-qc</pre></div>';
  }

  if (bc) {
    var bcDrag = false;
    // Through syncBox rather than writing the box directly: the barcode-rank
    // feature is usually also a MAD plot's feature, so moving this cut has to
    // show up in that plot's number box too.
    var setBcValue = function (v) {
      if (!isFinite(v)) return;
      state.thresholds[D.bcrank.feature] = { op: 'min', value: v };
      syncBox();
      drawBcrank(); drawHist(); drawScatter(); status();
    };
    var setBc = function (clientY) {
      if (!bcMeta) return;
      var y = clientY - bc.getBoundingClientRect().top;
      setBcValue(bcMeta.toCount(Math.max(bcMeta.padT, Math.min(bcMeta.padT + bcMeta.H, y))));
    };
    bc.addEventListener('pointerdown', function (e) {
      bcDrag = true; bc.setPointerCapture(e.pointerId); setBc(e.clientY);
    });
    bc.addEventListener('pointermove', function (e) { if (bcDrag) setBc(e.clientY); });
    bc.addEventListener('pointerup', function () { bcDrag = false; });

    // Typed entry, for the case the drag cannot serve: the same cut-off across
    // several samples. Dragging a log-log curve to land on 500 twice is not
    // realistic, and a barcode-rank cut is exactly the kind of number that
    // should be identical cohort-wide.
    var bcBox = $('ix-bcval'), bcSet = $('ix-bcset');
    if (bcBox) {
      var applyBc = function () {
        var raw = bcBox.value.trim();
        if (raw === '') {
          delete state.thresholds[D.bcrank.feature];
          syncBox();
          drawBcrank(); drawHist(); drawScatter(); status();
          return;
        }
        setBcValue(parseFloat(raw));
      };
      bcBox.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') { e.preventDefault(); applyBc(); }
      });
      if (bcSet) bcSet.onclick = applyBc;
    }
  }

  // --- doublet call overlap (Venn) -----------------------------------------
  // A dropdown plot like the histograms and scatters, drawn on the shared QC
  // canvas (hctx) from drawHist's dispatch -- not a standalone widget. SOLO left,
  // Scrublet right, over the cells both methods scored. The three disjoint
  // regions are click targets; clicking one toggles its cells into removal. The
  // lens is the SOLO-AND-Scrublet consensus (flag_doublet_consensus in the
  // artifact), which the panel otherwise gave no way to act on -- a reviewer
  // could threshold either score, never select the agreement set.
  var vennGeom = null;
  var VENN_COLORS = { solo_only: '#1f77b4', scrublet_only: '#ff7f0e', both: '#d62728' };
  var VENN_ORDER = ['solo_only', 'scrublet_only', 'both'];
  var VENN_NAMES = { solo_only: 'SOLO only', scrublet_only: 'Scrublet only', both: 'both' };

  function hexA(hex, a) {
    var n = parseInt(hex.slice(1), 16);
    return 'rgba(' + ((n >> 16) & 255) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')';
  }

  // Successive canvas clips intersect; an even-odd rect-with-a-circular-hole
  // clips to "everything except that circle", so (left circle) ∩ (not right
  // circle) is the SOLO-only crescent, and symmetrically for Scrublet-only.
  // Painting each region under its own clip keeps the crescents and lens exact.
  function vennClip(rk, g) {
    if (rk === 'both') {
      hctx.beginPath(); hctx.arc(g.cxL, g.cy, g.r, 0, 6.283); hctx.clip();
      hctx.beginPath(); hctx.arc(g.cxR, g.cy, g.r, 0, 6.283); hctx.clip();
      return;
    }
    var keep = rk === 'solo_only' ? g.cxL : g.cxR;
    var hole = rk === 'solo_only' ? g.cxR : g.cxL;
    hctx.beginPath(); hctx.arc(keep, g.cy, g.r, 0, 6.283); hctx.clip();
    hctx.beginPath();
    hctx.rect(-10, -10, g.w + 20, g.h + 20);
    hctx.arc(hole, g.cy, g.r, 0, 6.283);
    hctx.clip('evenodd');
  }

  function vennRegionAt(x, y) {
    if (!vennGeom) return null;
    var g = vennGeom;
    var inL = Math.hypot(x - g.cxL, y - g.cy) <= g.r;
    var inR = Math.hypot(x - g.cxR, y - g.cy) <= g.r;
    if (inL && inR) return 'both';
    if (inL) return 'solo_only';
    if (inR) return 'scrublet_only';
    return null;
  }

  // Drawn on the QC-plot canvas from drawHist; `size` is that canvas's CSS size,
  // already cleared by drawHist. Sets vennGeom for the hist canvas's click/hover
  // hit-testing (cleared again by drawHist when a different plot is shown).
  function drawVenn(size) {
    var d = D.doublets;
    if (!d) return;
    var r = Math.min(size.h * 0.32, size.w * 0.24);
    var cy = size.h * 0.48, cxL = size.w / 2 - r * 0.62, cxR = size.w / 2 + r * 0.62;
    var g = vennGeom = { cxL: cxL, cxR: cxR, cy: cy, r: r, w: size.w, h: size.h };
    var counts = { solo_only: d.solo_only.length, scrublet_only: d.scrublet_only.length,
                   both: d.both.length };

    VENN_ORDER.forEach(function (rk) {
      hctx.save(); vennClip(rk, g);
      var sel = state.doubletRegions[rk] && counts[rk];
      hctx.fillStyle = sel ? 'rgba(221,51,51,.55)' : hexA(VENN_COLORS[rk], 0.20);
      hctx.fillRect(0, 0, size.w, size.h);
      hctx.restore();
    });

    hctx.lineWidth = 2;
    hctx.strokeStyle = VENN_COLORS.solo_only;
    hctx.beginPath(); hctx.arc(cxL, cy, r, 0, 6.283); hctx.stroke();
    hctx.strokeStyle = VENN_COLORS.scrublet_only;
    hctx.beginPath(); hctx.arc(cxR, cy, r, 0, 6.283); hctx.stroke();

    hctx.textAlign = 'center';
    function label(rk, x) {
      var sel = state.doubletRegions[rk] && counts[rk];
      hctx.font = 'bold 12px sans-serif';
      hctx.fillStyle = sel ? '#d33' : VENN_COLORS[rk];
      hctx.fillText(VENN_NAMES[rk], x, cy - 3);
      hctx.font = '11px sans-serif'; hctx.fillStyle = sel ? '#d33' : '#888';
      hctx.fillText(counts[rk].toLocaleString() + (sel ? ' ✓ removing' : ' cells'), x, cy + 13);
    }
    label('solo_only', cxL - r * 0.42);
    label('scrublet_only', cxR + r * 0.42);
    label('both', size.w / 2);

    hctx.font = '11px sans-serif'; hctx.fillStyle = '#888';
    hctx.fillText('SOLO', cxL - r * 0.42, cy - r - 6);
    hctx.fillText('Scrublet', cxR + r * 0.42, cy - r - 6);
    hctx.fillText('click a region to toggle it for removal', size.w / 2, size.h - 6);
    hctx.textAlign = 'left';
    hist.style.cursor = 'default';
  }

  // Toggle the region under a click on the hist canvas. Wired from the hist
  // pointer/click handlers only while the venn plot is the current one.
  function vennToggleAt(clientX, clientY) {
    var b = hist.getBoundingClientRect();
    var rk = vennRegionAt(clientX - b.left, clientY - b.top);
    if (!rk || !D.doublets[rk].length) return;
    state.doubletRegions[rk] = !state.doubletRegions[rk];
    drawHist(); drawScatter(); status();
  }

  // Union of the selected Venn regions' barcodes, de-duplicated (regions are
  // disjoint, so there is nothing to de-dup between them, but a defensive set
  // keeps it robust if that ever changes).
  function doubletBarcodes() {
    if (!D.doublets) return [];
    var seen = {}, out = [];
    VENN_ORDER.forEach(function (rk) {
      if (!state.doubletRegions[rk]) return;
      (D.doublets[rk] || []).forEach(function (i) {
        var bc = D.barcodes[i];
        if (!(bc in seen)) { seen[bc] = 1; out.push(bc); }
      });
    });
    return out;
  }

  function selectedRegionLabel() {
    return VENN_ORDER.filter(function (rk) { return state.doubletRegions[rk]; })
                     .map(function (rk) { return VENN_NAMES[rk]; }).join(', ');
  }

  // --- decisions ------------------------------------------------------------
  function decisions() {
    var out = [], f;
    for (f in state.thresholds) {
      out.push({ kind: 'threshold', feature: f, op: state.thresholds[f].op,
                 value: Number(state.thresholds[f].value.toFixed(6)), units: 'raw' });
    }
    var ex = Object.keys(state.excluded), re = Object.keys(state.rescued);
    if (ex.length) out.push({ kind: 'exclude_barcodes', barcodes: ex, source_plot: embSel.value });
    // The Venn selection is a second exclude_barcodes decision, tagged so its
    // list entry and undo are distinct from a lasso's. refilter ORs every
    // exclude_barcodes together, so two of them compose without special-casing.
    var db = doubletBarcodes();
    if (db.length) out.push({ kind: 'exclude_barcodes', barcodes: db, source_plot: 'doublet_venn' });
    if (re.length) out.push({ kind: 'rescue_barcodes', barcodes: re, source_plot: embSel.value });
    return out;
  }

  function renderDecisions() {
    var ul = $('ix-decisions'); ul.innerHTML = '';
    decisions().forEach(function (d) {
      var li = document.createElement('li'), txt = document.createElement('span');
      var isVenn = d.kind === 'exclude_barcodes' && d.source_plot === 'doublet_venn';
      if (d.kind === 'threshold') txt.textContent = d.feature + ' ' + (d.op === 'max' ? '>' : '<') + ' ' + d.value + '  → remove';
      else if (isVenn) txt.textContent = 'Doublet overlap (' + selectedRegionLabel() + ') → remove ' + d.barcodes.length + ' cells';
      else txt.textContent = (d.kind === 'exclude_barcodes' ? 'Exclude ' : 'Rescue ') + d.barcodes.length + ' selected cells';
      var btn = document.createElement('button');
      btn.textContent = 'undo';
      btn.onclick = function () {
        if (d.kind === 'threshold') delete state.thresholds[d.feature];
        else if (isVenn) state.doubletRegions = {};
        else if (d.kind === 'exclude_barcodes') state.excluded = {};
        else state.rescued = {};
        drawHist(); drawScatter(); drawBcrank(); syncBox(); status();
      };
      li.appendChild(txt); li.appendChild(btn); ul.appendChild(li);
    });
    if (!ul.children.length) ul.innerHTML = '<li><span>No manual decisions yet.</span></li>';
  }

  $('ix-export').onclick = function () {
    var payload = {
      schema_version: '1.0',
      sample: D.sample,
      parent_artifact: D.sample + '_qcfilters.h5ad',
      parent_config_hash: D.parent_config_hash,
      created: new Date().toISOString(),
      decisions: decisions()
    };
    var blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = D.sample + '_decisions.json';
    a.click();
    URL.revokeObjectURL(a.href);
  };

  $('ix-reset').onclick = function () {
    state.thresholds = {}; state.excluded = {}; state.rescued = {}; state.doubletRegions = {};
    drawHist(); drawScatter(); drawBcrank(); syncBox(); status();
  };
  embSel.onchange = drawScatter;
  clustSel.onchange = drawScatter;
  plotSel.onchange = function () { plotLasso = null; drawHist(); syncBox(); };
  $('ix-op').onchange = function () {
    var f = histMeta ? histMeta.feature : null;
    if (state.thresholds[f]) { state.thresholds[f].op = this.value; drawHist(); drawScatter(); status(); }
  };
  $('ix-set').onclick = applyBox;
  $('ix-val').addEventListener('keydown', function (e) {
    if (e.key === 'Enter') { e.preventDefault(); applyBox(); }
  });
  window.addEventListener('resize', function () { drawScatter(); drawHist(); drawBcrank(); });

  drawScatter(); drawHist(); drawBcrank(); syncBox(); status();
})();
"""


def render_section(payload: dict) -> str:
    """Render the interactive section's HTML from a payload.

    Args:
        payload (dict): As returned by `build_payload`.

    Returns:
        str: HTML fragment, with the payload inlined as JSON.
    """
    if not payload.get('embeddings') and not payload.get('features'):
        return ('<h2>Interactive review</h2><p class="note">No embedding or numeric QC feature '
                'is available in this artifact, so there is nothing to select on. Embeddings are '
                'written from schema 1.2 onward; re-run this sample\'s QC plotting step '
                '(<code>--force-plotting</code>) to add them.</p>')

    # payload['embeddings'] is built scVI-first, so the first <option> -- the
    # one the select opens on -- is the scVI embedding whenever it exists.
    emb_options = ''.join('<option value="{}">{}</option>'.format(k, v['label'])
                          for k, v in payload['embeddings'].items())
    # The axis is named in the option itself, because the ticks alone do not say
    # whether the spacing is linear -- and on a logit axis the difference between
    # 0.02 and 0.05 is most of the width.
    _axis = {'logit': ' [logit axis]', 'log1p': ' [log1p axis]'}
    plot_options = ''.join(
        '<option value="{}">{}{}{}</option>'.format(
            i, html.escape(pl['label']), _axis.get(pl.get('scale'), ''),
            ' (advisory)' if pl['advisory'] else '')
        for i, pl in enumerate(payload.get('plots', [])))
    clust_options = '<option value="">no cluster colouring</option>' + ''.join(
        '<option value="{k}">{k}</option>'.format(k=k) for k in payload['clusterings'])

    no_emb = '' if payload['embeddings'] else (
        '<p class="note">This artifact carries no embedding, so lasso selection is unavailable; '
        'thresholds below still work.</p>')

    # Not a standalone section: the Venn is an entry in the QC-plot dropdown
    # (added to payload['plots'] by build_payload), so this is just the sentence
    # in the plots description that tells the reviewer it is there and clickable.
    venn_hint = ''
    dbl = payload.get('doublets') or {}
    if dbl:
        adv = [name for name, key in (('SOLO', 'solo_advisory'), ('Scrublet', 'scrublet_advisory'))
               if dbl.get(key)]
        if len(adv) == 2:
            adv_note = (' Both are advisory here (neither removes cells automatically), so this is '
                        'how you act on their calls by hand.')
        elif adv:
            adv_note = ' {} is advisory here (removes no cells automatically).'.format(adv[0])
        else:
            adv_note = ''
        venn_hint = (' The <strong>Doublet call overlap</strong> entry draws the SOLO &cap; Scrublet '
                     'Venn over the {n_assessed:,} cells both methods scored &mdash; click a region '
                     '(the lens is the consensus) to mark those cells for removal.{adv_note}').format(
                        n_assessed=int(dbl.get('n_assessed', 0)), adv_note=adv_note)

    bcrank_block = ''
    if payload.get('bcrank'):
        bcrank_block = (
            '<h3>Barcode rank</h3>'
            '<p class="sub">Drag to move the cut-off. Moving it <strong>right</strong> (removing '
            'more) applies directly. Moving it <strong>left</strong> admits cells the pipeline '
            'already removed &mdash; those were never seen by MAD, miQC or regression, so their '
            'flags are <code>False</code> by default rather than "passed". That direction cannot '
            'be applied as a decision; the panel will show you the config change to make '
            'instead.</p>'
            '<canvas id="ix-bcrank" class="ix-hist" style="height:260px"></canvas>'
            '<div class="ix-controls" style="margin-top:.6rem">'
            '<label>Keep cells with <code>{feature}</code> &ge; '
            '<input id="ix-bcval" type="number" step="any" placeholder="counts"></label>'
            '<button id="ix-bcset">Set</button>'
            '</div>'
            '<div id="ix-bcrank-warn"></div>').format(
                feature=html.escape(str(payload['bcrank'].get('feature', ''))))

    return """
<h2>Interactive review</h2>
<p class="note">Adjust the automatic filtering by hand, then <strong>Export decisions</strong> and
apply them with<br><code>python -m mapping.refilter -c configs/pipeline_config.yaml --sample {sample}</code>
<br>That writes <code>{sample}_filtered_manual.h5ad</code> <em>alongside</em> the automatic
artifact &mdash; nothing here overwrites it. Point integration at the result with
<code>--artifact-key filtered_manual_h5ad</code>. Decisions are a complete description of what
you want, not a running diff, so re-exporting after an undo genuinely undoes it.</p>
{no_emb}
<div class="ix-wrap">
  <div class="ix-controls">
    <label>Embedding <select id="ix-emb">{emb}</select></label>
    <label>Colour by <select id="ix-clust">{clust}</select></label>
    <button id="ix-reset">Reset all</button>
    <button id="ix-export" class="primary">Export decisions</button>
  </div>
  <div class="ix-legend">
    <span class="ix-kept">kept</span><span class="ix-cut">removed by your decisions</span>
    <span class="ix-auto">already removed automatically</span>
  </div>
  <canvas id="ix-scatter" class="ix-canvas"></canvas>
  <h3>QC diagnostic plots</h3>
  <p class="sub">The same figures as the Quality control section above, interactive &mdash; same
  axes, same units. Histograms take a cut-off, either by dragging or by typing an exact value;
  scatters (miQC, regression, DoD) have no single axis to threshold on, so select on those by
  lasso. <strong>Type the value when you want the same cut-off on several samples</strong> &mdash;
  dragging will not land on 0.5 twice.{venn_hint}</p>
  <p class="sub">Where a plot is marked <code>[logit axis]</code> or <code>[log1p axis]</code> the
  <em>spacing</em> is the transformed one MAD was fitted on, so the shape matches the figure above;
  the tick labels, the value you type and the exported decision are all in the feature's own raw
  units.</p>
  <div class="ix-controls" style="margin-top:.8rem">
    <label>Plot <select id="ix-plot">{plots}</select></label>
    <label>Remove cells <select id="ix-op">
      <option value="max">above the line</option>
      <option value="min">below the line</option>
    </select></label>
    <label>at <input id="ix-val" type="number" step="any" placeholder="value"></label>
    <button id="ix-set">Set</button>
  </div>
  <canvas id="ix-hist" class="ix-hist" style="height:320px"></canvas>
  {bcrank}
  <p class="ix-status" id="ix-status"></p>
  <h3>Pending decisions</h3>
  <ul class="ix-list" id="ix-decisions"></ul>
</div>
<script>window.__IX_DATA__ = {data};</script>
<script>{js}</script>
""".format(sample=payload['sample'], no_emb=no_emb, emb=emb_options, clust=clust_options,
           plots=plot_options, venn_hint=venn_hint,
           # separators= drops the whitespace json.dumps would otherwise put in a
           # payload with one entry per cell.
           bcrank=bcrank_block,
           data=json.dumps(payload, separators=(',', ':')),
           js=_JS)


def interactive_css() -> str:
    """The interactive section's stylesheet, appended to the report's own."""
    return _CSS
