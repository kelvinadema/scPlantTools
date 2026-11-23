#!/usr/bin/env python3

"""Per-sample doublet detection for the mapping stage QC cascade.

Runs SOLO (subprocess, scvi env), Scrublet (in-process) and their consensus.
Calls land in `.uns['qc']` and `.obs`; `solo.mode` in filter_config.yaml decides
whether they remove cells.

See docs/decisions/doublet-detection.md.
"""

import json
import logging
import time
import os
import subprocess

import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.mixture import GaussianMixture

from pathlib import Path

from shared.utils import load_config, log_exceptions, write_done_flag, get_sample_output_paths
from shared.samples import load_samples
from shared.io import write_h5ad
from shared.config_spec import FILTER_MODES
from shared.dimred import load_cached_hvg
from shared.hvg import HVGSelector
from .qc_filter import collect_removal_flags

# Runs under the integration env; invoked as `python -m` from the repo root.
_WORKER_MODULE = 'integration.doublet_model'

# Written onto every cell, including those the model never saw (see run()).
_RESULT_COLUMNS = ('solo_doublet_score', 'solo_doublet', 'leiden_scVI')


class DetectDoublets():
    """Detect doublets for one sample."""

    def __init__(self, sample_name: str, sample_base_dir: Path, sample_config: Path,
                 filter_config_path: Path, pipeline_config_path: Path):
        self.logger = logging.getLogger('SingleCellMappingPipeline')

        self.sample_name = sample_name
        self.sample_cfg = load_samples(sample_config)
        entry = self.sample_cfg[self.sample_name]
        self.mapper = entry.get('mapper', '').lower()
        self.config_paths = [sample_config, filter_config_path, pipeline_config_path]

        self.filter_profile = entry.get('qc_filter', 'default')
        filter_cfg = load_config(filter_config_path) or {}
        self.filter_cfg = (filter_cfg.get('QCfilterconfigs') or {}).get(self.filter_profile, {})
        # Three independent methods; collect_removal_flags ORs the flags of every
        # 'filter'-mode method. All advisory (the default) removes nothing.
        _filters = (self.filter_cfg.get('filters') or {})
        self.solo_cfg = _filters.get('solo', {}) or {}
        self.scrublet_cfg = _filters.get('scrublet', {}) or {}
        self.consensus_cfg = _filters.get('doublet_consensus', {}) or {}

        self.pipeline_cfg = load_config(pipeline_config_path) or {}
        self.doublet_cfg = self.pipeline_cfg.get('doublets') or {}
        # Model knobs live under doublets.scrublet; the filter block holds only
        # mode/threshold.
        self.scrublet_model_cfg = self.doublet_cfg.get('scrublet') or {}
        # Set when sample_quality forces this run to advisory; recorded in .uns.
        self._downgraded = False

        self.paths = get_sample_output_paths(Path(sample_base_dir) / self.sample_name,
                                             mapper=self.mapper)
        self.out_dir = self.paths['processed']
        self.qc_dir = self.paths['qc']
        self.adata_path = self.out_dir / '{}_qcfilters.h5ad'.format(self.sample_name)

    # -- helpers ------------------------------------------------------------

    _MODE_CFG = {'solo': 'solo_cfg', 'scrublet': 'scrublet_cfg',
                 'doublet_consensus': 'consensus_cfg'}

    def _mode(self, method: str = 'solo') -> str:
        """This method's configured execution mode: 'off', 'advisory' or 'filter'."""
        cfg = getattr(self, self._MODE_CFG[method])
        mode = cfg.get('mode', 'off')
        if mode not in FILTER_MODES:
            raise ValueError(
                '[DOUBLETS] filter "{}" in profile "{}" has mode {!r}; expected one of {}.'.format(
                    method, self.filter_profile, mode, ', '.join(repr(m) for m in FILTER_MODES)))
        return mode

    def _effective_mode(self, method: str, quality_flagged: bool):
        """Resolve a method's effective mode after the sample_quality downgrade.

        Returns (mode, downgrade_reason); the reason is empty when nothing was
        downgraded.
        """
        configured = self._mode(method)
        if quality_flagged and configured == 'filter':
            return 'advisory', (
                'sample_quality flagged this sample, so {} fit its threshold on a population too '
                'small or too shallow to trust; its calls are recorded for review but removed no '
                'cells'.format(method))
        return configured, ''

    def _env_python(self) -> Path:
        """Path to the interpreter in the doublet worker's conda env.

        scvi-tools cannot be installed alongside the mapping stage, so the worker runs
        out of process. See docs/decisions/doublet-detection.md.
        """
        env = os.environ.get('DOUBLET_ENV') or self.doublet_cfg.get('env')
        if not env:
            raise RuntimeError(
                '[DOUBLETS] solo is enabled but no doublets.env is configured. scvi-tools cannot '
                'be installed alongside the mapping stage (pandas 3.x), so the worker needs its '
                'own conda PREFIX -- set doublets.env in pipeline_config.yaml or $DOUBLET_ENV.')
        python = Path(env) / 'bin' / 'python'
        if not python.exists():
            raise RuntimeError(
                '[DOUBLETS] doublets.env {} has no bin/python (it must be the conda PREFIX, the '
                'directory containing bin/, not the binary).'.format(env))
        return python

    def _record(self, adata, method: str, metadata: dict, mode: str,
                downgrade_reason: str = '') -> None:
        """Write `.uns['qc'][method]`, stamping the effective exec_mode.

        `mode` is the effective mode; where it differs from the configured one,
        `downgrade_reason` (or a generic reason) is recorded with it.
        """
        if 'qc' not in adata.uns:
            adata.uns['qc'] = {}
        metadata = dict(metadata)
        configured = self._mode(method)
        metadata['exec_mode'] = mode
        if mode != configured:
            metadata['downgraded_from'] = configured
            metadata['downgrade_reason'] = downgrade_reason or (
                '{} ran on a population too small or too shallow to trust; its calls are '
                'recorded for manual review but removed no cells'.format(method))
        adata.uns['qc'][method] = metadata

    def _skip(self, adata, method: str, reason: str) -> None:
        """Record a skip in the shape every other filter uses."""
        self.logger.warning('[DOUBLETS] Skipping {} for sample "{}": {}'.format(
            method, self.sample_name, reason))
        self._record(adata, method, {'skipped': True, 'skip_reason': reason, 'features': []},
                     mode=self._mode(method))

    def _training_cells(self, adata) -> np.ndarray:
        """Boolean mask of cells the model should train on.

        Reuses the cascade's own verdict: flags of every method in exec_mode 'filter'.
        """
        keep = np.ones(adata.n_obs, dtype=bool)
        for flag in collect_removal_flags(adata.uns.get('qc', {})):
            if flag not in adata.obs:
                continue
            keep &= ~adata.obs[flag].astype(bool).values
        return keep

    # Keys this stage owns, cleared each run so a dropped method leaves nothing
    # behind. The cascade's own .uns['qc'] methods are untouched.
    _OWNED_OBS = ('solo_doublet_score', 'solo_doublet', 'flag_solo', 'leiden_scVI',
                  'scrublet_doublet_score', 'flag_scrublet', 'flag_doublet_consensus')
    _OWNED_OBSM = ('X_umap_scVI',)
    _OWNED_UNS = ('solo', 'scrublet', 'doublet_consensus')

    def _clear_previous(self, adata) -> None:
        """Drop any doublet columns and records left by an earlier run, in place."""
        qc = adata.uns.get('qc') or {}
        for key in self._OWNED_UNS:
            qc.pop(key, None)
        for col in self._OWNED_OBS:
            if col in adata.obs:
                del adata.obs[col]
        for key in self._OWNED_OBSM:
            if key in adata.obsm:
                del adata.obsm[key]

    # -- entrypoint ---------------------------------------------------------

    @log_exceptions()
    def run(self, force: bool=False):
        """Run every enabled doublet method for this sample.

        Returns None if detection completed or was deliberately skipped, or the
        failure reason if the SOLO worker faulted. Only a fault fails the run.
        """
        done_path = self.paths['done'] / 'doublets.done'
        if done_path.exists() and not force:
            self.logger.info('[DOUBLETS] skipping sample "{}"; doublets.done exists '
                             '(use --force-doublets to override)'.format(self.sample_name))
            return

        solo_mode = self._mode('solo')
        scrublet_mode = self._mode('scrublet')
        consensus_mode = self._mode('doublet_consensus')
        # The consensus is derived from the other two, so with both sources off
        # there is nothing to compute regardless of consensus_mode.
        if solo_mode == 'off' and scrublet_mode == 'off':
            self.logger.info('[DOUBLETS] solo and scrublet are both "off" for profile "{}"; not '
                             'running for sample "{}"'.format(self.filter_profile, self.sample_name))
            return

        adata = sc.read_h5ad(self.adata_path)
        self._clear_previous(adata)

        # Flagged sample: 'filter' -> 'advisory'. Still runs and records; removes
        # nothing.
        quality = (adata.uns.get('qc') or {}).get('sample_quality') or {}
        quality_flagged = bool(quality.get('gate_applied',
                                           quality.get('flagged_for_manual_review', False)))
        self._downgraded = quality_flagged

        train_mask = self._training_cells(adata)
        n_train = int(train_mask.sum())

        # ---- SOLO (subprocess under the scvi env) ----
        solo_ran = False
        solo_fault = None
        if solo_mode == 'off':
            self.logger.info('[DOUBLETS] solo.mode is "off"; skipping SOLO for "{}"'.format(
                self.sample_name))
        else:
            min_cells = int(self.doublet_cfg.get('min_cells', 200))
            if n_train < min_cells:
                # Technical floor: below ~10 cells scvi has no validation split.
                self._skip(adata, 'solo',
                           'only {} cells passed QC, below the {}-cell floor a per-sample '
                           'scVI/SOLO model needs to converge (doublets.min_cells)'.format(
                               n_train, min_cells))
            else:
                solo_eff, solo_reason = self._effective_mode('solo', quality_flagged)
                self.logger.info('[DOUBLETS] Sample "{}": training SOLO on {} QC-passed cells of '
                                 '{}'.format(self.sample_name, n_train, adata.n_obs))
                try:
                    summary, results = self._run_worker(adata, train_mask)
                except Exception as e:
                    # A fault, not a skip: done flag withheld so the next run retries, and the
                    # reason is returned so main.py exits 1.
                    self.logger.error(
                        '[DOUBLETS] SOLO FAILED for sample "{}". The rest of QC is unaffected and '
                        'the report shows the fault, but no SOLO calls exist and the run will exit '
                        'non-zero. "doublets.done" is deliberately not written, so re-running '
                        'retries without --force-doublets. Reason: {}'.format(self.sample_name, e))
                    self._record(adata, 'solo',
                                 {'skipped': True, 'failed': True, 'features': [],
                                  'failure_reason': str(e),
                                  'skip_reason': 'the SOLO worker failed (see failure_reason)'},
                                 mode=solo_eff, downgrade_reason=solo_reason)
                    solo_fault = str(e)
                else:
                    self._apply_solo(adata, results, summary, solo_eff, n_train, solo_reason)
                    solo_ran = True

        # ---- Scrublet (in-process, pure CPU, no env boundary) ----
        scrublet_ran = False
        if scrublet_mode == 'off':
            self.logger.info('[DOUBLETS] scrublet.mode is "off"; skipping Scrublet for "{}"'.format(
                self.sample_name))
        else:
            scr_min = int(self.scrublet_model_cfg.get('min_cells', 50))
            if n_train < scr_min:
                self._skip(adata, 'scrublet',
                           'only {} cells passed QC, below the {}-cell floor Scrublet needs to '
                           'simulate doublets and fit its PCA (doublets.scrublet.min_cells)'.format(
                               n_train, scr_min))
            else:
                scr_eff, scr_reason = self._effective_mode('scrublet', quality_flagged)
                self.logger.info('[DOUBLETS] Sample "{}": running Scrublet on {} QC-passed '
                                 'cells'.format(self.sample_name, n_train))
                try:
                    scores, scr_suggested = self._run_scrublet(adata, train_mask)
                except Exception as e:
                    # A Scrublet failure is a recorded skip, not a run fault. Retry with
                    # --force-doublets.
                    self.logger.warning(
                        '[DOUBLETS] Scrublet failed for sample "{}": {}. Recorded as a skip; SOLO '
                        'and the rest of QC are unaffected.'.format(self.sample_name, e))
                    self._record(adata, 'scrublet',
                                 {'skipped': True, 'failed': True, 'features': [],
                                  'failure_reason': str(e),
                                  'skip_reason': 'Scrublet failed (see failure_reason)'},
                                 mode=scr_eff, downgrade_reason=scr_reason)
                else:
                    self._apply_scrublet(adata, scores, scr_suggested, scr_eff, n_train, scr_reason)
                    scrublet_ran = True

        # ---- Consensus (strictly requires BOTH) ----
        if consensus_mode != 'off' and solo_ran and scrublet_ran:
            cons_eff, cons_reason = self._effective_mode('doublet_consensus', quality_flagged)
            self._apply_consensus(adata, cons_eff, cons_reason)
        elif consensus_mode != 'off':
            self.logger.info(
                '[DOUBLETS] consensus requires both SOLO and Scrublet; only {} produced calls for '
                '"{}", so no consensus entry is recorded.'.format(
                    'SOLO' if solo_ran else ('Scrublet' if scrublet_ran else 'neither'),
                    self.sample_name))

        # ---- write ----
        if solo_fault is not None:
            # Persisted so the report can render the fault, but WITHOUT the done
            # flag -- hence _persist, not _write.
            self._persist(adata)
            return solo_fault
        self._write(adata, done_path)
        return None

    def _select_hvgs(self, sub, n_hvg):
        """`sub` restricted to `n_hvg` genes chosen by the consensus HVG selector
        (cached per sample), falling back to the highest-variance log1p genes."""
        outfn = self.qc_dir / '{}_doublet_hvg_consensus.tsv'.format(self.sample_name)

        cached = load_cached_hvg(sub, outfn, n_hvg, self.logger)
        if cached is not None:
            return cached

        try:
            selector = HVGSelector(sub, outfn=str(outfn), n_hvg=n_hvg, logger=self.logger)
            # batch_key=None: one sample, so there is no .obs['batch'] to
            # aggregate over and the class default would raise a KeyError.
            return selector.select_hvg(batch_key=None)
        except Exception as e:
            self.logger.warning(
                '[DOUBLETS] consensus HVG selection failed ({}); falling back to the {} '
                'highest-variance genes on log1p counts, which needs no curve fit'.format(e, n_hvg))

        X = sub.X
        logged = np.log1p(X.toarray() if hasattr(X, 'toarray') else np.asarray(X))
        order = np.argsort(logged.var(axis=0))[::-1][:n_hvg]
        return sub[:, np.sort(order)].copy()

    def _run_worker(self, adata, train_mask):
        """Run the SOLO worker subprocess; returns (parsed summary, per-cell results)."""
        layer = self.doublet_cfg.get('layer', 'raw_counts')
        n_hvg = int(self.doublet_cfg.get('hvg_n_top_genes', 2000))

        sub = adata[train_mask].copy()
        if layer and layer in sub.layers:
            sub.X = sub.layers[layer].copy()
        elif layer:
            self.logger.warning(
                '[DOUBLETS] layer "{}" not present for sample "{}"; training on .X instead. '
                '(scVI expects counts -- check doublets.layer in pipeline_config.yaml.)'.format(
                    layer, self.sample_name))
        # Layers and .obsm are irrelevant to the worker and would bloat a file
        # that crosses a process and an environment boundary.
        sub = sc.AnnData(X=sub.X, obs=pd.DataFrame(index=sub.obs_names.astype(str)),
                         var=pd.DataFrame(index=sub.var_names.astype(str)))

        # HVG selection happens here, not in the worker, so the file handed over
        # carries a few thousand genes rather than the whole matrix.
        if n_hvg and n_hvg < sub.n_vars:
            sub = self._select_hvgs(sub, n_hvg)

        tmp_input = self.out_dir / '_temp_doublet_input.h5ad'
        tmp_output = self.out_dir / '_temp_doublet_output.tsv'
        sub.write_h5ad(tmp_input)

        cmd = [str(self._env_python()), '-m', _WORKER_MODULE,
               '--input', str(tmp_input),
               '--output', str(tmp_output),
               '--sample', self.sample_name,
               '--random-state', str(int(self.doublet_cfg.get('random_state', 42))),
               '--doublet-threshold', str(float(self.solo_cfg.get('doublet_threshold', 0.5)))]
        if self.doublet_cfg.get('tune', False):
            # No architecture flags here: the search space owns them.
            cmd += ['--tune',
                    '--n-trials', str(int(self.doublet_cfg.get('n_trials', 20))),
                    '--tune-metric', str(self.doublet_cfg.get('tune_metric', 'elbo_validation'))]
        else:
            cmd += ['--n-latent', str(int(self.doublet_cfg.get('n_latent', 10))),
                    '--n-hidden', str(int(self.doublet_cfg.get('n_hidden', 128))),
                    '--n-layers', str(int(self.doublet_cfg.get('n_layers', 2)))]
        if self.doublet_cfg.get('max_epochs') is not None:
            cmd += ['--max-epochs', str(int(self.doublet_cfg['max_epochs']))]
        if self.doublet_cfg.get('n_epochs_kl_warmup') is not None:
            cmd += ['--n-epochs-kl-warmup', str(int(self.doublet_cfg['n_epochs_kl_warmup']))]
        if self.doublet_cfg.get('use_cuda', True):
            cmd.append('--cuda')
        cmd += [str(a) for a in (self.doublet_cfg.get('extra_args') or [])]

        params = {'n_latent': int(self.doublet_cfg.get('n_latent', 10)),
                  'n_hidden': int(self.doublet_cfg.get('n_hidden', 128)),
                  'n_layers': int(self.doublet_cfg.get('n_layers', 2)),
                  'hvg_n_top_genes': sub.n_vars,
                  'tune': bool(self.doublet_cfg.get('tune', False))}
        self.logger.info(
            '[DOUBLETS] Training scVI for "{}" on {} cells x {} genes ({}); this is the long '
            'step.'.format(self.sample_name, sub.n_obs, sub.n_vars,
                           'autotuning, {} trials'.format(self.doublet_cfg.get('n_trials', 20))
                           if params['tune'] else
                           'fixed parameters: {}'.format(
                               ', '.join('{}={}'.format(k, params[k])
                                         for k in ('n_latent', 'n_hidden', 'n_layers')))))
        self.logger.info('[DOUBLETS] {}'.format(' '.join(cmd)))
        started = time.time()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=int(self.doublet_cfg.get('timeout', 7200)))
        if proc.returncode != 0:
            # Capture stderr: under SLURM the parent's goes to a different file than
            # this stage's log.
            stderr = proc.stderr.decode('utf-8', errors='replace').strip().splitlines()
            # Budget must clear several tracebacks -- a tuning run emits one error block
            # per failed trial.
            n_tail = 120
            tail = '\n    '.join(stderr[-n_tail:]) if stderr else '(no stderr)'
            # Temp files are left on failure: the only record of the worker input.
            raise RuntimeError(
                '[DOUBLETS] worker failed for sample "{}" (exit {}). Its input is left at {} for '
                'inspection.\n  Worker stderr (last {} of {} lines):\n    {}'.format(
                    self.sample_name, proc.returncode, tmp_input,
                    len(stderr[-n_tail:]), len(stderr), tail))

        summary = {}
        stdout = proc.stdout.decode('utf-8', errors='replace').strip().splitlines()
        if stdout:
            try:
                summary = json.loads(stdout[-1])
            except ValueError:
                self.logger.warning('[DOUBLETS] could not parse worker summary: {!r}'.format(stdout[-1]))

        # The worker's own progress went to its stderr; surface a trace of it in
        # this run's log rather than only on failure, so a long training step is
        # not silent.
        for line in proc.stderr.decode('utf-8', errors='replace').strip().splitlines()[-6:]:
            if line.strip():
                self.logger.info('[DOUBLETS]   {}'.format(line.strip()))
        self.logger.info('[DOUBLETS] Training finished for "{}" in {:.0f}s'.format(
            self.sample_name, time.time() - started))

        results = pd.read_csv(tmp_output, sep='\t', index_col=0)
        results.index = results.index.astype(str)

        for tmp in (tmp_input, tmp_output):
            tmp.unlink(missing_ok=True)
        return summary, results

    def _model_params_record(self, summary: dict) -> dict:
        """Build the model_params record for `.uns['qc']['solo']`.

        Architecture keys come from the worker summary, not the config: with `tune`
        enabled the search space picks them. `params_source` records which branch
        produced them.
        """
        reported = summary.get('model_params') or {}
        record = {
            'max_epochs': self.doublet_cfg.get('max_epochs'),
            'tuned': bool(summary.get('tuned', self.doublet_cfg.get('tune', False))),
            'n_trials': int(self.doublet_cfg.get('n_trials', 20)) if self.doublet_cfg.get('tune') else None,
            'hvg_n_top_genes': int(self.doublet_cfg.get('hvg_n_top_genes', 2000)),
            'layer': self.doublet_cfg.get('layer', 'raw_counts'),
            'random_state': int(self.doublet_cfg.get('random_state', 42)),
            'covariates': [],
            'scvi_version': summary.get('scvi_version', ''),
        }

        if reported:
            record.update(reported)
            record['params_source'] = 'trained'
        else:
            # No summary to read: an unparseable stdout line, or a worker
            # predating this field. Fall back to the request, but say so -- a
            # tuned run's real architecture is not recoverable from here, and
            # silently presenting the request as fact is the bug this method
            # exists to remove.
            record.update({
                'n_latent': int(self.doublet_cfg.get('n_latent', 10)),
                'n_hidden': int(self.doublet_cfg.get('n_hidden', 128)),
                'n_layers': int(self.doublet_cfg.get('n_layers', 2)),
            })
            record['params_source'] = 'requested (worker reported none)'
            if record['tuned']:
                self.logger.warning(
                    '[DOUBLETS] Sample "{}": worker reported no model_params but tuning was on; '
                    '.uns["qc"]["solo"]["model_params"] records the REQUESTED architecture, which '
                    'the tuner overrode. Treat it as unverified.'.format(self.sample_name))

        # anndata drops None-valued .uns keys on write, so a field would vanish
        # rather than record "not set" (see shared/io.py's same coercion).
        return {k: ('' if v is None else v) for k, v in record.items()}

    def _detect_doublet_threshold(self, scores: np.ndarray, default: float, cfg: dict = None):
        """Choose a doublet cut from this sample's own score distribution.

        Fits a 2-component Gaussian mixture and cuts at the Bayes boundary rather than
        using SOLO's fixed 0.5. See docs/decisions/doublet-detection.md.

        `default` is the fallback when no valley is found; `cfg` (default: the
        SOLO block) supplies the min_component_* guards. Returns (threshold,
        diagnostics); the diagnostics record how the cut was chosen, so a fallback
        is distinguishable from a real fit.
        """
        cfg = self.solo_cfg if cfg is None else cfg
        min_sep = float(cfg.get('min_component_separation', 0.25))
        min_w = float(cfg.get('min_component_weight', 0.05))
        diag = {'method': 'fixed_fallback', 'default_threshold': float(default),
                'min_component_separation': min_sep, 'min_component_weight': min_w}

        scores = np.asarray(scores, dtype=float)
        scores = scores[np.isfinite(scores)]
        if scores.size < 20:
            diag['skip_reason'] = 'too few scored cells ({}) to fit a distribution'.format(scores.size)
            return float(default), diag

        gmm = GaussianMixture(n_components=2, covariance_type='full', n_init=5,
                              random_state=int(self.doublet_cfg.get('random_state', 42)))
        gmm.fit(scores.reshape(-1, 1))

        means = gmm.means_.ravel()
        weights = gmm.weights_.ravel()
        lo, hi = np.argsort(means)          # low mode = singlets, high mode = doublets
        sep = float(means[hi] - means[lo])
        w_lo, w_hi = float(weights[lo]), float(weights[hi])
        diag.update({'component_means': [float(means[lo]), float(means[hi])],
                     'component_weights': [w_lo, w_hi], 'separation': sep})

        if sep < min_sep:
            diag['skip_reason'] = ('component separation {:.2f} < {:.2f} in doublet-probability '
                                   'units: no clear second mode'.format(sep, min_sep))
            return float(default), diag
        if min(w_lo, w_hi) < min_w:
            diag['skip_reason'] = ('a fitted component holds < {:.0%} of cells: the second mode is '
                                   'a numerical sliver, not a doublet population'.format(min_w))
            return float(default), diag

        # Bayes boundary: the point between the two means where the doublet
        # component's posterior overtakes the singlet's. predict_proba accounts for
        # the fitted weights and variances, so this is the valley given the actual
        # fit rather than a naive midpoint of the means.
        grid = np.linspace(means[lo], means[hi], 4001).reshape(-1, 1)
        post_hi = gmm.predict_proba(grid)[:, hi]
        threshold = float(np.clip(grid[np.argmin(np.abs(post_hi - 0.5)), 0], 0.0, 1.0))
        diag['method'] = 'gmm_valley'
        return threshold, diag

    def _apply_solo(self, adata, results, summary: dict, mode: str, n_train: int,
                    downgrade_reason: str = '') -> None:
        """Turn the worker's SOLO results into flags, the scVI UMAP/Leiden, and a record.

        `mode` is the effective mode; it is downgraded to advisory here if the
        doublet rate exceeds `max_plausible_rate`.
        """
        index = adata.obs_names.astype(str)
        aligned = results.reindex(index)

        adata.obs['solo_doublet_score'] = aligned['solo_doublet_score'].astype(float).values

        # Choose the hard-call threshold. SOLO's raw default is a fixed 0.5 on the
        # doublet probability, which over-calls (see _detect_doublet_threshold);
        # with solo.auto_threshold on (the default) the cut is placed at the valley
        # between this sample's singlet and doublet modes instead, and the worker's
        # own 0.5-based `solo_doublet` column is recomputed here rather than used.
        # The fixed value is the fallback when auto is off or the distribution is
        # not clearly bimodal.
        fixed = float(self.solo_cfg.get('doublet_threshold', 0.5))
        if self.solo_cfg.get('auto_threshold', True):
            threshold, thr_diag = self._detect_doublet_threshold(
                adata.obs['solo_doublet_score'].values, fixed)
        else:
            threshold, thr_diag = fixed, {'method': 'disabled', 'default_threshold': fixed}
        self.logger.info(
            '[DOUBLETS] Sample "{}": doublet-call threshold {:.3f} ({})'.format(
                self.sample_name, threshold,
                'auto-detected valley' if thr_diag.get('method') == 'gmm_valley' else
                'fixed {:.3f} -- {}'.format(fixed, thr_diag.get('skip_reason', thr_diag.get('method')))))

        # Cells outside the training set are not doublets as far as anything here
        # knows; False rather than NaN keeps the column boolean so it can feed a
        # removal mask without a dtype surprise. The score stays NaN, which is the
        # honest record of "not assessed".
        score = aligned['solo_doublet_score']
        flags = (score > threshold).where(score.notna(), False).astype(bool)
        adata.obs['flag_solo'] = flags.values
        adata.obs['solo_doublet'] = flags.values
        adata.obs['leiden_scVI'] = aligned['leiden'].astype('object').where(
            aligned['leiden'].notna(), '').astype(str).values

        umap = np.full((adata.n_obs, 2), np.nan, dtype=float)
        have = aligned[['umap1', 'umap2']].notna().all(axis=1).values
        umap[have] = aligned.loc[have, ['umap1', 'umap2']].to_numpy(dtype=float)
        # '_scVI' suffix marks this as a per-sample embedding: integration drops
        # such keys before concatenating, since forty per-sample UMAPs are forty
        # unrelated coordinate spaces.
        adata.obsm['X_umap_scVI'] = umap

        # Record the neighbours/UMAP/Leiden settings the worker resolved to, so
        # the HTML report can describe how this embedding was built. Merged into
        # a shared .uns['umap_params'] map (QCPlotter adds 'X_umap_qc' the same
        # way) rather than overwriting, and None-coerced since anndata drops
        # None-valued .uns keys on write.
        uparams = summary.get('umap_params') or {}
        if uparams:
            store = dict(adata.uns.get('umap_params', {}))
            store['X_umap_scVI'] = {k: ('' if v is None else v) for k, v in uparams.items()}
            adata.uns['umap_params'] = store

        n_flagged = int(flags.sum())
        rate = (n_flagged / n_train) if n_train else 0.0
        self.logger.info('[DOUBLETS] Sample "{}": {} doublets of {} assessed cells ({}); '
                         'mode={}'.format(self.sample_name, n_flagged, n_train,
                                          '{:.1%}'.format(rate) if n_train else 'n/a',
                                          mode))

        # A doublet rate this high is a diagnosis of the model, not of the
        # sample: real 10x doublet rates run ~1-30%, and a per-sample scVI/SOLO
        # model that has not converged calls essentially everything a doublet.
        # Observed on this cohort at 100.0% for every sample under ~400 QC-passed
        # cells, which is what doublets.min_cells now guards. This is the
        # backstop for a sample that clears the floor and still fails to
        # converge: downgrade to advisory so the calls cannot remove cells, and
        # record why, so the report shows the number next to the reason it is
        # not to be trusted rather than on its own.
        unconverged_rate = float(self.doublet_cfg.get('max_plausible_rate', 0.5))
        unconverged = bool(n_train and rate > unconverged_rate)
        if unconverged:
            self.logger.warning(
                '[DOUBLETS] Sample "{}": SOLO called {:.1%} of assessed cells doublets, above the '
                '{:.0%} plausibility ceiling. A per-sample model calling most of its input a '
                'doublet has not converged -- treating the calls as advisory (they remove no '
                'cells) and flagging them in the report. Inspect before trusting them.'.format(
                    self.sample_name, rate, unconverged_rate))
            if mode == 'filter':
                mode = 'advisory'
                downgrade_reason = (
                    'SOLO called {:.1%} of assessed cells doublets, above the {:.0%} '
                    'plausibility ceiling; a model calling most of its input a doublet has not '
                    'converged, so its calls are recorded for review but removed no '
                    'cells'.format(rate, unconverged_rate))

        self._record(adata, 'solo', {
            'unconverged': unconverged,
            'doublet_rate': float(rate),
            'max_plausible_rate': unconverged_rate,
            'flag_col': 'flag_solo',
            'score_column': 'solo_doublet_score',
            'features': ['solo_doublet_score'],
            'n_flagged': n_flagged,
            'n_assessed': n_train,
            'threshold': float(threshold),
            'threshold_method': thr_diag.get('method', 'fixed'),
            'threshold_diagnostics': thr_diag,
            # Recorded here rather than in the integration registry: this model
            # is disposable and per-sample, and the registry has no file locking.
            'model_params': self._model_params_record(summary),
            # Same shape every other filter records, so the report's "figures are
            # drawn on the cells that reached this stage" line works unchanged.
            'run_context': {'filtered_on_flags': collect_removal_flags(adata.uns.get('qc', {}))},
        }, mode=mode, downgrade_reason=downgrade_reason)

    def _run_scrublet(self, adata, train_mask):
        """Run Scrublet in-process on the QC-passed cells.

        Takes raw counts over all genes: Scrublet runs its own variability filter and
        PCA internally. Returns (scores, None): the second slot is always None, and
        the hard call is made by the shared valley detector in `_apply_scrublet`.
        """
        cfg = self.scrublet_model_cfg
        layer = cfg.get('layer', self.doublet_cfg.get('layer', 'raw_counts'))
        sub = adata[train_mask].copy()
        if layer and layer in sub.layers:
            sub.X = sub.layers[layer].copy()
        elif layer:
            self.logger.warning(
                '[DOUBLETS] scrublet layer "{}" not present for sample "{}"; using .X instead. '
                '(Scrublet expects counts -- check doublets.scrublet.layer.)'.format(
                    layer, self.sample_name))

        # Scrublet's simulation and PCA assume counts. CellBender's corrected
        # output is float, so round rather than let downstream normalization treat
        # fractional counts as real.
        X = sub.X
        data = X.data if hasattr(X, 'data') else X
        if getattr(data, 'size', 0) and not np.allclose(data, np.rint(data)):
            if hasattr(X, 'data'):
                X.data = np.rint(X.data)
            else:
                sub.X = np.rint(X)

        # n_prin_comps must stay below the cell count (PCA cannot ask for more
        # components than samples); on a small QC-passed population the default 30
        # can exceed it.
        n_pcs = int(min(int(cfg.get('n_prin_comps', 30)), max(2, sub.n_obs - 1)))
        sc.pp.scrublet(
            sub,
            sim_doublet_ratio=float(cfg.get('sim_doublet_ratio', 2.0)),
            expected_doublet_rate=float(cfg.get('expected_doublet_rate', 0.06)),
            n_prin_comps=n_pcs,
            random_state=int(self.doublet_cfg.get('random_state', 42)),
            # A concrete value, not None: None routes through scikit-image (absent
            # here). We ignore Scrublet's own call and threshold the score
            # ourselves, so this only decides an unused predicted_doublet column.
            threshold=0.5)

        scores = pd.Series(sub.obs['doublet_score'].astype(float).values,
                           index=sub.obs_names.astype(str))
        return scores, None

    def _apply_scrublet(self, adata, scores, scr_suggested, mode: str, n_train: int,
                        downgrade_reason: str = '') -> None:
        """Turn Scrublet scores into flags and record them.

        `mode` is the effective mode; it is downgraded to advisory here if the
        flagged rate exceeds `max_plausible_rate`.
        """
        index = adata.obs_names.astype(str)
        aligned = scores.reindex(index)
        adata.obs['scrublet_doublet_score'] = aligned.astype(float).values

        # Same valley policy as SOLO (shared _detect_doublet_threshold), but its
        # own guard thresholds and its own fixed fallback. Scrublet also proposes
        # its own cut; that is recorded as a cross-check but not used, so the two
        # methods speak one threshold language in the report.
        fixed = self.scrublet_cfg.get('doublet_threshold')
        if fixed is None:
            # Scrublet's own cut if one was returned (_run_scrublet returns none),
            # else a conservative default: when the valley fit is not trusted we
            # want to under-call, not fabricate doublets.
            fixed = scr_suggested if scr_suggested is not None else 0.25
        fixed = float(fixed)
        if self.scrublet_cfg.get('auto_threshold', True):
            threshold, thr_diag = self._detect_doublet_threshold(
                adata.obs['scrublet_doublet_score'].values, fixed, cfg=self.scrublet_cfg)
        else:
            threshold, thr_diag = fixed, {'method': 'disabled', 'default_threshold': fixed}
        thr_diag['scrublet_suggested_threshold'] = scr_suggested
        self.logger.info(
            '[DOUBLETS] Sample "{}": Scrublet call threshold {:.3f} ({})'.format(
                self.sample_name, threshold,
                'auto-detected valley' if thr_diag.get('method') == 'gmm_valley' else
                'fixed {:.3f} -- {}'.format(fixed, thr_diag.get('skip_reason', thr_diag.get('method')))))

        score = aligned
        flags = (score > threshold).where(score.notna(), False).astype(bool)
        adata.obs['flag_scrublet'] = flags.values

        n_flagged = int(flags.sum())
        rate = (n_flagged / n_train) if n_train else 0.0
        self.logger.info('[DOUBLETS] Sample "{}": Scrublet flagged {} of {} assessed cells ({}); '
                         'mode={}'.format(self.sample_name, n_flagged, n_train,
                                          '{:.1%}'.format(rate) if n_train else 'n/a', mode))

        # Same unconverged backstop as SOLO: a per-sample doublet method calling
        # most of its input a doublet has failed, not found a doublet-rich sample.
        unconverged_rate = float(self.doublet_cfg.get('max_plausible_rate', 0.5))
        unconverged = bool(n_train and rate > unconverged_rate)
        if unconverged:
            self.logger.warning(
                '[DOUBLETS] Sample "{}": Scrublet flagged {:.1%} of assessed cells, above the '
                '{:.0%} plausibility ceiling -- treating as advisory and flagging in the '
                'report.'.format(self.sample_name, rate, unconverged_rate))
            if mode == 'filter':
                mode = 'advisory'
                downgrade_reason = (
                    'Scrublet flagged {:.1%} of assessed cells, above the {:.0%} plausibility '
                    'ceiling; its calls are recorded for review but removed no cells'.format(
                        rate, unconverged_rate))

        self._record(adata, 'scrublet', {
            'unconverged': unconverged,
            'doublet_rate': float(rate),
            'max_plausible_rate': unconverged_rate,
            'flag_col': 'flag_scrublet',
            'score_column': 'scrublet_doublet_score',
            'features': ['scrublet_doublet_score'],
            'n_flagged': n_flagged,
            'n_assessed': n_train,
            'threshold': float(threshold),
            'threshold_method': thr_diag.get('method', 'fixed'),
            'threshold_diagnostics': thr_diag,
            'expected_doublet_rate': float(self.scrublet_model_cfg.get('expected_doublet_rate', 0.06)),
            'run_context': {'filtered_on_flags': collect_removal_flags(adata.uns.get('qc', {}))},
        }, mode=mode, downgrade_reason=downgrade_reason)

    def _apply_consensus(self, adata, mode: str, downgrade_reason: str = '') -> None:
        """Record the SOLO-and-Scrublet consensus call.

        A cell is a consensus doublet only where both methods assessed it and both
        called it. Cells either method did not assess are False.
        """
        solo = adata.obs['flag_solo'].fillna(False).astype(bool)
        scr = adata.obs['flag_scrublet'].fillna(False).astype(bool)
        both_assessed = (adata.obs['solo_doublet_score'].notna()
                         & adata.obs['scrublet_doublet_score'].notna())
        cons = (solo & scr & both_assessed)
        adata.obs['flag_doublet_consensus'] = cons.values

        n_flagged = int(cons.sum())
        n_assessed = int(both_assessed.sum())
        n_solo = int((solo & both_assessed).sum())
        n_scr = int((scr & both_assessed).sum())
        self.logger.info(
            '[DOUBLETS] Sample "{}": consensus (SOLO & Scrublet) flagged {} of {} jointly-assessed '
            'cells (SOLO {}, Scrublet {}); mode={}'.format(
                self.sample_name, n_flagged, n_assessed, n_solo, n_scr, mode))

        self._record(adata, 'doublet_consensus', {
            'method': 'intersection',
            'flag_col': 'flag_doublet_consensus',
            'features': ['solo_doublet_score', 'scrublet_doublet_score'],
            'n_flagged': n_flagged,
            'n_assessed': n_assessed,
            'n_solo_doublets': n_solo,
            'n_scrublet_doublets': n_scr,
            'run_context': {'filtered_on_flags': collect_removal_flags(adata.uns.get('qc', {}))},
        }, mode=mode, downgrade_reason=downgrade_reason)

    def _persist(self, adata) -> None:
        """Write the object without marking the step done.

        Used on the SOLO fault path so the next run retries.
        """
        write_h5ad(adata, self.adata_path,
                   sample=self.sample_name, mapper=self.mapper, stage='doublets',
                   config_paths=self.config_paths)

    def _write(self, adata, done_path: Path) -> None:
        """Write the object and mark the step done."""
        self._persist(adata)
        self.logger.info('[DOUBLETS] Wrote doublet results into "{}"'.format(self.adata_path))
        write_done_flag(done_path)
