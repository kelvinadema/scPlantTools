#!/usr/bin/env python3
"""
Stage 3 -- refinement: seed labels -> scVI -> scANVI -> embedding.

One run, from the reviewer's curated grouping: the six marker methods are six
ways of finding genesets, and stage 2 is where a person decides which of them
describe the same population. See docs/refine-design.md for why the six-way
comparison lives in the review page rather than in six tune+fit runs, and
docs/refine-handoff.md for current status.

Two ways in, one training path:

    build_group_seed_labels   the combined run: a cross-method grouping
    build_seed_labels         one method alone, from its own decision file --
                              kept for spot-checking a suspicious call, not
                              the normal route

What a run produces:

    {output_dir}/{sample}/{run}/
        {sample}_{run}.h5ad        cells x genes, with the columns below
        scvi/  scanvi/             the two trained models
        {sample}_{run}_labels.tsv  per-cell evidence, joinable on barcode
        {sample}_{run}_run.json    what actually ran, for the report

and in the h5ad, keyed by `{run}` (`combined` for the normal route):

    .obsm['X_scVI_{run}']      the intermediate latent, kept for comparison
    .obsm['X_scANVI_{run}']    the deliverable -- label-aware, which is the
                               whole reason to run scANVI at all
    .obsm['X_umap_{run}']      2-D, from the scANVI latent
    .obs['leiden_{run}']       clusters on the same graph the UMAP came from
    .obs['seed_{run}']         what was seeded, before the model saw anything
    .obs['seed_support_{run}'] how many methods independently seeded that cell
    .obs['prediction_{run}']   scANVI's raw call
    .obs['annotation_{run}']   the call after safeguards demote it
    .obs['confidence_{run}'], ['margin_{run}'], ['novelty_ratio_{run}'],
    .obs['flag_{run}']         the evidence, kept whole so thresholds can move
                               without retraining
"""

import json
import logging

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from shared.decisions import check_parent, read_decisions
from shared.dimred import DimReducer, UMAPPlotter
from shared.groups import check_parents, read as read_groups

from .labels import (UNASSIGNED, UNKNOWN_PREFIX, build_group_seed_labels,
                     build_seed_labels, summarise)
from .safeguards import apply_safeguards

logger = logging.getLogger('AnnotationLogging')


class RefineError(RuntimeError):
    """A fault in a run. Collected by main.py so one failure is reported, not fatal."""


def _read_scores(path: Path) -> pd.DataFrame:
    """Per-cell geneset scores, as stage 1 wrote them.

    Columns are geneset names and must stay strings: `hotspot_M1` and a bare `3`
    from a cluster-dependent method both appear, and pandas would read the second as
    an integer, after which no lookup against the decision file's string keys
    matches anything.

    Args:
        path (Path): The scores TSV.

    Returns:
        pandas.DataFrame: Cells x genesets, with string column names.
    """
    scores = pd.read_csv(path, sep='\t', index_col=0)
    scores.columns = [str(c) for c in scores.columns]
    return scores


def _name_map(po_ids, ontology_file, ontology_extensions=()) -> dict:
    """{po_id: readable name}, for the human-facing `.obs` columns only.

    The labels scANVI trains on stay ids: two genesets resolving to the same term
    must merge into one class, and names are not unique enough to rely on for that.
    Best-effort -- pronto is absent from the integration env on some installs, and a
    missing display name is not worth losing a run over.

    Args:
        po_ids (iterable[str]): Ontology ids to look up.
        ontology_file (Path): The base OBO.
        ontology_extensions (tuple): Extension OBOs, merged as the annotation stage
            merges them, so a grafted id shows its name too.

    Returns:
        dict: {po_id: name}, empty when pronto is unavailable.
    """
    if not ontology_file or not po_ids:
        return {}
    try:
        import io
        import pronto
        from shared.utils import merged_obo_bytes
    except ImportError:
        logger.info('[REFINE] pronto not importable; annotation columns will carry PO ids')
        return {}
    try:
        ont = pronto.Ontology(io.BytesIO(merged_obo_bytes(ontology_file, ontology_extensions)))
    except Exception as exc:                          # noqa: BLE001 - cosmetic only
        logger.warning('[REFINE] could not read %s (%s); using PO ids', ontology_file, exc)
        return {}
    out = {}
    for po_id in po_ids:
        term = ont.get(po_id)
        if term is not None and term.name:
            out[po_id] = term.name
    return out


class RefineRun:
    """One refinement of one sample: seeds -> scVI -> scANVI -> embedding.

    Normally the **combined** run, seeded from the reviewer's cross-method grouping.
    Passing `method=` switches to the single-method route, which reads that method's
    own per-geneset decision file -- for checking what one method would have said on
    its own, not for production.

    Split into `prepare` and `run` so the expensive half can be skipped: building
    the labels reads a few small files and is where configuration mistakes surface,
    while training is GPU-hours. `--dry-run` stops between the two.
    """

    def __init__(self, sample, adata_path, annotation_dir, output_dir, run='combined',
                 methods=None, batch_key=None, counts_layer='corrected_counts',
                 covariates=None, seeds=None, safeguards=None, model=None, embedding=None,
                 ontology_file=None, ontology_extensions=(), allow_stale=False,
                 groups_path=None, method=None):
        self.sample = sample
        # `run` names the outputs and every .obs/.obsm key. 'combined' for the
        # normal route; a method name when spot-checking one in isolation, so
        # the two can sit side by side in the same tree without collision.
        self.run = method or run
        self.method = method            # set only on the single-method route
        self.methods = list(methods or [])
        self.adata_path = Path(adata_path)
        self.annotation_dir = Path(annotation_dir)
        self.sample_dir = self.annotation_dir / sample
        self.out_dir = Path(output_dir) / sample / self.run
        self.groups_path = Path(groups_path) if groups_path else self.sample_dir
        self.batch_key = batch_key or None
        self.counts_layer = counts_layer
        self.covariates = list(covariates or [])
        self.seeds = dict(seeds or {})
        self.safeguards = dict(safeguards or {})
        self.model_cfg = dict(model or {})
        self.embedding_cfg = dict(embedding or {})
        self.ontology_file = ontology_file
        self.ontology_extensions = list(ontology_extensions or ())
        self.allow_stale = allow_stale

        self.adata = None
        self.labels = None
        self.support = None
        self.document = None
        self.decisions = None
        self.record = {'sample': sample, 'run': self.run}

    # ------------------------------------------------------------------ paths

    def method_dir(self, method):
        """This run's directory under the annotation tree."""
        return self.sample_dir / method

    def scores_path(self, method):
        """Path to the per-cell geneset scores stage 1 wrote."""
        return self.method_dir(method) / '{}_geneset_scores.tsv'.format(self.sample)

    def annotations_path(self, method):
        """Path to the per-cell evidence TSV this run writes."""
        return self.method_dir(method) / '{}_significant_annotations.tsv'.format(self.sample)

    @property
    def label_key(self):
        """The `.obs` column scANVI trains on."""
        return 'seed_{}'.format(self.run)

    def _discover_methods(self):
        """The marker methods with a decision file for this sample.

        Returns:
            list[str]: Method names.
        """
        if self.methods:
            return self.methods
        return sorted(p.name for p in self.sample_dir.iterdir()
                      if p.is_dir() and self.scores_path(p.name).exists())

    # ---------------------------------------------------------------- prepare

    def prepare(self, adata=None):
        """Build the seed labels and check their provenance, without training.

        Returns:
            dict: A summary of what was seeded, for the dry-run report.
        """
        if self.method:
            self._prepare_single()
        else:
            self._prepare_combined()

        n_classes = self.labels[self.labels != UNASSIGNED].nunique()
        if n_classes < 2:
            # scANVI classifies; one class is not a classification problem, and
            # the failure downstream would be an opaque shape error inside the
            # model rather than this sentence.
            raise RefineError(
                '{}/{}: only {} seed class(es) survived. scANVI needs at least 2. Either '
                'the grouping names one cell type, or seeds.min_seed_cells removed the '
                'rest.'.format(self.sample, self.run, n_classes))

        self.record['seeds'] = summarise(self.labels).to_dict('records')
        self.record['n_seed_classes'] = int(n_classes)
        self.record['n_cells'] = int(len(self.labels))
        self.record['n_seed_cells'] = int((self.labels != UNASSIGNED).sum())
        self.record['seed_fraction'] = round(float((self.labels != UNASSIGNED).mean()), 4)

        if adata is not None:
            self.adata = adata
        else:
            logger.info('[REFINE] Loading %s', self.adata_path)
            self.adata = sc.read_h5ad(self.adata_path)
        self._attach_labels()
        return self.labels

    def _prepare_combined(self):
        """Seed from the reviewer's cross-method grouping.

        Returns:
            pandas.Series: Seed label per cell.
        """
        methods = self._discover_methods()
        if not methods:
            raise RefineError(
                'No stage 1 geneset scores under {}. Run annotation first.'.format(
                    self.sample_dir))

        scores_by_method, available = {}, {}
        for method in methods:
            path = self.scores_path(method)
            if not path.exists():
                raise RefineError('{} has no geneset scores at {}'.format(method, path))
            scores_by_method[method] = _read_scores(path)
            available[method] = list(scores_by_method[method].columns)
        logger.info('[REFINE] %s: %d method(s), %d geneset(s) total, %d cells',
                    self.sample, len(methods), sum(len(v) for v in available.values()),
                    len(next(iter(scores_by_method.values()))))

        # `available=` makes a stale grouping an error rather than a set of
        # silently unseeded cells: every geneset needs a row, and a row naming a
        # geneset that no longer exists means the file predates a stage 1 rerun.
        self.document = read_groups(self.groups_path, available=available, logger=logger)
        check_parents(self.document,
                      {m: self.annotations_path(m) for m in methods},
                      allow_stale=self.allow_stale)
        logger.info('[REFINE] %s: %r', self.sample, self.document)

        names = _name_map(sorted({g['po_id'] for g in self.document.groups.values()
                                  if g.get('po_id')}), self.ontology_file,
                          self.ontology_extensions)
        self.labels, self.support = build_group_seed_labels(
            scores_by_method, self.document,
            quantile=self.seeds.get('quantile', 0.9),
            margin=self.seeds.get('margin', 0.25),
            min_seed_cells=self.seeds.get('min_seed_cells', 10),
            name_map=names)

        self.record.update({
            'mode': 'combined',
            'methods': methods,
            'group_hash': self.document.hash,
            'group_source': self.document.source,
            'n_groups': len(self.document.groups),
            'n_named_groups': len(self.document.named),
            'n_unnamed_groups': len(self.document.unnamed),
            'n_ungrouped': len(self.document.ungrouped),
            'n_dropped': len(self.document.dropped),
            'support_histogram': {
                str(k): int(v) for k, v in
                self.support[self.labels != UNASSIGNED].value_counts().sort_index().items()},
        })

    def _prepare_single(self):
        """Seed from one method's own per-geneset decision file.

        For checking what one method would have said alone, not for production.

        Returns:
            pandas.Series: Seed label per cell.
        """
        method = self.method
        path = self.scores_path(method)
        if not path.exists():
            raise RefineError('stage 1 output missing for {}/{}: {}'.format(
                self.sample, method, path))
        scores = _read_scores(path)
        logger.info('[REFINE] %s/%s (single-method): %d cells x %d geneset(s)',
                    self.sample, method, len(scores), scores.shape[1])

        self.decisions = read_decisions(self.method_dir(method),
                                        genesets=list(scores.columns),
                                        method=method, logger=logger)
        if self.annotations_path(method).exists():
            check_parent(self.decisions, self.annotations_path(method),
                         allow_stale=self.allow_stale)
        logger.info('[REFINE] %s/%s: %r', self.sample, method, self.decisions)

        names = _name_map(sorted(set(self.decisions.labelled.values())), self.ontology_file,
                          self.ontology_extensions)
        self.labels = build_seed_labels(
            scores, self.decisions,
            quantile=self.seeds.get('quantile', 0.9),
            margin=self.seeds.get('margin', 0.25),
            min_seed_cells=self.seeds.get('min_seed_cells', 10),
            name_map=names)
        # One method cannot corroborate itself; support is 1 wherever seeded.
        self.support = (self.labels != UNASSIGNED).astype(int)
        self.record.update({
            'mode': 'single-method', 'methods': [method],
            'decision_hash': self.decisions.hash,
            'decision_source': self.decisions.source,
            'unnamed_genesets': sorted(self.decisions.unnamed()),
            'dropped_genesets': sorted(self.decisions.dropped),
        })

    @property
    def gate_hash(self):
        """Hash of the inputs this run is gated on, for staleness checks."""
        return self.record.get('group_hash') or self.record.get('decision_hash')

    def _attach_labels(self):
        """Write the seed columns onto `.obs` and report their support.

        Args:
            seeds (pandas.Series): Seed label per cell.
        """
        missing = self.adata.obs_names.difference(self.labels.index)
        if len(missing):
            raise RefineError(
                '{}/{}: {} of {} cells in the AnnData have no geneset score. The scores '
                'were computed on a different object than the one being refined.'.format(
                    self.sample, self.run, len(missing), self.adata.n_obs))
        self.adata.obs[self.label_key] = (
            self.labels.reindex(self.adata.obs_names).fillna(UNASSIGNED).astype('category'))
        # How many methods independently seeded each cell. The reason for
        # combining at all: a 6/6 cell and a 1/6 cell both become seeds, and
        # only this column can tell them apart afterwards.
        self.adata.obs['seed_support_{}'.format(self.run)] = (
            self.support.reindex(self.adata.obs_names).fillna(0).astype(int).to_numpy())

    # -------------------------------------------------------------------- run

    def execute(self, registry=None):
        """Train scVI, then scANVI, then embed, then write everything.

        Returns:
            dict: What actually ran, as written to `{sample}_{run}_run.json`.
        """
        if self.labels is None:
            raise RefineError('execute() called before prepare()')
        self.out_dir.mkdir(parents=True, exist_ok=True)

        scvi_model = self._fit_scvi(registry)
        scanvi_model = self._fit_scanvi(scvi_model)
        evidence = self._predict(scanvi_model)
        self._embed()
        self._write(evidence)
        return self.record

    def _fit_scvi(self, registry=None):
        """Fit the unsupervised scVI model the scANVI model is built from.

        Returns:
            scvi.model.SCVI: The fitted model.
        """
        from integration.integrate import SingleCellIntegrate

        n_batches = (self.adata.obs[self.batch_key].nunique()
                     if self.batch_key and self.batch_key in self.adata.obs else 1)
        tune = bool(self.model_cfg.get('tune', False))
        metric = self.model_cfg.get('tune_metric', 'Total')
        if tune and n_batches < 2 and metric == 'Total':
            # scib's batch half needs >= 2 batches. Scoring 40% of the
            # objective as zero would make every trial look equally bad, which
            # is precisely the ambiguous failure the tuning diagnostics exist
            # to prevent -- so say so and change the objective instead.
            logger.warning(
                '[REFINE] %s has %d batch(es); falling back from "Total" to "Bio conservation". '
                'This run is NOT scored on batch correction.', self.sample, n_batches)
            metric = 'Bio conservation'
        self.record.update({'n_batches': int(n_batches), 'tune': tune,
                            'tune_metric': metric if tune else ''})

        train_args = {}
        if self.model_cfg.get('max_epochs'):
            train_args['max_epochs'] = int(self.model_cfg['max_epochs'])

        integrator = SingleCellIntegrate(
            adata=self.adata, target='{}_{}'.format(self.sample, self.run),
            do_tune=tune, do_hvg=False, output_dir=str(self.out_dir),
            # Continuous only, matching the integration stage's own convention:
            # every covariate configured here is a QC metric, and a categorical
            # one would need registering as such.
            covariates={'categorical': [], 'continuous': self.covariates},
            logger=logger, model_class='SCVI',
            n_trials=self.model_cfg.get('n_trials', 30),
            tune_metric=metric, model_args={'model_params': dict(
                self.model_cfg.get('model_params') or {})},
            train_args=train_args,
            registry=registry,
            # Refinement owns its own output tree; the integration registry is
            # keyed on integration targets and these are not those.
            persist=False, out_dir=self.out_dir / 'scvi')

        # labels_key is what makes the scib bio-conservation half meaningful
        # and what scANVI inherits below. See SingleCellIntegrate.integrate.
        integrator.integrate(batch_key=self.batch_key, layer=self.counts_layer,
                             labels_key=self.label_key)
        self.adata = integrator.adata
        self.record['scvi_params'] = integrator.resolved_model_params or {}

        model = integrator.model
        self.adata.obsm['X_scVI_{}'.format(self.run)] = model.get_latent_representation()
        model.save(str(self.out_dir / 'scvi'), overwrite=True)
        self._diagnostics('scVI', model, 'X_scVI_{}'.format(self.run))
        return model

    def _fit_scanvi(self, scvi_model):
        """Fit scANVI from the scVI model, using the seed labels.

        Args:
            scvi_model (scvi.model.SCVI): The source model.

        Returns:
            scvi.model.SCANVI: The fitted model.
        """
        from scvi.model import SCANVI

        params = dict(self.model_cfg.get('scanvi_params') or {})
        model = SCANVI.from_scvi_model(
            scvi_model, unlabeled_category=UNASSIGNED, labels_key=self.label_key, **params)

        train_args = {}
        if self.model_cfg.get('scanvi_max_epochs'):
            train_args['max_epochs'] = int(self.model_cfg['scanvi_max_epochs'])
        if self.model_cfg.get('n_samples_per_label'):
            # Seeds are a small minority of cells by design (~7.5% here), so
            # without oversampling the classifier head sees almost nothing but
            # unlabelled data and the labelled loss is swamped.
            train_args['n_samples_per_label'] = int(self.model_cfg['n_samples_per_label'])
        logger.info('[REFINE] Training scANVI (%s)', train_args or 'scvi defaults')
        model.train(**train_args)

        model.save(str(self.out_dir / 'scanvi'), overwrite=True)
        self.record['scanvi_train_args'] = train_args
        self._diagnostics('scANVI', model, 'X_scANVI_{}'.format(self.run))
        return model

    def _diagnostics(self, method, model, latent_key):
        """Compute the evidence columns the safeguards and the report read.

        Kept whole rather than reduced to a verdict, so thresholds can move without
        retraining.
        """
        from integration.diagnostics import run_diagnostics
        try:
            run_diagnostics(
                self.adata, method, self.out_dir,
                '{}_{}_{}'.format(self.sample, self.run, method.lower()),
                model=model, batch_key=self.batch_key, latent_keys=[latent_key],
                logger=logger, save_plot=True)
        except Exception as exc:                          # noqa: BLE001
            logger.warning('[REFINE] %s/%s: %s diagnostics failed: %s',
                           self.sample, self.run, method, exc)

    def _predict(self, scanvi_model):
        """Attach scANVI's calls and their confidence to `.obs`.

        Args:
            model (scvi.model.SCANVI): The fitted model.
        """
        latent = scanvi_model.get_latent_representation()
        self.adata.obsm['X_scANVI_{}'.format(self.run)] = latent

        proba = scanvi_model.predict(soft=True)
        # `predict` indexes by obs_names, but only when it is given an AnnData
        # rather than a dataloader. Re-index defensively -- a positional
        # mismatch here would be invisible in every downstream artifact.
        proba.index = self.adata.obs_names
        predictions = proba.columns.to_numpy()[proba.to_numpy().argmax(axis=1)]

        evidence = apply_safeguards(
            predictions=predictions, proba=proba, latent=latent,
            seed_labels=self.adata.obs[self.label_key],
            min_confidence=self.safeguards.get('min_confidence', 0.8),
            max_novelty_ratio=self.safeguards.get('max_novelty_ratio', 1.25),
            unassigned=UNASSIGNED,
            reference_quantile=self.safeguards.get('reference_quantile', 0.95))

        m = self.run
        self.adata.obs['prediction_{}'.format(m)] = pd.Categorical(evidence['prediction'])
        self.adata.obs['annotation_{}'.format(m)] = pd.Categorical(evidence['annotation'])
        self.adata.obs['confidence_{}'.format(m)] = evidence['confidence'].to_numpy()
        self.adata.obs['margin_{}'.format(m)] = evidence['margin'].to_numpy()
        self.adata.obs['novelty_ratio_{}'.format(m)] = evidence['novelty_ratio'].to_numpy()
        self.adata.obs['flag_{}'.format(m)] = pd.Categorical(evidence['flag'])

        counts = evidence['annotation'].value_counts()
        self.record['annotation_counts'] = {str(k): int(v) for k, v in counts.items()}
        self.record['n_demoted'] = int((evidence['flag'] != '').sum())
        # Confidence percentiles, because min_confidence is the one placeholder
        # that can only be calibrated once a real posterior exists. Recorded
        # every run so the number can be set from data rather than from the
        # nearest-centroid stand-in it was guessed against.
        self.record['confidence_percentiles'] = {
            str(q): round(float(np.percentile(evidence['confidence'], q)), 4)
            for q in (1, 5, 10, 25, 50, 75, 90, 99)}
        self.record['n_unknown_classes'] = int(sum(
            str(k).startswith(UNKNOWN_PREFIX) for k in counts.index))
        return evidence

    def _embed(self):
        """UMAP and Leiden on the scANVI latent, plus the figures."""
        cfg = self.embedding_cfg
        reducer = DimReducer(self.adata, logger=logger)
        result = reducer.compute(
            use_rep='X_scANVI_{}'.format(self.run),
            key=self.run,
            n_neighbors=cfg.get('n_neighbors'),
            metric=cfg.get('metric', 'correlation'),
            min_dist=cfg.get('min_dist', 0.0),
            leiden_resolution=cfg.get('leiden_resolution', 0.5),
            random_state=cfg.get('random_state', 42),
            do_leiden=True)
        self.adata = result['adata']
        self.record['embedding'] = result['params']

        # `_predict` (which ran before this) already wrote annotation_/
        # confidence_/flag_/seed_support_{run}, so they're available to colour
        # by here. UMAPPlotter is the same class mapping QC and integration
        # already use -- plotting only, no computation of its own -- so this
        # is the one new figure this stage adds beyond what run_diagnostics
        # (training curves, batch mixing) already covers.
        m = self.run
        extra = ['seed_{}'.format(m), 'seed_support_{}'.format(m),
                 'annotation_{}'.format(m), 'confidence_{}'.format(m),
                 'flag_{}'.format(m)]
        if self.batch_key and self.batch_key in self.adata.obs:
            extra.append(self.batch_key)
        try:
            UMAPPlotter(self.adata, self.out_dir, logger=logger).plot(
                umap_key=result['umap_key'], leiden_key=result['leiden_key'],
                extra_color_by=extra, prefix=self.sample)
        except Exception as exc:                          # noqa: BLE001
            logger.warning('[REFINE] %s/%s: embedding plots failed: %s',
                           self.sample, self.run, exc)
        return result

    def _write(self, evidence):
        """Write the h5ad, the per-cell evidence TSV and the run record."""
        h5ad_path = self.out_dir / '{}_{}.h5ad'.format(self.sample, self.run)
        self.adata.write_h5ad(h5ad_path)

        table = evidence.copy()
        table.insert(0, 'seed_support',
                     self.adata.obs['seed_support_{}'.format(self.run)].to_numpy())
        table.insert(0, 'seed', self.adata.obs[self.label_key].astype(str).to_numpy())
        table.insert(0, 'leiden', self.adata.obs['leiden_{}'.format(self.run)].astype(str).to_numpy())
        tsv_path = self.out_dir / '{}_{}_labels.tsv'.format(self.sample, self.run)
        table.to_csv(tsv_path, sep='\t')

        self.record['artifacts'] = {
            'h5ad': str(h5ad_path), 'labels_tsv': str(tsv_path),
            'scvi_model': str(self.out_dir / 'scvi'), 'scanvi_model': str(self.out_dir / 'scanvi')}
        run_path = self.out_dir / '{}_{}_run.json'.format(self.sample, self.run)
        run_path.write_text(json.dumps(self.record, indent=2, default=str))
        logger.info('[REFINE] %s/%s written -> %s', self.sample, self.run, self.out_dir)
        return h5ad_path
