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
        {sample}_{run}_labels.tsv  per-cell evidence, joinable on barcode
        {sample}_{run}_run.json    what actually ran, for the report

        SCVI/                      the base fit that seeds scANVI, untuned
        SCANVI/                    the deliverable -- model, diagnostics, AND
                                   the UMAP figures, which are built on its
                                   latent and used to sit at the run root under
                                   a name that identified no model at all
        SCVI_scibtuned/            optional comparison arm, scib-tuned
        JointEmbeddingSCVI/        optional comparison arm, scib-tuned

    One directory per trained model. Run-spanning artifacts stay at the root,
    because they are not any one model's. The tuned scVI arm cannot also be
    called SCVI -- the base fit owns that name.

and in the h5ad, keyed by `{run}` (`combined` for the normal route):

    .obsm['X_scVI_{run}']      the intermediate latent, kept for comparison
    .obsm['X_scANVI_{run}']    the deliverable -- label-aware, which is the
                               whole reason to run scANVI at all
    .obsm['X_umap_{run}']      2-D, from the scANVI latent
    .obs['leiden_{run}']       clusters on the same graph the UMAP came from
    .obsm['X_umap_scVI_{run}'], ['X_umap_scVItuned_{run}'],
    .obsm['X_umap_jointSCVI_{run}'] and their `leiden_` counterparts -- the
                               same embedding on each OTHER latent this run
                               fitted, one per model directory, so the arms can
                               be compared to the deliverable by eye rather
                               than only through a metrics table. Keyed by the
                               LATENT, never by the run alone. See
                               refine/embed_models.py.
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
from shared.groups import check_parents, read as read_groups

from .embed_models import COMPARISON_MODELS, embed_models, embed_one, latent_specs

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
                 groups_path=None, method=None, reuse_models=True):
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
        self.reuse_models = reuse_models

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

    def model_dir(self, name):
        """This run's directory for one model's artifacts.

            results/refine/{sample}/{run}/{MODEL}/

        One directory per trained model rather than a flat run directory,
        because the run produces up to four of them and the flat layout gave no
        way to tell whose a figure was. The UMAP is the case that actually
        misled: it is built on the scANVI latent, but sat at the run root under
        a name (`..._X_umap_{run}`) that named the RUN twice and no model at
        all.

        Names are the model class as scvi spells it, except the tuned scVI
        comparison arm, which cannot also be called `SCVI` -- the base fit owns
        that. Run-spanning artifacts (the h5ad, the labels TSV, the run record)
        stay at the run root, since they are not any one model's.

        Args:
            name (str): Directory name, e.g. 'SCANVI'.

        Returns:
            Path: The directory. Not created here; each writer mkdirs its own.
        """
        return self.out_dir / name

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
            name_map=names,
            quantile_scope=self.seeds.get('quantile_scope', 'group'))

        self.record.update({
            'mode': 'combined',
            'seed_quantile_scope': self.seeds.get('quantile_scope', 'group'),
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

        # THE DELIVERABLE CHAIN. scVI seeds scANVI, scANVI produces the labels,
        # and the UMAP is built on the scANVI latent. Untuned on purpose: this
        # scVI exists to seed scANVI, and comparing refine runs to each other
        # only works if the base model is held fixed.
        scvi_model = self._fit_scvi(registry)
        scanvi_model = self._fit_scanvi(scvi_model)
        evidence = self._predict(scanvi_model)
        self._embed()

        # WRITE THE DELIVERABLE NOW, BEFORE THE OPTIONAL ARMS RUN.
        #
        # This ordering was wrong until 2026-08-18 and cost a real run. The arms
        # sat between _embed and _write with a comment claiming they were
        # "ordered last so a failure cannot cost the labels" -- which they were
        # not, because _write came after them. Job 42068733 trained scVI and
        # scANVI over 4 h, produced the UMAP, then died in the scVI arm on a
        # library incompatibility, and because _write had never run the h5ad,
        # the labels TSV and the run record were all lost. 18 h for two model
        # checkpoints and some figures.
        #
        # A try/except around the arms is not sufficient on its own: it catches
        # an exception, not a SIGKILL, an OOM or the job hitting its wall clock,
        # and the arms are the part most likely to hit all three. So the
        # deliverable is written first and unconditionally.
        self._write(evidence)

        # THE COMPARISON ARMS. Scored on scib, which is only possible here:
        # scib's bio-conservation half needs labels, and until _predict has run
        # the only labels in the object are the cells carrying a seed. After it
        # every cell has a propagated call, so the metric is computed over a
        # full partition rather than a sparse one.
        #
        # Non-fatal by construction. They are a comparison, and no failure in
        # them should turn a completed refinement into a failed job.
        arms = {}
        try:
            arms = self._fit_comparison_arms(registry)
        except Exception as exc:                          # noqa: BLE001
            logger.error('[REFINE] %s/%s: comparison arms failed (%s: %s). The '
                         'deliverable above is unaffected and already written.',
                         self.sample, self.run, type(exc).__name__, exc)
            self.record['comparison'] = {'enabled': True, 'failed': True,
                                         'error': '{}: {}'.format(type(exc).__name__, exc)}

        # EMBED THE ARMS, which _embed above could not: it runs before they are
        # fitted, so at that point `X_scVItuned_{run}` and `X_jointSCVI_{run}`
        # are simply not in .obsm and `embed_models` finds only the base scVI.
        # That ordering is deliberate -- the deliverable is written to disk
        # before anything optional is attempted -- so the fix is a second
        # embedding pass here rather than moving _embed after the arms.
        #
        # Same `embed_models` the earlier pass uses, so the arms get the same
        # neighbour graph, metric, min_dist, resolution and plotnine figures as
        # the deliverable, and only the LATENT differs between them. Comparing
        # arms to the deliverable is the entire point of fitting them, and a
        # comparison across two different embedding recipes is not one.
        #
        # Non-fatal for the same reason the arms themselves are: the deliverable
        # is already on disk and a missing comparison figure must not fail a
        # completed refinement.
        if arms:
            try:
                record = dict(self.record.get('embedding_models') or {})
                embed_models(self.adata, sample=self.sample, run=self.run,
                             out_dir=self.out_dir, models=sorted(arms), cfg=self.embedding_cfg,
                             batch_key=self.batch_key, do_leiden=True, log=logger,
                             record=record)
                self.record['embedding_models'] = record
            except Exception as exc:                      # noqa: BLE001
                logger.error('[REFINE] %s/%s: embedding the comparison arms '
                             'failed (%s: %s). Their latents are in the h5ad; '
                             'run scripts/refine_embed_arms.py to backfill.',
                             self.sample, self.run, type(exc).__name__, exc)

        # Only if an arm actually produced a latent: fold it into the h5ad and
        # refresh the run record. Skipped entirely when both are off or both
        # failed, so the common path pays for one h5ad write, not two.
        if arms:
            self._write(evidence)
        return self.record

    def _reuse_model(self, model_class, out_dir, label):
        """A previously trained model from this run's own tree, or None.

        WHY THIS EXISTS HERE AND NOT IN SingleCellIntegrate. The integrator does
        have a skip -- `_integrate_scvi` returns a registry-cached model when one
        exists -- but it is gated on `persist`, and refine passes
        `persist=False` on purpose: the integration registry is keyed on
        integration targets and these are not those, so a disposable run must not
        resolve to some previously-registered model that happens to share a name.
        That guard is right, and it makes the registry skip unreachable from
        here. The integrator also never writes these models (`save=self.persist`
        is False); refine saves them itself.

        So the reuse is keyed on the FILESYSTEM instead, which is safe precisely
        where the registry was not: `results/refine/{sample}/{run}/{MODEL}/` is
        specific to one sample, one run and one model, so there is no name to
        collide over.

        Correctness rests on scvi's own check, not on ours: `cls.load(path,
        adata=...)` validates the saved setup registry against the AnnData it is
        handed and raises when they disagree, so a model trained on a different
        cohort, layer or batch key cannot be silently adopted. A load that fails
        for any reason falls through to training rather than killing the run.

        `--force` retrains everything.

        Args:
            model_class (str): A key of MODEL_CLASSES.
            out_dir (Path): Directory holding `model.pt`.
            label (str): Human-readable stage name for the log.

        Returns:
            The loaded model, or None to train.
        """
        if not self.reuse_models:
            return None
        path = Path(out_dir)
        if not (path / 'model.pt').exists():
            return None
        from integration.integrationutils import resolve_model_class
        try:
            model = resolve_model_class(model_class).load(str(path), adata=self.adata)
        except Exception as exc:                          # noqa: BLE001
            logger.warning('[REFINE] %s/%s: found a %s model at %s but could not '
                           'reuse it (%s: %s); retraining.', self.sample, self.run,
                           label, path, type(exc).__name__, exc)
            return None
        logger.info('[REFINE] %s/%s: reusing the existing %s model at %s -- not '
                    'retraining. Pass --force to override.',
                    self.sample, self.run, label, path)
        return model

    @staticmethod
    def _train_args(cfg):
        """`model.train()` kwargs from a config block: epochs and early stopping.

        EARLY STOPPING DOES NOT BITE ON ITS OWN, and the reason is worth stating
        where it will be read. `max_epochs` sets KL warmup too --
        normalize_kl_warmup derives it as 50% of the effective max_epochs when
        unset -- and the objective is a moving target while the KL term is still
        being annealed. scvi's patience is 45 epochs on `elbo_validation`, so the
        earliest a stop can fire is warmup + patience:

            max_epochs 400  -> warmup 200, earliest stop ~245 epochs
            max_epochs 120  -> warmup  60, earliest stop ~105
            max_epochs  60  -> warmup  30, earliest stop ~ 60 (i.e. never)

        At 400 that is 80 of the 131 minutes this fit costs on a 147k-cell
        object -- early stopping turns `max_epochs` into a ceiling rather than a
        target, but it cannot rescue a ceiling set far above what the data needs.
        scvi's own heuristic for 147,697 cells is 54 epochs. Lowering
        `max_epochs` is what captures the saving; this is the safety net.

        Args:
            cfg (dict): A config block carrying `max_epochs` and optionally
                `early_stopping`, `early_stopping_patience`,
                `check_val_every_n_epoch`.

        Returns:
            dict: kwargs for `model.train()`.
        """
        args = {}
        if cfg.get('max_epochs'):
            args['max_epochs'] = int(cfg['max_epochs'])
        # DATALOADER WORKERS. scvi's AnnDataLoader takes an explicit num_workers
        # if given and otherwise falls back to scvi.settings.dl_num_workers,
        # which defaults to 0 -- i.e. every batch is collated in the training
        # process, between GPU steps. Measured cost on job 42068733: four
        # concurrent tuning trials on 147k cells took ~2 h 20 m each against ~20
        # min for the same fit run solo, because four trials sharing 32 CPUs are
        # CPU-starved on collation rather than GPU-bound. lightning warns about
        # it in that run's log.
        #
        # Set per block rather than through scvi.settings, which is process
        # global: the base fit has the whole node and the tuning trials have a
        # quarter of it each, so they want different numbers, and an explicit
        # value lands in the run record where a global would not.
        workers = cfg.get('dataloader_workers')
        if workers:
            args['datasplitter_kwargs'] = {
                'num_workers': int(workers),
                # Rebuilding worker processes every epoch is most of what
                # workers cost on short epochs; keeping them alive is the point.
                'persistent_workers': True,
            }
        if cfg.get('early_stopping'):
            args['early_stopping'] = True
            # Validation must actually be computed for the monitor to see it.
            args['check_val_every_n_epoch'] = int(cfg.get('check_val_every_n_epoch', 1))
            if cfg.get('early_stopping_patience') is not None:
                args['early_stopping_patience'] = int(cfg['early_stopping_patience'])
        return args

    def _fit_comparison_arms(self, registry=None):
        """Fit the scib-tuned comparison latents beside the deliverable.

        Two optional arms, both off by default, both tuned on scib rather than
        on ELBO, and both scored against scANVI's PROPAGATED annotation:

            scvi                .obsm['X_scVItuned_{run}']
            joint_embedding     .obsm['X_jointSCVI_{run}']

        Neither replaces anything. `X_scANVI_{run}` stays the deliverable, the
        labels stay scANVI's, and the UMAP stays on the scANVI latent -- so
        disabling both returns the stage to exactly what it did before.

        WHY THE ARMS RUN HERE AND NOT BESIDE THE BASE FIT. scib's objective is
        half batch correction and half bio conservation, and the bio half is
        scored against labels. Before `_predict` the only labels in the object
        are the seeds, which are ~7.5% of cells; scoring a latent on those
        measures how well it organises a small, non-random subset. After
        `_predict` every cell carries a propagated call, so the same metric is
        computed over a complete partition. That is the whole reason this is a
        separate stage rather than a flag on `_fit_scvi`.

        WHY scib AND NOT elbo_validation. The base scVI is fitted at scvi's
        constructor defaults deliberately. These arms exist to be compared, and
        the parameters that distinguish them -- JointEmbeddingSCVI's four loss
        weights especially -- trade reconstruction against embedding structure,
        which is exactly what a likelihood cannot rank and what a composite
        metric can.

        Args:
            registry (CacheUtils, optional): Passed through; unused at
                `persist=False`.

        Returns:
            dict: {arm name: fitted model}, empty when both are disabled.
        """
        cfg = dict(self.model_cfg.get('comparison') or {})
        arms = (
            ('scvi', 'SCVI', 'SCVI_scibtuned', 'X_scVItuned_{}'.format(self.run)),
            ('joint_embedding', 'JointEmbeddingSCVI', 'JointEmbeddingSCVI',
             'X_jointSCVI_{}'.format(self.run)),
        )
        enabled = [a for a in arms if (cfg.get(a[0]) or {}).get('enabled', False)]
        if not enabled:
            self.record['comparison'] = {'enabled': False}
            return {}

        labels_key = self._comparison_labels_key(cfg)
        fitted, record = {}, {'enabled': True, 'labels_key': labels_key, 'arms': {}}
        for name, model_class, subdir, latent_key in enabled:
            # PER ARM, so one arm's failure does not silently cost the other.
            # On job 42083220 the scVI arm died on an ARPACK failure inside
            # scib's PCA; with a single try/except one level up, the
            # JointEmbeddingSCVI arm -- the one actually worth running -- was
            # never attempted. These are independent comparisons and a library
            # fault in one says nothing about the other.
            try:
                model, arm_record = self._fit_one_comparison_arm(
                    name=name, model_class=model_class, subdir=subdir,
                    latent_key=latent_key, cfg=cfg, labels_key=labels_key,
                    registry=registry)
            except Exception as exc:                      # noqa: BLE001
                logger.error('[REFINE] %s/%s: comparison arm %r failed (%s: %s); '
                             'continuing with the remaining arm(s).',
                             self.sample, self.run, name, type(exc).__name__, exc)
                record['arms'][name] = {'model_class': model_class, 'failed': True,
                                        'error': '{}: {}'.format(type(exc).__name__, exc)}
                continue
            fitted[name] = model
            record['arms'][name] = arm_record
        self.record['comparison'] = record
        return fitted

    def _comparison_labels_key(self, cfg):
        """Which `.obs` column the scib bio-conservation half is scored against.

        Defaults to `prediction_{run}` rather than `annotation_{run}`, and the
        difference matters. `annotation_` is the deliverable and carries
        UNASSIGNED for every cell the safeguards demoted -- a REJECT BIN, not a
        cell type, holding whatever was low-confidence or high-novelty. Handing
        it to scib as a label would reward a latent for making that bin compact,
        which is backwards: it is heterogeneous by construction.
        `prediction_` is the same calls before demotion, so every cell carries
        its best class and the partition is complete.

        Args:
            cfg (dict): The `comparison:` config block.

        Returns:
            str: An existing `.obs` column name.

        Raises:
            RefineError: If the configured column is not in `.obs`, which means
                the arms were asked to run before `_predict`.
        """
        stem = cfg.get('labels_key', 'prediction')
        key = '{}_{}'.format(stem, self.run)
        if key not in self.adata.obs:
            raise RefineError(
                "comparison arms need '{}' in .obs and it is absent. They run "
                'after _predict for exactly this reason -- scib cannot score bio '
                'conservation before the labels exist.'.format(key))
        return key

    def _fit_one_comparison_arm(self, name, model_class, subdir, latent_key,
                                cfg, labels_key, registry=None):
        """One comparison arm. See `_fit_comparison_arms` for why these exist.

        Args:
            name (str): Config key under `comparison:`.
            model_class (str): A key of `MODEL_CLASSES`.
            subdir (str): Output directory under the run's own tree.
            latent_key (str): `.obsm` key for the fitted latent.
            cfg (dict): The `comparison:` block.
            labels_key (str): `.obs` column for scib's bio half.
            registry (CacheUtils, optional): Passed through.

        Returns:
            tuple: (fitted model, record dict).
        """
        from integration.integrate import SingleCellIntegrate

        out = self.model_dir(subdir)
        reused = self._reuse_model(model_class, out, 'comparison arm {!r}'.format(name))
        if reused is not None:
            self.adata.obsm[latent_key] = reused.get_latent_representation()
            return reused, {'model_class': model_class, 'reused_from': str(out),
                            'latent_key': latent_key}

        arm = dict(cfg.get(name) or {})
        tune = bool(arm.get('tune', cfg.get('tune', True)))
        metric = arm.get('tune_metric', cfg.get('tune_metric', 'Total'))
        n_batches = (self.adata.obs[self.batch_key].nunique()
                     if self.batch_key and self.batch_key in self.adata.obs else 1)
        if tune and n_batches < 2 and metric == 'Total':
            # Same guard as the base fit: scib's batch half needs >= 2 batches,
            # and scoring 40% of the objective as zero makes every trial look
            # equally bad instead of failing loudly.
            logger.warning(
                '[REFINE] %s has %d batch(es); comparison arm %r falls back from '
                '"Total" to "Bio conservation" and is NOT scored on batch '
                'correction.', self.sample, n_batches, name)
            metric = 'Bio conservation'

        merged = {**cfg, **arm}
        train_args = self._train_args(merged)

        integrator = SingleCellIntegrate(
            adata=self.adata, target='{}_{}_{}'.format(self.sample, self.run, name),
            do_tune=tune, do_hvg=False, output_dir=str(self.out_dir),
            covariates={'categorical': [], 'continuous': self.covariates},
            logger=logger, model_class=model_class,
            n_trials=arm.get('n_trials', cfg.get('n_trials', 8)),
            tune_metric=metric,
            model_args={'model_params': dict(arm.get('model_params') or {})},
            train_args=train_args, registry=registry,
            # Per-arm search space, merged over the class default by
            # resolve_search_space. An arm needs this because the class default
            # is sized for a from-scratch integration target, while an arm here
            # has a fixed trial budget and a narrower question.
            search_space=(arm.get('search_space') or cfg.get('search_space')),
            persist=False, out_dir=out)

        logger.info('[REFINE] comparison arm %r: %s, tune=%s, metric=%s, labels=%s',
                    name, model_class, tune, metric if tune else '-', labels_key)
        integrator.integrate(batch_key=self.batch_key, layer=self.counts_layer,
                             labels_key=labels_key)
        self.adata = integrator.adata
        model = integrator.model
        self.adata.obsm[latent_key] = model.get_latent_representation()
        model.save(str(out), overwrite=True)
        self._diagnostics(model_class, model, latent_key, out_dir=out)
        return model, {'model_class': model_class, 'tune': tune,
                       'tune_metric': metric if tune else '',
                       'n_batches': int(n_batches), 'latent_key': latent_key,
                       'params': integrator.resolved_model_params or {}}

    def _fit_scvi(self, registry=None):
        """Fit the unsupervised scVI model the scANVI model is built from.

        Returns:
            scvi.model.SCVI: The fitted model.
        """
        from integration.integrate import SingleCellIntegrate

        reused = self._reuse_model('SCVI', self.model_dir('SCVI'), 'base scVI')
        if reused is not None:
            self.adata.obsm['X_scVI_{}'.format(self.run)] = reused.get_latent_representation()
            self.record['scvi_params'] = {'reused_from': str(self.model_dir('SCVI'))}
            return reused

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

        train_args = self._train_args(self.model_cfg)

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
            persist=False, out_dir=self.model_dir('SCVI'))

        # labels_key is what makes the scib bio-conservation half meaningful
        # and what scANVI inherits below. See SingleCellIntegrate.integrate.
        integrator.integrate(batch_key=self.batch_key, layer=self.counts_layer,
                             labels_key=self.label_key)
        self.adata = integrator.adata
        self.record['scvi_params'] = integrator.resolved_model_params or {}

        model = integrator.model
        self.adata.obsm['X_scVI_{}'.format(self.run)] = model.get_latent_representation()
        model.save(str(self.model_dir('SCVI')), overwrite=True)
        self._diagnostics('scVI', model, 'X_scVI_{}'.format(self.run),
                          out_dir=self.model_dir('SCVI'))
        return model

    def _fit_scanvi(self, scvi_model):
        """Fit scANVI from the scVI model, using the seed labels.

        Args:
            scvi_model (scvi.model.SCVI): The source model.

        Returns:
            scvi.model.SCANVI: The fitted model.
        """
        from scvi.model import SCANVI

        reused = self._reuse_model('SCANVI', self.model_dir('SCANVI'), 'scANVI')
        if reused is not None:
            return reused

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

        model.save(str(self.model_dir('SCANVI')), overwrite=True)
        self.record['scanvi_train_args'] = train_args
        self._diagnostics('scANVI', model, 'X_scANVI_{}'.format(self.run),
                          out_dir=self.model_dir('SCANVI'))
        return model

    def _diagnostics(self, method, model, latent_key, out_dir=None):
        """Compute the evidence columns the safeguards and the report read.

        Kept whole rather than reduced to a verdict, so thresholds can move without
        retraining.

        Args:
            method (str): Model name, used in the artifact names.
            model: The fitted model.
            latent_key (str): `.obsm` key to run the diagnostics on.
            out_dir (Path, optional): Where to write. Defaults to the run root
                for backward compatibility; every caller passes that model's own
                directory so a training curve sits beside the model it describes.
        """
        from integration.diagnostics import run_diagnostics
        target = Path(out_dir) if out_dir is not None else self.out_dir
        target.mkdir(parents=True, exist_ok=True)
        try:
            run_diagnostics(
                self.adata, method, target,
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
        """UMAP and Leiden on the scANVI latent, plus the figures.

        Then, unless `embedding.compare_models` is off, the same computation on
        every OTHER latent this run produced -- the base scVI fit and whichever
        comparison arms were enabled -- each into its own model directory. The
        arms exist to be compared against the deliverable and the comparison a
        person actually makes is visual; a training curve and a batch-mixing
        table answer neither "does it separate the same populations" nor "does
        it still mix the batches". See refine/embed_models.py for the key
        naming, and note that the deliverable's own keys are unchanged.

        Returns:
            dict: `DimReducer.compute`'s result for the scANVI latent, which is
                what the rest of the stage means by "the embedding".
        """
        cfg = self.embedding_cfg
        specs = {s.subdir: s for s in latent_specs(self.run)}

        # scANVI first and OUTSIDE the collect-and-continue below: it is the
        # deliverable, `.obsm['X_umap_{run}']` and `.obs['leiden_{run}']` are
        # what `_write` and the report read, and a failure here is a failed run
        # rather than a missing comparison.
        result = embed_one(self.adata, specs['SCANVI'], sample=self.sample, run=self.run,
                           out_dir=self.out_dir, cfg=cfg, batch_key=self.batch_key,
                           do_leiden=True, log=logger)
        self.adata = result['adata']
        self.record['embedding'] = result['params']

        if cfg.get('compare_models', True):
            record = {}
            embed_models(self.adata, sample=self.sample, run=self.run,
                         out_dir=self.out_dir, models=list(COMPARISON_MODELS), cfg=cfg,
                         batch_key=self.batch_key, do_leiden=True, log=logger,
                         record=record)
            self.record['embedding_models'] = record
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
            'scvi_model': str(self.model_dir('SCVI')),
            'scanvi_model': str(self.model_dir('SCANVI'))}
        run_path = self.out_dir / '{}_{}_run.json'.format(self.sample, self.run)
        run_path.write_text(json.dumps(self.record, indent=2, default=str))
        logger.info('[REFINE] %s/%s written -> %s', self.sample, self.run, self.out_dir)
        return h5ad_path
