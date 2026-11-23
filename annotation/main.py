import argparse
import logging

from pathlib import Path

import pandas as pd

from shared.species_profiles import resolve_species
from shared.utils import load_config, setup_logger, find_root_dir, pipeline_log_path
from shared.decisions import (DecisionError, build_consensus, build_template,
                              write_template)
from shared.geneset_scores import (associate_with_clusters, ensure_score_layer,
                                   read_genesets, score_genesets)
from .ortho import OrthoManager
from .omg import OMGBuilder
from .annotate import scAnnotator
from .markers_hook import ensure_marker_files
from .phylo import build as phylo_build
from .review import load_report_gene_groups, write_review
from .standardize import CellTypeStandardizer

def main(annotation_config_path: str, species: str=None, force:bool=False, validate_only:bool=False,
         orthofinder_only:bool=False, adata:str=None, sample_name:str=None,
         latent_key:str=None, cluster_key:str=None, methods:list=None,
         tissue:list=None, force_decisions:bool=False):
    """Run the annotation stage: proteomes, OrthoFinder, enrichment, review.

    Arguments mirror the CLI flags below; `tissue` is a list. Returns None, or
    the OrthoFinder results directory with `orthofinder_only`. Raises
    RuntimeError after the loop if any (sample, method) run failed.
    """

    # One config, several target species. resolve_species folds the selected
    # profile in and strips the target from `reference_species` -- see
    # shared/species_profiles.py. A single-species config is returned unchanged.
    config = resolve_species(load_config(Path(annotation_config_path)),
                             species)['AnnotateConfigs']
    root_dir = find_root_dir()
    # Stable, not date-stamped: OrthoFinder runs are content-addressed by the
    # proteome set (see OrthoManager), so a rerun with unchanged proteomes must
    # be able to find the previous run rather than start a new dated tree.
    # `output_dir:` decouples results from where the code is checked out;
    # unset, it is `<repo root>/results/annotation`.
    out_dir = Path(config.get('output_dir') or (root_dir / 'results/annotation'))

    # Regenerable caches (orthofinder/, primary_transcripts/, genemaps/,
    # validation/) live outside the deliverables tree, so it is clear what is
    # safe to delete. `cache_dir:` overrides; the default is a
    # `cache/annotation/` sibling of `output_dir`. Trees written before
    # 2026-08-06 hold these caches beside the per-dataset output.
    cache_dir = Path(config.get('cache_dir') or (out_dir.parent / 'cache' / 'annotation'))
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Logging is configured once, here: one file per run under
    # logs/annotation/ (see shared.utils.pipeline_log_path).
    log_dir = config.get('log_dir') or (root_dir / 'logs')
    log_file = pipeline_log_path('annotation', log_dir=log_dir)
    logger = setup_logger(name = 'AnnotationLogging', log_file=log_file, level=logging.INFO)
    logger.info('[MAIN] Loaded configs, starting annotation (log: %s)', log_file)

    target_species = config.get("target_species")
    target_species_files = config.get("target_species_files", {})
    reference_species = config.get("reference_species")
    of_cfg = config.get("orthofinder", {}) or {}
    marker_cfg = config.get("markers", {}) or {}
    # Reference tables and this pipeline's own per-sample marker tables have
    # different column names; both are declared in the config.
    reference_columns = marker_cfg.get("reference") or {"cluster": "Celltypes", "gene": "Genes"}
    target_columns = marker_cfg.get("target") or {"cluster": "clusterName", "gene": "gene"}
    q_threshold = marker_cfg.get("q_threshold", 0.05)
    # Tissue(s) of interest. Applied to the resulting *candidates*, not to the
    # reference markers -- filtering markers shrinks the orthogroup universe and
    # weakens every Fisher test (see OMGBuilder._record_tissues).
    tissue = tissue or marker_cfg.get("tissue")
    tissue_column = (marker_cfg.get("reference") or {}).get("tissue", "Tissues")
    # A candidate subsuming more than this many of the cell types we have
    # evidence for is too general to be a useful call.
    max_dataset_children = marker_cfg.get("max_dataset_children", 15)
    ontology_file = config.get("ontology_file") or str(
        Path(__file__).with_name("po_final.obo"))
    # Extra OBO files grafted into the ontology at load, for cell types PO does
    # not represent -- the nodule vocabulary. Relative paths resolve next to the
    # ontology file. Merged as text so their is_a/part_of resolve against PO;
    # see shared.utils.merged_obo_bytes and annotation/nodule.obo.
    ontology_extensions = [
        e if Path(e).is_absolute() else str(Path(ontology_file).with_name(e))
        for e in (config.get("ontology_extensions") or [])]

    proteomes = {}
    markers = {}
    gene_id_rules = {}
    # A species' GTF/GFF3, used to map marker gene *symbols* onto the locus IDs
    # the proteome writes, and to tell a non-coding marker (legitimately absent
    # from a proteome) from a genuine mismatch. Optional per species.
    annotations = {}
    alias_attributes = {}
    for sp, files in reference_species.items():
        proteomes[sp] = files["proteome_file"]
        # A reference with no marker_file joins the OrthoFinder run for its
        # orthogroup structure but votes on no cell type.
        if files.get("marker_file"):
            markers[sp] = files["marker_file"]
        gene_id_rules[sp] = files.get("gene_id")
        if files.get("annotation_file"):
            annotations[sp] = files["annotation_file"]
        if files.get("alias_attributes"):
            alias_attributes[sp] = files["alias_attributes"]
    if "proteome_file" in target_species_files:
        proteomes[target_species] = target_species_files["proteome_file"]
        gene_id_rules[target_species] = target_species_files.get("gene_id")
        if target_species_files.get("annotation_file"):
            annotations[target_species] = target_species_files["annotation_file"]
        if target_species_files.get("alias_attributes"):
            alias_attributes[target_species] = target_species_files["alias_attributes"]
    # The target species may also carry a *reference* marker table -- published,
    # independent cell-type markers for the same organism. It is the crispest
    # evidence in the run: same-species markers map to orthogroups exactly as
    # the target clusters do, with no cross-species ortholog inference. Keyed by
    # the species name so resolve_marker_tokens finds the Medicago genemap and
    # proteome already prepared above -- a distinct label would have nothing to
    # resolve against. The proteome set is unchanged, so no OrthoFinder rerun;
    # only the cheap Fisher pass sees it. It appears as its own voter in
    # `support_species` (self-corroboration is real evidence, not noise), and
    # its own universe is pairwise, so adding it does not shift any other
    # species' q-values.
    if target_species_files.get("marker_file"):
        markers[target_species] = target_species_files["marker_file"]
        logger.info('[MAIN] Target species %s also votes from its own reference marker file',
                    target_species)

    # Run Orthofinder
    orthogroups = OrthoManager(
        proteome_dict = proteomes,
        marker_dict = markers,
        gene_id_rules = gene_id_rules,
        # cache_dir, not out_dir: everything OrthoManager writes
        # (orthofinder/, primary_transcripts/, genemaps/, validation/) is a
        # regenerable cache keyed by proteome content, not a deliverable.
        out_dir = cache_dir,
        force_rerun = force,
        threads = of_cfg.get("threads", 16),
        analysis_threads = of_cfg.get("analysis_threads", 8),
        stop_after_orthogroups = of_cfg.get("stop_after_orthogroups", True),
        min_marker_coverage = marker_cfg.get("min_coverage", 0.8),
        min_marker_coverage_overrides = marker_cfg.get("min_coverage_overrides") or {},
        marker_gene_column = reference_columns["gene"],
        annotation_dict = annotations,
        alias_attributes = alias_attributes,
        processing = config.get("proteome_processing") or {},
        crosscheck = config.get("annotation_crosscheck") or {},
    )

    if validate_only:
        orthogroups.prepare()
        logger.info('[MAIN] Validation complete; skipping OrthoFinder (--validate-only)')
        return

    of_out = orthogroups.run_orthofinder()

    # Get OG mappings
    gene_to_og = orthogroups.parse_gene_to_og(results_dir = of_out)
    og_to_genes = orthogroups.parse_og_to_genes(results_dir = of_out)

    if orthofinder_only:
        # Parsing above is deliberately still done: it is seconds of work and
        # it proves Orthogroups.tsv is readable and keyed the way the rest of
        # the stage expects. Finding that out now is worth much more than
        # finding it out at the start of the next run.
        logger.info('[MAIN] OrthoFinder complete: %s', of_out)
        logger.info('[MAIN] %d genes mapped across %d orthogroups',
                    len(gene_to_og), len(og_to_genes))
        logger.info('[MAIN] Stopping before annotation (--orthofinder-only)')
        return of_out

    # Build orthologous marker groups
    marker_dir = target_species_files['marker_file_dir']
    # Built once and shared: parsing the 1.3 MB ontology per sample is repeat work.
    standardizer = CellTypeStandardizer(ontology_file, extensions=ontology_extensions)
    if ontology_extensions:
        logger.info('[MAIN] Ontology extended with %d file(s): %s',
                    len(ontology_extensions), ', '.join(Path(e).name for e in ontology_extensions))

    # Phylogenetic weighting of the reference-species vote (see phylo.py). Built
    # over the species that vote (those with a marker file), for the annotator
    # and the review report. Without a `phylogeny:` block every species weighs
    # 1, so scores equal the unweighted n_species.
    clade_weights = None
    phylo_cfg = config.get('phylogeny')
    if phylo_cfg:
        clade_weights = phylo_build(phylo_cfg, species=sorted(markers), logger_=logger)

    # Marker files are normally produced by markerid and simply read
    # here; this block lets annotation generate a missing one instead of
    # failing halfway through a run.
    gen_cfg = marker_cfg.get("generate", {}) or {}
    gen_samples = dict(gen_cfg.get("samples", {}) or {})

    # A target given on the command line replaces the config's sample list
    # entirely, rather than being appended to it: the CLI form exists to
    # annotate one dataset the user is holding, and silently also running the
    # 17 samples in the config would be a surprising amount of extra compute.
    if adata:
        if not sample_name:
            raise ValueError('--adata requires --sample (the name the outputs are keyed on)')
        samples = [sample_name]
        gen_samples[sample_name] = adata
    else:
        samples = target_species_files["samples"]

    methods = list(methods or marker_cfg.get("methods")
                   or [gen_cfg.get("method", "wilcoxon")])
    cluster_key = cluster_key or gen_cfg.get("cluster_key")
    latent_key = latent_key or gen_cfg.get("latent_key")
    logger.info('[MAIN] Annotating %d sample(s) x %d marker method(s): %s',
                len(samples), len(methods), ', '.join(methods))

    failures = []
    for sample in samples:
        # Generated once per sample for every method, before any annotation, so
        # a marker method that dies takes only its own annotation with it.
        marker_files, marker_failures = ensure_marker_files(
            sample = sample,
            marker_root = marker_dir,
            methods = methods,
            adata_path = gen_samples.get(sample),
            cluster_key = cluster_key,
            latent_key = latent_key,
            counts_layer = gen_cfg.get("counts_layer", "corrected_counts"),
            n_genes = gen_cfg.get("n_genes", 50),
            env = gen_cfg.get("env"),
            repo_root = find_root_dir())
        failures.extend(marker_failures)

        # Loaded once for all six methods; scoring needs .layers, so this
        # cannot be a backed read.
        adata = None
        adata_path = gen_samples.get(sample)
        score_layer = marker_cfg.get('score_layer', 'data_norm')
        if adata_path and cluster_key:
            import scanpy as sc
            logger.info('[MAIN] Loading %s for geneset scoring', adata_path)
            adata = sc.read_h5ad(adata_path)
            # Built once per sample, in-memory only -- nothing upstream (mapping,
            # integration, marker generation) ever writes a normalised layer into
            # this h5ad; see ensure_score_layer's docstring for why.
            ensure_score_layer(adata, layer=score_layer,
                               source_layer=marker_cfg.get('score_source_layer',
                                                            'corrected_counts'))

        templates = {}
        for method, target_marker_file in sorted(marker_files.items()):
            logger.info('[MAIN] Annotating sample %s (method=%s)', sample, method)
            try:
                templates[method] = _annotate_one(
                    sample = sample, method = method,
                    target_marker_file = target_marker_file,
                    gene_to_og = gene_to_og, og_to_genes = og_to_genes,
                    markers = markers, standardizer = standardizer,
                    target_species = target_species,
                    reference_columns = reference_columns,
                    target_columns = target_columns,
                    q_threshold = q_threshold,
                    tissue = tissue, tissue_column = tissue_column,
                    max_dataset_children = max_dataset_children,
                    force_decisions = force_decisions,
                    adata = adata, cluster_key = cluster_key,
                    score_layer = score_layer,
                    seed_quantile = marker_cfg.get('seed_quantile', 0.9),
                    seed_margin = marker_cfg.get('seed_margin', 0.25),
                    clade_weights = clade_weights,
                    orthogroups = orthogroups, logger = logger,
                    out_dir = out_dir)
            except Exception as exc:                  # noqa: BLE001 - collected below
                logger.exception('[MAIN] Annotation failed for %s/%s', sample, method)
                failures.append(f'{sample}/{method}: {type(exc).__name__}: {exc}')

        # Cross-method comparison. Only meaningful once genesets have been
        # mapped onto a shared partition, which is what `assoc_cluster` does.
        if adata is not None and templates:
            clusters = sorted(adata.obs[cluster_key].astype(str).unique(),
                              key=lambda x: (len(x), x))
            consensus = build_consensus(templates, clusters=clusters)
            path = out_dir / 'results' / sample / '{}_consensus.tsv'.format(sample)
            consensus.to_csv(path, sep='\t')
            disputed = int((consensus['n_distinct'] > 1).sum())
            logger.info('[MAIN] Cross-method consensus -> %s (%d/%d cluster(s) with '
                        'more than one distinct call)', path, disputed, len(consensus))

            # The review page. Written last because it consumes the consensus
            # matrix above and every template alongside it, and wrapped because
            # a fault in the report must not cost the annotations it describes
            # -- the same trade the mapping stage makes for its own report. It
            # is a pure consumer, so `python -m annotation.review`
            # rebuilds it at any time without re-running anything.
            try:
                # One dataset, one directory: the review page sits with the decisions
                # it is used to produce, rather than in a sibling `review/` tree.
                review_path = out_dir / sample / '{}_review.html'.format(sample)
                write_review(
                    sample=sample, adata=adata,
                    annotation_dir=out_dir / 'results',
                    methods=sorted(templates), cluster_key=cluster_key,
                    out_path=review_path,
                    gene_groups=load_report_gene_groups(marker_cfg.get('report_genes'),
                                                        root=root_dir, species=target_species),
                    seed_quantile=marker_cfg.get('seed_quantile', 0.9),
                    seed_margin=marker_cfg.get('seed_margin', 0.25),
                    clade_weights=clade_weights)
            except Exception:                         # noqa: BLE001 - report is not the deliverable
                logger.exception('[MAIN] Review report failed for %s; annotations are '
                                 'unaffected', sample)
        del adata

    if failures:
        logger.error('[MAIN] %d of %d (sample x method) run(s) failed:\n  %s',
                     len(failures), len(samples) * len(methods), '\n  '.join(failures))
        raise RuntimeError(
            '{} annotation run(s) failed; see the log for each. Successful runs '
            'have been written.'.format(len(failures)))
    logger.info('[MAIN] All %d annotation run(s) completed', len(samples) * len(methods))


def _annotate_one(sample, method, target_marker_file, gene_to_og, og_to_genes,
                  markers, standardizer, target_species, reference_columns,
                  target_columns, q_threshold, tissue, tissue_column,
                  max_dataset_children, orthogroups, logger, out_dir,
                  force_decisions=False, adata=None, cluster_key=None,
                  score_layer='data_norm', seed_quantile=0.9, seed_margin=0.25,
                  clade_weights=None):
    """One (sample, method) annotation. Outputs are keyed by both."""
    omg = OMGBuilder(
            gene_to_og = gene_to_og,
            og_to_gene = og_to_genes,
            marker_dict = markers,
            target_marker_file = target_marker_file,
            standardizer = standardizer,
            target_species = target_species,
            reference_columns = reference_columns,
            target_columns = target_columns,
            q_threshold = q_threshold,
            # Same resolution validation just reported, so a marker counted as
            # recovered there actually votes here.
        marker_resolver = orthogroups.resolve_marker_tokens,
        tissue_column = tissue_column)
    omg.build_omgs()
    # Keyed by method as well as sample: the six methods annotate the same
    # cells and would otherwise overwrite one another's output.
    method_out_dir = out_dir / 'results' / sample / method
    annotations = scAnnotator(
        omg = omg,
        out_dir = method_out_dir,
        # No '.tsv' here: scAnnotator appends '_significant_annotations.tsv' itself.
        outfn = sample,
        standardizer = standardizer,
        clade_weights = clade_weights,
        logger = logger).consensus_annotation(
            q_thresh = q_threshold, tissue = tissue,
            max_dataset_children = max_dataset_children)

    # Decision template. Built from the *full* geneset list rather than from
    # the annotations, so clusters with no significant hit still appear as
    # `none` rows -- their cells exist whether or not a cell type was found,
    # and on this data that is 10 of 16 cfmf modules.
    genesets = _read_genesets(target_marker_file, target_columns['cluster'])

    # Score each geneset per cell and find the cluster its confident cells sit
    # in. Cheap relative to everything else here, and it is what lets the six
    # methods be compared against one shared partition.
    association = None
    if adata is not None and cluster_key:
        sets = read_genesets(target_marker_file, cluster_column=target_columns['cluster'],
                             gene_column=target_columns['gene'])
        scores = score_genesets(adata, sets, layer=score_layer)
        scores.to_csv(method_out_dir / '{}_geneset_scores.tsv'.format(sample), sep='\t')
        association = associate_with_clusters(scores, adata.obs[cluster_key],
                                              quantile=seed_quantile, margin=seed_margin)
        logger.info('[ANNOTATION] %s/%s: %d geneset(s) associated with a cluster',
                    sample, method, int((association['assoc_cluster'] != '').sum()))

    template = build_template(annotations, genesets, association=association)
    path = method_out_dir / '{}_decisions.tsv'.format(sample)
    try:
        write_template(path, template, force=force_decisions)
        logger.info('[ANNOTATION] Decision template: %s (%d geneset(s), %d pre-filled)',
                    path, len(template), int((template['top_po_id'] != '').sum()))
    except DecisionError as exc:
        # An edited template is the human's work; never lose it to a rerun.
        logger.warning('[ANNOTATION] %s', exc)
    return template


def _read_genesets(marker_file, cluster_column):
    """Every cluster/module the marker method produced, in file order."""
    df = pd.read_csv(marker_file, sep='\t', dtype={cluster_column: str})
    return list(dict.fromkeys(df[cluster_column].astype(str)))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Peform single cell annotation')
    parser.add_argument('--acfg', required=True, help='Path(s) to annotation config')
    parser.add_argument('--species',
                        help='Which target species profile to use (see '
                             '`species_profiles:` in the annotation config). '
                             'Omit to use `default_species:`.')
    parser.add_argument('--force', required=False, action = 'store_true', help='Force rerun of orthofinder')
    parser.add_argument('--validate-only', required=False, action = 'store_true',
                        help='Prepare primary-transcript proteomes and check marker coverage, then stop')
    parser.add_argument('--orthofinder-only', required=False, action = 'store_true',
                        help='Run OrthoFinder and parse the orthogroups, then stop. Use when the '
                             'target marker files do not exist yet -- the annotation loop needs '
                             'them and would otherwise fail only after OrthoFinder had finished.')
    parser.add_argument('--adata', required=False,
                        help='Target .h5ad to annotate. Replaces the config sample list; '
                             'requires --sample.')
    parser.add_argument('--sample', dest='sample_name', required=False,
                        help='Name the outputs are keyed on (with --adata)')
    parser.add_argument('--latent-key', required=False,
                        help='.obsm key for the latent space the cluster-free marker methods '
                             'build their kNN graph on (e.g. X_scVI). Never pass a UMAP.')
    parser.add_argument('--cluster-key', required=False,
                        help='.obs key the cluster-dependent marker methods group by '
                             '(e.g. leiden_scVI)')
    parser.add_argument('--methods', required=False, nargs='+',
                        help='Marker methods to run. Default: markers.methods in the config.')
    parser.add_argument('--force-decisions', required=False, action='store_true',
                        help='Overwrite decision templates even if they have been edited')
    parser.add_argument('--tissue', required=False, nargs='+',
                        help='Tissue(s) of interest, e.g. Root (case-insensitive). Flags '
                             'candidates never seen in them as tissue_match=False. Applied '
                             'post-hoc: markers are NOT filtered, since that weakens every '
                             'Fisher test. Default: every tissue.')
    args = parser.parse_args()
    main(annotation_config_path = args.acfg, species=args.species, force=args.force, validate_only=args.validate_only,
         orthofinder_only=args.orthofinder_only, adata=args.adata,
         sample_name=args.sample_name, latent_key=args.latent_key,
         cluster_key=args.cluster_key, methods=args.methods, tissue=args.tissue,
         force_decisions=args.force_decisions)