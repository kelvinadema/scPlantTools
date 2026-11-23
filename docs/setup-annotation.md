# Annotation and refinement

Annotation assigns Plant Ontology cell types to clusters using marker genes from reference species, matched through orthogroups. It is optional and independent of mapping and QC. When reproducing a result, use the same ontology files, proteome versions and marker tables.

Annotation runs in `scAnnotationV3.12`, marker identification in `scMarkersV3.12` and refinement in `scIntegrationV1.12` (see [envs/README.md](../envs/README.md)). The `sc` command selects the right environment for each.

`configs/annotation_config.yaml` holds the species profiles and inputs of our own studies. Write a local configuration and run:

```bash
python -m annotation.main --help
python -m annotation.main --acfg configs/local/annotation.yaml --validate-only
python -m annotation.main --acfg configs/local/annotation.yaml --orthofinder-only
python -m annotation.main --acfg configs/local/annotation.yaml
```

`--validate-only` prepares and checks the reference inputs and may write cache files. `--orthofinder-only` runs the orthology step without the marker inputs needed for the full run. A species profile can set the target species and exclude it from the reference set.

The annotation results are reviewed on an HTML page (`python -m annotation.review --help`), where genesets from the different marker methods are grouped into cell types. Refinement uses that grouping to seed scANVI (`python -m refine.main --help`). The seeding rule can be set globally or per target with `seeds.quantile_scope`.

Background on the methods is in the [marker port notes](markerid-porting.md), the [proteome preparation notes](decisions/proteome-preparation.md) and the other [design notes](decisions/).

`integration/` also contains a few one-off maintenance scripts (barcode index backfills and label injection into existing objects). They are not part of a normal annotation run; read their targets and dry-run options before using them on existing results.
