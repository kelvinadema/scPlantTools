# Outputs and manual QC

Mapping writes one directory per sample under `output_dir`. The layout of the raw, processed and QC subdirectories is defined in `shared.utils.get_sample_output_paths`. Each sample's `manifest.json` lists its output files (as relative paths), their provenance and whether the run completed.

| File | Contents |
|---|---|
| `{sample}_filtered.h5ad` | Cells retained by the automatic filters, with all count layers and QC metadata |
| `{sample}_report.html` | QC report, including the interactive manual review |
| `metadata.tsv`, `MappingStats.tsv` | Per-cell QC metadata and mapping statistics |
| `{sample}_decisions.json` | Manual QC decisions exported from the report |
| `{sample}_filtered_manual.h5ad` | Result after applying the manual decisions; the automatic result is kept |

## Manual review

Open the HTML report, inspect the diagnostics and export the decisions as JSON into the sample's QC directory. Then apply them:

```bash
conda activate scMappingV3.12
python -m mapping.refilter -c configs/local/pipeline.yaml --sample example_cell
python -m sc status -c configs/local/pipeline.yaml --sample example_cell
```

Refiltering checks that the decisions were made on the same automatic result they are applied to. To use the manually filtered object for integration, set `artifact_key: filtered_manual_h5ad` in the integration target. `batch_scripts/run_refilter.sh` contains the sample list and options of one of our studies and is not meant for new data.

## Cohorts

Mapping accepts `--cohort`, but `mapping.refilter` and `sc status` read the sample sheet and output directory at the top level of the pipeline configuration. To use them on a cohort, make a local copy of the pipeline YAML in which `configs.samples` and `output_dir` point to that cohort, remove `cohorts` and `default_cohort`, and pass the copy with `-c`.

## Notes on stored results

The count layers, the order of cells and genes, the filter flags and the separation of automatic and manual results are what downstream analyses rely on. H5AD files can differ byte for byte while containing the same counts, because they store timestamps and provenance; do not rewrite old files only to update this metadata.

Integration, annotation, marker identification and refinement write to their own output directories, which are not covered by the mapping manifest. `sc status` reports what it can find there but cannot tell whether those stages finished.

`.done` flags and the integration model registry are used to skip work that has already been done. They do not need to be regenerated after reorganising files.
