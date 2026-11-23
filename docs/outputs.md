# Outputs and manual QC

Mapping writes beneath the configured `output_dir`, with one directory per sample. The mapper-specific raw/processed/QC subdirectories are resolved by `shared.utils.get_sample_output_paths`; preserve these names for downstream consumers. Each sample's `manifest.json` lists relative artifact paths, provenance, and completion information.

| Artifact | Purpose |
|---|---|
| `{sample}_filtered.h5ad` | Automatic filtered cells and recorded count layers/QC metadata |
| `{sample}_report.html` | QC report and interactive manual review |
| `metadata.tsv`, `MappingStats.tsv` | Per-cell QC metadata and mapping summaries |
| `{sample}_decisions.json` | Exported manual decisions; retain as a scientific input |
| `{sample}_filtered_manual.h5ad` | Separate manual result; automatic output is preserved |

Open the HTML report, review the diagnostics, and export the decisions JSON to the sample's QC directory. For the standalone example configuration:

```bash
conda activate scMappingV3.12
python -m mapping.refilter -c configs/local/pipeline.yaml --sample example_cell
python -m sc status -c configs/local/pipeline.yaml --sample example_cell
```

Integration opts into manual output with `artifact_key: filtered_manual_h5ad` in its target configuration. Review decisions are checked against parent provenance; do not bypass that check as a generic setup step. The local `run_refilter.sh` retains its historical study sample list and parent-mismatch override; it is not this generic workflow.

## Existing cohort-based study runs

Mapping honors `--cohort`, but current manual refilter and status read the top-level sheet/output root. For those commands, create a dedicated *local copy* of the pipeline YAML whose `configs.samples` and `output_dir` identify the selected cohort, and omit `cohorts`/`default_cohort`. Keep all other values identical. Use it explicitly with `-c`; do not edit the active study config or redirect existing output directories. This documented compatibility route avoids changing existing command semantics during organization.

## Preservation and optional downstream outputs

Count layers, ordered cell/gene identities, filtering flags, and automatic/manual artifact distinctions are part of the scientific contract. Raw-file hashes of H5AD can also change due to timestamps/provenance without count changes; never rewrite old outputs just to update their metadata.

Integration, annotation, markers, and refinement have separate output roots. Their output completeness is not fully covered by the mapping manifest contract. `sc status` reports those stages conservatively. Historical nested annotation directories are a site-layout diagnostic, not evidence that their contents may be deleted. Notebook links and consumers must be traced before any results relocation.

`.done` sentinels and model registries intentionally reuse existing work. Organization does not require regenerating them or retraining models.
