# Configuration

The maintained example is [configs/examples/minimal/](../configs/examples/minimal/README.md). Copy it into ignored `configs/local/` so adapting the pipeline does not overwrite the scientific record under `configs/*.yaml`.

| Category | Files/keys | What to change for a new run |
|---|---|---|
| Sample inputs | `samples`, `fastq1`, `fastq2`, `batch`, `sample_type` | Paired FASTQ lists in matching lane order; `protoplast` or `nuclei`; sample names and metadata |
| Chemistry | `technologies`, sample `tech` | Barcode/UMI geometry and the matching permit list |
| Reference | `indexes`, `mapping_config`, optional sample `index_ref` | FASTA, GTF, index destination, and intentionally selected reference profile |
| QC metrics | `QCconfigs`, sample `qc_metrics` | Species-appropriate organelle identifiers and annotation attributes; count layers |
| Filtering | `QCfilterconfigs`, sample `qc_filter` | Thresholds, modes, and existing method options |
| Machine/runtime | `output_dir`, `cellbender.env`, `doublets.env`, `tmp_dir` | Storage roots, conda prefixes, scratch location |

Sample `defaults:` are merged shallowly; non-null per-sample values replace defaults. Lists are replaced, not concatenated. Mapping cohorts select a sheet and optionally an output root. They do not introduce an independent set of filter defaults.

## Path and precedence rules

Mapping uses plain YAML loading. Relative subconfig and input paths resolve from the process working directory, which is why commands run from the checkout root. Arbitrary `${VAR}` or `~` strings in mapping YAML are not generally expanded. Use concrete paths. `CELLBENDER_ENV` and `DOUBLET_ENV` are explicit subprocess-environment fallbacks; YAML prefixes win when present.

Integration's `extends:` is different: its loader expands environment variables and resolves relative parent-config paths beside the overlay. Target/group name collisions are errors; `defaults:` merges per key. Preserve these existing rules when relocating configurations.

## Reference and whitelist records

Acquire FASTA and GTF/GFF from the chosen provider and keep the exact assembly/annotation releases together. The GTF used for metrics must describe the reference used for mapping. Obtain the chemistry's whitelist from its provider. Record provider/download URL, release, retrieval date, any preprocessing script, and SHA-256 of the resulting files. Do not substitute a current reference for an older one when reproducing results. Large inputs are intentionally not bundled.

For another species, verify the mitochondrial/chloroplast matching rules and available GTF attributes. The example preserves the existing Medicago default rules; it is not a claim that those identifiers occur in other plants. The existing index options, including intronic counting, are likewise copied rather than retuned.

## Existing study material

The top-level YAMLs, `C14/`, and chemistry-recheck configurations remain in place, unchanged. They are study inputs, not the generic quick start. Some cohorts are incomplete until fetching finishes (`cra007849`, `cra008337`); a `.partial` sheet is not a usable manifest. The `marcantia` example names an unavailable mapper profile and needs reference/input preparation. Run the explicit site diagnostics before reusing these inputs.

`configs/markerid/adata_paths.tsv` refers to an older study output layout. Verify intended inputs before using its associated study launcher. Keep existing names and paths until downstream notebook references have been checked.

## Schemas

`shared/config_spec.py` generates `configs/schemas/*.json`. Regenerate after updating the specification:

```bash
python -m mapping.validate --emit-schemas
```

The release check compares generated and checked-in schemas. Schema descriptions must follow existing supported behavior, including refinement seed options; changing a description must not change a runtime default.
