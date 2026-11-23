# Configuration

Start from the example in [configs/examples/minimal/](../configs/examples/minimal/README.md) and copy it to `configs/local/`, which Git ignores. This keeps your own settings separate from the study configurations under `configs/`.

| What | Where | What to set for a new dataset |
|---|---|---|
| Samples | `samples`; per sample `fastq1`, `fastq2`, `batch`, `sample_type` | Paired FASTQ lists in matching lane order, `protoplast` or `nuclei`, sample names and metadata |
| Chemistry | `technologies`; per sample `tech` | Barcode and UMI geometry and the matching whitelist |
| Reference | `indexes`, `mapping_config`; optionally per sample `index_ref` | FASTA, GTF, index location and reference profile |
| QC metrics | `QCconfigs`; per sample `qc_metrics` | Organelle gene identifiers and annotation attributes for the species, count layers |
| Filters | `QCfilterconfigs`; per sample `qc_filter` | Thresholds, modes and method options |
| Machine | `output_dir`, `cellbender.env`, `doublets.env`, `tmp_dir` | Output and scratch locations, conda environment prefixes |

Sample `defaults:` are merged shallowly: a non-null value on a sample replaces the default, and lists are replaced rather than combined. A mapping cohort selects a sample sheet and, optionally, an output directory; it does not carry its own filter defaults.

## Paths

The mapping stage reads its YAML files as they are. Relative paths, including those to other configuration files and inputs, are resolved from the working directory, which is why commands are run from the repository root. `${VAR}` and `~` are not expanded, so use full paths. `CELLBENDER_ENV` and `DOUBLET_ENV` are only used when `cellbender.env` or `doublets.env` is empty.

Integration configurations behave differently. Their `extends:` key expands environment variables and resolves a relative parent path from the file that contains it. Target and group names may not be defined twice, and `defaults:` are merged per key.

## References and whitelists

Download the FASTA and GTF/GFF from the same assembly and annotation release; the GTF used for QC metrics must describe the reference used for mapping. Obtain the whitelist for your chemistry from its provider. Record the source URL, release, download date, any preprocessing and the SHA-256 of each file. When reproducing an earlier result, use the reference it was produced with. Reference files are not included in the repository.

For another species, check the mitochondrial and chloroplast gene identifiers and the attributes available in your GTF. The example uses the Medicago rules, which will not match other genomes. Index options, including intronic counting, are taken from our own runs and have not been tuned for other data.

## Study configurations

The YAML files directly under `configs/`, `configs/C14/` and the chemistry re-check configurations are the inputs of our own studies and contain cluster-specific paths. Some public-data cohorts are incomplete until their downloads finish (`cra007849`, `cra008337`); a `.partial` sample sheet is not usable. The `marcantia` example refers to a mapper profile that does not exist and needs its own reference preparation. Run the site diagnostics (`python -m sc test site`) before reusing any of these.

`configs/markerid/adata_paths.tsv` refers to an older layout of our results. Check the paths before using it.

## Schemas

The JSON schemas in `configs/schemas/` provide editor hints and are generated from `shared/config_spec.py`. After changing the specification, regenerate them:

```bash
python -m mapping.validate --emit-schemas
```

`tools/check_release.py` checks that the committed schemas match the specification.
