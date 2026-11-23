# Minimal mapping and QC configuration

This example defines one cell and one nucleus sample, both 10x v3, with the default QC metrics and filters. No data are included. The QC settings are the defaults used in our own runs; the organelle gene identifiers are specific to Medicago and must be changed for other species.

From the repository root:

```bash
mkdir -p configs/local
cp -n configs/examples/minimal/*.yaml configs/local/
```

`cp -n` does not overwrite files that already exist. Then edit the copies:

- `samples.yaml`: sample names, preparation type, batches and paired FASTQ paths.
- `indexes.yaml`: the reference FASTA and GTF and where to build the index.
- `technologies.yaml`: the whitelist location. Change the chemistry only if your libraries use a different one.
- `qc.yaml`: the path to the same annotation and the organelle gene definitions for your species.
- `pipeline.yaml`: the output directory and the conda environment prefixes. The prefixes can be left empty if `CELLBENDER_ENV` and `DOUBLET_ENV` are set.
- `filter.yaml`: the QC thresholds and modes. Check whether they suit your data.

The paths in `pipeline.yaml` point to `configs/local/`, and commands are run from the repository root. The YAML files do not expand shell variables. Both samples share one reference; remove the one you do not need.

Then follow [mapping setup](../../../docs/setup-mapping.md). The example uses a GPU and per-sample model tuning, so it needs the same resources as our own runs. A test checks that the numerical settings here match the defaults in the study configuration.

For editor schema hints in the copied files, change `../../schemas/` to `../schemas/` in their first line.
