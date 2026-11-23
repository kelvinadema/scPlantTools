# Minimal mapping/QC configuration

This example contains paired cell/nucleus entries using the same existing 10x v3 and default QC/filter settings. Input data is not bundled. The scientific values are copied from the working default profile; organelle identifiers are Medicago-specific and require deliberate adaptation for another species.

From the checkout root:

```bash
mkdir -p configs/local
cp -n configs/examples/minimal/*.yaml configs/local/
```

`cp -n` leaves existing local files intact. Edit the copied files:

- `samples.yaml`: sample names, preparation, batches, paired FASTQ paths.
- `indexes.yaml`: matching reference FASTA/GTF and the index destination.
- `technologies.yaml`: whitelist location; change chemistry only to match your library.
- `qc.yaml`: the same annotation's path and species-specific feature definitions.
- `pipeline.yaml`: output location and environment prefixes, or leave prefixes null and set `CELLBENDER_ENV`/`DOUBLET_ENV`.
- `filter.yaml`: inspect the existing thresholds/modes before deciding whether they suit your data.

Paths in `pipeline.yaml` deliberately point to `configs/local/`. Commands run from the repository root; these YAML files do not perform general shell-variable expansion. Both samples share a reference in this example; remove either sample if it is not part of your experiment.

Follow [mapping setup](../../../docs/setup-mapping.md). No active study YAML needs editing. The example defaults preserve GPU use and per-sample tuning, so they retain the existing resource requirements. Tests verify equality of the copied numerical settings against the working defaults, allowing only the listed input/site substitutions.

For editor schema hints after copying, change `../../schemas/` to `../schemas/` in the copied YAML modelines. This affects editor hints only.
