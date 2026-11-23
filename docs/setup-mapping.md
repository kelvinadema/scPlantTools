# Mapping and QC setup

All commands are run from the repository root, the directory that contains `.root`, `mapping/` and `configs/`. If you run from elsewhere, set `SCRNASEQ_ROOT` to that directory. On our cluster the source and the data are kept on different file systems and joined with symbolic links; a normal clone needs no such links.

## Environments and patches

Create the environments from the version-pinned files in `envs/`. When reproducing an earlier analysis, do not update the package versions.

```bash
conda env create -f envs/scMappingV3.12.yml
conda env create -f envs/cellBenderENV.yml
conda env create -f envs/scIntegrationV1.12.yml
```

Two of these need an extra step, described in [envs/README.md](../envs/README.md): CellBender must be patched, and BD Rhapsody data need a patched piscem build. The stock and patched piscem report the same version, so check the barcode geometry rather than `--version`.

```bash
bash tools/piscem-bd/build.sh
conda activate scMappingV3.12
export CELLBENDER_ENV="/absolute/path/to/conda/envs/cellBenderENV"
export DOUBLET_ENV="/absolute/path/to/conda/envs/scIntegrationV1.12"
source batch_scripts/lib/mapping_setup.sh
sc_mapping_setup
```

CellBender and the SOLO doublet model run in their own environments. Their locations are taken from `cellbender.env` and `doublets.env` in the pipeline configuration; the two environment variables are only used when those keys are empty.

`sc_mapping_setup` puts `tools/bin` first on the `PATH` (so the patched piscem is used), initialises simpleaf's tool cache in `af_home`, disables HDF5 file locking and sets the open-file limit to 2048. Build indices with `TMPDIR` on node-local storage, and set the number of threads to match your scheduler allocation.

## Configure, check and run

Copy and edit the [minimal example](../configs/examples/minimal/README.md). It is a single self-contained configuration, so mapping, `sc status` and manual refiltering all use the same sample sheet and output directory.

```bash
python -m shared.mapping_setup -c configs/local/pipeline.yaml
python -m mapping.validate -c configs/local/pipeline.yaml
python -m mapping.main -c configs/local/pipeline.yaml --build-indexes
```

The first command checks the BD Rhapsody barcode geometry of the piscem that simpleaf will actually use, if the sample sheet contains BD samples. The second validates the configuration and checks that inputs, environments and indices are in place. On a login node it will report CUDA as unavailable when GPU use is configured; this is expected and does not need a configuration change.

On SLURM, check the resource directives in the launcher and submit from the repository root:

```bash
mkdir -p logs
sbatch batch_scripts/run_mapper.sh --config configs/local/pipeline.yaml
```

The launcher runs `sc_mapping_setup` and the BD geometry check with the same `--config` and `--cohort` it passes to mapping; `mapping.main` then runs its own preflight validation. `python -m sc map ...` selects the mapping environment but does not replace `sc_mapping_setup`.

## Reruns

Each completed step writes a `.done` flag and is skipped on the next run. Changing a configuration value does not invalidate earlier steps. To recompute a step and the steps after it, use one of `--force-mapping`, `--force-preprocessing`, `--force-qc`, `--force-plotting`, `--force-mapextraction`, `--force-report`, `--force-doublets` or `--force-index-rebuild NAME`.

Only the `alevin` mapper (simpleaf, piscem and alevin-fry) is implemented. kb-python and STARsolo appear in the configuration but are rejected before mapping starts.

See [outputs](outputs.md) for reviewing and refiltering samples and [configuration](configuration.md) for choosing QC profiles. QC thresholds are analysis choices and should be set for the data at hand, not adjusted to get through installation checks.
