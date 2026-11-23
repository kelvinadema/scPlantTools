# Mapping and QC setup

Run commands from the logical execution root containing `.root`, `mapping/`, and `configs/`. On the development cluster this is the nobackup tree; source files link to the backup checkout. A normal clone needs no such symlinks. `SCRNASEQ_ROOT`, when set, must identify the intended execution root.

## Environments and tool patches

Use the existing manifests; do not update versions while reproducing an existing analysis:

```bash
conda env create -f envs/scMappingV3.12.yml
conda env create -f envs/cellBenderENV.yml
conda env create -f envs/scIntegrationV1.12.yml
```

Follow [envs/README.md](../envs/README.md) for the CellBender checkpoint patch and patched piscem build. Stock and patched piscem share a version string; BD requires the geometry check, not just `--version`. Preserve the patch versions used by an existing run.

```bash
bash tools/piscem-bd/build.sh
conda activate scMappingV3.12
export CELLBENDER_ENV="/absolute/path/to/conda/envs/cellBenderENV"
export DOUBLET_ENV="/absolute/path/to/conda/envs/scIntegrationV1.12"
source batch_scripts/lib/mapping_setup.sh
sc_mapping_setup
```

The exports are fallbacks when the corresponding YAML `env` values are unset. An explicit YAML prefix takes precedence in the pipeline. The example leaves these values unset; existing study configs retain their prefixes.

The shared shell helper preserves the launcher's existing behavior: prefer `tools/bin`, initialize simpleaf's tool cache in `af_home`, disable HDF5 file locking, and request the existing file-handle limit. Use node-local `TMPDIR` for index construction. Set thread allocation consistently with your scheduler; this setup does not change analysis hyperparameters.

## Configure, preflight, and run

Copy and edit the [minimal example](../configs/examples/minimal/README.md). It is a standalone config with no cohort indirection, so mapping, status, and manual refilter all use the same sheet and output root.

```bash
python -m shared.mapping_setup -c configs/local/pipeline.yaml
python -m mapping.validate -c configs/local/pipeline.yaml
python -m mapping.main -c configs/local/pipeline.yaml --build-indexes
```

The first check verifies BD geometry using the executable actually cached by simpleaf, when the resolved sheet includes BD samples. The second validates the configuration and execution prerequisites. Missing CUDA is expected on a login node when GPU execution is configured; do not alter scientific settings merely to make that check pass there.

On SLURM, review the resource directives and run from the checkout root:

```bash
mkdir -p logs
sbatch batch_scripts/run_mapper.sh --config configs/local/pipeline.yaml
```

The launcher selects the same `--config` and `--cohort` for its setup checks and execution. The legacy invocation and resource directives remain available. `python -m sc map ...` activates the stage environment but does not replace the setup helper.

## Reruns and supported backends

Existing `.done` flags control reuse. Changing a YAML value does not automatically invalidate every dependent artifact. Force flags include `--force-mapping`, `--force-preprocessing`, `--force-qc`, `--force-plotting`, `--force-mapextraction`, `--force-report`, `--force-doublets`, and `--force-index-rebuild NAME`. Use them only when intentionally recomputing that step and its dependents.

Only `alevin` is implemented end-to-end. KB and STARsolo placeholders remain in source/configs and are deliberately rejected. No mapper functionality or QC method was added by the release cleanup.

See [outputs](outputs.md) for review/refiltering and [configuration](configuration.md) for profile selection. Existing tuning and small-sample safeguards are unchanged; profile values are scientific choices, not installation settings.
