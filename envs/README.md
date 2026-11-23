# Conda environments

There is one environment per stage. They are kept separate mainly because the stages need incompatible pandas versions. Each `.yml` file was exported from the cluster environments on 2026-07-21 with all versions pinned, so `conda env create -f <file>` rebuilds the environments that produced our results.

| Environment | Used for | Python | Reason for a separate environment |
|---|---|---|---|
| [`scMappingV3.12.yml`](scMappingV3.12.yml) | Mapping, QC and filtering (`mapping`) | 3.12.13 | pandas 3.0.3; contains simpleaf and alevin-fry, but not the patched piscem (see below) |
| [`cellBenderENV.yml`](cellBenderENV.yml) | CellBender, run as a subprocess during mapping | 3.10.20 | Pinned torch; needs a patch (see below) |
| [`scIntegrationV1.12.yml`](scIntegrationV1.12.yml) | Integration, refinement and the SOLO doublet model | 3.12.11 | scvi-tools does not support pandas 3 |
| [`scAnnotationV3.12.yml`](scAnnotationV3.12.yml) | Annotation (`annotation`) | 3.12.3 | Contains the OrthoFinder tools |
| [`scMarkersV3.12.yml`](scMarkersV3.12.yml) | Marker identification (`markerid`) | 3.12.13 | hotspot 1.1.3 requires pandas below 3 |

```bash
conda env create -f envs/scMappingV3.12.yml      # and likewise for the others
```

## Steps after creating the environments

Two environments do not work correctly as created. In both cases the run can finish without an error and still give wrong results, so both steps are needed, and they have to be repeated whenever the environment is rebuilt.

| Environment | Step | Without it |
|---|---|---|
| `scMappingV3.12` | `bash tools/piscem-bd/build.sh` | Stock piscem shortens BD Rhapsody 27 bp cell barcodes to 9 bp |
| `cellBenderENV` | Apply `cellbender-checkpoint-weakref.patch` | CellBender crashes after about 150 epochs |

The CellBender and doublet environments are referred to by path. Set `cellbender.env` and `doublets.env` in your pipeline configuration, and `env` in the annotation configuration, to the prefixes of your own installation.

## Patched piscem for BD Rhapsody data

The `piscem=0.21.1` in `scMappingV3.12.yml` is the stock conda build. For BD Rhapsody Enhanced Bead V2 data it keeps only the last of the three linker-separated blocks of the 27 bp cell label, without a warning. About 384³ possible cell labels are then reduced to about 384, and this cannot be detected later in the pipeline.

Our runs therefore use a patched piscem in `tools/bin/piscem`. `sc_mapping_setup` (in `batch_scripts/lib/mapping_setup.sh`, also called by `run_mapper.sh`) puts `tools/bin` first on the `PATH` before simpleaf records its tool paths. The patches, the cause of the problem and how the fix was validated are described in [piscem_bd_rhapsody/README.md](../piscem_bd_rhapsody/README.md).

The stock and patched binaries both report version 0.21.1, so `--version` does not tell them apart; the barcode geometry does (stock reports `bc_len=9`, patched `bc_len=27`). `tools/bin/` is not tracked by Git, so a fresh clone has to build it:

```bash
bash tools/piscem-bd/build.sh     # about 15 minutes, needs network access
```

The build script installs its own Rust toolchain under `tools/piscem-bd/rust/`, leaving `$HOME` and the conda environments untouched. It runs the upstream test suites against the patched code and checks for `bc_len=27` before installing. For an end-to-end check, run `sbatch tools/piscem-bd/smoke_test.sh`.

When the sample sheet contains BD samples, `run_mapper.sh` stops if piscem does not report `bc_len=27`. This check protects against a rebuilt environment silently falling back to the stock binary.

simpleaf 0.26.2 contains the same `seq_geom_parser` 1.1.0, but only validates the geometry and passes it on to piscem, so patching piscem is enough. 10x data are not affected.

## CellBender patch

CellBender 0.3.2 imports without problems but fails during a run: its checkpoint writer tries to pickle the model object and stops with `TypeError: cannot pickle 'weakref.ReferenceType' object` after about 150 epochs, followed by `AssertionError: Checkpoint file ckpt.tar.gz does not exist`. Changing the Python or torch version does not help (CellBender issues [#296](https://github.com/broadinstitute/CellBender/issues/296) and [#395](https://github.com/broadinstitute/CellBender/issues/395)).

The fix from [PR #444](https://github.com/broadinstitute/CellBender/pull/444) is included as [`cellbender-checkpoint-weakref.patch`](cellbender-checkpoint-weakref.patch):

```bash
conda activate cellBenderENV
patch -p0 -d "$CONDA_PREFIX/lib/python3.10/site-packages/cellbender/remove_background" \
      < envs/cellbender-checkpoint-weakref.patch
```

Adding `--forward` skips an installation that is already patched. The patch also applies to a Python 3.12 installation; only the `python3.10` part of the path changes.

CellBender is also installed, unpatched, in `scMappingV3.12`. The pipeline does not use that copy, because it runs CellBender from `cellBenderENV`; patch it as well if you run `remove-background` from the mapping environment.

The patch saves the model's `state_dict()` and the optimiser through `PyroOptim`'s own `.save()` instead of pickling the objects. A single `remove-background` run keeps the model in memory and never reloads it, so this works for the pipeline. It does break CellBender's option to resume training from a checkpoint, which the pipeline does not use.

## How the files were made

The files were exported with `conda env export --no-builds` and then corrected by hand. Leaving out build strings keeps the exact versions but avoids channel- and platform-specific builds, which often prevent an environment from solving on another machine. All environments are `linux-64`.

Three corrections were made:

1. The `prefix:` line, an absolute path in one user's home directory, was removed.
2. The `defaults` channel was removed. None of the packages in the five environments came from it (all are from conda-forge, bioconda or pip), and dropping it avoids the Anaconda terms of service for larger organisations.
3. Versions were checked against what is actually imported:
   - numpy in `scIntegrationV1.12` is set to 2.2.0. The package metadata in that environment is out of date and reports 1.26.4, while `numpy.__version__` is 2.2.0. An uncorrected export would therefore install numpy 1.x, a different major version, under torch 2.8 and scvi-tools 1.5. `shared/io.py` detects this mismatch at run time and records the imported version.
   - `torch` and `torchvision` keep their `+cu126` and `+cu130` versions, and the matching `--extra-index-url` is added to the pip section. Without it, pip either fails or installs a different torch build.

Because of the third correction, do not overwrite these files with a fresh `conda env export`. Either repeat the corrections or first repair the environment metadata.

## GPU

`scMappingV3.12` (CUDA 12.6), `cellBenderENV` (CUDA 13.0) and `scIntegrationV1.12` (CUDA 12.6) contain CUDA builds and need a GPU node. `scAnnotationV3.12` and `scMarkersV3.12` run on CPU only. If your driver needs a different CUDA version, change the `--extra-index-url` and the `+cuXXX` suffix together.
