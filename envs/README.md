# Conda environments

Five environments, one per stage. They are deliberately separate — the split is
forced by **pandas**, not by preference (see the table). Each `.yml` here was
captured from the live cluster env on **2026-07-21** and is fully version-pinned,
so `conda env create -f <file>` reproduces the environment that actually
produced this project's results.

| Env | Used by | Python | Why separate |
|---|---|---|---|
| [`scMappingV3.12.yml`](scMappingV3.12.yml) | Stage 1 mapping/QC/filter — `batch_scripts/run_mapper.sh` | 3.12.13 | pandas **3.0.3**; carries simpleaf/alevin-fry (**not** the piscem actually used — see below) |
| [`cellBenderENV.yml`](cellBenderENV.yml) | CellBender, as a subprocess of stage 1 | 3.10.20 | pins torch; **needs a patch** (below) |
| [`scIntegrationV1.12.yml`](scIntegrationV1.12.yml) | Stage 2 integration + the doublet worker | 3.12.11 | scvi-tools does not support pandas 3.x |
| [`scAnnotationV3.12.yml`](scAnnotationV3.12.yml) | Stage 3 annotation — `batch_scripts/run_annotation.sh` | 3.12.3 | carries the OrthoFinder toolchain |
| [`scMarkersV3.12.yml`](scMarkersV3.12.yml) | Marker ID — `batch_scripts/run_markerID.sh` | 3.12.13 | hotspot 1.1.3 requires pandas < 3 |

```bash
conda env create -f envs/scMappingV3.12.yml      # and so on for each
```

### `conda env create` alone is not enough

Two of these environments are **not usable straight out of the yml**. Both
failure modes are silent — the run exits 0 and produces wrong numbers — so
neither step is optional:

| After creating | Do this | Or else |
|---|---|---|
| `scMappingV3.12` | `bash tools/piscem-bd/build.sh` | stock piscem truncates BD Rhapsody 27bp barcodes to 9bp |
| `cellBenderENV` | apply `cellbender-checkpoint-weakref.patch` | CellBender crashes ~150 epochs into every run |

Both are detailed below. A conda rebuild silently undoes both, so re-run them
any time you recreate these envs.

Two envs are referenced by **path**, not by name, so after recreating them
update the prefixes in [`configs/pipeline_config.yaml`](../configs/pipeline_config.yaml)
(`cellbender.env`, `doublets.env`) and
[`configs/annotation_config.yaml`](../configs/annotation_config.yaml) (`env`) if
your conda root differs from `/home/WUR/adema026/miniconda3`.

## The mapping env's piscem is not the one that runs

**Creating `scMappingV3.12` is not sufficient to map BD Rhapsody samples
correctly.** The `piscem=0.21.1` pinned in that file is the stock conda build,
and stock piscem **silently truncates** BD Rhapsody Enhanced Bead V2's 27bp
split cell label to 9bp — it keeps only the third of the three
linker-separated blocks, logs nothing, and exits 0. The ~384³ possible cell
labels fold onto ~384, and no downstream stage can detect it afterwards.

Real runs use a **patched** piscem at `tools/bin/piscem`, which
[run_mapper.sh:42](../batch_scripts/run_mapper.sh#L42) puts ahead of the conda env on `$PATH`
— before `simpleaf set-paths`, which caches resolved tool paths. The patches,
root cause and validation are in
[piscem_bd_rhapsody/README.md](../piscem_bd_rhapsody/README.md).

Two things make this easy to get wrong:

- **Both binaries report `piscem 0.21.1`.** `--version` cannot distinguish them.
  The only reliable probe is the geometry itself: stock reports `bc_len=9`,
  patched reports `bc_len=27`.
- **`tools/bin/` is gitignored**, so a fresh clone has no patched binary at all.

So after creating the env, rebuild it:

```bash
bash tools/piscem-bd/build.sh     # ~15 min, needs network access
```

The script vendors its own Rust toolchain under `tools/piscem-bd/rust/` (it
never touches `$HOME` or the conda envs), runs both upstream test suites against
the patches, and self-verifies `bc_len=27` before installing. End-to-end check:
`sbatch tools/piscem-bd/smoke_test.sh`.

`batch_scripts/run_mapper.sh` independently refuses to start if piscem does not report
`bc_len=27` while a BD sample is in the sheet. **Keep that guard** — it is the
only thing standing between a routine conda rebuild and silently corrupted
output.

> simpleaf 0.26.2 links the same broken `seq_geom_parser` 1.1.0, but only
> validates and forwards the geometry to piscem, so patching piscem is
> sufficient. 10x samples are unaffected either way — the patch keeps the
> single-piece barcode path zero-copy.

## CellBender needs a patch after install

CellBender 0.3.2 imports fine, but **crashes during a real run** — its
checkpoint writer pickles the live model object and dies with
`TypeError: cannot pickle 'weakref.ReferenceType' object` roughly 150 epochs in,
after training has already burned the time, then takes the sample down with
`AssertionError: Checkpoint file ckpt.tar.gz does not exist`. This is not fixed
by changing Python or torch versions (CellBender
[#296](https://github.com/broadinstitute/CellBender/issues/296),
[#395](https://github.com/broadinstitute/CellBender/issues/395)).

`conda env create` installs stock CellBender, so **it gives you the bug back**.
The fix ([PR #444](https://github.com/broadinstitute/CellBender/pull/444)) is
vendored here as
[`cellbender-checkpoint-weakref.patch`](cellbender-checkpoint-weakref.patch):

```bash
conda activate cellBenderENV
patch -p0 -d "$CONDA_PREFIX/lib/python3.10/site-packages/cellbender/remove_background" \
      < envs/cellbender-checkpoint-weakref.patch
```

`--forward` is safe to add — an already-patched install is detected and skipped.
The patch applies cleanly to the 3.12 copy too; only the `python3.10` path
segment changes.

> CellBender is **also** installed in `scMappingV3.12`, where it imports but is
> **unpatched**. Harmless as things stand, because the pipeline shells out to
> `cellBenderENV`. Patch it as well before running remove-background from there.

The patch swaps the live-object saves for a `state_dict()` save plus
`PyroOptim`'s own dill-backed `.save()`. That is safe for this pipeline —
single-run `remove-background` reuses the in-memory model and never reconstructs
`'model'` from disk — but it **will break CellBender's mid-training resume
path**, which needs a real `nn.Module`. This pipeline passes no resume flag, so
that path is dead here.

## How these files were produced, and the three corrections

Generated with `conda env export --no-builds`, then corrected. `--no-builds`
keeps exact versions but drops build strings, which are channel- and
platform-specific and are the usual reason a pinned export fails to solve on
another machine. These are `linux-64` envs.

Three corrections were applied, each of which would otherwise be a silent
reproducibility bug:

1. **`prefix:` removed** — an absolute path into one user's home directory.
2. **`defaults` channel dropped** — verified 0 packages in any of the five envs
   came from it (all are conda-forge, bioconda, or pip). Dropping it also avoids
   the Anaconda ToS question for larger organisations.
3. **Pins corrected against what actually imports**, not against what the
   metadata claims:
   - `scIntegrationV1.12` numpy → **2.2.0**. This env's numpy `*.dist-info` is
     stale: `importlib.metadata` reports `1.26.4` while `numpy.__version__` is
     `2.2.0`. `conda env export` reads the metadata, so an uncorrected export
     rebuilds numpy **1.x — a different major version** — under torch 2.8 and
     scvi-tools 1.5. (`shared/io.py::_package_version` detects this at runtime
     and records the imported version for provenance. Repairing the live env is
     still worthwhile.)
   - `torch`/`torchvision` restored to their `+cu126` / `+cu130` local versions,
     with the matching `--extra-index-url` added to the pip section. PyPI's
     default index cannot serve a `+cuXXX` wheel, so without this the pip step
     fails outright or silently installs a differently-built torch.

Because of #3, **do not regenerate these by piping `conda env export` straight
over them.** Re-apply the corrections, or fix the underlying env metadata first.

## GPU

`scMappingV3.12` (cu126), `cellBenderENV` (cu130) and `scIntegrationV1.12`
(cu126) all carry CUDA builds and need a GPU node. `scAnnotationV3.12` and
`scMarkersV3.12` are CPU-only. If your driver does not match the CUDA version,
change the `--extra-index-url` and the `+cuXXX` suffix together — they must
agree.

## Not used by the pipeline

`/home/WUR/adema026/miniconda3/envs` also holds `scIntegrationV3.13`,
`scMappingV3.7`, `scMarkersV3.9`, `scVIenv`, `EZseqEnv`, `LabEnv` and `bulkRNA`.
None are referenced by any pipeline entrypoint or config. `scMappingV3.7`
appears only in legacy notebooks under `projects/`, and `scMarkersV3.9` was
replaced by `scMarkersV3.12` (it existed only because SEMITONES imports on
Python 3.9 alone; SEMITONES has since been dropped). They are not captured here.
