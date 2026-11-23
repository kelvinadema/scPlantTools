# Preparing a release

The pipeline is distributed as a source checkout with one conda environment per stage. `pyproject.toml` declares the packages and test settings but does not install a working pipeline. A release must contain `.root`, the `sc` command, the stage packages, `shared/`, the configuration schemas and the bundled ontology and patch files.

## Contents

The repository contains the README and guides, the design notes, the setup and launch scripts that are maintained, the tests, the example configuration and the current source. Private notes, study-specific batch jobs, incomplete download sheets, data, results, indices, caches and compiled binaries are excluded by `.gitignore`. The study configurations under `configs/` are included as a record and are kept apart from the example.

Run Git commands in the source checkout itself, not in a working directory that links to it. Review `git status --short -uall` and the diff before committing. None of the release tools stage, commit, push or run analyses.

## Checks

```bash
python tools/check_release.py
python -m sc test
python -m sc test site
```

`check_release.py` checks that the required files are present, that relative Markdown links resolve, that the schemas match `shared/config_spec.py`, that Python and shell files parse, and that no source file is a symbolic link. The GitHub Actions workflow runs it together with the portable tests. A release should run the end-to-end suite with the mapping tools installed, not skip it.

Before publishing, check an exported copy of the tree rather than only the working directory, and import the entry points in their own environments. kb-python and STARsolo must remain marked as not implemented. The licence and the third-party notices listed in [third-party.md](third-party.md) still need to be settled.

## Keeping existing results reproducible

Before changing the source, archive the current source (including untracked files), the Git status and diff, the exact package lists of the environments, the versions that are actually imported where these differ from the package metadata, and the paths and hashes of the binaries and patches. For the September 2026 release preparation this archive is kept locally in `_snapshots/`, outside the repository, because it contains private notes and paths.

Do not reformat the study YAML files: the configuration hash recorded in each output includes the raw YAML bytes, so changing comments changes the hash. Do not rewrite existing H5AD files to update their provenance; record release information separately.

A change to how the pipeline is run should leave the command, the resolved configuration, the selected binaries and the environments unchanged. A change to analysis code should be checked by running old and new code on the same frozen inputs into separate output directories and comparing cell and gene order, counts and layers, QC metrics and masks, doublet calls and manual decisions. GPU training is not bit-for-bit reproducible even with fixed seeds, so avoid retraining models when only reorganising code.

Moving result directories requires tracing what reads them, including the notebooks repository and `batch_scripts/lib/common.sh`, and is a separate task.
