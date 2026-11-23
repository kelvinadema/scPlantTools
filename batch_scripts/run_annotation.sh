#!/bin/bash
#SBATCH --job-name=scannot
#SBATCH --comment=3184300300
#SBATCH --time=4-00:00:00
#SBATCH --mem=256g
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/annotation-%j.out
#SBATCH --error=logs/annotation-%j.err

# ---------------------------------------------------------------------------
# Annotation runs on `main`, not `gpu`. OrthoFinder and DIAMOND are CPU-only --
# nothing in this stage touches a GPU -- and `main` is both the larger pool
# (109 nodes vs 7) and has no time limit, where `gpu` caps at 3 days.
#
# That cap mattered: this script used to request `--partition=gpu` with
# `--time=6000`, and a bare integer is *minutes* in SLURM, so it asked for
# 4d4h on a partition that allows 3d. The job was rejected at submission.
#
# --cpus-per-task must match `orthofinder.threads` in annotation_config.yaml
# (DIAMOND's search thread count, and the dominant cost of a 10-species run).
# ---------------------------------------------------------------------------

# Fail the job on an unset variable or a failed command. Without this a failed
# `conda activate` would sail on and run the pipeline against the wrong python.
set -euo pipefail

# ---------------------------------------------------------------------------
# Repo root. Two things depend on this being the working directory:
#   - `python -m annotation.main` only resolves from here
#   - find_root_dir() (shared/utils.py) walks *cwd's* parents for the `.root`
#     marker. Submitted from anywhere else, the run dies with "Project root
#     marker not found." (Output location itself is config-driven via
#     `output_dir:` in annotation_config.yaml, not cwd-relative.)
#
# Resolved without a hardcoded path: $SCRNASEQ_ROOT overrides explicitly,
# else $SLURM_SUBMIT_DIR (the directory `sbatch` was invoked from -- the
# normal case, since this is submitted from the repo root), else $PWD for a
# manual/interactive run.
# ---------------------------------------------------------------------------
REPO_ROOT="${SCRNASEQ_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
mkdir -p logs

# ---------------------------------------------------------------------------
# Environment. $CONDA_ROOT overrides the conda install location if it is not
# under this user's $HOME.
# ---------------------------------------------------------------------------
source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate scAnnotationV3.12

# HDF5 file locking is unreliable on Lustre. Only matters when marker
# generation is enabled (markers.generate.samples) and reads an h5ad.
export HDF5_USE_FILE_LOCKING=FALSE

# Match tool thread counts to the SLURM allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

# ---------------------------------------------------------------------------
# Provenance. OrthoFinder results are cached by a key derived from the prepared
# proteomes, so recording which binaries produced a run is the only way to tell
# two cache entries apart after the fact.
# ---------------------------------------------------------------------------
echo "--- versions ---"
python --version
orthofinder -h 2>&1 | sed -n 's/^OrthoFinder version \(.*\) Copyright.*/OrthoFinder \1/p'
diamond --version
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-<unset>}"
echo "----------------"

# ---------------------------------------------------------------------------
# Preflight: prepare primary-transcript proteomes, index each species'
# GTF/GFF3, and check marker coverage. Minutes, and it fails on a bad proteome
# path, a broken gene_id rule or a marker file that no longer matches its
# proteome release -- all of which would otherwise surface only after
# OrthoFinder had already burned hours.
#
# main.py runs this same validation itself, so this is a fast fail, not a
# separate step: everything it computes is cached and reused below.
# ---------------------------------------------------------------------------
# It must stay --validate-only. Setting it to --orthofinder-only makes the
# "fast fail" step run the full OrthoFinder, and then a submission with no
# arguments falls through to a *full* pipeline run below -- which is how job
# 41543025 burned 32 minutes on a successful OrthoFinder and still exited 1,
# on a missing target marker file.
python -m annotation.main \
    --acfg "$REPO_ROOT/configs/annotation_config.yaml" \
    --validate-only

# ---------------------------------------------------------------------------
# Run. Any argument given to this script is passed through, so:
#
#   sbatch run_annotation.sh --orthofinder-only
#       Run OrthoFinder, parse the orthogroups, stop. This is what you want
#       when results/markers/<method>/ has no target marker files yet -- the
#       annotation loop needs one per sample in target_species_files.samples
#       and raises if it is missing, which would otherwise happen only after
#       OrthoFinder had already finished.
#
#   sbatch run_annotation.sh --force
#       Ignore cached OrthoFinder results for this proteome set and rerun.
#
# The preflight above deliberately stays --validate-only regardless.
# ---------------------------------------------------------------------------
set +e
srun python -m annotation.main \
    --acfg "$REPO_ROOT/configs/annotation_config.yaml" "$@"
status=$?
set -e

echo "annotation pipeline exited with status $status"
exit $status
