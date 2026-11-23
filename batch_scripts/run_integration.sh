#!/bin/bash
#SBATCH --job-name=scint
#SBATCH --comment=3184300300
#SBATCH --time=48:00:00
#SBATCH --mem 512g
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/integration-%j.out
#SBATCH --error=logs/integration-%j.err

# Fail the job on an unset variable or a failed command. Without this a failed
# `conda activate` would sail on and run the pipeline against the wrong python.
set -euo pipefail

# ---------------------------------------------------------------------------
# Repo root. `python -m integration.main` only resolves from here,
# find_root_dir() walks cwd for `.root`, and both the pipeline's log_dir and
# the `logs/slurm-%j.{out,err}` paths above are relative to it.
# ---------------------------------------------------------------------------
REPO_ROOT="${SCRNASEQ_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
mkdir -p logs

# Disable HDF5 file locking (unreliable on Lustre).
export HDF5_USE_FILE_LOCKING=FALSE

# Node-local scratch. Ray Tune (scvi autotune) writes its experiment state and
# per-trial logs to storage_path; on Lustre that sync is slow enough to trip
# ray's 30s warning. IntegrationManager points it here instead. SLURM normally
# sets TMPDIR per job; fall back to a job-scoped dir if it does not.
export TMPDIR="${TMPDIR:-/tmp/$SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
echo "Using TMPDIR=$TMPDIR"

# Activate conda environment. $CONDA_ROOT overrides the conda install
# location if it is not under this user's $HOME.
source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate scIntegrationV1.12

# Run integration pipeline
#
# The config defines the model, the CLI chooses what to run. Model parameters
# are set with `--set key=value` when an entry is first created, and live in
# configs/integration_config.yaml from then on -- so the line below creates
# '2_tip' on its first run and replays it on every later one. Once it exists,
#   --target 2_tip --plot
# is the equivalent invocation, and is what you want for a rerun. A --set that
# disagrees with the existing entry is refused, not silently ignored.
#
# --run-solo is gone: doublet detection now runs per sample inside the mapping
# stage (mapping/doublets.py). To integrate hand-reviewed filtering
# instead of the automatic output, set `artifact_key: filtered_manual_h5ad` on
# the target (or file-wide) rather than passing a flag -- which artifact a model
# was trained on is part of what the model is.
srun python -m integration.main --icfg configs/integration_config.yaml --scfg configs/sample_metadata.yaml --target MtSZ --plot