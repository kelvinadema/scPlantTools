#!/bin/bash
#SBATCH --job-name=screfilter
#SBATCH --comment=3184300300
#SBATCH --time=12:00:00
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/refilter-%j.out
#SBATCH --error=logs/refilter-%j.err

# Fail the job on an unset variable or a failed command. Without this a failed
# `conda activate` would sail on and run the pipeline against the wrong python.
set -euo pipefail

# ---------------------------------------------------------------------------
# Repo root. `python -m mapping.main` only resolves from here, and the
# pipeline's log_dir ('logs/') is relative to the working directory.
# ---------------------------------------------------------------------------
REPO_ROOT="${SCRNASEQ_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
mkdir -p logs

# ---------------------------------------------------------------------------
# Environment. $CONDA_ROOT overrides the conda install location if it is not
# under this user's $HOME.
# ---------------------------------------------------------------------------
source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate scMappingV3.12

# ---------------------------------------------------------------------------
# Run. Exits 0 only if every sample succeeded, 1 if any failed.
# ---------------------------------------------------------------------------
set +e
srun python -m mapping.refilter \
    --config "$REPO_ROOT/configs/pipeline_config.yaml" \
    --sample 'SMu007' 'SMu008' 'SMu009' 'SMu010' 'SMu011' 'SMu012' \
    --allow-parent-mismatch
status=$?
set -e

echo "refiltering exited with status $status"
exit "$status"
