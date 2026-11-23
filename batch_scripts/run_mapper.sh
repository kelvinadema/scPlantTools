#!/bin/bash
#SBATCH --job-name=scmap
#SBATCH --comment=3184300300
#SBATCH --time=48:00:00
#SBATCH --mem=512g
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/mapping-%j.out
#SBATCH --error=logs/mapping-%j.err

# Fail the job on an unset variable or a failed command. Without this a failed
# `conda activate` would sail on and run the pipeline against the wrong python.
set -euo pipefail

# ---------------------------------------------------------------------------
# Repo root. `python -m mapping.main` only resolves from here, and the
# pipeline's log_dir ('logs/') is relative to the working directory.
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
conda activate scMappingV3.12

# Shared with interactive launches; preserve the logical execution root.
source "$REPO_ROOT/batch_scripts/lib/mapping_setup.sh"
sc_mapping_setup

# Node-local scratch for piscem index builds, which are unreliable on networked
# filesystems. SLURM normally sets TMPDIR per job; fall back to a job-scoped dir
# if it does not. IndexBuilder builds here and copies the finished index to the
# configured idx_path, so nothing partial ever lands on Lustre.
export TMPDIR="${TMPDIR:-/tmp/$SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
echo "Using TMPDIR=$TMPDIR"

# Match tool thread counts to the SLURM allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# CellBender lives in its own conda env (it pins torch and will not install
# alongside python 3.12 -- CellBender #371/#386/#395). The pipeline invokes it as
# a subprocess from the prefix set in pipeline_config.yaml's `cellbender.env`;
# export it here as a fallback when the YAML env is unset.
# export CELLBENDER_ENV="$REPO_ROOT/envs/cellbender310"

# ---------------------------------------------------------------------------
# Provenance. These land in .uns['pipeline']['tool_versions'] too, but printing
# them puts them in the job log next to any failure.
# ---------------------------------------------------------------------------
echo "--- versions ---"
python --version
simpleaf --version
piscem --version
alevin-fry --version
# CellBender is NOT in this env by design; ask the env the pipeline will actually use.
CB_ENV="$(python - "$@" <<'EOF'
import os, sys
from shared.mapping_setup import launch_config
cfg, _ = launch_config(sys.argv[1:])
print((cfg.get('cellbender') or {}).get('env') or os.environ.get('CELLBENDER_ENV', ''))
EOF
)"
if [ -n "$CB_ENV" ] && [ -x "$CB_ENV/bin/cellbender" ]; then
    "$CB_ENV/bin/python" --version
    "$CB_ENV/bin/cellbender" --version
else
    echo "WARNING: cellbender env not found at '${CB_ENV:-<unset>}' -- ambient removal will fail"
fi
# Doublet detection (scvi-tools SOLO) runs from its own env for the same reason
# CellBender does -- scvi-tools does not support the pandas 3.x this stage is
# on. Echoed here so the run log records which scvi-tools produced the calls.
# The preflight in mapping.main errors properly if a filter profile
# enables solo and this env is missing or has no scvi-tools; this is just the
# version line.
DBL_ENV="$(python - "$@" <<'EOF'
import os, sys
from shared.mapping_setup import launch_config
cfg, _ = launch_config(sys.argv[1:])
print((cfg.get('doublets') or {}).get('env') or os.environ.get('DOUBLET_ENV', ''))
EOF
)"
if [ -n "$DBL_ENV" ] && [ -x "$DBL_ENV/bin/python" ]; then
    "$DBL_ENV/bin/python" -c "import scvi; print('scvi-tools', scvi.__version__)" 2>/dev/null || \
        echo "WARNING: no importable scvi-tools in '$DBL_ENV' -- doublet detection will fail"
else
    echo "NOTE: no doublets env at '${DBL_ENV:-<unset>}' -- fine if every profile has solo.mode: off"
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || \
    echo "WARNING: no GPU visible -- CellBender runs with --cuda by default and will fail (as will doublet detection)"
echo "----------------"

# ---------------------------------------------------------------------------
# BD Rhapsody barcode preflight.
#
# The Enhanced Bead V2 cell label is one 27bp barcode carried in three 9bp
# blocks split by the GTGA/GACA linkers. Stock piscem accepts that geometry,
# silently keeps only the THIRD block, and still exits 0 -- so a green run
# against an unpatched binary yields a plausible-looking matrix in which ~384^3
# cell labels have collapsed onto ~384. Nothing downstream can detect this, and
# a conda env rebuild reinstates the unpatched binary without warning.
#
# So assert the patch is actually live, but only when the sample sheet has a BD
# sample -- a 10x-only run does not care.
# ---------------------------------------------------------------------------
# Resolve the same --config/--cohort (including sample defaults) as mapping.main.
# Probe the binary in simpleaf's cache after sc_mapping_setup refreshes it.
python -m shared.mapping_setup "$@"

# ---------------------------------------------------------------------------
# Run. Exits 0 only if every sample succeeded, 1 if any failed.
# Trailing arguments pass straight through to mapping.main, so
#   sbatch batch_scripts/run_mapper.sh --cohort <name>
# maps that cohort; with none, the run is unchanged (default cohort).
# ---------------------------------------------------------------------------
set +e
srun python -m mapping.main \
    --config "$REPO_ROOT/configs/pipeline_config.yaml" \
    --build-indexes "$@"
status=$?
set -e

echo "mapping pipeline exited with status $status"
exit "$status"
