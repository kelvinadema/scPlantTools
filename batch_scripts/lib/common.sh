# Shared boilerplate for batch_scripts/*.sh. Source it, don't execute it:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
#
# Everything here was copy-pasted across 38 scripts before 2026-08-06, with the
# usual consequence: the copies drifted. Some resolved the repo root from
# $SLURM_SUBMIT_DIR, some hardcoded /lustre/..., some did neither and relied on
# sbatch's cwd. Some set BLAS thread limits, most did not.
#
# Deliberately NOT a framework. Each function does one thing a script would
# otherwise open-code, and a script is free to skip any of them.

# ---------------------------------------------------------------------------
# Repo locations
# ---------------------------------------------------------------------------

# The pipeline checkout, as a LOGICAL path. $SCRNASEQ_ROOT overrides, else the
# directory sbatch was invoked from, else $PWD.
#
# DO NOT "improve" this to resolve from ${BASH_SOURCE[0]}. This repo is a
# symlink farm: jobs run from /lustre/nobackup/.../scRNAseq, where every
# code directory is a symlink into /lustre/backup/.../scRNAseq, and only the
# nobackup side has data/, results/, indices/ and logs/. Resolving through
# BASH_SOURCE with `cd ... && pwd` returns the PHYSICAL path, i.e. the backup
# tier, and a job that cds there finds no data at all. `pwd -L` on the submit
# directory keeps the logical path the user meant.
SC_REPO_ROOT="${SCRNASEQ_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

# The chapter-analysis repo. SEPARATE REPOSITORY as of 2026-08-06: it used to be
# scRNAseq/notebooks/ and is now a top-level sibling on the nobackup tier,
# following the same split labwork/ went through. The ch5/ch3 scripts `cd` into
# it and run its code, while its own configs point back here through
# `paths.scrnaseq_root` -- so such a job spans two repos and two commits, and
# neither records the other's SHA. $SCRNASEQ_NOTEBOOKS overrides.
#
# Hardcoded to the nobackup tier rather than derived from SC_REPO_ROOT: the
# notebooks tree carries an 18 GB results/ directory and exists only there.
SC_NOTEBOOKS="${SCRNASEQ_NOTEBOOKS:-/lustre/nobackup/WUR/PDB/adema026/notebooks}"

# ---------------------------------------------------------------------------

sc_init() {
    # cd to the repo root and make sure logs/ exists. `python -m <package>` and
    # every relative config path in this repo resolve from here, and
    # find_root_dir()'s `.root` walk starts here.
    cd "${SC_REPO_ROOT}"
    mkdir -p logs
}

sc_activate() {
    # Activate a conda env, failing loudly if it is not there.
    #
    # `conda activate` on a missing env prints a warning and returns non-zero
    # but does NOT stop an un-`set -e`'d script, which is how a job ends up
    # running against the base interpreter and failing hundreds of lines later
    # on an import. $CONDA_ROOT overrides the conda install location.
    local env_name="$1"
    local conda_root="${CONDA_ROOT:-$HOME/miniconda3}"
    if [[ ! -f "${conda_root}/etc/profile.d/conda.sh" ]]; then
        echo "conda not found at ${conda_root} -- set \$CONDA_ROOT" >&2
        return 1
    fi
    # shellcheck disable=SC1091
    source "${conda_root}/etc/profile.d/conda.sh"
    conda activate "${env_name}" || {
        echo "could not activate conda env '${env_name}'" >&2
        return 1
    }
    if [[ "${CONDA_DEFAULT_ENV:-}" != "${env_name}" ]]; then
        echo "conda env is '${CONDA_DEFAULT_ENV:-<none>}', expected '${env_name}'" >&2
        return 1
    fi
}

sc_threads() {
    # Pin BLAS/OpenMP to the SLURM allocation.
    #
    # Without this every BLAS call assumes it owns the node: scanpy's
    # neighbours, pygpcca's dense fallback and scvi's dataloader workers each
    # spawn $(nproc) threads on a 16-cpu allocation and thrash. Optional
    # argument overrides the count.
    local n="${1:-${SLURM_CPUS_PER_TASK:-16}}"
    export OMP_NUM_THREADS="${n}"
    export OPENBLAS_NUM_THREADS="${n}"
    export MKL_NUM_THREADS="${n}"
    export NUMEXPR_NUM_THREADS="${n}"
}

sc_banner() {
    # Provenance header. Any extra arguments are run as version commands.
    #
    # Worth the three lines: OrthoFinder results are cached by a key derived
    # from the prepared proteomes, so the binary versions are the only way to
    # tell two cache entries apart after the fact.
    echo "=== $(basename "${0}") on $(hostname) at $(date -Is) ==="
    echo "    repo:      ${SC_REPO_ROOT}"
    echo "    job:       ${SLURM_JOB_ID:-<interactive>}${SLURM_ARRAY_TASK_ID:+ task ${SLURM_ARRAY_TASK_ID}}"
    echo "    env:       ${CONDA_DEFAULT_ENV:-<none>} ($(python --version 2>&1))"
    echo "    cpus:      ${SLURM_CPUS_PER_TASK:-<unset>}"
    local cmd
    for cmd in "$@"; do
        echo "    $(eval "${cmd}" 2>&1 | head -1)"
    done
    echo "==="
}

sc_run() {
    # Run the pipeline command, report its exit status, and propagate it.
    #
    # The `set +e` / capture / `set -e` dance every script open-coded. Without
    # it a non-zero exit under `set -euo pipefail` kills the script before the
    # status can be reported, and SLURM records a bare failure with no line
    # saying which stage produced it.
    set +e
    "$@"
    local status=$?
    set -e
    echo "=== $(basename "${0}") exited with status ${status} at $(date -Is) ==="
    return ${status}
}

sc_task_from() {
    # Map $SLURM_ARRAY_TASK_ID to a line of a data file, 0-indexed, ignoring
    # blank lines and #-comments.
    #
    # WHY A FILE AND NOT A BASH ARRAY. The array index -> target mapping used to
    # be positional in a bash array inside the script. Editing that list
    # reorders it, and the queued tasks of an ALREADY-SUBMITTED array resolve
    # against the new positions -- so a running sweep silently continues against
    # the wrong targets. A file keeps the mapping reviewable, diffable, and
    # separate from the script that consumes it; it is still positional, but the
    # position is now data you can read rather than a side effect of editing code.
    # Trailing inline comments (`MtSZ_DRVI_full   # 265,287`) are stripped, so a
    # line can carry the cell count that justifies its position in the class.
    local file="$1"
    local idx="${2:-${SLURM_ARRAY_TASK_ID:-0}}"
    local line
    line=$(sc__entries "${file}" | sed -n "$((idx + 1))p")
    if [[ -z "${line}" ]]; then
        echo "array index ${idx} has no entry in ${file} (it has $(sc_task_count "${file}"), indices 0-$(( $(sc_task_count "${file}") - 1 )))" >&2
        return 1
    fi
    echo "${line}"
}

sc_task_count() {
    # Number of usable entries in such a file -- for `--array=0-$(( n - 1 ))`.
    sc__entries "$1" | wc -l
}

sc__entries() {
    # Non-comment, non-blank lines with inline comments and surrounding
    # whitespace removed. Internal.
    sed -e 's/#.*$//' -e 's/[[:space:]]*$//' -e 's/^[[:space:]]*//' "$1" \
        | grep -vE '^$'
}
