# Source after activating scMappingV3.12, from the logical execution root.
# These are the existing run_mapper.sh settings, shared with interactive runs.
sc_mapping_setup() {
    local root="${SCRNASEQ_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
    export PATH="$root/tools/bin:$PATH"
    export HDF5_USE_FILE_LOCKING=FALSE
    export ALEVIN_FRY_HOME="$root/af_home"
    mkdir -p "$ALEVIN_FRY_HOME" || return
    simpleaf set-paths || return
    ulimit -n 2048 || echo "WARNING: could not raise file handle limit (now $(ulimit -n))"
}
