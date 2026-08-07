#!/bin/bash
#SBATCH --job-name=bdsmoke
#SBATCH --time=02:00:00
#SBATCH --mem=120g
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/bdsmoke-%j.out
#SBATCH --error=logs/bdsmoke-%j.err

# End-to-end check of the BD Rhapsody path on a subsample of real reads:
# simpleaf -> patched piscem (27bp split barcode) -> alevin-fry permit list
# against the 56,623,104-barcode merged whitelist.
#
# Cheap proxy for "will the 24h production run actually work", and the only
# place the whitelist's memory footprint gets exercised. Run after any rebuild
# of tools/bin/piscem.
#
#   sbatch tools/piscem-bd/smoke_test.sh [N_READS]

set -euo pipefail

REPO_ROOT="${SCRNASEQ_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
mkdir -p logs

N_READS="${1:-1000000}"
SAMPLE=BDTip2mm
FQDIR="$REPO_ROOT/data/FQfiles/Medicago/Protoplast/Sophia/$SAMPLE"
INDEX="$REPO_ROOT/indices/AFdinduced/index"
OUT="$REPO_ROOT/results/smoketest/$SAMPLE"

source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate scMappingV3.12

# Patched piscem must win over the conda env's -- before set-paths caches it.
export PATH="$REPO_ROOT/tools/bin:$PATH"
export HDF5_USE_FILE_LOCKING=FALSE
export ALEVIN_FRY_HOME="$REPO_ROOT/af_home"
export TMPDIR="${TMPDIR:-/tmp/${SLURM_JOB_ID:-$$}}"
mkdir -p "$ALEVIN_FRY_HOME" "$TMPDIR"
simpleaf set-paths
ulimit -n 2048 || true

echo "--- versions ---"
echo "piscem resolved to: $(command -v piscem)"
piscem --version
simpleaf --version
alevin-fry --version
echo "----------------"

GEO=$(python - <<'EOF'
import yaml
cfg = yaml.safe_load(open('configs/pipeline_config.yaml'))
tech = yaml.safe_load(open(cfg['configs']['technologies']))['technologies']
print(tech['bd_rhapsody']['alevin']['chemistry'])
EOF
)

# Resolve the permit list exactly the way the pipeline does, so this exercises
# BDDataPrepper's cartesian-product merge rather than a hand-written path.
PL=$(python - <<'EOF'
import yaml
from mapping.mappingutils import BDDataPrepper
cfg = yaml.safe_load(open('configs/pipeline_config.yaml'))
tech = yaml.safe_load(open(cfg['configs']['technologies']))['technologies']
args = BDDataPrepper(dict(tech['bd_rhapsody']['alevin'])).PrepBDMapping()
print(args[args.index('--unfiltered-pl') + 1])
EOF
)
echo "geometry    : $GEO"
echo "permit list : $PL"

# Assert the patched binary is live before burning any time on mapping.
printf '@probe\nACGT\n+\nIIII\n' > "$TMPDIR/_probe.fq"
set +e
BC_LEN=$(piscem map-sc -i "$TMPDIR/_no_index" -g "$GEO" \
    -1 "$TMPDIR/_probe.fq" -2 "$TMPDIR/_probe.fq" -o "$TMPDIR/_probe_out" 2>&1 \
    | sed -n 's/.*bc_len=\([0-9]*\).*/\1/p' | head -1)
set -e
if [ "$BC_LEN" != "27" ]; then
    echo "FATAL: piscem reports bc_len=${BC_LEN:-<none>}, expected 27 -- unpatched build." >&2
    exit 1
fi
echo "preflight   : bc_len=$BC_LEN OK"

echo "--- subsampling $N_READS reads ---"
rm -rf "$OUT"
mkdir -p "$OUT"
NLINES=$((N_READS * 4))
# head closing the pipe makes zcat exit non-zero; that is expected here.
set +o pipefail
zcat "$FQDIR/Mt_2mm_tip_1.fq.gz" | head -n "$NLINES" | gzip > "$OUT/sub_R1.fq.gz"
zcat "$FQDIR/Mt_2mm_tip_2.fq.gz" | head -n "$NLINES" | gzip > "$OUT/sub_R2.fq.gz"
set -o pipefail
echo "subsample written"

echo "--- simpleaf quant ---"
/usr/bin/time -v simpleaf quant \
    --reads1 "$OUT/sub_R1.fq.gz" \
    --reads2 "$OUT/sub_R2.fq.gz" \
    --chemistry "$GEO" \
    --index "$INDEX" \
    --output "$OUT/quant" \
    --min-reads 10 \
    --expected-ori fw \
    --resolution cr-like \
    --threads "${SLURM_CPUS_PER_TASK:-16}" \
    --unfiltered-pl "$PL" 2>&1 | tail -40

echo "--- results ---"
python - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])

gpl = out / "quant" / "af_quant" / "generate_permit_list.json"
if gpl.is_file():
    print("generate_permit_list.json:", json.dumps(json.loads(gpl.read_text()), indent=2))

mi = out / "quant" / "af_map" / "map_info.json"
if mi.is_file():
    d = json.loads(mi.read_text())
    for k in ("num_reads", "num_mapped", "percent_mapped", "num_bootstraps"):
        if k in d:
            print(f"{k}: {d[k]}")

qm = out / "quant" / "af_quant" / "alevin" / "quants_mat_rows.txt"
if not qm.is_file():
    raise SystemExit(f"FAILED: no cell barcode matrix at {qm}")

bcs = qm.read_text().split()
bad = [b for b in bcs if len(b) != 27]
print(f"cells in matrix : {len(bcs)}")
print(f"barcodes != 27bp: {len(bad)}")
print("first barcodes  :", bcs[:3])
if bad or not bcs:
    raise SystemExit("FAILED: barcode width is wrong -- piscem truncated the cell label")
print("OK: every cell barcode is 27bp")
EOF

echo "smoke test finished"
