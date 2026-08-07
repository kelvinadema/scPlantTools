#!/bin/bash
# Rebuild the patched piscem used for BD Rhapsody Enhanced Bead V2 samples.
#
# Stock piscem silently truncates the 27bp split cell label to 9bp (it keeps
# only the third block) and still exits 0. Rationale, evidence and the patches
# themselves are in piscem_bd_rhapsody/ at the repo root.
#
# Run this after any conda env rebuild that replaces piscem, or to move to a
# newer upstream piscem. Produces tools/bin/piscem, which run_mapper.sh puts
# ahead of the conda env on PATH.
#
#   bash tools/piscem-bd/build.sh
#
# Needs network access and ~15 min. The Rust toolchain is installed under
# tools/piscem-bd/rust/ so it never touches $HOME or the conda envs.

set -euo pipefail

# Invoked directly (`bash tools/piscem-bd/build.sh`), not via sbatch, so
# $SLURM_SUBMIT_DIR does not apply here -- resolve relative to this script's
# own location instead ($SCRNASEQ_ROOT still overrides explicitly).
REPO_ROOT="${SCRNASEQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TOOLS="$REPO_ROOT/tools/piscem-bd"
PATCHES="$REPO_ROOT/piscem_bd_rhapsody"
SRC="$TOOLS/src"

# Versions this patch set was written and tested against. seq_geom_parser 1.1.0
# is crates.io-only -- its GitHub repo is stale at 0.3.0 with a different
# grammar, so do not substitute a git checkout for it.
PISCEM_TAG=v0.21.1
PISCEM_RS_VER=0.6.3
SGP_VER=1.1.0

export RUSTUP_HOME="$TOOLS/rust/rustup"
export CARGO_HOME="$TOOLS/rust/cargo"
export PATH="$CARGO_HOME/bin:$PATH"

if ! command -v cargo >/dev/null 2>&1; then
    echo "--- installing Rust toolchain into $TOOLS/rust ---"
    mkdir -p "$TOOLS/rust"
    curl -sSf https://sh.rustup.rs -o "$TOOLS/rust/rustup-init.sh"
    sh "$TOOLS/rust/rustup-init.sh" -y --no-modify-path --profile minimal --default-toolchain stable
fi
cargo --version

echo "--- fetching pristine sources ---"
rm -rf "$SRC"
mkdir -p "$SRC"
cd "$SRC"
curl -sL -o sgp.crate "https://static.crates.io/crates/seq_geom_parser/seq_geom_parser-${SGP_VER}.crate"
curl -sL -o prs.crate "https://static.crates.io/crates/piscem-rs/piscem-rs-${PISCEM_RS_VER}.crate"
tar xzf sgp.crate
tar xzf prs.crate
git clone -q https://github.com/COMBINE-lab/piscem.git piscem
git -C piscem checkout -q "$PISCEM_TAG"

echo "--- applying patches ---"
(cd "seq_geom_parser-${SGP_VER}" && patch -p1 -s < "$PATCHES/01-seq_geom_parser-1.1.0-multipiece-barcode.patch")
(cd "piscem-rs-${PISCEM_RS_VER}" && patch -p1 -s < "$PATCHES/02-piscem-rs-0.6.3-multipiece-barcode.patch")

# Redirect each crate at the patched copy of its dependency.
python3 - "$SRC" "$SGP_VER" "$PISCEM_RS_VER" <<'EOF'
import sys
src, sgp_ver, prs_ver = sys.argv[1], sys.argv[2], sys.argv[3]

p = f"{src}/piscem-rs-{prs_ver}/Cargo.toml"
s = open(p).read()
old = f'[dependencies.seq_geom_parser]\nversion = "{sgp_ver}"'
assert old in s, "seq_geom_parser dependency stanza not found -- upstream layout changed"
open(p, "w").write(s.replace(old, f'{old}\npath = "../seq_geom_parser-{sgp_ver}"'))

p = f"{src}/piscem/Cargo.toml"
s = open(p).read()
old = f'piscem-rs = "{prs_ver}"'
assert old in s, "piscem-rs dependency line not found -- upstream layout changed"
open(p, "w").write(s.replace(old, f'piscem-rs = {{ version = "{prs_ver}", path = "../piscem-rs-{prs_ver}" }}'))
print("path overrides written")
EOF

echo "--- running upstream test suites against the patches ---"
(cd "seq_geom_parser-${SGP_VER}" && cargo test --quiet)
(cd "piscem-rs-${PISCEM_RS_VER}" && cargo test --lib --quiet)

echo "--- building piscem ---"
cd "$SRC/piscem"
cargo build --release

mkdir -p "$REPO_ROOT/tools/bin"
install -m 0755 "$SRC/piscem/target/release/piscem" "$REPO_ROOT/tools/bin/piscem"

echo "--- verifying the split barcode is assembled ---"
PROBE=$(mktemp -d)
trap 'rm -rf "$PROBE"' EXIT
printf '@probe\nACGT\n+\nIIII\n' > "$PROBE/r.fq"
set +e
BC_LEN=$("$REPO_ROOT/tools/bin/piscem" map-sc \
    -i "$PROBE/no_such_index" \
    -g '1{x[0-3]b[9]f[GTGA]b[9]f[GACA]b[9]u[8]}2{r:}' \
    -1 "$PROBE/r.fq" -2 "$PROBE/r.fq" -o "$PROBE/out" 2>&1 \
    | sed -n 's/.*bc_len=\([0-9]*\).*/\1/p' | head -1)
set -e

if [ "$BC_LEN" != "27" ]; then
    echo "FAILED: built piscem reports bc_len=${BC_LEN:-<none>}, expected 27." >&2
    exit 1
fi

echo
echo "OK: $REPO_ROOT/tools/bin/piscem reports bc_len=27 for the BD Rhapsody geometry."
"$REPO_ROOT/tools/bin/piscem" --version
