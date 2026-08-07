# BD Rhapsody Enhanced Bead V2 support in piscem

Investigation and patches for mapping BD Rhapsody Enhanced Bead V2 data with
the geometry

```
1{x[0-3]b[9]f[GTGA]b[9]f[GACA]b[9]u[8]}2{r:}
```

Written against the versions installed in `scMappingV3.12`: **piscem 0.21.1**
and **simpleaf 0.26.2**.

## The finding

piscem does not reject this geometry. It accepts it, reports success, and
**silently extracts only the last of the three barcode blocks.** The 27bp cell
label is truncated to 9bp, with no warning in any log.

Observed directly from the installed binary:

```
$ piscem map-sc -g '1{x[0-3]b[9]f[GTGA]b[9]f[GACA]b[9]u[8]}2{r:}' ...
INFO piscem_rs::cli::map_scrna: Protocol: custom (bc_len=9, umi_len=8)
                                                  ^^^^^^^^^ should be 27
```

The same happens for a plain multi-piece barcode with no linkers at all
(`1{b[9]b[9]b[9]u[8]}2{r:}` also reports `bc_len=9`), which shows the problem is
barcode-piece handling generally, not anchor parsing.

**This is a silent data-corruption bug, not an unsupported-feature error.** A
run completes, exits 0, and produces a count matrix in which every cell sharing
a third block collapses into one barcode: the ~384³ possible cell labels are
folded onto ~384. Nothing downstream can detect this after the fact.

### What actually works, and what does not

The important nuance for anyone reading the piscem docs and concluding the
feature is missing: **almost all of it is implemented and correct.**

| Component | Status |
| --- | --- |
| Parsing `f[SEQ]` anchors | works |
| Locating anchors in the read | works |
| Variable diversity offset `x[0-3]` | works |
| Extracting a block positioned after two anchors | works |
| UMI after the third block | works |
| **Concatenating several `b[...]` into one barcode** | **broken** |

Verified by extracting from a synthetic BD read: the third block is returned
from exactly the right offset for every diversity-offset width 0–3, so anchor
resolution is sound. Only the assembly step is wrong.

## Root cause

Neither bug is in the piscem repository. piscem 0.21.1 is a thin CLI over
`piscem-rs 0.6.3`, which delegates all geometry work to **`seq_geom_parser`
1.1.0** (a crates.io-only release — the GitHub repo is stale at 0.3.0 and uses a
completely different pest grammar; do not read it as current).

That crate models a barcode as a set of *levels* (`s[...]` sample barcode,
`b[...]` cell barcode), and assumes **each level is one contiguous span**. When
several tags map to the same level, both the length metadata and the extracted
sequence overwrite rather than accumulate:

`seq_geom_parser-1.1.0/src/extract.rs`, in `record_target_info`:

```rust
bc_lens[level as usize] = len;          // last piece wins -> 9, not 27
```

and in `assign_target`:

```rust
result.barcodes[*level as usize] = Some(slice);   // last piece wins
```

`compute_barcode_levels` maps every plain `b` tag to the same level, so the
three blocks all collide on level 0 and the third overwrites the first two.

The deeper reason it cannot simply be made to append: `ExtractedSeqs.barcodes`
holds `Option<&'a [u8]>`, a slice borrowed from the read. Three blocks separated
by linkers are not contiguous in the read, so no single borrowed slice can
represent their concatenation. Fixing this requires a representation change, not
a one-line edit.

### simpleaf is affected identically

simpleaf 0.26.2 links the same `seq_geom_parser-1.1.0` (visible in the binary's
embedded build paths) and does **not** bundle `seq_geom_xform`. It validates the
geometry with the same crate and forwards it to piscem, so routing through
simpleaf does not avoid the problem — it reproduces it.

Note this differs from older simpleaf, which converted "complex" geometries by
rewriting the FASTQs into a simple geometry via `seq_geom_xform`. That path is
gone in 0.26.

## The patches

Two patches, applied to the vendored crate sources.

### `01-seq_geom_parser-1.1.0-multipiece-barcode.patch`

The substantive fix.

- Adds `BarcodeSeq<'a>`, holding up to `MAX_BC_PIECES` (4) borrowed slices for
  one barcode level, with `as_contiguous()` (zero-copy, single-piece fast path)
  and `assemble_into(&mut buf)` (concatenates into a caller-owned buffer).
- `assign_target` pushes pieces instead of overwriting; `record_target_info`
  accumulates widths; `build_geom_meta` sums a level's contributions.
- Adds validation for split levels: at most 4 pieces, at most 32 bases total
  (the limit of 2-bit packing into a `u64`), and at most one variable-length
  piece per level — padding encodes captured length in a suffix, which is only
  invertible when a single piece can vary.

Single-piece chemistries keep the existing zero-copy behaviour and allocate
nothing; only a genuinely split barcode touches an assembly buffer.

### `02-piscem-rs-0.6.3-multipiece-barcode.patch`

Mechanical follow-on for the changed type: updates the four barcode consumption
sites in `processors.rs` (with one reusable assembly buffer hoisted out of the
read loop), the `ExtractedSeqs` constructions in `scrna.rs`/`flex.rs`, and
`primary_barcode` in `protocols/mod.rs`.

`custom.rs::total_bc_len` needs no change — it already reads
`meta.bc_lens.last()`, which becomes 27 once widths accumulate.

## Verification

Both patches were compiled and tested on this cluster.

| Check | Result |
| --- | --- |
| `seq_geom_parser` existing tests | 39/39 pass (no regressions) |
| `seq_geom_parser` with new tests | 48/48 pass |
| `piscem-rs` full test suite | 193/193 pass |
| BD geometry declared barcode width | 27 (was 9) |
| BD extraction, diversity offset 0/1/2/3 | correct 27bp label in all four |
| Cells differing only in block 1 or 2 | stay distinct (previously collapsed) |
| 10x v3 barcode still borrows the read | yes, no copy |
| Guards: >32bp, >4 pieces, 2 variable pieces | all rejected with clear errors |

`verify_bd_geometry.rs` is the standalone program behind the last several rows;
the new in-crate regression tests are in the patch itself.

## Status: deployed

The patch route was chosen and is live.

| Thing | Where |
| --- | --- |
| Patched binary | `tools/bin/piscem` (reports `bc_len=27`) |
| Rebuild script | `tools/piscem-bd/build.sh` |
| Build tree + Rust toolchain | `tools/piscem-bd/{src,rust}/` (gitignored) |
| End-to-end smoke test | `tools/piscem-bd/smoke_test.sh` (sbatch) |

`batch_scripts/run_mapper.sh` puts `tools/bin` ahead of the conda env on `PATH` — before
`simpleaf set-paths`, which caches the resolved tool paths — and then **refuses
to start** if `piscem` does not report `bc_len=27` for the BD geometry while a
BD sample is in the sheet. That guard exists because the failure it protects
against is silent: a conda rebuild restores the truncating binary and the run
still exits 0.

### Validated on real reads

50,000 real `BDTip2mm` R1 reads through the patched extractor:

| Metric | Result |
| --- | --- |
| Barcode extracted | 95.6% of reads |
| Extracted barcodes exactly 27bp | 100% (0 wrong length) |
| Block 1 / 2 / 3 in its own whitelist | 88.0% / 91.5% / 89.7% |
| **All three blocks valid simultaneously** | **80.2%** |
| Distinct cell labels seen | 24,728 |

A misassembled barcode would essentially never hit all three per-block
whitelists at once, so 80.2% on uncorrected barcodes is conclusive. The
remaining ~20% is ordinary sequencing error, which is what alevin-fry's
mismatch correction against the permit list is for — and getting that
correction is a large part of why the patch route was chosen over
preprocessing.

Spot-checking the raw reads by hand agrees: read 1 has `GTGA` at offset 10 and
`GACA` at 23 (diversity offset 1), read 2 at 11 and 24 (offset 2), each
followed by the 8bp UMI and then polyT.

### Validated end to end through alevin-fry

`tools/piscem-bd/smoke_test.sh`, 1,000,000 real `BDTip2mm` read pairs against
the `AFdinduced` index (SLURM job 41427104):

| Metric | Result |
| --- | --- |
| Wall clock / peak RSS | 1m51s / **3.4 GB** |
| Reads mapped | 916,718 of 1,000,000 (91.67%) |
| Quantified cells | 20,061 |
| Genes | 163,971 |
| Cell barcode width in `quants_mat_rows.txt` | 27bp for **all** 20,061 rows |
| Spot-checked barcodes present in `BD_CLSmerged.txt` | 5 of 5 |
| USA mode | true (spliced/unspliced/ambiguous layers, as the pipeline expects) |

The memory figure is the useful surprise: the 56,623,104-barcode unfiltered
permit list costs about 3.4 GB peak, not the tens of GB feared, so it is not a
constraint on the production run.

## Applying and building

`tools/piscem-bd/build.sh` does all of the below in one step and verifies the
result; prefer it. The manual sequence is kept for reference and for building
somewhere other than this repo.

```bash
export CARGO_HOME=/some/scratch/cargo RUSTUP_HOME=/some/scratch/rustup
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
export PATH="$CARGO_HOME/bin:$PATH"

# fetch the exact sources
curl -sL -o sgp.crate https://static.crates.io/crates/seq_geom_parser/seq_geom_parser-1.1.0.crate
curl -sL -o prs.crate https://static.crates.io/crates/piscem-rs/piscem-rs-0.6.3.crate
tar xzf sgp.crate && tar xzf prs.crate

(cd seq_geom_parser-1.1.0 && patch -p1 < ../01-seq_geom_parser-1.1.0-multipiece-barcode.patch)
(cd piscem-rs-0.6.3      && patch -p1 < ../02-piscem-rs-0.6.3-multipiece-barcode.patch)

# point piscem-rs at the patched parser, and piscem at the patched piscem-rs
#   piscem-rs-0.6.3/Cargo.toml : [dependencies.seq_geom_parser] add
#       path = "../seq_geom_parser-1.1.0"
#   piscem/Cargo.toml          : [dependencies] piscem-rs -> add
#       path = "../piscem-rs-0.6.3"

git clone https://github.com/COMBINE-lab/piscem.git
(cd piscem && cargo build --release)
```

The resulting `piscem/target/release/piscem` replaces the conda one. Keep the
conda binary — see the caveat about it being overwritten below.

Confirm the fix took with a one-line smoke test; it must now print `bc_len=27`:

```bash
./piscem map-sc -i /nonexistent -g '1{x[0-3]b[9]f[GTGA]b[9]f[GACA]b[9]u[8]}2{r:}' \
  -1 r1.fq -2 r2.fq -o /tmp/out 2>&1 | grep Protocol
```

## Before committing to this route — read this

**These patches make you the maintainer of a fork of two upstream crates.**
Consider whether you want that before the biology deadline. Specific concerns:

1. **Upstream first.** This is a genuine upstream bug and a serious one (silent
   miscounting, not a crash). Filing it against
   [COMBINE-lab/piscem](https://github.com/COMBINE-lab/piscem) with the
   `bc_len=9` reproduction is worth doing regardless of which route you take —
   the maintainers may prefer a different API shape, and a merged fix costs you
   nothing to maintain. The reproduction in this directory is written to be
   attachable to an issue as-is.

2. **A conda env rebuild silently reverts you.** Same failure mode as the
   CellBender `checkpoint.py` patch already documented in `CLAUDE.md`: nothing
   in this repo pins it, so reinstalling `piscem` from conda restores the
   truncating binary and the run still exits 0. If you deploy the fork, install
   it somewhere the env cannot overwrite and pin the path explicitly.

3. **The lower-risk alternative: preprocess the FASTQs.** Rewrite R1 into a
   simple geometry — strip the diversity offset, drop the `GTGA`/`GACA` linkers,
   concatenate the three blocks + UMI — and hand piscem
   `1{b[27]u[8]}2{r:}`, which it handles correctly today with no patched
   binaries anywhere. This is exactly what `seq_geom_xform` used to do. It costs
   one pass over the FASTQs and some disk, and it keeps you on stock, conda-
   managed tools. For a single chemistry on a deadline this is probably the
   right call; the patch route is right if you want it fixed properly and
   upstreamed.

4. **The barcode whitelist was already handled — do not "fix" it.** This was
   initially misread here as a broken config, so it is worth stating plainly.
   `technology_config.yaml`'s `bd_rhapsody.alevin.extra_mapping_args` keeps the
   three per-block whitelists space-joined inside **one** list entry, which
   looks wrong but is deliberate: BD samples go through
   `mapping.py::_run_alevin_bd` → `BDDataPrepper.PrepBDMapping`
   (`mappingutils.py`), which splits that value, takes the cartesian product
   into `data/Whitelists/BD_CLSmerged.txt` (384³ = 56,623,104 complete 27bp
   barcodes, cached if present) and substitutes that single path before
   simpleaf is invoked. `PrepBDMapping` reads `extra_args[idx + 1]`, so
   splitting the paths into separate YAML entries would break it.

   Consequence worth noting: with the whitelist path always having been fine,
   **piscem's truncation was the only thing standing between this pipeline and
   a completed-but-silently-wrong BD run.**
