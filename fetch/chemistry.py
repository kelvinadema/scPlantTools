"""Whitelist vote on the 10x chemistry of every sample in a sheet.

``fetch download`` infers ``tech`` from the R1 length (26 bp -> 10xv2, 28 bp ->
10xv3). That is blind when the archive holds full-length R1 (GSE210881: 150 bp
R1, GEO said v2, the barcodes were v3 at 92-95 %), and GEO's own chemistry
field is unreliable. This reads the first N R1 reads of each sample, counts how
many of ``R1[0:16]`` sit in each 10x whitelist and reports the vote per sample.
With ``--apply`` the winning chemistry is written into the sheet's ``tech``
when it disagrees with what is there.

The v2 and v3 whitelists overlap a little, so a v3 library scores ~1-2 % on v2
and vice versa; the call is the larger fraction, and only when it clears
``--min-frac`` (default 0.5). Below that the sample is reported as
``undetermined`` and the sheet is left alone -- a DNBelab / BD library, or a
read that is not R1 at all, looks exactly like that.

Usage::

    python -m fetch.chemistry --sheet configs/external/GSE224539_samples.yaml \\
        --out data/FQfiles/external/manifests/GSE224539_chemistry.tsv --apply
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys

import yaml

from .util import log, setup_logger

WHITELISTS = {
    "10xv2": "data/Whitelists/10xv2Whitelist.txt",
    "10xv3": "data/Whitelists/10xv3Whitelist.txt",
}
BC_LEN = 16
_TENX = set(WHITELISTS)


def _load(path: str) -> set[str]:
    with open(path) as fh:
        return {line.strip() for line in fh if line.strip()}


def vote(fastq1: list[str], lists: dict[str, set[str]], n_reads: int) -> dict:
    """Fractions of the first ``n_reads`` R1 barcodes (pooled over lanes) in
    each whitelist, plus the modal R1 length."""
    hits = {k: 0 for k in lists}
    total = 0
    lengths: dict[int, int] = {}
    per_lane = max(1, n_reads // max(1, len(fastq1)))
    for path in fastq1:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as fh:
            for i, line in enumerate(fh):
                if i % 4 == 1:
                    seq = line.rstrip("\n")
                    lengths[len(seq)] = lengths.get(len(seq), 0) + 1
                    bc = seq[:BC_LEN]
                    total += 1
                    for k, wl in lists.items():
                        if bc in wl:
                            hits[k] += 1
                    if i // 4 + 1 >= per_lane:
                        break
    fr = {k: (hits[k] / total if total else 0.0) for k in lists}
    r1_len = max(lengths, key=lengths.get) if lengths else 0
    return {"n": total, "r1_len": r1_len, **{f"frac_{k}": v for k, v in fr.items()}}


def call(fr: dict, min_frac: float) -> str:
    best = max(_TENX, key=lambda k: fr[f"frac_{k}"])
    return best if fr[f"frac_{best}"] >= min_frac else "undetermined"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--out", help="TSV of per-sample votes (default: log only)")
    ap.add_argument("--reads", type=int, default=200_000, help="R1 reads per sample (pooled over lanes)")
    ap.add_argument("--min-frac", type=float, default=0.5)
    ap.add_argument("--apply", action="store_true",
                    help="rewrite `tech` in the sheet where the vote disagrees")
    for k, p in WHITELISTS.items():
        ap.add_argument(f"--{k}", default=p, help=f"{k} whitelist (default {p})")
    args = ap.parse_args(argv)
    setup_logger()

    lists = {k: _load(getattr(args, k)) for k in WHITELISTS}
    log().info("[CHEM] whitelists: %s", {k: len(v) for k, v in lists.items()})

    sheet = yaml.safe_load(open(args.sheet)) or {}
    defaults = sheet.get("defaults") or {}
    samples = sheet.get("samples") or {}
    rows = []
    changed = 0
    for name, entry in samples.items():
        current = entry.get("tech", defaults.get("tech"))
        if current not in _TENX:
            log().info("[CHEM] %s: tech %r is not 10x, skipped", name, current)
            continue
        fastq1 = [p for p in (entry.get("fastq1") or []) if os.path.exists(p)]
        if not fastq1:
            log().warning("[CHEM] %s: no readable fastq1, skipped", name)
            continue
        fr = vote(fastq1, lists, args.reads)
        verdict = call(fr, args.min_frac)
        rows.append({"sample": name, "sheet_tech": current, "call": verdict, **fr})
        log().info("[CHEM] %s: R1 %d bp, v2 %.3f, v3 %.3f -> %s (sheet: %s)",
                   name, fr["r1_len"], fr["frac_10xv2"], fr["frac_10xv3"], verdict, current)
        if args.apply and verdict != "undetermined" and verdict != current:
            entry["tech"] = verdict
            changed += 1
            log().warning("[CHEM] %s: tech %s -> %s (whitelist vote)", name, current, verdict)

    if args.out and rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        cols = ["sample", "sheet_tech", "call", "n", "r1_len", "frac_10xv2", "frac_10xv3"]
        with open(args.out, "w") as fh:
            fh.write("\t".join(cols) + "\n")
            for r in rows:
                fh.write("\t".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                                   for c in cols) + "\n")
        log().info("[CHEM] wrote %s", args.out)

    if changed:
        # Rewrite in place, keeping the comment header (yaml drops comments).
        header = []
        with open(args.sheet) as fh:
            for line in fh:
                if line.startswith("#"):
                    header.append(line)
                else:
                    break
        header.append(f"# tech for {changed} sample(s) set by fetch.chemistry (whitelist vote).\n")
        with open(args.sheet, "w") as fh:
            fh.writelines(header)
            yaml.dump(sheet, fh, default_flow_style=False, sort_keys=False, width=200)
        log().info("[CHEM] %s rewritten with %d tech change(s)", args.sheet, changed)

    undetermined = [r["sample"] for r in rows if r["call"] == "undetermined"]
    if undetermined:
        log().warning("[CHEM] undetermined chemistry (sheet untouched): %s", ", ".join(undetermined))
    return 0


if __name__ == "__main__":
    sys.exit(main())
