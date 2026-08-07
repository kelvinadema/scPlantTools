"""External-data fetch stage.

Two subcommands, run from the repo root like the other stages:

    # 1. build a curatable run table for a public accession
    python -m fetch.main metadata GSE123013

    # ... edit the TSV: set `download` to no for runs you don't want ...

    # 2. download the kept runs and emit a ready-to-map sample sheet
    python -m fetch.main download -m data/FQfiles/external/manifests/GSE123013_runtable.tsv

    # URL mode -- archives with no API (CNGB/CNSA), or any direct FASTQ URL.
    # Pass CNSA *run* directory URLs (.../CNR....../); --species is required.
    python -m fetch.main metadata --species Gossypium --dataset CNP0007662 \
        --url http://ftp.cngb.org/pub/CNSA/data7/CNP0007662/CNS.../CNX.../CNR....../
    python -m fetch.main download -m data/FQfiles/external/manifests/CNP0007662_runtable.tsv

FASTQs land in data/FQfiles/external/{species}/{dataset}/{sample}/ and the
sheet in configs/external/{dataset}_samples.yaml. See the module docstrings in
ena.py (accession source decision), cngb.py (URL/CNSA listing) and download.py
(SRA `--include-technical`) for the parts that are easy to get wrong.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

from . import cngb, ena, sheet
from .download import assign_read_roles, download_run
from .util import (is_truthy, log, read_tsv, run_id, sanitize, setup_logger,
                   write_tsv)

DEFAULT_EXTERNAL_ROOT = "data/FQfiles/external"
DEFAULT_MANIFEST_DIR = "data/FQfiles/external/manifests"
DEFAULT_SHEET_DIR = "configs/external"


# ---------------------------------------------------------------------------
# metadata subcommand
# ---------------------------------------------------------------------------
def cmd_metadata(args: argparse.Namespace) -> int:
    """Resolve a dataset accession to a per-run metadata TSV.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        int: Process exit status.
    """
    if bool(args.accession) == bool(args.url):
        log().error("Provide exactly one of: an accession, or --url URL... "
                    "(got %s).", "both" if args.accession else "neither")
        return 1
    if args.url and not args.species:
        log().error("--url mode requires --species (the listing has no organism).")
        return 1
    if args.url:
        # URL mode: CNSA run directories or direct FASTQ URLs. No ENA/GEO query;
        # the archive has no metadata API, so species must be supplied.
        records = cngb.records_from_urls(args.url, species=args.species,
                                         dataset=args.dataset)
        label = args.dataset or (records[0]["dataset"] if records else "external")
    else:
        query_acc, context = ena.to_ena_query_accession(args.accession)
        log().info("[METADATA] querying ENA for %s (from %s)", query_acc, args.accession)
        rows = ena.ena_filereport(query_acc)
        records = ena.build_records(rows, context, args.species)
        label = args.accession

    out = args.output or os.path.join(
        DEFAULT_MANIFEST_DIR, f"{sanitize(label)}_runtable.tsv")
    write_tsv(out, records, ena.MANIFEST_FIELDS)

    # A short human summary so the curator knows what they're editing.
    by_source: dict[str, int] = defaultdict(int)
    by_species: dict[str, int] = defaultdict(int)
    for r in records:
        by_source[r["source"]] += 1
        by_species[r["species"]] += 1
    log().info("[METADATA] %d runs -> %s", len(records), out)
    log().info("[METADATA]   species: %s",
               ", ".join(f"{k}={v}" for k, v in sorted(by_species.items())))
    log().info("[METADATA]   sources: %s",
               ", ".join(f"{k}={v}" for k, v in sorted(by_source.items())))
    if by_source.get("sra"):
        log().info("[METADATA]   note: %d run(s) will use the SRA route "
                   "(prefetch + fasterq-dump --include-technical); needs the "
                   "scFetch env and scratch space ~3x the .sra size.",
                   by_source["sra"])
    print(f"\nRun table written to: {out}")
    print("Edit the 'download' column (yes/no), then run:")
    print(f"  python -m fetch.main download -m {out}\n")
    return 0


# ---------------------------------------------------------------------------
# download subcommand
# ---------------------------------------------------------------------------
def _dest_dir(root: str, record: dict) -> str:
    """Where one run's FASTQs belong under the fetch root.

    Args:
        root (Path): Fetch root.
        record (dict): The run record.

    Returns:
        Path: The destination directory.
    """
    return os.path.join(root, record["species"], record["dataset"], record["sample_name"])


def cmd_download(args: argparse.Namespace) -> int:
    """Download every run in a metadata TSV and write a sample sheet.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        int: Process exit status.
    """
    records = [r for r in read_tsv(args.metadata) if is_truthy(r.get("download"))]
    if not records:
        log().error("No rows with download=yes in %s", args.metadata)
        return 1
    log().info("[DOWNLOAD] %d run(s) selected from %s", len(records), args.metadata)

    scratch = args.scratch or os.environ.get("TMPDIR") or f"/tmp/{run_id()}"
    os.makedirs(scratch, exist_ok=True)

    # Per (species, dataset, sample) -> accumulated R1/R2 lists + chemistry votes.
    grouped: dict[tuple, dict] = defaultdict(
        lambda: {"fastq1": [], "fastq2": [], "chem": [], "runs": []})
    failures: list[tuple[str, str]] = []

    for record in records:
        dest = _dest_dir(args.external_root, record)
        key = (record["species"], record["dataset"], record["sample_name"])
        if args.dry_run:
            log().info("[DRY-RUN] would download %s (%s) via %s -> %s",
                       record["run_accession"], record["sample_name"],
                       record["source"], dest)
            continue
        try:
            paths = download_run(record, dest, threads=args.threads,
                                 scratch=scratch, check_md5=args.check_md5)
            roles = assign_read_roles(paths)
            g = grouped[key]
            g["runs"].append(record["run_accession"])
            if roles["barcode"]:
                g["fastq1"].append(os.path.abspath(roles["barcode"]))
            if roles["cdna"]:
                g["fastq2"].append(os.path.abspath(roles["cdna"]))
            if roles["chemistry"]:
                g["chem"].append(roles["chemistry"])
        except Exception as exc:  # noqa: BLE001 - collect, keep going
            log().error("[FAIL] %s (%s): %s", record["run_accession"],
                        record["sample_name"], exc)
            failures.append((record["run_accession"], str(exc)))

    if args.dry_run:
        return 0

    if args.emit_sheet:
        _emit_sheets(grouped, records, args)

    if failures:
        log().error("[DOWNLOAD] %d run(s) failed:", len(failures))
        for run, why in failures:
            log().error("           %s: %s", run, why)
        return 1
    log().info("[DOWNLOAD] all selected runs complete.")
    return 0


def _emit_sheets(grouped: dict, records: list[dict], args: argparse.Namespace) -> None:
    """Build one sample sheet per dataset present in the download."""
    # Map each group key back to its dataset for filename grouping.
    per_dataset: dict[str, dict] = defaultdict(dict)
    for (species, dataset, sample_name), g in grouped.items():
        if not g["fastq1"] or not g["fastq2"]:
            log().warning("[SHEET] %s/%s/%s: unresolved R1/R2 (R1=%d R2=%d) -- "
                          "omitted from the sheet; add it manually.",
                          species, dataset, sample_name, len(g["fastq1"]), len(g["fastq2"]))
            continue
        chem = _majority(g["chem"])
        tech = args.tech or chem  # explicit override beats detection
        if tech is None:
            log().warning("[SHEET] %s: chemistry undetected; leaving tech to the "
                          "sheet default. Set --tech to be explicit.", sample_name)
        if chem == "10xv2":
            log().warning("[SHEET] %s: detected 10xv2 (26bp R1). technology_config.yaml "
                          "defines 10xv3 and bd_rhapsody only -- add a 10xv2 entry "
                          "before mapping.", sample_name)
        entry = sheet.sample_entry(
            fastq1=g["fastq1"], fastq2=g["fastq2"],
            batch=f"{dataset}_{sample_name}",
            tech=(tech if tech and tech != args.default_tech else None),
            sample_type=None,
        )
        per_dataset[dataset][f"{dataset}_{sample_name}"] = entry

    for dataset, entries in per_dataset.items():
        doc = sheet.build_sheet(entries, default_tech=args.default_tech,
                                sample_type=args.sample_type)
        out = args.sheet_out or os.path.join(DEFAULT_SHEET_DIR, f"{dataset}_samples.yaml")
        sheet.write_sheet(doc, out)
        print(f"\nSample sheet written to: {out}")
        print("Review it, then point pipeline_config.yaml's configs.samples at it "
              "(or copy the entries into an existing sheet).\n")


def _majority(votes: list[str]) -> str | None:
    """The most common value in `votes`, for collapsing per-run fields.

    Args:
        votes (iterable): The values.

    Returns:
        The most common value, or None when there are none.
    """
    if not votes:
        return None
    return max(set(votes), key=votes.count)


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """The `fetch` argument parser.

    Returns:
        argparse.ArgumentParser: The parser.
    """
    p = argparse.ArgumentParser(
        prog="python -m fetch.main",
        description="Fetch external scRNA-seq datasets from GEO/SRA/ENA.")
    sub = p.add_subparsers(dest="command", required=True)

    m = sub.add_parser(
        "metadata", help="build a curatable run table for an accession or URL(s)",
        description="Two input modes: an accession (GEO/SRA/ENA, queried via "
                    "their APIs) OR --url for archives with no API (CNGB/CNSA "
                    "run directories, or any direct FASTQ URL).")
    m.add_argument("accession", nargs="?",
                   help="GSE.../GSM.../SRP/ERP/DRP/PRJ.../SRX.../SRR...")
    m.add_argument("--url", nargs="+", metavar="URL",
                   help="CNSA run-directory URL(s) (.../CNR....../) or direct "
                        "FASTQ URL(s). Requires --species. Mutually exclusive "
                        "with the positional accession.")
    m.add_argument("--dataset", help="dataset dir name for --url mode "
                   "(default: CNP from the URL, else 'external')")
    m.add_argument("-o", "--output", help="run-table TSV path "
                   f"(default {DEFAULT_MANIFEST_DIR}/<acc-or-dataset>_runtable.tsv)")
    m.add_argument("--species", help="species dir name. Optional for an "
                   "accession (defaults to the ENA scientific_name's genus); "
                   "REQUIRED for --url mode (a listing has no organism).")
    m.set_defaults(func=cmd_metadata)

    d = sub.add_parser("download", help="download the kept runs + emit a sample sheet")
    d.add_argument("-m", "--metadata", required=True, help="curated run-table TSV")
    d.add_argument("--external-root", default=DEFAULT_EXTERNAL_ROOT,
                   help=f"FASTQ tree root (default {DEFAULT_EXTERNAL_ROOT})")
    d.add_argument("--threads", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
                   help="threads for fasterq-dump / pigz (default: SLURM_CPUS_PER_TASK or 8)")
    d.add_argument("--scratch", help="scratch dir for fasterq-dump temp "
                   "(default: $TMPDIR or /tmp/<run_id>)")
    d.add_argument("--check-md5", action="store_true",
                   help="verify ENA md5s after download (slow on large files)")
    d.add_argument("--tech", help="force the 10x chemistry written to the sheet "
                   "(e.g. 10xv3), overriding R1-length detection")
    d.add_argument("--default-tech", default="10xv3",
                   help="sheet default tech when a sample's chemistry matches it "
                        "(default 10xv3)")
    d.add_argument("--sample-type", default="protoplast",
                   choices=["protoplast", "nuclei"],
                   help="sample_type for the emitted sheet (default protoplast)")
    d.add_argument("--no-sheet", dest="emit_sheet", action="store_false",
                   help="download only; do not emit a sample sheet")
    d.add_argument("--sheet-out", help="sample-sheet path "
                   f"(default {DEFAULT_SHEET_DIR}/<dataset>_samples.yaml)")
    d.add_argument("--dry-run", action="store_true",
                   help="show what would be downloaded, transfer nothing")
    d.set_defaults(func=cmd_download, emit_sheet=True)
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the fetch stage.

    Args:
        argv (list[str], optional): Argument vector.

    Returns:
        int: Process exit status.
    """
    args = build_parser().parse_args(argv)
    # One log file per run, in its own logs/fetch/ subdirectory -- same
    # [pipeline]-[run_id].log convention as shared.utils.pipeline_log_path,
    # kept as a separate implementation here since fetch deliberately
    # imports nothing from shared/ (see util.py's own module docstring).
    setup_logger(log_file=os.path.join("logs", "fetch", f"fetch-{run_id()}.log"))
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001 - top-level guard
        log().error("[FATAL] %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
