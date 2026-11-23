"""Resolve a public accession to a per-run table with download instructions.

Accepts a GEO series/sample (GSE.../GSM...), an SRA/ENA study (SRP.../ERP.../
DRP.../PRJ...), or an experiment/run/sample accession, and returns one
normalised record per sequencing run. Each record decides *where* the run's
FASTQs will come from (``source``):

    ena_submitted : ENA has the original submitter FASTQs (best; e.g. the
                    CellRanger R1/R2/I1 files) -> download those URLs directly.
    ena_fastq     : ENA has >=2 archive-generated FASTQs (already split into
                    R1/R2) -> download those URLs directly.
    sra           : ENA has only a single collapsed FASTQ (or none). For 10x
                    that file is unusable -- the cell-barcode read is not
                    recoverable from it -- so the run is reconstructed from the
                    .sra object with ``prefetch`` + ``fasterq-dump
                    --split-files --include-technical`` at download time.

The ``sra`` case is the common one for older 10x submissions (it is what
GSE123013 resolves to), which is why the download step must carry the SRA
toolkit even though the policy is "ENA first".
"""

from __future__ import annotations

import re

from .util import genus_dir, http_get, log, sanitize

ENA_PORTAL = "https://www.ebi.ac.uk/ena/portal/api/filereport"
NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
GEO_ACC = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"

# Fields pulled from the ENA portal. Order is the TSV column order returned.
ENA_FIELDS = [
    "run_accession", "experiment_accession", "sample_accession",
    "secondary_sample_accession", "study_accession", "sample_title",
    "sample_alias", "scientific_name", "tax_id", "library_strategy",
    "library_source", "library_selection", "library_layout",
    "instrument_platform", "instrument_model", "read_count", "base_count",
    "fastq_ftp", "fastq_bytes", "fastq_md5",
    "submitted_ftp", "submitted_bytes", "submitted_md5", "submitted_format",
    "sra_ftp", "sra_md5",
]


# ---------------------------------------------------------------------------
# Accession classification + GEO resolution
# ---------------------------------------------------------------------------
def classify(accession: str) -> str:
    """Accession kind from its prefix: 'geo_series', 'geo_sample', 'study',
    'experiment', 'run', 'sample' or 'unknown'."""
    acc = accession.strip().upper()
    if acc.startswith("GSE"):
        return "geo_series"
    if acc.startswith("GSM"):
        return "geo_sample"
    if re.match(r"^(SR|ER|DR)P\d+$", acc) or acc.startswith("PRJ"):
        return "study"
    if re.match(r"^(SR|ER|DR)X\d+$", acc):
        return "experiment"
    if re.match(r"^(SR|ER|DR)R\d+$", acc):
        return "run"
    if re.match(r"^(SR|ER|DR)S\d+$", acc) or acc.startswith("SAMN"):
        return "sample"
    return "unknown"


def resolve_geo_series(gse: str) -> dict:
    """Fetch the GEO SOFT text for a series and pull out its SRA study relation.

    ``!Series_relation = SRA: https://www.ncbi.nlm.nih.gov/sra?term=SRP171040``
    is the reliable GSE->SRP bridge. (NCBI ``esearch db=sra term=GSE...``
    returns nothing -- a GSE is not indexed in the SRA db.) Also returns the
    BioProject, organism and member GSMs for context.
    """
    text = http_get(GEO_ACC, {"acc": gse, "targ": "self", "form": "text", "view": "quick"})
    srp = re.search(r"Series_relation = SRA:\s*\S*term=([A-Z]+\d+)", text)
    prj = re.search(r"Series_relation = BioProject:\s*\S*/bioproject/(PRJ[A-Z]+\d+)", text)
    organism = re.search(r"Series_sample_organism = (.+)", text)
    gsms = re.findall(r"Series_sample_id = (GSM\d+)", text)
    if not (srp or prj):
        raise RuntimeError(
            f"{gse}: no SRA/BioProject relation found in GEO SOFT. Either the "
            f"series has no raw data on SRA, or the accession is wrong."
        )
    resolved = {
        "study": (srp.group(1) if srp else None),
        "bioproject": (prj.group(1) if prj else None),
        "organism": (organism.group(1).strip() if organism else ""),
        "gsms": gsms,
    }
    log().info("[RESOLVE] %s -> study=%s bioproject=%s (%d GSMs, %s)",
               gse, resolved["study"], resolved["bioproject"], len(gsms),
               resolved["organism"] or "organism unknown")
    return resolved


def resolve_geo_sample(gsm: str) -> str:
    """Map a single GSM to its SRA experiment/run via the GSM SOFT relation."""
    text = http_get(GEO_ACC, {"acc": gsm, "targ": "self", "form": "text", "view": "quick"})
    m = re.search(r"Sample_relation = SRA:\s*\S*term=([A-Z]+\d+)", text)
    if not m:
        raise RuntimeError(f"{gsm}: no SRA relation in GEO SOFT.")
    log().info("[RESOLVE] %s -> %s", gsm, m.group(1))
    return m.group(1)


def to_ena_query_accession(accession: str) -> tuple[str, dict]:
    """Return the accession to hand to the ENA portal, plus GEO context.

    GEO accessions are resolved to their SRA study/experiment first; everything
    else (SRP/PRJ/SRX/SRR/SRS) the ENA portal accepts directly.
    """
    kind = classify(accession)
    context: dict = {"input": accession, "kind": kind}
    if kind == "geo_series":
        info = resolve_geo_series(accession)
        context.update(info)
        # Prefer the BioProject: ENA indexes PRJ* reliably; a secondary SRP
        # occasionally lags. Fall back to the SRP.
        return (info["bioproject"] or info["study"]), context
    if kind == "geo_sample":
        return resolve_geo_sample(accession), context
    if kind == "unknown":
        raise RuntimeError(
            f"Unrecognised accession '{accession}'. Expected a GEO series/sample "
            f"(GSE.../GSM...) or an SRA/ENA study/experiment/run "
            f"(SRP/ERP/DRP/PRJ.../SRX.../SRR...)."
        )
    return accession, context


# ---------------------------------------------------------------------------
# ENA portal
# ---------------------------------------------------------------------------
def ena_filereport(accession: str) -> list[dict]:
    """Query the ENA portal for every read run under ``accession``."""
    tsv = http_get(ENA_PORTAL, {
        "accession": accession,
        "result": "read_run",
        "fields": ",".join(ENA_FIELDS),
        "format": "tsv",
    })
    lines = [ln for ln in tsv.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"ENA returned no runs for '{accession}'.")
    header = lines[0].split("\t")
    rows = []
    for ln in lines[1:]:
        cells = ln.split("\t")
        cells += [""] * (len(header) - len(cells))  # pad short rows
        rows.append(dict(zip(header, cells)))
    return rows


# ---------------------------------------------------------------------------
# Source decision + normalisation
# ---------------------------------------------------------------------------
def _looks_like_fastq_pair(urls: list[str]) -> bool:
    """Two-plus files that look like real FASTQs (not an index-only pair)."""
    fq = [u for u in urls if re.search(r"\.f(ast)?q(\.gz)?$", u, re.I)]
    return len(fq) >= 2


def decide_source(row: dict) -> tuple[str, list[str], list[str]]:
    """Choose a download source for one ENA run row.

    Returns ``(source, file_urls, md5s)``. For the ``sra`` source ``file_urls``
    is empty -- the run is reconstructed from the .sra at download time.
    """
    submitted = [u for u in row.get("submitted_ftp", "").split(";") if u]
    fastq = [u for u in row.get("fastq_ftp", "").split(";") if u]
    submitted_md5 = [u for u in row.get("submitted_md5", "").split(";") if u]
    fastq_md5 = [u for u in row.get("fastq_md5", "").split(";") if u]

    if _looks_like_fastq_pair(submitted):
        return "ena_submitted", submitted, submitted_md5
    if _looks_like_fastq_pair(fastq):
        return "ena_fastq", fastq, fastq_md5
    # Single collapsed FASTQ or nothing -> rebuild from the .sra. This is the
    # correct path for 10x: a lone fastq.gz has lost the barcode read.
    return "sra", [], []


def build_records(rows: list[dict], context: dict, species_override: str | None) -> list[dict]:
    """Turn raw ENA rows into curatable manifest records."""
    # Dataset dir = the accession the user actually typed, so the tree is
    # predictable: GSE123013 -> external/{species}/GSE123013/. For a direct
    # study/run accession that is the SRP/PRJ/SRR itself.
    dataset = context["input"]

    records: list[dict] = []
    for row in rows:
        source, urls, md5s = decide_source(row)
        sci = row.get("scientific_name", "")
        species = sanitize(species_override) if species_override else genus_dir(sci)
        # Sample grouping/naming: prefer the human title (GEO sets it to the GSM
        # title, e.g. "WT_rep1"), fall back to the alias then the sample id.
        sample = (row.get("sample_title") or row.get("sample_alias")
                  or row.get("sample_accession") or row.get("run_accession"))
        records.append({
            "download": "yes",
            "run_accession": row.get("run_accession", ""),
            "sample_name": sanitize(sample),
            "species": species,
            "dataset": sanitize(dataset),
            "source": source,
            "scientific_name": sci,
            "library_layout": row.get("library_layout", ""),
            "library_strategy": row.get("library_strategy", ""),
            "instrument_platform": row.get("instrument_platform", ""),
            "instrument_model": row.get("instrument_model", ""),
            "read_count": row.get("read_count", ""),
            "sample_accession": row.get("sample_accession", ""),
            "experiment_accession": row.get("experiment_accession", ""),
            "study_accession": row.get("study_accession", ""),
            "sample_alias": row.get("sample_alias", ""),
            "file_urls": ";".join(urls),
            "file_md5": ";".join(md5s),
            "fastq_bytes": row.get("fastq_bytes", ""),
            "notes": "",
        })
    return records


# Column order for the written manifest -- the columns a curator edits come
# first (download / run / sample / species / source).
MANIFEST_FIELDS = [
    "download", "run_accession", "sample_name", "species", "dataset", "source",
    "scientific_name", "library_layout", "library_strategy",
    "instrument_platform", "instrument_model", "read_count",
    "sample_accession", "experiment_accession", "study_accession",
    "sample_alias", "file_urls", "file_md5", "fastq_bytes", "notes",
]
