"""CNGB / CNSA support, and a generic direct-URL fallback.

CNGB (db.cngb.org / ftp.cngb.org) is a separate archive from GEO/SRA/ENA with
its own accession space (CNP project / CNS sample / CNX experiment / CNR run).
It exposes **no documented public metadata API**, ENA does not mirror it, and
sra-tools cannot fetch a CNR run -- so the ENA/SRA machinery in ena.py cannot
reach it. The `db.cngb.org/data_resources/...` pages are a JavaScript app that
injects nothing server-side, so they are not scrapable either.

What *does* work: the **ftp.cngb.org file tree** is server-rendered HTML at
every level. A directory page lists its subdirectories as
``<tr id="CN..." class="folder">`` rows and its files as
``<tr id="NAME" class="file">`` rows, and the files download over ordinary HTTP
with no auth and no raysync client. So given any node in the tree -- a project
(``.../CNP....../``), sample (``.../CNS....../``), experiment or run -- this
module crawls down to every FASTQ and builds one manifest record per run.

The one thing that is *not* derivable is the entry point: the ``dataN`` storage
bucket is not encoded in the accession, and the `db.cngb.org` sample page does
not expose the ftp path. So the caller supplies an ftp.cngb.org URL (grab it
from the CNSA site's download button for any sample in the project, then
truncate to the CNP/CNS level to fetch more). A whole project is one URL.

The same path handles any direct FASTQ URL (journal supplementary files, a lab
server, ...), which is why the source is the generic ``url`` rather than
``cngb``.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit

from .util import genus_dir, http_get, log, sanitize

# Directory rows: subdirectories are class="folder", files class="file".
_FILE_ROW = re.compile(
    r'<tr id="([^"]+)" class="file">.*?<td class="text-right">\s*([^<\s][^<]*)',
    re.S)
_FOLDER_ROW = re.compile(r'<tr id="([^"]+)" class="folder">')
_FASTQ = re.compile(r"\.(fq|fastq)(\.gz)?$", re.I)
_CNR = re.compile(r"(CNR\d+)", re.I)
_CNP = re.compile(r"(CNP\d+)", re.I)
_MAX_DEPTH = 6  # CNP -> CNS -> CNX -> CNR -> files is 4; leave headroom

# Feature-barcode / hashing / multiplexing libraries are sequenced alongside the
# cDNA library but are NOT gene expression -- they must not be mapped through the
# normal alevin GEX path. Flagged in the manifest's `notes` so curation sees it.
_FEATURE_TOKENS = ("oligo", "hto", "hto", "adt", "feature", "hashing", "hash",
                   "cmo", "cellplex", "multiplex", "tag")
_GEX_TOKENS = ("cdna", "gex", "rna", "expression", "transcript")


def _is_fastq(name: str) -> bool:
    """Whether a filename looks like a (gzipped) FASTQ."""
    return bool(_FASTQ.search(name))


def _dir_url(url: str) -> str:
    """Normalise a URL to its directory form (trailing slash, no query)."""
    parts = urlsplit(url)
    path = parts.path if parts.path.endswith("/") else parts.path + "/"
    return urlunsplit((parts.scheme or "http", parts.netloc, path, "", ""))


def _library_note(filenames: list[str]) -> str:
    """A short note classifying a run's files (feature/oligo, GEX, or unknown) for the manifest."""
    low = " ".join(filenames).lower()
    if any(t in low for t in _FEATURE_TOKENS):
        return "feature/oligo library -- NOT gene expression, do not map as GEX"
    if any(t in low for t in _GEX_TOKENS):
        return "gene expression (cDNA)"
    return "cngb/url"


# ---------------------------------------------------------------------------
# Crawl
# ---------------------------------------------------------------------------
def _crawl(url: str, found: dict[str, list[tuple[str, str]]], depth: int = 0) -> None:
    """Recursively collect FASTQ files, grouped by their containing directory.

    ``found`` maps a leaf directory URL to ``[(filename, file_url), ...]``.
    Follows only ``class="folder"`` subdirectories (which go strictly deeper),
    so the walk terminates; ``_MAX_DEPTH`` is a belt-and-braces cap.
    """
    base = _dir_url(url)
    html = http_get(base)

    files = [(n.strip(), base + n.strip())
             for n, _size in _FILE_ROW.findall(html) if _is_fastq(n)]
    if files:
        found.setdefault(base, []).extend(files)

    if depth < _MAX_DEPTH:
        for folder in _FOLDER_ROW.findall(html):
            _crawl(base + folder.strip("/") + "/", found, depth + 1)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
def _sample_name(filenames: list[str], run_acc: str) -> str:
    """Sample name from a run's FASTQ filenames (drops the read-index suffix).

    ``sc_root_fresh_nuclei_b1_cDNA_1.fq.gz`` -> ``sc_root_fresh_nuclei_b1_cDNA``.
    Distinct cDNA/oligo libraries therefore get distinct sample names and are
    never merged. Falls back to the run accession if names are inconsistent.
    """
    bases = set()
    for name in filenames:
        stem = _FASTQ.sub("", name)
        stem = re.sub(r"[._-]?[Rr]?[12]$", "", stem)
        bases.add(stem)
    if len(bases) == 1:
        return sanitize(bases.pop())
    return sanitize(run_acc or "sample")


def records_from_urls(urls: list[str], *, species: str, dataset: str | None) -> list[dict]:
    """Build ``source=url`` manifest records from ftp.cngb.org URLs (crawled) or
    direct FASTQ URLs. One record per run directory.
    """
    if not species:
        raise RuntimeError("URL mode needs --species: the file listing carries "
                           "no organism, so the species dir cannot be inferred.")

    groups: dict[str, list[tuple[str, str]]] = {}
    for url in urls:
        if _is_fastq(urlsplit(url).path):  # a direct file URL
            parent = _dir_url(url.rsplit("/", 1)[0])
            groups.setdefault(parent, []).append((url.rsplit("/", 1)[-1], url))
        else:                               # a directory URL -> crawl it
            log().info("[CNGB] crawling %s ...", _dir_url(url))
            _crawl(url, groups)

    if not groups:
        raise RuntimeError(
            "No FASTQ files found. Pass an ftp.cngb.org URL (project/sample/"
            "run directory) or a direct FASTQ URL -- the db.cngb.org site pages "
            "are a JS app and cannot be crawled.")

    species_dir = sanitize(species) if " " not in species else genus_dir(species)
    records: list[dict] = []
    for parent in sorted(groups):
        files = sorted(groups[parent])
        filenames = [n for n, _ in files]
        file_urls = [u for _, u in files]
        cnr = _CNR.search(parent)
        run_acc = cnr.group(1) if cnr else sanitize(parent.rstrip("/").rsplit("/", 1)[-1])
        cnp = _CNP.search(parent)
        ds = dataset or (cnp.group(1) if cnp else "external")
        records.append({
            "download": "yes",
            "run_accession": run_acc,
            "sample_name": _sample_name(filenames, run_acc),
            "species": species_dir,
            "dataset": sanitize(ds),
            "source": "url",
            "scientific_name": species,
            "library_layout": "PAIRED" if len(file_urls) >= 2 else "SINGLE",
            "library_strategy": "",
            "instrument_platform": "",   # filled by read-role detection at download
            "instrument_model": "",
            "read_count": "",
            "sample_accession": "",
            "experiment_accession": "",
            "study_accession": "",
            "sample_alias": "",
            "file_urls": ";".join(file_urls),
            "file_md5": "",
            "fastq_bytes": "",
            "notes": _library_note(filenames),
        })
    n_feat = sum(1 for r in records if "feature/oligo" in r["notes"])
    log().info("[CNGB] built %d run record(s)%s", len(records),
               f" ({n_feat} feature/oligo -- not GEX)" if n_feat else "")
    return records
