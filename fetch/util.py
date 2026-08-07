"""Shared helpers for the external-data fetch stage.

Deliberately dependency-light: HTTP is done with the standard library
(``urllib``) and the big FASTQ transfers shell out to ``curl``. The only
third-party import is ``yaml`` (for emitting the sample-sheet fragment), which
is already present in the ``scFetch`` env. Nothing here imports from
``shared/`` so the tool can run in its own small conda env without dragging in
the mapping stage's scientific stack.
"""

from __future__ import annotations

import csv
import gzip
import logging
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request

LOGGER_NAME = "scFetch"

# NCBI asks unauthenticated callers to stay under 3 requests/second and to
# identify themselves. We identify via tool=/email= and self-throttle.
_NCBI_MIN_INTERVAL = 0.34  # seconds between NCBI calls (~3/s)
_last_ncbi_call = 0.0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logger(log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure and return the ``scFetch`` logger. Call once at the entrypoint.

    Everywhere else use ``logging.getLogger("scFetch")``. Messages carry a
    ``[FETCH]``-style bracket prefix by convention, matching the other stages.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")  # append: one file per run_id
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def log() -> logging.Logger:
    """The fetch stage's logger.

    Returns:
        logging.Logger: The logger.
    """
    return logging.getLogger(LOGGER_NAME)


def run_id() -> str:
    """A stable id for this invocation: SLURM job id if set, else a timestamp.

    Matches the other stages' ``shared.utils.pipeline_log_path`` convention so
    ``logs/fetch/fetch-<run_id>.log`` lines up with ``logs/fetch-<jobid>.out``.
    """
    return os.environ.get("SLURM_JOB_ID") or time.strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------------------
# HTTP (stdlib)
# ---------------------------------------------------------------------------
def http_get(
    url: str,
    params: dict | None = None,
    *,
    is_ncbi: bool = False,
    retries: int = 4,
    timeout: int = 60,
) -> str:
    """GET ``url`` (with optional query ``params``) and return the decoded body.

    Retries with exponential backoff on transient failures. When ``is_ncbi`` is
    set the call is throttled to respect NCBI's rate limit and stamped with
    ``tool``/``email`` (and ``api_key`` if ``NCBI_API_KEY`` is exported).
    """
    global _last_ncbi_call
    params = dict(params or {})

    if is_ncbi:
        params.setdefault("tool", "scRNAseq_fetch")
        email = os.environ.get("NCBI_EMAIL")
        if email:
            params.setdefault("email", email)
        api_key = os.environ.get("NCBI_API_KEY")
        if api_key:
            params.setdefault("api_key", api_key)

    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"

    last_err: Exception | None = None
    for attempt in range(retries):
        if is_ncbi:
            wait = _NCBI_MIN_INTERVAL - (time.time() - _last_ncbi_call)
            if wait > 0:
                time.sleep(wait)
            _last_ncbi_call = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "scRNAseq_fetch/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 - report and back off
            last_err = exc
            sleep = 2 ** attempt
            log().warning("HTTP GET failed (attempt %d/%d): %s -- retrying in %ds",
                          attempt + 1, retries, exc, sleep)
            time.sleep(sleep)
    raise RuntimeError(f"HTTP GET failed after {retries} attempts: {url}\n  {last_err}")


# ---------------------------------------------------------------------------
# Subprocess
# ---------------------------------------------------------------------------
def which(binary: str) -> str | None:
    """Path to a binary on PATH, or None.

    Args:
        binary (str): Binary name.

    Returns:
        str or None: Its path.
    """
    from shutil import which as _which
    return _which(binary)


def run_cmd(cmd: list[str], *, cwd: str | None = None, check: bool = True) -> int:
    """Run ``cmd`` (a list), streaming its output to the log. Returns the code."""
    log().info("[CMD] %s", " ".join(str(c) for c in cmd))
    proc = subprocess.run(cmd, cwd=cwd)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(map(str, cmd))}")
    return proc.returncode


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------
def sanitize(name: str) -> str:
    """Filesystem-safe token: keep word chars, collapse the rest to underscores."""
    name = re.sub(r"[^\w.-]+", "_", (name or "").strip())
    return re.sub(r"_+", "_", name).strip("_") or "unnamed"


def genus_dir(scientific_name: str) -> str:
    """Short species directory name = capitalised genus, matching the existing
    ``data/FQfiles/{Medicago,Marcantia}`` convention.

    ``"Arabidopsis thaliana"`` -> ``"Arabidopsis"``. An empty/unknown name
    becomes ``"Unknown"`` so nothing lands at the tree root by accident.
    """
    token = (scientific_name or "").strip().split()
    if not token:
        return "Unknown"
    return sanitize(token[0]).capitalize()


# ---------------------------------------------------------------------------
# TSV manifest IO
# ---------------------------------------------------------------------------
def write_tsv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    """Write rows to a TSV.

    Args:
        path (Path): Destination.
        rows (iterable[dict]): The rows.
        fieldnames (list[str], optional): Column order.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_tsv(path: str) -> list[dict]:
    """Read a TSV into a list of dicts.

    Args:
        path (Path): The file.

    Returns:
        list[dict]: The rows.
    """
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def is_truthy(value: str | None) -> bool:
    """Interpret the manifest's ``download`` column leniently."""
    return str(value).strip().lower() in {"yes", "y", "true", "1", "keep", "x"}


# ---------------------------------------------------------------------------
# FASTQ inspection
# ---------------------------------------------------------------------------
def modal_read_length(fastq_path: str, sample_reads: int = 400) -> int:
    """Modal sequence length over the first ``sample_reads`` reads of a FASTQ.

    Used to assign 10x read roles after download (barcode vs cDNA vs index)
    without trusting the ``_1/_2/_3`` numbering, which fasterq-dump orders by
    read index, not by biological role.
    """
    opener = gzip.open if fastq_path.endswith(".gz") else open
    lengths: dict[int, int] = {}
    with opener(fastq_path, "rt") as fh:
        for i, line in enumerate(fh):
            if i >= sample_reads * 4:
                break
            if i % 4 == 1:  # sequence line
                length = len(line.rstrip("\n"))
                lengths[length] = lengths.get(length, 0) + 1
    if not lengths:
        return 0
    return max(lengths, key=lengths.get)

