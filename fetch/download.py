"""Download one run's FASTQs and identify its 10x read roles.

Two routes, chosen per run by the ``source`` recorded in the manifest:

* ``ena_submitted`` / ``ena_fastq`` -- ``curl`` each ENA URL directly
  (resumable, optional md5 check).
* ``sra`` -- ``prefetch`` the .sra from NCBI, then ``fasterq-dump
  --split-files --include-technical`` to rebuild the reads, then ``pigz``.
  ``--include-technical`` is load-bearing: without it fasterq-dump drops the
  10x cell-barcode read (it is "technical" in SRA's model) and you are left
  with cDNA only.

After the files land, ``assign_read_roles`` labels them by modal read length
rather than by the ``_1/_2/_3`` numbering, which reflects read index and not
biological role.
"""

from __future__ import annotations

import hashlib
import os
import subprocess

from .util import log, modal_read_length, run_cmd, which

# 10x cell-barcode read lengths: v2 = 16bp CB + 10bp UMI, v3 = 16 + 12.
CHEMISTRY_BY_R1_LEN = {26: "10xv2", 28: "10xv3"}
_INDEX_MAX_LEN = 12  # I1/I2 sample indices are 8bp (dual up to ~10)


# ---------------------------------------------------------------------------
# Transfer primitives
# ---------------------------------------------------------------------------
def _with_scheme(url: str) -> str:
    """Prefix a bare host path with a scheme so curl accepts it.

    Args:
        url (str): The URL as given.

    Returns:
        str: The URL with a scheme.
    """
    return url if url.startswith(("http://", "https://", "ftp://")) else "https://" + url


def _curl(url: str, dest: str) -> None:
    """Fetch one URL to `dest` with curl, resuming a partial file.

    Args:
        url (str): Source.
        dest (Path): Destination.

    Returns:
        bool: Whether the download succeeded.
    """
    # --retry-all-errors is load-bearing: ENA throttles/blocks an IP after a
    # sustained multi-hour transfer and then returns a *non-transient* HTTP error
    # (e.g. 403) instantly. Plain --retry does not retry those (nor a mid-stream
    # recv failure, curl exit 56), so a whole batch fails the moment the block
    # kicks in. --retry-all-errors makes every error retryable, and -C - resumes
    # the partial file across each retry rather than restarting from zero.
    run_cmd(["curl", "-L", "--fail",
             "--retry", "10", "--retry-delay", "15", "--retry-all-errors",
             "--retry-connrefused", "-C", "-", "-o", dest, _with_scheme(url)])


def _remote_size(url: str) -> int | None:
    """Content-Length for ``url`` via a HEAD request, or None if unavailable.

    Used to tell a *complete* file (skip) from a *truncated* one (resume) --
    without it a partial download left by an interrupted transfer looks
    identical to a finished one and gets silently accepted.
    """
    try:
        out = subprocess.run(["curl", "-sIL", "--max-time", "60", _with_scheme(url)],
                             capture_output=True, text=True, timeout=90)
        sizes = [ln.split(":", 1)[1].strip() for ln in out.stdout.splitlines()
                 if ln.lower().startswith("content-length:")]
        return int(sizes[-1]) if sizes else None  # last hop after redirects
    except Exception:  # noqa: BLE001 - verification is best-effort
        return None


def _md5(path: str) -> str:
    """md5 of a file, read in chunks.

    Args:
        path (Path): The file.

    Returns:
        str: Hex digest.
    """
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Route: direct URL download (ENA submitter/archive files, CNGB, or any host)
# ---------------------------------------------------------------------------
def _download_urls(record: dict, dest_dir: str, check_md5: bool) -> list[str]:
    """Fetch a record's FASTQs from their direct URLs.

    Args:
        record (dict): The run record.
        dest_dir (Path): Where to write.
        check_md5 (bool): Verify against the recorded checksums.

    Returns:
        list[Path]: The downloaded files.
    """
    urls = [u for u in record["file_urls"].split(";") if u]
    md5s = [m for m in record.get("file_md5", "").split(";") if m]
    # Prefix with the run accession so runs sharing a sample dir (multi-lane, or
    # a CNGB sample whose files have generic names) never overwrite each other --
    # mirrors the ENA route's SRR-prefixed filenames.
    run = record.get("run_accession", "")
    paths: list[str] = []
    for i, url in enumerate(urls):
        fname = os.path.basename(url)
        if run and not fname.startswith(run):
            fname = f"{run}_{fname}"
        dest = os.path.join(dest_dir, fname)

        # Decide skip / resume / (re)download by comparing the file on disk to
        # the remote size. The old code skipped any file with size > 0, so an
        # interrupted transfer left a truncated FASTQ that was silently reused.
        expected = _remote_size(url)
        need_download = True
        if os.path.exists(dest):
            local = os.path.getsize(dest)
            if expected is not None and local == expected:
                log().info("[SKIP] complete (%d bytes): %s", local, dest)
                need_download = False
            elif expected is not None and local > expected:
                log().warning("[REDOWNLOAD] %s is larger than expected (%d > %d); "
                              "removing and refetching.", dest, local, expected)
                os.remove(dest)
            elif local > 0:
                log().info("[RESUME] %s (%d/%s bytes on disk)", dest, local,
                           expected if expected is not None else "?")

        if need_download:
            _curl(url, dest)  # -C - resumes from any partial left above
            if expected is not None:
                got = os.path.getsize(dest)
                if got != expected:
                    # curl returned success but the file is short (e.g. a
                    # truncated proxy response) -- fail loudly rather than accept
                    # it. The partial stays on disk so the next run resumes it.
                    raise RuntimeError(
                        f"incomplete download: {dest} is {got} bytes, expected {expected}")

        if check_md5 and i < len(md5s):
            got_md5 = _md5(dest)
            if got_md5 != md5s[i]:
                raise RuntimeError(f"md5 mismatch for {dest}: expected {md5s[i]}, got {got_md5}")
            log().info("[MD5] ok: %s", fname)
        paths.append(dest)
    return paths


# ---------------------------------------------------------------------------
# Route: SRA toolkit reconstruction
# ---------------------------------------------------------------------------
def _download_sra(record: dict, dest_dir: str, threads: int, scratch: str) -> list[str]:
    """Fetch a record via sra-tools, converting to FASTQ.

    Args:
        record (dict): The run record.
        dest_dir (Path): Where to write.
        threads (int): Worker threads.
        scratch (Path, optional): Scratch root for the conversion.

    Returns:
        list[Path]: The downloaded files.
    """
    run = record["run_accession"]
    for tool in ("prefetch", "fasterq-dump", "pigz"):
        if which(tool) is None:
            raise RuntimeError(
                f"'{tool}' not found on PATH. The SRA route needs sra-tools + pigz "
                f"(see batch_scripts/run_fetch.sh / the scFetch env)."
            )
    tmp = os.path.join(scratch, f"_sra_{run}")
    os.makedirs(tmp, exist_ok=True)

    # prefetch downloads the .sra into {tmp}/{run}/{run}.sra
    run_cmd(["prefetch", "--max-size", "u", "-O", tmp, run])
    sra_path = os.path.join(tmp, run, f"{run}.sra")
    if not os.path.exists(sra_path):
        sra_path = run  # let fasterq-dump resolve it from the cache

    # --include-technical keeps the barcode read; --split-files writes _1/_2/...
    run_cmd(["fasterq-dump", "--split-files", "--include-technical",
             "--threads", str(threads), "--temp", scratch,
             "--outdir", dest_dir, sra_path])

    produced = sorted(
        os.path.join(dest_dir, f) for f in os.listdir(dest_dir)
        if f.startswith(run) and f.endswith(".fastq")
    )
    if not produced:
        raise RuntimeError(f"fasterq-dump produced no FASTQ for {run}.")
    run_cmd(["pigz", "-f", "-p", str(threads), *produced])  # -f: overwrite on re-run
    gzipped = [p + ".gz" for p in produced]

    # Clean the .sra; the reads are what we keep.
    try:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass
    return gzipped


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def download_run(record: dict, dest_dir: str, *, threads: int, scratch: str,
                 check_md5: bool = False) -> list[str]:
    """Fetch one run, preferring direct URLs and falling back to SRA.

    Args:
        record (dict): The run record.
        dest_dir (Path): Where to write.
        threads (int): Worker threads.
        scratch (Path, optional): Scratch root.
        check_md5 (bool): Verify against the recorded checksums.

    Returns:
        list[Path]: The downloaded files.
    """
    os.makedirs(dest_dir, exist_ok=True)
    source = record["source"]
    log().info("[DOWNLOAD] %s (%s) via %s -> %s",
               record["run_accession"], record["sample_name"], source, dest_dir)
    if source in ("ena_submitted", "ena_fastq", "url"):
        return _download_urls(record, dest_dir, check_md5)
    if source == "sra":
        return _download_sra(record, dest_dir, threads, scratch)
    raise RuntimeError(f"Unknown source '{source}' for run {record['run_accession']}.")


# ---------------------------------------------------------------------------
# Read-role assignment
# ---------------------------------------------------------------------------
def assign_read_roles(paths: list[str]) -> dict:
    """Classify a run's FASTQs into barcode (R1) / cDNA (R2) / index by length.

    Returns ``{"barcode": path|None, "cdna": path|None, "index": [paths],
    "chemistry": "10xv2"|"10xv3"|None, "lengths": {path: len}}``. Best-effort:
    warns and leaves roles ``None`` when the geometry is unexpected.
    """
    lengths = {p: modal_read_length(p) for p in paths}
    result = {"barcode": None, "cdna": None, "index": [], "chemistry": None, "lengths": lengths}

    for path, length in sorted(lengths.items(), key=lambda kv: kv[1]):
        if length in CHEMISTRY_BY_R1_LEN and result["barcode"] is None:
            result["barcode"] = path
            result["chemistry"] = CHEMISTRY_BY_R1_LEN[length]
        elif length <= _INDEX_MAX_LEN:
            result["index"].append(path)
        else:  # longest wins as cDNA
            result["cdna"] = path

    # If nothing matched a known barcode length, fall back to shortest-non-index.
    if result["barcode"] is None:
        candidates = [p for p, l in lengths.items()
                      if l > _INDEX_MAX_LEN and p != result["cdna"]]
        if candidates:
            result["barcode"] = min(candidates, key=lambda p: lengths[p])
            blen = lengths[result["barcode"]]
            # A ~30bp barcode read is the DNBSEQ/MGI DNBelab C4 geometry
            # (20bp split cell barcode + 10bp UMI), not 10x -- flag it so the
            # sheet is not mislabelled and a chemistry profile gets added.
            hint = (" This looks like DNBSEQ/DNBelab (not 10x): mapping needs a "
                    "custom alevin chemistry + whitelist, not a 10x profile."
                    if 20 <= blen <= 32 else "")
            log().warning("[ROLES] no 26/28bp (10x) barcode read among %s; guessed "
                          "R1=%s (len %d).%s Verify chemistry before mapping.",
                          [os.path.basename(p) for p in paths],
                          os.path.basename(result["barcode"]), blen, hint)
    if result["barcode"] is None or result["cdna"] is None:
        log().warning("[ROLES] could not fully resolve R1/R2 for %s (lengths %s).",
                      [os.path.basename(p) for p in paths],
                      {os.path.basename(p): l for p, l in lengths.items()})
    return result
