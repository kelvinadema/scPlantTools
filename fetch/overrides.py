"""Apply curated per-dataset / per-sample overrides to a fetch-emitted sheet.

``fetch download`` writes one sheet per dataset with a single ``defaults:``
block and per-sample entries carrying only what it could infer from the
archive (batch, tech, FASTQ paths). Design metadata -- which index to map
against (``mapping_config``), ``treatment``, ``time_h``, ``sample_type`` --
comes from reading the paper, not from ENA, so it lives in a small curated
YAML (``configs/external/atlas_overrides.yaml``) and is merged here, inside
the fetch job, right after the download. Keeping the curation in its own file
instead of hand-editing the emitted sheet means the sheet can be regenerated
(a refetch, an extra lane) without losing it.

Overrides file shape::

    datasets:
      GSE224539:
        defaults:                 # merged into the sheet's defaults: block
          sample_type: nuclei
        samples:                  # keyed by the EMITTED sample name
          GSE224539_A17_0h:
            treatment: none
          GSE224539_A17_24h:
            treatment: S_meliloti
            time_h: 24
            mapping_config: induced

A value of ``null`` deletes that key from the entry. Usage::

    python -m fetch.overrides --sheet configs/external/GSE224539_samples.yaml \\
        --overrides configs/external/atlas_overrides.yaml --dataset GSE224539 --strict

``--strict`` exits 1 when an override names a sample the sheet does not have:
a renamed sample would otherwise keep the sheet default and map against the
wrong index without anyone noticing.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from collections import OrderedDict

import yaml

from .sheet import _represent_ordereddict  # noqa: F401  (registers the representer)
from .util import log, setup_logger

# Keys written in this order so a sheet reads the same way everywhere; anything
# not listed keeps its relative order after these, FASTQ lists last.
_KEY_ORDER = ["batch", "tech", "sample_type", "mapping_config", "index_ref",
              "qc_filter", "qc_metrics", "prep", "treatment", "time_h", "genotype"]
_TAIL = ["fastq1", "fastq2"]


def _ordered(entry: dict) -> OrderedDict:
    out = OrderedDict()
    for k in _KEY_ORDER:
        if k in entry:
            out[k] = entry[k]
    for k in entry:
        if k not in out and k not in _TAIL:
            out[k] = entry[k]
    for k in _TAIL:
        if k in entry:
            out[k] = entry[k]
    return out


def _merge(target: dict, patch: dict) -> None:
    for k, v in (patch or {}).items():
        if v is None:
            target.pop(k, None)
        else:
            target[k] = v


def apply_overrides(sheet: dict, spec: dict, *, strict: bool) -> tuple[dict, list[str]]:
    """Return ``(new_sheet, unmatched_sample_keys)``."""
    doc = OrderedDict()
    defaults = dict(sheet.get("defaults") or {})
    _merge(defaults, spec.get("defaults") or {})
    doc["defaults"] = OrderedDict(defaults)

    samples = OrderedDict((k, dict(v)) for k, v in (sheet.get("samples") or {}).items())
    unmatched: list[str] = []
    for name, patch in (spec.get("samples") or {}).items():
        if name not in samples:
            unmatched.append(name)
            continue
        _merge(samples[name], patch or {})
    doc["samples"] = OrderedDict((k, _ordered(v)) for k, v in samples.items())
    if strict and unmatched:
        raise KeyError(unmatched)
    return doc, unmatched


def _leading_comments(path: str) -> str:
    lines = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                lines.append(line)
            else:
                break
    return "".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sheet", required=True, help="fetch-emitted sample sheet (rewritten in place)")
    ap.add_argument("--overrides", required=True, help="curated overrides YAML")
    ap.add_argument("--dataset", required=True, help="key under datasets: in the overrides file")
    ap.add_argument("--strict", action="store_true",
                    help="fail if an override names a sample missing from the sheet")
    ap.add_argument("--out", help="write here instead of rewriting --sheet")
    args = ap.parse_args(argv)
    setup_logger()

    sheet = yaml.safe_load(open(args.sheet)) or {}
    allspec = yaml.safe_load(open(args.overrides)) or {}
    spec = (allspec.get("datasets") or {}).get(args.dataset)
    if spec is None:
        log().error("[OVERRIDES] dataset %r not in %s (have: %s)", args.dataset,
                    args.overrides, ", ".join(sorted((allspec.get("datasets") or {}))))
        return 1

    try:
        doc, unmatched = apply_overrides(sheet, spec, strict=args.strict)
    except KeyError as exc:
        missing = exc.args[0]
        log().error("[OVERRIDES] %d override key(s) match no sample in %s: %s",
                    len(missing), args.sheet, ", ".join(missing))
        log().error("            sheet has: %s", ", ".join((sheet.get("samples") or {})))
        return 1
    for name in unmatched:
        log().warning("[OVERRIDES] no sample %r in sheet; override ignored", name)

    header = _leading_comments(args.sheet)
    stamp = _dt.date.today().isoformat()
    header += (f"# Overrides from {os.path.relpath(args.overrides)} "
               f"(dataset {args.dataset}) applied {stamp} by fetch.overrides.\n")
    out = args.out or args.sheet
    with open(out, "w") as fh:
        fh.write(header)
        yaml.dump(doc, fh, default_flow_style=False, sort_keys=False, width=200)
    n = sum(1 for k in (spec.get("samples") or {}) if k not in unmatched)
    log().info("[OVERRIDES] %s: defaults %s, %d sample override(s) applied -> %s",
               args.dataset, dict(spec.get("defaults") or {}), n, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
