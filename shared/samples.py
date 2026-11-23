#!/usr/bin/env python3

"""
Single resolution point for the sample sheet.

Every stage loads samples through here, so the sheet's `defaults:` block is
merged identically everywhere. A site that read an unresolved entry would fall
back to `mapper=''`, and `get_sample_output_paths(mapper=...)` would then
silently place its outputs in a different directory than the one mapping wrote
to.

`mapper` is required rather than defaulted for the same reason: an entry that
reaches `get_sample_output_paths` without one is unrecoverable, so it is worth
failing loudly at load time instead of writing to `{sample}//`.
"""


from pathlib import Path

from .utils import load_config


# Parsed sheets, keyed by (resolved path, mtime). The mtime is in the key so an
# edited sheet is re-read rather than served stale; within one run the file does
# not change, so every stage's load of a sample shares one parse.
_SAMPLE_CACHE = {}


class SampleConfigError(ValueError):
    """Raised for a sample sheet that cannot be resolved into usable entries."""


def _apply_defaults(defaults: dict, entry: dict) -> dict:
    """
    Merge one sample entry over the sheet-level defaults.

    The rule, deliberately kept to one sentence so it stays explainable:
    `defaults: {k: v}` is exactly equivalent to writing `k: v` on every sample
    entry. Nothing deeper -- the merge is shallow, so a sample that declares a
    list key replaces the default list rather than extending it.

    A key present but null in a sample entry counts as *absent*, matching
    `SingleCellMapper._merge_mapper_defaults`, which treats
    `self.config[key] is None` as unset. A sample therefore cannot null out a
    default; omit the key from `defaults:` and state it per sample instead.
    """
    resolved = dict(defaults)
    for key, value in (entry or {}).items():
        if value is not None:
            resolved[key] = value
    return resolved


def load_samples(path) -> dict:
    """
    Load a sample sheet and return {sample_name: fully-resolved entry}.

    The returned entries already have the sheet's `defaults:` block folded in,
    so a consumer can read them exactly as if every key had been written out
    per sample.
    """
    resolved_path = Path(path).resolve()
    try:
        mtime = resolved_path.stat().st_mtime
    except OSError as e:
        raise SampleConfigError('Cannot read sample sheet {}: {}'.format(resolved_path, e))

    cache_key = (str(resolved_path), mtime)
    if cache_key in _SAMPLE_CACHE:
        return _SAMPLE_CACHE[cache_key]

    raw = load_config(resolved_path) or {}
    if 'samples' not in raw:
        raise SampleConfigError(
            "Sample sheet {} has no top-level 'samples:' key.".format(resolved_path))

    defaults = raw.get('defaults') or {}
    if not isinstance(defaults, dict):
        raise SampleConfigError(
            "'defaults:' in {} must be a mapping of key: value, got {}.".format(
                resolved_path, type(defaults).__name__))

    samples = raw['samples'] or {}
    if not isinstance(samples, dict):
        raise SampleConfigError(
            "'samples:' in {} must be a mapping of sample_name: {{...}}, got {}.".format(
                resolved_path, type(samples).__name__))

    resolved = {name: _apply_defaults(defaults, entry) for name, entry in samples.items()}

    # Report every offending sample at once rather than failing on the first --
    # a 40-sample sheet should need one round-trip to fix, same convention as
    # mapping.main.validate_samples.
    missing = sorted(name for name, entry in resolved.items() if not entry.get('mapper'))
    if missing:
        raise SampleConfigError(
            "No 'mapper' for sample(s) {} in {}. Set it per sample, or once for the whole "
            "sheet under the top-level 'defaults:' block. It cannot be defaulted in code: "
            'the mapper name is a path component of every output directory, so an empty '
            'value would silently scatter this run across different paths than the ones '
            'mapping writes to.'.format(', '.join(repr(m) for m in missing), resolved_path))

    _SAMPLE_CACHE[cache_key] = resolved
    return resolved


def resolve_sample(sample_name: str, path) -> dict:
    """Return one fully-resolved sample entry from the sheet at `path`."""
    samples = load_samples(path)
    if sample_name not in samples:
        raise SampleConfigError(
            "Sample '{}' not found in {}. Known samples: {}".format(
                sample_name, Path(path).resolve(), ', '.join(sorted(samples)) or '(none)'))
    return samples[sample_name]


def resolve_index_ref(sample_entry: dict, mapper_cfg: dict):
    """
    Resolve which index a sample maps against, in precedence order.

    1. an `index_ref:` written directly on the sample -- the ad hoc escape
       hatch, for a one-off that does not warrant its own profile;
    2. the `index_ref` of the mapper profile the sample selects, i.e.
       `mappers.<mapper>.<mapping_config>` in mapping_config.yaml -- the
       reusable route, and how a second species should normally be wired in
       (add the index, add a profile naming it, point the sample's
       `mapping_config:` at that profile).

    Returns None when neither supplies one; callers decide whether that is
    fatal (`SingleCellMapper._resolve_index`) or merely nothing to build
    (`build_required_index`). Both go through here so that `--build-indexes`
    builds exactly the index a sample maps against.
    """
    direct = sample_entry.get('index_ref')
    if direct:
        return direct

    mapper = (sample_entry.get('mapper') or '').lower()
    profile = sample_entry.get('mapping_config', 'default')
    profile_entry = (mapper_cfg.get('mappers', {}) or {}).get(mapper, {}) or {}
    return (profile_entry.get(profile) or {}).get('index_ref')


def clear_cache() -> None:
    """Drop the parsed-sheet cache. For tests and long-lived processes."""
    _SAMPLE_CACHE.clear()
