"""
Named cohorts -- which dataset a mapping run operates on.

`configs.samples` alone is a single pointer to a sample sheet, so the command
line would not say which dataset it maps. A `cohorts:` block lets it:

    python -m mapping.main --cohort medicago
    python -m mapping.main --cohort external

A `pipeline_config.yaml` with no `cohorts:` block uses `configs.samples`, and
`--cohort` is rejected. A config WITH the block resolves `--cohort`, falling
back to `default_cohort:` when the flag is omitted, and errors listing the valid
names when neither is available. Nothing silently picks a cohort for you.

WHAT A COHORT MAY OVERRIDE. `samples` (required) and `output_dir` (optional).
Deliberately not the other sub-configs: qc/filter/technology settings are
statistical choices that should not fork per dataset without someone noticing,
and `index_config.yaml` is already keyed by name so a cohort selects its
reference through each sample's own `index_ref`, not here.

Kept free of stage imports so `shared/validate.py` can use it during preflight,
which runs outside the stage environments.
"""

from __future__ import annotations


class CohortConfigError(ValueError):
    """Raised for an unknown cohort name, or a cohorts block that is malformed."""


def cohort_names(pipeline_cfg: dict) -> list:
    """Declared cohort names, in file order. Empty if the config declares none."""
    block = pipeline_cfg.get('cohorts') or {}
    return list(block.keys())


def resolve_cohort(pipeline_cfg: dict, cohort: str = None) -> dict:
    """
    Return the effective `{'samples': ..., 'output_dir': ...}` for this run.

    `output_dir` is only present when the cohort or the file sets one; the caller
    keeps its own fallback for the file-level value. Returns a `name` key too, so
    callers can log and stamp which cohort a run used rather than inferring it
    from the sheet path.
    """
    block = pipeline_cfg.get('cohorts') or {}
    configs = pipeline_cfg.get('configs') or {}

    if not block:
        if cohort:
            raise CohortConfigError(
                "--cohort {!r} was given but this pipeline config declares no `cohorts:` "
                "block. Either add one, or drop the flag and let `configs.samples` "
                "({}) select the dataset.".format(cohort, configs.get('samples')))
        return {'name': None, 'samples': configs.get('samples'),
                'output_dir': pipeline_cfg.get('output_dir')}

    name = cohort or pipeline_cfg.get('default_cohort')
    if not name:
        raise CohortConfigError(
            "this pipeline config declares cohorts but no `default_cohort:`, so a "
            "cohort must be named explicitly. Use --cohort with one of: {}.".format(
                ', '.join(sorted(block))))
    if name not in block:
        raise CohortConfigError(
            "unknown cohort {!r}. Declared cohorts are: {}.".format(
                name, ', '.join(sorted(block))))

    entry = block[name] or {}
    if not isinstance(entry, dict):
        raise CohortConfigError(
            "cohort {!r} must be a mapping with at least a `samples:` key, got {}.".format(
                name, type(entry).__name__))
    samples = entry.get('samples')
    if not samples:
        raise CohortConfigError(
            "cohort {!r} does not set `samples:`, which is the sample sheet it "
            "selects and has no default.".format(name))

    return {'name': name,
            'samples': samples,
            'output_dir': entry.get('output_dir') or pipeline_cfg.get('output_dir')}
