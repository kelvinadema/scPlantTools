"""
Enforce the two-layer package rule.

    shared/     layer 0 -- imports nothing from a stage package, ever
    <stage>/    layer 1 -- never imports another stage at MODULE scope

A test rather than a convention, because as a convention it decayed (`hvg.py`
was once filed under integration and imported from shared/ and mapping/).

WHY MODULE SCOPE IS THE LINE. Each stage runs in its OWN conda environment with
an incompatible dependency set (scMappingV3.12, scIntegrationV1.12,
scAnnotationV3.12, scMarkersV3.12, scFetch). A module-scope import of another
stage therefore fails at import time in any env that lacks that stage's
dependencies -- after SLURM has already scheduled the job and the allocation is
burning. A deferred import inside a function only runs on the code path that
needs it, in a context where the caller has established the env is right.

So a function-scope cross-stage import is CORRECT and this test allows it. Three
exist deliberately:

    integration/report_embedding.py -> mapping.report
    annotation/markers_hook.py      -> markerid.markers
    refine/refine.py                -> integration.{integrate,diagnostics}

HOW TO FIX A FAILURE. Two legal options, in order of preference:

  1. If the imported thing has no stage-specific dependencies, PROMOTE it to
     `shared/`. That is what `hvg.py` needed -- it imports only scanpy, numpy,
     pandas, scipy and statsmodels.
  2. Otherwise, move the import inside the function that uses it and add a
     comment naming the env it requires.

Do NOT add an exemption here to make a failure go away. The whole value of this
file is that the exemption list stays short and argued.

These tests need no fixtures and no pipeline dependencies -- they parse source
with `ast` and never import it, so they run in any environment.
"""

from __future__ import annotations

import ast
import pathlib

import pytest


def _find_root():
    """Nearest ancestor with a `.root` marker -- depth-independent, so this file
    can move between tests/ and tests/<subdir>/ without resolving one level short."""
    here = pathlib.Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / '.root').exists():
            return parent
    raise RuntimeError('no .root marker above {}'.format(here))


REPO_ROOT = _find_root()

SHARED = 'shared'
STAGES = [
    'mapping',
    'integration',
    'annotation',
    'markerid',
    'refine',
    'fetch',
]
ALL_PKGS = [SHARED] + STAGES


def _source_files(package: str):
    for path in sorted((REPO_ROOT / package).rglob('*.py')):
        if '__pycache__' in path.parts:
            continue
        yield path


def _imports(path: pathlib.Path):
    """
    Yield (imported_top_level_package, lineno, is_module_scope) for one file.

    Module scope is `col_offset == 0`: an import indented at all is inside a
    function, class or conditional, so it does not run at import time.
    Relative imports (`from .hvg import ...`) are same-package by definition and
    are skipped.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:                      # relative import -> same package
                continue
            names = [node.module] if node.module else []
        else:
            continue
        for name in names:
            top = name.split('.')[0]
            if top in ALL_PKGS:
                yield top, node.lineno, node.col_offset == 0


def test_shared_imports_no_stage_package():
    """
    `shared/` is layer 0. It may not import a stage package at ANY scope.

    Stricter than the stage-to-stage rule on purpose: `shared/validate.py` and
    `shared/config_spec.py` are imported by preflight, which by design runs
    OUTSIDE the stage environments so that a bad config fails in seconds rather
    than after hours of GPU time. A deferred import in `shared/` would put that
    guarantee at the mercy of which code path happened to run.
    """
    violations = []
    for path in _source_files(SHARED):
        for imported, lineno, _ in _imports(path):
            if imported in STAGES:
                violations.append(
                    '{}:{} imports {}'.format(
                        path.relative_to(REPO_ROOT), lineno, imported))
    assert not violations, (
        'shared/ must not import a stage package -- it is layer 0 and preflight '
        'runs it outside the stage envs. Promote what is needed into shared/, or '
        'invert the dependency.\n  ' + '\n  '.join(violations))


@pytest.mark.parametrize('stage', STAGES)
def test_stage_does_not_import_another_stage_at_module_scope(stage):
    """
    A stage package may not import another stage package at module scope.

    Function-scope imports are permitted -- see this module's docstring for why
    the env split makes that the correct pattern rather than a workaround.
    """
    violations = []
    for path in _source_files(stage):
        for imported, lineno, module_scope in _imports(path):
            if imported in STAGES and imported != stage and module_scope:
                violations.append(
                    '{}:{} imports {} at module scope'.format(
                        path.relative_to(REPO_ROOT), lineno, imported))
    assert not violations, (
        '{} imports another stage at module scope. Each stage runs in its own '
        'conda env, so this fails at import time wherever that env is not '
        'installed -- after the job is already scheduled. Move the import inside '
        'the function that needs it, or promote the shared part to shared/.'
        '\n  {}'.format(stage, '\n  '.join(violations)))


def test_deferred_cross_stage_imports_are_the_known_set():
    """
    Inventory the function-scope cross-stage imports.

    Not a prohibition -- these are legal. It is a tripwire: each one is a place
    where two conda environments have to agree at runtime, and that is worth
    noticing when a new one appears rather than discovering it from a failed
    job. Update the expected set deliberately, with a note on which env pair the
    new edge assumes.
    """
    expected = {
        ('integration', 'mapping'),      # report_embedding -> report
        ('annotation', 'markerid'),      # markers_hook -> markers
        ('refine', 'integration'),       # refine -> integrate, diagnostics
    }
    found = set()
    for stage in STAGES:
        for path in _source_files(stage):
            for imported, _lineno, module_scope in _imports(path):
                if imported in STAGES and imported != stage and not module_scope:
                    found.add((stage, imported))
    assert found == expected, (
        'the set of deferred cross-stage imports changed.\n'
        '  added:   {}\n  removed: {}\n'
        'Each edge means two conda envs must both satisfy the import at runtime. '
        'If the addition is intended, add it to `expected` with a comment naming '
        'the env pair.'.format(
            sorted(found - expected) or 'none', sorted(expected - found) or 'none'))


def test_every_package_is_importable_as_a_package():
    """
    Each package carries an `__init__.py`.

    An implicit namespace package works for `python -m` from the repo root but
    fails in most other contexts, including pytest's own collection.
    """
    missing = [p for p in ALL_PKGS if not (REPO_ROOT / p / '__init__.py').exists()]
    assert not missing, 'missing __init__.py in: {}'.format(', '.join(missing))
