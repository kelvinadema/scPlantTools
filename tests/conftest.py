"""
Shared test configuration.

DELIBERATELY IMPORTS NOTHING FROM THE PIPELINE. pytest imports this before
collecting anything, so a heavy import here would break collection in every
environment that lacks the dependency -- and the whole point of the directory
split below is that different subtrees need different environments.

    tests/portable/     stdlib + shared/ only. Runs in ANY pipeline env.
    tests/mapping/      needs scMappingV3.12
    tests/integration/  needs scIntegrationV1.12
    tests/annotation/   needs scAnnotationV3.12

WHY DIRECTORIES AND NOT MARKERS. A module-scope `import mapping.qc_filter` fails
at COLLECTION time in the wrong environment, and a collection error fails the
whole run rather than one file. `pytest.importorskip` would avoid that, but then
running in the wrong env silently skips a whole tier and still prints green --
worse than failing. Directories make the omission structural: `python -m sc test`
iterates them explicitly and reports per environment, so "all passed" means all
of them actually ran.
"""

from __future__ import annotations

import pathlib

import pytest


def find_repo_root(start: pathlib.Path = None) -> pathlib.Path:
    """
    Nearest ancestor containing the `.root` marker.

    Walks rather than counting `.parent`s so a test can move between
    `tests/` and `tests/<subdir>/` without silently resolving one level short --
    which is exactly what happened when these files moved into `portable/`.
    Same marker `shared.utils.find_root_dir` and `batch_scripts/lib/common.sh`
    use, so all three agree on what the root is.
    """
    here = (start or pathlib.Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / '.root').exists():
            return parent
    raise RuntimeError('no .root marker above {}'.format(here))


REPO_ROOT = find_repo_root()


@pytest.fixture(scope='session')
def repo_root() -> pathlib.Path:
    return REPO_ROOT


def pytest_configure(config):
    config.addinivalue_line(
        'markers', 'slow: takes more than a second or two; excluded by `sc test --fast`')
    config.addinivalue_line(
        'markers', 'needs_gpu: requires a visible GPU; skipped on login nodes')
