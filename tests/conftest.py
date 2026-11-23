"""
Shared test configuration.

Imports nothing from the pipeline: pytest loads this before collection, and
each test subtree needs a different environment (see docs/testing.md).

Directories, not markers, separate the environments. A module-scope import of
the wrong stage fails at collection and fails the whole run; `importorskip`
would instead skip a whole tier silently and still print green. `python -m sc
test` runs each directory in its own env, so "all passed" means all of them ran.
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
