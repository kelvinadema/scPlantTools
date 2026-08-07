"""
    python -m sc <command> [args...]

Run `python -m sc` with no arguments for the command list.

WHAT THIS IS. A dispatcher, not a workflow engine. It does two things the bare
`python -m <stage>.main` invocations cannot:

  1. Activates the right conda environment. Five stages, five mutually
     incompatible environments, and until now the mapping from stage to env
     lived only inside `batch_scripts/*.sh` -- so running a stage by hand meant
     knowing which env from memory, and getting it wrong failed on an import
     hundreds of lines in.

  2. Reports state (`sc status`), by reading the manifests, `.done` flags and
     registry the stages already write.

It deliberately does NOT sequence stages, infer what should run next, or rerun
anything on change. The stages are run deliberately, weeks apart, on different
cohorts; a DAG would be modelling a workflow that does not exist. Unrecognised
arguments are passed through untouched, so every stage's own flags keep working
and this file does not become a second place to add them.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import STAGE_ENVS


def _repo_root() -> Path:
    """
    Same resolution as `shared.utils.find_root_dir` and
    `batch_scripts/lib/common.sh`: $SCRNASEQ_ROOT, else the nearest ancestor
    with a `.root` marker. Not `__file__` -- see find_root_dir's docstring for
    why that is wrong under this repo's symlink layout.
    """
    env = os.environ.get('SCRNASEQ_ROOT')
    if env:
        return Path(env)
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / '.root').exists():
            return parent
    sys.exit("repo root not found: run from the repo root, or set $SCRNASEQ_ROOT.")


def _run_stage(command: str, passthrough: list, dry_run: bool = False) -> int:
    """Dispatch one stage into its conda env via `conda run`.

    Args:
        command (str): Stage name.
        passthrough (list[str]): Arguments forwarded untouched.
        dry_run (bool): Print the command instead of running it.

    Returns:
        int: The stage's exit status.
    """
    module, env = STAGE_ENVS[command]
    root = _repo_root()

    if shutil.which('conda') is None:
        # Fall back to the current interpreter. Correct when the caller has
        # already activated the env (which every batch script does), wrong and
        # loud otherwise -- better than silently using the base interpreter.
        print('[sc] conda not on PATH; using the active interpreter. '
              'Expected env: {}'.format(env), file=sys.stderr)
        cmd = [sys.executable, '-m', module] + passthrough
    else:
        # `--no-capture-output` so the stage's own stdout/stderr stream live
        # instead of being buffered to the end -- these are multi-hour jobs and
        # a buffered log is useless while one is running.
        cmd = ['conda', 'run', '-n', env, '--no-capture-output',
               'python', '-m', module] + passthrough

    print('[sc] env={} cwd={}\n[sc] {}'.format(env, root, ' '.join(cmd)), file=sys.stderr)
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=str(root))


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def _load_pipeline_cfg(root: Path, config: str):
    """Read pipeline_config.yaml from the repo root.

    Args:
        root (Path): Repo root.
        config (Path, optional): Explicit config path.

    Returns:
        dict: The parsed config.
    """
    import yaml
    path = Path(config)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        return {}, path
    return (yaml.safe_load(path.read_text()) or {}), path


def _tick(ok: bool) -> str:
    """A check or cross glyph for a boolean.

    Args:
        ok (bool): The state.

    Returns:
        str: The glyph.
    """
    return 'x' if ok else '.'


def cmd_status(args) -> int:
    """Print the sample x step matrix, or one sample's or target's detail.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        int: Process exit status.
    """
    sys.path.insert(0, str(_repo_root()))
    from shared.status import (mapping_status, integration_status,
                               simple_stage_status, MAPPING_STEPS)

    root = _repo_root()
    cfg, cfg_path = _load_pipeline_cfg(root, args.config)
    out_dir = cfg.get('output_dir') or str(root / 'results/mapping')
    results = Path(out_dir).parent

    if args.sample:
        state = mapping_status(out_dir).get(args.sample)
        if state is None:
            print('no mapping output for sample {!r} under {}'.format(args.sample, out_dir))
            return 1
        print('sample:  {}'.format(args.sample))
        print('mapper:  {}'.format(state['mapper'] or '<unknown>'))
        for step in MAPPING_STEPS:
            print('  [{}] {}'.format(_tick(state['steps'][step]), step))
        m = state['manifest']
        if not m:
            print('\n  no manifest.json -- this sample predates the output contract, '
                  'or the run did not reach a terminal write.')
            return 0
        print('\nmanifest:')
        print('  schema {}   status {}   {}'.format(
            m.get('schema_version', '?'), m.get('status', '?'), m.get('timestamp', '')))
        print('  batch  {}'.format(m.get('batch', '-')))
        print('  config_hash {}'.format((m.get('config_hash') or '')[:16]))
        arts = m.get('artifacts') or {}
        if arts:
            print('  artifacts:')
            for k, v in sorted(arts.items()):
                exists = (Path(out_dir) / args.sample / v).exists() if isinstance(v, str) else False
                print('    [{}] {:<24} {}'.format(_tick(exists), k, v))
        return 0

    if args.targets:
        targets = integration_status(results / 'integration')
        if not targets:
            print('no integration output under {}'.format(results / 'integration'))
            return 0
        print('{:<34} {:<22} {:>5} {:>6} {:>9}'.format(
            'TARGET', 'MODEL', 'h5ad', 'model', 'registry'))
        for name, t in sorted(targets.items()):
            print('{:<34} {:<22} {:>5} {:>6} {:>9}'.format(
                name[:34], t['model_class'][:22], _tick(t['has_h5ad']),
                _tick(t['has_model']), _tick(t['registry'] is not None)))
        print('\n{} target(s). A registry entry with no model on disk retrains on the next '
              'run; a model with no entry was a disposable (persist: False) run.'.format(len(targets)))
        return 0

    # default: mapping matrix + one-line summaries for the stages that do not
    # write manifests
    samples = mapping_status(out_dir)
    print('config:  {}'.format(cfg_path))
    if cfg.get('cohorts'):
        print('cohorts: {}   (default: {})'.format(
            ', '.join(sorted(cfg['cohorts'])), cfg.get('default_cohort', '<none>')))
    print('mapping: {}\n'.format(out_dir))
    if not samples:
        print('  no mapping output yet.')
    else:
        # Abbreviations, not truncations: `mapping` and `mappingstats` both
        # truncate to 'mapp', which makes two columns indistinguishable.
        abbrev = {'mapping': 'map', 'preprocessing': 'prep', 'mappingstats': 'stat',
                  'qcmetrics': 'qcm', 'qcfilter': 'qcf', 'qcplot': 'qcp',
                  'doublets': 'dbl', 'report': 'rpt'}
        head = ''.join(abbrev.get(s, s[:4]).rjust(5) for s in MAPPING_STEPS)
        print('{:<42}{}'.format('SAMPLE', head))
        for name, st in sorted(samples.items()):
            row = ''.join(_tick(st['steps'][s]).rjust(5) for s in MAPPING_STEPS)
            print('{:<42}{}'.format(name[:42], row))
        complete = sum(1 for s in samples.values() if all(s['steps'].values()))
        print('\n  {}/{} samples complete through all {} steps'.format(
            complete, len(samples), len(MAPPING_STEPS)))
        print('  steps: {}'.format('  '.join(
            '{}={}'.format(abbrev.get(x, x[:4]), x) for x in MAPPING_STEPS)))

    for label, sub in (('integration', 'integration'), ('annotation', 'annotation'),
                       ('refine', 'refine'), ('markers', 'markers')):
        if label == 'integration':
            n = len(integration_status(results / 'integration'))
            print('\n{:<12} {} target(s)   (sc status --targets for detail)'.format(label, n))
        else:
            entries = simple_stage_status(results / sub)
            print('{:<12} {} entr{}'.format(label, len(entries), 'y' if len(entries) == 1 else 'ies'),
                  end='')
            print('   -- no manifests written by this stage yet, so completeness is unknown'
                  if entries else '')
    return 0




# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------

# Test subtree -> the conda env it must run in. `portable` runs in whatever is
# already active: it imports only the stdlib and `shared/`, so any pipeline env
# will do. The rest import a stage package and therefore need that stage's env.
#
# This mapping is the reason `sc test` exists at all. `pytest tests/` from one
# environment cannot collect a subtree that imports another stage -- collection
# fails before markers are ever evaluated -- so a single pytest invocation can
# never mean "everything passed".
TEST_SUITES = [
    ('portable',    None),
    ('mapping',     'scMappingV3.12'),
    ('integration', 'scIntegrationV1.12'),
    ('annotation',  'scAnnotationV3.12'),
    ('markerid',    'scMarkersV3.12'),
    ('e2e',         'scMappingV3.12'),
]


def cmd_test(args) -> int:
    """Run each test directory in the conda env that stage needs.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        int: 0 when every suite passed, else 1.
    """
    root = _repo_root()
    suites = TEST_SUITES
    if args.suite:
        suites = [s for s in TEST_SUITES if s[0] == args.suite]
        if not suites:
            print('unknown suite {!r}; known: {}'.format(
                args.suite, ', '.join(s for s, _ in TEST_SUITES)), file=sys.stderr)
            return 2

    base = ['-q']
    if args.fast:
        base += ['-m', 'not slow']
    if args.verbose:
        base = [a for a in base if a != '-q'] + ['-v']

    results, worst = [], 0
    for name, env in suites:
        subtree = root / 'tests' / name
        if not subtree.is_dir():
            results.append((name, env, 'absent', 0)); continue

        argv = ['python', '-m', 'pytest', str(subtree)] + base
        if env and shutil.which('conda'):
            argv = ['conda', 'run', '-n', env, '--no-capture-output'] + argv

        print('\n=== tests/{}  (env: {}) ==='.format(name, env or CURRENT_ENV()), file=sys.stderr)
        code = subprocess.call(argv, cwd=str(root))
        # 5 is pytest's "no tests collected", which is not a failure here.
        ok = 'pass' if code in (0, 5) else 'FAIL'
        worst = worst if code in (0, 5) else 1
        results.append((name, env, ok, code))

    print('\n' + '=' * 58)
    for name, env, ok, code in results:
        print('  {:<12} {:<22} {}{}'.format(
            name, env or '(current)', ok, '' if code in (0, 5) else ' (exit {})'.format(code)))
    print('=' * 58)
    print('ALL PASSED' if worst == 0 else 'FAILURES ABOVE')
    return worst


def CURRENT_ENV():
    """The conda env this process is running in, or '' if none."""
    return os.environ.get('CONDA_DEFAULT_ENV', '<none>')


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """The `sc` argument parser.

    Returns:
        argparse.ArgumentParser: The parser.
    """
    p = argparse.ArgumentParser(
        prog='python -m sc',
        description='Front-end for the scRNAseq pipeline. Activates each stage\'s conda '
                    'environment and passes every other argument through to it unchanged.',
        epilog='Examples:\n'
               '  python -m sc test\n'
               '  python -m sc test --fast\n'
               '  python -m sc status\n'
               '  python -m sc status --sample SMu006\n'
               '  python -m sc status --targets\n'
               '  python -m sc map --cohort medicago\n'
               '  python -m sc integrate --icfg configs/integration_config.yaml '
               '--scfg configs/sample_metadata.yaml --target MtSZ --plot\n'
               '  python -m sc annotate --acfg configs/annotation_config.yaml --validate-only\n',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command')

    t = sub.add_parser('test', help='run the test suite, each subtree in its own env')
    t.add_argument('suite', nargs='?',
                   help='one of: {} (default: all)'.format(', '.join(n for n, _ in TEST_SUITES)))
    t.add_argument('--fast', action='store_true', help='skip tests marked slow')
    t.add_argument('--verbose', '-v', action='store_true')
    t.set_defaults(func=cmd_test)

    s = sub.add_parser('status', help='what has been produced so far')
    s.add_argument('-c', '--config', default='configs/pipeline_config.yaml')
    s.add_argument('--sample', help='detail for one sample, including its manifest')
    s.add_argument('--targets', action='store_true', help='integration targets instead')
    s.set_defaults(func=cmd_status)

    for name, (module, env) in STAGE_ENVS.items():
        sp = sub.add_parser(name, add_help=False,
                            help='{}  (env: {})'.format(module, env))
        sp.set_defaults(stage=name)
    return p


def main(argv=None) -> int:
    """CLI entry point: resolve the stage to its env and dispatch.

    Args:
        argv (list[str], optional): Argument vector.

    Returns:
        int: Process exit status.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help'):
        build_parser().print_help()
        return 0

    command = argv[0]
    if command in STAGE_ENVS:
        # Everything after the command belongs to the stage. Parsed here only
        # far enough to peel off --dry-run, so `sc` never has to know a stage's
        # flags and cannot drift out of sync with them.
        rest = argv[1:]
        dry = '--dry-run' in rest
        if dry:
            rest = [a for a in rest if a != '--dry-run']
        return _run_stage(command, rest, dry_run=dry)

    args = build_parser().parse_args(argv)
    if getattr(args, 'func', None):
        return args.func(args)
    build_parser().print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
