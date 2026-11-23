"""Release/setup changes must preserve scientific config and stage commands."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from shared.mapping_setup import launch_config, bd_geometry, check_bd_geometry
from shared import config_spec
from sc import __main__ as cli


def test_example_preserves_scientific_defaults(repo_root):
    def load(name):
        return yaml.safe_load((repo_root / 'configs' / name).read_text())
    original = load('pipeline_config.yaml')
    example = load('examples/minimal/pipeline.yaml')
    for name in ['cellbender', 'doublets']:
        assert {k: v for k, v in original[name].items() if k != 'env'} == {
            k: v for k, v in example[name].items() if k != 'env'}
    assert load('examples/minimal/filter.yaml') == load('filter_config.yaml')
    original_qc = load('qc_config.yaml')['QCconfigs']['default']
    example_qc = load('examples/minimal/qc.yaml')['QCconfigs']['default']
    assert original_qc['qc_metrics'] == example_qc['qc_metrics']
    index = load('index_config.yaml')
    assert load('examples/minimal/indexes.yaml')['indexes']['AFdefault']['extra_index_args'] == index['indexes']['AFdefault']['extra_index_args']
    assert load('examples/minimal/technologies.yaml')['technologies']['10xv3']['alevin']['chemistry'] == load('technology_config.yaml')['technologies']['10xv3']['alevin']['chemistry']
    assert config_spec.IMPLEMENTED_MAPPERS == ('alevin',)
    assert {'kb', 'starsolo'} <= set(config_spec.KNOWN_MAPPERS)


def test_selected_cohort_and_sheet_defaults_drive_bd_check(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'bd.yaml').write_text(yaml.safe_dump({
        'defaults': {'tech': 'bd_rhapsody', 'mapper': 'alevin'},
        'samples': {'S1': {}}}))
    (tmp_path / 'tenx.yaml').write_text(yaml.safe_dump({
        'defaults': {'tech': '10xv3', 'mapper': 'alevin'}, 'samples': {'S2': {}}}))
    (tmp_path / 'tech.yaml').write_text(yaml.safe_dump({'technologies': {
        'bd_rhapsody': {'alevin': {'chemistry': 'BD_GEOMETRY'}}}}))
    config = {'configs': {'samples': 'tenx.yaml', 'technologies': 'tech.yaml'},
              'default_cohort': 'tenx', 'cohorts': {
                  'tenx': {'samples': 'tenx.yaml'}, 'bd': {'samples': 'bd.yaml'}}}
    (tmp_path / 'pipeline.yaml').write_text(yaml.safe_dump(config))
    cfg, selected = launch_config(['--config', 'pipeline.yaml', '--cohort', 'bd', '--force-qc'])
    assert bd_geometry(cfg, selected) == (['S1'], 'BD_GEOMETRY')
    cfg, selected = launch_config(['-c', 'pipeline.yaml'])
    assert bd_geometry(cfg, selected) == ([], None)


@pytest.mark.parametrize('length,accept', [(27, True), (9, False), (0, False)])
def test_probe_checks_simpleaf_cached_binary(tmp_path, monkeypatch, length, accept):
    (tmp_path / 'simpleaf_info.json').write_text(json.dumps({
        'prog_info': {'piscem': {'exe_path': '/cached/piscem'}}}))
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, '', f'bc_len={length}')
    monkeypatch.setattr('shared.mapping_setup.subprocess.run', run)
    if accept:
        assert check_bd_geometry('GEOMETRY', tmp_path) == '/cached/piscem'
    else:
        with pytest.raises(ValueError, match='requires bc_len=27'):
            check_bd_geometry('GEOMETRY', tmp_path)
    assert calls[0][0] == '/cached/piscem'
    assert calls[0][calls[0].index('-g') + 1] == 'GEOMETRY'


@pytest.mark.parametrize('script,module', [('run_mapper.sh', 'mapping.main'),
                                           ('run_refilter.sh', 'mapping.refilter')])
@pytest.mark.parametrize('status', [0, 7])
def test_launchers_preserve_analysis_command_and_exit_status(repo_root, tmp_path, script, module, status):
    """Exercise the real shell launcher with fake tools; never submit or map."""
    import shutil
    bin_dir = tmp_path / 'bin'; bin_dir.mkdir()
    lib = tmp_path / 'batch_scripts/lib'; lib.mkdir(parents=True)
    shutil.copyfile(repo_root / 'batch_scripts/lib/mapping_setup.sh', lib / 'mapping_setup.sh')
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/pipeline_config.yaml').write_text('configs: {samples: unused.yaml}\ncohorts: {selected: {samples: unused.yaml}}\ndefault_cohort: selected\n')
    conda = tmp_path / 'conda/etc/profile.d'; conda.mkdir(parents=True)
    (conda / 'conda.sh').write_text('conda() { return 0; }\n')
    for binary in ['simpleaf', 'piscem', 'alevin-fry', 'nvidia-smi']:
        p = bin_dir / binary; p.write_text('#!/bin/sh\nexit 0\n'); p.chmod(0o755)
    p = bin_dir / 'python'
    p.write_text('#!' + sys.executable + '\nimport os, sys\n'
                 'if sys.argv[1:3] == ["-m", "shared.mapping_setup"]: sys.exit(0)\n'
                 'os.execv(sys.executable, [sys.executable] + sys.argv[1:])\n')
    p.chmod(0o755)
    p = bin_dir / 'srun'
    p.write_text('#!' + sys.executable + '\nimport json, os, sys\n'
                 'from pathlib import Path\n'
                 'Path(os.environ["TEST_COMMAND"]).write_text(json.dumps({"argv":sys.argv[1:],'
                 '"af_home":os.environ.get("ALEVIN_FRY_HOME"),"path":os.environ["PATH"]}))\n'
                 'sys.exit(int(os.environ["TEST_STATUS"]))\n')
    p.chmod(0o755)
    env = dict(os.environ, SCRNASEQ_ROOT=str(tmp_path), CONDA_ROOT=str(tmp_path / 'conda'),
               PATH=str(bin_dir) + ':' + os.environ['PATH'], PYTHONPATH=str(repo_root),
               TMPDIR=str(tmp_path / 'scratch'), TEST_COMMAND=str(tmp_path / 'command.json'),
               TEST_STATUS=str(status), CELLBENDER_ENV='/absent', DOUBLET_ENV='/absent')
    args = ['--cohort', 'selected', '--force-qc'] if module == 'mapping.main' else []
    result = subprocess.run(['bash', str(repo_root / 'batch_scripts' / script), *args],
                            env=env, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == status, result.stderr + result.stdout
    command = json.loads((tmp_path / 'command.json').read_text())
    prefix = ['python', '-m', module, '--config', str(tmp_path / 'configs/pipeline_config.yaml')]
    if module == 'mapping.main':
        assert command['argv'] == prefix + ['--build-indexes', *args]
        assert command['af_home'] == str(tmp_path / 'af_home')
        assert command['path'].split(':')[0] == str(tmp_path / 'tools/bin')
    else:
        assert command['argv'] == prefix + ['--sample', 'SMu007', 'SMu008', 'SMu009',
                                            'SMu010', 'SMu011', 'SMu012', '--allow-parent-mismatch']


@pytest.mark.parametrize('present,code,expected', [(False, 0, 1), (True, 5, 1), (True, 0, 0)])
def test_required_test_suite_cannot_pass_without_collection(tmp_path, monkeypatch, present, code, expected):
    if present:
        (tmp_path / 'tests/portable').mkdir(parents=True)
    monkeypatch.setattr(cli, '_repo_root', lambda: tmp_path)
    monkeypatch.setattr(cli.subprocess, 'call', lambda *a, **kw: code)
    assert cli.cmd_test(argparse.Namespace(suite='portable', fast=False, verbose=False)) == expected


def test_fast_e2e_exclusion_is_explicit(tmp_path, monkeypatch, capsys):
    (tmp_path / 'tests/e2e').mkdir(parents=True)
    monkeypatch.setattr(cli, '_repo_root', lambda: tmp_path)
    monkeypatch.setattr(cli.subprocess, 'call', lambda *a, **kw: pytest.fail('should be excluded'))
    assert cli.cmd_test(argparse.Namespace(suite='e2e', fast=True, verbose=False)) == 0
    assert 'excluded by --fast' in capsys.readouterr().out
