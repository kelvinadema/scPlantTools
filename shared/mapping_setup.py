"""Resolve a mapping launch and verify the existing BD barcode requirement.

No analysis is performed. Run after simpleaf set-paths, from the repository
root. Unknown arguments belong to mapping.main and are left untouched.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from .cohorts import resolve_cohort
from .samples import load_samples
from .utils import load_config


def launch_config(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-c', '--config', default='configs/pipeline_config.yaml')
    parser.add_argument('--cohort')
    args, _ = parser.parse_known_args(argv)
    config = load_config(args.config)
    return config, resolve_cohort(config, args.cohort)


def bd_geometry(config, selected):
    samples = load_samples(selected['samples'])
    names = [name for name, entry in samples.items()
             if entry.get('tech') == 'bd_rhapsody']
    if not names:
        return names, None
    tech = load_config(config['configs']['technologies'])
    return names, tech['technologies']['bd_rhapsody']['alevin']['chemistry']


def check_bd_geometry(geometry, af_home):
    """Probe the executable cached by simpleaf, not a second PATH lookup."""
    info = json.loads((Path(af_home) / 'simpleaf_info.json').read_text())
    binary = info['prog_info']['piscem']['exe_path']
    with tempfile.TemporaryDirectory(prefix='scrnaseq-bcprobe-') as tmp:
        root = Path(tmp)
        reads = root / 'probe.fq'
        reads.write_text('@probe\nACGT\n+\nIIII\n')
        result = subprocess.run(
            [binary, 'map-sc', '-i', str(root / 'no_such_index'),
             '-g', geometry, '-1', str(reads), '-2', str(reads),
             '-o', str(root / 'out')],
            capture_output=True, text=True, timeout=30)
    # The missing index is intentional; geometry is reported before loading it.
    match = re.search(r'bc_len=(\d+)', result.stdout + result.stderr)
    observed = match.group(1) if match else '<none>'
    if observed != '27':
        raise ValueError(
            f'piscem cached by simpleaf ({binary}) reports bc_len={observed}; '
            'BD Rhapsody requires bc_len=27. Rebuild with '
            'bash tools/piscem-bd/build.sh, then repeat mapping setup.')
    return binary


def main(argv=None):
    try:
        config, selected = launch_config(argv)
        names, geometry = bd_geometry(config, selected)
        if names:
            binary = check_bd_geometry(geometry, os.environ['ALEVIN_FRY_HOME'])
            print(f'BD Rhapsody preflight OK: bc_len=27 for {names}; {binary}')
        else:
            print('BD Rhapsody preflight: no BD samples in the selected sheet.')
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        print(f'Mapping setup failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
