#!/usr/bin/env python3
"""Read-only checks of the public source tree; no data, GPU, or stage imports."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.validate import emit_schemas


def public_files(root):
    if (root / '.git').exists():
        result = subprocess.run(
            ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
            cwd=root, capture_output=True, check=True)
        return set(result.stdout.decode().rstrip('\0').split('\0'))
    # An exported source archive has no Git metadata or ignored private files.
    return {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
            and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}


def check(root=ROOT):
    files = public_files(root)
    required = {
        '.root', 'README.md', 'pyproject.toml', 'sc/__main__.py',
        'batch_scripts/run_mapper.sh', 'batch_scripts/lib/common.sh',
        'batch_scripts/lib/mapping_setup.sh', 'shared/mapping_setup.py',
        'docs/setup-mapping.md', 'docs/configuration.md', 'docs/outputs.md',
        'docs/testing.md', 'docs/releasing.md', 'docs/third-party.md',
        'annotation/po_final.obo', 'annotation/nodule.obo',
        'refine/embed_models.py', 'fetch/chemistry.py', 'fetch/overrides.py',
        'tools/piscem-bd/build.sh', '.github/workflows/portable.yml',
    }
    required.update('configs/examples/minimal/' + f + '.yaml'
                    for f in ['pipeline', 'samples', 'mapping', 'indexes',
                              'technologies', 'qc', 'filter'])
    errors = [f'missing public file: {name}' for name in sorted(required - files)]
    for name in sorted(files):
        path = root / name
        if not path.is_file():
            errors.append(f'missing file: {name}')
            continue
        if path.is_symlink():
            errors.append(f'public file is a symlink: {name}')
        if path.suffix == '.py':
            try:
                ast.parse(path.read_text(), filename=name)
            except SyntaxError as exc:
                errors.append(f'{name}: {exc}')
        elif path.suffix == '.sh':
            result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
            if result.returncode:
                errors.append(f'{name}: {result.stderr.strip()}')
        elif path.suffix == '.md':
            for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
                target = urlsplit(link.strip('<>'))
                if target.scheme or target.netloc or not target.path:
                    continue
                resolved = (path.parent / unquote(target.path)).resolve()
                try:
                    rel = resolved.relative_to(root).as_posix()
                except ValueError:
                    errors.append(f'{name}: link outside release: {link}')
                    continue
                if rel not in files and not any(f.startswith(rel.rstrip('/') + '/') for f in files):
                    errors.append(f'{name}: link missing from public files: {link}')
    with tempfile.TemporaryDirectory(prefix='scrnaseq-schema-check-') as temp:
        for generated in emit_schemas(temp):
            path = Path(generated)
            name = 'configs/schemas/' + path.name
            if name not in files or json.loads((root / name).read_text()) != json.loads(path.read_text()):
                errors.append(f'{name}: does not match shared/config_spec.py')
    return errors


if __name__ == '__main__':
    errors = check()
    for error in errors:
        print(error, file=sys.stderr)
    print(f'Release structure: {len(errors)} error(s). License selection remains separate.')
    sys.exit(bool(errors))
