#!/usr/bin/env python3
"""Check that a change to Python files touched only docstrings and comments.

    python tools/ast_docs_only.py [REV] [PATHSPEC ...]     # default REV: HEAD

For every tracked .py file that differs between REV and the working tree, the
AST is compared with docstrings removed and source positions ignored, and the
lines carrying `noqa`/`type:`/`pragma` directives are compared as text. Exit
status 1 on any difference.

Limitations:
- Comments are invisible to the AST, so any comment edit passes. That is the
  point, but it also means review of comment content is still manual.
- A docstring is only the first string statement of a module, class or
  function. Any other string literal (log messages, help=, error text) is code
  and is flagged.
- Docstrings that are read at runtime (`description=__doc__`, `help()`,
  `inspect.getsource` in tests) are NOT protected; find those consumers by hand.
- Untracked or newly added files and non-.py changes are listed, not checked.
- Directives are matched where a comment starts with `noqa`, `type:` or
  `pragma`; a directive later in a comment is treated as prose.
- Uses the running interpreter's parser; run it with the Python the code
  targets (3.12).
"""

import ast
import re
import subprocess
import sys


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return tree


def code_dump(source, name):
    return ast.dump(strip_docstrings(ast.parse(source, filename=name)),
                    include_attributes=False)


DIRECTIVE = re.compile(r'#\s*(?:noqa\b|type:|pragma\b).*')


def directives(source):
    return sorted(m.group(0) for line in source.splitlines()
                  for m in [DIRECTIVE.search(line)] if m)


def git(root, *args):
    return subprocess.run(['git', '-C', root, *args], capture_output=True,
                          text=True, check=True).stdout


def main(argv):
    rev, specs = (argv[0], argv[1:]) if argv else ('HEAD', [])
    root = git('.', 'rev-parse', '--show-toplevel').strip()
    changed = git(root, 'diff', '--name-only', rev, '--', *specs).split()
    untracked = git(root, 'ls-files', '--others', '--exclude-standard', '--', *specs).split()

    failures = 0
    for path in changed:
        if not path.endswith('.py'):
            print(f'NOT CHECKED  {path} (not Python)')
            continue
        try:
            new = open(f'{root}/{path}').read()
        except FileNotFoundError:
            print(f'FAIL  {path} (deleted)')
            failures += 1
            continue
        try:
            old = git(root, 'show', f'{rev}:{path}')
        except subprocess.CalledProcessError:
            print(f'NOT CHECKED  {path} (added since {rev})')
            continue
        same_code = code_dump(old, path) == code_dump(new, path)
        same_directives = directives(old) == directives(new)
        ok = same_code and same_directives
        failures += not ok
        print(f"{'OK  ' if ok else 'FAIL'}  {path}  code={'same' if same_code else 'DIFFERS'}"
              f"  directives={'same' if same_directives else 'DIFFER'}")
    for path in untracked:
        print(f'NOT CHECKED  {path} (untracked)')
    print(f'{len(changed)} changed file(s), {failures} failure(s)')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
