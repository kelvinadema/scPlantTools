#!/usr/bin/env python3

"""
Comment-preserving write-back into configs/integration_config.yaml.

A --sample/--reference ad hoc run that trains a genuinely new model documents
itself here under its --name, so `--target NAME` can replay it later.

Deliberately NOT a full-file YAML round-trip. ruamel.yaml's round-trip mode
was tried first, but it still normalizes every existing True/False in the file
to lowercase true/false -- boolean scalars get re-serialized through its own
representer rather than preserved verbatim, confirmed empirically, and no
`yaml.version` setting avoids it. Instead this renders the new entry as a text
block matching the file's existing conventions (2-space indent under
IntegrateConfigs, 4-space under the new target, single-quoted strings,
True/False capitalization, inline flow-style lists) and splices it in as plain
text immediately before the first top-level key that follows
`IntegrateConfigs:` -- every other byte of the file is untouched.
"""

import re

from pathlib import Path


def _scalar(value) -> str:
    """Render one scalar as YAML, quoting only where needed.

    Args:
        value: The scalar.

    Returns:
        str: YAML text.
    """
    if isinstance(value, bool):
        return 'True' if value else 'False'
    if isinstance(value, str):
        return "'{}'".format(value)
    return str(value)


def _flow_list(items) -> str:
    """Render a list in YAML flow style.

    Args:
        items (iterable): The items.

    Returns:
        str: YAML text.
    """
    return '[' + ', '.join(_scalar(i) for i in items) + ']'


def _render_entry(name: str, entry: dict) -> list:
    """Render one target entry as YAML text.

    Written textually rather than via yaml.dump so the file's comments survive.

    Args:
        name (str): Target name.
        entry (dict): The target's keys.

    Returns:
        str: YAML text.
    """
    lines = ['  {}:'.format(name)]
    for key, value in entry.items():
        if isinstance(value, list):
            lines.append('    {}: {}'.format(key, _flow_list(value)))
        elif isinstance(value, dict):
            lines.append('    {}:'.format(key))
            for k2, v2 in value.items():
                if isinstance(v2, dict):
                    lines.append('      {}:'.format(k2))
                    for k3, v3 in v2.items():
                        lines.append('        {}: {}'.format(k3, _scalar(v3)))
                else:
                    lines.append('      {}: {}'.format(k2, _scalar(v2)))
        else:
            lines.append('    {}: {}'.format(key, _scalar(value)))
    return lines


def write_target_entry(config_path, name: str, entry: dict) -> None:
    """Append a target entry to integration_config.yaml.

    Args:
        config_path (Path): The config.
        name (str): Target name.
        entry (dict): The target's keys.
    """
    path = Path(config_path)
    text = path.read_text()

    if re.search(r'(?m)^  {}:'.format(re.escape(name)), text):
        raise ValueError(
            "'{}' already exists in IntegrateConfigs; write_target_entry is only "
            "for brand-new names (an existing name should be run via --target instead).".format(name))

    lines = text.split('\n')
    # Find the first top-level (column-0) key after IntegrateConfigs: itself --
    # that marks the end of the IntegrateConfigs mapping.
    insert_at = len(lines)
    seen_integrate_configs = False
    for i, line in enumerate(lines):
        if re.match(r'^IntegrateConfigs:\s*$', line):
            seen_integrate_configs = True
            continue
        if seen_integrate_configs and re.match(r'^\S', line):
            insert_at = i
            break

    block = _render_entry(name, entry)
    new_lines = lines[:insert_at] + block + [''] + lines[insert_at:]
    path.write_text('\n'.join(new_lines))
