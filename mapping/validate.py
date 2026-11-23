#!/usr/bin/env python3

"""
Standalone config preflight.

    python -m mapping.validate -c configs/pipeline_config.yaml
    python -m mapping.validate -c configs/pipeline_config.yaml \
        --icfg configs/integration_config.yaml
    python -m mapping.validate --emit-schemas

`main()` runs the mapping half of this automatically at startup, so this
entrypoint exists for two things the pipeline run does not do: checking a
config tree without launching anything (useful from a login node before
queueing an sbatch), and validating the integration config, which belongs to a
stage with its own entrypoint.

Exit status is 1 if any error was found, 0 otherwise. Warnings never change the
exit status -- a missing index is a normal state before --build-indexes.
"""

import argparse
import sys

from pathlib import Path

from shared.validate import validate_mapping, validate_integration, emit_schemas


def main(argv=None) -> int:
    """CLI entry point: preflight the config tree, or emit the editor schemas."""
    parser = argparse.ArgumentParser(description='Validate the pipeline config tree')
    parser.add_argument('-c', '--config', help='Path to pipeline_config.yaml')
    parser.add_argument('--cohort', help='Validate the sheet this cohort selects '
                                        '(see `cohorts:` in the pipeline config).')
    parser.add_argument('--icfg', help='Also validate this integration_config.yaml')
    # Deliberately NOT defaulted from -c: pipeline_config.yaml often points
    # `samples:` at a dev subset while integration runs against the full sheet,
    # so deriving it yields spurious "sample not in the sample sheet" errors.
    parser.add_argument('--scfg', help='Sample sheet to validate --icfg against '
                                       '(usually configs/sample_metadata.yaml)')
    parser.add_argument('--emit-schemas', action='store_true',
                        help='Write configs/schemas/*.schema.json for the YAML language server')
    parser.add_argument('--schema-dir', default='configs/schemas',
                        help='Where --emit-schemas writes (default: configs/schemas)')
    args = parser.parse_args(argv)

    if args.emit_schemas:
        for path in emit_schemas(args.schema_dir):
            print('wrote {}'.format(path))
        if not args.config:
            return 0

    if not args.config and not args.icfg:
        parser.error('need -c/--config, --icfg, or --emit-schemas')

    failed = False

    if args.config:
        report = validate_mapping(args.config, cohort=args.cohort)
        print('mapping config ({}):'.format(args.config))
        print(report.render())
        failed = failed or not report.ok

    if args.icfg:
        if not args.scfg:
            parser.error(
                '--icfg needs --scfg naming the sample sheet the integration targets refer to '
                '(usually configs/sample_metadata.yaml). It is not taken from -c, because the '
                'mapping pipeline config often points at a dev subset.')
        report = validate_integration(args.icfg, args.scfg)
        print('\nintegration config ({} against {}):'.format(args.icfg, Path(args.scfg).name))
        print(report.render())
        failed = failed or not report.ok

    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
