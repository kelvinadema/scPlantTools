"""
`sc` -- one front-end for the whole pipeline.

Layer 2. It imports NO stage code: every stage runs in its own conda
environment, so `sc` dispatches with `conda run -n <env> python -m <stage>.main`
rather than importing anything it drives. That is what lets a single command
work across five mutually-incompatible environments, and it is why this package
does not appear in `tests/portable/test_layering.py`'s stage list.

The only pipeline module it touches is `shared/status.py`, which reads JSON off
disk and needs nothing installed.
"""

STAGE_ENVS = {
    'map':       ('mapping.main',     'scMappingV3.12'),
    'refilter':  ('mapping.refilter', 'scMappingV3.12'),
    'integrate': ('integration.main', 'scIntegrationV1.12'),
    'annotate':  ('annotation.main',  'scAnnotationV3.12'),
    'review':    ('annotation.review','scAnnotationV3.12'),
    'markers':   ('markerid.main',    'scMarkersV3.12'),
    'refine':    ('refine.main',      'scIntegrationV1.12'),
    'fetch':     ('fetch.main',       'scFetch'),
}
