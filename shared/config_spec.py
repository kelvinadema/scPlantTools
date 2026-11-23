#!/usr/bin/env python3

"""
Declarative description of the config files: one source of truth, two consumers.

`shared/validate.py` walks these specs for its structural pass, and
`--emit-schemas` renders them to JSON Schema for the editor. Writing the
structure once means the red squiggle in VSCode and the preflight error from
the pipeline cannot disagree about what a valid config looks like.

Deliberately dependency-free -- the structural checks are plain Python rather
than `jsonschema` -- so validation runs in any environment, outside the stage
envs, and cannot itself fail on a missing package.

Coverage is uneven on purpose:

  full          sample sheets, index_config, mapping_config, pipeline_config,
                technology_config, integration_config -- the files that are
                edited routinely and whose mistakes are expensive
  structure     qc_config, filter_config -- top-level shape and profile names
                only, permissive within. Their parameters are statistical and
                their inline comments are the real documentation; a
                half-understood enum here would reject valid science.

A note on `required`: these specs describe a config file *as written on disk*.
A sample entry may legitimately omit `mapper` because the sheet's `defaults:`
block supplies it, so nothing in SAMPLE_ENTRY is marked required. Keys that
must exist once defaults have been folded in are listed separately in
REQUIRED_RESOLVED_SAMPLE_KEYS and checked by validate.py against resolved
entries.
"""


KNOWN_MAPPERS = ('alevin', 'kb', 'starsolo')
# kb and starsolo are wired into the config indirection but not implemented
# end-to-end. Kept here rather than in mapping.main so the validator and
# the entrypoint gate cannot drift apart.
IMPLEMENTED_MAPPERS = ('alevin',)
SAMPLE_TYPES = ('protoplast', 'nuclei')
INTEGRATION_METHODS = ('scVI', 'scArches', 'contrastiveVI')
# Selectable scVI model classes. Mirrors integrationutils.MODEL_CLASSES, which
# cannot be imported here: it binds the actual scvi classes, and this module is
# stdlib-only so the validator runs outside the integration env.
#
# DRVI is a model_class rather than a method because it subclasses scvi.model.SCVI
# (setup_anndata, train() and ArchesMixin all inherited unchanged), exactly like
# JointEmbeddingSCVI. contrastiveVI is a *method* instead, because it subclasses
# BaseModelClass only and its train() takes two required positional index lists --
# SingleCellIntegrate's `model.train(**train_args)` cannot drive it.
#
# SysVI is admitted despite NOT subclassing SCVI. It meets the two requirements
# that actually matter -- the uniform setup/init/train path and ArchesMixin --
# and integrate.py handles the four places it departs from the scVI contract
# explicitly: a Gaussian reconstruction likelihood (so log-normalized input,
# not counts), no labels_key, a REQUIRED batch_key, and a train() that resets
# KL warmup to 0. Each of those is silent if unhandled, which is why they are
# named sets there rather than isinstance checks.
INTEGRATION_MODEL_CLASSES = ('SCVI', 'JointEmbeddingSCVI', 'DRVI', 'SysVI')

# Per-filter execution modes, set as `mode:` on each block in
# filter_config.yaml. Replaced a boolean `enabled:`:
#
#   'filter'    run it and remove the cells it flags      (was enabled: true)
#   'advisory'  run it, record everything, remove nothing (no boolean equivalent)
#   'off'       do not run it                            (was enabled: false)
#
# 'advisory' is what lets a method produce a diagnostic without making the
# decision, so the interactive QC report can offer its scores for a human to
# threshold by hand. Deliberately one key with three values rather than
# `enabled` plus a separate `mode`, so the contradictory `enabled: false` +
# `mode: advisory` cannot be expressed at all.
#
# Lives here rather than in mapping.qc_filter because both the
# validator (shared/) and the filter itself need it, and shared/ must not
# import from a stage package.
FILTER_MODES = ('off', 'advisory', 'filter')

# Checked against entries that have already been through shared.samples, i.e.
# after the sheet's `defaults:` block has been applied.
REQUIRED_RESOLVED_SAMPLE_KEYS = ('mapper', 'tech', 'fastq1', 'fastq2')
# Absent-but-not-fatal: each has a working fallback in code, but omitting them
# is more often an oversight than a decision.
ADVISORY_RESOLVED_SAMPLE_KEYS = ('sample_type', 'batch')


_PY_TYPES = {
    'string': str,
    'integer': int,
    'number': (int, float),
    'boolean': bool,
    'array': list,
    'object': dict,
}


class Field:
    """
    One key's expected shape.

    types       tuple of JSON-Schema type names this key may take
    required    key must be present in the object as written on disk
    enum        allowed scalar values
    doc         one line; becomes the hover text in the editor
    items       Field describing each element, for arrays
    properties  {name: Field} for objects with a fixed key set
    values      Field describing every value, for objects with arbitrary keys
                (a sample sheet's `samples:`, an index config's `indexes:`)
    additional  whether keys outside `properties` are allowed
    """

    __slots__ = ('types', 'required', 'enum', 'doc', 'items', 'properties', 'values', 'additional')

    def __init__(self, *types, required=False, enum=None, doc='', items=None,
                 properties=None, values=None, additional=True):
        self.types = types
        self.required = required
        self.enum = enum
        self.doc = doc
        self.items = items
        self.properties = properties
        self.values = values
        self.additional = additional


def _type_name(value):
    """The readable type name for a value, used in the schema and error text."""
    for name, py in _PY_TYPES.items():
        if name == 'number':
            continue
        if isinstance(value, py) and not (name != 'boolean' and isinstance(value, bool)):
            return name
    return type(value).__name__


def check(field: Field, value, path: str) -> list:
    """Validate `value` against `field`. Returns a list of human-readable errors."""
    errors = []

    if value is None:
        return errors  # an explicitly-null key counts as unset; see shared.samples

    if field.types:
        allowed = tuple(
            t for name in field.types
            for t in (_PY_TYPES[name] if isinstance(_PY_TYPES[name], tuple) else (_PY_TYPES[name],)))
        # bool is a subclass of int; never let it satisfy integer/number.
        is_bool = isinstance(value, bool)
        wants_bool = 'boolean' in field.types
        if is_bool and not wants_bool:
            errors.append('{}: expected {}, got boolean'.format(path, ' or '.join(field.types)))
            return errors
        if not isinstance(value, allowed):
            errors.append('{}: expected {}, got {}'.format(
                path, ' or '.join(field.types), _type_name(value)))
            return errors

    if field.enum is not None and value not in field.enum:
        errors.append('{}: {!r} is not one of {}'.format(
            path, value, ', '.join(repr(e) for e in field.enum)))

    if isinstance(value, list) and field.items is not None:
        for i, item in enumerate(value):
            errors.extend(check(field.items, item, '{}[{}]'.format(path, i)))

    if isinstance(value, dict):
        if field.properties:
            for key, sub in field.properties.items():
                if sub.required and key not in value:
                    errors.append('{}: missing required key {!r}'.format(path, key))
            for key, item in value.items():
                if key in field.properties:
                    errors.extend(check(field.properties[key], item, '{}.{}'.format(path, key)))
                elif not field.additional:
                    errors.append('{}: unknown key {!r} (allowed: {})'.format(
                        path, key, ', '.join(sorted(field.properties))))
        if field.values is not None:
            for key, item in value.items():
                errors.extend(check(field.values, item, '{}.{}'.format(path, key)))

    return errors


def to_json_schema(field: Field, title: str = '') -> dict:
    """Render a Field tree as a JSON Schema document for the YAML language server."""
    schema = {}
    if title:
        schema['$schema'] = 'http://json-schema.org/draft-07/schema#'
        schema['title'] = title
    if field.types:
        schema['type'] = list(field.types) if len(field.types) > 1 else field.types[0]
    if field.doc:
        schema['description'] = field.doc
    if field.enum is not None:
        schema['enum'] = list(field.enum)
    if field.items is not None:
        schema['items'] = to_json_schema(field.items)
    if field.properties:
        schema['properties'] = {k: to_json_schema(v) for k, v in field.properties.items()}
        required = sorted(k for k, v in field.properties.items() if v.required)
        if required:
            schema['required'] = required
        if not field.additional:
            schema['additionalProperties'] = False
    if field.values is not None:
        schema['additionalProperties'] = to_json_schema(field.values)
    return schema


# ---------------------------------------------------------------------------
# Sample sheets: sample_metadata.yaml, sample_subset.yaml, MarcantiaSample.yaml
# ---------------------------------------------------------------------------

_PATH_LIST = Field('array', items=Field('string'), doc='Absolute paths.')

SAMPLE_ENTRY = Field(
    'object',
    doc='One sample. Any key omitted here falls back to the sheet-level `defaults:` block.',
    properties={
        'mapper': Field('string', enum=KNOWN_MAPPERS,
                        doc='Mapper to use. Only "alevin" is implemented end-to-end; '
                            '"kb" and "starsolo" are gated at startup.'),
        'mapping_config': Field('string',
                                doc='Profile under mappers.<mapper> in mapping_config.yaml. '
                                    'Selects the index; this is how a second species is wired in.'),
        'index_ref': Field('string',
                           doc='Optional. Names an entry in index_config.yaml directly, '
                               'bypassing the mapper profile. For one-offs that do not '
                               'warrant their own profile.'),
        'tech': Field('string',
                      doc='Chemistry key into technology_config.yaml (e.g. 10xv3, bd_rhapsody).'),
        'sample_type': Field('string', enum=SAMPLE_TYPES,
                             doc='Drives the ambiguous-read fallback in preprocess.py and '
                                 'sample-type-keyed thresholds in filter_config.yaml.'),
        # The three below are stamped onto .obs by integration/
        # orchestrator.py during concatenation, alongside tech/sample_type.
        # They are PARTIAL -- stamped only where declared -- which validate.py
        # checks before allowing one to be named as a batch_key.
        'prep': Field('string',
                      doc='Protoplasting protocol. Two values: "1h_2h_mix" (a 1 h + 2 h '
                          'protoplast mix) and "2h_only". Corrected 2026-08-01 -- the old '
                          '"1h_post_protoplasting"/"standard" vocabulary was wrong in both '
                          'directions: it implied SMu005/006/008 differed from the rest of the '
                          'SMu series (they do not -- ALL EIGHT SMu samples are 1h_2h_mix), and '
                          '"standard" hid that BDMock/BDNod are the minority protocol within BD. '
                          'Every 10x protoplast sample and both BD tip samples are 2h_only. '
                          'Legal as a batch_key only where it is CROSSED with condition, which '
                          'inside BD it now is (mock and NF each appear at both levels) but only '
                          'because SMu005/006/008 are in `induced_bd` -- with the 7-sample group '
                          'it was a perfect proxy for rhizobia-vs-control and correcting on it '
                          'would have deleted the infection axis. Omit where a cohort has only '
                          'one variant.'),
        'treatment': Field('string',
                           doc='Experimental treatment applied, as a value rather than a substring '
                               'of `batch`: mock, NF, BAP, S_meliloti, none. Stamped onto .obs so '
                               'results can be grouped by it. NEVER a batch_key -- correcting on '
                               'it deletes the effect being measured, and validate.py errors.'),
        'time_h': Field('number',
                        doc='Hours since treatment applied. NF/BAP at 3 hpt and S. meliloti at '
                            '12-96 hpi share this numeric axis while being different timelines, so '
                            'it is interpretable only together with `treatment` -- never order '
                            'cells on time_h across treatments. Omitted entirely on untreated '
                            'samples. Mock is 0 by convention; under the staggered-inoculation / '
                            'common-harvest design it was harvested alongside every timepoint and '
                            'is a control for all of them, not a t=0 baseline. NEVER a batch_key.'),
        'fallback': Field('string',
                          doc='NOT READ BY ANY CODE PATH. The ambiguous-read fallback chain '
                              '(per-cell, per-gene, global, sample_type) is hardcoded in '
                              'preprocess.py and keyed off sample_type. Retained because it is '
                              'written on every sample; safe to delete.'),
        'qc_filter': Field('string',
                           doc='Profile under QCfilterconfigs in filter_config.yaml.'),
        'qc_metrics': Field('string',
                            doc='Profile under QCconfigs in qc_config.yaml. Defaults to "default".'),
        'batch': Field('string',
                       doc='Batch label carried into integration. Falls back to the sample name.'),
        'fastq1': Field('array', items=Field('string'), doc='R1 FASTQ paths.'),
        'fastq2': Field('array', items=Field('string'),
                        doc='R2 FASTQ paths; must be the same length as fastq1.'),
    })

SAMPLE_SHEET = Field(
    'object',
    properties={
        'defaults': Field('object', doc='Applied to every sample below. Exactly equivalent to '
                                        'writing each key on every sample entry; a sample that '
                                        'states the key wins.',
                          properties=SAMPLE_ENTRY.properties),
        'samples': Field('object', required=True, values=SAMPLE_ENTRY,
                         doc='Mapping of sample name to sample entry.'),
    },
    additional=False)


# ---------------------------------------------------------------------------
# mapping_config.yaml
# ---------------------------------------------------------------------------

MAPPING_PROFILE = Field(
    'object',
    doc='A named mapper profile. Merged into the sample entry, sample wins.',
    properties={
        'index_ref': Field('string', required=True,
                           doc='Entry in index_config.yaml this profile maps against.'),
        'extra_mapping_args': Field('array', items=Field('string', 'integer'),
                                    doc='Passed through to the mapper verbatim.'),
    })

MAPPING_CONFIG = Field(
    'object',
    properties={
        'mappers': Field('object', required=True,
                         values=Field('object', values=MAPPING_PROFILE),
                         doc='mappers.<mapper>.<profile> -> profile body.'),
    },
    additional=False)


# ---------------------------------------------------------------------------
# index_config.yaml
# ---------------------------------------------------------------------------

INDEX_ENTRY = Field(
    'object',
    properties={
        'mapper': Field('string', required=True, enum=KNOWN_MAPPERS,
                        doc='Which mapper this index is built for. Must match the mapper of '
                            'every sample that references it.'),
        'idx_path': Field('string', required=True, doc='Absolute path to the built index.'),
        'genome': Field('array', required=True, items=Field('string'),
                        doc='FASTA path(s). More than one triggers a multi-reference merge.'),
        'gtf': Field('array', required=True, items=Field('string'),
                     doc='GTF path(s); must be the same length as genome.'),
        'extra_index_args': Field('array', items=Field('string', 'integer'),
                                  doc='Passed through to the index builder verbatim.'),
    })

INDEX_CONFIG = Field(
    'object',
    properties={'indexes': Field('object', required=True, values=INDEX_ENTRY)},
    additional=False)


# ---------------------------------------------------------------------------
# technology_config.yaml
# ---------------------------------------------------------------------------

TECH_MAPPER_ENTRY = Field(
    'object',
    doc='Per-mapper settings for one chemistry. Merged into the sample entry, sample wins.',
    properties={
        'chemistry': Field('string', doc='alevin geometry string.'),
        'x_flag': Field('string', doc='kb -x technology flag.'),
        'solo_type': Field('string', doc='STARsolo --soloType.'),
        'soloCBwhitelist': Field('string', doc='STARsolo whitelist path(s).'),
        'unsupported': Field('boolean', doc='Set true to reject this mapper/chemistry pair.'),
        'extra_mapping_args': Field('array', items=Field('string', 'integer')),
    })

TECHNOLOGY_CONFIG = Field(
    'object',
    properties={
        'technologies': Field('object', required=True,
                              values=Field('object', values=TECH_MAPPER_ENTRY),
                              doc='technologies.<tech>.<mapper> -> settings.'),
    },
    additional=False)


# ---------------------------------------------------------------------------
# pipeline_config.yaml
# ---------------------------------------------------------------------------

PIPELINE_CONFIG = Field(
    'object',
    properties={
        'cohorts': Field('object', additional=True,
                         doc='Named datasets this pipeline config can run on, so the command '
                             'line says which one a run uses instead of it depending on a '
                             'pointer inside this file. Each value is a mapping with a required '
                             '`samples:` (sample sheet path) and an optional `output_dir:`. '
                             'Select with --cohort; `default_cohort:` is used when the flag is '
                             'omitted. Omit this block entirely to keep the old behaviour, where '
                             '`configs.samples` alone selects the dataset.'),
        'default_cohort': Field('string',
                                doc='Cohort used when --cohort is not given. Without it, a '
                                    'config that declares cohorts REQUIRES the flag -- '
                                    'deliberately, so nothing silently picks a dataset.'),
        'configs': Field('object', required=True, additional=False,
                         doc='Paths to the sub-configs this stage reads.',
                         properties={
                             'samples': Field('string',
                                              doc='Sample sheet. Optional only when `cohorts:` '
                                                  'is declared, in which case the selected '
                                                  'cohort supplies it.'),
                             'mapping': Field('string', required=True),
                             'indexes': Field('string', required=True),
                             'technologies': Field('string', required=True),
                             'qc': Field('string', required=True),
                             'filter': Field('string', required=True),
                         }),
        'output_dir': Field('string', required=True,
                           doc='Mapping stage output root. No default -- a site-specific data '
                               'location with no sensible fallback (main.py raises without it).'),
        'log_dir': Field('string', doc='Defaults to "logs", relative to cwd.'),
        'tmp_dir': Field('string',
                         doc='Scratch root for index builds. Falls back to $TMPDIR then the '
                             'system temp dir. piscem index builds are unreliable on Lustre.'),
        'cellbender': Field('object', doc='CellBender runs from its own conda env as a subprocess.',
                            properties={
                                'env': Field('string',
                                             doc='Conda PREFIX (the directory containing bin/), '
                                                 'not the binary. Overridable with $CELLBENDER_ENV.'),
                                'use_cuda': Field('boolean', doc='Passes --cuda; needs a GPU node.'),
                                'learning_rate': Field('number', 'string'),
                                'extra_args': Field('array', items=Field('string', 'integer')),
                            }),
        'doublets': Field('object',
                          doc='Per-sample doublet detection (scvi-tools SOLO). Runs from its own '
                              'conda env as a subprocess, for the same reason CellBender does: '
                              'scvi-tools does not support the pandas 3.x the mapping stage is on. '
                              'Whether its calls remove cells is set by `solo.mode` in '
                              'filter_config.yaml, not here.',
                          properties={
                              'env': Field('string',
                                           doc='Conda PREFIX (the directory containing bin/), not '
                                               'the binary. Overridable with $DOUBLET_ENV. Must '
                                               'have scvi-tools installed.'),
                              'use_cuda': Field('boolean',
                                                doc='Train on GPU. Mapping already needs a GPU node '
                                                    'for CellBender, so this costs no extra '
                                                    'infrastructure.'),
                              'max_epochs': Field('integer',
                                                  doc='scVI training epochs. Left unset, scvi-tools '
                                                      'picks a cell-count-dependent default.'),
                              'n_epochs_kl_warmup': Field('integer',
                                                          doc='KL warmup CEILING in epochs for the '
                                                              'throwaway scVI model (into '
                                                              'plan_kwargs). SingleCellIntegrate '
                                                              'scales warmup to half the effective '
                                                              'max_epochs and clamps this value down '
                                                              'if it would not complete, so warmup '
                                                              'always finishes; left unset it is '
                                                              'derived the same way. scvi\'s own '
                                                              'default (400) never completes on '
                                                              'these small models and drove the '
                                                              '100%-doublet failure (scvi-tools '
                                                              '#507). Applies to the final fit and, '
                                                              'when tune is on, the tuning trials.'),
                              'n_latent': Field('integer', doc='scVI latent dimensions.'),
                              'n_hidden': Field('integer', doc='scVI hidden units per layer.'),
                              'hvg_n_top_genes': Field('integer',
                                                       doc='HVGs the model is trained on. Selected '
                                                           'on the mapping side with scanpy so the '
                                                           'temp exchange file stays small.'),
                              'layer': Field('string',
                                             doc='Layer the model trains on. scVI\'s likelihood is '
                                                 'defined on counts, so "raw_counts" (integers) '
                                                 'rather than CellBender-corrected floats.'),
                              'n_layers': Field('integer'),
                              'random_state': Field('integer'),
                              'min_cells': Field('integer',
                                                 doc='Fewest QC-passed cells SOLO will run on '
                                                     '(default 200). A technical floor, not a '
                                                     'quality judgement, so falling below it is a '
                                                     'recorded skip rather than the advisory '
                                                     'downgrade sample_quality applies. Below ~10 '
                                                     'cells scvi-tools has no validation split at '
                                                     'all and autotune dies in ray; well above '
                                                     'that the model still fails to converge and '
                                                     'calls ~100% of cells doublets.'),
                              'max_plausible_rate': Field('number',
                                                          doc='Doublet rate above which the fitted '
                                                              'model is judged unconverged '
                                                              '(default 0.5): its calls are '
                                                              'downgraded to advisory and flagged '
                                                              'in the report. Real 10x rates run '
                                                              '~1-30%.'),
                              'tune': Field('boolean',
                                            doc='Autotune hyperparameters. Runs PER SAMPLE, so a '
                                                '40-sample cohort means 40 tuning jobs; off by '
                                                'default.'),
                              'n_trials': Field('integer', doc='Autotune trials per sample.'),
                              'tune_metric': Field('string'),
                              'timeout': Field('integer',
                                               doc='Seconds before the worker subprocess is killed.'),
                              'extra_args': Field('array', items=Field('string', 'integer')),
                              'scrublet': Field('object',
                                                doc='Scrublet, the second per-sample doublet caller '
                                                    '(scanpy). Runs in-process (CPU, no torch/GPU/'
                                                    'subprocess) so it also covers samples SOLO '
                                                    'skips; its calls form the SOLO+Scrublet '
                                                    'consensus. Whether each of SOLO/Scrublet/'
                                                    'consensus removes cells is set per-method by '
                                                    'mode: in filter_config.yaml, not here.',
                                                properties={
                                                    'expected_doublet_rate': Field('number',
                                                        doc='Scrublet\'s prior on the simulated '
                                                            'doublet fraction (default 0.06).'),
                                                    'sim_doublet_ratio': Field('number',
                                                        doc='Simulated doublets per observed cell '
                                                            '(default 2.0).'),
                                                    'n_prin_comps': Field('integer',
                                                        doc='PCA components for Scrublet\'s kNN. '
                                                            'Auto-reduced below the cell count on '
                                                            'small samples.'),
                                                    'layer': Field('string',
                                                        doc='Counts layer Scrublet simulates from. '
                                                            'Unset follows doublets.layer. All '
                                                            'genes are used -- Scrublet does its own '
                                                            'gene-variability selection.'),
                                                    'min_cells': Field('integer',
                                                        doc='Fewest QC-passed cells Scrublet runs '
                                                            'on (default 50). Below it is a recorded '
                                                            'skip; lower than SOLO\'s floor since '
                                                            'Scrublet has no validation split.'),
                                                }),
                          }),
    })


# ---------------------------------------------------------------------------
# integration_config.yaml
# ---------------------------------------------------------------------------

INTEGRATION_TARGET = Field(
    'object',
    doc='One integration target. Any key omitted falls back to the top-level `defaults:` block. '
        'Sample lists may name a group from `groups:` instead of listing samples.',
    properties={
        'method': Field('string', enum=INTEGRATION_METHODS,
                        doc='scVI trains a new model; scArches maps a query onto a reference model; '
                            'contrastiveVI fits one Background-vs-Target contrast. '
                            'scANVI is not selectable here -- it is a refinement applied on top of '
                            'a trained scVI model. DRVI is not a method: it is a model_class on the '
                            'scVI path, since it subclasses SCVI.'),
        'Samples': Field('array', 'string', items=Field('string'),
                         doc='scVI targets. A group name, or a list of sample/group names.'),
        'Reference': Field('array', 'string', items=Field('string'),
                           doc='scArches targets: samples the reference model was trained on.'),
        'Query': Field('array', 'string', items=Field('string'),
                       doc='scArches targets: samples to map onto the reference.'),
        'Reference_model': Field('string',
                                 doc='Name of the scVI target whose trained model to map onto.'),
        'Background': Field('array', 'string', items=Field('string'),
                            doc='contrastiveVI targets: the control samples defining shared '
                                'variation. Their salient latent is fixed at 0 in the generative '
                                'model. Must be disjoint from Target.'),
        'Target': Field('array', 'string', items=Field('string'),
                        doc='contrastiveVI targets: the perturbed samples whose target-specific '
                            'variation the salient latent should isolate. Must be disjoint from '
                            'Background.'),
        'covariates': Field('array', 'string', 'object', items=Field('string'),
                            properties={
                                'categorical': Field('array', items=Field('string'),
                                                     doc='.obs columns registered as CATEGORICAL '
                                                         'covariates.'),
                                'continuous': Field('array', items=Field('string'),
                                                    doc='.obs columns registered as CONTINUOUS '
                                                        'covariates.')},
                            doc='.obs columns passed to scVI as covariates. A bare LIST means '
                                'all-continuous, which is what almost every target wants and why '
                                'it is the common form. The OBJECT form -- {categorical: [...], '
                                'continuous: [...]} -- has been supported by '
                                'targets.py::parse_covariates all along but was rejected by this '
                                'schema until 2026-08-17, so a target needing a categorical '
                                'covariate could not be written. SysVI targets need it: their '
                                'batch_key is the SYSTEM (sample_type) and the per-sample batch '
                                'goes here as categorical instead. A name that is not present in '
                                '.obs is silently dropped at integrate.py:79 and the model trains '
                                'without it -- validate.py checks these.'),
        'batch_key': Field('string', 'array', 'object',
                           doc="The .obs column scVI corrects on. Unstated means 'batch' (the "
                               'sample sheet\'s own batch string, backfilled by resolve_batch); '
                               'an explicit null trains with no batch correction at all. Stated '
                               'here rather than assumed because `batch` is one-to-one with '
                               'condition on a treated cohort, where correcting on it deletes the '
                               'effect being measured. `tech` (10xv3/bd_rhapsody) is the useful '
                               'alternative wherever platform is crossed with condition rather '
                               'than confounded with it. validate.py checks the column exists. '
                               'THREE FORMS beyond the plain string, all resolved by '
                               'shared/targets.py::parse_batch_key and materialised onto .obs by '
                               'orchestrator._apply_batch_key: a LIST (["tech","prep"]) derives '
                               'one column whose levels are the joined values, for when two '
                               'nuisance variables are each crossed with condition but neither '
                               'alone identifies the grouping; a MAPPING ({tip: sample, '
                               'induced_all: [tech, prep]}) applies different keys to different '
                               'parts of the cohort, for joint reference+query fits where '
                               'per-sample correction is right on the untreated reference and '
                               'ruinous on the treated query; and null. Every key named in any '
                               'form faces the same checks a lone batch_key does -- pairing '
                               '`treatment` with a legal key does not make it legal.'),
        'sweep': Field('boolean',
                       doc='contrastiveVI only. Choose `wasserstein_penalty` and '
                           '`n_salient_latent` by fitting a grid of DISPOSABLE models and '
                           'ranking them, then train the winner as this target\'s real model. '
                           'This is what `tune: true` would be if it worked here -- it does '
                           'not, because contrastiveVI\'s objective has no term rewarding '
                           'salient/background separation, so elbo_validation selects maximum '
                           'salient capacity and zero penalty (the leak). Every criterion the '
                           'sweep ranks on is computed outside the objective: separation, '
                           'identity leakage measured on BACKGROUND cells (which received no '
                           'perturbation, so any structure there is leakage by construction), '
                           'and salient effective dimensionality, which catches the opposite '
                           'degeneracy where a too-large penalty collapses the salient space '
                           'into two blobs. The full grid is written to '
                           '{target}_contrast_grid.tsv and the chosen point recorded in the '
                           'registry, so an automatic selection stays auditable.'),
        'sweep_grid': Field('object',
                            doc='Optional axes for `sweep`, e.g. '
                                '{wasserstein_penalty: [0, 0.1, 0.5], n_salient_latent: [10]}. '
                                'Unstated uses integration/cvi_sweep.py\'s defaults, '
                                'which after the 2026-08-03 CH5 sweep are a FINE LADDER UNDER 1 '
                                '(0, 0.05, 0.1, 0.25, 0.5, 1): every penalty >= 1 collapsed the '
                                'salient space on all five contrasts, while 0 leaked cell '
                                'identity, and the transition between the two is a step rather '
                                'than a slope.'),
        'keep_layers': Field('array', 'string', items=Field('string'),
                             doc='Count layers to carry into the concatenated target; everything '
                                 'else is dropped per sample before ad.concat. Unstated keeps all '
                                 'eight the mapping stage writes, which means nine same-shape '
                                 'sparse matrices (with X) held in RAM and written to disk while '
                                 'the model reads one -- the usual cause of an OOM on a large '
                                 "cohort. ['corrected_counts'] is the safe minimum: it is what "
                                 'integrate/contrastive read and what every annotation and refine '
                                 'config names as counts_layer. Mapping artifacts are not modified, '
                                 'so a dropped layer can always be re-derived from them.'),
        'min_cells': Field('integer',
                           doc='Drop genes detected in fewer than this many cells, applied ONCE to '
                               'the concatenated target (orchestrator._concatenate_adata) and so to '
                               'every method. Unset means no gene filtering -- the historical '
                               'behaviour, and the default precisely because this runs for every '
                               'target: a value here would change the input to every existing fit. '
                               'Counted on the layer the model trains on (keep_layers[0], normally '
                               'corrected_counts), not on .X, since those differ after ad.concat. '
                               'Exists for the do_hvg: False case, where nothing else filters genes '
                               "at all -- the min_cells=10 in hvg.py's initialize_views lives INSIDE "
                               'HVGSelector and is skipped along with the selection. On a 265k-cell '
                               'cohort a floor of 10 is 0.004% of cells: it removes genes carrying '
                               'no information rather than genes carrying rare-population signal, '
                               'which is what makes it a safe substitute for a variance ranking.'),
        # Four label-space keys. All default to the single-layer behaviour, so a
        # target that states none of them is unaffected. See
        # docs/decisions/two-layer-label-spaces.md.
        'derive_corrected_layers': Field(
            'boolean',
            doc='Derive ambient-corrected spliced/unspliced layers per sample before '
                'concatenation: corrected_rspliced = floor(corrected_counts * rspliced / '
                '(rspliced + runspliced)) and corrected_runspliced = corrected_counts - '
                'corrected_rspliced, per gene per cell, so the two sum exactly to '
                'corrected_counts. rspliced + runspliced == 0 with corrected_counts > 0 is '
                'an error (CellBender removes counts, never adds). Reads rspliced/runspliced '
                'off the mapping artifact BEFORE keep_layers drops anything, and the two '
                'derived layers are kept automatically. Second-order exact: CellBender '
                'removed 1-7% of counts per cell here, so the misallocation is bounded by '
                'that fraction. Default False.'),
        'layer': Field(
            'string', 'array', items=Field('string'),
            doc='Count layer the scVI-path model trains on. Unstated means corrected_counts, '
                'as it always was. A LIST OF TWO layers trains on a 2 x genes feature matrix '
                '-- every gene contributes one feature per layer, named <gene>__s and '
                '<gene>__u -- so the model sees spliced and unspliced counts as separate '
                'inputs rather than their sum, and HVG selection runs on that feature space. '
                'The fitted latent is written back onto the ordinary gene-space object, with '
                '.var[hvg_spliced] / .var[hvg_unspliced] marking which features trained and '
                'the derivation in .uns[labelspace]. SysVI derives its log-normalised input '
                'from whatever matrix it is given, so it works on the doubled matrix unchanged.'),
        'hvg_batch_key': Field(
            'string',
            doc='The .obs column HVG rankings are aggregated over when do_hvg is on '
                '(shared/hvg.py::select_hvg(batch_key=...)). Unstated means batch, as it '
                'always was. Distinct from batch_key on purpose: the model may correct per '
                'library while the gene set is balanced per modality (sample_type), so the '
                'larger libraries do not dominate which genes are chosen.'),
        'exclude_genes': Field(
            'string',
            doc='Path to a gene-list file (one id per line, # comments) removed from the '
                'training object after concatenation and before HVG selection; with a '
                'two-layer `layer` both features of a listed gene go. The SAVED h5ad keeps '
                'every gene, with .var[excluded] marking the list, so marker calling and '
                'annotation scoring still see them. ${VAR} is expanded from the environment. '
                'Used for organelle and ribosomal-protein genes, which otherwise rank as '
                'modality-discriminating HVGs.'),
        'do_hvg': Field('boolean',
                        doc='Run the 8-method HVG consensus before training. Default False. Note '
                            'the gene floor that runs with it is HVGSelector-internal, so turning '
                            'this off removes it too -- see min_cells.'),
        'tune': Field('boolean',
                      doc='Ray/hyperopt autotuning instead of fixed model_params. Default False; '
                          'the top-level `defaults:` block in this file currently turns it on. '
                          'Refused on a contrastiveVI target -- see validate.py for why the ELBO '
                          'is the wrong selection criterion there.'),
        'early_stopping': Field('boolean',
                                doc='Stop training when the validation ELBO stops improving. Passed '
                                    'to model.train(). Distinct from `tune`: this is a convergence '
                                    'criterion within one fixed architecture, which is a sound use '
                                    'of elbo_validation even where searching over architectures on '
                                    'it is not.'),
        'model_class': Field('string', enum=INTEGRATION_MODEL_CLASSES,
                             doc='scVI variant to train. Also names the output folder under '
                                 'results/integration/. Default SCVI.'),
        'n_trials': Field('integer', doc='Autotune trial budget. Only read when tune is on.'),
        'search_space': Field('object',
                              doc='Override the autotune search space, merged PER KEY over this '
                                  "model_class's built-in default (integrate.py's "
                                  '_SEARCH_SPACE_DEFAULTS, one per '
                                  'SCVI/JointEmbeddingSCVI/DRVI/SysVI). '
                                  'State only what differs. Values: a LIST becomes tune.choice over '
                                  'it, so a 1-element list pins a parameter; a single-key mapping '
                                  'like {loguniform: [1.0e-4, 1.0e-2]} becomes that ray sampler; '
                                  'null drops the key entirely, leaving scvi\'s own constructor '
                                  'default and not naming it at all. Only read when tune is on -- '
                                  'with tune off the parameters come from model_params. Keys are '
                                  'not checked against the model constructor, so a name the model '
                                  'does not accept fails at the first trial, not here.'),
        'tune_metric': Field('string',
                             doc='Metric the tuner selects on. Default elbo_validation. A scib '
                                 'metric name selects the scib tuner instead.'),
        'tune_mode': Field('string', enum=('min', 'max'),
                           doc='Optimisation direction. Derived from tune_metric when unset '
                               '(min for elbo_validation, max otherwise) -- setting it wrong is '
                               'silent, and min on a scib score selects the WORST trial.'),
        'max_epochs': Field('integer',
                            doc='Total training epochs, applied to BOTH the autotune trials and the '
                                'final fit so the tuner selects in the regime the chosen model is '
                                'trained in. Unset means scvi\'s own cell-count heuristic '
                                '(min(round(20000/n_obs*400), 400)) -- ~123 at 65k cells. KL warmup '
                                'is now scaled to half the effective max_epochs (and any longer '
                                'n_epochs_kl_warmup clamped down) so warmup always completes, so '
                                'this no longer has to be hand-tuned against warmup (scvi-tools '
                                '#507).'),
        'check_val_every_n_epoch': Field('integer',
                                         doc='Log validation metrics every N epochs on the FINAL fit '
                                             '(tuning already logs them for its metric). Set to 1 to '
                                             'get *_validation loss curves in model.history and see '
                                             'the train/val gap. scvi holds out train_size=0.9 by '
                                             'default but logs nothing without this.'),
        'plan_kwargs': Field('object',
                             doc='Passed to model.train(plan_kwargs=...) on both tuning trials and '
                                 'the final fit. Home for the KL-warmup control -- n_epochs_kl_warmup '
                                 '(epoch-based) or n_steps_kl_warmup (iteration-based, preferred for '
                                 'large datasets per scvi-tools #507) -- and any other TrainingPlan '
                                 'arg. `lr` is added to the search space automatically when tuning, '
                                 'so set it here only to pin it. n_epochs_kl_warmup is auto-scaled '
                                 'to half the effective max_epochs and clamped down if set longer, '
                                 'so it need not be kept below max_epochs by hand.'),
        'model_params': Field('object',
                              doc='Passed to the scVI constructor when tune is off. Nested one '
                                  'level: `model_params: {model_params: {n_latent: 20, ...}}`, '
                                  'matching the tuner\'s search-space shape. Empty means scvi\'s '
                                  'own defaults.'),
        'hvg_path': Field('string',
                          doc='scArches targets: gene list the reference model was trained on. '
                              'Normally written by the reference run, not set by hand.'),
        'artifact_key': Field('string',
                              doc='Which mapping artifact this target integrates. Overrides the '
                                  'file-level artifact_key; see that key for the values.'),
        # No `run_solo`: doublets are called per sample in the mapping stage
        # (docs/decisions/doublet-detection.md).
    },
    # A target's keys are a closed set, so a typo such as `model_calss:` fails
    # validation instead of silently training plain SCVI.
    additional=False)

INTEGRATION_CONFIG = Field(
    'object',
    properties={
        'extends': Field('string',
                         doc='Treat this file as an OVERLAY on the config at this path. The base '
                             'supplies `groups:`, `defaults:` and the site paths; this file adds '
                             'targets. `${VAR}` is expanded from the environment and a relative '
                             'path resolves against this file\'s own directory. A group or target '
                             'name defined in both files is an error, not an override. Used by '
                             'the thesis chapters in the notebooks repo, which own their targets '
                             'without owning a copy of the cohort definitions.'),
        'IntegrateConfigs': Field('object', required=True, values=INTEGRATION_TARGET),
        # Same closed key set as a target -- `defaults:` is defined as "exactly
        # equivalent to writing each key on every target", so a key that is not
        # valid on a target is not valid here either.
        'defaults': Field('object', properties=INTEGRATION_TARGET.properties, additional=False,
                          doc='Applied to every target. Exactly equivalent to writing each key '
                              'on every target; a target that states the key wins. The merge is '
                              'shallow, which is why every model key above is flat rather than '
                              'nested under a `model:` block.'),
        'groups': Field('object', values=Field('array', items=Field('string')),
                        doc='Named, reusable sample lists. Referenced by name from Samples/'
                            'Reference/Query. Groups may reference other groups.'),
        'input_dir': Field('string', required=True,
                          doc='Mapping stage output root; where manifests are read from. No '
                              'default -- IntegrationManager raises without it (a site-specific '
                              'path with no sensible fallback).'),
        'output_dir': Field('string', required=True,
                           doc='Same reasoning as input_dir: no default, must be set explicitly.'),
        'artifact_key': Field('string',
                              doc='Which per-sample manifest artifact to integrate: '
                                  '"filtered_h5ad" (default, automatic QC) or '
                                  '"filtered_manual_h5ad" (hand-reviewed via '
                                  'mapping.refilter). Overridable with --artifact-key.'),
    })


# ---------------------------------------------------------------------------
# refine_config.yaml -- stage 3, the scVI -> scANVI refinement of one target.
# ---------------------------------------------------------------------------

REFINE_TARGET = Field(
    'object',
    doc='One refinement target: an annotated dataset, its decisions, and where the six '
        'per-method models are written.',
    properties={
        'seeds': Field('object', doc="Per-target override of the top-level `seeds` block (same keys; the target's values "
                                     'win). Added 2026-09-14: chapter 3 runs `quantile_scope: geneset` for At_tip_ls_H '
                                     'and `group` for the Medicago target.'),
        'adata': Field('string', required=True,
                       doc='The .h5ad stage 1 annotated. Must be the same object -- the geneset '
                           'scores and decisions are keyed on its barcodes.'),
        'batch_key': Field('string',
                           doc='.obs column scVI corrects on. Null/absent for a single-sample '
                               'target, which also forces the single-batch scib fallback.'),
        'counts_layer': Field('string',
                              doc='Raw counts layer for scVI. NOT a normalised one -- scVI models '
                                  'counts directly.'),
        'covariates': Field('array', 'string', items=Field('string'),
                            doc='.obs columns passed to scVI as continuous covariates. Verified '
                                'safe here: SCANVI.from_scvi_model copies the scVI setup args '
                                'through, so unlike SOLO it does not reject a covariate-carrying '
                                'source model.'),
        'methods': Field('array', items=Field('string'),
                         doc='Marker methods to refine. Each is an independent tune+fit, so this '
                             'is the SLURM array width.'),
    })

REFINE_CONFIG = Field(
    'object',
    properties={
        'RefineConfigs': Field(
            'object', required=True,
            properties={
                'targets': Field('object', required=True, values=REFINE_TARGET,
                                 doc='sample name -> target. The name must match the one stage 1 '
                                     'keyed its outputs on.'),
                'annotation_dir': Field('string',
                                        doc='Stage 1 output root holding {sample}/{method}/. '
                                            'Default: results/annotation.'),
                'output_dir': Field('string', doc='Default: results/refine.'),
        'cache_dir': Field('string',
                           doc='Regenerable caches: OrthoFinder runs, prepared primary-transcript '
                               'proteomes, gene maps and preflight validation. Kept out of '
                               '`output_dir` so the deliverables tree holds only what you would '
                               'read, and so this directory is safe to delete at the cost of one '
                               'OrthoFinder rerun. Default: a cache/annotation/ sibling of output_dir.'),
                'log_dir': Field('string'),
                'ontology_file': Field('string',
                                       doc='Plant Ontology .obo, read only to turn PO ids into '
                                           'readable class names for .obs. Optional: without it '
                                           'the annotation columns carry bare PO ids.'),
                'ontology_extensions': Field('array', items=Field('string'),
                                             doc='Extension .obo files grafted into the ontology, '
                                                 'matching annotation_config so a group named with '
                                                 'a grafted id (a NODULE: nodule cell type) shows '
                                                 'its name in .obs rather than the bare id. '
                                                 'Relative paths resolve next to ontology_file.'),
                'seeds': Field(
                    'object',
                    doc='How the confident core of each geneset is chosen. Both values are '
                        'placeholders -- they yield ~7.5% of cells as seeds on MtTip_noHVG, which '
                        'is deliberate but untested against propagation quality.',
                    properties={
                        'quantile_scope': Field('string', enum=['group', 'geneset'],
                                                doc="Combined run only. 'group' (default): the seed quantile is taken over all cells a "
                                                    'review group wins, so a group uniting two clusters with different score scales '
                                                    "seeds only the higher-scoring one. 'geneset': the quantile is taken per winning "
                                                    'member geneset within the group (falls back to the group cutoff below 10 cells); '
                                                    'exclusivity is unchanged. Added 2026-09-14 for chapter 3 (C2 addendum: nuclei-only '
                                                    'pericycle on At_tip_ls_H).'),
                        'quantile': Field('number',
                                          doc='Per-geneset score quantile a cell must clear. Per '
                                              'geneset, not global: scores are not comparable '
                                              'across genesets.'),
                        'margin': Field('number',
                                        doc='Z-score gap the best geneset must beat the runner-up '
                                            'by, else the cell is left for scANVI.'),
                        'min_seed_cells': Field('integer',
                                                doc='Classes with fewer seeds are returned to '
                                                    'Unassigned. Guess; scANVI\'s real floor is '
                                                    'unknown.'),
                    }),
                'safeguards': Field(
                    'object',
                    properties={
                        'min_confidence': Field('number',
                                                doc='Max-posterior floor. PLACEHOLDER -- the 0.8 '
                                                    'default was set against a nearest-centroid '
                                                    'softmax, which is far flatter than a real '
                                                    'scANVI posterior. Calibrate on the first run.'),
                        'max_novelty_ratio': Field('number',
                                                   doc='Measured: 1.25 catches 50.3% of held-out '
                                                       'orphans for 0.7% false positives.'),
                        'reference_quantile': Field('number'),
                    }),
                'model': Field(
                    'object',
                    properties={
                        'tune': Field('boolean',
                                      doc='scib-metric autotuning. Off until step 5 lands.'),
                        'n_trials': Field('integer',
                                          doc='30 is a screening budget, not an optimum -- 6 '
                                              'methods x 200 trials with scib per trial is '
                                              '240-600 GPU-hours.'),
                        'tune_metric': Field('string',
                                             enum=('Total', 'Bio conservation', 'Batch correction',
                                                   'elbo_validation'),
                                             doc='The scib strings select the scib tuner (scvi '
                                                 'dispatches on the metric name). "Total" is the '
                                                 '0.6 bio / 0.4 batch aggregate. A single-batch '
                                                 'target falls back to "Bio conservation".'),
                        'max_epochs': Field('integer'),
                        'model_params': Field('object',
                                              doc='Passed to the scVI constructor when tune is '
                                                  'off. Empty means scvi\'s own defaults.'),
                        'scanvi_params': Field('object',
                                               doc='Passed to SCANVI.from_scvi_model. n_hidden/'
                                                   'n_latent etc. are inherited from the scVI '
                                                   'model and warned about if repeated here.'),
                        'scanvi_max_epochs': Field('integer'),
                        'n_samples_per_label': Field('integer',
                                                     doc='scANVI oversamples labelled cells per '
                                                         'epoch. Guards against the seed set being '
                                                         'a small fraction of the data.'),
                    }),
                'embedding': Field(
                    'object',
                    doc='Neighbours/UMAP/Leiden on the scANVI latent, via shared/dimred.py.',
                    properties={
                        'n_neighbors': Field('integer'),
                        'metric': Field('string'),
                        'min_dist': Field('number'),
                        'leiden_resolution': Field('number'),
                        'random_state': Field('integer'),
                        'compare_models': Field(
                            'boolean',
                            doc='Also embed every OTHER latent the run fitted -- the base '
                                'scVI and whichever comparison arms ran -- with these same '
                                'settings, into that model\'s own directory. On by default: '
                                'the arms exist to be compared and the comparison is visual. '
                                'Costs one neighbours+UMAP per extra latent.'),
                    }),
                'report': Field(
                    'object',
                    properties={
                        'genes': Field('array', items=Field('string'),
                                       doc='Genes whose expression is embedded in the HTML. Full '
                                           'expression cannot be inlined (62,750 x 52,689), so '
                                           'only these columns travel. No real list exists yet -- '
                                           'until one does, treat the panels as a UI test.'),
                        'max_points': Field('integer',
                                            doc='Cells drawn in the report scatter. Above this the '
                                                'payload is subsampled.'),
                    }),
            }),
    },
    additional=False)


# ---------------------------------------------------------------------------
# qc_config.yaml / filter_config.yaml -- structure only, permissive within.
# ---------------------------------------------------------------------------

QC_CONFIG = Field(
    'object',
    properties={'QCconfigs': Field('object', required=True, values=Field('object'),
                                   doc='Named QC-metric profiles, selected per sample by '
                                       '`qc_metrics:`.')},
    additional=False)

FILTER_CONFIG = Field(
    'object',
    properties={'QCfilterconfigs': Field('object', required=True, values=Field('object'),
                                         doc='Named QC-filter profiles, selected per sample by '
                                             '`qc_filter:`.')},
    additional=False)


# Filename stem -> (spec, schema title). Drives both validation dispatch and
# `--emit-schemas`.
SPECS = {
    'sample_sheet': (SAMPLE_SHEET, 'Sample sheet'),
    'mapping_config': (MAPPING_CONFIG, 'Mapper profiles'),
    'index_config': (INDEX_CONFIG, 'Index definitions'),
    'technology_config': (TECHNOLOGY_CONFIG, 'Chemistry definitions'),
    'pipeline_config': (PIPELINE_CONFIG, 'Mapping stage config-of-configs'),
    'integration_config': (INTEGRATION_CONFIG, 'Integration targets'),
    'refine_config': (REFINE_CONFIG, 'Annotation refinement targets'),
    'qc_config': (QC_CONFIG, 'QC metric profiles'),
    'filter_config': (FILTER_CONFIG, 'QC filter profiles'),
}
