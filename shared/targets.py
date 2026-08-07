#!/usr/bin/env python3

"""
Resolution of integration targets: `extends:`, `defaults:` and `groups:`.

The counterpart to shared/samples.py, for integration_config.yaml. Same reason
for existing: the orchestrator and the validator must agree on what a target
says, and that agreement has to survive `Samples: tip_protoplast` meaning nine
samples rather than one named "tip_protoplast".

Three mechanisms, all purely notational -- they change how a target is written,
never what it resolves to:

  extends:   this file is an OVERLAY on another config. The base supplies the
             cohort groups, the `defaults:` block and the site paths; the
             overlay adds targets. A thesis chapter owns its own targets without
             owning a copy of the pipeline's cohort definitions -- see
             load_integration_config for the merge rules.
  defaults:  applied to every target. Exactly equivalent to writing each key on
             every target; a target that states the key wins.
  groups:    named, reusable sample lists, referenced by name from Samples /
             Reference / Query. Groups may contain other group names. A name
             that is not a group is passed through untouched, so a literal
             sample list keeps working and a group name and a sample name never
             need to be told apart at the call site.

Group expansion preserves order and removes duplicates, so overlapping groups
compose without the caller having to care.
"""

import os
from string import Template
from pathlib import Path


class TargetConfigError(ValueError):
    """Raised for an integration config that cannot be resolved into targets."""


# Top-level keys merged per key rather than replaced wholesale when a config
# `extends:` another. Everything else is a scalar the overlay simply overrides.
#
# The two mapping keys are UNIONED and a name present in both files is an
# ERROR, not an override: two files claiming the same target or group name is
# precisely the ambiguity that makes a run train on something nobody chose, and
# the overlay cannot see what it would be shadowing. `defaults:` is the one
# exception -- overriding a default is what a defaults block is for.
_MERGE_UNION = ('groups', 'IntegrateConfigs')
_MERGE_OVERRIDE = ('defaults',)


def _resolve_extends(raw: str, source: Path) -> Path:
    """The path an `extends:` value points at.

    `${VAR}` is expanded from the environment, so a chapter overlay can name the
    pipeline checkout as `${SCRNASEQ_ROOT}/configs/...` rather than hardcoding
    one machine's layout. A relative path resolves against the directory of the
    file that wrote it, not the process cwd -- an overlay is submitted from
    wherever the job happens to start.

    `SCRNASEQ_ROOT` falls back to this checkout when it is not in the
    environment. The batch scripts export it, but the interactive case --
    `python -m integration.main --icfg <chapter file>` from the repo root -- has
    no reason to, and the code resolving the config is already running inside
    the tree the variable names.

    Args:
        raw (str): The `extends:` value as written.
        source (Path): The config file that wrote it.

    Returns:
        Path: The base config's path.

    Raises:
        TargetConfigError: If an expansion leaves an unresolved `${VAR}`.
    """
    env = dict(os.environ)
    if not env.get('SCRNASEQ_ROOT'):
        try:
            from .utils import find_root_dir
            env['SCRNASEQ_ROOT'] = str(find_root_dir())
        except Exception:                    # not inside a checkout; leave it unset
            pass
    expanded = Template(str(raw)).safe_substitute(env)
    if '$' in expanded:
        raise TargetConfigError(
            "{}: extends: {!r} still contains an unexpanded variable after expansion ({!r}). "
            'Set it in the environment, or write an absolute path.'.format(
                source, raw, expanded))
    path = Path(expanded).expanduser()
    return path if path.is_absolute() else (source.parent / path).resolve()


def load_integration_config(path, _seen=None) -> dict:
    """An integration config with any `extends:` base merged underneath it.

    `groups` and `IntegrateConfigs` are unioned and a name defined in both files
    raises; `defaults` merges per key with the overlay winning; every other key
    the overlay states replaces the base's. The returned dict carries no
    `extends` key, so every consumer downstream sees one flat config and cannot
    tell an overlay from a standalone file.

    Args:
        path (str | Path): The config to load. May itself be a base.
        _seen (tuple, optional): Internal, for cycle detection.

    Returns:
        dict: The merged config.

    Raises:
        TargetConfigError: On a missing base, a cycle, or a name defined twice.
    """
    from .utils import load_config          # deferred: shared/utils imports yaml

    path = Path(path).resolve()
    _seen = _seen or ()
    if path in _seen:
        raise TargetConfigError(
            'extends: cycle {}'.format(' -> '.join(str(p) for p in _seen + (path,))))

    config = load_config(path) or {}
    base_ref = config.pop('extends', None)
    if not base_ref:
        return config

    base_path = _resolve_extends(base_ref, path)
    if not base_path.is_file():
        raise TargetConfigError(
            '{}: extends: {!r} resolves to {}, which is not a file.'.format(
                path, base_ref, base_path))

    merged = load_integration_config(base_path, _seen + (path,))

    for key in _MERGE_UNION:
        overlay = config.pop(key, None) or {}
        base = merged.get(key) or {}
        clashes = sorted(set(base) & set(overlay))
        if clashes:
            raise TargetConfigError(
                "{}: {} redefines {} already defined in {}: {}. A name may be defined in one "
                'file only -- an overlay cannot see what it would be shadowing, and silently '
                'winning is how a run trains on something nobody chose. Rename it, or remove '
                'it from the base.'.format(
                    path, key, 'entries' if len(clashes) > 1 else 'an entry', base_path,
                    ', '.join(clashes)))
        if base or overlay:
            merged[key] = {**base, **overlay}

    for key in _MERGE_OVERRIDE:
        overlay = config.pop(key, None)
        if overlay:
            merged[key] = {**(merged.get(key) or {}), **overlay}

    merged.update(config)
    return merged


SAMPLE_LIST_KEYS = ('Samples', 'Reference', 'Query', 'Background', 'Target')

# Keys where an explicit `key: null` is a real instruction rather than an
# omission, and must override `defaults:` instead of falling through to it.
# See docs/decisions/batch-keys-and-confounds.md.
NULLABLE_KEYS = frozenset({'batch_key'})


def expand_group(name, groups, _seen=None):
    """Resolve one name to a list of sample names, following nested groups.

    A name that is not a group resolves to itself -- that is what lets
    `Samples: ['tip_protoplast', 'nMOCK']` mix a group with a bare sample.

    Args:
        name (str): Group or sample name.
        groups (dict): The `groups:` block.
        _seen (tuple): Internal, for cycle detection.

    Returns:
        list[str]: Sample names.

    Raises:
        TargetConfigError: If a group references itself.
    """
    _seen = _seen or ()
    if name in _seen:
        raise TargetConfigError(
            "group '{}' references itself (via {})".format(name, ' -> '.join(_seen)))
    if name not in (groups or {}):
        return [name]

    members = groups[name]
    if isinstance(members, str):
        members = [members]

    out = []
    for member in members or []:
        out.extend(expand_group(member, groups, _seen + (name,)))
    return out


def expand_sample_list(value, groups):
    """Expand a Samples/Reference/Query value, de-duplicated, order preserved.

    Args:
        value (str | list | None): The value as written.
        groups (dict): The `groups:` block.

    Returns:
        list[str] or None: Sample names, or None if `value` was None.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]

    seen = set()
    out = []
    for item in value:
        for sample in expand_group(item, groups):
            if sample not in seen:
                seen.add(sample)
                out.append(sample)
    return out


def resolve_target(config: dict, target: str) -> dict:
    """One fully-resolved target: defaults applied, group names expanded.

    The returned dict is a copy, so a caller may mutate it without touching the
    loaded config. A key present but null is treated as unstated unless it is in
    NULLABLE_KEYS.

    Args:
        config (dict): The whole integration config.
        target (str): Target name.

    Returns:
        dict: The resolved entry.

    Raises:
        TargetConfigError: If the target is not in the config.
    """
    targets = config.get('IntegrateConfigs') or {}
    if target not in targets:
        raise TargetConfigError(
            "Target '{}' not found in config. Known targets: {}".format(
                target, ', '.join(sorted(targets)) or '(none)'))

    defaults = config.get('defaults') or {}
    groups = config.get('groups') or {}

    resolved = dict(defaults)
    for key, value in (targets[target] or {}).items():
        if value is not None or key in NULLABLE_KEYS:
            resolved[key] = value

    for key in SAMPLE_LIST_KEYS:
        if key in resolved:
            resolved[key] = expand_sample_list(resolved[key], groups)

    return resolved


def resolve_all(config: dict) -> dict:
    """Every target in the config, resolved. Useful for validation and listing.

    Args:
        config (dict): The whole integration config.

    Returns:
        dict: {target name: resolved entry}.
    """
    return {name: resolve_target(config, name)
            for name in (config.get('IntegrateConfigs') or {})}


# ---------------------------------------------------------------------------
# From "what the entry says" to "what the run will therefore do".
# ---------------------------------------------------------------------------

INTEGRATION_METHODS = ('scVI', 'scArches', 'contrastiveVI')

# Code-level defaults for keys a target may omit. Here rather than inline in
# orchestrator._get_type so the run, the validator and the CLI's --set checking
# all read ONE set of values.
DEFAULT_DO_HVG = False
DEFAULT_TUNE = False
DEFAULT_MODEL_CLASS = 'SCVI'
# Kept as the default so every existing target resolves as it did before
# batch_key became expressible; an explicit null reaches here as None.
DEFAULT_BATCH_KEY = 'batch'


def parse_covariates(raw):
    """Normalize a target's `covariates:` into categorical and continuous lists.

    Args:
        raw: The `covariates:` block, in any of its accepted shapes.

    Returns:
        dict: {'categorical': list[str], 'continuous': list[str]}.

    Raises:
        TargetConfigError: If the block is malformed.
    """
    if raw is None:
        return {'categorical': [], 'continuous': []}
    if isinstance(raw, list):
        return {'categorical': [], 'continuous': list(raw)}
    if isinstance(raw, dict):
        return {'categorical': list(raw.get('categorical') or []),
                'continuous': list(raw.get('continuous') or [])}
    raise TargetConfigError("unsupported 'covariates' schema: {}".format(type(raw)))


# ---------------------------------------------------------------------------
# batch_key: one column, several combined into one, or a per-cohort mapping.
# ---------------------------------------------------------------------------

# Joins both the derived column's NAME and the values inside it, so a level of
# `tech+prep` reads '10xv3+2h_only' and can be split back apart by eye in a plot
# legend. A single character that appears in no existing key or value.
BATCH_KEY_SEP = '+'

# Column name for the mapped form. Unlike the combined form there is no single
# key list to name it after -- that is the whole point of the form -- so it gets
# a fixed name. Written into .obs, so it shows up in diagnostics under this name.
MAPPED_BATCH_COLUMN = 'batch_mixed'


def parse_batch_key(raw, groups=None):
    """Normalize a target's `batch_key:` into a spec the run and the validator share.

    Four forms -- a column name, null, a list to combine, or a mapping resolved per
    sample. See docs/decisions/batch-keys-and-confounds.md for what each is for and
    why a sample claimed by two mapping entries is an error.

    Args:
        raw: The `batch_key:` value as written.
        groups (dict, optional): The `groups:` block, for the mapped form.

    Returns:
        dict: 'mode' ('none'|'single'|'combined'|'mapped'), 'column' (the .obs
            column the model will use, or None), 'components' (every key
            referenced, for validation), and 'assignments' ({sample: (key, ...)}
            for mapped, else None).

    Raises:
        TargetConfigError: On an empty, malformed or overlapping specification.
    """
    if raw is None:
        return {'mode': 'none', 'column': None, 'components': (), 'assignments': None}

    if isinstance(raw, str):
        if not raw:
            raise TargetConfigError("batch_key: empty string; use an explicit null to mean "
                                    '"no batch correction".')
        return {'mode': 'single', 'column': raw, 'components': (raw,), 'assignments': None}

    if isinstance(raw, (list, tuple)):
        keys = _batch_key_list(raw, 'batch_key')
        # One key in a list is just that key. Collapsing here rather than
        # deriving a one-component column keeps `['tech']` and `'tech'` the same
        # run, so a target can gain and lose a second key without its .obs
        # column name changing under it.
        if len(keys) == 1:
            return {'mode': 'single', 'column': keys[0],
                    'components': tuple(keys), 'assignments': None}
        return {'mode': 'combined', 'column': BATCH_KEY_SEP.join(keys),
                'components': tuple(keys), 'assignments': None}

    if isinstance(raw, dict):
        if not raw:
            raise TargetConfigError("batch_key: empty mapping; use an explicit null to mean "
                                    '"no batch correction".')
        assignments = {}
        owner = {}
        components = []
        for cohort, value in raw.items():
            keys = _batch_key_list(
                [value] if isinstance(value, str) else value,
                "batch_key mapping entry '{}'".format(cohort))
            for key in keys:
                if key not in components:
                    components.append(key)
            for sample in expand_group(cohort, groups or {}):
                if sample in assignments:
                    raise TargetConfigError(
                        "batch_key mapping: sample '{}' is matched by both '{}' and '{}'. "
                        'Overlapping entries have no defined precedence -- split the groups '
                        'so every sample is named exactly once.'.format(
                            sample, owner[sample], cohort))
                assignments[sample] = tuple(keys)
                owner[sample] = cohort
        return {'mode': 'mapped', 'column': MAPPED_BATCH_COLUMN,
                'components': tuple(components), 'assignments': assignments}

    raise TargetConfigError("unsupported 'batch_key' schema: {}. Use a string, a list of "
                            'strings, a mapping of group -> string-or-list, or '
                            'null.'.format(type(raw)))


def _batch_key_list(raw, label):
    """Validate one list of batch key names: non-empty strings, no duplicates.

    Args:
        raw (list): Keys as written.
        label (str): Prefix for the error messages.

    Returns:
        list[str]: The validated keys.

    Raises:
        TargetConfigError: If empty, non-string, or containing duplicates.
    """
    if raw is None:
        raise TargetConfigError('{}: no keys given.'.format(label))
    keys = list(raw)
    if not keys:
        raise TargetConfigError('{}: empty list; use an explicit null to mean '
                                '"no batch correction".'.format(label))
    for key in keys:
        if not isinstance(key, str) or not key:
            raise TargetConfigError('{}: {!r} is not a column name.'.format(label, key))
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise TargetConfigError(
            '{}: repeated key(s) {}. Each key may appear once -- repeating one would '
            'duplicate its values inside every derived level without changing the '
            'grouping.'.format(label, ', '.join(duplicates)))
    return keys


def resolve_batch_assignments(spec, samples):
    """For a mapped batch_key, the per-sample key tuples, checked for full coverage.

    Separate from `parse_batch_key` because coverage can only be judged against the
    target's own resolved sample list, which the parser does not see. Raises rather
    than falling back to a default key.

    Args:
        spec (dict): As returned by `parse_batch_key`.
        samples (list[str]): The target's resolved samples.

    Returns:
        dict or None: {sample: (key, ...)}, or None when the mode is not 'mapped'.

    Raises:
        TargetConfigError: If any sample is uncovered.
    """
    if spec.get('mode') != 'mapped':
        return None
    assignments = spec['assignments']
    missing = [s for s in samples if s not in assignments]
    if missing:
        raise TargetConfigError(
            'batch_key mapping does not cover {} of the target\'s samples: {}. Every sample '
            'must be named by exactly one entry -- there is no default, because a sample '
            'quietly inheriting one would be corrected on a key nobody chose for '
            'it.'.format(len(missing), ', '.join(sorted(missing))))
    return {s: assignments[s] for s in samples}


def _require(cfg, target, key):
    """Fetch a key a method requires, raising if it is missing or empty.

    Args:
        cfg (dict): The resolved target.
        target (str): Target name, for the error message.
        key (str): Key to fetch.

    Returns:
        The value.

    Raises:
        TargetConfigError: If the key is absent or falsy.
    """
    value = cfg.get(key)
    if not value:
        raise TargetConfigError("target '{}': method='{}' requires '{}'.".format(
            target, cfg.get('method'), key))
    return value


def _train_args(cfg) -> dict:
    """kwargs for scvi's `model.train()` on the FINAL fit.

    Built as a dict rather than a scalar so further train() kwargs get an obvious
    home. Kept import-free of scvi/anndata: preflight resolves specs outside the
    integration env.

    Args:
        cfg (dict): The resolved target.

    Returns:
        dict: kwargs for `model.train()`, including `plan_kwargs` when set.
    """
    train_args = {}
    if cfg.get('max_epochs') is not None:
        train_args['max_epochs'] = int(cfg['max_epochs'])
    if cfg.get('check_val_every_n_epoch') is not None:
        train_args['check_val_every_n_epoch'] = int(cfg['check_val_every_n_epoch'])
    if cfg.get('early_stopping') is not None:
        train_args['early_stopping'] = bool(cfg['early_stopping'])
    if cfg.get('plan_kwargs'):
        train_args['plan_kwargs'] = dict(cfg['plan_kwargs'])
    return train_args


def resolve_run_spec(config: dict, target: str) -> dict:
    """One target resolved all the way down to what the run actually uses.

    `resolve_target` answers "what does this entry say"; this answers "what will
    therefore happen" -- defaults applied, groups expanded, code-level defaults
    filled in. Deliberately free of scvi/anndata imports: the model class is *named*
    here and resolved to a class by the caller, so preflight can resolve a full spec
    outside the integration env.

    Args:
        config (dict): The whole integration config.
        target (str): Target name.

    Returns:
        dict: Everything the orchestrator needs, including 'method', 'samples',
            'covariates', 'batch_key' and 'batch_key_spec'.

    Raises:
        TargetConfigError: If the method is missing or unsupported.
    """
    cfg = resolve_target(config, target)
    _batch_spec = parse_batch_key(cfg.get('batch_key', DEFAULT_BATCH_KEY),
                                  config.get('groups') or {})

    method = cfg.get('method')
    if method is None:
        raise TargetConfigError("target '{}' has no 'method' defined.".format(target))
    if method not in INTEGRATION_METHODS:
        raise TargetConfigError("target '{}': unsupported method '{}'; must be one of {}.".format(
            target, method, ', '.join(INTEGRATION_METHODS)))

    spec = {
        'method': method,
        'artifact_key': cfg.get('artifact_key'),
        'covariates': parse_covariates(cfg.get('covariates')),
        # Stays the name of the ONE .obs column the model corrects on, or None,
        # so the list and mapping forms leave every existing consumer untouched.
        # For those the column does not exist yet -- _apply_batch_key derives it.
        'batch_key': _batch_spec['column'],
        'batch_key_spec': _batch_spec,
        # Resolved here rather than per method, since _concatenate_adata runs
        # on all three. None keeps every layer -- the usual OOM on a large
        # cohort. See docs/decisions/concatenation-hygiene.md.
        'keep_layers': cfg.get('keep_layers'),
        # Same placement reason as keep_layers. None means no filtering: every
        # target predating this key must keep the gene set it was fitted on.
        'min_cells': cfg.get('min_cells'),
    }

    if method == 'scVI':
        spec.update({
            'samples': _require(cfg, target, 'Samples'),
            'do_hvg': cfg.get('do_hvg', DEFAULT_DO_HVG),
            'tune': cfg.get('tune', DEFAULT_TUNE),
            'model_class': cfg.get('model_class', DEFAULT_MODEL_CLASS),
            'n_trials': cfg.get('n_trials'),
            'tune_metric': cfg.get('tune_metric'),
            'tune_mode': cfg.get('tune_mode'),
            'model_params': cfg.get('model_params'),
            # Merged over the model class's built-in default PER KEY, by
            # integrate.resolve_search_space -- not here, because the defaults
            # are keyed on model_class and live beside the model code. Only read
            # when tune is on. Deliberately resolved as None-if-unstated rather
            # than materialised: this module must stay free of scvi imports (see
            # the note above), and the defaults are ray tune values.
            'search_space': cfg.get('search_space'),
            'train_args': _train_args(cfg),
        })
    elif method == 'contrastiveVI':
        background = _require(cfg, target, 'Background')
        target_samples = _require(cfg, target, 'Target')
        # Both lists are needed as lists (to build the two index vectors) AND
        # concatenated as `samples` (the orchestrator loads one AnnData covering
        # the whole contrast, and the indices are positions into it).
        spec.update({
            'background': background,
            'target_samples': target_samples,
            'samples': list(background) + [s for s in target_samples if s not in background],
            'do_hvg': cfg.get('do_hvg', DEFAULT_DO_HVG),
            # Deliberately not tunable: contrastiveVI's objective is a sum of a
            # target ELBO and a background ELBO and contains no term rewarding
            # salient/background separation, so an elbo_validation search picks
            # the largest salient space and a zero Wasserstein penalty -- the
            # documented leak failure mode. validate.py refuses `tune: true`
            # here and explains why; resolving it to a constant False keeps the
            # file's `defaults: {tune: True}` block from turning it on silently.
            'tune': False,
            # The supported way to choose `wasserstein_penalty` and
            # `n_salient_latent`, and the reason `tune` above is hard-False. A
            # sweep fits a grid of disposable models, ranks them on criteria
            # computed OUTSIDE the objective (separation, identity leakage
            # measured on background cells, salient effective dimensionality),
            # and trains the winner as the target's real model. `sweep_grid`
            # overrides the axes; unstated uses cvi_sweep's defaults.
            'sweep': bool(cfg.get('sweep', False)),
            'sweep_grid': cfg.get('sweep_grid') or {},
            'model_params': cfg.get('model_params'),
            'train_args': _train_args(cfg),
        })
    else:
        spec.update({
            'reference_model': _require(cfg, target, 'Reference_model'),
            'reference': _require(cfg, target, 'Reference'),
            'query': _require(cfg, target, 'Query'),
            'hvg_path': cfg.get('hvg_path'),
            # The query fine-tune trains through model.train() too, so it honours
            # the same max_epochs / plan_kwargs (KL warmup) / check_val_every_n_epoch
            # as the scVI path. refmapping merges the canonical weight_decay=0.0
            # into whatever plan_kwargs come through here.
            'train_args': _train_args(cfg),
        })

    # Coverage can only be judged now, against the target's own resolved sample
    # list. Raising here rather than at run time means the validator catches an
    # uncovered sample without loading a single artifact -- and that a scArches
    # target, whose `samples` key does not exist, is checked against the union
    # its run actually concatenates.
    if _batch_spec['mode'] == 'mapped':
        covered = spec.get('samples')
        if covered is None:
            covered = list(spec.get('reference') or []) + [
                s for s in (spec.get('query') or []) if s not in (spec.get('reference') or [])]
        resolve_batch_assignments(_batch_spec, covered)

    return spec
