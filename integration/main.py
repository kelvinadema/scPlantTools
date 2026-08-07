#!/usr/bin/env python3

"""
Unified integration entrypoint.

The interface rule, in one sentence: **the config defines the model, the CLI
chooses what to run.** No parameter is settable from both places. Model
parameters (tune, model_class, do_hvg, n_trials, covariates, artifact_key, ...)
live only in configs/integration_config.yaml; the flags here select a target,
control this invocation, and nothing else.

Three mutually exclusive modes:
  --target T [T2 ...]                            replay one or more named IntegrateConfigs entries
  --sample S [S2 ...] --name NAME                 ad hoc scVI run (1 sample = single-sample, >1 = integration)
  --reference REF --query Q [Q2 ...] --name NAME  ad hoc scArches run

--sample/--reference runs that train a genuinely new model are written into
IntegrateConfigs under --name, so every trained model is documented and
rerunnable later via --target NAME -- there is no "ad hoc, undocumented" scratch
path. `--set key=value` seeds that new entry, and applies *only* when creating
one.

If --name already exists, the entry owns the model: drop the conflicting --set
to replay it as written, edit the YAML to change it, or pick a new --name. A
disagreement is refused, not warned about. Five flags used to be accepted here
and silently ignored in favour of the config, which meant `--target X --tune`
looked like it enabled tuning and did not -- a warning in a SLURM log nobody
reads is not a safeguard.
"""

import argparse
import logging
import os
import sys

import pandas as pd
import yaml

from pathlib import Path
from shared import config_spec
from shared.targets import resolve_run_spec, resolve_target, expand_sample_list, TargetConfigError
from shared.utils import load_config, setup_logger, find_root_dir, pipeline_log_path
from shared.targets import load_integration_config
from .orchestrator import IntegrationManager
from .integrationutils import CacheUtils
from .config_writer import write_target_entry

# Architecture written into a brand-new, non-tuned ad hoc entry so it is a
# complete record of what was trained rather than a shrug at scvi's constructor
# defaults. Only a seed: once written it is ordinary config and `--set
# model_params=...` overrides it.
DEFAULT_MODEL_PARAMS = {
    'n_layers': 2, 'n_latent': 20, 'n_hidden': 128,
    'dropout_rate': 0.1, 'dispersion': 'gene', 'gene_likelihood': 'nb'}

# What `--set` may name. The structural keys are excluded because the mode flags
# already determine them: --sample gives Samples, --reference/--query give
# Reference/Query, and the mode gives method. Background/Target are structural
# for the same reason and have no mode of their own -- there is deliberately no
# ad hoc CLI path for a contrastiveVI contrast: which samples are the control
# side is the definition of the experiment, not an invocation detail, so a
# contrast is written into the config by hand or not at all.
_STRUCTURAL_KEYS = ('method', 'Samples', 'Reference', 'Query', 'Reference_model',
                    'Background', 'Target')
SETTABLE_KEYS = tuple(k for k in config_spec.INTEGRATION_TARGET.properties
                      if k not in _STRUCTURAL_KEYS)


def parse_set_overrides(pairs) -> dict:
    """
    Parse repeated `--set key=value` into a target-entry fragment.

    Values go through yaml.safe_load, so `tune=true` is a bool, `n_trials=200`
    an int and `covariates=[a, b]` a list -- no bespoke coercion, and the same
    syntax as the config file the value is about to be written into.

    Keys are checked against the integration-target spec. A typo is refused here
    rather than written into the YAML as an inert key that would then be read by
    nothing and reported by nothing.
    """
    overrides = {}
    for pair in pairs or []:
        if '=' not in pair:
            raise ValueError(
                "--set expects key=value, got {!r}. Example: --set tune=true".format(pair))
        key, _, raw = pair.partition('=')
        key, raw = key.strip(), raw.strip()
        if key not in SETTABLE_KEYS:
            raise ValueError(
                "--set: unknown target key {!r}. Settable keys are: {}".format(
                    key, ', '.join(sorted(SETTABLE_KEYS))))
        try:
            value = yaml.safe_load(raw) if raw != '' else None
        except yaml.YAMLError as e:
            raise ValueError("--set {}: could not parse value {!r}: {}".format(key, raw, e)) from e
        overrides[key] = value
    return overrides


def _check_no_conflicts(name, config, overrides, cli_samples, resolved_samples, config_path):
    """
    Refuse a --set or sample list that disagrees with an entry that already exists.

    Compared against the *resolved* target (defaults applied, groups expanded),
    so `--sample p1 p2` against `Samples: ['tip']` is compared as nine samples
    rather than one group name.

    Raising rather than warning is the point of this rewrite: the old code
    logged the conflict and used the config's value, so the flag appeared to
    take effect and did not.
    """
    resolved = resolve_target(config, name)
    conflicts = []
    for key, value in (overrides or {}).items():
        if key in resolved and resolved[key] != value:
            conflicts.append('{}: config says {!r}, --set asks for {!r}'.format(
                key, resolved[key], value))
    if cli_samples is not None and list(cli_samples) != list(resolved_samples or []):
        conflicts.append('sample list: config says {}, CLI asks for {}'.format(
            list(resolved_samples or []), list(cli_samples)))
    if conflicts:
        raise ValueError(
            "'{}' already exists in IntegrateConfigs and disagrees with this invocation:\n"
            "  {}\n"
            "The config defines the model. Drop the conflicting flag to replay the entry as "
            "written, edit {} to change it, or run under a new --name.".format(
                name, '\n  '.join(conflicts), config_path))


def _new_scvi_entry(samples, overrides: dict, file_defaults: dict) -> dict:
    """
    A brand-new scVI entry: the mode flags give its structure, `--set` the rest.

    Whatever `--set` does not state is deliberately left out of the entry, so it
    falls through to the file's own `defaults:` block exactly as a hand-written
    target would. The previous version always wrote explicit tune/do_hvg/
    model_class, which meant a CLI-created entry silently bypassed `defaults:` --
    omitting --tune wrote `tune: False` into a file whose defaults say True.
    """
    entry = {'Samples': list(samples), 'method': 'scVI'}
    entry.update(overrides)
    # Complete the record with whichever of n_trials/model_params the *resolved*
    # tune setting makes meaningful -- read through `defaults:` for the reason
    # above, not assumed off.
    tune = entry.get('tune', file_defaults.get('tune', False))
    if tune:
        entry.setdefault('n_trials', 50)
    else:
        entry.setdefault('model_params', {'model_params': dict(DEFAULT_MODEL_PARAMS)})
    return entry


def _new_scarches_entry(reference, query, icfg, scfg, overrides, logger) -> dict:
    """Build a scArches target entry from --reference/--query on the CLI.

    Args:
        reference (str): Reference target name.
        query (list[str]): Query sample names.
        icfg (dict): The integration config.
        scfg (Path): The sample sheet.
        overrides (dict): `--set` overrides.
        logger (logging.Logger): Where the notes go.

    Returns:
        dict: The new target entry.
    """
    cache = CacheUtils()
    ref_entry = cache.find_entry(reference)
    if ref_entry is None:
        # Not registered yet -- if it's a known IntegrateConfigs target, train it
        # first (same auto-train-on-demand behavior --target scArches already has).
        # The reference trains on whatever artifact_key its own entry names: it
        # is a target like any other, and a reference should be reproducible from
        # its own config rather than from this invocation's flags.
        im = IntegrationManager(icfg, scfg, reference, 'umap', False)
        try:
            im._ensure_reference(reference)
        except Exception as e:
            raise ValueError(
                "Reference '{}' is not a registered model and could not be auto-trained "
                "(it must already exist as a --target to be trained automatically): {}".format(reference, e)) from e
        ref_entry = cache.find_entry(reference)
    ref_samples = ref_entry.get('samples') if ref_entry else None
    if not ref_samples:
        raise ValueError(
            "Reference '{}' has no recorded sample list in its registry entry "
            "(rerun it once with the current code to backfill this field).".format(reference))
    entry = {
        'Reference_model': reference,
        'Reference': list(ref_samples),
        'Query': list(query),
        'method': 'scArches'}
    entry.update(overrides)
    return entry


def _resolve_run_name(args, icfg, scfg, logger):
    """
    For --sample/--reference: ensure the target this invocation will run actually
    exists in IntegrateConfigs, writing a new entry if needed. Returns the name
    to hand to IntegrationManager.

    An existing --name is replayed as written, and any --set or sample list that
    disagrees with it raises. Resolving the spec (rather than just reading the
    raw entry) also fails fast on a target that cannot be resolved at all,
    before any GPU work.
    """
    cfg = load_integration_config(icfg)
    existing = cfg.get('IntegrateConfigs', {}) or {}
    groups = cfg.get('groups') or {}
    overrides = parse_set_overrides(args.set_)
    name = args.name

    wanted, cli_samples = ('scVI', args.sample) if args.sample else ('scArches', args.query)

    if name in existing:
        spec = resolve_run_spec(cfg, name)
        if spec['method'] != wanted:
            raise ValueError(
                "'{}' already exists as a method='{}' target; --{} only applies to {} "
                "targets.".format(name, spec['method'],
                                  'sample' if args.sample else 'reference', wanted))
        resolved_samples = spec['samples'] if wanted == 'scVI' else spec['query']
        _check_no_conflicts(name, cfg, overrides, expand_sample_list(cli_samples, groups),
                            resolved_samples, icfg)
        logger.info("[MAIN] '{}' already exists in IntegrateConfigs; replaying it as "
                    "written.".format(name))
        return name

    if args.sample:
        entry = _new_scvi_entry(args.sample, overrides, cfg.get('defaults') or {})
    else:
        entry = _new_scarches_entry(args.reference, args.query, icfg, scfg, overrides, logger)
    write_target_entry(icfg, name, entry)
    logger.info("[MAIN] Wrote new IntegrateConfigs entry '{}' to {}".format(name, icfg))
    return name


def _enable_tf32(logger) -> None:
    """
    Let float32 matmuls use TF32 tensor cores.

    PyTorch defaults `float32_matmul_precision` to 'highest', which runs every
    float32 matmul in true FP32 and leaves the tensor cores idle. The GPUs this
    stage runs on (L40S, Ada/sm_89, and any Ampere+) implement TF32 in hardware:
    same 8-bit exponent, mantissa truncated to 10 bits, accumulation still in
    FP32. On a matmul-bound model -- which an encoder/decoder over ~31k genes
    very much is -- 'high' is typically a 1.5-2x throughput win for free.

    Why this is safe here and where it would not be. TF32 applies ONLY to
    matmul/conv inputs; reductions, the NB/ZINB likelihood, the KL term and the
    optimiser all stay FP32, so this is not mixed-precision training and there
    is no loss-scaling or overflow exposure. What it does cost is bitwise
    reproducibility against a run made without it -- which for an autotune whose
    ASHA scheduler is already asynchronous and explicitly documented as
    non-reproducible across GPU counts (see run_ch5_integrations.sh) is not a
    property this stage had to begin with.

    Set INTEGRATION_TF32=0 to opt out -- e.g. when deliberately reproducing an
    older fit's numerics rather than training a new one.

    Called once at the entrypoint, following the same convention as
    setup_logger: it is a process-global torch setting, so a per-model call
    site would be both redundant and harder to find. Deliberately NOT applied
    in doublet_model.py, the other entrypoint into this env -- those are small
    disposable per-sample models where the runtime is not the constraint, and
    their calls feed a hard filtering decision.
    """
    if os.environ.get('INTEGRATION_TF32', '1') == '0':
        logger.info('[MAIN] INTEGRATION_TF32=0; leaving float32 matmul precision at torch default.')
        return
    try:
        import torch
        if not torch.cuda.is_available():
            return
        torch.set_float32_matmul_precision('high')
        logger.info(
            '[MAIN] float32 matmul precision set to "high" (TF32) on {}. '
            'Set INTEGRATION_TF32=0 to disable.'.format(torch.cuda.get_device_name(0)))
    except Exception as e:
        # Never worth failing a run over: the fits are correct either way, this
        # only changes how fast they get there.
        logger.warning('[MAIN] Could not enable TF32 matmuls ({}); continuing at torch default.'.format(e))


def main(icfg, scfg, args) -> int:
    """CLI entry point: resolve the requested targets and run each.

    Args:
        icfg (Path): integration_config.yaml.
        scfg (Path): The sample sheet.
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        int: Process exit status.
    """
    # One log file per run, in its own logs/integration/ subdirectory -- see
    # shared.utils.pipeline_log_path for why (every stage now shares this
    # convention) and batch_scripts/run_integration.sh's
    # logs/integration-<jobid>.out/.err for the matching SLURM-side naming.
    log_file = pipeline_log_path('integration', log_dir=find_root_dir() / 'logs')
    # The only place setup_logger is called for 'IntegrateLogging' -- every
    # IntegrationManager just fetches this already-configured logger via
    # logging.getLogger(), per setup_logger's own "call once at the
    # entrypoint" convention.
    logger = setup_logger(name='IntegrateLogging', log_file=log_file, level=logging.INFO)
    _enable_tf32(logger)

    if args.target:
        run_names = list(args.target)
    else:
        # Writes the new entry, or validates that the existing one agrees with
        # this invocation. Either way it can fail before any GPU work, so it is
        # not inside the per-target loop's try/except -- there is nothing to
        # carry on to.
        try:
            run_names = [_resolve_run_name(args, icfg, scfg, logger)]
        except (ValueError, TargetConfigError) as e:
            logger.error('[MAIN] {}'.format(e))
            return 1

    failures = []
    for name in run_names:
        try:
            integrator = IntegrationManager(icfg, scfg, name, args.dimred, args.plot,
                                             overwrite=args.overwrite, force=args.force,
                                             artifact_fallback=args.artifact_fallback)
            adata = integrator.run()
            # Overwrite the h5ad already written mid-run with this fully final
            # version (it now also carries the UMAP/leiden columns _dispatch_reduction
            # adds after that earlier save) at the same path the model itself
            # lives at -- looked up from the registry rather than recomputing a
            # path formula that must independently agree with integrate.py's/
            # refmapping.py's own folder-naming logic.
            entry = CacheUtils().find_entry(name)
            out_dir = Path(entry['model_path']) if entry and entry.get('model_path') else (
                find_root_dir() / "results/integration/scVI" / name)
            adata.write_h5ad(out_dir / "{}.h5ad".format(name))
            write_anndata_metadata(adata, outfile=out_dir / "{}_metadata.tsv".format(name))
        except Exception as e:
            logger.exception("[MAIN] Error processing '{}': {}".format(name, e))
            failures.append((name, e))

    n_ok = len(run_names) - len(failures)
    if failures:
        logger.error("[MAIN] Finished with failures: {}/{} succeeded. Failed: {}".format(
            n_ok, len(run_names), ', '.join(n for n, _ in failures)))
    else:
        logger.info("[MAIN] Finished: {}/{} succeeded".format(n_ok, len(run_names)))
    return len(failures)


def write_anndata_metadata(adata, outfile="metadata.tsv"):
    """Write a per-target metadata TSV beside the h5ad.

    Args:
        adata (AnnData): The integrated object.
        outfile (Path): Where to write.
    """
    metadata = adata.obs.copy()
    for key in adata.obsm:
        arr = adata.obsm[key]
        if arr.ndim == 2:
            obsm_df = pd.DataFrame(
                arr,
                index=adata.obs.index,
                columns=[f"{key}_{i+1}" for i in range(arr.shape[1])])
            metadata = pd.concat([metadata, obsm_df], axis=1)
        elif arr.ndim == 1:
            metadata[key] = arr
    metadata.to_csv(outfile, sep="\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform single cell integration')
    parser.add_argument('--icfg', required=True, help='Path to integration config.yaml')
    parser.add_argument('--scfg', required=True, help='Path to sample metadata config.yaml')

    parser.add_argument('--target', nargs='+', default=None,
                         help='Run one or more named targets from IntegrateConfigs')
    parser.add_argument('--sample', nargs='+', default=None,
                         help='Ad hoc scVI run over these sample(s); requires --name. '
                              '1 sample = single-sample mode, >1 = multi-sample integration')
    parser.add_argument('--reference', default=None,
                         help='Ad hoc scArches run: name of an existing registered/curated '
                              'reference model; requires --query and --name')
    parser.add_argument('--query', nargs='+', default=None, help='Query sample(s) for --reference mode')
    parser.add_argument('--name', default=None,
                         help="Name under which an ad hoc --sample/--reference run is "
                              "registered and documented in IntegrateConfigs")

    # Model parameters are NOT flags. They live in the target entry, and --set
    # is the one way to state them -- when creating that entry, and only then.
    # --tune/--model-class/--do-hvg/--n-trials/--covariates/--artifact-key and
    # --samples-override are all gone; see this module's docstring. The
    # replacement for --samples-override is `--sample S1 S2 --name NEW`, which
    # writes an entry and is therefore reproducible.
    parser.add_argument('--set', dest='set_', action='append', metavar='KEY=VALUE', default=None,
                         help='Set a key on a NEW --sample/--reference entry, e.g. '
                              '--set tune=true --set n_trials=200. Repeatable. Values use YAML '
                              'syntax. Settable keys: {}'.format(', '.join(sorted(SETTABLE_KEYS))))

    # --run-solo was removed: doublet detection moved into the mapping stage as
    # a per-sample step. `.obs['solo_doublet']` arrives with the mapping
    # artifact; see mapping/doublets.py and solo.mode in
    # filter_config.yaml.
    #
    # --artifact-fallback stays a flag rather than a config key on purpose: it
    # is a permission ("I accept a model trained on mixed filtering"), not a
    # parameter, and the default is to refuse.
    parser.add_argument('--artifact-fallback', dest='artifact_fallback', action='store_true',
                         help='Use the automatic artifact for samples that have no manual one. '
                              'Off by default: silently mixing manually- and auto-filtered '
                              'samples trains a quietly different model.')

    # 'pca' was accepted here and silently did nothing -- _dispatch_reduction
    # has no branch for it. Only what is implemented is offered.
    parser.add_argument('--dimred', type=str, choices=['umap'], default='umap',
                         help='Dimensionality reduction to compute after integration')
    parser.add_argument('--plot', action='store_true', help='Generate dimred and metric plots after integration')
    parser.add_argument('--overwrite', action='store_true',
                         help='Force a full retrain + re-tune, bypassing both registry shortcuts')
    parser.add_argument('--force', action='store_true',
                         help="Bypass the report-embedding done-sentinel for a 1-sample run")

    args = parser.parse_args()

    mode_count = sum(x is not None for x in (args.target, args.sample, args.reference))
    if mode_count != 1:
        parser.error("Specify exactly one of --target, --sample, or --reference")
    if bool(args.reference) != bool(args.query):
        parser.error("--reference and --query must be given together")
    if (args.sample or args.reference) and not args.name:
        parser.error("--sample/--reference require --name")
    if args.set_ and args.target:
        parser.error(
            "--set only applies when creating a new --sample/--reference entry. A --target "
            "entry already documents its own parameters; edit the config to change them.")

    n_failed = main(args.icfg, args.scfg, args)
    sys.exit(1 if n_failed else 0)
