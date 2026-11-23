# Optional integration and reference mapping

Use `scIntegrationV1.12` and run from the checkout root. Integration consumes mapping artifacts; it is not required to map/QC new data. Existing model implementations, parameters, and registry behavior are unchanged.

Prepare a local integration config and a sheet containing the samples it references:

```bash
python -m mapping.validate --icfg configs/local/integration.yaml --scfg configs/local/samples.yaml
python -m integration.main --icfg configs/local/integration.yaml --scfg configs/local/samples.yaml --target YOUR_TARGET
```

`configs/integration_config.yaml` is the preserved study reference, not a minimal portable example. `shared/targets.py` resolves defaults, named groups, targets, covariates, and layer selection. `python -m integration.main --help` lists existing options. `python -m sc integrate ...` dispatches to the same entry point in its environment.

An `extends:` overlay can add targets/groups to an existing base. Environment variables in `extends` are expanded and relative parent paths resolve beside the overlay. Duplicate target/group names are errors; defaults merge per key. See [configuration](configuration.md).

The existing registry can reuse a persisted model and its cached AnnData when the target already exists and overwrite is not requested. Editing a sample list is not a universal cache invalidation mechanism. Use new target names or explicit overwrite only when intentionally recomputing. Preserve model files and registry state during organization.

Choose covariates deliberately; an absent required column fails validation/runtime checks. Consult the [batch/confound rationale](decisions/batch-keys-and-confounds.md), [two-layer input rationale](decisions/two-layer-label-spaces.md), and other decision documents before changing scientific settings. Mapping's automatic and manual outputs are selected through the existing artifact-key configuration.

Study overlays and downstream analyses live in the separate notebooks repository. Its launchers still use `batch_scripts/lib/common.sh`; that cross-repository interface is retained.
