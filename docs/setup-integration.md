# Integration and reference mapping

Integration is optional and runs in `scIntegrationV1.12` from the repository root. It reads the filtered objects written by mapping.

Write a local integration configuration and a sample sheet that contains the samples it uses, then validate and run a target:

```bash
python -m mapping.validate --icfg configs/local/integration.yaml --scfg configs/local/samples.yaml
python -m integration.main --icfg configs/local/integration.yaml --scfg configs/local/samples.yaml --target YOUR_TARGET
```

`python -m integration.main --help` lists all options, and `python -m sc integrate ...` runs the same command in the right environment. `configs/integration_config.yaml` is the configuration of our own studies and shows the available keys, but it is not a minimal example. Defaults, named sample groups, covariates and layer selection are resolved by `shared/targets.py`.

A configuration can build on another with `extends:`. Environment variables in the path are expanded, and a relative path is resolved from the file that contains it. A target or group may not be defined in both files; `defaults:` are merged per key. See [configuration](configuration.md).

Trained models are recorded in a registry. If a target has already been trained, the stored model and its AnnData are reused unless you ask for them to be overwritten; changing the sample list alone does not trigger retraining. Use a new target name, or overwrite explicitly, when you want a new model.

Batch keys and covariates must exist as `.obs` columns; a missing column fails validation or training. The choice of batch key and of the two-layer (spliced and unspliced) input is discussed in [batch keys and confounds](decisions/batch-keys-and-confounds.md) and [two-layer label spaces](decisions/two-layer-label-spaces.md). Which mapping output a target reads (automatic or manually filtered) is set with `artifact_key`.

Our study-specific integration configurations and analyses are kept in a separate notebooks repository, whose launchers use `batch_scripts/lib/common.sh` from this repository.
