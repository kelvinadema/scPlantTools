# Configurations

For a new dataset, copy [examples/minimal/](examples/minimal/README.md) to `configs/local/`, which Git ignores. The path and precedence rules are described in [configuration](../docs/configuration.md).

| Location | Contents |
|---|---|
| `examples/minimal/` | Complete mapping and QC example with the default settings |
| `local/` | Your own configuration (not tracked) |
| `*_config.yaml` | Settings and QC profiles used in our studies |
| `sample_metadata*.yaml`, `MarcantiaSample.yaml` | Sample sheets of our studies |
| `external/` | Sample sheets and metadata for public datasets |
| `C14/` | Read-depth subsampling cohorts from one study |
| `chemistry_recheck_2026-09-21/` | Configurations from a re-check of library chemistries |
| `markerid/` | Input lists from an earlier study; their result paths may be out of date |
| `schemas/` | Editor schemas generated from `shared/config_spec.py` |

The study files are kept as they were used and have not been re-tuned. The kb-python and STARsolo entries are placeholders and do not run. The FASTQ and reference files these configurations point to are not included; run the preflight check and `python -m sc test site` before rerunning a study.
