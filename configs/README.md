# Configuration inventory

Start a new dataset from [examples/minimal/](examples/minimal/README.md), copied
into ignored `configs/local/`. The full path/precedence rules are in
[configuration](../docs/configuration.md).

| Location | Role |
|---|---|
| `examples/minimal/` | Complete editable mapping/QC example using existing scientific defaults |
| `local/` | Your machine/input configuration; excluded from Git |
| Top-level `*_config.yaml` | Preserved working study settings and profiles |
| `sample_metadata*.yaml`, `MarcantiaSample.yaml` | Preserved study sample sheets |
| `external/` | Public-study manifests and curated metadata; incomplete fetch files remain local |
| `C14/` | Study-specific read-budget cohorts |
| `chemistry_recheck_2026-09-21/` | Historical chemistry comparison configurations |
| `markerid/` | Historical study input lists; verify their older output paths before reuse |
| `schemas/` | Generated editor metadata from `shared/config_spec.py` |

Existing files were kept at their original paths and their values were not
retuned. KB and STARsolo entries remain unimplemented placeholders. A file's
presence here does not imply its external FASTQs/reference data are bundled or
available; use preflight and the opt-in site diagnostics before a study rerun.
