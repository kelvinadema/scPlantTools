# Optional annotation and refinement

Annotation uses orthogroups, marker tables, and Plant Ontology; it is independent of the basic mapping/QC run. Keep the existing ontology assets, proteome versions, marker inputs, and patches when reproducing a result.

Use `scAnnotationV3.12` for annotation and `scMarkersV3.12` for marker discovery. Refinement runs in `scIntegrationV1.12`. Follow [environment setup](../envs/README.md). The `sc` dispatcher selects these environments without importing the other stages into one interpreter.

The preserved `configs/annotation_config.yaml` documents the study's species profiles and inputs. Prepare your local configuration explicitly and inspect available arguments:

```bash
python -m annotation.main --help
python -m annotation.main --acfg configs/local/annotation.yaml --validate-only
python -m annotation.main --acfg configs/local/annotation.yaml --orthofinder-only
python -m annotation.main --acfg configs/local/annotation.yaml
```

Annotation's `--validate-only` prepares/checks reference inputs and may write caches; it is not a read-only source inspection. Orthofinder-only prepares the orthology stage without requiring all downstream marker outputs. Species profiles can select the target and remove it from the reference pool.

Review annotations through the existing review workflow (`python -m annotation.review --help`). Refinement uses the resulting decisions and its own config (`python -m refine.main --help`). Existing `seeds.quantile_scope` and per-target seed overrides remain supported; no seed policy or training method was changed for release preparation.

The [marker port notes](markerid-porting.md), [proteome preparation rationale](decisions/proteome-preparation.md), and [decision documents](decisions/) preserve scientific context. Old study handoff notes are historical records rather than a statement that current functionality is unimplemented.

Standalone maintenance utilities under `integration/` include barcode-index backfills and label injection. They are not automatically part of a new annotation run. Inspect their explicit targets and dry-run behavior before using them on existing artifacts.
