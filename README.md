# scRNAseq

Configurable mapping and quality control for plant single-cell and single-nucleus RNA-seq, with optional integration, marker discovery, annotation, and refinement workflows.

The same mapping/QC implementation handles cells and nuclei through sample metadata, reference/chemistry configuration, and selectable QC profiles. Parameters remain explicit: modality-agnostic does not mean every species, preparation, or chemistry should use identical settings.

## Supported scope

| Component | Status |
|---|---|
| Mapping | `alevin`: simpleaf + piscem + alevin-fry |
| Chemistries | 10x v2/v3, BD Rhapsody Enhanced Bead V2, DNBelab C4 |
| QC | Configurable metrics and filters; `off`, `advisory`, and `filter` modes; HTML review and separate manual-refilter output |
| Ambient RNA / doublets | Existing CellBender and Scrublet/SOLO workflows, configured per run |
| KB / STARsolo | **Not implemented end-to-end.** Placeholders retained; execution is deliberately blocked. |
| Optional downstream stages | Integration/reference mapping, marker identification, ontology annotation, refinement, public-data fetching |

## Start here

Run from a **source checkout**, using the stage-specific conda environments in [envs/](envs/README.md). `pip install` is not a complete supported installation. The current scientific environments target Linux; configured CellBender/SOLO runs need a GPU. The documented CellBender and BD piscem patches are part of the reproducible setup.

1. Follow [mapping setup](docs/setup-mapping.md), including environment creation and patch verification.
2. Copy the [minimal example](configs/examples/minimal/README.md) to `configs/local/`. Set FASTQ, reference, whitelist, and environment locations. Check the species-specific organelle identifiers before using another plant reference.
3. From the checkout root, in `scMappingV3.12`, initialize the mapping tools and validate the chosen run:

```bash
source batch_scripts/lib/mapping_setup.sh
sc_mapping_setup
python -m shared.mapping_setup -c configs/local/pipeline.yaml
python -m mapping.validate -c configs/local/pipeline.yaml
python -m mapping.main -c configs/local/pipeline.yaml --build-indexes
```

The example contains no FASTQs/reference data and does not run until those inputs and environment prefixes are supplied. It preserves existing default scientific parameters as a starting example, not a universal plant QC recommendation. GPU preflight can report unavailable CUDA on a login node; perform the run on the configured compute resource.

Read the per-sample HTML reports, then use the [manual-review workflow](docs/outputs.md). Existing study configurations at `configs/*.yaml` are preserved as the research record and contain cluster-specific paths; the quick start does not use them.

## Commands and configuration

`python -m sc --help` lists the convenience dispatcher. `python -m sc map ...` selects the mapping environment; it still requires the mapping-tool setup above. Direct `python -m <stage>.main` commands remain supported. `sc` does not schedule dependencies or automatically invalidate cached results.

```bash
python -m sc status -c configs/local/pipeline.yaml
python -m sc test portable
python -m sc test
```

See [configuration and path rules](docs/configuration.md), [output contracts](docs/outputs.md), [testing](docs/testing.md), and [batch launchers](batch_scripts/README.md). Optional workflows have separate [integration](docs/setup-integration.md) and [annotation](docs/setup-annotation.md) guides. The [documentation index](docs/README.md) links the scientific rationale.

## Reuse and attribution

A license for the project's own code is **not yet selected**. This release preparation does not assign one. See the [third-party inventory](docs/third-party.md) for bundled ontology/patch origins and the remaining attribution checks before public release.

For release preparation and preserving existing results, see [releasing](docs/releasing.md). Raw data, indices, results, private notes, and compiled tools are excluded from Git.
