# scPlantTools

A pipeline for mapping and quality control (QC) of plant single-cell and single-nucleus RNA-seq data, with optional stages for integration, marker gene identification, orthology-based cell type annotation and label refinement.

Cells and nuclei go through the same mapping and QC code. Differences between preparations, species and chemistries are set in the sample sheet, the reference and chemistry definitions and the QC profiles, not in the code. The default QC settings were chosen for Medicago and Arabidopsis root data and should be checked before use on other material.

## Stages

| Stage | Command | Conda environment |
|---|---|---|
| Mapping and QC | `python -m mapping.main` | `scMappingV3.12` |
| Manual refiltering | `python -m mapping.refilter` | `scMappingV3.12` |
| Integration and reference mapping | `python -m integration.main` | `scIntegrationV1.12` |
| Marker identification | `python -m markerid.main` | `scMarkersV3.12` |
| Annotation | `python -m annotation.main` | `scAnnotationV3.12` |
| Refinement | `python -m refine.main` | `scIntegrationV1.12` |
| Public data download | `python -m fetch.main` | `scFetch` |

Mapping uses simpleaf, piscem and alevin-fry, keeping spliced, unspliced and ambiguous counts. Ambient RNA is removed with CellBender, doublets are scored with SOLO and Scrublet, and each sample gets an HTML report that can also be used to select cells for removal by hand. Every QC filter can be set to `off`, `advisory` (computed and reported, but removes no cells) or `filter`.

Supported chemistries are 10x v2 and v3, BD Rhapsody Enhanced Bead V2 and DNBelab C4. kb-python and STARsolo appear in the configuration as placeholders only; they are not implemented, and samples that request them are rejected before mapping starts.

## Installation

Run the pipeline from a clone of this repository; it is not installed with `pip`. Each stage has its own conda environment, defined in [envs/](envs/README.md), because their dependencies cannot be combined. The environments target Linux, and CellBender and SOLO need a GPU. Two environments need an extra step after creation (a patched piscem for BD Rhapsody data and a CellBender patch); both are described in [envs/README.md](envs/README.md).

## Quick start

1. Create the environments and apply the patches as described in [mapping setup](docs/setup-mapping.md).
2. Copy the [minimal example](configs/examples/minimal/README.md) to `configs/local/` and fill in the FASTQ, reference, whitelist and environment paths. If you work with a species other than Medicago, adapt the organelle gene identifiers.
3. From the repository root, in `scMappingV3.12`, set up the mapping tools, validate the configuration and run:

```bash
source batch_scripts/lib/mapping_setup.sh
sc_mapping_setup
python -m shared.mapping_setup -c configs/local/pipeline.yaml
python -m mapping.validate -c configs/local/pipeline.yaml
python -m mapping.main -c configs/local/pipeline.yaml --build-indexes
```

The example contains no data and will not run until those paths are set. On a login node the GPU check may report that CUDA is unavailable; run the pipeline itself on a GPU node.

4. Inspect the per-sample HTML reports and, if needed, refilter by hand as described in [outputs](docs/outputs.md).

The YAML files directly under `configs/` are the configurations used for our own studies. They contain cluster-specific paths and are not needed for a new dataset.

## The `sc` command

`python -m sc` runs a stage in its own conda environment and reports what has been produced so far. It does not schedule stages or decide what needs rerunning.

```bash
python -m sc --help
python -m sc status -c configs/local/pipeline.yaml
python -m sc test portable
python -m sc test
```

`python -m sc map ...` still requires the mapping setup above. The stages can also be run directly with `python -m <stage>.main`.

## Documentation

- [Mapping setup](docs/setup-mapping.md), [configuration](docs/configuration.md) and [outputs](docs/outputs.md)
- [Integration](docs/setup-integration.md) and [annotation](docs/setup-annotation.md)
- [Testing](docs/testing.md) and [batch scripts](batch_scripts/README.md)
- The [documentation index](docs/README.md), which also links the design notes in `docs/decisions/`
