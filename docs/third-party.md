# Third-party material

A licence for this project's own code has not yet been chosen. Before the code is published for reuse, a licence needs to be selected and the notices below checked.

| Material | Origin | Still to check |
|---|---|---|
| `annotation/po_final.obo` | Plant Ontology, release `2024-04-17` | The header states CC BY 4.0. Keep the file unchanged and record its checksum. |
| `annotation/nodule.obo` | Ontology extension written for this project | Keep its provenance separate from Plant Ontology. |
| `envs/cellbender-checkpoint-weakref.patch` | CellBender checkpoint fix, upstream PR #444 | Include the upstream notice. |
| `piscem_bd_rhapsody/*.patch` | Patches to piscem v0.21.1, piscem-rs 0.6.3 and seq_geom_parser 1.1.0 | Include the notice of each patched component. |
| `markerid/lmd.py`, `markerid/cfmf.py` | Python ports of LocalizedMarkerDetector and project_CFMF | Confirm the upstream terms; the [port notes](markerid-porting.md) list citations and deviations. |

Source links and installation details are in the [environment guide](../envs/README.md) and the [piscem patch notes](../piscem_bd_rhapsody/README.md). The licence chosen for this code will not by itself cover the third-party material above.

FASTQ files, genome annotations, whitelists, proteomes and marker tables are supplied by the user and are not distributed here. Record their source, release and checksum; their terms of use are separate from this repository. `resources/ontology/attic` and `resources/manual_annotations` are not part of this repository.
