# Third-party material and attribution

The project owner has left the license for original code undecided. No project-wide license has been assigned by this cleanup. Before publication as reusable open-source software, select the intended terms and complete the upstream notice review below.

| Material | Recorded origin | What is preserved / remains to verify |
|---|---|---|
| `annotation/po_final.obo` | Plant Ontology, header release `2024-04-17` | Header declares CC BY 4.0; local processing is described in prior research notes. Preserve the file bytes and record its checksum. |
| `annotation/nodule.obo` | Project-specific ontology extension | Keep separate provenance from Plant Ontology. |
| `envs/cellbender-checkpoint-weakref.patch` | CellBender checkpoint fix, upstream PR #444 recorded in the environment guide | Preserve patch and origin; verify/carry the applicable upstream notice before release. |
| `piscem_bd_rhapsody/*.patch` | piscem v0.21.1 / piscem-rs 0.6.3 / seq_geom_parser 1.1.0 | Preserve the patch set, rebuild instructions, and geometry probes; verify the notices for each modified upstream component. |
| `markerid/lmd.py`, `markerid/cfmf.py` | Python ports of LocalizedMarkerDetector and project_CFMF | Keep [port notes](markerid-porting.md), citations, and documented deviations; confirm upstream reuse terms and required notices before licensing the ports. |

The [environment guide](../envs/README.md) and [piscem patch documentation](../piscem_bd_rhapsody/README.md) record source links and installation details. This inventory does not assert that one future project license covers third-party material.

FASTQs, genome annotations, whitelists, proteomes, and marker references are user-supplied inputs. Record their source/release/checksum separately; their redistribution terms are not decided by this repository's software license. Previously described `resources/ontology/attic` and `resources/manual_annotations` are absent from this checkout and are not advertised as bundled assets.
