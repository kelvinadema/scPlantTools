# Testing

The tests are split by conda environment, because the stages cannot be imported into one interpreter. Run them from the repository root. The portable tests need only Python 3.12, pytest and PyYAML, and do not depend on local data or the stage environments.

```bash
python -m pytest tests/portable -q
python -m sc test                 # portable and all stage suites
python -m sc test --fast          # without the slow end-to-end suite
python -m sc test mapping         # one suite
python -m sc test site            # checks of the local installation and study data
```

| Suite | Environment | Covers |
|---|---|---|
| portable | any | Configuration resolution, package layering, launcher and setup checks, release consistency |
| mapping | scMappingV3.12 | QC filters and output contracts |
| integration | scIntegrationV1.12 | Targets, model configuration and derived layers |
| annotation | scAnnotationV3.12 | Ontology traversal and proteome preparation |
| markerid | scMarkersV3.12 | Marker methods and label input |
| e2e | scMappingV3.12 | Synthetic FASTQ to count matrix with the real simpleaf, piscem and alevin-fry |
| site | any | Installed environments, availability of study sample sheets, layout of existing results |

The site suite checks our own installation and may fail for downloads that have not finished or for older result layouts. It is not part of the default run.

The end-to-end test generates a whitelist and reads in a temporary directory and checks which genes and cells receive counts. It fails, rather than skips, when the mapping tools are missing. It does not run CellBender or SOLO and does not cover every chemistry.

`sc test` fails when a suite is missing or collects no tests. Passing tests do not show that a change to analysis code leaves results unchanged; compare outputs for that.

The GitHub Actions workflow runs the portable suite and `tools/check_release.py` with Python 3.12. It does not install the stage environments.
