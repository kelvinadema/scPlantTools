# Testing

Suites are separated by environment because stage dependencies differ. Run from the checkout root. Portable tests need Python 3.12, pytest, and PyYAML; they must not require the developer's live results or installed stage environments.

```bash
python -m pytest tests/portable -q
python -m sc test                 # required portable + stage suites
python -m sc test --fast          # explicitly excludes slow end-to-end suite
python -m sc test mapping         # one declared environment
python -m sc test site            # opt-in diagnostics of the local study setup
```

| Suite | Environment | Scope |
|---|---|---|
| portable | active interpreter | Config resolution, package layering, launch/setup controls, release consistency |
| mapping | scMappingV3.12 | Existing QC and artifact contracts |
| integration | scIntegrationV1.12 | Existing target/model/config and derived-layer behavior |
| annotation | scAnnotationV3.12 | Existing ontology/proteome behavior |
| markerid | scMarkersV3.12 | Existing marker-method and label interfaces |
| e2e | scMappingV3.12 | Synthetic FASTQ → count matrix with real simpleaf/piscem/alevin-fry |
| site | active interpreter | Installed environments, study cohort-sheet availability, live output-layout checks |

Site diagnostics retain the original checks and may fail for unfinished fetches or historical output layouts. They are not hidden by the portable release result. Investigate failures without deleting data or changing scientific defaults.

The synthetic smoke uses a deterministic generated permit list and data in a temporary directory. Its assertions still check which genes and cells receive counts. Missing tools fail the explicitly selected end-to-end suite; all seven tests must run for a release. It does not train CellBender/SOLO or verify all chemistries.

The dispatcher fails on missing required suites and pytest's no-tests-collected status. `--fast` reports its deliberate end-to-end exclusion. Test results alone are not proof of numerical equivalence after a scientific-code change; the release procedure also protects existing source/configuration bytes.

The portable CI workflow uses GitHub's [checkout](https://github.com/actions/checkout) and [setup-python](https://github.com/actions/setup-python) actions with Python 3.12. CI setup does not install or upgrade the scientific stage environments.
