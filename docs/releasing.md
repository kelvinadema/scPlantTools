# Preparing a release without changing results

The supported distribution is a source checkout with per-stage conda environments. The package metadata is tooling/build metadata, not a complete installed runtime. Keep `.root`, the `sc` dispatcher, stage packages, shared helpers, configuration schemas, and bundled ontology/patch assets.

## Candidate contents

Public files include the README, selected guides, scientific decision documents, maintained setup/launch helpers, tests, example configs, and current source. Private working notes, inspection reports, study-specific batch jobs, incomplete fetch sheets, data/results, indices, caches, and compiled binaries remain ignored. Existing study configurations are retained and clearly distinguished from the minimal example. Current untracked source files must be included when staging the release; the old commit does not represent the current pipeline.

Git operations belong in the actual source checkout, not the execution workspace's symlink farm. Review `git status --short -uall` and the intended diff. No automated release command stages, commits, pushes, or runs production analyses.

## Verification

```bash
python tools/check_release.py
python -m sc test
python -m sc test site
```

`check_release.py` checks required files, public Markdown links, schema agreement, Python/shell syntax, and unintended source symlinks. CI runs the portable suite and the release check on the checkout. Site failures are reported separately; they do not justify deleting results or changing QC values. A full release run must execute the synthetic end-to-end suite rather than merely skip it for missing tools.

Before publishing, inspect an exported candidate tree, not only the developer workspace. Test all public entry points/imports that matter in their existing stage environments. Keep KB/STARsolo explicitly unimplemented. License selection and the notice checks in [third-party inventory](third-party.md) remain owner decisions before an open-source release.

## Baseline and scientific preservation

Before edits, archive the current source (including uncommitted/untracked modules), Git status/diff, exact environment package records, imported versions where package metadata disagrees, binary paths/hashes, and patch hashes. The September 2026 preparation captured these locally under `_snapshots/` outside the repository. Keep the archive private; it contains local working notes and paths.

Preserve active study YAML bytes, analysis modules, ontology/patch assets, and environment manifests. Config hashes currently include raw YAML bytes; reformatting comments can invalidate provenance expectations. Keep timestamps and old provenance inside existing artifacts intact. Store release provenance separately rather than rewriting H5AD files.

For operational changes, prove the successful analysis command, configuration resolution, binary selection, and environment remain equivalent. For an eventual numerical refactor, compare frozen inputs/intermediates in isolated outputs: ordered cell/gene identities, counts/layers, QC metrics and masks, doublet calls, and manual decisions. Seeds do not guarantee bitwise-identical GPU retraining; avoid retraining as an organizational action.

Mapping sentinels and downstream model registries remain unchanged. New file layouts must preserve notebook links and `batch_scripts/lib/common.sh`. A result-directory cleanup is a separate task requiring producer/consumer tracing.
