# Batch scripts

Submit from the logical execution root; `SCRNASEQ_ROOT` overrides it. Create `logs/` before `sbatch` so the scheduler can open its output files. Review the existing SLURM resource/account directives for your site.

| File | Role |
|---|---|
| `run_mapper.sh` | Maintained mapping launcher; forwards `--config`, `--cohort`, and other mapping arguments |
| `lib/mapping_setup.sh` | Shared interactive/batch tool initialization; source after activating the mapping environment |
| `lib/common.sh` | Existing launch helpers; also sourced by the sibling notebooks repository |
| `run_refilter.sh` | Preserved **study-specific** sample list and parent-mismatch override; use the documented direct command for new datasets |
| `run_integration.sh` | Preserved site/study invocation; use `python -m integration.main` for another target |
| `run_annotation.sh` | Existing site launcher; inspect config and resource settings before reuse |

Other local scripts drive specific studies or maintenance and remain outside the public file set. Their locations and behavior have not been changed. The mapping and refilter launchers propagate the underlying process exit status, including failures.

The portable setup path and generic commands are in [mapping setup](../docs/setup-mapping.md) and [outputs](../docs/outputs.md).
