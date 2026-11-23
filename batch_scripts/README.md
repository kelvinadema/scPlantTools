# Batch scripts

Submit from the repository root, or set `SCRNASEQ_ROOT`. Create `logs/` before calling `sbatch` so SLURM can write its output files, and adjust the resource and account directives for your cluster.

| File | Purpose |
|---|---|
| `run_mapper.sh` | Mapping launcher; passes `--config`, `--cohort` and any other arguments on to `mapping.main` |
| `lib/mapping_setup.sh` | Tool setup for interactive and batch runs; source it after activating the mapping environment |
| `lib/common.sh` | Shared launcher functions, also used by the notebooks repository |
| `run_refilter.sh` | Manual refiltering with the sample list of one of our studies; for new data use the command in [outputs](../docs/outputs.md) |
| `run_integration.sh` | Integration of one of our study targets; for other targets run `python -m integration.main` |
| `run_annotation.sh` | Annotation launcher; check its configuration and resources before use |

The mapping and refilter launchers exit with the status of the command they run, so failures are visible to SLURM. Other launchers used for specific studies are kept outside this repository.

The general setup and commands are described in [mapping setup](../docs/setup-mapping.md) and [outputs](../docs/outputs.md).
