# Running Ray Tune under this pipeline

Rationale extracted from `integration/integrate.py` comments, 2026-08-06.
See also `autotune-search-spaces.md` and `autotune-failure-diagnosis.md`.

## `RAY_TQDM_PATCH_PRINT=0`, set before `import ray`

Ray Tune's progress reporter permanently replaces `builtins.print` with its own
tqdm-safe wrapper (`ray/experimental/tqdm_ray.py::instance`, guarded only by
this env var) and never restores it — not on `ray.shutdown()`, not at exit.

numba can only type the genuine builtin `print`, so once it is a plain Python
function **every subsequent JIT compile of a function containing a `print` call
fails** with `Untyped global name 'print'`. pynndescent's
`nn_descent_internal_low_memory_parallel` has exactly that (under `if verbose:`,
but numba types all branches) and is decorated with a bare `@numba.njit()` — no
`cache=`, so it recompiles in every process and there is no cached artifact to
fall back on.

The result is that any `sc.pp.neighbors`/UMAP **after** a tuning run dies — but
only above 4096 cells, where scanpy switches from exact sklearn neighbours to
pynndescent. That is why it surfaced on a large sample (BDTip2mm, 46,799 cells)
while small ones tuned and embedded fine.

Set unconditionally and before `import ray`: there is no case where we want Ray
owning `print`.

## Sizing `ray.init` from the real allocation

`_cluster_resources` reports what the job actually has; `_get_allocated_resources`
divides that up per trial.

Both used to be guessed — `ray.init(num_cpus=16, num_gpus=2)` was hardcoded,
which fails outright on any allocation that is not exactly two GPUs. Ray
cross-checks the count against `CUDA_VISIBLE_DEVICES` and raises *"Attempting to
start raylet with 2 GPU, but CUDA_VISIBLE_DEVICES contains ['0']"*. A single-GPU
job — the normal `--gres=gpu:1` case, and what per-sample doublet tuning runs
under — therefore could never tune.

GPUs come from `torch.cuda.device_count()`, which already respects
`CUDA_VISIBLE_DEVICES`, so it cannot disagree with what ray sees. SLURM's own
variables are consulted only for CPUs, where there is no equivalent runtime
probe that respects the allocation — `os.cpu_count()` reports the whole node.

`ray.is_initialized()` guards the per-sample case: doublet tuning constructs one
`SingleCellIntegrate` per sample in a single process, and a second `ray.init`
would raise.

## Trial logs are staged through node-local scratch

scvi's `AutotuneExperiment` hands `logging_dir` straight to Ray Tune's
`RunConfig(storage_path=...)`, where Ray writes both the per-trial results (one
subdir per trial, up to `num_samples` of them) and the global
`experiment_state.json` it re-serialises on **every trial event**.

Left at scvi's default it lands in `./scvi_log/` under cwd; pointed at `out_dir`
it lands on Lustre — and Lustre is slow enough at rewriting that growing state
file that Ray's own 30s sync check fires (*"Saving the experiment state has
already taken N seconds"*). So it is staged through `$TMPDIR`, the same reason
mapping's `IndexBuilder` does: fast local disk for the churn, nothing partial on
the network filesystem. Scoped by target so two parallel tuning runs on one node
cannot collide.

**Scratch is wiped when the job ends**, so the failure path copies the trial
logs to `out_dir/_tmp_tune` before raising — best-effort, since a copy failure
must not mask the real tuning failure. A clean success deletes them instead.
This follows CellBender's temp-file convention: a run worth inspecting is a
failed one.

## `max_t` tracks `max_epochs`

ASHA's `max_t` is the per-trial epoch budget the successive-halving rungs are
laid out against, so it is read from `train_args['max_epochs']` (falling back to
400 when unset). Left at a hardcoded 400 while trials stopped at scvi's
~123-epoch heuristic, the highest rungs never got trials.
