# Running hotspot's two phases in parallel

Rationale extracted from `markerid/methods.py`, 2026-08-06.

## Why it is worth doing

Both hotspot phases are parallelised, and both used to run serially. They are
roughly 50/50 of hotspot's runtime, measured off `logs/markerID/*.log` on the
2026-08-05 `_fixed` fits:

| Cells | Autocorrelation | Pairwise |
|---|---|---|
| 35,115 | 31 min | 33 min |
| 77,883 | 50 min | 45 min |

Hotspot is **83–90% of the whole six-method marker run**, so this is the only
marker knob that matters at this scale.

## Why it is safe now, having been pinned to `jobs=1`

The pin was attributed to numba's threading layer being initialised before
hotspot forks its Pool. **That attribution was wrong**: hotspot uses only plain
`@jit`/`@njit` — no `parallel=True`, no `prange`, no `set_num_threads` anywhere
in `local_stats.py`, `local_stats_pairs.py` or `knn.py` — so there is no numba
threading layer to inherit.

What the process *does* carry is an **OpenMP/BLAS thread pool**, because every
batch script here exports `OMP_NUM_THREADS`, and `fork()`ing a process with live
OpenMP threads is a textbook reliable SIGSEGV.

`forkserver` forks each worker from a clean single-threaded server process
instead, so no thread pool is ever inherited and the crash cannot occur.

Hotspot needs no patch for this: every Pool it creates passes its state
explicitly through `initargs=[...]` — that is what its `initializer`/
`initializer1/2/3` functions are for — rather than relying on fork-inherited
globals, so it is already start-method agnostic.

## Results are bit-identical, not approximately equal

Both parallel paths are `pool.imap` — order-preserving — over deterministic
per-gene and per-pair functions, with the BH correction applied afterwards on
the assembled frame. No order-dependent reduction and no RNG.

## The cost is memory, and it caps `jobs`

It lands on the pairwise phase. Under `forkserver` the initargs are pickled per
worker instead of shared copy-on-write, and the pairwise initargs include the
dense `n_top_genes x n_cells` counts matrix — ~2.7 GB at 343k cells, so ~22 GB
at 8 workers on top of baseline. That, rather than the CPU count, is what caps
`jobs`.
