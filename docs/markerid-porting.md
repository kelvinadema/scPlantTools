# LMD and CFMF — Python port notes

Both methods are published only as R packages. They are reimplemented here in
Python rather than shelled out to R, so this stage needs no R interpreter.
These notes record what the original code does, so the ports stay auditable
against the source rather than against a paraphrase of the paper.

Sources read (clone and diff if either is updated):

- LMD — `github.com/KlugerLab/LocalizedMarkerDetector`, `R/calculate_lmds.R`,
  `R/construct_cell_graph.R`, `R/matrix_operation.R`, `src/fastKL.cpp`.
  Paper: Commun Biol 2025, doi:10.1038/s42003-025-08485-y
- CFMF — `github.com/ShuminYin/project_CFMF`, `CFMF/R/core_functions.R`.
  Preprint: doi:10.1101/2025.10.26.683836

---

## LMD (Localized Marker Detector)

Idea: a marker gene's expression, treated as a probability distribution over
cells, stays put under diffusion on the cell-cell graph. A ubiquitous or noisy
gene's distribution washes out immediately. Score each gene by how far its
distribution has moved after 2^t diffusion steps, summed over scales.

### Algorithm, as implemented in `LMD()` -> `fast_get_lmds()`

1. **Cell graph** (`ConstructKnnGraph`, default `knn=5`)
   - kNN on `feature_space` (the cell x coordinate matrix -- this is where
     `X_scVI` goes; the R docs say "e.g. 20 principal components").
   - Directed adjacency `A[i, neighbours(i)] = 1`, then `diag(A) = self_loop`.
   - Symmetrised **unweighted**: `W = min(A + A.T, 1)`, `diag(W) = self_loop`.
     Note `pmin(..., 1)`, not `(A + A.T)/2` -- the weighted form is commented
     out in the original.
   - Connected components smaller than 10 cells are dropped; the rest are
     joined by an MST over the shortest inter-component edges.

2. **Diffusion operator**: `P = Doubly_stochastic(W)`, i.e. symmetric
   Sinkhorn-Knopp to a doubly stochastic matrix (`niter=100`, `tol=1e-8`,
   with the symmetric averaging step `c <- (c + r)/2`).

3. **Initial state**: `rho = RowwiseNormalize(expression)` -- the gene x cell
   matrix with **each gene row summed to 1**. Genes expressed in fewer than
   `min_cell=5` cells are dropped.

4. **Dyadic diffusion**: for `t = 1 .. log2(max_time)`, square the operator
   (`P <- P @ P`) and take `state_t = rho @ P`. Early stop when the diagonal
   has converged to 1/n.

5. **Score per scale**: `score0_t = KL(rho || state_t)` per gene.
   Scale factors, both computed once: `max_score0 = KL(rho || uniform)`
   and `entropy = H(rho)`.

6. **LMDS** (`obtain_lmds`, `correction=FALSE`, the default):

       LMDS = sum_t [ score0_t / max_score0 ]

   The `correction=TRUE` variant divides by `entropy` instead and subtracts a
   `delta_correction` term; it is not the default and is not ported.

### Direction — easy to get backwards

**Lower LMDS = more localized = better marker.** `show_result_lmd` sorts
ascending and ranks 1..n from the smallest score, and the knee point is taken
on that ascending curve. A localized gene's mass stays inside its own
neighbourhood, so `state_t` remains close to `rho` and the KL terms stay small.

### Scale

`P` densifies after a few squarings, so this is O(n^2) memory: 62,750 cells
would need ~31 GB per dense operator. The original handles this with
`CoarseGrain(feature_space, expression, graph.affinity, N=5000)` -- k-means to
N metacells, then `W' = M W M.T` and `expression' = expression @ binarize(M).T`
with `M` row-normalised. The port applies coarse-graining above a cell-count
threshold, defaulting to the original's `N=5000`.

---

## CFMF (Clustering-Free Marker Finding)

Idea: score each gene by whether the cells expressing it sit together in the
embedding. Two statistics per gene -- neighbourhood *purity* and a *silhouette*
on the expressed/not-expressed split.

### Algorithm, as implemented in `RunCFMF()`

1. **Gene filter**: detected in `> 3` cells and in `< 100%` of cells.

2. **Purity** (`calculate.purity`, `neighbor=50`)
   - kNN on the PCA cell embeddings (`BiocNeighbors::findKNN`).
   - For each gene, positive cells = expression `> 0`.
   - For each positive cell `j`: a 2x2 table over all other cells of
     (is a kNN of j) x (is positive), tested with `fisher.test`.
   - Per-gene combination by **Fisher's method**:
     `chisq = -2 * sum(log(p))`, `df = 2 * len(p)`,
     `meta.p = pchisq(chisq, df, lower.tail=FALSE)`, then BH across genes.

3. **Silhouette** (`calculate.silhouette`)
   - Full pairwise distance matrix on the PCA embeddings.
   - Per gene, a 2-group partition (expressed = 1, not = 2), then the
     **median silhouette width over the expressed cells**.

4. Returns gene / silhouette / meta.p / FDR / pct / count.

### Two divergences from the original, both deliberate

**One-sided test.** The original calls `fisher.test(x, y)` with R's default,
which is **two-sided**. A two-sided test also assigns small p-values to genes
whose expressing cells actively *avoid* each other -- the opposite of purity --
and Fisher's method then folds that into the same "pure" score. The port
defaults to `alternative='greater'`; pass `alternative='two-sided'` to
reproduce the original exactly.

The per-cell test has fixed margins (k neighbours, m positive cells, n-1
cells), so one-sided Fisher is exactly the hypergeometric tail. The port
vectorises it with `scipy.stats.hypergeom.sf` over all positive cells at once
instead of looping `fisher.test` per cell -- identical values, and the
difference between minutes and days at this cell count.

**Fisher's method assumes independent p-values, and these are not.**
Neighbouring cells share most of their kNN, so the per-cell tests are strongly
dependent and the combined `meta.p` is anti-conservative -- it will look far
more significant than the evidence supports. This is inherent to the published
method; the port reproduces it but **ranks genes by the statistic rather than
treating `meta.p`/FDR as calibrated**, and records both so the inflation is
visible. Do not report CFMF FDRs as if they were calibrated q-values.

### Scale

Both steps are O(n^2): `stats::dist` on 62,750 cells is ~2e9 pairwise
distances (~16 GB) *per gene* for the silhouette. The published method targets
much smaller datasets. The port subsamples cells for the silhouette step above
a threshold, following the same convention as `qc_filter.py`'s `MAX_DOD_CELLS`
for the other quadratic method in this repo.
