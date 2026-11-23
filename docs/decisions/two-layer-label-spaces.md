# Two-layer label spaces: spliced and unspliced as separate features

Added 2026-09-10 for Chapter 3 session C11 (`notebooks/03-sc-vs-sn/plan/subplans/C11-label-space-construction.md`).
Four additive target keys, one per-sample derivation, tests in
`tests/integration/test_two_layer_features.py` and `tests/portable/test_labelspace_keys.py`.
**No target that states none of the keys trains differently than before.**

## The problem

A shared nuclei + protoplast label space trains on `corrected_counts`, which
for a nucleus is ≈ s_g + u_g with u_g dominant and u_g/s_g a property of the
gene (intron length, retention, transcription rate). The same cell type
therefore has a different gene ranking in the two modalities, HVG selection on
the concatenated object picks modality-discriminating genes first, and the
encoder has to learn a per-gene modality map. Restricting to spliced counts
would cost nuclei 50–65 % of their detected genes in Medicago and maize. The
user chose (2026-09-10) to give the model **both layers as separate features**
rather than drop one.

## What was added

| key | where it acts | default |
|---|---|---|
| `derive_corrected_layers: true` | `orchestrator._concatenate_adata`, per sample, before `keep_layers` drops anything | off |
| `layer: [a, b]` | `integrate._prep_scvi`, builds the 2 × genes feature object | `corrected_counts` |
| `hvg_batch_key:` | `HVGSelector.select_hvg(batch_key=…)` | `batch` |
| `exclude_genes:` | `integrate._prep_scvi`, after concatenation, before HVG | none |

**Ambient-corrected spliced/unspliced layers.** `rspliced`/`runspliced` are
pre-CellBender; `corrected_counts` is post-CellBender on their sum. Per gene per
cell, `corrected_rspliced = floor(corrected × rs/(rs+ru))` and
`corrected_runspliced = corrected − corrected_rspliced`. The two are
non-negative integers that sum exactly to `corrected_counts`. `rs + ru == 0`
with `corrected > 0` raises: CellBender removes counts, never adds, so that is
a layer mismatch and not a case to allocate. Exact only if the removed ambient
counts share the cell's own s:u ratio; the misallocation is bounded by the
removed fraction (1–7 % of counts per cell on this cohort), i.e. second order.
Floor-the-spliced-share was chosen over stochastic rounding because it needs
no seed and is reproducible byte-for-byte. The derivation record goes to
`.uns['corrected_layers']` (per sample on the concatenated target).

**Two layers as features.** `stack_layers` builds a derived AnnData whose var is
the gene space doubled, features named `<gene>__s` / `<gene>__u`, `.var['gene']`
mapping each back. HVG selection, `min_cells` inside `HVGSelector`, and the
model all see that object. The gene-space object is kept aside and the latent
is written back onto it, with `.var['hvg_spliced']`, `.var['hvg_unspliced']`,
`.var['hvg']` (either) and `.uns['labelspace']`. So markerid, annotation scoring
and refine still find `corrected_counts` and every gene, exactly as after the
2026-08-18 "save every gene" change. SysVI derives its log-normalised input from
whatever matrix it is given (`_ensure_gaussian_input`), so it works on the
doubled matrix unchanged; the CP10K total is then over 2 × genes, i.e. the
cell's corrected total.

**Per-modality HVG aggregation.** `select_hvg` always aggregated rankings per
batch, but on the target's `batch_key` (per library). `hvg_batch_key:
sample_type` balances the gene set between the two modalities instead of
letting the larger libraries dominate it, while the model still corrects per
library. Two separate choices, two separate keys.

**Exclusions.** A gene-list file (organelle + ribosomal-protein genes here;
later C9's isolation-affected sets) removed from the training object only. The
saved h5ad keeps them, flagged `.var['excluded']`.

## Not done, on purpose

- No spliced-only arm (revivable as a fallback if the two-layer arms do not
  beat the baselines).
- The registry reload path (`_integrate_scvi`'s cached-model branch) loads the
  saved gene-space h5ad into the model; for any HVG-trained model that object
  has more vars than the model, which is a pre-existing condition of that path
  and not changed here.
- `exclude_genes` matches ids exactly as written in `.var_names`; no
  normalisation. The chapter builds its lists from one sample's `.var`.
