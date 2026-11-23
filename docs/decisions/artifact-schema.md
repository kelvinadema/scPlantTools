# `.uns['pipeline']` schema history

Extracted from `shared/io.py`, 2026-08-06. `SCHEMA_VERSION` there is the current
value; this is what each bump meant. Bump it on any breaking change to the
`.uns['pipeline']` payload or the manifest.

## 1.1 — tool versions

Added `tool_versions`. Additive; 1.0 consumers are unaffected.

## 1.2 — embeddings in the terminal artifact

Added `.obsm` embeddings: `X_umap_qc`/`leiden_qc` always, plus
`X_umap_scVI`/`leiden_scVI`/`solo_doublet*` when doublet detection ran.

Additive — nothing was removed, and the neighbour graph (`.obsp`,
`uns['neighbors']`) is still deliberately excluded. See
`qc-terminal-umap.md`.

## 1.3 — the embedding became conditional

This is a bump rather than another additive note because **1.2 promised
`X_umap_qc`/`leiden_qc` on every artifact** and 1.3 does not.

A sample the QC cascade reduces below `qc_plot._MIN_CELLS_FOR_EMBEDDING` has no
embedding to record. Previously that case raised inside `DimReducer` and cost
the whole sample — artifact and manifest included.

**Consumers must now check for the key.** Why it is absent is recorded in
`.uns['umap_qc_status']` (`{computed, skip_reason, n_cells}`), so "no embedding"
can be told apart from "older artifact".

Also additive: `.uns['qc']['post_cascade']`, the count of cells surviving the
whole cascade.

## 1.4 — a second doublet caller

Adds Scrublet alongside SOLO, and their consensus. All **additive**:

- `.obs['scrublet_doublet_score']`, `.obs['flag_scrublet']`
- `.obs['flag_doublet_consensus']` — the SOLO-AND-Scrublet intersection
- `.uns['qc']['scrublet']` and `.uns['qc']['doublet_consensus']`

Any of the three (solo / scrublet / consensus) is present only when that method
ran, and can remove cells only in `filter` mode; consensus exists only when
**both** callers produced calls. Consumers unaware of these columns are
unaffected, and the existing `solo_*` columns are unchanged.
