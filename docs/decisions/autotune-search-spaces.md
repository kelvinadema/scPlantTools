# Autotune search spaces

Rationale extracted from `integration/integrate.py` docstrings and comments,
2026-08-06. See also `autotune-ray-runtime.md` and
`autotune-failure-diagnosis.md`.

## Why the spaces are plain data

`_BASE_SPACE` / `_SEARCH_SPACE_DEFAULTS` hold lists and dicts, not `tune.*`
objects, so the built-in defaults and a `search_space:` block in
`integration_config.yaml` go through exactly one converter (`_to_sampler`).
Building the defaults as ray objects directly and the config ones separately is
how the two drift into meaning different things.

The value grammar is therefore identical in code and in YAML:

| Written | Becomes |
|---|---|
| `[a, b, c]` | `tune.choice([a, b, c])` — a 1-element list pins the parameter |
| `{loguniform: [lo, hi]}` | `tune.loguniform(lo, hi)`, for any sampler in `_SAMPLERS` |
| `null` | the key is dropped entirely, leaving scvi's own default |

`_SAMPLERS` is a whitelist rather than any `getattr(tune, name)`: the search
space is executed, so an arbitrary attribute name out of a YAML file is not
something to hand to `getattr`.

A target's `search_space:` merges over its model class's defaults **per key**.
Note this differs from `resolve_target`'s shallow top-level merge, which
replaces a dict-valued key wholesale — the per-key merge happens in
`resolve_search_space`, against the code defaults.

## The 2026-08-04 narrowing

The DRVI and JointEmbeddingSCVI spaces were narrowed against the tuned configs
already in `results/integration/scvi_tune_registry.json` — 6 DRVI fits and 24
JointEmbeddingSCVI fits. Two patterns drove it:

1. **A value chosen unanimously carries no information and costs a factor.**
   DRVI picked `dispersion: gene-cell` 6/6, so it is pinned.
2. **A value pinned at the *edge* of its range means the range is truncated,
   not that the value is optimal.** Those are extended in that direction rather
   than fixed.

### What the votes were

| Model | Key | Evidence | Action |
|---|---|---|---|
| JointEmbeddingSCVI | `n_layers` | 1 in 18/24, 3 in only 2/24 | narrowed to `[1, 2]` |
| JointEmbeddingSCVI | `dispersion` | gene-cell 20/24, plain gene 1/24 | dropped plain `gene` |
| JointEmbeddingSCVI | `joint_embedding_weight` | floor-pinned at 10.0, 13/24 | extended down to 1.0 |
| JointEmbeddingSCVI | `lambda_off_diag` | 0.01/0.1 split 11/10 | all three kept |
| JointEmbeddingSCVI | `reconstruction_weight` | floor-pinned at 0.5, 17/24 | extended down to 0.25 |
| JointEmbeddingSCVI | `variance_weight` | 13/11, genuinely undecided | both kept |
| DRVI | `n_layers` | ceiling 3 won 6/6; 1 never won | shifted up to `[2, 3, 4]` |
| DRVI | `n_hidden` | ceiling 256 won 5/6 | extended up to 512 |
| DRVI | `dispersion` | unanimous 6/6 | pinned to `gene-cell` |
| DRVI | `gene_likelihood` | `pnb` / `nb` split 3/3 | both kept |
| DRVI | `n_split_latent` | 16 never selected in 6 fits | `[None, 8]` |

Both sets of evidence come from fits on 2000 HVGs. The noise-model choice should
transfer; the capacity parameters are the ones least likely to, which is a
second argument for extending DRVI's ranges upward now that it trains on ~31k
genes rather than pinning them where a much narrower model landed.

**SCVI is unrevised**: the registry holds no tuned plain-SCVI fit, so there is
no evidence here to revise it with. It stays the scvi-tools documented space.

### DRVI specifics

- `n_latent` is kept deliberately wide (`[32, 64, 128]`). DRVI is built for an
  *over-complete* latent — unused splits collapse and you read the active ones —
  so under-sizing is the real risk, and this is the parameter the chapter
  actually reads.
- `pnb` (log-space parametrized NB) is DRVI's own default and what it was tuned
  against; the base space does not offer it.
- `n_split_latent`, `split_aggregation` and `split_method` are the
  disentanglement controls: how the latent is split and how the per-split
  decodings recombine. `None` vs `8` is a structural choice (no splitting at all
  vs splitting) and is kept for exactly that reason.
- DRVI accepts the whole base space — it inherits SCVI's `__init__` and
  `DRVIModule` takes `**kwargs` — so its entry widens the base rather than
  replacing it. Verified against scvi-tools 1.5.0.post1; pinned by
  `tests/integration/test_search_space.py`.

### Base-space single-element keys

`use_layer_norm`, `use_batch_norm` and `encode_covariates` are 1-element lists on
purpose: they are `DRVIModule`'s own defaults, named explicitly so the
constructor receives them rather than being left to scvi's, but there is nothing
to search.
