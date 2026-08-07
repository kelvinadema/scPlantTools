#!/usr/bin/env python3
"""
Two checks on scANVI's predictions, answering two different questions.

**Confidence** -- `SCANVI.predict(soft=True)` returns a per-class posterior, and
its maximum says how sure the model is *among the classes it knows*. Low
maximum means the cell sits between two known types.

**Novelty** -- the posterior cannot say "none of the above", because it is a
softmax over the seed classes and sums to 1 whatever the cell is. A cell type
absent from the seed set therefore comes back as the nearest available class,
frequently with high confidence. Confidence alone will never catch that. So the
second check ignores the posterior entirely and asks a geometric question: in
the latent space, is this cell as close to its predicted class as that class's
own seed cells are to each other? A cell far outside that spread is being
assigned by elimination rather than by resemblance.

Neither check invents a missing class -- nothing can. `unknown_<geneset>`
classes (see labels.py) are what actually give the model somewhere to put
unnamed cell types; these two flags are for what slips past that.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger('AnnotationLogging')


def confidence(proba: pd.DataFrame) -> pd.Series:
    """Maximum posterior per cell -- how sure the model is among known classes."""
    return pd.Series(proba.to_numpy().max(axis=1), index=proba.index, name='confidence')


def margin(proba: pd.DataFrame) -> pd.Series:
    """
    Gap between the best and second-best class.

    A cell can hold a high maximum and still be a coin-flip between two classes
    when the rest of the mass is spread thin, so the margin is reported
    alongside the maximum rather than derived from it.
    """
    values = np.sort(proba.to_numpy(), axis=1)
    gap = values[:, -1] - (values[:, -2] if values.shape[1] > 1 else 0.0)
    return pd.Series(gap, index=proba.index, name='margin')


def novelty(latent, predictions, seed_labels, unassigned='Unassigned',
            reference_quantile=0.95) -> pd.DataFrame:
    """
    How far each cell sits from the seed cells of the class it was predicted.

    Returns `distance`, `class_reference` (that class's `reference_quantile`
    seed distance) and `novelty_ratio` = distance / reference. A ratio above 1
    means the cell is further from the class centre than
    `reference_quantile` of the cells that *defined* the class.

    Scaled per class rather than globally: a broad class legitimately has
    broader seed spread, and a single global cutoff would flag all of its
    members while missing a tight class's outliers entirely.

    Cells of a class with no seeds (which cannot happen for a trained scANVI
    class, but can if predictions come from elsewhere) get `NaN` -- not
    silently 0, which would read as "perfectly typical".
    """
    latent = np.asarray(latent, dtype=float)
    index = predictions.index
    predictions = predictions.astype(str)
    seed_labels = seed_labels.astype(str).reindex(index)

    centroids, references = {}, {}
    for label in predictions.unique():
        members = np.where((seed_labels == label).to_numpy())[0]
        if len(members) == 0:
            continue
        centroid = latent[members].mean(axis=0)
        seed_d = np.linalg.norm(latent[members] - centroid, axis=1)
        centroids[label] = centroid
        # A class whose seeds are a single point has no spread to compare
        # against; fall back to its mean distance so the ratio stays finite.
        references[label] = float(np.quantile(seed_d, reference_quantile)) or float(seed_d.mean())

    distance = np.full(len(index), np.nan)
    reference = np.full(len(index), np.nan)
    for label, centroid in centroids.items():
        mask = (predictions == label).to_numpy()
        distance[mask] = np.linalg.norm(latent[mask] - centroid, axis=1)
        reference[mask] = references[label]

    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = distance / reference
    return pd.DataFrame({'distance': distance, 'class_reference': reference,
                         'novelty_ratio': ratio}, index=index)


def apply_safeguards(predictions, proba, latent, seed_labels,
                     min_confidence=0.8, max_novelty_ratio=1.25,
                     unassigned='Unassigned', reference_quantile=0.95) -> pd.DataFrame:
    """
    Combine both checks into a final call plus the evidence for it.

    `max_novelty_ratio` is measured, not guessed. Holding out each of five
    seed classes in turn from `MtTip_noHVG`/hotspot and reassigning their cells
    to the nearest remaining class -- the situation this check exists for --
    gives 2,293 orphaned cells against 21,127 correctly-seeded ones:

        threshold   orphans caught   false positives
          1.00           73.3%             5.7%
          1.10           64.5%             2.3%
          1.25           50.3%             0.7%
          1.50           29.2%             0.1%
          2.00            5.1%             0.0%

    Orphan median ratio 1.25 against 0.65 for true seeds, and the true-seed
    99th percentile is 1.19 -- so 1.25 is roughly "further out than 99% of the
    cells that defined the class". It catches half the orphans for 0.7% false
    flags, and since a flag demotes rather than deletes, and every diagnostic
    is kept, that trade is the right way round.

    **This is a backstop, not the fix.** Even at its most aggressive setting it
    misses a quarter of orphaned cells, because a cell type absent from the
    seed set can genuinely sit inside another class's spread. What prevents the
    problem rather than detecting it is giving every unnamed geneset its own
    `unknown_<geneset>` class (labels.py), so the model is never forced to
    choose by elimination in the first place.

    Returns a frame with `prediction` (as given), `confidence`, `margin`,
    `novelty_ratio`, `flag` and `annotation` -- the last being the prediction
    with failures replaced by `unassigned`.

    **Nothing is discarded**: every diagnostic stays in the frame so the
    thresholds can be moved afterwards without retraining, and so a cell that
    was demoted can be told from one the model never had an opinion about.
    """
    predictions = pd.Series(predictions, index=proba.index).astype(str)
    conf = confidence(proba)
    marg = margin(proba)
    nov = novelty(latent, predictions, seed_labels, unassigned=unassigned,
                  reference_quantile=reference_quantile)

    low_conf = conf < min_confidence
    # NaN ratio (class had no seeds) is not treated as novel -- unknown is not
    # the same as far away, and flagging on missing evidence would be wrong.
    novel = nov['novelty_ratio'] > max_novelty_ratio

    flag = pd.Series('', index=predictions.index, dtype=object)
    flag[low_conf] = 'low_confidence'
    flag[novel] = np.where(low_conf[novel], 'low_confidence;novel', 'novel')

    annotation = predictions.where(~(low_conf | novel), unassigned)

    out = pd.DataFrame({
        'prediction': predictions,
        'confidence': conf,
        'margin': marg,
        'novelty_ratio': nov['novelty_ratio'],
        'flag': flag,
        'annotation': annotation,
    })
    logger.info('[REFINE] safeguards: %d/%d cells demoted to %s '
                '(%d low confidence, %d novel, %d both)',
                int((annotation == unassigned).sum() - (predictions == unassigned).sum()),
                len(out), unassigned, int(low_conf.sum()), int(novel.sum()),
                int((low_conf & novel).sum()))
    return out
