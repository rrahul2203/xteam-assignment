"""The grouped cross-validation protocol, in one place.

Every Part A score is measured this way. `build` is a `labels -> unfitted pipeline` callable;
passing a factory keeps this module model-agnostic and forces weights to be fitted per fold.
"""
import logging

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .data import groups

log = logging.getLogger(__name__)

N_SPLITS = 5
SEEDS = (0, 1, 2)


def grouped_predict(texts, labels, build, seed=0, n_splits=N_SPLITS, groups_=None):
    """Out-of-fold predictions with every template confined to one fold."""
    X, y = np.asarray(texts), np.asarray(labels)
    g = np.asarray(groups(texts) if groups_ is None else groups_)
    preds = np.empty(len(y), dtype=object)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_i, test_i in cv.split(X, y, groups=g):
        pipe = build(y[train_i])
        preds[test_i] = pipe.fit(X[train_i], y[train_i]).predict(X[test_i])
    return y, preds


def over_seeds(texts, labels, build, score, seeds=SEEDS, groups_=None, **kwargs):
    """`score(y, preds)` per seed, since at this group count one split picks by noise."""
    return [
        score(*grouped_predict(texts, labels, build, seed=seed, groups_=groups_, **kwargs))
        for seed in seeds
    ]
