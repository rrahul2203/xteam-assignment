"""Search the hyperparameters under grouped CV and log the ranking.

    python3 src/tune.py

The winner is what WORD_NGRAMS / CHAR_NGRAMS / C in model.py are set to, and the solved
skew is what FRAUD_SKEW is set to. Re-run after the training data changes. Fraud recall is
logged alongside macro-F1 because a config that wins on macro-F1 while shedding fraud
recall is not an improvement here.
"""
import logging

import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

from data import FRAUD, default_data_path, groups, load
from model import (
    MIN_FRAUD_PRECISION,
    build_pipeline,
    fraud_first_weights,
    solve_skew,
    tune,
)

log = logging.getLogger(__name__)

SEEDS = (0, 1, 2, 3, 4)


def fraud_metrics(texts, labels, groups_, params, seeds=SEEDS):
    """Fraud recall and precision for one config, same protocol as the ranking."""
    X, y, g = np.asarray(texts), np.asarray(labels), np.asarray(groups_)
    recalls, precisions = [], []

    for seed in seeds:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        preds = np.empty(len(y), dtype=object)
        for train_i, test_i in cv.split(X, y, groups=g):
            pipe = build_pipeline(fraud_first_weights(y[train_i]))
            pipe.set_params(**params)
            preds[test_i] = pipe.fit(X[train_i], y[train_i]).predict(X[test_i])
        recalls.append(recall_score(y, preds, labels=[FRAUD], average="macro"))
        precisions.append(
            precision_score(y, preds, labels=[FRAUD], average="macro", zero_division=0)
        )
    return float(np.mean(recalls)), float(np.mean(precisions))


def main():
    texts, labels = load(default_data_path())
    g = groups(texts)
    log.info("%d messages, %d templates, %d seeds", len(texts), len(set(g)), len(SEEDS))

    best, ranked = tune(texts, labels, g, seeds=SEEDS)

    log.info("%10s %7s   config", "macro-F1", "std")
    for mean, std, params in ranked[:8]:
        log.info(
            "%10.4f %7.4f   word=%s char=%s C=%g",
            mean, std,
            params["features__word__ngram_range"],
            params["features__char__ngram_range"],
            params["classifier__C"],
        )

    recall, precision = fraud_metrics(texts, labels, g, best)
    log.info("best config %s", best)
    log.info("fraud recall %.4f, precision %.4f", recall, precision)

    skew, _ = solve_skew(texts, labels, g, seeds=SEEDS)
    log.info("solved fraud skew %.2f at a %.2f precision floor -> FRAUD_SKEW",
             skew, MIN_FRAUD_PRECISION)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
