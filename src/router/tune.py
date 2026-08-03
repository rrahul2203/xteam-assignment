"""Search the hyperparameters under grouped CV and log the ranking.

    python3 -m src.router.tune

Sets WORD_NGRAMS / CHAR_NGRAMS / C and FRAUD_SKEW in model.py; re-run when the data changes.
Fraud recall is logged beside macro-F1, since a config can win on macro-F1 by shedding it.
"""
import logging

import numpy as np
from sklearn.metrics import precision_score, recall_score

from .crossval import over_seeds
from .data import FRAUD, default_data_path, groups, load
from .model import MIN_FRAUD_PRECISION, pipeline_factory, solve_skew, tune

log = logging.getLogger(__name__)

SEEDS = (0, 1, 2, 3, 4)


def fraud_metrics(texts, labels, groups_, params, seeds=SEEDS):
    """Fraud recall and precision for one config, same protocol as the ranking."""
    per_seed = over_seeds(
        texts, labels, pipeline_factory(params=params),
        lambda y, preds: (
            recall_score(y, preds, labels=[FRAUD], average="macro", zero_division=0),
            precision_score(y, preds, labels=[FRAUD], average="macro", zero_division=0),
        ),
        seeds=seeds, groups_=groups_,
    )
    recalls, precisions = zip(*per_seed)
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
