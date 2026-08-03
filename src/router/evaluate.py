"""Scoring under template-grouped cross-validation.

    python3 -m src.router.evaluate

Macro-F1 weights every route equally regardless of size; fraud-report recall is reported
separately as the guardrail, next to both split protocols so the leakage gap stays visible.
"""
import logging

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_predict

from .crossval import N_SPLITS, SEEDS, grouped_predict, over_seeds
from .data import FRAUD, default_data_path, groups, load
from .model import build_pipeline, pipeline_factory

log = logging.getLogger(__name__)


def _fraud(metric):
    """Scope a metric to the fraud route alone."""
    def score(y, preds):
        return metric(y, preds, labels=[FRAUD], average="macro", zero_division=0)
    return score


METRICS = {
    "macro_f1": lambda y, p: f1_score(y, p, average="macro"),
    "fraud_recall": _fraud(recall_score),
    "fraud_precision": _fraud(precision_score),
}


def grouped_scores(texts, labels, seeds=SEEDS, build=None, metrics=None):
    """Average grouped scores over seeds, all metrics off one CV pass per seed."""
    metrics = metrics or METRICS
    per_seed = over_seeds(
        texts, labels, build or pipeline_factory(),
        lambda y, preds: {name: score(y, preds) for name, score in metrics.items()},
        seeds=seeds,
    )

    summary = {}
    for name in metrics:
        values = [seed_scores[name] for seed_scores in per_seed]
        summary[name] = float(np.mean(values))
        summary[f"{name}_std"] = float(np.std(values))
    return summary


def leakage_gap(texts, labels):
    """Random split (what the handover measured) against grouped (what generalises)."""
    X, y = np.asarray(texts), np.asarray(labels)
    random_preds = cross_val_predict(build_pipeline(labels=y), X, y, cv=N_SPLITS)
    _, grouped = grouped_predict(texts, labels, pipeline_factory())
    return {
        "random_split_macro_f1": float(f1_score(y, random_preds, average="macro")),
        "grouped_split_macro_f1": float(f1_score(y, grouped, average="macro")),
    }


def report(data_path=None):
    texts, labels = load(data_path or default_data_path())
    n_groups = len(set(groups(texts)))

    scores = grouped_scores(texts, labels)
    gap = leakage_gap(texts, labels)
    y, preds = grouped_predict(texts, labels, pipeline_factory())

    log.info("%d messages, %d intent templates", len(texts), n_groups)
    log.info("grouped %d-fold over seeds %s", N_SPLITS, list(SEEDS))
    log.info("macro-F1             %.4f +/- %.4f", scores["macro_f1"], scores["macro_f1_std"])
    log.info("fraud-report recall  %.4f +/- %.4f   <- release guardrail",
             scores["fraud_recall"], scores["fraud_recall_std"])
    log.info("fraud-report prec.   %.4f            <- floor bounds analyst load",
             scores["fraud_precision"])
    log.info("random split macro-F1  %.4f  (what the handover measured)",
             gap["random_split_macro_f1"])
    log.info("grouped     macro-F1  %.4f  (what generalises)",
             gap["grouped_split_macro_f1"])
    log.info("per-class breakdown\n%s", classification_report(y, preds, zero_division=0))

    order = sorted(str(c) for c in set(y))
    log.info("confusion, rows true cols predicted %s\n%s",
             order, confusion_matrix(y, preds, labels=order))
    return scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report()
