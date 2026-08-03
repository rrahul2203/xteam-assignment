"""Scoring under template-grouped cross-validation.

Reports macro-F1, which weights every route equally regardless of size, and fraud-report
recall separately as the guardrail. `leakage_gap` scores the same data both ways -- random
split and grouped split -- to show what the split choice is worth.
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
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from data import FRAUD, groups, load
from model import build_pipeline

log = logging.getLogger(__name__)

SEEDS = (0, 1, 2)
N_SPLITS = 5


def grouped_predict(texts, labels, seed=0, n_splits=N_SPLITS, class_weight=None):
    """Cross-validated predictions with every template confined to one fold."""
    X, y = np.asarray(texts), np.asarray(labels)
    g = np.asarray(groups(texts))
    preds = np.empty(len(y), dtype=object)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_i, test_i in cv.split(X, y, groups=g):
        # Weights come from the training fold only, like every other fitted quantity.
        pipe = build_pipeline(class_weight, labels=y[train_i]).fit(X[train_i], y[train_i])
        preds[test_i] = pipe.predict(X[test_i])
    return y, preds


def grouped_scores(texts, labels, seeds=SEEDS, class_weight=None):
    """Average grouped scores over several seeds, since 400 rows is a small sample."""
    f1s, recalls, precisions = [], [], []
    for seed in seeds:
        y, preds = grouped_predict(texts, labels, seed=seed, class_weight=class_weight)
        f1s.append(f1_score(y, preds, average="macro"))
        recalls.append(recall_score(y, preds, labels=[FRAUD], average="macro"))
        precisions.append(
            precision_score(y, preds, labels=[FRAUD], average="macro", zero_division=0)
        )
        log.debug("seed %d macro-F1 %.4f fraud recall %.4f", seed, f1s[-1], recalls[-1])
    return {
        "macro_f1": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "fraud_recall": float(np.mean(recalls)),
        "fraud_recall_std": float(np.std(recalls)),
        "fraud_precision": float(np.mean(precisions)),
    }


def leakage_gap(texts, labels):
    """Score the same model under a random split and a grouped split."""
    X, y = np.asarray(texts), np.asarray(labels)
    random_preds = cross_val_predict(build_pipeline(labels=y), X, y, cv=N_SPLITS)
    _, grouped = grouped_predict(texts, labels)
    return {
        "random_split_macro_f1": float(f1_score(y, random_preds, average="macro")),
        "grouped_split_macro_f1": float(f1_score(y, grouped, average="macro")),
    }


def report(data_path):
    texts, labels = load(data_path)
    n_groups = len(set(groups(texts)))

    scores = grouped_scores(texts, labels)
    gap = leakage_gap(texts, labels)
    y, preds = grouped_predict(texts, labels)

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
