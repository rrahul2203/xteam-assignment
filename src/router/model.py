"""The classifier pipeline.

Word and character TF-IDF features into a logistic regression. Character n-grams let an
unseen phrasing match on shared substrings ("unauthoriz", "log in"/"login") and absorb
typos, which word features cannot.

Class weights are inverse-frequency, then skewed further toward fraud-report: the four
routes do not cost the same, so equalising them is not the right target. Missing a fraud
report loses money, over-flagging costs an analyst minutes. `solve_skew` sizes that skew
from data -- the largest one whose fraud precision still clears MIN_FRAUD_PRECISION -- so
the input is how much analyst load the queue can absorb, and the weight follows.

The vectorisers live inside the Pipeline, so a train/test split can never fit them on
held-out rows.
"""
import logging
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.utils.class_weight import compute_class_weight

from .crossval import N_SPLITS, over_seeds
from .data import FRAUD

log = logging.getLogger(__name__)

# Set from `python3 -m src.router.tune`, which grid-searches these under the grouped CV used
# to report scores. C is held one step below the search's winner on purpose: the top-ranked
# value buys its macro-F1 by shedding fraud recall, and recall is the guardrail.
WORD_NGRAMS = (1, 2)
CHAR_NGRAMS = (2, 4)
C = 10.0

SEARCH_SPACE = {
    "features__word__ngram_range": [(1, 1), (1, 2)],
    "features__char__ngram_range": [(2, 4), (2, 5), (3, 5)],
    "classifier__C": [1.0, 3.0, 10.0, 30.0],
}

# The one number that is a business input rather than a measurement: how much fraud-queue
# precision the analysts can absorb. The fraud weighting follows from it.
MIN_FRAUD_PRECISION = 0.60
SKEW_CANDIDATES = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)

# Cached output of solve_skew() so build_pipeline() stays cheap. Re-run
# `python3 -m src.router.tune` to refresh it after the data or feature settings change.
FRAUD_SKEW = 2.5

TUNE_SEEDS = (0, 1, 2, 3, 4)


def fraud_first_weights(labels, skew=None, target=FRAUD):
    """Inverse-frequency weights from the observed labels, then push `target` above parity.

    `skew=None` uses the cached FRAUD_SKEW; solve_skew passes explicit candidates instead.
    """
    classes = sorted(Counter(labels))
    weights = dict(zip(
        classes,
        compute_class_weight("balanced", classes=np.asarray(classes), y=np.asarray(labels)),
    ))
    if target in weights:
        weights[target] *= FRAUD_SKEW if skew is None else skew
    return {c: float(w) for c, w in weights.items()}


def pipeline_factory(skew=None, params=None):
    """A `build(labels) -> unfitted pipeline` callable for crossval.

    Weights come from the fold's own labels, so not even the class distribution leaks.
    """
    def build(fold_labels):
        pipe = build_pipeline(labels=fold_labels, skew=skew)
        if params:
            pipe.set_params(**params)
        return pipe
    return build


def build_pipeline(class_weight=None, labels=None, skew=None):
    """TF-IDF word+char features into a logistic regression.

    With no explicit `class_weight`, derives the fraud-first weights from `labels`.
    """
    if class_weight is None and labels is not None:
        class_weight = fraud_first_weights(labels, skew=skew)
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(
                ngram_range=WORD_NGRAMS, min_df=1, sublinear_tf=True)),
            ("char", TfidfVectorizer(
                analyzer="char_wb", ngram_range=CHAR_NGRAMS, min_df=2, sublinear_tf=True)),
        ])),
        ("classifier", LogisticRegression(
            max_iter=2000, C=C, class_weight=class_weight)),
    ])


def train(texts, labels, class_weight=None, skew=None):
    """Fit a pipeline on labelled messages."""
    return build_pipeline(class_weight, labels=labels, skew=skew).fit(texts, labels)


def solve_skew(texts, labels, groups_, floor=MIN_FRAUD_PRECISION,
               candidates=SKEW_CANDIDATES, seeds=TUNE_SEEDS, target=FRAUD):
    """Largest skew whose fraud precision still clears `floor`, measured under grouped CV.

    Skew trades fraud precision for fraud recall, so the floor is the binding constraint.
    Falls back to the smallest candidate if none clears it.
    """
    measured = []

    for skew in candidates:
        precisions = over_seeds(
            texts, labels, pipeline_factory(skew=skew),
            lambda y, preds: precision_score(
                y, preds, labels=[target], average="macro", zero_division=0),
            seeds=seeds, groups_=groups_,
        )
        precision = float(np.mean(precisions))
        measured.append((skew, precision))
        log.info("skew %.2f -> fraud precision %.4f", skew, precision)

    passing = [s for s, p in measured if p >= floor]
    if not passing:
        log.warning(
            "no skew clears the %.2f fraud-precision floor; falling back to %.2f",
            floor, candidates[0],
        )
        return candidates[0], measured
    return max(passing), measured


def tune(texts, labels, groups_, seeds=TUNE_SEEDS, space=None):
    """Grid-search the feature and regularisation settings under grouped CV.

    Averaged over several seeds, because at this group count a single split picks its
    winner by noise. Returns (best_params, ranked) with ranked as (mean, std, params).

    GridSearchCV needs a splitter, not out-of-fold predictions, so it can't use crossval.
    Weights here come from all labels, which is why reported scores come from `evaluate`.
    """
    space = space or SEARCH_SPACE
    X, y, g = np.asarray(texts), np.asarray(labels), np.asarray(groups_)
    scores = {}

    for seed in seeds:
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        search = GridSearchCV(
            build_pipeline(labels=y), space, scoring="f1_macro", cv=cv,
            n_jobs=-1, refit=False,
        )
        search.fit(X, y, groups=g)
        for score, params in zip(
            search.cv_results_["mean_test_score"], search.cv_results_["params"]
        ):
            scores.setdefault(tuple(sorted(params.items())), []).append(score)

    ranked = sorted(
        ((float(np.mean(v)), float(np.std(v)), dict(k)) for k, v in scores.items()),
        key=lambda row: -row[0],
    )
    log.info("best config %s at macro-F1 %.4f", ranked[0][2], ranked[0][0])
    return ranked[0][2], ranked
