"""Saving and loading the trained router, with the metadata needed to trust it.

    python3 -m src.router.artifact          # train and write models/router.joblib

Training takes about a tenth of a second and is deterministic, so this is not a speed
optimisation. It exists for provenance: a prediction is only auditable if you can say which
model produced it, what it was trained on and with which settings.

The metadata travels inside the same file as the estimator. Recording the training-data digest
and the library versions is what lets `load_model` warn that an artifact no longer matches the
data or the installed scikit-learn, instead of silently predicting from a stale model.
"""
import hashlib
import logging
from pathlib import Path

import joblib
import sklearn

from . import model as model_module
from .data import default_data_path, load

log = logging.getLogger(__name__)

ARTIFACT_VERSION = 1


def default_model_path():
    return Path(__file__).resolve().parents[2] / "models" / "router.joblib"


def data_digest(path=None):
    """Short SHA-256 of the training file, used to detect an artifact trained on other data."""
    path = Path(path or default_data_path())
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def save_model(path=None, train_path=None):
    """Trains on the labelled data and writes the estimator plus its metadata."""
    train_path = Path(train_path or default_data_path())
    texts, labels = load(train_path)
    pipeline = model_module.train(texts, labels)

    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "pipeline": pipeline,
        "labels": sorted(set(labels)),
        "n_training_rows": len(texts),
        "train_file": train_path.name,
        "data_digest": data_digest(train_path),
        # Settings that produced these coefficients, so a saved model stays interpretable
        # after the module constants are retuned.
        "settings": {
            "word_ngrams": model_module.WORD_NGRAMS,
            "char_ngrams": model_module.CHAR_NGRAMS,
            "C": model_module.C,
            "fraud_skew": model_module.FRAUD_SKEW,
            "min_fraud_precision": model_module.MIN_FRAUD_PRECISION,
        },
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
    }

    path = Path(path or default_model_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    log.info("wrote %s (%d rows, digest %s, sklearn %s)",
             path, len(texts), payload["data_digest"], payload["sklearn_version"])
    return path


def load_model(path=None, check_digest=True):
    """Loads the artifact and returns (pipeline, metadata).

    Raises FileNotFoundError when absent, so the caller decides between training and failing.
    Version and digest mismatches warn rather than raise: a stale artifact still predicts, and
    refusing to load one would make the CLI unusable after any edit to train.csv.
    """
    path = Path(path or default_model_path())
    if not path.exists():
        raise FileNotFoundError(
            f"no saved model at {path}; run python3 -m src.router.artifact to create it")

    payload = joblib.load(path)
    if payload.get("artifact_version") != ARTIFACT_VERSION:
        log.warning("artifact version %s, expected %s; retrain if predictions look wrong",
                    payload.get("artifact_version"), ARTIFACT_VERSION)
    if payload.get("sklearn_version") != sklearn.__version__:
        log.warning("artifact was built with scikit-learn %s, running %s",
                    payload.get("sklearn_version"), sklearn.__version__)
    if check_digest and payload.get("data_digest") != data_digest():
        log.warning("training data has changed since this model was saved; "
                    "re-run python3 -m src.router.artifact")

    metadata = {k: v for k, v in payload.items() if k != "pipeline"}
    return payload["pipeline"], metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    save_model()
