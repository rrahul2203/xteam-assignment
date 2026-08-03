"""Tests for saving and loading the router model.

The point of the artifact is that prediction reads a model instead of fitting one, so these check
the two things that would break that: a round trip has to predict identically to the model it was
saved from, and the prediction path must not train.
"""
import logging

import pytest

from src.router.artifact import data_digest, load_model, save_model
from src.router.data import default_data_path, load
from src.router.model import train
from src.router.predict import get_pipeline

MESSAGES = [
    "I see a transfer I never made",
    "I can't log in to my account",
    "how do I enable price alerts",
    "this charge on my statement is wrong",
]


@pytest.fixture(scope="module")
def training_data():
    return load(default_data_path())


@pytest.fixture
def saved(tmp_path):
    """An artifact written to a temporary path, so tests never touch models/router.joblib."""
    return save_model(path=tmp_path / "router.joblib")


class TestRoundTrip:
    def test_a_loaded_model_predicts_identically_to_the_one_saved(self, saved, training_data):
        """Persistence is only useful if it changes nothing, in label or in confidence.

        Confidence is checked as well as the label because it gates the review queue.
        """
        texts, labels = training_data
        pipeline, _ = load_model(saved)
        expected = train(texts, labels)

        assert list(pipeline.predict(MESSAGES)) == list(expected.predict(MESSAGES))
        assert pipeline.predict_proba(MESSAGES) == pytest.approx(expected.predict_proba(MESSAGES))

    def test_metadata_records_what_the_model_was_trained_on(self, saved):
        """Provenance is the reason to save at all: an audit needs data and settings."""
        _, metadata = load_model(saved)
        assert metadata["n_training_rows"] == 400
        assert metadata["data_digest"] == data_digest()
        assert metadata["settings"]["fraud_skew"] > 1.0


class TestPredictionDoesNotTrain:
    """Fitting is proportional to the training set, so it must not happen per prediction."""

    def test_the_prediction_path_loads_or_raises_but_never_trains(self, saved, tmp_path,
                                                                  monkeypatch):
        """With training sabotaged, loading still works and a missing artifact raises.

        Both halves guard the same thing from opposite sides: silently training on a cache miss
        would hide the very cost this artifact exists to remove.
        """
        monkeypatch.setattr("src.router.predict.train",
                            lambda *a, **k: pytest.fail("prediction path trained a model"))
        assert get_pipeline(model_path=saved) is not None
        with pytest.raises(FileNotFoundError, match="src.router.artifact"):
            get_pipeline(model_path=tmp_path / "absent.joblib")

    def test_retrain_opts_back_into_training(self, saved):
        """The escape hatch has to work, for development against edited data."""
        assert list(get_pipeline(model_path=saved, retrain=True).predict(MESSAGES))


class TestStaleArtifacts:
    def test_a_stale_artifact_warns_but_still_loads(self, saved, monkeypatch, caplog):
        """A stale model must stay usable: refusing to load would break the CLI after any edit."""
        monkeypatch.setattr("src.router.artifact.data_digest", lambda *a, **k: "0000deadbeef")
        with caplog.at_level(logging.WARNING):
            pipeline, _ = load_model(saved)
        assert pipeline is not None
        assert "training data has changed" in caplog.text
