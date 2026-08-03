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
        """Persistence is only useful if it changes nothing about the predictions."""
        texts, labels = training_data
        pipeline, _ = load_model(saved)
        assert list(pipeline.predict(MESSAGES)) == list(train(texts, labels).predict(MESSAGES))

    def test_probabilities_survive_the_round_trip(self, saved, training_data):
        """Confidence gates the review queue, so it has to be preserved, not just the label."""
        texts, labels = training_data
        pipeline, _ = load_model(saved)
        expected = train(texts, labels).predict_proba(MESSAGES)
        assert pipeline.predict_proba(MESSAGES) == pytest.approx(expected)

    def test_metadata_records_what_the_model_was_trained_on(self, saved):
        """Provenance is the reason to save at all: an audit needs data and settings."""
        _, metadata = load_model(saved)
        assert metadata["n_training_rows"] == 400
        assert metadata["data_digest"] == data_digest()
        assert metadata["settings"]["fraud_skew"] > 1.0
        assert sorted(metadata["labels"]) == metadata["labels"]


class TestPredictionDoesNotTrain:
    """Fitting is proportional to the training set, so it must not happen per prediction."""

    def test_get_pipeline_never_calls_train(self, saved, monkeypatch):
        def fail(*args, **kwargs):
            raise AssertionError("prediction path trained a model instead of loading one")

        monkeypatch.setattr("src.router.predict.train", fail)
        assert get_pipeline(model_path=saved) is not None

    def test_a_missing_artifact_raises_instead_of_training(self, tmp_path, monkeypatch):
        """Silently training would hide the cost this artifact exists to remove."""
        monkeypatch.setattr("src.router.predict.train",
                            lambda *a, **k: pytest.fail("trained instead of raising"))
        with pytest.raises(FileNotFoundError, match="src.router.artifact"):
            get_pipeline(model_path=tmp_path / "absent.joblib")

    def test_retrain_opts_back_into_training(self, saved, training_data):
        """The escape hatch has to work, for development against edited data."""
        texts, _ = training_data
        pipeline = get_pipeline(model_path=saved, retrain=True)
        assert list(pipeline.predict(MESSAGES))


class TestStaleArtifacts:
    def test_changed_training_data_warns_but_still_loads(self, saved, monkeypatch, caplog):
        """A stale model must stay usable: refusing to load would break the CLI after any edit."""
        monkeypatch.setattr("src.router.artifact.data_digest", lambda *a, **k: "0000deadbeef")
        with caplog.at_level(logging.WARNING):
            pipeline, _ = load_model(saved)
        assert pipeline is not None
        assert "training data has changed" in caplog.text

    def test_a_version_mismatch_warns(self, saved, monkeypatch, caplog):
        monkeypatch.setattr("src.router.artifact.ARTIFACT_VERSION", 99)
        with caplog.at_level(logging.WARNING):
            load_model(saved)
        assert "artifact version" in caplog.text
