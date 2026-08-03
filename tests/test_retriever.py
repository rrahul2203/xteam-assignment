"""Tests for the ranking layer: fusion arithmetic, mode selection, and the no-torch fallback.

The fallback tests matter most. The brief requires the repo to run without the optional
dependency, so they simulate its absence rather than trusting that the code path works.
"""
from datetime import date

import numpy as np
import pytest

from src.qa.embeddings import DocumentVectors, can_encode, doc_text, load_model
from src.qa.fetch_model import fetch_file
from src.qa.retriever import (
    MODE_EMBEDDING,
    MODE_HYBRID,
    MODE_TFIDF,
    Retriever,
    validate_question,
)

AS_OF = date(2026, 7, 28)
DISPUTE = "How many days does a customer have to raise a dispute?"

# Tests that need the query encoded are skipped rather than failed on a clone that installed only
# requirements.txt, so a reviewer without the optional extra still sees a green suite.
needs_model = pytest.mark.skipif(
    not can_encode(), reason="needs sentence-transformers: pip install -r requirements-embeddings.txt"
)


@pytest.fixture
def block_sentence_transformers(monkeypatch):
    """Simulates a clone without the optional extra, where the import left the name None."""
    monkeypatch.setattr("src.qa.embeddings.SentenceTransformer", None)
    # load_model caches its result, including failure, so the cache is cleared either side.
    monkeypatch.setattr("src.qa.embeddings._model", None)
    monkeypatch.setattr("src.qa.embeddings._model_tried", False)
    yield
    monkeypatch.setattr("src.qa.embeddings._model_tried", False)


class TestModeSelection:
    def test_unknown_mode_is_rejected(self, kb):
        with pytest.raises(ValueError, match="mode must be one of"):
            Retriever(kb, mode="magic")

    def test_tfidf_mode_never_touches_embeddings(self, kb):
        retriever = Retriever(kb, mode=MODE_TFIDF)
        assert retriever.vectors is None
        assert retriever.describe() == MODE_TFIDF

    def test_describe_reports_lexical_when_embeddings_are_unavailable(
            self, kb, block_sentence_transformers):
        """Hybrid asked for, lexical delivered: describe() must not overstate what ran."""
        retriever = Retriever(kb, mode=MODE_HYBRID)
        assert retriever.describe() == MODE_TFIDF

    def test_embedding_mode_raises_rather_than_silently_degrading(
            self, kb, block_sentence_transformers):
        """An explicit request for semantic ranking must fail loudly if it cannot be honoured."""
        with pytest.raises(RuntimeError, match="sentence-transformers"):
            Retriever(kb, mode=MODE_EMBEDDING)


class TestFallback:
    """Without the optional dependency the service still answers, using lexical ranking."""

    def test_search_still_returns_dated_hits(self, kb, block_sentence_transformers):
        hits = Retriever(kb, mode=MODE_HYBRID).search(DISPUTE, AS_OF, k=3)
        assert hits
        assert all(h.as_of == AS_OF for h in hits)
        assert all(h.semantic == 0.0 for h in hits)
        assert hits[0].lexical > 0

    def test_date_resolution_is_unaffected_by_the_missing_dependency(
            self, kb, block_sentence_transformers):
        """The date layer is independent of ranking, so the fallback must resolve dates too."""
        retriever = Retriever(kb, mode=MODE_HYBRID)
        assert retriever.search(DISPUTE, date(2026, 6, 30))[0].doc_id == "kb-031"
        assert retriever.search(DISPUTE, date(2026, 7, 1))[0].doc_id == "kb-032"


@needs_model
class TestFusion:
    """Fusion arithmetic, which only has two signals to combine when the model is present."""

    def test_weight_zero_and_one_isolate_the_two_signals(self, kb):
        """A hybrid at the extremes has to reproduce each pure mode's ranking."""
        lexical = Retriever(kb, mode=MODE_TFIDF).search(DISPUTE, AS_OF, k=3)
        semantic = Retriever(kb, mode=MODE_EMBEDDING).search(DISPUTE, AS_OF, k=3)

        at_zero = Retriever(kb, mode=MODE_HYBRID, embedding_weight=0.0)
        at_one = Retriever(kb, mode=MODE_HYBRID, embedding_weight=1.0)
        assert [h.doc_id for h in at_zero.search(DISPUTE, AS_OF, k=3)] == \
            [h.doc_id for h in lexical]
        assert [h.doc_id for h in at_one.search(DISPUTE, AS_OF, k=3)] == \
            [h.doc_id for h in semantic]

    def test_fused_score_is_the_weighted_sum_of_the_parts(self, kb):
        """Hit.score has to be reproducible from Hit.lexical and Hit.semantic."""
        weight = 0.45
        top = Retriever(kb, mode=MODE_HYBRID, embedding_weight=weight).search(
            DISPUTE, AS_OF, k=1)[0]
        # float32 vectors, so the tolerance is single-precision rather than double.
        assert top.score == pytest.approx(
            (1 - weight) * top.lexical + weight * top.semantic, rel=1e-6)

    def test_scores_are_ordered_and_bounded(self, kb):
        hits = Retriever(kb, mode=MODE_HYBRID).search(DISPUTE, AS_OF, k=5)
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
        # Both parts are cosines over non-negative features, so the fusion stays in [0, 1].
        assert all(0.0 <= h.score <= 1.0 for h in hits)

    def test_embeddings_bridge_a_vocabulary_gap_tfidf_cannot(self, kb):
        """The question says "withdraw"; the document in force says "transfers out".

        This is the case that motivated adding embeddings, so it is pinned as a test: with the
        semantic signal the right fee document wins, without it a different one does.
        """
        question = "What fee do I pay to withdraw crypto from my account?"
        semantic = Retriever(kb, mode=MODE_EMBEDDING).search(question, AS_OF, k=1)[0]
        lexical = Retriever(kb, mode=MODE_TFIDF).search(question, AS_OF, k=1)[0]
        assert semantic.doc_id == "kb-013"
        assert lexical.doc_id != "kb-013"


class TestModelLoading:
    """The model must be constructed once per process, not once per question."""

    def test_the_model_is_built_once_and_then_cached(self, monkeypatch):
        builds = []

        def counting(source, *args, **kwargs):
            builds.append(source)
            return object()

        monkeypatch.setattr("src.qa.embeddings.SentenceTransformer", counting)
        monkeypatch.setattr("src.qa.embeddings._model", None)
        monkeypatch.setattr("src.qa.embeddings._model_tried", False)

        assert [load_model() for _ in range(5)].count(None) == 0
        assert len(builds) == 1
        monkeypatch.setattr("src.qa.embeddings._model_tried", False)

    def test_a_local_checkout_is_preferred_over_the_hub(self, monkeypatch, tmp_path):
        """Loading from disk is what keeps the first question off the network."""
        builds = []
        monkeypatch.setattr("src.qa.embeddings.LOCAL_MODEL_DIR", tmp_path)
        monkeypatch.setattr("src.qa.embeddings.SentenceTransformer",
                            lambda source, *a, **k: builds.append(source) or object())
        monkeypatch.setattr("src.qa.embeddings._model", None)
        monkeypatch.setattr("src.qa.embeddings._model_tried", False)

        load_model()
        assert builds == [str(tmp_path)]
        monkeypatch.setattr("src.qa.embeddings._model_tried", False)

    def test_fetch_model_skips_files_already_present(self, tmp_path):
        """Re-running the fetch must not re-download, so it is cheap to put in a setup script."""
        (tmp_path / "vocab.txt").write_text("already here")
        assert fetch_file("vocab.txt", tmp_path).read_text() == "already here"


class TestDocumentVectors:
    def test_fixture_covers_the_whole_kb(self, kb):
        vectors = DocumentVectors(kb)
        assert vectors.available()
        assert vectors.covered == set(kb), "run python3 -m src.qa.build_vectors after editing kb/"

    def test_vectors_are_l2_normalised(self, kb):
        """Ranking treats a dot product as the cosine, which only holds for unit vectors."""
        matrix, _ = DocumentVectors(kb).matrix(list(kb.values()))
        assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4)

    @needs_model
    def test_a_missing_fixture_falls_back_to_encoding(self, kb, tmp_path):
        """With the model installed, an absent fixture is re-encoded rather than being fatal."""
        vectors = DocumentVectors(kb, path=tmp_path / "absent.npz")
        assert not vectors.from_fixture
        assert vectors.available()

    def test_a_corrupt_fixture_is_ignored_rather_than_raising(self, kb, tmp_path):
        """Construction must survive an unreadable fixture whether or not a model can replace it."""
        bad = tmp_path / "bad.npz"
        bad.write_bytes(b"not an npz file")
        vectors = DocumentVectors(kb, path=bad)
        assert not vectors.from_fixture
        assert vectors.available() == can_encode()

    def test_documents_without_vectors_are_dropped_not_zero_filled(self, kb):
        """A zero row would score zero against every query and rank as merely irrelevant."""
        vectors = DocumentVectors(kb)
        vectors.vectors.pop("kb-031")
        matrix, usable = vectors.matrix(list(kb.values()))
        assert "kb-031" not in {d.doc_id for d in usable}
        assert matrix.shape[0] == len(usable)

    def test_embedded_text_matches_the_lexical_corpus(self, kb):
        """Both signals must index the same text, or they rank different things."""
        doc = kb["kb-031"]
        assert doc_text(doc).startswith(f"{doc.title}. {doc.title}.")


class TestInputValidation:
    @pytest.mark.parametrize("bad, exception", [
        (None, TypeError), (42, TypeError), ("", ValueError), ("!!!", ValueError),
    ])
    def test_bad_questions_are_rejected_before_ranking(self, bad, exception):
        with pytest.raises(exception):
            validate_question(bad)

    def test_an_empty_candidate_pool_returns_no_hits(self, kb):
        """A date before the KB begins has nothing in force, which is not an error."""
        assert Retriever(kb, mode=MODE_HYBRID).search(DISPUTE, date(1999, 1, 1)) == []
