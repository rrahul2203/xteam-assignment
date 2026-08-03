"""Date-filtered retrieval over the KB, lexical and semantic scores fused.

Filters to the documents in force for the query date, then ranks only those. Ranking first and
dropping out-of-window hits afterwards would let a superseded document win and be discarded,
leaving a worse in-force document unranked.

Two signals, because they fail differently. TF-IDF matches shared words, so it is precise on doc
ids, figures and exact terms, and blind when the question and the document name the same thing
differently. Embeddings match meaning, so they bridge that gap and are correspondingly vaguer
about which of several similar documents is meant. Both are cosines, so they are fused as a
weighted sum of the raw values rather than rescaled per query: min-max normalising would map the
best candidate to 1.0 on every question, and the abstention gate in answerer.py reads an
absolute score.

MODE_HYBRID degrades to lexical scoring on its own when no embedding vectors are available, so
the repo runs with only scikit-learn installed. `Retriever.describe()` reports which applied.
"""
import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion

from .embeddings import DocumentVectors, can_encode, doc_text, encode
from .kb import Doc, in_force, lapsed, load_kb, require_date

log = logging.getLogger(__name__)

WORD_NGRAMS = (1, 2)
CHAR_NGRAMS = (3, 5)

MODE_HYBRID = "hybrid"
MODE_TFIDF = "tfidf"
MODE_EMBEDDING = "embedding"
MODES = (MODE_HYBRID, MODE_TFIDF, MODE_EMBEDDING)

# Weight on the embedding signal when both are present. Centred in the flat region of the sweep
# in eval_answers.py rather than at its edge, so the choice does not sit on a cliff.
EMBEDDING_WEIGHT = 0.45

# Cosine below which the top hit is treated as no hit. Set from the sweep in eval_answers.py.
MIN_SCORE = 0.08


_warned_lexical = False


def _warn_lexical_once():
    """Logs the fallback notice once per process rather than once per Retriever.

    The evaluation builds a Retriever per mode and per swept weight, which would otherwise bury
    the report under twenty copies of the same line.
    """
    global _warned_lexical
    if not _warned_lexical:
        _warned_lexical = True
        log.info("embeddings unavailable, ranking lexically only "
                 "(pip install -r requirements-embeddings.txt to enable them)")


@dataclass(frozen=True)
class Hit:
    doc: Doc
    score: float
    as_of: date
    lexical: float = 0.0
    semantic: float = 0.0

    @property
    def doc_id(self):
        return self.doc.doc_id

    @property
    def is_lapsed(self):
        """True when this document's window had already closed by `as_of`."""
        return self.doc.has_lapsed(self.as_of)


def validate_question(question):
    """A question must be a non-empty string with something wordlike in it."""
    if not isinstance(question, str):
        raise TypeError(f"question must be a string, got {type(question).__name__}")
    text = question.strip()
    if not text:
        raise ValueError("question is empty")
    if not any(c.isalnum() for c in text):
        raise ValueError(f"question has no alphanumeric content: {question!r}")
    return text


def _build_vectoriser():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=WORD_NGRAMS, sublinear_tf=True,
                                 stop_words="english")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=CHAR_NGRAMS,
                                 sublinear_tf=True, min_df=2)),
    ])


class Retriever:
    """Ranks KB documents against a question, as at a date."""

    def __init__(self, docs=None, mode=MODE_HYBRID, embedding_weight=EMBEDDING_WEIGHT,
                 vectors=None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.docs = docs if docs is not None else load_kb()
        self.mode = mode
        self.embedding_weight = embedding_weight

        self.vectors = None
        if mode in (MODE_HYBRID, MODE_EMBEDDING):
            self.vectors = vectors if vectors is not None else DocumentVectors(self.docs)
            # Document vectors alone are not enough: the query has to be encoded per call, which
            # needs the model. With the fixture but no model, the semantic signal cannot fire.
            usable = self.vectors.available() and can_encode()
            if not usable:
                if mode == MODE_EMBEDDING:
                    raise RuntimeError(
                        "embedding mode needs sentence-transformers installed to encode the "
                        "question; only document vectors are cached in eval/doc_vectors.npz")
                _warn_lexical_once()
                self.vectors = None

    def describe(self):
        """Which signals are actually in use, for the CLI and the evaluation header."""
        if self.mode == MODE_TFIDF or self.vectors is None:
            return MODE_TFIDF
        if self.mode == MODE_EMBEDDING:
            return MODE_EMBEDDING
        return MODE_HYBRID

    def candidates(self, as_of):
        """Documents eligible to answer for `as_of`: in force, plus expired-and-never-replaced.

        Returned as one pool so a single fitted vectoriser ranks both kinds. Cosines from two
        separate fits are not comparable, and scoring a small pool on its own fit inflates it.
        `Hit.is_lapsed` tells the caller which kind won.
        """
        return in_force(self.docs, as_of) + lapsed(self.docs, as_of)

    def _lexical(self, text, docs):
        """Cosine over word and character TF-IDF, refitted on this candidate set."""
        vectoriser = _build_vectoriser()
        try:
            matrix = vectoriser.fit_transform([doc_text(d) for d in docs])
            query = vectoriser.transform([text])
        except ValueError:
            # Raised when every term is a stop word or below min_df, leaving no shared features.
            log.debug("no usable lexical features for %r", text)
            return np.zeros(len(docs))
        return cosine_similarity(query, matrix)[0]

    def _semantic(self, text, docs):
        """Cosine between the encoded question and the cached document vectors.

        Returns zeros when the query cannot be encoded, which is the case for a question typed
        at the CLI with no model installed, since the fixture only covers documents.
        """
        if self.vectors is None:
            return np.zeros(len(docs))
        matrix, usable = self.vectors.matrix(docs)
        if matrix is None:
            return np.zeros(len(docs))
        query = encode([text])
        if query is None:
            log.debug("no model to encode the query, lexical scoring only")
            return np.zeros(len(docs))

        # Scatter back to the full candidate order, leaving uncovered documents at zero.
        scored = dict(zip((d.doc_id for d in usable), matrix @ query[0]))
        return np.array([scored.get(d.doc_id, 0.0) for d in docs])

    def search(self, question, as_of, k=3, pool=None):
        """Top `k` hits by fused score, highest first."""
        text = validate_question(question)
        as_of = require_date(as_of)
        docs = list(pool) if pool is not None else self.candidates(as_of)
        if not docs:
            return []

        lexical = self._lexical(text, docs)
        semantic = self._semantic(text, docs)

        if self.mode == MODE_TFIDF:
            fused = lexical
        elif self.mode == MODE_EMBEDDING:
            fused = semantic
        else:
            # Raw cosines, so the fused score stays on a comparable scale across questions and
            # the absolute abstention threshold keeps its meaning.
            weight = self.embedding_weight if semantic.any() else 0.0
            fused = (1 - weight) * lexical + weight * semantic

        ranked = sorted(range(len(docs)), key=lambda i: -fused[i])[:k]
        return [
            Hit(doc=docs[i], score=float(fused[i]), as_of=as_of,
                lexical=float(lexical[i]), semantic=float(semantic[i]))
            for i in ranked
        ]
