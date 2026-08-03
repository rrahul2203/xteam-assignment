"""Date-filtered TF-IDF retrieval over the KB.

Filters to the documents in force for the query date, then ranks only those. Ranking first and
dropping out-of-window hits afterwards would let a superseded document win and be discarded,
leaving a worse in-force document unranked.

Features are word plus character n-grams, for the reason Part A's model.py gives: character
features let an unseen phrasing match on shared substrings and absorb typos. The vectoriser is
refitted per query date, since the candidate corpus changes with `as_of`.
"""
import logging
from dataclasses import dataclass
from datetime import date

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion

from .kb import Doc, in_force, lapsed, load_kb, require_date

log = logging.getLogger(__name__)

WORD_NGRAMS = (1, 2)
CHAR_NGRAMS = (3, 5)

# Cosine below which the top hit is treated as no hit. Set from the sweep in eval_answers.py.
MIN_SCORE = 0.08


@dataclass(frozen=True)
class Hit:
    doc: Doc
    score: float
    as_of: date

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

    def __init__(self, docs=None):
        self.docs = docs if docs is not None else load_kb()

    def candidates(self, as_of):
        """Documents eligible to answer for `as_of`: in force, plus expired-and-never-replaced.

        Returned as one pool so a single fitted vectoriser ranks both kinds. Cosines from two
        separate fits are not comparable, and scoring a small pool on its own fit inflates it.
        `Hit.is_lapsed` tells the caller which kind won.
        """
        return in_force(self.docs, as_of) + lapsed(self.docs, as_of)

    def search(self, question, as_of, k=3, pool=None):
        """Top `k` hits by cosine, highest first."""
        text = validate_question(question)
        as_of = require_date(as_of)
        docs = list(pool) if pool is not None else self.candidates(as_of)
        if not docs:
            return []

        corpus = [f"{d.title}. {d.title}. {d.body}" for d in docs]
        vectoriser = _build_vectoriser()
        try:
            matrix = vectoriser.fit_transform(corpus)
            query = vectoriser.transform([text])
        except ValueError:
            # Raised when every term is a stop word or below min_df, leaving no shared features.
            log.debug("no usable features for %r", text)
            return []

        scores = cosine_similarity(query, matrix)[0]
        ranked = sorted(zip(docs, scores), key=lambda pair: -pair[1])
        return [Hit(doc=d, score=float(s), as_of=as_of) for d, s in ranked[:k]]
