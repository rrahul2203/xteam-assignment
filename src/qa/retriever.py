"""Date-filtered TF-IDF retrieval over the KB.

Filter first, rank second. Ranking the whole KB and then dropping out-of-window hits would
let a superseded document win and be discarded, leaving a worse in-force document unranked;
filtering first means the candidate set only ever contains documents that spoke for the date
being asked about.

Features are word plus character n-grams, for the reason Part A's model.py gives: character
features let an unseen phrasing match on shared substrings and absorb typos. The vectoriser
is refitted per query date, because the corpus itself changes with `as_of`.
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

# Below this cosine, the top hit is treated as no hit. Set by the sweep in eval_answers.py.
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
        """Everything eligible to answer for `as_of`.

        In-force documents plus closed-and-never-replaced notices. Both go into one pool so
        they are ranked by a single fitted vectoriser: cosines from two different fits are not
        comparable, and comparing them across pools makes a small pool look artificially
        strong. `Hit.is_lapsed` tells the caller which kind won.
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
            # Every term was a stop word or below min_df, so nothing is comparable.
            log.debug("no usable features for %r", text)
            return []

        scores = cosine_similarity(query, matrix)[0]
        ranked = sorted(zip(docs, scores), key=lambda pair: -pair[1])
        return [Hit(doc=d, score=float(s), as_of=as_of) for d, s in ranked[:k]]
