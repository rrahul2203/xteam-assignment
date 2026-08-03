"""Turning a ranked hit into a grounded answer, or declining to.

Answers are extractive: the text returned is sentences copied from the document. Nothing
paraphrases, so an answer cannot drift from its source, and `doc_ids` is checkable by reading
the file. That is the whole grounding argument -- no LLM is involved.

Three outcomes, not two. ANSWERED and UNANSWERED are obvious; LAPSED is the one worth
arguing for. When the only relevant document expired before `as_of`, "no, and here is the
window that closed" is a real answer carrying real information, so abstaining there would
throw away something the system knows.

Abstention thresholds two signals together rather than one. Cosine on its own is a poor
discriminator here: it is length-normalised, so a question sharing a couple of words with a
short document scores like one sharing many with a long document.

The signal that does discriminate is *which* question words are missing rather than how many.
An undocumented topic tends to carry a domain noun found nowhere in the corpus, while an
answerable question is typically missing only filler and dates, and a plain count treats those
alike. So both must hold to decline: a long unmatched term is present AND the top document
covers little of the question's rare vocabulary. eval_answers.py prints the distributions and
the threshold sweep the cutoffs were read off.

A lapsed document faces the same coverage bar without the absent-term escape, since announcing
an unrelated expiry is a confidently wrong answer rather than a helpful one.
"""
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .kb import ancestors, require_date
from .retriever import MIN_SCORE, Retriever, validate_question

log = logging.getLogger(__name__)

ANSWERED = "answered"
LAPSED = "lapsed"
UNANSWERED = "unanswered"

MAX_SENTENCES = 2

# Both cutoffs come from the sweep in eval_answers.py.
# Minimum length for an absent question word to count as a domain noun rather than filler,
# since short unmatched tokens tend to be dates and contractions.
MIN_DOMAIN_TERM_LENGTH = 6
# Share of the question's IDF-weighted vocabulary that the top document must cover.
MIN_SALIENT_COVERAGE = 0.35

REFUSAL = (
    "Not answerable from the knowledge base. No document in force on {as_of} covers this, "
    "so this needs a human agent."
)

# sklearn's list rather than a hand-rolled one: it covers the grammar words that inflate the
# coverage signal while leaving domain terms alone, which is the split this metric needs.
# Extended with ordinary English words a financial KB would not contain, since their absence
# says nothing about whether the topic is documented.
_STOP = frozenset(ENGLISH_STOP_WORDS) | frozenset("""
money moved fast happens pay long need want know like make going lost use using make sure
""".split())

# Question words that ask for a quantity, so an answer without a number is suspect.
_NUMERIC_CUES = frozenset("""
how much how many what fee rate limit cap ceiling cost charge percent long
""".split())

_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class Answer:
    question: str
    as_of: date
    text: str
    doc_ids: list = field(default_factory=list)
    status: str = UNANSWERED
    score: float = 0.0
    missing_mass: float = 0.0
    coverage: float = 0.0

    @property
    def answered(self):
        return self.status in (ANSWERED, LAPSED)

    def doc_ids_field(self):
        """Semicolon-separated, per the output contract."""
        return ";".join(self.doc_ids)


def content_words(text):
    """Lowercased alphanumeric tokens with stop words and bare numbers removed."""
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and not w.isdigit()]


def _stem(word):
    """Crude suffix strip, so an inflected question word matches its KB form.

    Applied to the KB and the query through the same function, so the two sides always agree.
    Longest suffixes first and stripping iterated, so plural and past forms of a word converge
    on the stem its base form reaches. The length guard stops short words eroding into
    collisions.
    """
    stem = word
    for _ in range(2):
        for suffix in ("ations", "ation", "ures", "ure", "ings", "ing", "ers", "er",
                       "als", "al", "ed", "es", "s", "e"):
            if stem.endswith(suffix) and len(stem) - len(suffix) >= 3:
                stem = stem[: -len(suffix)]
                break
        else:
            break
    return stem


def kb_vocabulary(docs):
    """Every stemmed content word appearing anywhere in the KB, at any date.

    Deliberately date-blind: the question is whether the product documents this topic at all,
    which is separate from which version applies.
    """
    vocabulary = set()
    for doc in _values(docs):
        for word in content_words(f"{doc.title} {doc.body}"):
            vocabulary.add(_stem(word))
    return vocabulary


def missing_mass(question, vocabulary):
    """Share of the question's content words that appear nowhere in the KB.

    Reported for diagnostics only. It is not thresholded on, because a count cannot tell a
    missing domain noun from a missing date, which is why the abstention rule pairs signals.
    """
    words = content_words(question)
    if not words:
        return 1.0
    unknown = [w for w in words if _stem(w) not in vocabulary]
    return len(unknown) / len(words)


def absent_domain_terms(question, vocabulary, min_length=MIN_DOMAIN_TERM_LENGTH):
    """Question words long enough to be domain nouns that appear nowhere in the KB."""
    return [
        w for w in content_words(question)
        if len(w) >= min_length and _stem(w) not in vocabulary
    ]


def document_frequencies(docs):
    """How many documents each stemmed term appears in, for IDF weighting."""
    frequencies = Counter()
    for doc in _values(docs):
        for stem in {_stem(w) for w in content_words(f"{doc.title} {doc.body}")}:
            frequencies[stem] += 1
    return frequencies


def salient_coverage(question, doc, frequencies, n_docs):
    """Share of the question's IDF-weighted vocabulary that `doc` contains.

    Weighting by rarity is the point: matching a question's common words while missing its one
    rare noun should score as covering nothing that mattered.
    """
    asked = {_stem(w) for w in content_words(question)}
    if not asked:
        return 0.0

    def idf(stem):
        return math.log((n_docs + 1) / (frequencies.get(stem, 0) + 1))

    have = {_stem(w) for w in content_words(f"{doc.title} {doc.body}")}
    total = sum(idf(stem) for stem in asked)
    if total <= 0:
        return 0.0
    return sum(idf(stem) for stem in asked if stem in have) / total


def _wants_number(question):
    words = set(content_words(question)) | {question.lower()[:8].strip()}
    return bool(words & _NUMERIC_CUES) or "how much" in question.lower()


def extract(question, doc, limit=MAX_SENTENCES):
    """The `limit` body sentences that best overlap the question, in document order.

    Overlap is on stems. A sentence carrying a number is nudged up when the question asks for a
    quantity, which guards against citing the right document but quoting a sentence that omits
    the figure asked for.
    """
    sentences = doc.sentences()
    if not sentences:
        return doc.title

    asked = {_stem(w) for w in content_words(question)}
    wants_number = _wants_number(question)

    scored = []
    for index, sentence in enumerate(sentences):
        words = {_stem(w) for w in content_words(sentence)}
        overlap = len(asked & words) / (len(asked) or 1)
        if wants_number and re.search(r"\d", sentence):
            overlap += 0.15
        scored.append((overlap, index, sentence))

    best = sorted(scored, key=lambda row: (-row[0], row[1]))[:limit]
    if not any(row[0] > 0 for row in best):
        return sentences[0]
    return " ".join(sentence for _, _, sentence in sorted(best, key=lambda row: row[1]))


class AnswerService:
    """Holds the KB and the derived vocabulary so batch runs load once."""

    def __init__(self, docs=None, min_score=MIN_SCORE,
                 min_coverage=MIN_SALIENT_COVERAGE, require_absent_term=True):
        self.retriever = Retriever(docs)
        self.docs = self.retriever.docs
        self.vocabulary = kb_vocabulary(self.docs)
        self.frequencies = document_frequencies(self.docs)
        self.min_score = min_score
        self.min_coverage = min_coverage
        # Switchable so the ablation can show what the conjunction is worth.
        self.require_absent_term = require_absent_term

    def answer(self, question, as_of):
        """Answer `question` as at `as_of`, or decline. Returns an Answer."""
        text = validate_question(question)
        when = require_date(as_of)

        gap = missing_mass(text, self.vocabulary)
        hits = self.retriever.search(text, when, k=3)
        top = hits[0] if hits else None

        if top is None or top.score < self.min_score:
            return self._refuse(text, when, top, gap)

        coverage = salient_coverage(text, top.doc, self.frequencies, len(self.docs))
        absent = absent_domain_terms(text, self.vocabulary)
        # Decline only when both signals agree; see the module docstring for why either alone
        # is too weak. `require_absent_term` off drops the conjunction back to coverage.
        if coverage < self.min_coverage and (absent or not self.require_absent_term):
            return self._refuse(text, when, top, gap, coverage, absent)

        if top.is_lapsed:
            # A closed notice speaks only when clearly on topic, so the same coverage bar
            # applies here without the absent-term escape above. Below it, fall through to the
            # best in-force hit rather than announce an expiry the question never asked about.
            if coverage >= self.min_coverage:
                return self._as_lapsed(text, when, top, gap, coverage)
            in_force_hits = [h for h in hits if not h.is_lapsed]
            if not in_force_hits:
                return self._refuse(text, when, top, gap, coverage, absent)
            top = in_force_hits[0]
            coverage = salient_coverage(text, top.doc, self.frequencies, len(self.docs))

        return Answer(
            question=text, as_of=when, text=extract(text, top.doc),
            doc_ids=[top.doc_id], status=ANSWERED, score=top.score, missing_mass=gap,
            coverage=coverage,
        )

    def _as_lapsed(self, text, when, hit, gap, coverage=0.0):
        """Report a closed window as a negative answer rather than an abstention."""
        doc = hit.doc
        ended = doc.valid_until.isoformat() if doc.valid_until else "an earlier date"
        body = (
            f"No, not as at {when.isoformat()}. "
            f"The relevant notice ({doc.title}) applied until {ended} and is no longer "
            f"in force. For the record, it said: {extract(text, doc, limit=1)}"
        )
        return Answer(
            question=text, as_of=when, text=body, doc_ids=[doc.doc_id],
            status=LAPSED, score=hit.score, missing_mass=gap, coverage=coverage,
        )

    def _refuse(self, text, when, top, gap, coverage=0.0, absent=()):
        log.debug("declining %r (top=%s coverage=%.2f absent=%s)", text,
                  f"{top.doc_id}:{top.score:.3f}" if top else None, coverage, list(absent))
        return Answer(
            question=text, as_of=when, text=REFUSAL.format(as_of=when.isoformat()),
            doc_ids=[], status=UNANSWERED,
            score=top.score if top else 0.0, missing_mass=gap, coverage=coverage,
        )

    def history(self, doc_id):
        """Earlier versions of a document, for explaining what changed."""
        doc = self.docs.get(doc_id)
        return ancestors(self.docs, doc) if doc else []


def _values(docs):
    return list(docs.values()) if isinstance(docs, dict) else list(docs)


def answer(question, as_of, service=None):
    """Module-level convenience: `answer(question, as_of) -> Answer`.

    Builds a service per call, so batch work should construct AnswerService directly.
    """
    return (service or AnswerService()).answer(question, as_of)
