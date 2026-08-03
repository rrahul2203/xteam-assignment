"""Turns a ranked hit into a grounded answer, or declines to answer.

Answers are extractive: the returned text is sentences copied from the cited document, so no
LLM is involved and grounding is structural rather than promised.

Returns one of three statuses. ANSWERED and UNANSWERED are the usual pair; LAPSED covers the
case where the only relevant document expired before `as_of`, which is reported as a negative
answer naming the closed window rather than as an abstention.

Abstention thresholds two signals jointly: an unmatched question word long enough to be a
domain noun, and the share of the question's rare vocabulary the top document covers. Cosine is
not used for this, being length-normalised and so insensitive to how much of the question a
document accounts for. Cutoffs come from the sweep in eval_answers.py.
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

# Both cutoffs are set from the threshold sweep in eval_answers.py.
# Length at which an unmatched question word counts as a domain noun rather than filler.
MIN_DOMAIN_TERM_LENGTH = 6
# Share of the question's IDF-weighted vocabulary the top document must cover to answer.
MIN_SALIENT_COVERAGE = 0.35

REFUSAL = (
    "Not answerable from the knowledge base. No document in force on {as_of} covers this, "
    "so this needs a human agent."
)

# Words excluded from the coverage and absent-term signals. sklearn's list supplies the grammar
# words; the additions are ordinary English a financial KB would not contain anyway.
_STOP = frozenset(ENGLISH_STOP_WORDS) | frozenset("""
money moved fast happens pay long need want know like make going lost use using make sure
""".split())

# Words marking a question as asking for a quantity, used to prefer sentences with numbers.
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
    """Strips inflectional suffixes so a question word matches its KB form.

    Tries the longest suffix first and repeats, so plural and past forms converge on the stem
    the base form reaches. The length guard stops short words eroding into collisions. Both the
    KB and the query pass through here, so the two sides always agree.
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

    Built date-blind on purpose: it answers whether a topic is documented at all, which is a
    separate question from which version of it applies.
    """
    vocabulary = set()
    for doc in _values(docs):
        for word in content_words(f"{doc.title} {doc.body}"):
            vocabulary.add(_stem(word))
    return vocabulary


def missing_mass(question, vocabulary):
    """Share of the question's content words that appear nowhere in the KB.

    Carried on the Answer for diagnostics and not thresholded on: a count cannot tell a missing
    domain noun from a missing date, which is what absent_domain_terms looks at instead.
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

    Weighting each term by rarity means a document matching only the question's common words
    scores near zero, where an unweighted overlap would credit it.
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
    """The `limit` body sentences that best overlap the question, returned in document order.

    Overlap is counted on stems. Sentences containing a digit get a small bonus when the
    question asks for a quantity, so the quoted text carries the figure asked for.
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
        # Switchable so eval_answers.py can ablate the second abstention signal.
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
        # Decline only when both signals agree, unless the conjunction is ablated off.
        if coverage < self.min_coverage and (absent or not self.require_absent_term):
            return self._refuse(text, when, top, gap, coverage, absent)

        if top.is_lapsed:
            # Apply the coverage bar again here without the absent-term escape, so a thinly
            # matching expired notice falls through to the best in-force hit instead.
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
