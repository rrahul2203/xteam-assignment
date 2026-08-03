"""Loading KB documents and deciding which of them speak for a given date.

The front matter is flat `key: value`, so it is parsed here rather than pulling in a YAML
dependency.

The rule that matters: `status` is never consulted when deciding what is in force. `status`
describes today, and a question can ask about any date, so only `effective_date` and
`valid_until` decide. Treating `status: superseded` as a filter is precisely how a system
answers a 2025 question with a 2026 policy.
"""
import csv
import logging
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

REQUIRED_FIELDS = ("doc_id", "title", "effective_date")

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)
# Split on sentence punctuation only when the next sentence starts, so decimals, money and
# ISO dates stay intact.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    category: str
    version: str
    status: str
    effective_date: date
    valid_until: date | None
    supersedes: str
    superseded_by: str
    body: str

    def in_force(self, as_of):
        """True when `as_of` falls inside the document's window, both ends inclusive."""
        if as_of < self.effective_date:
            return False
        return self.valid_until is None or as_of <= self.valid_until

    def has_lapsed(self, as_of):
        """Window has closed and no successor took over, so the document is the answer."""
        return (
            not self.superseded_by
            and self.valid_until is not None
            and as_of > self.valid_until
        )

    def sentences(self):
        """Body sentences, headings and blank lines dropped."""
        text = " ".join(
            line.strip() for line in self.body.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def parse_date(value, field="date"):
    """ISO date, or None when the field is blank."""
    text = (value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field} is not an ISO date: {text!r}") from exc


def require_date(value, field="as_of"):
    """Same, but the field is mandatory. Accepts a date object unchanged."""
    if isinstance(value, date):
        return value
    parsed = parse_date(value, field)
    if parsed is None:
        raise ValueError(f"{field} is required")
    return parsed


def _parse_front_matter(raw, path):
    match = _FRONT_MATTER.match(raw)
    if not match:
        raise ValueError(f"{path} has no YAML front matter")

    fields = {}
    for line in match.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"{path} has a front-matter line without a colon: {line!r}")
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields, match.group(2)


def load_doc(path):
    """Parse one markdown document. Raises rather than defaulting on bad metadata."""
    fields, body = _parse_front_matter(Path(path).read_text(encoding="utf-8"), path)

    missing = [f for f in REQUIRED_FIELDS if not fields.get(f)]
    if missing:
        raise ValueError(f"{path} is missing front-matter field(s) {missing}")

    effective = require_date(fields["effective_date"], f"{path} effective_date")
    valid_until = parse_date(fields.get("valid_until"), f"{path} valid_until")
    if valid_until is not None and valid_until < effective:
        raise ValueError(f"{path} has valid_until before effective_date")

    return Doc(
        doc_id=fields["doc_id"],
        title=fields["title"],
        category=fields.get("category", ""),
        version=fields.get("version", ""),
        status=fields.get("status", ""),
        effective_date=effective,
        valid_until=valid_until,
        supersedes=fields.get("supersedes", ""),
        superseded_by=fields.get("superseded_by", ""),
        body=body.strip(),
    )


def load_kb(directory=None):
    """Load every .md in `directory`, keyed by doc_id."""
    directory = Path(directory or default_kb_path())
    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise ValueError(f"no .md documents found in {directory}")

    docs = {}
    for path in paths:
        doc = load_doc(path)
        if doc.doc_id in docs:
            raise ValueError(f"duplicate doc_id {doc.doc_id} at {path}")
        docs[doc.doc_id] = doc

    _warn_on_broken_chains(docs)
    return docs


def _warn_on_broken_chains(docs):
    """Log supersession links that do not resolve or do not point back."""
    for doc in docs.values():
        for field in ("supersedes", "superseded_by"):
            target = getattr(doc, field)
            if not target:
                continue
            other = docs.get(target)
            if other is None:
                log.warning("%s %s points at unknown %s", doc.doc_id, field, target)
                continue
            mirror = "superseded_by" if field == "supersedes" else "supersedes"
            if getattr(other, mirror) != doc.doc_id:
                log.warning("%s %s %s is not mirrored by %s.%s",
                            doc.doc_id, field, target, target, mirror)


def load_questions(path=None):
    """Read qid/question/as_of, validating columns and rejecting duplicate qids."""
    with open(path or default_questions_path(), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("question file contains no rows")

    missing = {"qid", "question", "as_of"} - set(rows[0])
    if missing:
        raise ValueError(f"question file is missing column(s) {sorted(missing)}")

    seen, questions = set(), []
    for row in rows:
        qid = (row["qid"] or "").strip()
        if not qid:
            raise ValueError("question file has a row with an empty qid")
        if qid in seen:
            raise ValueError(f"duplicate qid {qid}")
        seen.add(qid)
        questions.append({
            "qid": qid,
            "question": (row["question"] or "").strip(),
            "as_of": require_date(row["as_of"], f"{qid} as_of"),
        })
    return questions


def in_force(docs, as_of):
    """Documents whose window contains `as_of`."""
    return [d for d in _values(docs) if d.in_force(as_of)]


def lapsed(docs, as_of):
    """Expired documents with no successor -- their expiry is itself the answer.

    Superseded documents are excluded: their replacement is in force and will be retrieved
    normally, so reporting the old one as "lapsed" would be wrong.
    """
    return [d for d in _values(docs) if d.has_lapsed(as_of)]


def ancestors(docs, doc):
    """The chain of versions `doc` replaced, oldest last."""
    chain, seen = [], {doc.doc_id}
    current = docs.get(doc.supersedes) if doc.supersedes else None
    while current is not None and current.doc_id not in seen:
        chain.append(current)
        seen.add(current.doc_id)
        current = docs.get(current.supersedes) if current.supersedes else None
    return chain


def _values(docs):
    return list(docs.values()) if isinstance(docs, dict) else list(docs)


def default_kb_path():
    return Path(__file__).resolve().parents[2] / "starter" / "kb"


def default_questions_path():
    return Path(__file__).resolve().parents[2] / "starter" / "questions.csv"
