"""Tests for the date layer: which documents speak for a given date.

Covered closely because a bug here is silent: a wrong window does not raise, it returns a
confident answer from the wrong version of a policy.

Cases that probe one behaviour at several inputs assert in a loop rather than parametrising, so
the suite reads as one test per behaviour. Each carries the failing input in its message.
"""
from datetime import date, timedelta

import pytest

from src.qa.kb import in_force, lapsed, load_doc, load_kb

DOC = """---
doc_id: kb-999
title: Test Policy
category: testing
version: 1
effective_date: 2026-01-10
valid_until: 2026-01-20
status: current
supersedes:
superseded_by:
---

# Test Policy

The fee is 1.5%. It applies to all transfers.
"""


def write(tmp_path, text, name="kb-999.md"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestWindowBoundaries:
    @pytest.fixture
    def doc(self, tmp_path):
        return load_doc(write(tmp_path, DOC))

    def test_the_window_is_inclusive_at_both_ends_and_lapses_after_it(self, doc, tmp_path):
        """Probe the days either side of each end, then the same day once a successor exists.

        Both ends being inclusive and lapsing starting the day after are the same boundary seen
        twice, and lapsing is what turns an expired notice into a negative answer rather than
        silence -- so an off-by-one here changes an answer without raising.
        """
        expected = {
            date(2026, 1, 9): False,   # day before
            date(2026, 1, 10): True,   # effective_date itself
            date(2026, 1, 15): True,
            date(2026, 1, 20): True,   # valid_until itself
            date(2026, 1, 21): False,  # day after
        }
        for day, want in expected.items():
            assert doc.in_force(day) is want, f"in_force({day}) should be {want}"

        assert not doc.has_lapsed(date(2026, 1, 20))
        assert doc.has_lapsed(date(2026, 1, 21))

        replaced = load_doc(write(tmp_path, DOC.replace("superseded_by:", "superseded_by: kb-998")))
        # With a successor set the replacement is retrieved instead, so this is not lapsed.
        assert not replaced.has_lapsed(date(2026, 1, 21))

    def test_an_open_ended_document_never_expires(self, tmp_path):
        """A blank valid_until means still in force, not expired on the epoch."""
        doc = load_doc(write(tmp_path, DOC.replace("valid_until: 2026-01-20", "valid_until:")))
        assert doc.valid_until is None
        assert doc.in_force(date(2099, 1, 1))
        assert not doc.in_force(date(2026, 1, 9))


class TestMalformedInput:
    def test_bad_metadata_raises_at_load_time(self, tmp_path):
        """Bad metadata must raise rather than default to something plausible.

        The last two cases load a directory rather than a file, since an ambiguous citation and an
        empty knowledge base are only visible once the documents are collected together.
        """
        cases = [
            ("front matter", "# Just a heading\n\nSome text.\n"),
            ("effective_date", DOC.replace("effective_date: 2026-01-10", "effective_date:")),
            ("ISO date", DOC.replace("2026-01-10", "10/01/2026")),
            ("valid_until before effective_date",
             DOC.replace("valid_until: 2026-01-20", "valid_until: 2026-01-01")),
        ]
        for expected, text in cases:
            with pytest.raises(ValueError, match=expected):
                load_doc(write(tmp_path, text))

        write(tmp_path, DOC, "a.md")
        write(tmp_path, DOC, "b.md")
        with pytest.raises(ValueError, match="duplicate doc_id"):
            load_kb(tmp_path)
        with pytest.raises(ValueError, match="no .md documents"):
            load_kb(tmp_path / "empty")


class TestRealKnowledgeBase:
    """Properties the shipped KB has to hold for date resolution to be well defined."""

    def test_one_version_of_a_lineage_is_in_force_at_a_time(self, kb):
        """Group docs into chains by their root, then walk every day each chain spans.

        Two versions of one policy live on the same day would leave the ranking to choose
        between them by wording, and a gap would leave the question unanswerable. The closing
        assertion is why retrieval ignores `status`: it describes today rather than the query
        date, so filtering on it could never answer a historical question.
        """
        chains = {}
        for doc in kb.values():
            if not (doc.supersedes or doc.superseded_by):
                continue
            root = doc
            while root.supersedes and root.supersedes in kb:
                root = kb[root.supersedes]
            chains.setdefault(root.doc_id, []).append(doc)

        assert chains, "expected the KB to contain supersession chains"
        for root_id, members in chains.items():
            start = min(d.effective_date for d in members)
            last = max(d.valid_until or d.effective_date for d in members)
            for offset in range((last - start).days + 1):
                day = start + timedelta(days=offset)
                live = [d for d in members if d.in_force(day)]
                assert len(live) == 1, f"{root_id} has {len(live)} versions in force on {day}"

        superseded = [d for d in kb.values() if d.status == "superseded"]
        assert superseded, "expected the KB to contain superseded versions"
        assert all(d.in_force(d.effective_date) for d in superseded)

    def test_the_two_candidate_pools_stay_disjoint(self, kb):
        """In-force and lapsed are ranked as one pool, and a superseded doc is in neither.

        An overlap would rank one document twice; a superseded doc offered as lapsed would
        announce an expiry for a policy that was actually replaced.
        """
        for day in (date(2025, 6, 1), date(2026, 3, 31), date(2026, 7, 28)):
            live = {d.doc_id for d in in_force(kb, day)}
            expired = {d.doc_id for d in lapsed(kb, day)}
            assert not live & expired, f"pools overlap on {day}"
            assert all(not kb[i].superseded_by for i in expired), f"lapsed offers a replaced doc"
