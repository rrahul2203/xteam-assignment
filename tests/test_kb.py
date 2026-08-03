"""Tests for the date layer: which documents speak for a given date, and input validation.

These are the tests worth having, because a bug here is silent. A wrong window does not
raise -- it returns a confident answer from the wrong version of a policy.
"""
from datetime import date, timedelta

import pytest

from src.qa.kb import (
    in_force,
    lapsed,
    load_doc,
    load_kb,
    load_questions,
    parse_date,
    require_date,
)

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
    """Both ends of the window are inclusive. Off-by-one here is a wrong-version answer."""

    @pytest.fixture
    def doc(self, tmp_path):
        return load_doc(write(tmp_path, DOC))

    @pytest.mark.parametrize("day, expected", [
        (date(2026, 1, 9), False),   # day before
        (date(2026, 1, 10), True),   # effective_date itself
        (date(2026, 1, 15), True),
        (date(2026, 1, 20), True),   # valid_until itself
        (date(2026, 1, 21), False),  # day after
    ])
    def test_in_force_is_inclusive_at_both_ends(self, doc, day, expected):
        assert doc.in_force(day) is expected

    def test_open_ended_document_never_expires(self, tmp_path):
        doc = load_doc(write(tmp_path, DOC.replace("valid_until: 2026-01-20", "valid_until:")))
        assert doc.valid_until is None
        assert doc.in_force(date(2099, 1, 1))
        assert not doc.in_force(date(2026, 1, 9))

    def test_lapses_only_after_the_window_and_only_without_a_successor(self, doc, tmp_path):
        assert not doc.has_lapsed(date(2026, 1, 20))
        assert doc.has_lapsed(date(2026, 1, 21))

        replaced = load_doc(write(tmp_path, DOC.replace("superseded_by:", "superseded_by: kb-998")))
        # The successor is in force and will be retrieved instead, so this is not the answer.
        assert not replaced.has_lapsed(date(2026, 1, 21))


class TestMalformedInput:
    """Bad metadata raises at load time rather than defaulting to something plausible."""

    def test_missing_front_matter_raises(self, tmp_path):
        with pytest.raises(ValueError, match="front matter"):
            load_doc(write(tmp_path, "# Just a heading\n\nSome text.\n"))

    def test_missing_required_field_raises(self, tmp_path):
        with pytest.raises(ValueError, match="effective_date"):
            load_doc(write(tmp_path, DOC.replace("effective_date: 2026-01-10", "effective_date:")))

    def test_unparseable_date_raises(self, tmp_path):
        with pytest.raises(ValueError, match="ISO date"):
            load_doc(write(tmp_path, DOC.replace("2026-01-10", "10/01/2026")))

    def test_window_ending_before_it_starts_raises(self, tmp_path):
        with pytest.raises(ValueError, match="valid_until before effective_date"):
            load_doc(write(tmp_path, DOC.replace("valid_until: 2026-01-20",
                                                 "valid_until: 2026-01-01")))

    def test_duplicate_doc_id_raises(self, tmp_path):
        write(tmp_path, DOC, "a.md")
        write(tmp_path, DOC, "b.md")
        with pytest.raises(ValueError, match="duplicate doc_id"):
            load_kb(tmp_path)

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no .md documents"):
            load_kb(tmp_path)

    @pytest.mark.parametrize("value", ["", None, "   "])
    def test_require_date_rejects_blank(self, value):
        with pytest.raises(ValueError, match="required"):
            require_date(value)

    def test_parse_date_passes_blank_through_as_none(self):
        assert parse_date("") is None
        assert parse_date(None) is None


class TestRealKnowledgeBase:
    """Properties the shipped KB has to satisfy for date resolution to mean anything."""

    def test_status_is_not_a_proxy_for_in_force(self, kb):
        """Assert docs marked superseded were live on their own effective_date.

        This is why retrieval ignores `status`: it describes today, not the query date, so
        filtering on it could never answer a historical question correctly.
        """
        superseded = [d for d in kb.values() if d.status == "superseded"]
        assert superseded, "expected the KB to contain superseded versions"
        assert all(d.in_force(d.effective_date) for d in superseded)

    def test_one_version_of_a_lineage_is_in_force_at_a_time(self, kb):
        """Group docs into chains by their root, then walk every day each chain spans.

        Two versions of one policy live on the same day would leave the ranking to choose
        between them by wording, and a gap would leave the question unanswerable.
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

    def test_a_superseded_document_is_never_offered_as_lapsed(self, kb):
        for day in (date(2025, 6, 1), date(2026, 3, 1), date(2026, 7, 28)):
            assert all(not d.superseded_by for d in lapsed(kb, day))

    def test_candidate_pools_do_not_overlap(self, kb):
        """In-force and lapsed are disjoint by construction, since one pool is ranked."""
        for day in (date(2025, 1, 1), date(2026, 3, 31), date(2026, 7, 28)):
            assert not {d.doc_id for d in in_force(kb, day)} & {d.doc_id for d in lapsed(kb, day)}


class TestQuestionFile:
    def test_missing_column_raises(self, tmp_path):
        path = tmp_path / "q.csv"
        path.write_text("qid,question\nq1,What is the fee?\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing column"):
            load_questions(path)

    def test_duplicate_qid_raises(self, tmp_path):
        path = tmp_path / "q.csv"
        path.write_text("qid,question,as_of\nq1,A?,2026-01-01\nq1,B?,2026-01-01\n",
                        encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate qid"):
            load_questions(path)

    def test_as_of_is_parsed_to_a_date(self):
        questions = load_questions()
        assert questions and all(isinstance(q["as_of"], date) for q in questions)
