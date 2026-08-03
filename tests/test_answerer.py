"""Tests for answering: date resolution end to end, abstention, and the output contract.

`test_same_question_two_dates_two_answers` is the central one: it asks a single question either
side of a policy change and requires different documents back. That covers the failure mode the
brief names, and is the one a test can pin down directly.
"""
from datetime import date

import pytest

from src.qa.answerer import (
    ANSWERED,
    LAPSED,
    UNANSWERED,
    absent_domain_terms,
    answer,
    content_words,
    extract,
    salient_coverage,
)
from src.qa.kb import load_questions
from src.qa.retriever import validate_question

DISPUTE_QUESTION = "How many days does a customer have to raise a dispute?"


class TestDateResolution:
    """Answers must resolve to the document in force on the date asked, not on today."""

    def test_same_question_two_dates_two_answers(self, service):
        """Ask one question at two dates spanning a version change and compare the citations."""
        during_v1 = service.answer(DISPUTE_QUESTION, date(2026, 3, 1))
        during_v2 = service.answer(DISPUTE_QUESTION, date(2026, 7, 28))

        assert during_v1.doc_ids == ["kb-031"]
        assert during_v2.doc_ids == ["kb-032"]
        assert during_v1.text != during_v2.text

    def test_the_day_a_policy_changes_switches_the_answer(self, service):
        """Probe the two adjacent days where one version's window ends and the next begins."""
        assert service.answer(DISPUTE_QUESTION, date(2026, 6, 30)).doc_ids == ["kb-031"]
        assert service.answer(DISPUTE_QUESTION, date(2026, 7, 1)).doc_ids == ["kb-032"]

    def test_a_future_document_cannot_answer_an_earlier_question(self, service):
        """Check no cited document has an effective_date later than the query date."""
        when = date(2026, 3, 1)
        for question in (DISPUTE_QUESTION, "What is the fee to withdraw funds?"):
            result = service.answer(question, when)
            for doc_id in result.doc_ids:
                assert service.docs[doc_id].effective_date <= when

    def test_every_cited_document_was_in_force_or_a_lapsed_notice(self, service):
        """Run the shipped question set and check every citation was live on its own date."""
        for question in load_questions():
            result = service.answer(question["question"], question["as_of"])
            for doc_id in result.doc_ids:
                doc = service.docs[doc_id]
                assert doc.in_force(result.as_of) or doc.has_lapsed(result.as_of), (
                    f"{question['qid']} cited {doc_id}, which was not live on {result.as_of}")


class TestLapsedNotices:
    """An expired notice with no successor answers in the negative instead of abstaining."""

    def test_expired_promotion_answers_in_the_negative(self, service):
        result = service.answer("Is the invite a friend promotion still running?",
                                date(2026, 7, 28))
        assert result.status == LAPSED
        assert result.doc_ids == ["kb-092"]
        assert result.answered
        assert "no longer" in result.text.lower()
        # Read the closing date off the document, so the assertion survives a KB edit.
        assert service.docs["kb-092"].valid_until.isoformat() in result.text

    def test_the_same_promotion_answers_positively_inside_its_window(self, service):
        result = service.answer("Is the invite a friend promotion still running?",
                                date(2026, 1, 15))
        assert result.status == ANSWERED
        assert result.doc_ids == ["kb-092"]

    def test_an_off_topic_lapsed_notice_does_not_speak(self, service):
        """Ask an unrelated question and check no expiry is announced for it."""
        result = service.answer(
            "I lost my phone and cannot complete two-factor authentication. "
            "How do I get back into my account?", date(2026, 7, 28))
        assert result.status != LAPSED


class TestAbstention:
    """Declining is a designed outcome, so it has an output contract of its own."""

    @pytest.mark.parametrize("question", [
        "What interest rate do you charge on a crypto backed margin loan?",
        "Who is your qualifying custodian for client assets?",
        "How do I pay my electricity bill through the app?",
    ])
    def test_topics_absent_from_the_kb_are_declined(self, service, question):
        result = service.answer(question, date(2026, 7, 28))
        assert result.status == UNANSWERED
        assert result.doc_ids == []
        assert "human agent" in result.text

    def test_a_declined_answer_cites_nothing(self, service):
        result = service.answer("What is your policy on quantum computing?", date(2026, 7, 28))
        assert not result.answered and result.doc_ids_field() == ""

    def test_absent_domain_terms_ignores_short_filler(self, service):
        """Short absent words must not count as domain terms, long ones must."""
        assert absent_domain_terms("what is the fee today in march", service.vocabulary) == []
        assert "chargeback" in absent_domain_terms(
            "how do I file a chargeback", service.vocabulary)


class TestOutputContract:
    def test_doc_ids_field_is_semicolon_separated(self, service):
        result = service.answer(DISPUTE_QUESTION, date(2026, 3, 1))
        assert ";" not in result.doc_ids_field()  # one doc, so no separator
        result.doc_ids = ["kb-031", "kb-032"]
        assert result.doc_ids_field() == "kb-031;kb-032"

    def test_answer_text_is_copied_from_the_cited_document(self, service):
        """Check each returned sentence appears verbatim in the cited body."""
        result = service.answer(DISPUTE_QUESTION, date(2026, 3, 1))
        body = service.docs[result.doc_ids[0]].body
        for sentence in result.text.split(". "):
            assert sentence.strip(". ") in " ".join(body.split())

    def test_as_of_accepts_an_iso_string_or_a_date(self, service):
        assert (service.answer(DISPUTE_QUESTION, "2026-03-01").doc_ids
                == service.answer(DISPUTE_QUESTION, date(2026, 3, 1)).doc_ids)

    def test_module_level_answer_matches_the_service(self, service):
        assert answer(DISPUTE_QUESTION, "2026-03-01", service=service).doc_ids == ["kb-031"]


class TestInputValidation:
    @pytest.mark.parametrize("bad, exception", [
        (None, TypeError),
        (42, TypeError),
        (["a question?"], TypeError),
        ("", ValueError),
        ("   ", ValueError),
        ("???", ValueError),
    ])
    def test_bad_questions_are_rejected(self, bad, exception):
        with pytest.raises(exception):
            validate_question(bad)

    def test_validation_happens_before_retrieval(self, service):
        with pytest.raises(ValueError):
            service.answer("  ", date(2026, 7, 28))

    def test_missing_as_of_is_rejected(self, service):
        with pytest.raises(ValueError, match="required"):
            service.answer(DISPUTE_QUESTION, "")

    def test_malformed_as_of_is_rejected(self, service):
        with pytest.raises(ValueError, match="ISO date"):
            service.answer(DISPUTE_QUESTION, "28-07-2026")


class TestSignals:
    def test_content_words_drop_stop_words_and_bare_numbers(self):
        words = content_words("What is the fee for 2 transfers in 2026?")
        assert "fee" in words and "transfers" in words
        assert "the" not in words and "2026" not in words

    def test_coverage_rewards_the_document_that_has_the_rare_terms(self, service):
        question = "How many days to raise a dispute?"
        on_topic = salient_coverage(question, service.docs["kb-031"],
                                    service.frequencies, len(service.docs))
        off_topic = salient_coverage(question, service.docs["kb-091"],
                                     service.frequencies, len(service.docs))
        assert on_topic > off_topic

    def test_extract_prefers_a_sentence_carrying_the_number(self, service):
        text = extract("How many days do I have to raise a dispute?", service.docs["kb-031"])
        assert "60" in text
