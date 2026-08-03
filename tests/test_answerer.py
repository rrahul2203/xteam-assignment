"""Tests for answering: date resolution end to end, the three outcomes, and abstention.

`test_same_question_two_dates_two_answers` is the central one: it asks a single question either
side of a policy change and requires different documents back. That covers the failure mode the
brief names, and is the one a test can pin down directly.
"""
from datetime import date

import pytest

from src.qa.answerer import ANSWERED, LAPSED, UNANSWERED, absent_domain_terms, answer
from src.qa.kb import load_questions

DISPUTE_QUESTION = "How many days does a customer have to raise a dispute?"


class TestDateResolution:
    """Answers must resolve to the document in force on the date asked, not on today."""

    def test_same_question_two_dates_two_answers(self, service):
        """Ask one question at two dates spanning a version change and compare the citations.

        Also probes the two adjacent days where one version's window ends and the next begins,
        since an off-by-one in the boundary is the likeliest way this breaks.
        """
        during_v1 = service.answer(DISPUTE_QUESTION, date(2026, 3, 1))
        during_v2 = service.answer(DISPUTE_QUESTION, date(2026, 7, 28))

        assert during_v1.doc_ids == ["kb-031"]
        assert during_v2.doc_ids == ["kb-032"]
        assert during_v1.text != during_v2.text

        assert service.answer(DISPUTE_QUESTION, date(2026, 6, 30)).doc_ids == ["kb-031"]
        assert service.answer(DISPUTE_QUESTION, date(2026, 7, 1)).doc_ids == ["kb-032"]

    def test_every_cited_document_was_live_on_the_date_asked(self, service):
        """Run the shipped question set and check every citation was in force or lapsed."""
        for question in load_questions():
            result = service.answer(question["question"], question["as_of"])
            for doc_id in result.doc_ids:
                doc = service.docs[doc_id]
                assert doc.in_force(result.as_of) or doc.has_lapsed(result.as_of), (
                    f"{question['qid']} cited {doc_id}, which was not live on {result.as_of}")
                assert doc.effective_date <= result.as_of, (
                    f"{question['qid']} cited {doc_id}, which did not exist yet")


class TestLapsedNotices:
    """An expired notice with no successor answers in the negative instead of abstaining."""

    def test_an_expired_promotion_answers_in_the_negative_and_only_on_its_own_topic(self, service):
        """Ask the promotion after its window, inside it, then ask something unrelated.

        The three cases are one claim: the negative answer has to come from the date rather than
        the wording, and a lapsed notice sits in the candidate pool for every query, so it also
        has to stay silent on questions it has nothing to do with.
        """
        promotion = "Is the invite a friend promotion still running?"

        after = service.answer(promotion, date(2026, 7, 28))
        assert after.status == LAPSED
        assert after.doc_ids == ["kb-092"]
        assert after.answered
        assert "no longer" in after.text.lower()
        # Read the closing date off the document, so the assertion survives a KB edit.
        assert service.docs["kb-092"].valid_until.isoformat() in after.text

        inside = service.answer(promotion, date(2026, 1, 15))
        assert inside.status == ANSWERED
        assert inside.doc_ids == ["kb-092"]

        unrelated = service.answer(
            "I lost my phone and cannot complete two-factor authentication. "
            "How do I get back into my account?", date(2026, 7, 28))
        assert unrelated.status != LAPSED


class TestAbstention:
    """Declining is a designed outcome, so it has an output contract of its own."""

    def test_topics_absent_from_the_kb_are_declined(self, service):
        questions = [
            "What interest rate do you charge on a crypto backed margin loan?",
            "Who is your qualifying custodian for client assets?",
            "How do I pay my electricity bill through the app?",
        ]
        for question in questions:
            result = service.answer(question, date(2026, 7, 28))
            assert result.status == UNANSWERED, f"answered {question!r}"
            assert result.doc_ids == [] and result.doc_ids_field() == ""
            assert "human agent" in result.text

    def test_absent_domain_terms_ignores_short_filler(self, service):
        """Short absent words must not count as domain terms, long ones must.

        This is half the abstention conjunction, and the half that would silently stop working:
        if every short word counted, the service would decline almost everything.
        """
        assert absent_domain_terms("what is the fee today in march", service.vocabulary) == []
        assert "chargeback" in absent_domain_terms(
            "how do I file a chargeback", service.vocabulary)


class TestOutputContract:
    def test_answer_text_is_copied_from_the_cited_document(self, service):
        """Nothing is generated: each returned sentence appears verbatim in the cited body.

        The closing lines cover the two entry points and the CSV field: the CLI passes the date as
        a string and the library passes a date, and both have to reach the same answer.
        """
        result = service.answer(DISPUTE_QUESTION, date(2026, 3, 1))
        body = service.docs[result.doc_ids[0]].body
        for sentence in result.text.split(". "):
            assert sentence.strip(". ") in " ".join(body.split())

        result.doc_ids = ["kb-031", "kb-032"]
        assert result.doc_ids_field() == "kb-031;kb-032"

        assert (service.answer(DISPUTE_QUESTION, "2026-03-01").doc_ids
                == service.answer(DISPUTE_QUESTION, date(2026, 3, 1)).doc_ids
                == answer(DISPUTE_QUESTION, "2026-03-01", service=service).doc_ids)


class TestInputValidation:
    def test_unusable_input_is_rejected_before_retrieval(self, service):
        """A question with no content words cannot be ranked, so it raises rather than abstaining.

        The date cases are here too, since a missing or non-ISO date decides the answer and
        guessing one would be the wrong kind of helpful.
        """
        for bad, exception in ((None, TypeError), (42, TypeError), ("  ", ValueError),
                               ("???", ValueError)):
            with pytest.raises(exception):
                service.answer(bad, date(2026, 7, 28))

        with pytest.raises(ValueError, match="required"):
            service.answer(DISPUTE_QUESTION, "")
        with pytest.raises(ValueError, match="ISO date"):
            service.answer(DISPUTE_QUESTION, "28-07-2026")
