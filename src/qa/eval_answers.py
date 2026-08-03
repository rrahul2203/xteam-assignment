"""Measuring the retriever against a hand-labelled gold set.

    python3 -m src.qa.eval_answers

Reported per stratum, not just in aggregate: one number would hide the failure modes, and the
strata fail for different reasons. Historical and lapsed questions are the ones the date logic
exists for and there are few of them, so every rate carries a Wilson interval.

The ablations matter as much as the headline. If switching the date filter off does not move
the score, the score is not measuring the thing this exercise is about.

Honest limitation, stated rather than buried: the same person wrote the system and the labels,
and the gold set is small. This measures whether the date logic works. It cannot support a
claim about generalisation to unseen questions.
"""
import csv
import logging
import math
import re
from pathlib import Path

from .answerer import AnswerService
from .kb import load_kb, load_questions

log = logging.getLogger(__name__)

STRATA = ("current", "historical", "lapsed", "unanswerable")


def default_gold_path():
    return Path(__file__).resolve().parents[2] / "eval" / "gold.csv"


def load_gold(path=None):
    rows = list(csv.DictReader(open(path or default_gold_path(), encoding="utf-8")))
    if not rows:
        raise ValueError("gold set is empty")
    gold = {}
    for row in rows:
        if row["stratum"] not in STRATA:
            raise ValueError(f"{row['qid']} has unknown stratum {row['stratum']!r}")
        pattern = (row["must_match"] or "").strip()
        if pattern:
            # Fail here rather than midway through scoring.
            re.compile(pattern)
        gold[row["qid"]] = {
            "stratum": row["stratum"],
            "doc_ids": [d for d in row["doc_ids"].split(";") if d],
            "must_match": pattern,
        }
    return gold


def wilson(successes, total, z=1.96):
    """95% Wilson interval, chosen because the normal approximation breaks down at small n."""
    if not total:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def score_one(question, gold, answer, service):
    """Per-question outcome flags."""
    expected = set(gold["doc_ids"])
    returned = set(answer.doc_ids)
    should_answer = bool(expected)

    top_correct = bool(returned) and bool(expected & returned)
    pattern = gold["must_match"]
    fact_ok = bool(pattern) and bool(re.search(pattern, answer.text, re.IGNORECASE))

    return {
        "qid": question["qid"],
        "stratum": gold["stratum"],
        "should_answer": should_answer,
        "did_answer": answer.answered,
        "doc_correct": top_correct,
        "fact_checked": bool(pattern),
        "fact_ok": fact_ok,
        # The metric that decides whether this is shippable: answered confidently, wrong doc.
        "wrong_answer": answer.answered and should_answer and not top_correct,
        # Answered something that has no answer in the KB.
        "false_answer": answer.answered and not should_answer,
        "status": answer.status,
        "score": answer.score,
        "missing_mass": answer.missing_mass,
    }


def evaluate(service=None, questions=None, gold=None):
    service = service or AnswerService()
    questions = questions or load_questions()
    gold = gold or load_gold()

    rows = []
    for question in questions:
        if question["qid"] not in gold:
            log.warning("no gold label for %s, skipping", question["qid"])
            continue
        answer = service.answer(question["question"], question["as_of"])
        rows.append(score_one(question, gold[question["qid"]], answer, service))
    return rows


def summarise(rows):
    """Aggregate and per-stratum rates."""
    answerable = [r for r in rows if r["should_answer"]]
    unanswerable = [r for r in rows if not r["should_answer"]]
    checked = [r for r in answerable if r["fact_checked"]]

    summary = {
        "n": len(rows),
        "doc_accuracy": _rate(answerable, "doc_correct"),
        "wrong_answer_rate": _rate(rows, "wrong_answer"),
        "abstention_recall": (
            sum(1 for r in unanswerable if not r["did_answer"]) / len(unanswerable)
            if unanswerable else 0.0
        ),
        "answer_recall": _rate(answerable, "did_answer"),
        "fact_accuracy": _rate(checked, "fact_ok"),
        "by_stratum": {},
    }
    for stratum in STRATA:
        group = [r for r in rows if r["stratum"] == stratum]
        if not group:
            continue
        if stratum == "unanswerable":
            hits = sum(1 for r in group if not r["did_answer"])
        else:
            hits = sum(1 for r in group if r["doc_correct"])
        summary["by_stratum"][stratum] = {
            "n": len(group),
            "correct": hits,
            "rate": hits / len(group),
            "ci": wilson(hits, len(group)),
        }
    return summary


def _rate(rows, key):
    return sum(1 for r in rows if r[key]) / len(rows) if rows else 0.0


def ablations(questions, gold):
    """Configurations that should each cost something measurable."""
    docs = load_kb()
    results = {}

    baseline = AnswerService(docs)
    results["shipped"] = summarise(evaluate(baseline, questions, gold))

    # Date filter off: rank the whole KB regardless of as_of. This is the ablation that
    # matters -- it should wreck the historical stratum specifically.
    undated = AnswerService(docs)
    undated.retriever.candidates = lambda as_of: list(docs.values())
    results["no_date_filter"] = summarise(evaluate(undated, questions, gold))

    # Abstention off: always answer the top hit, however weak.
    eager = AnswerService(docs, min_score=0.0, min_coverage=0.0)
    results["no_abstention"] = summarise(evaluate(eager, questions, gold))

    # Coverage alone, without requiring an absent domain term: shows what the conjunction
    # buys over the single signal.
    coverage_only = AnswerService(docs, require_absent_term=False)
    results["coverage_only"] = summarise(evaluate(coverage_only, questions, gold))

    return results


def threshold_sweep(questions, gold, thresholds=(0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.70)):
    """What each coverage threshold buys and costs, so the choice is a curve not a claim."""
    docs = load_kb()
    sweep = []
    for threshold in thresholds:
        summary = summarise(
            evaluate(AnswerService(docs, min_coverage=threshold), questions, gold))
        sweep.append((threshold, summary))
    return sweep


def report():
    questions, gold = load_questions(), load_gold()
    rows = evaluate(AnswerService(), questions, gold)
    summary = summarise(rows)

    log.info("gold set: %d questions", summary["n"])
    log.info("document accuracy      %.3f   (answerable questions, right doc cited)",
             summary["doc_accuracy"])
    log.info("wrong-answer rate      %.3f   <- the one that decides shippability",
             summary["wrong_answer_rate"])
    log.info("answer recall          %.3f   (answerable questions actually answered)",
             summary["answer_recall"])
    log.info("abstention recall      %.3f   (unanswerable questions declined)",
             summary["abstention_recall"])
    log.info("fact accuracy          %.3f   (quoted text contains the expected value)",
             summary["fact_accuracy"])

    log.info("\nby stratum")
    for stratum, stats in summary["by_stratum"].items():
        low, high = stats["ci"]
        log.info("  %-13s %d/%d = %.3f   95%% CI [%.2f, %.2f]",
                 stratum, stats["correct"], stats["n"], stats["rate"], low, high)

    log.info("\nablations (document accuracy / wrong-answer rate)")
    for name, summ in ablations(questions, gold).items():
        historical = summ["by_stratum"].get("historical", {})
        log.info("  %-16s %.3f / %.3f   historical %.3f",
                 name, summ["doc_accuracy"], summ["wrong_answer_rate"],
                 historical.get("rate", 0.0))

    log.info("\ncoverage threshold sweep")
    log.info("  %-12s %-10s %-10s %s", "min_coverage", "answered", "wrong", "abstain_recall")
    for threshold, summ in threshold_sweep(questions, gold):
        log.info("  %-12.2f %-10.3f %-10.3f %.3f", threshold, summ["answer_recall"],
                 summ["wrong_answer_rate"], summ["abstention_recall"])

    failures = [r for r in rows if r["wrong_answer"] or r["false_answer"]
                or (r["should_answer"] and not r["did_answer"])]
    if failures:
        log.info("\nfailures")
        for row in failures:
            log.info("  %s (%s) status=%s score=%.3f gap=%.2f",
                     row["qid"], row["stratum"], row["status"], row["score"],
                     row["missing_mass"])
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report()
