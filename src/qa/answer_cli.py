"""Question answering CLI.

    python3 -m src.qa.answer_cli --input questions.csv --output answers.csv
    python3 -m src.qa.answer_cli --question "What is the withdrawal fee?" --as-of 2025-06-01

Writes `qid`, `answer` and `doc_ids` (semicolon separated), per the brief. Declined questions get
an empty `doc_ids` and a plain-words answer, so abstention is distinguishable from a crash.
`--verbose` adds the status and score columns that the evaluation thresholds on.
"""
import argparse
import csv
import logging

from .answerer import AnswerService
from .kb import default_questions_path, load_kb, load_questions, require_date
from .retriever import MODE_HYBRID, MODES, Retriever

log = logging.getLogger(__name__)

FIELDS = ("qid", "answer", "doc_ids")
VERBOSE_FIELDS = FIELDS + ("status", "score", "coverage")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Answer questions from the KB, as at a date.")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", nargs="?", const=str(default_questions_path()),
                        help="CSV with qid, question and as_of columns")
    source.add_argument("--question", help="answer a single question and print it")
    p.add_argument("--as-of", help="date for --question, ISO format")
    p.add_argument("--output", default="answers.csv", help="where to write answers")
    p.add_argument("--kb", default=None, help="KB directory, defaults to starter/kb")
    p.add_argument("--mode", default=MODE_HYBRID, choices=MODES,
                   help="ranking signals to use, defaults to hybrid")
    p.add_argument("--verbose", action="store_true", help="add status and score columns")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.question and not args.as_of:
        raise SystemExit("--question requires --as-of")

    docs = load_kb(args.kb) if args.kb else load_kb()
    service = AnswerService(docs)
    service.retriever = Retriever(docs, mode=args.mode)
    log.info("ranking mode: %s", service.retriever.describe())

    if args.question:
        result = service.answer(args.question, require_date(args.as_of, "--as-of"))
        # Answer to stdout so it can be piped; diagnostics go to the log instead.
        print(result.text)
        print(f"\ndoc_ids: {result.doc_ids_field() or '(none)'}  [{result.status}]")
        return

    questions = load_questions(args.input)
    fields = VERBOSE_FIELDS if args.verbose else FIELDS

    declined = 0
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for question in questions:
            result = service.answer(question["question"], question["as_of"])
            declined += not result.answered
            row = {
                "qid": question["qid"],
                "answer": result.text,
                "doc_ids": result.doc_ids_field(),
            }
            if args.verbose:
                row.update(status=result.status, score=round(result.score, 4),
                           coverage=round(result.coverage, 4))
            writer.writerow(row)

    log.info("wrote %d answers to %s (%d declined)", len(questions), args.output, declined)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
