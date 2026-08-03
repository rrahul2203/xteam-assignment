"""Route classification CLI.

    python3 -m src.router.predict --input messages.csv --output predictions.csv
    python3 -m src.router.predict --text "I can't log in"

Writes `text` and `prediction`, per the brief. `--confidence` adds the top-class
probability, which is what a review queue would threshold on.
"""
import argparse
import csv
import logging

from .data import default_data_path, load, load_texts
from .model import train

log = logging.getLogger(__name__)


def classify(pipeline, texts):
    """Return (label, confidence) for each message in one vectorisation pass."""
    proba = pipeline.predict_proba(texts)
    classes = list(pipeline.classes_)
    return [(classes[int(row.argmax())], float(row.max())) for row in proba]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Route support messages.")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="CSV with a text column")
    source.add_argument("--text", help="classify a single message and print the route")
    p.add_argument("--output", default="predictions.csv", help="where to write predictions")
    p.add_argument("--train", default=None, help="labelled CSV to train on")
    p.add_argument("--confidence", action="store_true", help="add a confidence column")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    texts, labels = load(args.train or default_data_path())
    pipeline = train(texts, labels)

    if args.text:
        label, confidence = classify(pipeline, [args.text])[0]
        # The route is the payload, so it goes to stdout; status goes to the log.
        print(f"{label}\t{confidence:.4f}")
        return

    messages = load_texts(args.input)
    results = classify(pipeline, messages)

    fields = ["text", "prediction"] + (["confidence"] if args.confidence else [])
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for message, (label, confidence) in zip(messages, results):
            row = {"text": message, "prediction": label}
            if args.confidence:
                row["confidence"] = round(confidence, 4)
            w.writerow(row)

    log.info("wrote %d rows to %s", len(results), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
