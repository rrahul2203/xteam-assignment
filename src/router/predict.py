"""Route classification CLI.

    python3 -m src.router.predict --input messages.csv --output predictions.csv
    python3 -m src.router.predict --text "I can't log in"

Writes `text` and `prediction`, per the brief. `--confidence` adds the top-class
probability, which is what a review queue would threshold on.

Loads the model from models/router.joblib and does not train. Fitting is proportional to the
training set, so training per prediction call does not scale to a large corpus or high request
volume. Build the artifact once with `python3 -m src.router.artifact`; `--retrain` overrides this
for development against edited data.
"""
import argparse
import csv
import logging

from .artifact import load_model
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
    p.add_argument("--model", default=None, help="saved model to load, defaults to models/")
    p.add_argument("--retrain", action="store_true",
                   help="train now instead of loading the saved model")
    p.add_argument("--confidence", action="store_true", help="add a confidence column")
    return p.parse_args(argv)


def get_pipeline(model_path=None, train_path=None, retrain=False):
    """The pipeline to predict with, loaded from the saved artifact.

    Loading is the only default path. Training costs time proportional to the training set, so
    doing it per call does not survive a large corpus or a high prediction volume -- prediction
    reads a model, it does not build one. A missing artifact therefore raises with the command to
    build it rather than quietly training, which would hide that cost instead of removing it.

    `--retrain` and `--train` opt into training explicitly, for development against edited data.
    """
    if retrain or train_path:
        log.info("training now, as asked (this is not the prediction path)")
        texts, labels = load(train_path or default_data_path())
        return train(texts, labels)

    pipeline, metadata = load_model(model_path)
    log.info("loaded model trained on %d rows of %s (digest %s)",
             metadata["n_training_rows"], metadata["train_file"], metadata["data_digest"])
    return pipeline


def main(argv=None):
    args = parse_args(argv)

    try:
        pipeline = get_pipeline(args.model, args.train, args.retrain)
    except FileNotFoundError as exc:
        # A missing artifact is a setup mistake, so report it as one rather than as a traceback.
        raise SystemExit(str(exc))

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
