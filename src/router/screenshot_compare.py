"""Comparing OCR against a vision language model for routing screenshots.

    python3 -m src.router.screenshot_compare --vlm /path/to/SmolVLM-500M-Instruct

Two ways to get from screenshot pixels to one of the four routes:

  ocr  the shipped path. Tesseract extracts text, then the Part A classifier routes it.
  vlm  a vision language model shown the image and asked for the route, with no OCR stage.

These are not two qualities of the same thing. The first reuses a classifier whose accuracy was
measured on 400 labelled tickets in Part A, so the only new failure mode is the text extraction.
The second replaces both stages with one model's judgement, which cannot be evaluated against that
baseline and has no intermediate output to inspect when it is wrong.

Two things are reported per tier: whether the route is right across a blur sweep, and how many
distinct routes the tier ever produced. The second column is there because a tier that answers the
same route everywhere still scores on whichever assets happen to match it. A transcription probe
follows, asking the model to read the screenshot rather than route it, since the case for a vision
model is that it reads what OCR cannot.

Both tiers call the same code the shipped router calls -- `screenshots.extract_text` and
`vlm_reader` -- so what is measured here is what `--reader ocr` and `--reader vlm` actually do.

The VLM tier is skipped unless --vlm points at a local model directory, so this runs on a clone
without a 1GB download and reports what it could not measure rather than failing.
"""
import argparse
import logging
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageFilter

from . import vlm_reader
from .artifact import load_model
from .predict import classify
from .screenshots import can_read, extract_text, find_images, redact

log = logging.getLogger(__name__)

MEDIA_DIR = "starter/media/screenshots"

# Blur radii for the degradation sweep, in pixels against a 780x1688 screenshot.
BLUR_RADII = (0, 2, 3, 4)

EXPECTED = {
    "login-error.png": "account-access",
    "phishing-sms.png": "fraud-report",
    "txn-failed.png": "transaction-dispute",
}

ROUTES = ("account-access", "fraud-report", "general", "transaction-dispute")

ROUTE_PROMPT = ("This is a screenshot from a customer support ticket. Choose exactly one route "
                f"from: {', '.join(ROUTES)}. Answer with the route name only.")

# The line that decides the fraud route on phishing-sms.png: the customer admitting they passed on
# a one-time code. It is white on saturated blue, which is what the second OCR pass exists for.
BUBBLE_MARKER = "448120"


def vlm_route(loaded, image):
    """Route an image with the model directly, reading the label out of its prose reply.

    This is the one-step alternative the shipped pipeline does not offer: no text is produced, so
    there is nothing to redact or show a reviewer. It is measured here to show what that buys.
    """
    reply = vlm_reader.ask(*loaded, image, ROUTE_PROMPT, max_new_tokens=16)
    lowered = reply.lower()
    for label in ROUTES:
        if label in lowered:
            return label
    return None


def ocr_route(pipeline, image):
    """Extract text, redact it, and classify it, which is the shipped pipeline end to end."""
    redacted, _ = redact(" ".join(ocr_text(image).split()))
    if not redacted:
        return None
    return classify(pipeline, [redacted])[0][0]


def ocr_text(image):
    """Run the shipped two-pass extraction over an in-memory image.

    `extract_text` takes a path because that is what the CLI has, while the sweep holds blurred
    variants in memory, so each is written out once to reuse the same code path.
    """
    with tempfile.NamedTemporaryFile(suffix=".png") as handle:
        image.save(handle.name)
        return extract_text(handle.name)[0]


def blurred(path, radius):
    image = Image.open(path)
    image.load()
    return image.filter(ImageFilter.GaussianBlur(radius)) if radius else image


def measure(paths, route_fn):
    """Route every asset at every blur radius, recording hits, answers and clean-image latency."""
    hits_by_radius = {}
    answers = []
    latencies = []
    for path in paths:
        for radius in BLUR_RADII:
            image = blurred(path, radius)
            start = time.perf_counter()
            route = route_fn(image)
            if radius == 0:
                latencies.append((time.perf_counter() - start) * 1000)
            hits_by_radius.setdefault(radius, []).append(route == EXPECTED[path.name])
            answers.append(route)
    return hits_by_radius, answers, latencies


def transcription_probe(paths, loaded):
    """Print what each reader returns when asked to read one screenshot rather than route it.

    This runs the vision model through the same `vlm_reader.transcribe` the shipped `--reader vlm`
    uses, so what is printed here is what that backend would hand the classifier.
    """
    print("\nAsked to transcribe rather than route:")
    for path in paths:
        words = len(ocr_text(Image.open(path)).split())
        reply, _ = vlm_reader.transcribe(loaded, path)
        print(f"  {path.name}")
        print(f"    ocr : {words} words")
        print(f"    vlm : {reply!r}")


def report(vlm_path=None):
    pipeline, _ = load_model()
    paths = find_images(MEDIA_DIR)
    tiers = []

    if can_read():
        tiers.append(("ocr", lambda image: ocr_route(pipeline, image)))
    else:
        log.warning("skipping ocr: needs the tesseract binary, brew install tesseract")

    vlm = None
    if vlm_reader.can_read(vlm_path):
        vlm = vlm_reader.load(vlm_path)
        tiers.append(("vlm", lambda image: vlm_route(vlm, image)))
    else:
        log.warning("skipping vlm: pass --vlm with a local model directory")

    results = {label: measure(paths, route_fn) for label, route_fn in tiers}

    scans = len(paths) * len(BLUR_RADII)
    print(f"\n{len(paths)} assets x {len(BLUR_RADII)} blur radii = {scans} scans per tier\n")
    print(f"{'tier':6}{'correct':>9}{'latency':>10}{'routes used':>13}   accuracy by blur radius")
    for label, (hits_by_radius, answers, latencies) in results.items():
        correct = sum(sum(hits) for hits in hits_by_radius.values())
        by_blur = "  ".join(
            f"r={r}:{sum(hits)}/{len(hits)}" for r, hits in sorted(hits_by_radius.items()))
        distinct = len({a for a in answers if a})
        latency = statistics.median(latencies) if latencies else 0
        print(f"{label:6}{correct:>4}/{scans:<4}{latency:>8.0f}ms{distinct:>10}/4     {by_blur}")

    for label, (_, answers, _) in results.items():
        route, times = Counter(answers).most_common(1)[0]
        if times > scans / 2:
            print(f"\n{label} answered {route!r} on {times} of {scans} scans, so its score is "
                  f"one guess landing on the assets that happen to match it, not discrimination.")

    if "ocr" in results:
        marker_path = Path(MEDIA_DIR) / "phishing-sms.png"
        found = BUBBLE_MARKER in extract_text(marker_path)[0]
        print(f"\nThe {BUBBLE_MARKER} line, white on blue, is "
              f"{'read' if found else 'missed'} by the two OCR passes.")

    if vlm:
        transcription_probe(paths, vlm)

    print(f"\n{len(results)} of 2 tiers measured. Everything here ran on-device, so none of these "
          f"numbers carries a per-ticket fee.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compare OCR against a vision model for routing.")
    p.add_argument("--vlm", default=None,
                   help="directory of a local vision language model to include as a second tier")
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report(parse_args().vlm)
