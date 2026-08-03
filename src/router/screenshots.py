"""Routing support screenshots into the four text routes, via local OCR.

    python3 -m src.router.screenshots --dir starter/media/screenshots
    python3 -m src.router.screenshots --image starter/media/screenshots/txn-failed.png

Screenshots are the chosen modality because their text is rendered rather than photographed, so
OCR reads back what a layout engine drew instead of inferring glyphs from a camera. The approach
follows from that: extract text locally, then hand it to the Part A classifier, so this part adds
a text-extraction stage rather than a second model to train and evaluate.

Extraction runs Tesseract twice per image and unions the lines, because these screenshots are
light-on-dark and the two passes fail on different text -- the default pass misses low-contrast
bubbles, and the luminance pass reads those but loses more on a soft image. Identifiers are
redacted from the text before it is classified or written.

Both stages are local: OCR is a system binary and the classifier is a saved artifact, so pixels
and extracted text never leave the machine. See the README for the measured cost of each choice.
"""
import argparse
import csv
import logging
import re
from pathlib import Path

import numpy as np

from .artifact import load_model
from .predict import classify

# Optional dependencies, imported at module scope so a missing one is detected once at import
# time. Pillow decodes the PNG; pytesseract shells out to the `tesseract` binary, which is a
# system package rather than a wheel, so its absence is reported separately by `can_read`.
try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

# Luminance cut for the second OCR pass, on a 0-255 scale. Sits above the panel fills these
# screenshots use and below their body text, so text of any hue binarises to black-on-white.
LUMINANCE_THRESHOLD = 140

# Mean per-word OCR confidence, 0-100, below which a route is held for review instead of returned.
# Set from the blur sweep in screenshot_eval. The gate measures whether the scan was legible, not
# whether the route is right -- see `route_image` for why that distinction matters.
MIN_OCR_CONFIDENCE = 60.0

# Identifiers replaced before classification, each by its own category name so the route still
# sees that a credential was present. Ordered so `wallet` runs before `phone`, whose digit run
# would otherwise consume the middle of an all-digit address.
REDACTIONS = (
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w]{2,}\b")),
    ("card", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    # Bitcoin-style addresses, allowing the ellipsis a UI inserts when it truncates one.
    ("wallet", re.compile(r"\b(?:bc1|0x)[a-zA-Z0-9]{2,}(?:\.{2,}[a-zA-Z0-9]+)?\b")),
    ("phone", re.compile(r"\+?\d[\d\s().-]{7,}\d")),
    # One-time codes, matched via a nearby keyword rather than on shape alone, since a bare
    # 6-digit run is also a reference number or an amount the route argues from.
    ("otp", re.compile(r"\b(code|otp|pin)\b(?:\s+\w+){0,3}?\s+(\d{4,8})\b", re.IGNORECASE)),
)


def can_read():
    """Whether OCR can run: the Python bindings import and the tesseract binary answers."""
    if pytesseract is None:
        return False
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        # pytesseract raises its own EnvironmentError subclass when the binary is absent, and
        # the underlying call can also fail on a broken install, so neither is worth narrowing.
        return False
    return True


def binarise(image):
    """Flatten an image to black text on white by thresholding luminance.

    Weighted luminance rather than a single channel, because a saturated colour can be bright in
    one channel and dark in the others, which leaves text on it below any per-channel cut.
    """
    rgb = np.asarray(image.convert("RGB")).astype(np.int32)
    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return Image.fromarray(((luminance > LUMINANCE_THRESHOLD) * 255).astype(np.uint8))


def _read_lines(image):
    """OCR one image into [(line text, mean word confidence)], dropping unrecognised boxes."""
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    lines = {}
    for line_number, text, confidence in zip(data["line_num"], data["text"], data["conf"]):
        # Tesseract emits a box per word, scores an unrecognised one -1, and pads the layout
        # with blanks, so both are filtered before the words are joined back into a line.
        if str(text).strip() and float(confidence) >= 0:
            lines.setdefault(line_number, []).append((str(text).strip(), float(confidence)))
    return [(" ".join(w for w, _ in words), float(np.mean([c for _, c in words])))
            for words in lines.values()]


def extract_text(path):
    """Read an image with both passes and return (text, mean word confidence).

    Lines are deduplicated case-insensitively, so text both passes see is counted once and the
    confidence average is not weighted toward whatever happens to be legible twice.
    """
    with Image.open(path) as image:
        image.load()
        passes = (_read_lines(image), _read_lines(binarise(image)))

    seen, kept = set(), []
    for lines in passes:
        for text, confidence in lines:
            key = " ".join(text.lower().split())
            if key and key not in seen:
                seen.add(key)
                kept.append((text, confidence))

    if not kept:
        return "", 0.0
    return " ".join(text for text, _ in kept), float(np.mean([c for _, c in kept]))


def redact(text):
    """Replace identifiers with their category name. Returns (text, categories found)."""
    found = []
    for name, pattern in REDACTIONS:
        text, count = pattern.subn(f"[{name}]", text)
        if count:
            found.append(name)
    return text, found


def route_image(pipeline, path):
    """Route one screenshot, returning the redacted text, the route and both confidences.

    `review` marks the scan as too degraded to trust rather than the route as wrong. The
    distinction is deliberate: a low-confidence scan sometimes routes correctly anyway, but it
    does so on a fragment, so the gate is there to keep an unreviewed route from resting on text
    nobody could read, not to predict which routes are mistakes.
    """
    text, ocr_confidence = extract_text(path)
    text, redacted = redact(" ".join(text.split()))
    if redacted:
        log.info("%s: redacted %s before classifying", Path(path).name, ", ".join(redacted))

    if not text:
        # No text at all is a failed read, not a `general` ticket, so it never reaches the model.
        log.warning("%s: no text recognised, sending to review", Path(path).name)
        return {"file": Path(path).name, "route": None, "confidence": 0.0,
                "ocr_confidence": 0.0, "review": True, "text": ""}

    route, confidence = classify(pipeline, [text])[0]
    review = ocr_confidence < MIN_OCR_CONFIDENCE
    if review:
        log.warning("%s: OCR confidence %.1f below %.0f, sending %s to review",
                    Path(path).name, ocr_confidence, MIN_OCR_CONFIDENCE, route)
    return {"file": Path(path).name, "route": route, "confidence": confidence,
            "ocr_confidence": ocr_confidence, "review": review, "text": text}


def find_images(directory):
    return sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Route support screenshots via local OCR.")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--dir", help="directory of screenshots to route")
    source.add_argument("--image", help="route a single screenshot and print the route")
    p.add_argument("--output", default=None, help="CSV to write, defaults to stdout only")
    p.add_argument("--model", default=None, help="saved model to load, defaults to models/")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not can_read():
        raise SystemExit(
            "OCR unavailable: needs the tesseract binary (brew install tesseract) and "
            "pip install -r requirements.txt")

    try:
        pipeline, _ = load_model(args.model)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    paths = [Path(args.image)] if args.image else find_images(args.dir)
    if not paths:
        raise SystemExit(f"no images found in {args.dir}")

    results = [route_image(pipeline, p) for p in paths]
    for r in results:
        # The route is the payload, so it goes to stdout; the review flag rides with it because
        # a caller acting on the route needs to know whether it was trusted.
        flag = "\treview" if r["review"] else ""
        print(f"{r['file']}\t{r['route']}\t{r['confidence']:.4f}\t{r['ocr_confidence']:.1f}{flag}")

    if args.output:
        fields = ["file", "route", "confidence", "ocr_confidence", "review", "text"]
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in results:
                w.writerow({**r, "confidence": round(r["confidence"], 4),
                            "ocr_confidence": round(r["ocr_confidence"], 1)})
        log.info("wrote %d rows to %s", len(results), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
