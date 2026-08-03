"""Measuring how screenshot routing degrades, and what the review gate catches.

    python3 -m src.router.screenshot_eval

Three assets is too few to report an accuracy on, so this reports the shape of the failure
instead: each screenshot is blurred progressively and re-routed, and the outcome column records
whether the review gate fired before the route went wrong. Blur stands in for a photo of a screen,
a downscaled upload or a compressed forward, since all of those degrade the high-frequency detail
glyph recognition depends on.
"""
import logging

from PIL import ImageFilter

from .artifact import load_model
from .data import LABELS
from .predict import classify
from .screenshots import (MIN_OCR_CONFIDENCE, Image, _read_lines, binarise, can_read,
                          find_images, redact)

log = logging.getLogger(__name__)

MEDIA_DIR = "starter/media/screenshots"

# Gaussian blur radii in pixels, against a 780x1688 screenshot. The range runs past the point
# where body text stops being legible to a person, to bound the sweep on something meaningful.
BLUR_RADII = (0, 1, 2, 2.5, 3, 4)

# The route each asset should reach, read off the screenshot contents by hand. These are the
# labels being defended, not a sample anything is estimated from.
EXPECTED = {
    "login-error.png": "account-access",
    "phishing-sms.png": "fraud-report",
    "txn-failed.png": "transaction-dispute",
}


def route_blurred(pipeline, path, radius):
    """Route one screenshot at one blur radius, reusing the extraction the CLI uses."""
    with Image.open(path) as image:
        image.load()
        if radius:
            image = image.filter(ImageFilter.GaussianBlur(radius))
        lines = _read_lines(image) + _read_lines(binarise(image))

    seen, kept = set(), []
    for text, confidence in lines:
        key = " ".join(text.lower().split())
        if key and key not in seen:
            seen.add(key)
            kept.append((text, confidence))

    if not kept:
        return None, 0.0, 0.0
    ocr_confidence = sum(c for _, c in kept) / len(kept)
    text, _ = redact(" ".join(" ".join(t for t, _ in kept).split()))
    route, confidence = classify(pipeline, [text])[0]
    return route, confidence, ocr_confidence


def report():
    pipeline, _ = load_model()
    rows = []

    for path in find_images(MEDIA_DIR):
        expected = EXPECTED.get(path.name)
        print(f"\n{path.name}  expected {expected}")
        print(f"  {'blur':>5}  {'ocr conf':>8}  {'route':<20} {'conf':>6}  outcome")
        for radius in BLUR_RADII:
            route, confidence, ocr_confidence = route_blurred(pipeline, path, radius)
            correct = route == expected
            review = ocr_confidence < MIN_OCR_CONFIDENCE
            rows.append((correct, review))

            if review:
                outcome = "held for review"
            elif correct:
                outcome = "routed, correct"
            else:
                outcome = "ROUTED WRONG, unflagged"
            print(f"  {radius:>5}  {ocr_confidence:>8.1f}  {route or '<no text>':<20} "
                  f"{confidence:>6.3f}  {outcome}")

    auto = [(c, r) for c, r in rows if not r]
    held = [(c, r) for c, r in rows if r]
    wrong_unflagged = sum(1 for c, r in rows if not c and not r)

    print(f"\n{len(rows)} scans over {len(EXPECTED)} assets and {len(BLUR_RADII)} blur levels")
    print(f"  routed without review : {len(auto):2}, of which correct {sum(c for c, _ in auto)}")
    print(f"  held for review       : {len(held):2}, of which would have been correct "
          f"{sum(c for c, _ in held)}")
    print(f"  wrong and unflagged   : {wrong_unflagged}")

    # States the gate's headline result directly rather than leaving it to be read off the table.
    if wrong_unflagged == 0:
        print(f"\nNo misroute reached an unreviewed queue: every scan clearing "
              f"{MIN_OCR_CONFIDENCE:.0f} mean OCR confidence routed correctly.")
    print("The gate reads legibility, not correctness -- some held scans routed correctly anyway,"
          "\nand that is the cost paid for the line above.")
    print(f"\nRoutes are the same four as the text classifier: {', '.join(LABELS)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    if not can_read():
        raise SystemExit("OCR unavailable: needs the tesseract binary (brew install tesseract)")
    report()
