"""Tests for screenshot routing: redaction, the two-pass read, and the review gate.

Redaction is tested hardest. It runs on every image and it is the control that keeps credentials
out of the routed text and the output CSV, so a silent regression there is a leak rather than a
wrong answer. Those tests need no OCR and run everywhere.

The OCR tests are skipped rather than failed where the tesseract binary is absent, since it is a
system package and not something `pip install -r requirements.txt` can supply.
"""
from pathlib import Path

import pytest
from PIL import Image, ImageFilter

from src.router.artifact import load_model, save_model
from src.router.screenshots import (MIN_OCR_CONFIDENCE, can_read, extract_text, find_images,
                                    redact, route_image)

MEDIA_DIR = Path(__file__).resolve().parents[1] / "starter" / "media" / "screenshots"

needs_ocr = pytest.mark.skipif(
    not can_read(), reason="needs the tesseract binary: brew install tesseract"
)

# The route each asset should reach, duplicated from screenshot_eval so a change to the reporting
# script cannot quietly change what the tests assert.
EXPECTED = {
    "login-error.png": "account-access",
    "phishing-sms.png": "fraud-report",
    "txn-failed.png": "transaction-dispute",
}


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """A model saved to a temporary path, so tests never depend on models/router.joblib."""
    path = save_model(path=tmp_path_factory.mktemp("model") / "router.joblib")
    return load_model(path)[0]


class TestRedaction:
    """What must never survive into the routed text or the output CSV."""

    def test_identifiers_do_not_survive(self):
        cases = [
            ("j.moreira@example.com", "Email j.moreira@example.com Password"),
            ("448120", "I didn't make that withdrawal. The code is 448120"),
            ("bc1qxy2k", "move your balance to this holding wallet bc1qxy2k...9df8"),
            ("4155550142", "alert from 4155550142 about your account"),
            ("4111111111111111", "charge to card 4111111111111111 disputed"),
        ]
        for secret, text in cases:
            redacted, found = redact(text)
            assert secret not in redacted, f"{secret} survived redaction"
            assert found

    def test_the_category_is_kept_and_a_reference_number_is_left_alone(self):
        """A code collapses to its category; a reference number of the same shape survives.

        Both directions are the same design decision. The route needs to know a credential was
        sent, because that is the fraud signal itself, but redacting every digit run on shape
        alone would also strip the evidence a dispute is argued from.
        """
        redacted, found = redact("The code is 448120")
        assert "[otp]" in redacted
        assert found == ["otp"]

        redacted, found = redact("Reference TX-88410-QQ Asset 0.061 BTC Pending 2 days")
        assert "88410" in redacted
        assert "otp" not in found

    def test_a_wallet_address_is_redacted_before_the_phone_pattern_splits_it(self):
        """Ordering matters where an address body is all digits.

        The phone pattern matches that digit run too. Running it first consumes the middle and
        leaves the `bc1q` prefix behind as literal text, which is a partial address in the output
        rather than a redacted one, so the address pattern has to go first.
        """
        redacted, found = redact("send to bc1q98765432109876543210 now")
        assert "98765432109876543210" not in redacted
        assert "bc1q" not in redacted
        assert found == ["wallet"]


class TestRouting:
    @needs_ocr
    def test_each_asset_reaches_its_route_without_review(self, pipeline):
        """The gate has to pass legible scans, or it just moves every ticket to a human."""
        for name, expected in sorted(EXPECTED.items()):
            result = route_image(pipeline, MEDIA_DIR / name)
            assert result["route"] == expected, f"{name} routed to {result['route']}"
            assert result["ocr_confidence"] > MIN_OCR_CONFIDENCE
            assert not result["review"]

    @needs_ocr
    def test_low_contrast_text_is_read_but_never_routed_verbatim(self, pipeline):
        """The customer's own bubble is white on blue and the default pass drops it.

        Recovering it is the whole reason extraction runs twice, so it is asserted on the real
        asset rather than a synthetic image that might not reproduce the failure. The same code
        then has to disappear from the routed text, since that text is what a downstream system
        stores -- recovering a credential and then leaking it would be worse than missing it.
        """
        text, _ = extract_text(MEDIA_DIR / "phishing-sms.png")
        assert "448120" in text

        result = route_image(pipeline, MEDIA_DIR / "phishing-sms.png")
        assert "448120" not in result["text"]
        assert "@example.com" not in result["text"]

    @needs_ocr
    def test_a_blurred_screenshot_is_held_rather_than_routed_silently(self, pipeline, tmp_path):
        """Degradation must surface as review, not as a confident route on a fragment."""
        blurred = tmp_path / "blurred.png"
        with Image.open(MEDIA_DIR / "login-error.png") as image:
            image.filter(ImageFilter.GaussianBlur(4)).save(blurred)
        assert route_image(pipeline, blurred)["review"]

    def test_an_unreadable_image_never_reaches_the_classifier(self, pipeline, tmp_path,
                                                              monkeypatch):
        """No recognised text is a failed read, not a `general` ticket."""
        monkeypatch.setattr("src.router.screenshots.extract_text", lambda path: ("", 0.0))
        result = route_image(pipeline, tmp_path / "blank.png")
        assert result["route"] is None
        assert result["review"]

    def test_a_reader_that_cannot_score_itself_is_always_held(self, pipeline, tmp_path):
        """Swap in a reader returning None for the confidence, as the vision backend does.

        The route still has to be produced, since the text was readable, but the legibility gate
        has nothing to threshold -- and an unscored read is a weaker claim than a legible one, so
        it must not inherit the pass that a scored read would get.
        """
        result = route_image(pipeline, tmp_path / "unused.png",
                             reader=lambda path: ("I cannot log in to my account", None))
        assert result["route"] == "account-access"
        assert result["ocr_confidence"] is None
        assert result["review"]

    def test_only_image_files_are_picked_up(self, tmp_path):
        for name in ("a.png", "b.jpg", "notes.txt", "data.csv"):
            (tmp_path / name).touch()
        assert [p.name for p in find_images(tmp_path)] == ["a.png", "b.jpg"]
