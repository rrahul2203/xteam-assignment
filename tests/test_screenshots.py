"""Tests for screenshot routing: redaction, the two-pass read, and the review gate.

Redaction is tested hardest. It runs on every image and it is the control that keeps credentials
out of the routed text and the output CSV, so a silent regression there is a leak rather than a
wrong answer. Those tests need no OCR and run everywhere.

The OCR tests are skipped rather than failed where the tesseract binary is absent, since it is a
system package and not something `pip install -r requirements.txt` can supply.
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from src.router.artifact import load_model, save_model
from src.router.data import LABELS
from src.router.screenshots import (MIN_OCR_CONFIDENCE, binarise, can_read, extract_text,
                                    find_images, redact, route_image)

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

    @pytest.mark.parametrize("secret, text", [
        ("j.moreira@example.com", "Email j.moreira@example.com Password"),
        ("448120", "I didn't make that withdrawal. The code is 448120"),
        ("bc1qxy2k", "move your balance to this holding wallet bc1qxy2k...9df8"),
        ("4155550142", "alert from 4155550142 about your account"),
        ("4111111111111111", "charge to card 4111111111111111 disputed"),
    ])
    def test_identifiers_do_not_survive(self, secret, text):
        redacted, found = redact(text)
        assert secret not in redacted
        assert found

    def test_the_category_is_kept_when_the_value_is_removed(self):
        """The route needs to know a credential was sent, which is the fraud signal itself."""
        redacted, found = redact("The code is 448120")
        assert "[otp]" in redacted
        assert found == ["otp"]

    def test_a_reference_number_is_not_mistaken_for_a_code(self):
        """Redacting every digit run would strip the evidence a dispute is argued from."""
        redacted, found = redact("Reference TX-88410-QQ Asset 0.061 BTC Pending 2 days")
        assert "88410" in redacted
        assert "otp" not in found

    def test_clean_text_is_returned_unchanged(self):
        text = "the transfer has not been broadcast and support chat keeps timing out"
        assert redact(text) == (text, [])

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


class TestBinarise:
    @needs_ocr
    def test_low_contrast_text_is_recovered_by_the_second_pass(self):
        """The customer's own bubble is white on blue and the default pass drops it.

        This is the whole reason extraction runs twice, so it is asserted on the real asset
        rather than on a synthetic image that might not reproduce the failure.
        """
        text, _ = extract_text(MEDIA_DIR / "phishing-sms.png")
        assert "448120" in text or "didn" in text.lower()

    @needs_ocr
    def test_binarising_produces_a_two_tone_image(self):
        with Image.open(MEDIA_DIR / "phishing-sms.png") as image:
            values = set(np.asarray(binarise(image)).ravel().tolist())
        assert values <= {0, 255}


class TestRouting:
    @needs_ocr
    @pytest.mark.parametrize("name", sorted(EXPECTED))
    def test_each_asset_reaches_its_route(self, pipeline, name):
        result = route_image(pipeline, MEDIA_DIR / name)
        assert result["route"] == EXPECTED[name]
        assert result["route"] in LABELS

    @needs_ocr
    def test_a_clean_screenshot_is_not_held_for_review(self, pipeline):
        """The gate has to pass legible scans, or it just moves every ticket to a human."""
        result = route_image(pipeline, MEDIA_DIR / "txn-failed.png")
        assert result["ocr_confidence"] > MIN_OCR_CONFIDENCE
        assert not result["review"]

    @needs_ocr
    def test_the_routed_text_carries_no_identifiers(self, pipeline):
        """Whatever is written to the CSV is what a downstream system stores."""
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


class TestDiscovery:
    def test_only_image_files_are_picked_up(self, tmp_path):
        for name in ("a.png", "b.jpg", "notes.txt", "data.csv"):
            (tmp_path / name).touch()
        assert [p.name for p in find_images(tmp_path)] == ["a.png", "b.jpg"]

    @needs_ocr
    def test_the_media_directory_holds_the_three_assets(self):
        assert [p.name for p in find_images(MEDIA_DIR)] == sorted(EXPECTED)
