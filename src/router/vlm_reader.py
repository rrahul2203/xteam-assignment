"""Reading screenshot text with a local vision language model, as an alternative to OCR.

This is the second extraction backend behind `screenshots.py --reader vlm`. It occupies the same
slot as OCR -- image in, text out, then the Part A classifier routes that text -- so switching
readers changes how the text is obtained and nothing else about the pipeline.

Transcription rather than routing is what this asks the model for, deliberately. A vision model can
be asked for the route in one step, and `screenshot_compare` measures that too, but then the
pipeline has no intermediate text to redact, to log, or to show a human reviewing the decision.
Asking it to read keeps the parts separable and keeps one classifier for both readers.

The model reports no per-word confidence, so a reader built here declares that it cannot support
the legibility gate. `route_image` holds every such scan for review rather than inventing a score.

The dependencies are optional and heavy, so absence is reported by `can_read` rather than raised at
import. See the README for what this backend measures at 500M parameters, which is worse than OCR
on every axis: it captions the screen instead of transcribing it.
"""
import logging
from pathlib import Path

from PIL import Image

# Optional and heavy: torch plus a local model checkout of ~1GB. Imported at module scope so a
# missing dependency is detected once, and reported through `can_read` rather than at the call site.
try:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
except ImportError:
    torch = None
    AutoModelForImageTextToText = None
    AutoProcessor = None

log = logging.getLogger(__name__)

READ_PROMPT = "Transcribe all of the text visible in this screenshot, exactly as it appears."

# Cap on generated tokens for a transcription. Set well above the ~140 words OCR reads from these
# assets, so a short reply is the model's own failure to transcribe rather than a truncation.
MAX_NEW_TOKENS = 512


def can_read(model_path):
    """Whether this backend can run: the transformers stack imports and the checkout exists."""
    return AutoProcessor is not None and model_path is not None and Path(model_path).is_dir()


def load(model_path):
    """Load the processor and model onto the best available device, once per run."""
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.float32)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("loaded vision model from %s onto %s", model_path, device)
    return processor, model.to(device).eval(), device


def ask(processor, model, device, image, prompt, max_new_tokens=MAX_NEW_TOKENS):
    """Put one image and one instruction to the model and return its reply as text."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=chat, images=[image.convert("RGB")], return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0].strip()


def transcribe(loaded, path):
    """Read one screenshot's text with the model. Returns (text, None) for the missing confidence."""
    with Image.open(path) as image:
        image.load()
        return ask(*loaded, image, READ_PROMPT), None
