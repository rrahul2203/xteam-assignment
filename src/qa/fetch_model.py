"""Downloads the embedding model into models/ so inference never reaches the network.

    python3 -m src.qa.fetch_model

Without a local copy, the first question of the first run downloads ~90MB from the hub inside the
answering path. This vendors the model once, up front, and `load_model()` prefers that directory.

Files are fetched over urllib rather than huggingface_hub: the hub client is an extra dependency
for a fixed nine-file download, and only sentence-transformers needs to be installed to use the
result.
"""
import logging
import urllib.error
import urllib.request
from pathlib import Path

from .embeddings import LOCAL_MODEL_DIR, MODEL_NAME

log = logging.getLogger(__name__)

BASE_URL = f"https://huggingface.co/{MODEL_NAME}/resolve/main"

# Everything SentenceTransformer reads when loading from a directory. 1_Pooling/config.json
# defines the pooling layer, without which the model loads but produces no sentence vector.
FILES = (
    "config.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "1_Pooling/config.json",
)


def fetch_file(name, target_dir, timeout=120, attempts=3):
    """Downloads one file, skipping it when already present. Returns the path written.

    Retried because the weights file is ~87MB and a dropped connection part way through would
    otherwise fail the whole command.
    """
    destination = target_dir / name
    if destination.exists() and destination.stat().st_size > 0:
        log.info("  have %s", name)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Written to a temporary name first, so an interrupted download cannot look complete.
    partial = destination.with_suffix(destination.suffix + ".part")
    log.info("  get  %s", name)
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(f"{BASE_URL}/{name}", timeout=timeout) as response:
                partial.write_bytes(response.read())
            break
        except (urllib.error.URLError, OSError) as exc:
            partial.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            log.info("       attempt %d/%d failed (%s), retrying", attempt, attempts, exc)
    partial.replace(destination)
    return destination


def fetch_model(target_dir=None):
    """Downloads every file the model needs into `target_dir`."""
    target_dir = Path(target_dir or LOCAL_MODEL_DIR)
    log.info("fetching %s into %s", MODEL_NAME, target_dir)
    for name in FILES:
        try:
            fetch_file(name, target_dir)
        except (urllib.error.URLError, OSError) as exc:
            raise SystemExit(f"could not fetch {name}: {exc}\nthe repo still runs with --mode tfidf")
    log.info("done, %d files in %s", len(FILES), target_dir)
    return target_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fetch_model()
