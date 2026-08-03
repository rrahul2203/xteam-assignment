"""Sentence embeddings for the KB documents, with a cached fixture for use without torch.

Vectors are resolved in order, first hit winning: the committed fixture, then the model if it is
installed, then nothing. `DocumentVectors.available()` reports whether any were found, and the
retriever drops to TF-IDF alone when none were.

The fixture holds document vectors only. Encoding a question needs the model, so `can_encode()`
is a separate check from `available()`.

Document vectors do not depend on `as_of`, so they are encoded once and date filtering selects
rows from them. Only the query is encoded per call.
"""
import logging
from pathlib import Path

import numpy as np

# Optional dependency, imported at module scope so the failure is detected once at import time.
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

log = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Preferred over the hub name, so a vendored copy is used when present and works offline.
LOCAL_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "all-MiniLM-L6-v2"

_model = None
_model_tried = False


def default_vector_path():
    return Path(__file__).resolve().parents[2] / "eval" / "doc_vectors.npz"


def load_model():
    """Loads the model once and returns it, or None if it is unavailable.

    The result is cached including failure, so an absent dependency costs one attempt rather
    than one per question, and a present model is read from disk once per process.
    """
    global _model, _model_tried
    if _model_tried:
        return _model
    _model_tried = True
    if SentenceTransformer is None:
        log.info("sentence-transformers not installed, embeddings unavailable")
        return None
    source = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.is_dir() else MODEL_NAME
    log.info("loading embedding model from %s", source)
    try:
        _model = SentenceTransformer(source)
    except Exception as exc:
        # Catches a missing local copy, an unreachable hub, and hub client errors alike.
        log.warning("could not load embedding model from %s: %s", source, exc)
        _model = None
    return _model


def can_encode():
    """Whether arbitrary text can be embedded, which requires the model and not just vectors."""
    return load_model() is not None


def encode(texts):
    """Encodes `texts` to L2-normalised float32 rows, or returns None with no model.

    Normalising here makes a dot product the cosine, so ranking applies no further scaling.
    """
    model = load_model()
    if model is None:
        return None
    return np.asarray(
        model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )


def doc_text(doc):
    """Builds the text a document is embedded from, repeating the title to weight it."""
    return f"{doc.title}. {doc.title}. {doc.body}"


def save_vectors(docs, path=None):
    """Encodes every document and writes the fixture. Requires the model."""
    docs = list(docs.values()) if isinstance(docs, dict) else list(docs)
    vectors = encode([doc_text(d) for d in docs])
    if vectors is None:
        raise RuntimeError("cannot build the fixture without sentence-transformers installed")
    path = Path(path or default_vector_path())
    np.savez_compressed(
        path,
        doc_ids=np.array([d.doc_id for d in docs]),
        vectors=vectors,
        model=np.array(MODEL_NAME),
    )
    log.info("wrote %d vectors to %s", len(docs), path)
    return path


def load_vectors(path=None):
    """Reads the fixture into `{doc_id: vector}`, or returns None if absent or unreadable."""
    path = Path(path or default_vector_path())
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return dict(zip(data["doc_ids"].tolist(), data["vectors"]))
    except (OSError, ValueError, KeyError) as exc:
        log.warning("ignoring unreadable vector fixture %s: %s", path, exc)
        return None


class DocumentVectors:
    """Holds a vector per document, taken from the fixture and encoding only what it lacks.

    A KB edited after the fixture was written leaves documents uncovered. Those are encoded on
    construction when the model is present, and excluded from the embedding signal when not.
    """

    def __init__(self, docs, path=None):
        docs = list(docs.values()) if isinstance(docs, dict) else list(docs)
        self.vectors = load_vectors(path) or {}
        self.from_fixture = bool(self.vectors)

        missing = [d for d in docs if d.doc_id not in self.vectors]
        if missing:
            encoded = encode([doc_text(d) for d in missing])
            if encoded is None:
                log.info("no vectors for %d document(s) and no model to encode them",
                         len(missing))
            else:
                self.vectors.update(zip((d.doc_id for d in missing), encoded))

        self.covered = {d.doc_id for d in docs if d.doc_id in self.vectors}

    def available(self):
        """Whether any document has a vector, i.e. whether the signal can contribute at all."""
        return bool(self.covered)

    def matrix(self, docs):
        """Stacks the vectors for `docs` and returns them with the subset they correspond to.

        Documents without a vector are dropped rather than zero-filled: a zero row scores zero
        against every query, which ranks as merely irrelevant instead of as missing.
        """
        usable = [d for d in docs if d.doc_id in self.vectors]
        if not usable:
            return None, []
        return np.vstack([self.vectors[d.doc_id] for d in usable]), usable
