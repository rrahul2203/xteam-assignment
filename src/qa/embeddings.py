"""Sentence embeddings for the KB, with a cached fixture so the repo runs without torch.

Loading order, first hit wins: the committed vector fixture, then sentence-transformers if it
is installed, then nothing. `available()` reports which happened, and the retriever falls back
to TF-IDF alone when the answer is nothing, so a clone with only scikit-learn still works.

The fixture holds document vectors keyed by doc_id, which covers batch answering and the
evaluation. A question typed at the CLI is text the fixture cannot know, so single-question mode
needs the model itself; without it that path is lexical only.

Document vectors do not depend on `as_of`, so they are encoded once and date filtering becomes a
row selection. Only the query has to be encoded per call.
"""
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Local checkout used in preference to the hub, so a cached copy works offline.
LOCAL_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "all-MiniLM-L6-v2"

_model = None
_model_tried = False


def default_vector_path():
    return Path(__file__).resolve().parents[2] / "eval" / "doc_vectors.npz"


def load_model():
    """The sentence-transformers model, or None when it cannot be loaded.

    Cached including the failure, so a missing dependency costs one import attempt rather than
    one per question.
    """
    global _model, _model_tried
    if _model_tried:
        return _model
    _model_tried = True
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.info("sentence-transformers not installed, embeddings unavailable")
        return None
    source = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.is_dir() else MODEL_NAME
    try:
        _model = SentenceTransformer(source)
    except Exception as exc:
        # Covers a missing local copy, no network for the hub, and hub client failures.
        log.warning("could not load embedding model from %s: %s", source, exc)
        _model = None
    return _model


def can_encode():
    """True when arbitrary text can be embedded, which needs the model rather than the fixture.

    The fixture covers documents only, so a question typed at the CLI cannot be encoded from it.
    """
    return load_model() is not None


def encode(texts):
    """L2-normalised vectors for `texts`, or None when no model is available.

    Normalising here means a dot product is the cosine, so ranking needs no further scaling.
    """
    model = load_model()
    if model is None:
        return None
    return np.asarray(
        model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )


def doc_text(doc):
    """The text a document is embedded from. Title twice, matching the TF-IDF corpus."""
    return f"{doc.title}. {doc.title}. {doc.body}"


def save_vectors(docs, path=None):
    """Encode every document and write the fixture. Requires the model."""
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
    """`{doc_id: vector}` from the fixture, or None when it is absent or unreadable."""
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
    """Document vectors from the fixture where possible, encoding only what is missing.

    A KB edited after the fixture was written leaves some documents uncovered; those are encoded
    on construction if the model is present and dropped from the embedding signal if it is not.
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
        """True when the embedding signal can contribute at all."""
        return bool(self.covered)

    def matrix(self, docs):
        """Stacked vectors for `docs`, and the subset they correspond to.

        Documents without a vector are dropped rather than zero-filled, since a zero row scores
        zero against every query and would silently rank as merely irrelevant.
        """
        usable = [d for d in docs if d.doc_id in self.vectors]
        if not usable:
            return None, []
        return np.vstack([self.vectors[d.doc_id] for d in usable]), usable
