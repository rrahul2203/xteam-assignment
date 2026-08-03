"""Encodes the KB and writes eval/doc_vectors.npz.

    python3 -m src.qa.build_vectors

Run this after editing the KB or changing the model. Needs sentence-transformers installed; the
committed fixture is what lets everything else run without it.
"""
import logging

from .embeddings import MODEL_NAME, save_vectors
from .kb import load_kb

log = logging.getLogger(__name__)


def main(kb_path=None, out_path=None):
    docs = load_kb(kb_path)
    path = save_vectors(docs, out_path)
    log.info("encoded %d documents with %s", len(docs), MODEL_NAME)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
