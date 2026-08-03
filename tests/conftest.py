"""Shared fixtures. Session-scoped because parsing the KB is the slow part of a run."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qa.answerer import AnswerService  # noqa: E402
from src.qa.kb import load_kb  # noqa: E402


@pytest.fixture(scope="session")
def kb():
    return load_kb()


@pytest.fixture(scope="session")
def service(kb):
    return AnswerService(kb)
