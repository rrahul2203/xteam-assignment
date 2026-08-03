"""Loading training data and deriving template groups for honest validation.

train.csv is synthetic: rows come from a small set of base templates with slots filled in
(coin, device, amount) and a greeting or sign-off attached. Grouping by template and
splitting on the group forces every validation message to come from a template the model
never saw.
"""
import csv
import re
from pathlib import Path

LABELS = ("account-access", "fraud-report", "general", "transaction-dispute")
FRAUD = "fraud-report"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

_GREETING = re.compile(
    r"^(hi team|hello team|hey team|dear team|dear support|hi there|hi|hey|hello|team|"
    r"quick question|urgent:|please help\.|good morning|good afternoon)[,.:!]?\s*",
    re.IGNORECASE,
)
_SIGNOFF = re.compile(
    r"\s*(thanks in advance|thanks|thank you|please advise|appreciate any help|"
    r"any help appreciated|let me know|this is time sensitive|any update|best regards|"
    r"regards|many thanks|cheers)[.!,]?$",
    re.IGNORECASE,
)
_COIN = re.compile(
    r"\b(btc|eth|sol|ada|usdc|usdt|matic|ltc|xrp|doge|bnb|dot|xlm|bitcoin|ethereum|"
    r"solana|cardano|polygon|litecoin|ripple|dogecoin|tether|arbitrum|avalanche|"
    r"optimism|polkadot|stellar|tron)\b",
    re.IGNORECASE,
)
_DEVICE = re.compile(
    r"\b(android app|new iphone|iphone|android|work computer|desktop browser|browser|"
    r"laptop|tablet|phone|app)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"[$£€]?\s?\d[\d,]*(\.\d+)?")


def template_key(text):
    """Reduce a message to the template it was generated from.

    Greetings and sign-offs stack, so peel repeatedly.
    """
    s = " ".join(str(text).split())
    for _ in range(4):
        peeled = _SIGNOFF.sub("", _GREETING.sub("", s)).strip()
        if peeled == s:
            break
        s = peeled

    key = _COIN.sub("<coin>", s)
    key = _DEVICE.sub("<device>", key)
    key = _NUMBER.sub("<num>", key)
    return re.sub(r"[^a-z0-9<>]+", " ", key.lower()).strip()


def groups(texts):
    """Group ids so siblings of one template never straddle a split."""
    return [template_key(t) for t in texts]


def load(path):
    """Read a labelled CSV into (texts, labels), dropping exact repeats."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} contains no rows")

    missing = {TEXT_COLUMN, LABEL_COLUMN} - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing column(s) {sorted(missing)}")

    seen = set()
    texts, labels = [], []
    for r in rows:
        pair = (r[TEXT_COLUMN].strip(), r[LABEL_COLUMN].strip())
        if not pair[0] or pair in seen:
            continue
        seen.add(pair)
        texts.append(pair[0])
        labels.append(pair[1])

    unknown = set(labels) - set(LABELS)
    if unknown:
        raise ValueError(f"unknown label(s) {sorted(unknown)}")
    return texts, labels


def load_texts(path, text_column=None):
    """Read an unlabelled CSV. Falls back to the first column if there is no `text`."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} contains no rows to score")

    column = text_column or (TEXT_COLUMN if TEXT_COLUMN in rows[0] else list(rows[0])[0])
    if column not in rows[0]:
        raise ValueError(f"column {column!r} not found in {path}")
    return [r[column] for r in rows]


def default_data_path():
    """starter/data/train.csv, resolved from this file: src/router/data.py -> repo root."""
    return Path(__file__).resolve().parents[2] / "starter" / "data" / "train.csv"
