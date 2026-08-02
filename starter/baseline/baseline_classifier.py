"""Support-ticket route classifier, baseline.

Handed over from a previous engineer along with eval_report.md.
Trains on data/train.csv and reports how well it does.

    python3 baseline/baseline_classifier.py
"""
import csv
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
ROUTES = ["account-access", "transaction-dispute", "fraud-report", "general"]


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [r["label"] for r in rows]


def main():
    texts, labels = load(DATA)
    print(f"loaded {len(texts)} rows")

    # Vectorise the corpus, then split into train and test.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X = vectorizer.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=0
    )

    clf = LogisticRegression(max_iter=2000, C=10.0)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"test accuracy: {acc:.4f}")
    return acc


def predict(text):
    """predict(text) -> route label."""
    texts, labels = load(DATA)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X = vectorizer.fit_transform(texts)
    clf = LogisticRegression(max_iter=2000, C=10.0).fit(X, labels)
    return clf.predict(vectorizer.transform([text]))[0]


if __name__ == "__main__":
    main()
