"""Run the honest evaluation of the baseline.

    python3 src/evaluate_baseline.py
"""
import logging

from data import default_data_path
from evaluate import report

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report(default_data_path())
