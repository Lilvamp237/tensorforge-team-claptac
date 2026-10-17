"""Checks for data/handwritten.csv, the small real-world test set (test-only, never used for training)."""
import csv
import re
from pathlib import Path

import pytest

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "handwritten.csv"
COLUMNS = ["text", "label", "category", "language", "source_url", "label_basis", "needs_review"]
LABELS = {"scam", "legit"}
CATEGORIES = {"local_job", "foreign_job", "student_visa", "visa_pr"}
LANGUAGES = {"en", "si", "ta", "singlish", "mixed"}

# Sri Lankan phone numbers: 07X XXXXXXX (with optional spaces/dashes), landlines 0XX XXXXXXX, +94 / 0094 forms.
PHONE_RE = re.compile(r"(?:\+94|0094|\b0)[\s-]?\d{2}[\s-]?\d{3}[\s-]?\d{4}\b")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Old NIC: 9 digits + V/X; new NIC: 12 digits.
NIC_RE = re.compile(r"\b(?:\d{9}[VvXx]|\d{12})\b")


@pytest.fixture(scope="module")
def rows():
    with CSV_PATH.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == COLUMNS
        return list(reader)


def test_enough_rows(rows):
    assert len(rows) >= 40


def test_both_labels_present(rows):
    assert {r["label"] for r in rows} == LABELS


def test_allowed_values(rows):
    for i, r in enumerate(rows, start=2):
        assert r["label"] in LABELS, i
        assert r["category"] in CATEGORIES, i
        assert r["language"] in LANGUAGES, i
        assert r["needs_review"] in {"yes", "no"}, i
        assert r["text"].strip(), i
        assert r["source_url"].startswith("http"), i
        assert r["label_basis"].strip(), i


def test_texts_are_short(rows):
    for i, r in enumerate(rows, start=2):
        assert len(r["text"]) <= 800, i


@pytest.mark.parametrize("pattern", [PHONE_RE, EMAIL_RE, NIC_RE], ids=["phone", "email", "nic"])
def test_no_personal_data(rows, pattern):
    hits = [(i, m.group()) for i, r in enumerate(rows, start=2) for m in [pattern.search(r["text"])] if m]
    assert not hits, hits


def test_patterns_catch_examples():
    assert PHONE_RE.search("call 077 1234567")
    assert PHONE_RE.search("call 0771234567")
    assert PHONE_RE.search("+94 77 123 4567")
    assert PHONE_RE.search("Tel: 011 2677380")
    assert EMAIL_RE.search("cv@example.lk")
    assert NIC_RE.search("NIC 912345678V")
    assert NIC_RE.search("NIC 199112345678")
    assert not PHONE_RE.search("Rs.1,235,532")
