import pytest

from safepassage import drift
from safepassage.novelty import NoveltyIndex

REFERENCE = {
    "language": {"en": 0.6, "singlish": 0.25, "si": 0.15, "ta": 0.0, "mixed": 0.0, "unknown": 0.0},
    "category": {"local_job": 0.25, "foreign_job": 0.25, "student_visa": 0.25, "visa_pr": 0.25, "unknown": 0.0},
    "mean_confidence": 0.9,
    "novelty_share": 0.02,
    "thresholds": {"language_psi": 0.5, "category_psi": 0.4, "confidence_drop": 0.05, "novelty_share": 0.13},
}


def row(lang="en", cat="foreign_job", p=0.95, unfamiliar=False):
    return {"risk": {"probability": p}, "content": {"language": lang, "type": cat},
            "novelty": {"status": "unfamiliar_pattern" if unfamiliar else "familiar"}}


@pytest.fixture(autouse=True)
def fixed_reference(monkeypatch):
    monkeypatch.setattr(drift, "load_reference", lambda: REFERENCE)


def normal_rows(n=200):
    langs = ["en"] * 12 + ["singlish"] * 5 + ["si"] * 3
    cats = ["local_job", "foreign_job", "student_visa", "visa_pr"]
    return [row(langs[i % 20], cats[i % 4], 0.95 if i % 2 else 0.05) for i in range(n)]


def test_psi_zero_for_identical():
    assert drift.psi(REFERENCE["language"], REFERENCE["language"]) == pytest.approx(0, abs=1e-9)


def test_normal_traffic_does_not_fire():
    report = drift.drift_report(rows=normal_rows())
    assert report["drift"] is False and report["triggered"] == []


def test_shifted_traffic_fires():
    rows = normal_rows(100) + [row("ta", "foreign_job", 0.6, unfamiliar=True) for _ in range(100)]
    report = drift.drift_report(rows=rows)
    assert report["drift"] is True
    assert {"language_psi", "confidence_drop", "novelty_share"} <= set(report["triggered"])


def test_empty_log():
    assert drift.drift_report(rows=[])["drift"] is False


def test_novelty_scores_unfamiliar_text_higher():
    familiar = [f"Driver jobs in Qatar, salary QAR {1000 + i}. Interview at our office." for i in range(60)]
    familiar += [f"Study nursing in Australia, intake {i}. Offer letter from the university." for i in range(60)]
    index = NoveltyIndex(dims=8).fit(familiar[::2], familiar[1::2])
    texts = ["Driver jobs in Qatar, salary QAR 1500. Interview at our office.",
             "பதிவு கட்டணம் உடனடியாக செலுத்தவும், USDT wallet"]
    seen, new = index.distances(texts)
    assert new > seen
    assert all(0.0 <= s <= 1.0 for s in index.scores(texts))
