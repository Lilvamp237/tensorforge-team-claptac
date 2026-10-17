"""Pure-function tests for the evaluation helpers in scripts/evaluate.py and scripts/make_dataset.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate as ev  # noqa: E402
import make_dataset as md  # noqa: E402


def test_binary_metrics_known_confusion():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.3, 0.2])  # TP=2 FN=1 FP=1 TN=4
    m = ev.binary_metrics(y, p, 0.5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 5)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(6 / 8)


def test_binary_metrics_single_class_has_no_auc():
    assert ev.binary_metrics(np.ones(4), np.full(4, 0.9))["roc_auc"] is None


def test_precision_at_base_rate_bayes():
    # TPR 0.9, FPR 0.1 at 50% prevalence -> 0.9; at 1% -> 0.009 / (0.009 + 0.099)
    assert ev.precision_at_base_rate(0.9, 0.1, 0.5) == pytest.approx(0.9)
    assert ev.precision_at_base_rate(0.9, 0.1, 0.01) == pytest.approx(0.009 / 0.108)
    assert ev.precision_at_base_rate(0.0, 0.0, 0.05) == 0.0
    assert ev.precision_at_base_rate(0.9, 0.0, 0.01) == pytest.approx(1.0)


def test_base_rate_table_counts():
    t = ev.base_rate_table(0.8, 0.05, rates=(0.1,))["10%"]
    assert t["scams_caught_per_1000_offers"] == pytest.approx(80.0)
    assert t["scams_missed_per_1000_offers"] == pytest.approx(20.0)
    assert t["false_alarms_per_1000_offers"] == pytest.approx(45.0)
    assert t["false_alarms_per_scam_caught"] == pytest.approx(45 / 80, abs=0.01)


def test_bootstrap_ci_contains_point_estimate_and_is_deterministic():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.25, 400), 0, 1)
    ci = ev.bootstrap_ci(y, p, n_boot=200, seed=3)
    m = ev.binary_metrics(y, p)
    for k in ev.CI_METRICS:
        lo, hi = ci[k]
        assert lo <= round(m[k], 3) <= hi
        assert 0 <= lo <= hi <= 1
    assert ci == ev.bootstrap_ci(y, p, n_boot=200, seed=3)
    assert "precision_at_5%" in ci


def test_bootstrap_ci_perfect_classifier_is_degenerate():
    y = np.array([0, 1] * 50)
    ci = ev.bootstrap_ci(y, y.astype(float), n_boot=50)
    assert ci["f1"] == [1.0, 1.0] and ci["roc_auc"] == [1.0, 1.0]


def test_recommend_thresholds():
    y = np.array([0, 0, 1, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    rec = ev.recommend_thresholds(y, p, min_recall=0.9)
    a, b = rec["recall_at_least_0.9"], rec["best_f1"]
    # Recall >= 0.9 needs every scam, so the highest such threshold is 0.3 (P=4/6); best F1 is at 0.6 (P=1, R=0.75).
    assert a["threshold"] == pytest.approx(0.3) and a["recall"] == 1.0 and a["precision"] == pytest.approx(0.667)
    assert b["threshold"] == pytest.approx(0.6) and b["f1"] == pytest.approx(6 / 7, abs=1e-3)


def test_pr_curve_points_bounded():
    y = np.array([0, 1] * 100)
    p = np.linspace(0, 1, 200)
    pts = ev.pr_curve_points(y, p, n=20)
    assert 2 <= len(pts) <= 20
    assert all(0 <= q["precision"] <= 1 and 0 <= q["recall"] <= 1 for q in pts)


def test_mask_shortcuts_and_skeleton():
    a = "Ceylon Horizon Manpower - SLBFE licence 1187. Salary USD 1,100. Call 07X XXX XXXX."
    b = "Blue Lotus Foreign Employment - SLBFE licence 1542. Salary QAR 1,400. Call 07X XXX XXXX."
    masked = md.mask_shortcuts(a)
    for leak in ("Ceylon Horizon", "1187", "1,100", "07X"):
        assert leak not in masked
    assert "<AGENCY>" in masked and "<LICNO>" in masked and "<AMOUNT>" in masked and "<PHONE>" in masked
    assert md.skeleton(a) == md.skeleton(b)
    assert md.skeleton("Welder job in Qatar!") == md.skeleton("welder job in Kuwait")


def test_near_duplicates():
    ref = pd.Series(["Immediate vacancies for welders in Malta, monthly salary EUR 900, apply now",
                     "Study nursing in Canada with a full scholarship, pay the deposit today"])
    others = pd.Series(["Immediate vacancies for welders in Malta, monthly salary EUR 950, apply now",
                        "Data entry operator wanted in Colombo, send CV to HR"])
    sim, pairs = md.near_duplicates(ref, others)
    assert sim[0] > 0.9 > sim[1] and pairs == 1
