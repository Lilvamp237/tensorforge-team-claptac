"""Fit the novelty index and calibrate drift thresholds. Run after train_baseline.py.

1. Split train.csv 90/10. Fit the kNN novelty index on the 90%; the 10% calibration split sets the score scale.
2. Split the held-out test set 50/50. Half A is scored like live traffic -> reference distributions (language,
   category, confidence, novelty). It must be data the scam model never trained on, or the reference confidence is
   inflated and the confidence signal false-alarms.
3. Bootstrap 200-item windows from half A (normal) and the shifted set (drift). For each signal the threshold sits
   between the normal p99 and the shifted median.
4. Check on data the thresholds never saw: false-alarm rate on half B windows, detection rate on shifted windows and
   on mixed windows (50% and 25% shifted).
Writes models/novelty.joblib, models/drift_reference.json and reports/drift_calibration.json.
With SAFEPASSAGE_MODEL_DIR set, reads the models from and writes all three files to that directory instead.
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from safepassage import extract
from safepassage.config import DATA_DIR, MODEL_PATH, NOVELTY_THRESHOLD, REPORTS_DIR
from safepassage.drift import REFERENCE_PATH, compute_signals, reference_from_rows, triggered
from safepassage.extract import classify_category, detect_language
from safepassage.novelty import NOVELTY_PATH, NoveltyIndex

MODEL_DIR = os.environ.get("SAFEPASSAGE_MODEL_DIR")
if MODEL_DIR:  # candidate models: read and write everything inside that directory
    MODEL_DIR = Path(MODEL_DIR)
    MODEL_PATH, NOVELTY_PATH, REFERENCE_PATH = (MODEL_DIR / p.name for p in (MODEL_PATH, NOVELTY_PATH, REFERENCE_PATH))
    extract.CATEGORY_PATH = MODEL_DIR / extract.CATEGORY_PATH.name
    REPORTS_DIR = MODEL_DIR

WINDOW = 200
N_WINDOWS = 300
SIGNALS = ["language_psi", "category_psi", "confidence_drop", "novelty_share"]


def score_rows(texts: list[str], scam_model, index: NoveltyIndex) -> list[dict]:
    """Rows shaped like the prediction log, so drift.py treats them exactly like live traffic."""
    p = scam_model.predict_proba(texts)[:, list(scam_model.classes_).index("scam")]
    nov = index.scores(texts)
    return [{
        "risk": {"probability": float(pi)},
        "content": {"language": detect_language(t), "type": classify_category(t)},
        "novelty": {"score": float(ni), "status": "unfamiliar_pattern" if ni >= NOVELTY_THRESHOLD else "familiar"},
    } for t, pi, ni in zip(texts, p, nov)]


def windows(rows: list[dict], rng: np.random.Generator, n: int = N_WINDOWS, mix: list[dict] | None = None, share=0.5):
    for _ in range(n):
        if mix is None:
            idx = rng.integers(0, len(rows), WINDOW)
            yield [rows[i] for i in idx]
        else:
            k = int(WINDOW * share)
            yield ([rows[i] for i in rng.integers(0, len(rows), WINDOW - k)]
                   + [mix[i] for i in rng.integers(0, len(mix), k)])


def main():
    rng = np.random.default_rng(42)
    train = pd.read_csv(DATA_DIR / "processed" / "train.csv")
    test = pd.read_csv(DATA_DIR / "processed" / "test.csv")
    shifted = pd.read_csv(DATA_DIR / "processed" / "shifted.csv")

    fit, calib = train_test_split(train, test_size=0.1, stratify=train["source"] + train["label"], random_state=42)
    index = NoveltyIndex().fit(fit["text"], calib["text"])
    index.save(NOVELTY_PATH)
    print(f"Novelty index: {len(fit)} items, calib distance median={index.median:.3f} p99={index.p99:.3f}")

    scam_model = joblib.load(MODEL_PATH)
    ref_half, check_half = train_test_split(test, test_size=0.5, stratify=test["source"] + test["label"],
                                            random_state=42)
    ref_rows = score_rows(ref_half["text"].tolist(), scam_model, index)
    test_rows = score_rows(check_half["text"].tolist(), scam_model, index)
    shifted_rows = score_rows(shifted["text"].tolist(), scam_model, index)

    reference = reference_from_rows(ref_rows)
    normal = pd.DataFrame([compute_signals(w, reference) for w in windows(ref_rows, rng)])
    drifted = pd.DataFrame([compute_signals(w, reference) for w in windows(shifted_rows, rng)])

    thresholds, separation = {}, {}
    for s in SIGNALS:
        hi_normal, mid_shift = normal[s].quantile(0.99), drifted[s].median()
        thr = hi_normal + 0.25 * (mid_shift - hi_normal) if mid_shift > hi_normal else hi_normal * 1.5 + 1e-3
        thresholds[s] = round(float(thr), 4)
        separation[s] = {"normal_p99": round(float(hi_normal), 4), "shifted_median": round(float(mid_shift), 4)}
    reference["thresholds"] = thresholds
    REFERENCE_PATH.write_text(json.dumps(reference, indent=2))

    def fire_rate(rows, **kw):
        fired = [triggered(compute_signals(w, reference), thresholds) for w in windows(rows, rng, **kw)]
        per_signal = {s: round(float(np.mean([s in f for f in fired])), 3) for s in SIGNALS}
        return {"any": round(float(np.mean([bool(f) for f in fired])), 3), "per_signal": per_signal}

    report = {
        "window": WINDOW,
        "n_windows": N_WINDOWS,
        "thresholds": thresholds,
        "separation": separation,
        "false_alarm_rate_heldout_test": fire_rate(test_rows),
        "detection_rate_shifted": fire_rate(shifted_rows),
        "detection_rate_mixed_50pct_shifted": fire_rate(test_rows, mix=shifted_rows, share=0.5),
        "detection_rate_mixed_25pct_shifted": fire_rate(test_rows, mix=shifted_rows, share=0.25),
        "full_set_signals": {
            "heldout_test": compute_signals(test_rows, reference),
            "shifted": compute_signals(shifted_rows, reference),
        },
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "drift_calibration.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ["thresholds", "separation", "false_alarm_rate_heldout_test",
                                             "detection_rate_shifted", "detection_rate_mixed_50pct_shifted",
                                             "detection_rate_mixed_25pct_shifted", "full_set_signals"]}, indent=1))


if __name__ == "__main__":
    main()
