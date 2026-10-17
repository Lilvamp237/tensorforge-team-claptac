"""Train the stacking meta-model (src/safepassage/stacker.py). Run after train_baseline.py.

    PYTHONPATH=src python scripts/train_stacker.py

1. Out-of-fold text-model scores on train: GroupKFold(5) grouped by a text skeleton, so near-identical
   synthetic templates never sit on both sides of a fold.
2. Meta features per row: text score, red flags, official-record verification.
3. Logistic regression (standardised features, balanced classes) on train.
4. Threshold chosen on VAL only (lowest false-alarm rate with recall >= 0.95). The real-world set and the test
   sets are scored once at the end, never used to choose anything.
Writes models/stacker.joblib and reports/stacker.json.
"""

import json
import re
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from safepassage import stacker
from safepassage.config import DATA_DIR, MODEL_PATH, REPORTS_DIR
from safepassage.extract import extract_content
from safepassage.features import find_red_flags
from safepassage.verify import verify

TARGET_RECALL = 0.95


def skeleton(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "0", text.lower()))[:120]


def meta_matrix(texts: pd.Series, probs: np.ndarray) -> np.ndarray:
    rows = []
    for text, p in zip(texts.astype(str), probs):
        rows.append(stacker.features(p, find_red_flags(text), verify(text, extract_content(text))))
    return np.array(rows)


def counts(y, p, thr):
    pred = p >= thr
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    fp, tn = int((pred & (y == 0)).sum()), int((~pred & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"threshold": round(float(thr), 3), "missed_scams": fn, "scams": tp + fn, "false_alarms": fp,
            "legit": fp + tn, "precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 3), "false_alarm_rate": round(fp / max(fp + tn, 1), 3)}


def pick_threshold(y, p) -> float:
    best = 0.5
    for thr in np.linspace(0.05, 0.95, 181):
        if (p[y == 1] >= thr).mean() >= TARGET_RECALL:
            best = thr  # the highest threshold that still keeps recall >= target has the fewest false alarms
    return float(best)


def main():
    t0 = time.time()
    load = lambda name: pd.read_csv(DATA_DIR / "processed" / f"{name}.csv")
    train, val = load("train"), load("val")
    y_train = (train["label"] == "scam").astype(int).to_numpy()
    base = joblib.load(MODEL_PATH)

    oof = np.zeros(len(train))
    groups = train["text"].astype(str).map(skeleton)
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(train, y_train, groups)):
        model = clone(base).fit(train["text"].iloc[tr], train["label"].iloc[tr])
        oof[te] = model.predict_proba(train["text"].iloc[te])[:, list(model.classes_).index("scam")]
        print(f"fold {fold + 1}/5 done ({time.time() - t0:.0f}s)", flush=True)

    scam_idx = list(base.classes_).index("scam")
    X_train = meta_matrix(train["text"], oof)
    meta = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    meta.fit(X_train, y_train)
    print(f"meta features built and fitted ({time.time() - t0:.0f}s)", flush=True)

    def both(df):
        y = (df["label"] == "scam").astype(int).to_numpy()
        p_text = base.predict_proba(df["text"].astype(str))[:, scam_idx]
        p_stack = meta.predict_proba(meta_matrix(df["text"], p_text))[:, 1]
        return y, p_text, p_stack

    y_val, pt_val, ps_val = both(val)
    threshold = pick_threshold(y_val, ps_val)

    report = {"threshold_rule": f"highest threshold with val recall >= {TARGET_RECALL}", "threshold": threshold,
              "weights": dict(zip(stacker.FEATURES, np.round(meta[-1].coef_[0], 3).tolist()))}
    sets = {"val": None, "heldout_test": DATA_DIR / "processed" / "test.csv",
            "shifted": DATA_DIR / "processed" / "shifted.csv", "real_world_TEST_ONLY": DATA_DIR / "handwritten.csv"}
    for name, path in sets.items():
        if name == "val":
            y, pt, ps = y_val, pt_val, ps_val
        elif path.exists():
            y, pt, ps = both(pd.read_csv(path))
        else:
            continue
        report[name] = {"text_model_at_0.5": counts(y, pt, 0.5), "stacker_at_0.5": counts(y, ps, 0.5),
                        "stacker_at_val_threshold": counts(y, ps, threshold)}
        r = report[name]
        print(f"{name}: text@0.5 missed {r['text_model_at_0.5']['missed_scams']}/{r['text_model_at_0.5']['scams']} "
              f"FA {r['text_model_at_0.5']['false_alarms']}/{r['text_model_at_0.5']['legit']} | "
              f"stack@0.5 missed {r['stacker_at_0.5']['missed_scams']} FA {r['stacker_at_0.5']['false_alarms']} | "
              f"stack@{threshold:.2f} missed {r['stacker_at_val_threshold']['missed_scams']} "
              f"FA {r['stacker_at_val_threshold']['false_alarms']}")

    joblib.dump({"model": meta, "features": stacker.FEATURES, "threshold": threshold}, stacker.STACKER_PATH)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "stacker.json").write_text(json.dumps(report, indent=1))
    print(f"saved {stacker.STACKER_PATH} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
