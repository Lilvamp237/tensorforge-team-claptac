"""Evaluate the trained models on the held-out test set and the shifted set -> reports/eval.json.

The shifted set is scored but never trained on: its numbers show how far quality drops under drift and
give the reference values used to calibrate the drift thresholds (mean confidence, scam rate).
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from safepassage.config import DATA_DIR, MODEL_PATH, MODELS_DIR, MODEL_VERSION, REPORTS_DIR

CATEGORY_PATH = MODELS_DIR / "category.joblib"


def scam_metrics(y_true: pd.Series, p_scam: np.ndarray, threshold: float = 0.5) -> dict:
    y = (y_true == "scam").astype(int).to_numpy()
    pred = (p_scam >= threshold).astype(int)
    out = {
        "n": int(len(y)),
        "scam_share": round(float(y.mean()), 3),
        "accuracy": round(accuracy_score(y, pred), 3),
        "precision": round(precision_score(y, pred, zero_division=0), 3),
        "recall": round(recall_score(y, pred, zero_division=0), 3),
        "f1": round(f1_score(y, pred, zero_division=0), 3),
        "mean_confidence": round(float(np.maximum(p_scam, 1 - p_scam).mean()), 3),
        "predicted_scam_rate": round(float(pred.mean()), 3),
    }
    if len(set(y)) == 2:
        out["roc_auc"] = round(roc_auc_score(y, p_scam), 3)
    return out


def breakdown(df: pd.DataFrame, p: np.ndarray, by: str) -> dict:
    return {str(k): scam_metrics(g["label"], p[g.index.to_numpy()]) for k, g in df.groupby(by)}


def main():
    scam_model = joblib.load(MODEL_PATH)
    category_model = joblib.load(CATEGORY_PATH)
    scam_idx = list(scam_model.classes_).index("scam")

    report = {"model_version": MODEL_VERSION}
    for name in ["test", "shifted"]:
        df = pd.read_csv(DATA_DIR / "processed" / f"{name}.csv").reset_index(drop=True)
        p = scam_model.predict_proba(df["text"])[:, scam_idx]
        section = {
            "overall": scam_metrics(df["label"], p),
            "by_source": breakdown(df, p, "source"),
            "by_language": breakdown(df, p, "language"),
        }
        if name == "shifted":
            section["by_tactic"] = breakdown(df, p, "tactic")
        sl = df[df["source"].str.startswith("synthetic")]
        if len(sl):
            section["category_accuracy"] = round(accuracy_score(sl["category"], category_model.predict(sl["text"])), 3)
        report[name] = section

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "eval.json").write_text(json.dumps(report, indent=2))
    for name in ["test", "shifted"]:
        o = report[name]["overall"]
        print(f"{name:8s} n={o['n']:5d}  acc={o['accuracy']}  P={o['precision']}  R={o['recall']}  F1={o['f1']}  "
              f"AUC={o.get('roc_auc')}  conf={o['mean_confidence']}  category_acc={report[name].get('category_accuracy')}")
    print(f"Wrote {REPORTS_DIR / 'eval.json'}")


if __name__ == "__main__":
    main()
