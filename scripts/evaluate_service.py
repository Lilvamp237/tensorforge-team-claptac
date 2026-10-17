"""Evaluate the full scoring path (text model + official-record trust adjustment) against the model alone.

    PYTHONPATH=src python scripts/evaluate_service.py

Runs on data/handwritten.csv (real-world, TEST ONLY), data/processed/test.csv and shifted.csv at threshold 0.5.
Nothing is logged to the live prediction log. Writes reports/eval_service.json.
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from safepassage import trust
from safepassage.config import DATA_DIR, REPORTS_DIR
from safepassage.extract import extract_content
from safepassage.features import find_red_flags
from safepassage.model import ScamModel
from safepassage.verify import verify

THRESHOLD = 0.5
SETS = {"real_world_TEST_ONLY": DATA_DIR / "handwritten.csv",
        "heldout_test": DATA_DIR / "processed" / "test.csv",
        "shifted": DATA_DIR / "processed" / "shifted.csv"}


def metrics(y, p, n_boot=0, seed=0):
    pred = (p >= THRESHOLD).astype(int)
    out = {"n": int(len(y)), "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
           "false_alarm_rate": float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)),
           "roc_auc": roc_auc_score(y, p) if len(set(y)) > 1 else None}
    out = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}
    if n_boot:
        rng, boots = np.random.default_rng(seed), {"f1": [], "false_alarm_rate": [], "recall": []}
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            m = metrics(y[i], p[i])
            for k in boots:
                boots[k].append(m[k])
        out["ci95"] = {k: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
                       for k, v in boots.items()}
    return out


def score(df: pd.DataFrame, model: ScamModel) -> tuple[np.ndarray, np.ndarray, dict]:
    raw, adjusted, effects = [], [], {"raised": 0, "lowered": 0, "unchanged": 0}
    for text in df["text"].astype(str):
        p = model.score(text)
        p_adj, _ = trust.adjust(p, verify(text, extract_content(text)), find_red_flags(text))
        raw.append(p)
        adjusted.append(p_adj)
        effects["raised" if p_adj > p else "lowered" if p_adj < p else "unchanged"] += 1
    return np.array(raw), np.array(adjusted), effects


def main():
    model, report = ScamModel(), {"threshold": THRESHOLD}
    for name, path in SETS.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        y = (df["label"] == "scam").astype(int).to_numpy()
        raw, adj, effects = score(df, model)
        boot = 1000 if name.startswith("real") else 0
        report[name] = {"model_only": metrics(y, raw, boot), "with_official_records": metrics(y, adj, boot),
                        "adjustments": effects}
        print(f"{name}: model F1 {report[name]['model_only']['f1']} FAR {report[name]['model_only']['false_alarm_rate']}"
              f" -> with records F1 {report[name]['with_official_records']['f1']}"
              f" FAR {report[name]['with_official_records']['false_alarm_rate']} | {effects}")
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "eval_service.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
