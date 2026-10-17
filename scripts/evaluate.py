"""Evaluate the trained models on the held-out test set and the shifted set -> reports/eval.json.

The shifted set is scored but never trained on: its numbers show how far quality drops under drift and
give the reference values used to calibrate the drift thresholds (mean confidence, scam rate).

Honesty checks (added keys; the original keys are unchanged)
  overall_ci              bootstrap 95% confidence intervals (accuracy, precision, recall, F1, ROC-AUC)
  base_rates              precision and false alarms if 1% / 5% / 10% of offers were scams (from TPR and FPR);
                          the test set itself is ~30% scam, which flatters precision
  by_phrase_novelty       synthetic test rows split by how many of their lines also occur in train
  val                     precision-recall curve on data/processed/val.csv
  recommended_thresholds  thresholds picked on VAL (recall >= 0.9, best F1) and their test metrics.
                          Informational only: config.py is not changed here.
  ablations               cross-domain (EMSCAD-only <-> synthetic-only) and shortcut masking
  handwritten_TEST_ONLY   data/handwritten.csv, if present. Never used for training or threshold choice.

Usage: python scripts/evaluate.py [--skip-ablations] [--n-boot 1000]
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

from safepassage.config import (DATA_DIR, HIGH_RISK, MEDIUM_RISK, MODEL_PATH, MODELS_DIR, MODEL_VERSION,
                                REPORTS_DIR)

CATEGORY_PATH = MODELS_DIR / "category.joblib"
HANDWRITTEN = DATA_DIR / "handwritten.csv"
BASE_RATES = (0.01, 0.05, 0.10)
CI_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


# ---------------------------------------------------------------- pure helpers (tested in tests/test_eval_utils.py)

def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    """Unrounded metrics for 0/1 labels `y` and scores `p`. roc_auc is None when only one class is present."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    pos, neg = int(y.sum()), int((1 - y).sum())
    return {
        "accuracy": float((pred == y).mean()) if len(y) else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / pos if pos else 0.0,
        "f1": 2 * tp / (2 * tp + fp + (pos - tp)) if pos + fp else 0.0,
        "fpr": fp / neg if neg else 0.0,
        "roc_auc": float(roc_auc_score(y, p)) if 0 < pos < len(y) else None,
    }


def precision_at_base_rate(tpr: float, fpr: float, prevalence: float) -> float:
    """Precision (PPV) the classifier would have if `prevalence` of offers were scams (Bayes' rule)."""
    hits, false_alarms = prevalence * tpr, (1 - prevalence) * fpr
    return hits / (hits + false_alarms) if hits + false_alarms else 0.0


def base_rate_table(tpr: float, fpr: float, rates=BASE_RATES) -> dict:
    """Per base rate: precision, false alarms and missed scams per 1,000 offers, false alarms per real scam caught."""
    out = {}
    for r in rates:
        caught, false_alarms = r * tpr, (1 - r) * fpr
        out[f"{r:.0%}"] = {
            "precision": round(precision_at_base_rate(tpr, fpr, r), 3),
            "false_alarms_per_1000_offers": round(1000 * false_alarms, 1),
            "scams_caught_per_1000_offers": round(1000 * caught, 1),
            "scams_missed_per_1000_offers": round(1000 * r * (1 - tpr), 1),
            "false_alarms_per_scam_caught": round(false_alarms / caught, 2) if caught else None,
        }
    return out


def bootstrap_ci(y, p, threshold: float = 0.5, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
                 rates=BASE_RATES) -> dict:
    """Percentile bootstrap (1 - alpha) intervals for accuracy, precision, recall, F1, ROC-AUC and the
    precision at each base rate in `rates`. Resamples rows with replacement."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    samples: dict[str, list] = {k: [] for k in CI_METRICS + tuple(f"precision_at_{r:.0%}" for r in rates)}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        m = binary_metrics(y[idx], p[idx], threshold)
        for k in CI_METRICS:
            if m[k] is not None:
                samples[k].append(m[k])
        for r in rates:
            samples[f"precision_at_{r:.0%}"].append(precision_at_base_rate(m["recall"], m["fpr"], r))
    q = [alpha / 2, 1 - alpha / 2]
    return {k: [round(float(x), 3) for x in np.quantile(v, q)] for k, v in samples.items() if v}


def recommend_thresholds(y, p, min_recall: float = 0.9) -> dict:
    """Pick thresholds on a validation set: (a) the highest threshold whose recall is still >= min_recall
    (the most precise operating point that meets the recall target), (b) the threshold with the best F1."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]  # the last point has no threshold
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    ok = np.flatnonzero(rec >= min_recall)
    a = ok[np.argmax(thr[ok])] if len(ok) else int(np.argmax(rec))
    b = int(np.argmax(f1))
    point = lambda i: {"threshold": round(float(thr[i]), 3), "precision": round(float(prec[i]), 3),  # noqa: E731
                       "recall": round(float(rec[i]), 3), "f1": round(float(f1[i]), 3)}
    return {f"recall_at_least_{min_recall:g}": point(a), "best_f1": point(b)}


def pr_curve_points(y, p, n: int = 40) -> list[dict]:
    """Down-sampled precision-recall curve (about n points) for the report."""
    prec, rec, thr = precision_recall_curve(np.asarray(y).astype(int), np.asarray(p, dtype=float))
    idx = np.unique(np.linspace(0, len(thr) - 1, min(n, len(thr))).astype(int))
    return [{"threshold": round(float(thr[i]), 3), "precision": round(float(prec[i]), 3),
             "recall": round(float(rec[i]), 3)} for i in idx]


# ---------------------------------------------------------------- report sections

def is_scam(labels: pd.Series) -> np.ndarray:
    return (labels == "scam").astype(int).to_numpy()


def scam_metrics(y_true: pd.Series, p_scam: np.ndarray, threshold: float = 0.5) -> dict:
    y = is_scam(y_true)
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


def honest_block(labels: pd.Series, p: np.ndarray, n_boot: int, threshold: float = 0.5) -> dict:
    """Metrics + bootstrap CIs + base-rate projection at one threshold."""
    y = is_scam(labels)
    m = binary_metrics(y, p, threshold)
    return {"threshold": threshold, **scam_metrics(labels, p, threshold), "fpr": round(m["fpr"], 3),
            "ci95": bootstrap_ci(y, p, threshold, n_boot=n_boot),
            "base_rates": base_rate_table(m["recall"], m["fpr"])}


def phrase_novelty(train: pd.DataFrame, df: pd.DataFrame, p: np.ndarray) -> dict:
    """Synthetic rows bucketed by the share of their lines that also occur in train (skeleton-normalised)."""
    from leakage_check import offer_lines
    seen = {x for t in train.loc[train["source"] == "synthetic", "text"] for x in offer_lines(t)}
    syn = df[df["source"] == "synthetic"]
    share = syn["text"].map(lambda t: np.mean([x in seen for x in offer_lines(t)] or [1.0]))
    buckets = pd.cut(share, [-0.01, 0.5, 0.99, 1.0], labels=["<=50% lines seen", "51-99% lines seen",
                                                                 "all lines seen in train"])
    return {str(b): scam_metrics(g["label"], p[g.index.to_numpy()]) for b, g in syn.groupby(buckets, observed=True)}


def fit_score(model, train: pd.DataFrame, tests: dict[str, pd.DataFrame], transform=None) -> dict:
    """Fit a fresh copy of `model` on `train` and score each test frame (optionally after `transform`)."""
    f = transform or (lambda s: s)
    m = clone(model).fit(f(train["text"]), train["label"])
    idx = list(m.classes_).index("scam")
    return {k: scam_metrics(d["label"], m.predict_proba(f(d["text"]))[:, idx]) for k, d in tests.items() if len(d)}


def ablations(model, scam_idx: int, train: pd.DataFrame, test: pd.DataFrame, p_test: np.ndarray) -> dict:
    from make_dataset import mask_shortcuts
    masked = lambda s: s.map(mask_shortcuts)  # noqa: E731
    em_tr, syn_tr = train[train["source"] == "emscad"], train[train["source"] == "synthetic"]
    em_te, syn_te = test[test["source"] == "emscad"], test[test["source"] == "synthetic"]
    print("ablation: cross-domain (2 fits)")
    cross = {
        "train_emscad_only": fit_score(model, em_tr, {"test_emscad": em_te, "test_synthetic": syn_te}),
        "train_synthetic_only": fit_score(model, syn_tr, {"test_synthetic": syn_te, "test_emscad": em_te}),
        "train_both (main model)": {"test_emscad": scam_metrics(em_te["label"], p_test[em_te.index.to_numpy()]),
                                    "test_synthetic": scam_metrics(syn_te["label"], p_test[syn_te.index.to_numpy()])},
    }
    print("ablation: masking (1 fit)")
    p_masked = model.predict_proba(masked(test["text"]))[:, scam_idx]
    mask = {
        "masked_tokens": "fictional agency names, licence numbers, phone numbers/placeholders, money amounts",
        "share_of_test_rows_changed": round(float((masked(test["text"]) != test["text"]).mean()), 3),
        "main_model_unmasked_test": {"overall": scam_metrics(test["label"], p_test),
                                     "by_source": breakdown(test, p_test, "source")},
        "main_model_masked_test": {"overall": scam_metrics(test["label"], p_masked),
                                   "by_source": breakdown(test, p_masked, "source")},
        "retrained_on_masked_train_masked_test": fit_score(model, train, {"overall": test, "synthetic": syn_te,
                                                                         "emscad": em_te}, transform=masked),
    }
    return {"cross_domain": cross, "masking": mask}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-ablations", action="store_true", help="skip the retraining ablations (faster)")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    scam_model = joblib.load(MODEL_PATH)
    category_model = joblib.load(CATEGORY_PATH)
    scam_idx = list(scam_model.classes_).index("scam")
    score = lambda texts: scam_model.predict_proba(texts)[:, scam_idx]  # noqa: E731
    processed = DATA_DIR / "processed"
    train = pd.read_csv(processed / "train.csv")

    report = {"model_version": MODEL_VERSION}
    preds = {}
    for name in ["test", "shifted"]:
        df = pd.read_csv(processed / f"{name}.csv").reset_index(drop=True)
        p = preds[name] = score(df["text"])
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
        section["overall_ci"] = bootstrap_ci(is_scam(df["label"]), p, n_boot=args.n_boot)
        m = binary_metrics(is_scam(df["label"]), p)
        section["base_rates"] = base_rate_table(m["recall"], m["fpr"])
        section["by_source_ci"] = {str(k): bootstrap_ci(is_scam(g["label"]), p[g.index.to_numpy()],
                                                        n_boot=args.n_boot) for k, g in df.groupby("source")}
        if name == "test":
            section["by_phrase_novelty"] = phrase_novelty(train, df, p)
            test_df = df
        report[name] = section

    # Threshold choice on validation data only.
    val_path = processed / "val.csv"
    if val_path.exists():
        val = pd.read_csv(val_path)
        p_val = score(val["text"])
        rec = recommend_thresholds(is_scam(val["label"]), p_val)
        report["val"] = {"n": len(val), "scam_share": round(float(is_scam(val["label"]).mean()), 3),
                         "roc_auc": round(roc_auc_score(is_scam(val["label"]), p_val), 3),
                         "pr_curve": pr_curve_points(is_scam(val["label"]), p_val)}
        report["recommended_thresholds"] = {
            "note": "Chosen on VAL, evaluated on TEST. Informational: config.py is unchanged; the team decides.",
            "current_config": {"MEDIUM_RISK": MEDIUM_RISK, "HIGH_RISK": HIGH_RISK},
            **{k: {"val": v, "test": honest_block(test_df["label"], preds["test"], args.n_boot, v["threshold"])}
               for k, v in rec.items()},
            "config_MEDIUM_RISK_on_test": honest_block(test_df["label"], preds["test"], args.n_boot, MEDIUM_RISK),
            "config_HIGH_RISK_on_test": honest_block(test_df["label"], preds["test"], args.n_boot, HIGH_RISK),
        }

    # Real hand-written offers: test only.
    if HANDWRITTEN.exists():
        hw = pd.read_csv(HANDWRITTEN).reset_index(drop=True)
        hw = hw[hw["label"].isin(["scam", "legit"])].reset_index(drop=True)
        p_hw = score(hw["text"])
        sec = {"note": "TEST-ONLY: never used for training or threshold selection.",
               "rows_also_in_train": int(hw["text"].isin(train["text"]).sum()),
               "at_0.5": honest_block(hw["label"], p_hw, args.n_boot)}
        for k, v in report.get("recommended_thresholds", {}).items():
            if isinstance(v, dict) and "val" in v:
                sec[f"at_{k}"] = honest_block(hw["label"], p_hw, args.n_boot, v["val"]["threshold"])
        for col in ("category", "language"):
            if col in hw:
                sec[f"by_{col}"] = breakdown(hw, p_hw, col)
        report["handwritten_TEST_ONLY"] = sec

    if not args.skip_ablations:
        report["ablations"] = ablations(scam_model, scam_idx, train, test_df, preds["test"])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "eval.json").write_text(json.dumps(report, indent=2))
    for name in ["test", "shifted"]:
        o, ci = report[name]["overall"], report[name]["overall_ci"]
        print(f"{name:8s} n={o['n']:5d}  acc={o['accuracy']}  P={o['precision']}  R={o['recall']}  F1={o['f1']} "
              f"{ci['f1']}  AUC={o.get('roc_auc')} {ci.get('roc_auc')}  conf={o['mean_confidence']}  "
              f"category_acc={report[name].get('category_accuracy')}")
    for k, v in report.get("recommended_thresholds", {}).items():
        if isinstance(v, dict) and "val" in v:
            t = v["test"]
            print(f"threshold {k}: {v['val']['threshold']}  test P={t['precision']} R={t['recall']} F1={t['f1']}")
    if "handwritten_TEST_ONLY" in report:
        h = report["handwritten_TEST_ONLY"]["at_0.5"]
        print(f"handwritten (TEST-ONLY) n={h['n']}  P={h['precision']}  R={h['recall']}  F1={h['f1']} {h['ci95']['f1']}")
    print(f"Wrote {REPORTS_DIR / 'eval.json'}")


if __name__ == "__main__":
    main()
