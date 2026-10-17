"""Drift signals over the prediction log. Owner: Person A (metrics), Person B (alerting + UI).

Signals over the latest `window` predictions, compared with the reference built by scripts/calibrate_drift.py:
  language_psi     PSI of the language mix (en / singlish / si / ta / mixed / unknown)
  category_psi     PSI of the content-type mix
  confidence_drop  reference mean confidence minus current mean confidence, where confidence = max(p, 1 - p)
  novelty_share    share of inputs flagged unfamiliar_pattern
Thresholds are calibrated on bootstrap windows: above normal held-out traffic, below the shifted set.
"""

import json
from functools import lru_cache

import numpy as np

from .config import MODELS_DIR
from .monitoring import read_log

REFERENCE_PATH = MODELS_DIR / "drift_reference.json"
LANGUAGES = ["en", "singlish", "si", "ta", "mixed", "unknown"]
CATEGORIES = ["local_job", "foreign_job", "student_visa", "visa_pr", "unknown"]
# Used only when no calibrated reference exists yet.
DEFAULT_THRESHOLDS = {"language_psi": 0.25, "category_psi": 0.25, "confidence_drop": 0.08, "novelty_share": 0.2}
EPS = 1e-4


def distribution(values: list[str], levels: list[str]) -> dict[str, float]:
    n = max(len(values), 1)
    return {lv: sum(v == lv for v in values) / n for lv in levels}


def psi(expected: dict[str, float], actual: dict[str, float]) -> float:
    """Population stability index; > 0.25 is the usual rule of thumb for a major shift."""
    total = 0.0
    for k in expected:
        e, a = max(expected[k], EPS), max(actual.get(k, 0.0), EPS)
        total += (a - e) * np.log(a / e)
    return float(total)


def _fields(rows: list[dict]):
    probs = np.array([r["risk"]["probability"] for r in rows], dtype=float)
    langs = [r["content"]["language"] for r in rows]
    cats = [r["content"]["type"] for r in rows]
    unfamiliar = np.array([r["novelty"]["status"] == "unfamiliar_pattern" for r in rows], dtype=float)
    return probs, langs, cats, unfamiliar


def reference_from_rows(rows: list[dict]) -> dict:
    probs, langs, cats, unfamiliar = _fields(rows)
    return {
        "n": len(rows),
        "language": distribution(langs, LANGUAGES),
        "category": distribution(cats, CATEGORIES),
        "mean_confidence": float(np.maximum(probs, 1 - probs).mean()),
        "novelty_share": float(unfamiliar.mean()),
    }


def compute_signals(rows: list[dict], reference: dict) -> dict[str, float]:
    probs, langs, cats, unfamiliar = _fields(rows)
    mean_conf = float(np.maximum(probs, 1 - probs).mean())
    return {
        "language_psi": round(psi(reference["language"], distribution(langs, LANGUAGES)), 4),
        "category_psi": round(psi(reference["category"], distribution(cats, CATEGORIES)), 4),
        "mean_confidence": round(mean_conf, 4),
        "confidence_drop": round(reference["mean_confidence"] - mean_conf, 4),
        "novelty_share": round(float(unfamiliar.mean()), 4),
    }


def triggered(signals: dict[str, float], thresholds: dict[str, float]) -> list[str]:
    return [name for name, thr in thresholds.items() if signals.get(name, 0.0) > thr]


@lru_cache(maxsize=1)
def load_reference() -> dict | None:
    return json.loads(REFERENCE_PATH.read_text()) if REFERENCE_PATH.exists() else None


def drift_report(window: int = 200, rows: list[dict] | None = None) -> dict:
    """Compare the latest `window` predictions with the training reference."""
    recent = (read_log() if rows is None else rows)[-window:]
    if not recent:
        return {"n": 0, "drift": False, "signals": {}, "triggered": [], "thresholds": {}}
    reference = load_reference()
    if reference is None:  # not calibrated yet: compare against a neutral reference
        probs, langs, cats, _ = _fields(recent)
        reference = {"language": distribution(langs, LANGUAGES), "category": distribution(cats, CATEGORIES),
                     "mean_confidence": 0.9}
        thresholds = DEFAULT_THRESHOLDS
    else:
        thresholds = reference["thresholds"]
    signals = compute_signals(recent, reference)
    fired = triggered(signals, thresholds)
    return {"n": len(recent), "drift": bool(fired), "signals": signals, "triggered": fired, "thresholds": thresholds}
