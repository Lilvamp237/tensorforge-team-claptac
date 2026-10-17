"""Drift response: alert on drift onset/recovery, safe fallback while drifting, and retraining. Owner: Person B."""

import csv

from . import config
from .drift import drift_report
from .monitoring import raise_alert, read_alerts, reviewed

_DRIFT_KINDS = ("drift_detected", "drift_cleared")


def drift_active() -> bool:
    """Drift state lives in the alert log, so the API, Streamlit and scripts (separate processes) agree on it."""
    last = next((a["kind"] for a in reversed(read_alerts()) if a["kind"] in _DRIFT_KINDS), None)
    return last == "drift_detected"


CLEAR_MARGIN = 0.8  # once drifting, every signal must fall below 80% of its threshold before the alert clears


def check_drift() -> dict:
    """Recompute drift; alert only when the state changes so the log isn't flooded.
    Hysteresis: a window that dips just under a threshold doesn't clear an active alert, so it can't flap."""
    report = drift_report()
    now = report["n"] >= config.DRIFT_MIN_SCANS and report["drift"]
    if not now and drift_active():
        sig, thr = report.get("signals", {}), report.get("thresholds", {})
        now = any(sig.get(k, 0.0) >= CLEAR_MARGIN * t for k, t in thr.items())
    if now != drift_active():
        raise_alert("drift_detected" if now else "drift_cleared", report)
    return {**report, "drift": now}


def confirmed_labels() -> list[tuple[str, str]]:
    """(text, label) for scans an analyst marked scam or legit. Raw user reports are never included."""
    return [(r["text"], r["decision"]) for r in reviewed() if r.get("decision") in ("scam", "legit") and r.get("text")]


def export_feedback() -> int:
    """Write analyst-confirmed scan labels to FEEDBACK_PATH (text,label). Raw user reports are never used,
    so nobody can poison the training data just by submitting reports."""
    rows = confirmed_labels()
    config.FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.FEEDBACK_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    return len(rows)


def trigger_retrain(reason: str = "manual") -> dict:
    """Start the full retrain loop (export, train, calibrate, evaluate, gate, promote, reload) in a background
    thread. The heavy steps run as subprocesses, so a failure cannot take the API down."""
    from .retrain import start_retrain_job

    return start_retrain_job(reason)
