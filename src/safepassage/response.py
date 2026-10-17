"""Drift response: alert on drift onset/recovery, safe fallback while drifting, and retraining. Owner: Person B."""

import csv
import os
import subprocess
import sys

from . import config
from .drift import drift_report
from .monitoring import raise_alert, reviewed

_drifting = False


def drift_active() -> bool:
    return _drifting


def check_drift() -> dict:
    """Recompute drift; alert only when the state changes so the log isn't flooded."""
    global _drifting
    report = drift_report()
    now = report["n"] >= config.DRIFT_MIN_SCANS and report["drift"]
    if now != _drifting:
        raise_alert("drift_detected" if now else "drift_cleared", report)
        _drifting = now
    return {**report, "drift": now}


def export_feedback() -> int:
    """Write analyst-confirmed scan labels to FEEDBACK_PATH (text,label). Raw user reports are never used,
    so nobody can poison the training data just by submitting reports."""
    rows = [(r["text"], r["decision"]) for r in reviewed() if r.get("decision") in ("scam", "legit") and r.get("text")]
    config.FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.FEEDBACK_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    return len(rows)


def trigger_retrain() -> dict:
    """Export feedback and run the training script in the background; reload the model when it finishes."""
    n = export_feedback()
    log = (config.LOGS_DIR / "retrain.log").open("a", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(config.ROOT / "src"), "SAFEPASSAGE_FEEDBACK": str(config.FEEDBACK_PATH)}
    proc = subprocess.Popen([sys.executable, str(config.ROOT / "scripts" / "train_baseline.py")],
                            cwd=config.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    raise_alert("retrain_started", {"pid": proc.pid, "feedback_rows": n})
    return {"started": True, "pid": proc.pid, "feedback_rows": n}


def reset() -> None:
    global _drifting
    _drifting = False
