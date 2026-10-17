"""Drift response: alert on drift onset/recovery and switch to a safe fallback while drifting. Owner: Person B."""

from . import config
from .drift import drift_report
from .monitoring import raise_alert

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


def reset() -> None:
    global _drifting
    _drifting = False
