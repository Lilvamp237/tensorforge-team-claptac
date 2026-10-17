"""Prediction logging, review queue and alerts. Owner: Person B."""

import json
from datetime import datetime, timezone

import requests

from . import config


def _append(path, record: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def _read(path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def log_prediction(record: dict) -> None:
    _append(config.PREDICTION_LOG, record)


def read_log() -> list[dict]:
    return _read(config.PREDICTION_LOG)


def enqueue_review(record: dict) -> None:
    """Queue an item for a human. `review_id` is the scan_id for scans, or `domain:<host>` for reported sites."""
    _append(config.REVIEW_QUEUE, {"review_id": record.get("scan_id"), **record})


def pending_reviews() -> list[dict]:
    done = {r["review_id"] for r in _read(config.REVIEWED_LOG)}
    pending = {}
    for item in _read(config.REVIEW_QUEUE):
        if item["review_id"] not in done:
            pending[item["review_id"]] = item  # latest entry wins for repeated items
    return list(pending.values())


def resolve_review(review_id: str, decision: str, reviewer: str = "analyst") -> dict:
    item = next((i for i in pending_reviews() if i["review_id"] == review_id), None)
    if item is None:
        raise KeyError(review_id)
    return _append(config.REVIEWED_LOG, {**item, "decision": decision, "reviewer": reviewer})


def reviewed() -> list[dict]:
    return _read(config.REVIEWED_LOG)


def raise_alert(kind: str, details: dict) -> None:
    record = _append(config.ALERT_LOG, {"kind": kind, **details})
    if config.ALERT_WEBHOOK_URL:
        try:
            requests.post(config.ALERT_WEBHOOK_URL, json=record, timeout=3)
        except requests.RequestException:
            pass  # the alert is already in the log; a dead webhook must not break scanning


def read_alerts() -> list[dict]:
    return _read(config.ALERT_LOG)
