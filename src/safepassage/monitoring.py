"""Prediction logging, review queue and alerts. Owner: Person B."""

import json
from datetime import datetime, timezone

from .config import ALERT_LOG, PREDICTION_LOG, REVIEW_QUEUE


def _append(path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read(path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def log_prediction(record: dict) -> None:
    _append(PREDICTION_LOG, record)


def read_log() -> list[dict]:
    return _read(PREDICTION_LOG)


def enqueue_review(record: dict) -> None:
    _append(REVIEW_QUEUE, record)


def raise_alert(kind: str, details: dict) -> None:
    # TODO(B): also POST to a webhook if ALERT_WEBHOOK_URL is set
    _append(ALERT_LOG, {"kind": kind, **details})
