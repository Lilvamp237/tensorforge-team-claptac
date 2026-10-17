"""Prediction logging. Drift signals are computed over this log."""

import json
from datetime import datetime, timezone

from .config import PREDICTION_LOG


def log_prediction(record: dict, path=PREDICTION_LOG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_log(path=PREDICTION_LOG) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
