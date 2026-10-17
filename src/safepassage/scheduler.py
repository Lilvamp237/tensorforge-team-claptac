"""Periodic retraining. Each tick retrains when any of these holds, otherwise it logs "skip" with the reason:
  - at least --min-new-labels analyst-confirmed labels since the last training run
  - drift is active and at least DRIFT_MIN_NEW_LABELS new labels exist
  - --force is given

Run: python -m safepassage.scheduler [--interval-min 60] [--min-new-labels 20] [--once] [--force]
Every retrain run is recorded in logs/retrain_history.jsonl (see safepassage.retrain).
"""

import argparse
import json
import time
from datetime import datetime, timezone

from . import config
from .response import check_drift, confirmed_labels
from .retrain import read_history, run_retrain

DRIFT_MIN_NEW_LABELS = 5


def labels_at_last_training() -> int:
    """Confirmed-label count when the model was last trained: the latest completed run in the retrain history,
    else the feedback rows recorded in models/meta.json, else 0."""
    runs = [h for h in read_history() if h.get("status") in ("promoted", "rejected")]
    if runs:
        return int(runs[-1].get("labels_total", 0))
    meta = config.MODELS_DIR / "meta.json"
    try:
        return int(json.loads(meta.read_text(encoding="utf-8")).get("feedback_rows", 0))
    except (OSError, ValueError, AttributeError):
        return 0


def decide(new_labels: int, drift: bool, min_new_labels: int, force: bool = False) -> tuple[bool, str]:
    if force:
        return True, "forced"
    if new_labels >= min_new_labels:
        return True, f"{new_labels} new confirmed labels (>= {min_new_labels})"
    if drift and new_labels >= DRIFT_MIN_NEW_LABELS:
        return True, f"drift active and {new_labels} new confirmed labels (>= {DRIFT_MIN_NEW_LABELS})"
    why = f"{new_labels} new confirmed labels (< {min_new_labels})"
    if drift:
        why += f"; drift active but fewer than {DRIFT_MIN_NEW_LABELS} new labels"
    return False, why


def tick(min_new_labels: int = 20, force: bool = False) -> dict:
    new_labels = max(0, len(confirmed_labels()) - labels_at_last_training())
    drift = check_drift()["drift"]
    go, why = decide(new_labels, drift, min_new_labels, force)
    if not go:
        return {"status": "skip", "reason": why, "new_labels": new_labels, "drift": drift}
    return {**run_retrain(f"scheduler: {why}"), "new_labels": new_labels, "drift": drift}


def _log(result: dict) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if result["status"] == "skip":
        print(f"{now} skip: {result['reason']}", flush=True)
        return
    keys = ("status", "version", "reason", "gate", "current_version", "error")
    print(f"{now} {json.dumps({k: result[k] for k in keys if k in result})}", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval-min", type=float, default=60)
    ap.add_argument("--min-new-labels", type=int, default=20)
    ap.add_argument("--once", action="store_true", help="run one tick and exit")
    ap.add_argument("--force", action="store_true", help="retrain regardless of labels and drift")
    args = ap.parse_args(argv)
    while True:
        _log(tick(args.min_new_labels, args.force))
        if args.once:
            return
        time.sleep(args.interval_min * 60)


if __name__ == "__main__":
    main()
