"""Retraining loop: feedback -> candidate model -> drift recalibration -> evaluation -> promotion gate -> hot reload.

Every candidate lives in models/versions/<UTC timestamp>/. Only a candidate that passes the gate is copied into
models/ (the files the service loads), so a bad retrain never reaches production and `rollback()` can always
restore the previous version. Training, calibration and evaluation run as subprocesses: if one crashes, the API
process keeps serving the current model.

Run one evaluation by hand: python -m safepassage.retrain evaluate --model-dir models/versions/<v>
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .monitoring import _append, _read, raise_alert
from .response import confirmed_labels, export_feedback

# Files that make up one model version (as written by train_baseline.py and calibrate_drift.py).
MODEL_FILES = ("baseline.joblib", "category.joblib", "meta.json", "novelty.joblib", "drift_reference.json")
MAX_DROP = 0.02  # candidate F1 / recall on the held-out set may be at most this much below production
STEP_TIMEOUT_S = 1800
LOCK_STALE_S = 2 * 3600  # a lock file older than this is from a crashed run

_lock = threading.Lock()
_jobs: dict[str, dict] = {}


# ---------------------------------------------------------------- paths
def versions_dir() -> Path:
    return config.MODELS_DIR / "versions"


def history_path() -> Path:
    return config.LOGS_DIR / "retrain_history.jsonl"


def _promotions_path() -> Path:
    return versions_dir() / "promotions.json"


def current_version() -> str | None:
    path = config.MODELS_DIR / "CURRENT_VERSION"
    return (path.read_text(encoding="utf-8").strip() or None) if path.exists() else None


def _promotions() -> list[str]:
    path = _promotions_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_history() -> list[dict]:
    return _read(history_path())


# ---------------------------------------------------------------- pipeline steps (subprocesses)
def _env(model_dir: Path) -> dict:
    return {**os.environ, "PYTHONPATH": str(config.ROOT / "src"), "SAFEPASSAGE_MODEL_DIR": str(model_dir),
            "SAFEPASSAGE_FEEDBACK": str(config.FEEDBACK_PATH)}


def _run_step(args: list[str], model_dir: Path) -> str:
    """Run one pipeline step in its own process; raise with the tail of its output if it fails."""
    proc = subprocess.run([sys.executable, *args], cwd=config.ROOT, env=_env(model_dir), capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=STEP_TIMEOUT_S)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with (config.LOGS_DIR / "retrain.log").open("a", encoding="utf-8") as log:
        log.write(f"$ {' '.join(args)} (SAFEPASSAGE_MODEL_DIR={model_dir})\n{proc.stdout}{proc.stderr}\n")
    if proc.returncode != 0:
        step = args[1] if args[0] == "-m" else Path(args[0]).name
        raise RuntimeError(f"{step} failed: {(proc.stderr or proc.stdout).strip()[-500:]}")
    return proc.stdout


def train(model_dir: Path) -> None:
    _run_step([str(config.ROOT / "scripts" / "train_baseline.py")], model_dir)


def calibrate(model_dir: Path) -> None:
    _run_step([str(config.ROOT / "scripts" / "calibrate_drift.py")], model_dir)


def evaluate(model_dir: Path) -> dict:
    """Score a model dir on the held-out test set (and data/handwritten.csv if present) in a subprocess."""
    out = _run_step(["-m", "safepassage.retrain", "evaluate", "--model-dir", str(model_dir)], model_dir)
    return json.loads(out.strip().splitlines()[-1])


def evaluate_in_process(model_dir: Path) -> dict:
    """What `evaluate` runs inside the subprocess. Test-only sets: nothing here is ever trained on."""
    import joblib
    import pandas as pd
    from sklearn.metrics import f1_score, precision_score, recall_score

    model = joblib.load(Path(model_dir) / "baseline.joblib")
    scam_idx = list(model.classes_).index("scam")

    def metrics(df) -> dict:
        y = (df["label"] == "scam").astype(int)
        pred = (model.predict_proba(df["text"])[:, scam_idx] >= 0.5).astype(int)
        return {"n": int(len(df)), "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
                "precision": round(float(precision_score(y, pred, zero_division=0)), 4)}

    result = {"heldout": metrics(pd.read_csv(config.DATA_DIR / "processed" / "test.csv"))}
    real = config.DATA_DIR / "handwritten.csv"
    if real.exists():
        result["real"] = metrics(pd.read_csv(real).dropna(subset=["text", "label"]))
    return result


# ---------------------------------------------------------------- gate, promotion, rollback
def gate(candidate: dict, production: dict | None) -> tuple[bool, list[str]]:
    """Promote only if held-out F1 and recall are at most MAX_DROP below production, and recall on the real
    (handwritten) set does not drop. With no production model, any candidate that trained is promoted."""
    if not production:
        return True, ["no production model to compare against"]
    reasons, ok = [], True
    for metric in ("f1", "recall"):
        cand, prod = candidate["heldout"][metric], production["heldout"][metric]
        passed = cand >= prod - MAX_DROP
        ok &= passed
        reasons.append(f"heldout {metric} {cand:.4f} vs {prod:.4f} ({'ok' if passed else f'drop > {MAX_DROP}'})")
    if "real" in candidate and "real" in production:
        cand, prod = candidate["real"]["recall"], production["real"]["recall"]
        passed = cand >= prod
        ok &= passed
        reasons.append(f"real-set recall {cand:.4f} vs {prod:.4f} ({'ok' if passed else 'dropped'})")
    return ok, reasons


def _snapshot_production() -> str | None:
    """Models trained straight into models/ have no version dir yet; copy them into one so rollback can reach them."""
    if current_version() or not (config.MODELS_DIR / "baseline.joblib").exists():
        return current_version()
    version = "initial-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = versions_dir() / version
    target.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        if (config.MODELS_DIR / name).exists():
            shutil.copy2(config.MODELS_DIR / name, target / name)
    _write_atomic(config.MODELS_DIR / "CURRENT_VERSION", version)
    _write_atomic(_promotions_path(), json.dumps([version]))
    return version


def _install(version: str) -> None:
    """Copy a version's files into models/. Each file is staged next to its target and swapped in with os.replace
    (atomic per file), and CURRENT_VERSION is written last, so a reader never sees a half-written file."""
    source = versions_dir() / version
    missing = [n for n in ("baseline.joblib",) if not (source / n).exists()]
    if missing:
        raise FileNotFoundError(f"version {version} is missing {missing}")
    for name in MODEL_FILES:
        if (source / name).exists():
            tmp = config.MODELS_DIR / f".{name}.staging"
            shutil.copy2(source / name, tmp)
            os.replace(tmp, config.MODELS_DIR / name)
    _write_atomic(config.MODELS_DIR / "CURRENT_VERSION", version)


def promote(version: str) -> str | None:
    """Make `version` production. Returns the version it replaced (kept on disk for rollback)."""
    previous = _snapshot_production()
    _install(version)
    _write_atomic(_promotions_path(), json.dumps([*_promotions(), version]))
    return previous


def reload_in_process() -> bool:
    from . import service

    return service.reload_model()


def rollback() -> dict:
    """Restore the version that was in production before the current one."""
    with _lock:
        stack = _promotions()
        if len(stack) < 2:
            return {"rolled_back": False, "reason": "no previous version", "current": current_version()}
        bad, previous = stack[-1], stack[-2]
        _install(previous)
        _write_atomic(_promotions_path(), json.dumps(stack[:-1]))
        reload_in_process()
        summary = {"rolled_back": True, "from_version": bad, "to_version": previous}
        raise_alert("retrain_rollback", summary)
        _append(history_path(), {"status": "rolled_back", **summary})
        return summary


# ---------------------------------------------------------------- the loop
def new_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _acquire_file_lock() -> bool:
    """The API and the scheduler are separate processes; a lock file stops them retraining at the same time."""
    path = versions_dir() / ".retrain.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and datetime.now().timestamp() - path.stat().st_mtime > LOCK_STALE_S:
        path.unlink(missing_ok=True)
    try:
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def run_retrain(reason: str = "manual") -> dict:
    """Run the whole loop once. Never raises: failures come back as {"status": "failed"} and an alert."""
    if not _lock.acquire(blocking=False):
        return {"status": "busy", "reason": reason}
    if not _acquire_file_lock():
        _lock.release()
        return {"status": "busy", "reason": reason, "note": "another process is retraining"}
    version = new_version()
    summary = {"version": version, "reason": reason, "labels_total": len(confirmed_labels())}
    try:
        summary["feedback_rows"] = export_feedback()
        model_dir = versions_dir() / version
        model_dir.mkdir(parents=True, exist_ok=True)
        train(model_dir)
        calibrate(model_dir)
        candidate = evaluate(model_dir)
        (model_dir / "eval.json").write_text(json.dumps(candidate, indent=2), encoding="utf-8")
        production = evaluate(config.MODELS_DIR) if (config.MODELS_DIR / "baseline.joblib").exists() else None
        promoted, reasons = gate(candidate, production)
        summary.update(candidate=candidate, production=production, gate=reasons,
                       status="promoted" if promoted else "rejected")
        if promoted:
            summary["previous_version"] = promote(version)
            summary["model_loaded"] = reload_in_process()
        raise_alert(f"retrain_{summary['status']}", summary)
    except Exception as e:  # noqa: BLE001 - any failure must leave production untouched and be reported
        summary.update(status="failed", error=str(e)[-1000:])
        shutil.rmtree(versions_dir() / version, ignore_errors=True)
        raise_alert("retrain_failed", summary)
    finally:
        (versions_dir() / ".retrain.lock").unlink(missing_ok=True)
        _lock.release()
    summary["current_version"] = current_version()
    _append(history_path(), summary)
    return summary


def start_retrain_job(reason: str = "manual") -> dict:
    """Run `run_retrain` in a background thread; poll `job_status(job_id)` for the result."""
    running = next((j for j in _jobs.values() if j["status"] == "running"), None)
    if running:
        return {**running, "note": "a retrain is already running"}
    job_id = f"retrain_{uuid.uuid4().hex[:8]}"
    _jobs[job_id] = {"job_id": job_id, "status": "running", "reason": reason,
                     "started_at": datetime.now(timezone.utc).isoformat()}

    def work():
        _jobs[job_id].update(status="done", result=run_retrain(reason))

    threading.Thread(target=work, name=job_id, daemon=True).start()
    return dict(_jobs[job_id])


def job_status(job_id: str) -> dict | None:
    return _jobs.get(job_id)


def status() -> dict:
    """Versions on disk, the one in production, the last retrain result and the full history."""
    history = read_history()
    runs = [h for h in history if h.get("status") in ("promoted", "rejected", "failed")]
    versions = sorted(p.name for p in versions_dir().glob("*") if p.is_dir()) if versions_dir().exists() else []
    stack = _promotions()
    return {"current": current_version(), "previous": stack[-2] if len(stack) > 1 else None, "versions": versions,
            "last_retrain": runs[-1] if runs else None, "history": history,
            "jobs": list(_jobs.values())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("evaluate", help="print held-out (and real-set) metrics for a model dir as JSON")
    ev.add_argument("--model-dir", type=Path, default=config.MODELS_DIR)
    run = sub.add_parser("run", help="run the retrain loop once")
    run.add_argument("--reason", default="manual")
    sub.add_parser("rollback", help="restore the previous model version")
    args = ap.parse_args()
    if args.cmd == "evaluate":
        print(json.dumps(evaluate_in_process(args.model_dir)))
    elif args.cmd == "run":
        print(json.dumps(run_retrain(args.reason), indent=1))
    else:
        print(json.dumps(rollback(), indent=1))


if __name__ == "__main__":
    main()
