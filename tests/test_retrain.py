"""Retrain loop: feedback merging, promotion gate, rollback and scheduler decisions. The subprocess pipeline steps
are replaced by fakes that write tiny placeholder model files, so these tests run in milliseconds."""

import importlib.util
import json

import pandas as pd
import pytest

from safepassage import config, retrain, scheduler
from safepassage.monitoring import _append, read_alerts
from safepassage.response import export_feedback

ROOT = config.ROOT


def load_train_baseline():
    spec = importlib.util.spec_from_file_location("train_baseline", ROOT / "scripts" / "train_baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(f1, recall, real_recall=None):
    out = {"heldout": {"n": 100, "f1": f1, "recall": recall, "precision": f1}}
    if real_recall is not None:
        out["real"] = {"n": 10, "f1": real_recall, "recall": real_recall, "precision": 1.0}
    return out


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """Temp models/ with an initial production model, and fake train/calibrate/evaluate/reload steps."""
    models = tmp_path / "models"
    models.mkdir()
    for name in retrain.MODEL_FILES:
        (models / name).write_text("production-0")
    monkeypatch.setattr(config, "MODELS_DIR", models)
    state = {"candidate": metrics(0.90, 0.90), "production": metrics(0.90, 0.90), "reloads": 0, "n": 0}

    def fake_train(model_dir):
        state["n"] += 1
        for name in ("baseline.joblib", "category.joblib", "meta.json"):
            (model_dir / name).write_text(f"candidate-{state['n']}")

    def fake_calibrate(model_dir):
        for name in ("novelty.joblib", "drift_reference.json"):
            (model_dir / name).write_text(f"candidate-{state['n']}")

    def fake_evaluate(model_dir):
        return state["production"] if model_dir == models else state["candidate"]

    def fake_reload():
        state["reloads"] += 1
        return True

    monkeypatch.setattr(retrain, "train", fake_train)
    monkeypatch.setattr(retrain, "calibrate", fake_calibrate)
    monkeypatch.setattr(retrain, "evaluate", fake_evaluate)
    monkeypatch.setattr(retrain, "reload_in_process", fake_reload)
    # Distinct version names even when two runs start within the same second.
    names = iter(f"2026101{i}T000000Z" for i in range(10))
    monkeypatch.setattr(retrain, "new_version", lambda: next(names))
    return state


def confirm(n, label="scam", start=0):
    for i in range(start, start + n):
        _append(config.REVIEWED_LOG, {"review_id": f"scan_{i}", "text": f"offer number {i}", "decision": label})


# ---------------------------------------------------------------- feedback merging (train_baseline)
def test_merge_feedback_appends_dedupes_and_analyst_wins():
    tb = load_train_baseline()
    train = pd.DataFrame({"text": ["a", "b", "c"], "label": ["legit", "legit", "scam"], "category": "local_job",
                          "language": "en", "source": "synthetic", "tactic": "x"})
    feedback = pd.DataFrame({"text": ["b", "d", "d", "e"], "label": ["scam", "legit", "scam", "maybe"]})
    merged, info = tb.merge_feedback(train, feedback)
    assert info == {"feedback_rows": 2, "feedback_replaced_rows": 1}
    labels = dict(zip(merged["text"], merged["label"]))
    assert labels == {"a": "legit", "b": "scam", "c": "scam", "d": "scam"}  # analyst wins, latest decision wins
    assert merged["text"].is_unique
    assert set(merged.loc[merged["text"].isin(["b", "d"]), "source"]) == {"feedback"}


def test_export_feedback_uses_only_confirmed_labels():
    confirm(2, "scam")
    _append(config.REVIEWED_LOG, {"review_id": "domain:x.lk", "domain": "x.lk", "decision": "block"})
    _append(config.REVIEWED_LOG, {"review_id": "scan_d", "text": "dismissed", "decision": "dismiss"})
    assert export_feedback() == 2
    assert len(pd.read_csv(config.FEEDBACK_PATH)) == 2


# ---------------------------------------------------------------- gate
def test_gate_rules():
    prod = metrics(0.90, 0.90, real_recall=0.8)
    assert retrain.gate(metrics(0.885, 0.881, 0.8), prod)[0]  # within 0.02, real recall equal
    assert not retrain.gate(metrics(0.87, 0.95, 0.9), prod)[0]  # F1 dropped more than 0.02
    assert not retrain.gate(metrics(0.95, 0.87, 0.9), prod)[0]  # recall dropped more than 0.02
    assert not retrain.gate(metrics(0.95, 0.95, 0.79), prod)[0]  # real-set recall dropped
    assert retrain.gate(metrics(0.5, 0.5), None)[0]  # nothing in production yet


def test_promotes_good_candidate(pipeline):
    confirm(3)
    result = retrain.run_retrain("test")
    assert result["status"] == "promoted"
    assert (config.MODELS_DIR / "baseline.joblib").read_text() == "candidate-1"
    assert retrain.current_version() == result["version"]
    assert result["previous_version"].startswith("initial-")  # pre-versioning model kept for rollback
    assert pipeline["reloads"] == 1
    assert read_alerts()[-1]["kind"] == "retrain_promoted"
    assert retrain.read_history()[-1]["labels_total"] == 3


def test_rejects_worse_candidate(pipeline):
    pipeline["candidate"] = metrics(0.80, 0.95)
    result = retrain.run_retrain("test")
    assert result["status"] == "rejected"
    assert (config.MODELS_DIR / "baseline.joblib").read_text() == "production-0"  # production untouched
    assert (retrain.versions_dir() / result["version"] / "baseline.joblib").exists()  # candidate kept for audit
    assert pipeline["reloads"] == 0
    alert = read_alerts()[-1]
    assert alert["kind"] == "retrain_rejected" and alert["candidate"]["heldout"]["f1"] == 0.80


def test_failed_step_leaves_production_alone(pipeline, monkeypatch):
    monkeypatch.setattr(retrain, "calibrate", lambda d: (_ for _ in ()).throw(RuntimeError("boom")))
    result = retrain.run_retrain("test")
    assert result["status"] == "failed" and "boom" in result["error"]
    assert (config.MODELS_DIR / "baseline.joblib").read_text() == "production-0"
    assert read_alerts()[-1]["kind"] == "retrain_failed"
    assert not (retrain.versions_dir() / ".retrain.lock").exists()


def test_rollback_restores_previous_version(pipeline):
    first = retrain.run_retrain("test")
    second = retrain.run_retrain("test")
    assert (config.MODELS_DIR / "baseline.joblib").read_text() == "candidate-2"
    assert retrain.status()["previous"] == first["version"]

    out = retrain.rollback()
    assert out == {"rolled_back": True, "from_version": second["version"], "to_version": first["version"]}
    assert (config.MODELS_DIR / "baseline.joblib").read_text() == "candidate-1"
    assert retrain.current_version() == first["version"]

    retrain.rollback()  # back to the original, pre-versioning model
    assert (config.MODELS_DIR / "baseline.joblib").read_text() == "production-0"
    assert retrain.rollback()["rolled_back"] is False
    assert read_alerts()[-1]["kind"] == "retrain_rollback"


def test_status_lists_versions_and_last_result(pipeline):
    pipeline["candidate"] = metrics(0.5, 0.5)
    retrain.run_retrain("test")
    s = retrain.status()
    assert s["last_retrain"]["status"] == "rejected"
    assert len(s["versions"]) == 1 and s["current"] is None


# ---------------------------------------------------------------- scheduler
@pytest.mark.parametrize("new, drift, force, expected", [
    (20, False, False, True),   # enough new labels
    (19, False, False, False),  # not enough
    (5, True, False, True),     # drift + a few labels
    (4, True, False, False),    # drift but too few labels
    (0, False, True, True),     # forced
])
def test_scheduler_decision(new, drift, force, expected):
    go, reason = scheduler.decide(new, drift, min_new_labels=20, force=force)
    assert go is expected and reason


def test_scheduler_counts_labels_since_last_run(pipeline, monkeypatch):
    monkeypatch.setattr(scheduler, "check_drift", lambda: {"drift": False})
    confirm(3)
    assert scheduler.tick(min_new_labels=3)["status"] == "promoted"
    skipped = scheduler.tick(min_new_labels=3)
    assert skipped["status"] == "skip" and skipped["new_labels"] == 0
    confirm(3, start=3)
    assert scheduler.tick(min_new_labels=3)["status"] == "promoted"
    assert [h["status"] for h in retrain.read_history()] == ["promoted", "promoted"]


def test_scheduler_force_once(pipeline, monkeypatch, capsys):
    monkeypatch.setattr(scheduler, "check_drift", lambda: {"drift": False})
    scheduler.main(["--once", "--force"])
    assert json.loads(capsys.readouterr().out.split(" ", 1)[1])["status"] == "promoted"


# ---------------------------------------------------------------- API
def test_api_retrain_is_async_and_models_lists_state(pipeline):
    import time

    from fastapi.testclient import TestClient

    from safepassage.api import app

    client = TestClient(app)
    resp = client.post("/api/v1/retrain")
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    for _ in range(200):
        job = client.get(f"/api/v1/retrain/{job_id}").json()
        if job["status"] == "done":
            break
        time.sleep(0.01)
    assert job["result"]["status"] == "promoted"
    models = client.get("/api/v1/models").json()
    assert models["current"] == job["result"]["version"] and models["last_retrain"]["status"] == "promoted"
    assert client.post("/api/v1/models/rollback").json()["rolled_back"]
    assert client.post("/api/v1/models/rollback").status_code == 409
