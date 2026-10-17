import pytest

from safepassage import config, drift, service
from safepassage.schema import Novelty


@pytest.fixture(autouse=True)
def isolated_logs(tmp_path, monkeypatch):
    """Point every log file at a temp dir so tests never touch logs/ (this also resets the drift state)."""
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(config, "ALERT_WEBHOOK_URL", None)
    monkeypatch.setattr(config, "FEEDBACK_PATH", tmp_path / "feedback.csv")
    for name in ("PREDICTION_LOG", "REVIEW_QUEUE", "ALERT_LOG", "REVIEWED_LOG", "REPORTS_LOG", "REPUTATION_PATH"):
        monkeypatch.setattr(config, name, tmp_path / getattr(config, name).name)
    return tmp_path


@pytest.fixture(autouse=True)
def deterministic_pipeline(monkeypatch):
    """Pipeline-logic tests use the red-flag fallback score, a neutral novelty score and the default drift
    thresholds, so they behave the same whether or not trained models exist in models/."""
    monkeypatch.setattr(service, "_model", None)
    monkeypatch.setattr(service, "novelty", lambda text: Novelty())
    monkeypatch.setattr(drift, "load_reference", lambda: None, raising=False)
