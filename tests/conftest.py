import pytest

from safepassage import config


@pytest.fixture(autouse=True)
def isolated_logs(tmp_path, monkeypatch):
    """Point every log file at a temp dir so tests never touch logs/ (this also resets the drift state)."""
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(config, "ALERT_WEBHOOK_URL", None)
    monkeypatch.setattr(config, "FEEDBACK_PATH", tmp_path / "feedback.csv")
    for name in ("PREDICTION_LOG", "REVIEW_QUEUE", "ALERT_LOG", "REVIEWED_LOG", "REPORTS_LOG", "REPUTATION_PATH"):
        monkeypatch.setattr(config, name, tmp_path / getattr(config, name).name)
    return tmp_path
