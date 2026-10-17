import pytest

from safepassage import config, response


@pytest.fixture(autouse=True)
def isolated_logs(tmp_path, monkeypatch):
    """Point every log file at a temp dir so tests never touch logs/, and start each test drift-free."""
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)
    for name in ("PREDICTION_LOG", "REVIEW_QUEUE", "ALERT_LOG"):
        monkeypatch.setattr(config, name, tmp_path / getattr(config, name).name)
    monkeypatch.setattr(config, "ALERT_WEBHOOK_URL", None)
    response.reset()
    yield tmp_path
    response.reset()
