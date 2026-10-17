import pytest

from safepassage import config


@pytest.fixture(autouse=True)
def isolated_logs(tmp_path, monkeypatch):
    """Point every log file at a temp dir so tests never touch logs/."""
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)
    for name in ("PREDICTION_LOG", "REVIEW_QUEUE", "ALERT_LOG"):
        monkeypatch.setattr(config, name, tmp_path / getattr(config, name).name)
    return tmp_path
