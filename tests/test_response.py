import pytest

from safepassage import config, response
from safepassage.monitoring import read_alerts
from safepassage.service import scan_text

# Two red flags -> stub score 0.5 -> medium risk, i.e. low-confidence predictions.
UNCERTAIN = "Guaranteed visa, contact us on WhatsApp."


@pytest.fixture(autouse=True)
def small_window(monkeypatch):
    monkeypatch.setattr(config, "DRIFT_MIN_SCANS", 5)


def test_no_verdict_before_min_scans():
    for _ in range(4):
        scan_text(UNCERTAIN)
    assert not response.drift_active()
    assert read_alerts() == []


def test_low_confidence_stream_raises_one_alert():
    for _ in range(8):
        scan_text(UNCERTAIN)
    assert response.drift_active()
    assert [a["kind"] for a in read_alerts()] == ["drift_detected"]


def test_drifting_routes_high_risk_to_review():
    for _ in range(5):
        scan_text(UNCERTAIN)
    assert response.drift_active()
    high = scan_text("Guaranteed visa! Pay processing fee now, only 2 seats left, WhatsApp, no interview needed.")
    assert high.risk.label == "high"
    assert high.needs_review
