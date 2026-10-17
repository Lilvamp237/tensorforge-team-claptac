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


def test_drift_alert_does_not_flap_just_under_threshold(monkeypatch):
    from safepassage import response
    thresholds = {"language_psi": 1.0}
    reports = iter([
        {"n": 50, "drift": True, "signals": {"language_psi": 1.2}, "thresholds": thresholds, "triggered": ["language_psi"]},
        {"n": 50, "drift": False, "signals": {"language_psi": 0.95}, "thresholds": thresholds, "triggered": []},
        {"n": 50, "drift": False, "signals": {"language_psi": 0.8}, "thresholds": thresholds, "triggered": []},
        {"n": 50, "drift": False, "signals": {"language_psi": 0.5}, "thresholds": thresholds, "triggered": []},
    ])
    monkeypatch.setattr(response, "drift_report", lambda: next(reports))
    assert response.check_drift()["drift"] is True
    assert response.check_drift()["drift"] is True   # 0.95 is under the threshold but above 80% of it: still drifting
    assert response.check_drift()["drift"] is True   # exactly 80% is not below the clearing margin
    assert response.check_drift()["drift"] is False  # clearly back to normal
    kinds = [a["kind"] for a in response.read_alerts()]
    assert kinds == ["drift_detected", "drift_cleared"]
