from fastapi.testclient import TestClient

from safepassage.api import app
from safepassage.features import find_red_flags

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_red_flags():
    text = "100% visa guaranteed! Pay registration fee now, only 3 seats left. WhatsApp us."
    flags = find_red_flags(text)
    assert {"guaranteed", "upfront_fee", "urgency", "contact_offplatform"} <= set(flags)
