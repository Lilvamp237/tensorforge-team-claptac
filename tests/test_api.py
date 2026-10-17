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
    assert {"guaranteed", "upfront_fee", "urgency", "contact_offplatform"} <= set(find_red_flags(text))


def test_local_job_red_flags():
    text = "Part-time job! Like YouTube videos and earn daily commission. Send your NIC copy on Telegram."
    assert {"task_scam", "id_document_request", "contact_offplatform"} <= set(find_red_flags(text))


def test_scan_text_returns_contract():
    resp = client.post("/api/v1/scans/text", json={"text": "Guaranteed visa! Pay processing fee now on WhatsApp."})
    assert resp.status_code == 200
    body = resp.json()
    assert body["risk"]["label"] in {"low", "medium", "high"}
    assert body["disclaimer"]


def test_url_scan_blocks_private_hosts():
    resp = client.post("/api/v1/scans/url", json={"url": "http://127.0.0.1/admin"})
    assert resp.status_code == 400
