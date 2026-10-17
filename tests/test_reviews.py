from fastapi.testclient import TestClient

from safepassage.api import app

client = TestClient(app)
MEDIUM = "Guaranteed visa, contact us on WhatsApp."  # stub score 0.5 -> medium -> queued


def test_medium_risk_scan_is_queued_and_resolvable():
    scan = client.post("/api/v1/scans/text", json={"text": MEDIUM}).json()
    pending = client.get("/api/v1/reviews").json()
    assert [p["review_id"] for p in pending] == [scan["scan_id"]]
    assert pending[0]["text"] == MEDIUM

    resp = client.post(f"/api/v1/reviews/{scan['scan_id']}", json={"decision": "scam"})
    assert resp.status_code == 200
    assert client.get("/api/v1/reviews").json() == []


def test_unknown_review_is_404():
    assert client.post("/api/v1/reviews/nope", json={"decision": "scam"}).status_code == 404
