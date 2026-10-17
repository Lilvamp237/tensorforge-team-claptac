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


def test_only_analyst_labels_reach_feedback(isolated_logs):
    from safepassage import config
    from safepassage.response import export_feedback

    queued = [client.post("/api/v1/scans/text", json={"text": f"{MEDIUM} #{i}"}).json() for i in range(3)]
    client.post(f"/api/v1/reviews/{queued[0]['scan_id']}", json={"decision": "scam"})
    client.post(f"/api/v1/reviews/{queued[1]['scan_id']}", json={"decision": "legit"})
    client.post("/api/v1/reports", json={"url": "x.example", "verdict": "legit"})  # reports never become labels

    assert export_feedback() == 2
    lines = config.FEEDBACK_PATH.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "text,label" and {l.rsplit(",", 1)[1] for l in lines[1:]} == {"scam", "legit"}


def test_unknown_review_is_404():
    assert client.post("/api/v1/reviews/nope", json={"decision": "scam"}).status_code == 404
