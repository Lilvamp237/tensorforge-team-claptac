from fastapi.testclient import TestClient

from safepassage.api import app
from safepassage.reputation import domain_of

client = TestClient(app)
SITE = "https://www.fake-gulf-jobs.example/offer/12"
CLEAN = "Accountant vacancy at our Colombo office."


def report(verdict="scam", url=SITE):
    return client.post("/api/v1/reports", json={"url": url, "verdict": verdict}).json()


def test_domain_normalisation():
    assert domain_of(SITE) == domain_of("fake-gulf-jobs.example") == "fake-gulf-jobs.example"


def test_reports_escalate_to_analyst_but_never_auto_list():
    for _ in range(2):
        report()
    assert client.get("/api/v1/reviews").json() == []
    report()
    pending = client.get("/api/v1/reviews").json()
    assert [p["review_id"] for p in pending] == ["domain:fake-gulf-jobs.example"]
    assert client.get("/api/v1/reputation").json() == {"allow": {}, "block": {}}
    report()  # a 4th report must not queue the site twice
    assert len(client.get("/api/v1/reviews").json()) == 1


def test_analyst_block_raises_risk_for_that_site():
    for _ in range(3):
        report()
    client.post("/api/v1/reviews/domain:fake-gulf-jobs.example", json={"decision": "block"})
    assert "fake-gulf-jobs.example" in client.get("/api/v1/reputation").json()["block"]

    scan = client.post("/api/v1/scans/text", json={"text": CLEAN, "url": SITE}).json()
    assert scan["risk"]["label"] == "high"
    assert "blocklist" in scan["indicators"][0]


def test_allowlist_is_informational_only():
    for _ in range(3):
        report()
    client.post("/api/v1/reviews/domain:fake-gulf-jobs.example", json={"decision": "allow"})
    scan = client.post("/api/v1/scans/text", json={"text": "Guaranteed visa! Pay processing fee now on WhatsApp, only 2 seats left.", "url": SITE}).json()
    assert scan["risk"]["label"] == "high"
    assert "verified list" in scan["indicators"][0]
