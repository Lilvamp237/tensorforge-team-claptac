from safepassage import community, reputation
from safepassage.community import MAX_SHIFT, rater_hash, signal, standing, verify_rater
from safepassage.schema import Claim, Verification

SITE = "https://jobs-offer.example/ad/1"


def rate(n, verdict="scam", rater="r1", url=SITE):
    for _ in range(n):
        reputation.add_report(url, verdict, rater_id=rater)


def test_no_reports_means_no_signal():
    assert signal(SITE) == (0.0, None)


def test_shift_is_capped_and_grows_with_evidence():
    rate(1)
    small, _ = signal(SITE)
    rate(8, rater="r2")
    big, note = signal(SITE)
    assert 0 < small < big <= MAX_SHIFT and "trust-weighted" in note


def test_verified_raters_count_more():
    rate(2, rater="anon")
    unverified, _ = signal(SITE)
    verify_rater("ngo-1")
    rate(2, rater="ngo-1", url="https://other.example")
    verified_site, _ = signal("https://other.example")
    assert verified_site > unverified


def test_raters_who_matched_analysts_gain_weight():
    reports = [{"rater": "a", "domain": "x.example", "verdict": "scam"}]
    good = community.rater_weight("a", reports, {"x.example": "scam"})
    bad = community.rater_weight("a", reports, {"x.example": "legit"})
    assert good > community.UNVERIFIED_WEIGHT > bad


def test_burst_of_reports_is_frozen():
    rate(community.BURST_REPORTS, rater="bot")
    shift, note = signal(SITE)
    assert shift == 0.0 and "frozen" in note


def test_rater_ids_are_never_stored_raw():
    rate(1, rater="person@example.com")
    stored = reputation.reports()[0]["rater"]
    assert stored == rater_hash("person@example.com") and "@" not in stored


def test_standing_uses_official_facts_and_analyst_decisions_only():
    cancelled = Verification(agency_status="not_matched", claims=[
        Claim(claim="Agency 'X' is licensed", status="contradicted", source="Registry lists 'X' as cancelled")])
    assert standing(None, cancelled) == ["SLBFE records: licence cancelled or suspended"]
    rate(3)  # reports alone only mark the site as under review, never as a scam
    assert "Reported, under review" in standing(SITE, Verification())


def test_blank_rater_id_is_anonymous():
    reputation.add_report(SITE, "scam", rater_id=None)
    assert reputation.reports()[0]["rater"] is None
