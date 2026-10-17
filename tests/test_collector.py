"""ikman collector: parsing, PII stripping, robots handling and dedupe. Synthetic fixtures only, no network."""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "ikman"

_spec = importlib.util.spec_from_file_location("collect_ikman", ROOT / "scripts" / "collect_ikman.py")
ci = importlib.util.module_from_spec(_spec)
sys.modules["collect_ikman"] = ci
_spec.loader.exec_module(ci)


def read(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


class FakeScanner:
    def __init__(self):
        self.calls = []

    def __call__(self, text, url):
        self.calls.append((text, url))
        return {"risk": "high" if "guaranteed" in text.lower() else "low", "needs_review": "fee" in text.lower()}


# ------------------------------------------------------------------------------------------- parsing


def test_parse_listing_from_initial_data():
    cards, pagination = ci.parse_listing(read("overseas-jobs.html"))
    assert [c["slug"] for c in cards] == [
        "hotel-cook-qatar-colombo-test",  # top ad first, duplicate in the regular list dropped
        "housemaid-kuwait-kurunegala-test",
        "factory-worker-romania-colombo-test",
    ]
    cook = cards[0]
    assert cook["id"] == "f1000000000000000000000a"
    assert cook["url"] == "https://ikman.lk/en/ad/hotel-cook-qatar-colombo-test"
    assert cook["agency"] == "Example Overseas Agency"
    assert cook["salary"] == "Rs 180,000 - 200,000"
    assert cook["auth_agent"] is True
    assert cards[1]["auth_agent"] is False and cards[1]["member"] is False
    assert pagination == {"total": 4, "pageSize": 3, "activePage": 1}


def test_parse_listing_html_fallback():
    cards, pagination = ci.parse_listing(read("jobs.html"))
    assert pagination == {}
    assert [c["slug"] for c in cards] == ["data-entry-work-from-home-test", "cashier-kandy-test"]
    assert cards[0]["title"] == "Data Entry - Work From Home"
    assert cards[0]["agency"] == "Easy Earn Online"
    assert cards[0]["salary"] == "Rs 5,000 per day"
    assert cards[0]["location"] == "Gampaha"
    assert cards[1]["member"] is True and cards[1]["auth_agent"] is False


def test_parse_ad_fields_and_ignores_contact_card():
    ad = ci.parse_ad(read("ad-hotel-cook-qatar-colombo-test.html"))
    assert ad["title"] == "Hotel Cook - Qatar"
    assert ad["agency"] == "Example Overseas Agency"
    assert ad["properties"]["company_name"] == "Example Hotels Group"
    assert ad["salary"] == "Rs 180,000 - 200,000"
    assert ad["location"] == "Colombo 10"
    assert ad["auth_agent"] is True
    assert ad["poster_type"].startswith("Authorised agent")
    assert ad["properties"]["slbfe_license_number"] == "0000"
    assert "Doha" in ad["description"]
    dumped = json.dumps(ad)
    assert "0700000099" not in dumped and "shop-owner@example.com" not in dumped
    assert "Fake Contact Person" not in dumped


def test_parse_ad_meta_fallback_uses_card():
    card = {"id": "slug:cashier-kandy-test", "slug": "cashier-kandy-test", "agency": "Sample Supermarket",
            "member": True, "auth_agent": False, "salary": "Rs 45,000", "location": "Kandy"}
    ad = ci.parse_ad(read("ad-cashier-kandy-test.html"), card)
    assert ad["title"] == "Cashier - Kandy"
    assert "supermarket in Kandy" in ad["description"]
    assert ad["poster_type"] == "Member (MEMBER badge)"


def test_build_scan_text_includes_signals_and_is_scrubbed():
    text = ci.build_scan_text(ci.parse_ad(read("ad-housemaid-kuwait-kurunegala-test.html")))
    assert text.startswith("Housemaid - Kuwait")
    assert "Poster type: Non-member" in text
    assert "Country: Kuwait" in text and "Salary: Rs 150,000" in text
    assert "100% visa guaranteed" in text
    assert "000000000V" not in text and "0710000000" not in text and "+94 70" not in text
    assert "<NIC>" in text and "<PHONE>" in text and "<NAME>" in text
    assert "Test Person" not in text


# ------------------------------------------------------------------------------------------- privacy


@pytest.mark.parametrize("raw", [
    "077 000 0000",
    "0770000000",
    "077-000-0000",
    "+94 77 000 0000",
    "+94770000000",
    "0094 77 000 0000",
    "+94 (0) 77 000 0000",
    "011-2000000",
    "011 2 000 000",
    "(011) 200 0000",
    "081-2000000",
    "wa.me/94770000000",
    "+971 50 000 0000",
])
def test_scrub_phone_formats(raw):
    out = ci.scrub_pii(f"Call {raw} today")
    assert "<PHONE>" in out
    assert not any(ch.isdigit() for ch in out), out


def test_scrub_email_nic_and_names():
    out = ci.scrub_pii(
        "Email hr.team@example.com. NIC 000000000V or 200000000000. contact Nimal on 0770000000. "
        "Contact: Kamal Perera. Ask for Mrs. Silva. WhatsApp Sunil now."
    )
    assert "example.com" not in out and "<EMAIL>" in out
    assert out.count("<NIC>") == 2
    for name in ("Nimal", "Kamal", "Perera", "Silva", "Sunil"):
        assert name not in out
    assert "contact <NAME> on <PHONE>" in out


def test_scrub_keeps_offer_content():
    text = ("Contact Us today. Salary Rs 313,000 - 350,000, deadline 2026-11-24, SLBFE licence 1228. "
            "Call now for interviews in Colombo.")
    assert ci.scrub_pii(text) == text


# ------------------------------------------------------------------------------------------- robots


def test_robots_wildcards_from_fixture():
    robots = ci.RobotsPolicy(read("robots.txt"))
    assert robots.allowed("https://ikman.lk/en/ads/sri-lanka/overseas-jobs")
    assert robots.allowed("https://ikman.lk/en/ads/sri-lanka/overseas-jobs?page=2")
    assert not robots.allowed("https://ikman.lk/en/ads/sri-lanka/overseas-jobs?page=2&sort=date")
    assert not robots.allowed("https://ikman.lk/en/ads/sri-lanka/jobs?query=driver")
    assert not robots.allowed("https://ikman.lk/en/ad/some--thing")
    assert not robots.allowed("https://ikman.lk/en/ad/x/report")
    assert not robots.allowed("https://ikman.lk/en/ad/blocked-ad-test")
    assert robots.allowed("https://ikman.lk/en/ad/hotel-cook-qatar-colombo-test")


def _policy(disallowed):
    parser = MagicMock()
    parser.can_fetch.side_effect = lambda ua, url: not any(d in url for d in disallowed)
    return ci.RobotsPolicy("", parser=parser)


def test_robotparser_disallow_blocks_pagination_and_ads(tmp_path):
    source = ci.FixtureSource(FIXTURES)
    source.load_robots = lambda: _policy(["?page=", "/en/ad/housemaid"])
    scan = FakeScanner()
    stats = ci.run_once(source, scan, ci.SeenStore(tmp_path / "seen.json"), ["overseas-jobs"], log=lambda *_: None)
    urls = [u for _, u in scan.calls]
    assert not any("housemaid" in u for u in urls)
    assert not any("driver-saudi" in u for u in urls)  # only on page 2, which robots disallows
    assert stats.scanned == 2
    assert stats.robots_skipped == 2  # the housemaid ad and the page-2 listing


def test_robotparser_disallow_category_fetches_nothing(tmp_path):
    source = ci.FixtureSource(FIXTURES)
    source.load_robots = lambda: _policy(["/en/ads/"])
    stats = ci.run_once(source, FakeScanner(), ci.SeenStore(tmp_path / "s.json"), ["overseas-jobs"],
                        log=lambda *_: None)
    assert source.fetched == 0 and stats.scanned == 0 and stats.robots_skipped == 1


# ------------------------------------------------------------------------------------------- runs + dedupe


def test_run_paginates_scans_and_stores_only_clean_fields(tmp_path):
    seen_path = tmp_path / "collector_seen.json"
    scan = FakeScanner()
    stats = ci.run_once(ci.FixtureSource(FIXTURES), scan, ci.SeenStore(seen_path),
                        ["overseas-jobs", "jobs"], log=lambda *_: None)
    assert stats.scanned == 6 and stats.new == 6
    assert stats.risk["high"] == 1 and stats.risk["low"] == 5
    assert stats.review == 2  # housemaid + data-entry mention a fee
    stored = json.loads(seen_path.read_text(encoding="utf-8"))["ads"]
    assert len(stored) == 6
    for entry in stored.values():
        assert set(entry) == {"url", "text", "first_seen", "last_seen", "scanned_at"}
    blob = seen_path.read_text(encoding="utf-8")
    for secret in ("0770000000", "077 000 0000", "jobs@example.com", "000000000V", "200000000000",
                   "0700000099", "Nimal", "Sunil", "081-2000000", "075 000 0000"):
        assert secret not in blob
    assert not any("<html" in t or "<script" in t for t, _ in scan.calls)


def test_dedupe_skips_seen_ads(tmp_path):
    seen_path = tmp_path / "seen.json"
    first = ci.run_once(ci.FixtureSource(FIXTURES), FakeScanner(), ci.SeenStore(seen_path), ["overseas-jobs"],
                        log=lambda *_: None)
    assert first.scanned == 4
    source, scan = ci.FixtureSource(FIXTURES), FakeScanner()
    second = ci.run_once(source, scan, ci.SeenStore(seen_path), ["overseas-jobs"], log=lambda *_: None)
    assert second.scanned == 0 and second.new == 0 and second.already_seen == 4
    assert scan.calls == []
    assert source.fetched == 2  # listing pages only, no ad pages


def test_max_ads_limits_run(tmp_path):
    scan = FakeScanner()
    stats = ci.run_once(ci.FixtureSource(FIXTURES), scan, ci.SeenStore(tmp_path / "s.json"),
                        ["overseas-jobs", "jobs"], max_ads=2, log=lambda *_: None)
    assert stats.scanned == 2 and len(scan.calls) == 2


def test_in_process_scan_logs_to_prediction_log(isolated_logs, tmp_path):
    stats = ci.run_once(ci.FixtureSource(FIXTURES), ci.in_process_scanner(), ci.SeenStore(tmp_path / "s.json"),
                        ["jobs"], log=lambda *_: None)
    assert stats.scanned == 2
    lines = (isolated_logs / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and all(json.loads(line)["source"] == "collector" for line in lines)
    assert "scanned 2" in stats.summary()


# ------------------------------------------------------------------------------------------- live source


class _Resp:
    def __init__(self, status, text=""):
        self.status_code, self.text = status, text


def test_live_source_stops_on_429_and_identifies_itself(monkeypatch):
    monkeypatch.setattr(ci.time, "sleep", lambda s: None)
    session = MagicMock()
    session.headers = {}
    session.get.return_value = _Resp(429)
    src = ci.LiveSource(delay=0, session=session)
    assert src.delay == ci.MIN_DELAY
    assert session.headers["User-Agent"] == "SafePassageLK-collector/0.1 (hackathon research)"
    with pytest.raises(ci.StopCollection):
        src.listing("overseas-jobs", 1)
    assert session.get.call_count == 1  # no retry after 429


def test_live_source_retries_server_errors(monkeypatch):
    monkeypatch.setattr(ci.time, "sleep", lambda s: None)
    session = MagicMock()
    session.headers = {}
    session.get.side_effect = [_Resp(503), _Resp(502), _Resp(200, "ok")]
    assert ci.LiveSource(session=session).ad("x") == "ok"
    assert session.get.call_count == 3


def test_missing_robots_stops_live_run(monkeypatch, tmp_path):
    monkeypatch.setattr(ci.time, "sleep", lambda s: None)
    session = MagicMock()
    session.headers = {}
    session.get.return_value = _Resp(500)
    stats = ci.run_once(ci.LiveSource(session=session, retries=1), FakeScanner(), ci.SeenStore(tmp_path / "s.json"),
                        ["overseas-jobs"], log=lambda *_: None)
    assert "robots.txt" in stats.stopped and stats.scanned == 0


def test_cli_enforces_minimum_delay():
    args = ci.parse_args(["--delay", "0.2", "--once"])
    assert args.delay == ci.MIN_DELAY and args.category == ["overseas-jobs"]
    with pytest.raises(SystemExit):
        ci.parse_args(["--category", "../etc"])
