import pandas as pd
import pytest

from safepassage import verify as V
from safepassage.extract import extract_content


_LOAD_REGISTRY = V.load_registry.__wrapped__
JO = "https://applications.slbfe.lk/jobbank/jsearchdisplay_an_m.asp?an="
# Small fixture job orders (same columns as data/kb/slbfe_job_orders.csv); licences match the demo registry.
JOB_ORDERS = pd.DataFrame([
    ["1023", "Qatar", "Welder", 300000, "2026-01-10", "2099-01-09", JO + "1023"],
    ["1023", "Japan", "Specified Skilled Worker (SSW) -Nursing Care", 1119985, "2025-11-27", "2099-11-26", JO + "1023"],
    ["1023", "Japan", "Specified Skilled Worker (SSW) -Agriculture", 5000000, "2018-01-01", "2020-01-01", JO + "1023"],
    ["1187", "U A E", "Driver - Heavy Vehicle", 1500000, "2026-03-01", "2099-03-01", JO + "1187"],
], columns=["licence_no", "country", "job_title", "max_fee_lkr", "approved_date", "valid_until", "source"])


@pytest.fixture(autouse=True)
def demo_registry(monkeypatch):
    """Always test against the committed demo registry and the fixture job orders, never crawled or network data."""
    df = pd.read_csv(V.DEMO_REGISTRY, dtype=str).fillna("")
    df["key"] = df["name"].map(V.normalise)
    monkeypatch.setattr(V, "load_registry", lambda: df)
    monkeypatch.setattr(V, "load_job_orders", lambda: JOB_ORDERS.copy())


def run(text):
    v = V.verify(text, extract_content(text))
    return v.agency_status, {(c.status, c.claim.split(" ")[0]) for c in v.claims}, v


def test_licensed_agency_and_registration_fee():
    status, _, v = run("Lanka Bright Overseas (Pvt) Ltd, SLBFE licence no. 1023. Welders for Qatar. "
                       "Only the SLBFE registration fee of Rs. 22,027 is payable.")
    assert status == "matched"
    assert all(c.status == "consistent" for c in v.claims)


def test_borrowed_licence_number_is_contradicted():
    status, claims, _ = run("Ceylon Horizon Manpower licence 2051 - staff nurse jobs in Japan.")
    assert status == "not_matched" and ("contradicted", "Licence") in claims


def test_suspended_agency():
    status, claims, _ = run("Island Star Recruiters licence 0987: drivers for Dubai.")
    assert status == "not_matched" and ("contradicted", "Agency") in claims


def test_unknown_agency_is_unverifiable_not_accused():
    status, claims, _ = run("Accountant jobs in Qatar via Sunrise Overseas Agency.")
    assert status == "not_matched" and ("unverifiable", "Agency") in claims


def test_no_agency_claimed():
    assert run("Drivers needed in Qatar.")[0] == "unknown"


def test_domestic_middle_east_fee_and_personal_account():
    _, claims, v = run("Housemaid jobs in Kuwait. Pay the registration fee of Rs. 150,000 to the personal account below.")
    assert ("contradicted", "Fee") in claims
    assert sum(c.status == "contradicted" for c in v.claims) == 2


def test_fee_above_highest_cap_and_no_slbfe():
    _, _, v = run("Factory workers for Romania. Placement fee Rs. 2,500,000. No need to register with SLBFE.")
    assert [c.status for c in v.claims] == ["contradicted", "contradicted"]


def test_fee_under_cap_is_unverifiable():
    _, claims, _ = run("Drivers for Dubai, processing fee Rs. 80,000.")
    assert claims == {("unverifiable", "Fee")}


def fee_claims(v):
    return [c for c in v.claims if c.claim.startswith("Fee")]


def test_fee_above_job_order_maximum_is_contradicted():
    status, _, v = run("Lanka Bright Overseas (Pvt) Ltd, licence no. 1023. Welders for Qatar. Agency fee Rs. 350,000.")
    [fee] = fee_claims(v)
    assert status == "matched" and fee.status == "contradicted"
    assert "Rs. 300,000" in fee.source and JO + "1023" in fee.source


def test_fee_within_job_order_maximum_is_consistent():
    _, _, v = run("Lanka Bright Overseas (Pvt) Ltd, licence no. 1023. Nursing care workers for Japan. "
                  "Agency fee Rs. 1,000,000.")
    [fee] = fee_claims(v)
    assert fee.status == "consistent" and "Nursing Care" in fee.source


def test_job_order_country_alias_and_exact_cap():
    _, _, v = run("Ceylon Horizon Manpower licence 1187. Heavy vehicle drivers for Dubai. Service fee Rs. 1,500,000.")
    [fee] = fee_claims(v)
    assert fee.status == "consistent" and "Driver - Heavy Vehicle" in fee.source


def test_no_job_order_for_that_country_falls_back_to_global_cap():
    _, claims, _ = run("Lanka Bright Overseas (Pvt) Ltd, licence no. 1023. Welders for Oman. Agency fee Rs. 350,000.")
    assert ("unverifiable", "Fee") in claims


def test_global_cap_is_highest_currently_valid_job_order():
    value, source = V.highest_valid_cap(JOB_ORDERS)
    assert value == 1500000 and source == JO + "1187"  # the expired Rs. 5,000,000 order is ignored
    _, _, v = run("Drivers for Dubai, processing fee Rs. 1,600,000.")
    [fee] = fee_claims(v)
    assert fee.status == "contradicted" and "Rs. 1,500,000" in fee.source


def test_registry_prefers_slbfe_list_over_demo(tmp_path, monkeypatch):
    real = tmp_path / "slbfe_registry.csv"
    real.write_text("name,licence_no,status,source,retrieved\nREAL AGENCY (PVT) LTD,3532,licensed,https://x,2026-10-17\n")
    demo_copy = tmp_path / "registry.csv"
    demo_copy.write_text(V.DEMO_REGISTRY.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(V, "SLBFE_REGISTRY", real)
    monkeypatch.setattr(V, "REGISTRY_PATH", demo_copy)  # a stale copy of the demo list must not win
    assert list(_LOAD_REGISTRY()["licence_no"]) == ["3532"]
    monkeypatch.setattr(V, "SLBFE_REGISTRY", tmp_path / "missing.csv")
    assert "1023" in set(_LOAD_REGISTRY()["licence_no"])  # demo is the last fallback
