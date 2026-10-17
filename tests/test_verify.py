import pandas as pd
import pytest

from safepassage import verify as V
from safepassage.extract import extract_content


@pytest.fixture(autouse=True)
def demo_registry(monkeypatch):
    """Always test against the committed demo registry, whatever is in data/processed."""
    df = pd.read_csv(V.DEMO_REGISTRY, dtype=str).fillna("")
    df["key"] = df["name"].map(V.normalise)
    monkeypatch.setattr(V, "load_registry", lambda: df)


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
    _, _, v = run("Factory workers for Romania. Placement fee Rs. 750,000. No need to register with SLBFE.")
    assert [c.status for c in v.claims] == ["contradicted", "contradicted"]


def test_fee_under_cap_is_unverifiable():
    _, claims, _ = run("Drivers for Dubai, processing fee Rs. 80,000.")
    assert claims == {("unverifiable", "Fee")}
