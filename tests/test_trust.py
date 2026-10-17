from safepassage.schema import Claim, Verification
from safepassage.trust import CONTRADICTED_FLOOR, adjust

LICENSED = Claim(claim="Agency 'X' is licensed", status="consistent")
FEE_OK = Claim(claim="Fee of Rs. 300,000 is within the approved maximum", status="consistent")
FEE_BAD = Claim(claim="Fee of Rs. 1,500,000", status="contradicted")


def test_contradicted_check_raises_score():
    p, notes = adjust(0.2, Verification(agency_status="matched", claims=[LICENSED, FEE_BAD]), [])
    assert p == CONTRADICTED_FLOOR and "contradict" in notes[0]


def test_licensed_agency_lowers_score_more_with_fee_inside_cap():
    p_lic, _ = adjust(0.8, Verification(agency_status="matched", claims=[LICENSED]), ["contact_offplatform"])
    p_fee, notes = adjust(0.8, Verification(agency_status="matched", claims=[LICENSED, FEE_OK]), [])
    assert p_fee < p_lic < 0.8 and "job-order cap" in notes[0]


def test_severe_flag_blocks_the_discount():
    p, notes = adjust(0.8, Verification(agency_status="matched", claims=[LICENSED]), ["personal_account"])
    assert p == 0.8 and notes == []


def test_unverified_agency_is_unchanged():
    assert adjust(0.6, Verification(agency_status="not_matched"), []) == (0.6, [])
