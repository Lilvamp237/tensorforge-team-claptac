import pytest

from safepassage.extract import detect_language, extract_details, find_agency, find_country, find_money
from safepassage.features import find_red_flags


@pytest.mark.parametrize("text,lang", [
    ("Heavy vehicle drivers needed in Qatar, salary QAR 1,400.", "en"),
    ("Qatar wala driver rassawal, padi Rs. 450,000. Registration fee eka ada ma gewanna.", "singlish"),
    ("කටාර් සඳහා රියදුරු පුරප්පාඩු, මාසික වැටුප රු. 450,000.", "si"),
    ("கத்தார் இல் ஓட்டுநர் வேலை வாய்ப்பு, மாத சம்பளம் 3 லட்சம் ரூபாய்.", "ta"),
])
def test_language(text, lang):
    assert detect_language(text) == lang


def test_country_aliases():
    assert find_country("Work in Dubai as a cleaner") == "United Arab Emirates"
    assert find_country("ජපාන රැකියා") == "Japan"
    assert find_country("Cashier needed in Kandy") == "Sri Lanka"
    assert find_country("No location here") is None


def test_agency_and_licence():
    text = "Contact Lanka Bright Overseas (Pvt) Ltd, SLBFE licence no. 1023. Tel 011 XXX XXXX."
    d = extract_details(text)
    assert d.content.claimed_agency == "Lanka Bright Overseas (Pvt) Ltd"
    assert d.licence_no == "1023"
    assert find_agency("We are recruiting drivers.") is None


def test_fee_vs_salary():
    d = extract_details("Salary USD 3,000 per month. Pay the registration fee of Rs. 150,000 today.")
    assert [m.amount for m in d.salaries] == [3000]
    assert [m.amount for m in d.fees] == [150000]
    assert d.max_fee_lkr == 150000


def test_lakh_amounts():
    (m,) = find_money("மாதம் 5 லட்சம் ரூபாய்")
    assert m.lkr == 500_000 and m.currency == "LKR"
    assert find_money("Bags up to 20 kg allowed") == []


@pytest.mark.parametrize("text,flags", [
    ("Registration fee eka ada ma gewanna. Visa eka 100% guarantee. Salli personal account ekata dapan.",
     {"upfront_fee", "urgency", "guaranteed", "personal_account"}),
    ("ලියාපදිංචි ගාස්තුව අදම ගෙවන්න. 100% වීසා සහතිකයි. සම්මුඛ පරීක්ෂණයක් අවශ්‍ය නැත.",
     {"upfront_fee", "urgency", "guaranteed", "no_interview"}),
    ("பதிவு கட்டணம் உடனடியாக செலுத்தவும். 100% விசா உத்தரவாதம்.", {"upfront_fee", "urgency", "guaranteed"}),
    ("Visit visa eken gihin work visa ekata convert karanawa. SLBFE register wenna ona na.",
     {"visa_conversion", "no_slbfe"}),
    ("Pay the processing fee in USDT (TRC20) to the wallet address below.", {"crypto_payment", "upfront_fee"}),
])
def test_multilingual_red_flags(text, flags):
    assert flags <= set(find_red_flags(text))


def test_legit_disclaimer_is_not_guarantee():
    assert "guaranteed" not in find_red_flags("Visa decision is made only by the embassy, we cannot guarantee it.")
