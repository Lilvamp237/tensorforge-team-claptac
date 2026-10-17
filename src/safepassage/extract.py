"""Entity extraction: category, language, country, agency, licence, fee, salary. Owner: Person A.

`extract_content()` fills the shared `Content` contract. `extract_details()` also returns the money amounts and
licence number, which verify.py uses for the SLBFE registry match and fee-cap check.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache

import joblib

from .config import MODELS_DIR
from .schema import Content

CATEGORY_PATH = MODELS_DIR / "category.joblib"
CATEGORY_MIN_PROB = 0.4

# Rough LKR rates, only used to compare a fee with the SLBFE cap. Not for display as exact values.
LKR_PER_UNIT = {"LKR": 1, "USD": 300, "EUR": 330, "GBP": 390, "QAR": 82, "AED": 82, "SAR": 80, "KWD": 980,
                "OMR": 780, "JPY": 2.0, "KRW": 0.22, "AUD": 195, "CAD": 215, "NZD": 180, "THB": 9, "RUB": 3.5,
                "GEL": 110, "MYR": 68}
_CURRENCY_ALIASES = {"rs": "LKR", "rs.": "LKR", "lkr": "LKR", "රු": "LKR", "රු.": "LKR", "ரூ": "LKR", "ரூ.": "LKR",
                     "$": "USD", "us$": "USD", "€": "EUR", "£": "GBP"}
_MULTIPLIERS = {"lakh": 100_000, "lakhs": 100_000, "lak": 100_000, "laks": 100_000, "லட்சம்": 100_000,
                "ලක්ෂ": 100_000, "mn": 1_000_000, "million": 1_000_000, "k": 1_000, "thousand": 1_000,
                "ஆயிரம்": 1_000, "දහස": 1_000}

_CUR = r"(Rs\.?|LKR|රු\.?|ரூ\.?|US\$|\$|€|£|" + "|".join(c for c in LKR_PER_UNIT if c != "LKR") + r")"
_NUM = r"(\d[\d,]*(?:\.\d+)?)"
_MULT = r"(lakhs?|laks?|mn|million|k|thousand|லட்சம்|ලක්ෂ|ஆயிரம்|දහස)"
_MONEY_RX = [
    re.compile(_CUR + r"\s*" + _NUM + r"(?:\s*" + _MULT + r"\b)?", re.IGNORECASE),  # Rs. 150,000 / USD 3,000
    re.compile(_NUM + r"\s*" + _MULT + r"(?![A-Za-z])(?:\s*(?:rupees|ரூபாய்|රුපියල්))?", re.IGNORECASE),  # 5 லட்சம்
]
_FEE_WORDS = re.compile(r"fee|deposit|pay|advance|charge|gewanna|gewala|gewwama|ගාස්තු|ගෙව|තැන්පතු|අත්තිකාරම්|"
                        r"கட்டணம்|செலுத்த", re.IGNORECASE)
_SALARY_WORDS = re.compile(r"salary|padi|earn|income|per (month|day)|monthly|daily|wage|වැටුප|උපයන්න|දිනකට|"
                           r"சம்பளம்|மாதம்|நாளொன்றுக்கு", re.IGNORECASE)

_SINHALA = re.compile(r"[඀-෿]")
_TAMIL = re.compile(r"[஀-௿]")
_LATIN = re.compile(r"[A-Za-z]")
_SINGLISH_WORDS = {"karanna", "ona", "na", "eka", "ekak", "ekata", "wenna", "witharai", "gewanna", "rassawal",
                   "rassawak", "rassawa", "padi", "salli", "yanna", "thiyenawa", "thiyenne", "denawa", "ada", "hadisi",
                   "kenek", "kenekuta", "wala", "gedara", "inna", "gaman", "api", "enna", "epa", "ikmanata",
                   "awashyai", "kiyanna", "dapan", "eken", "gewala", "hambu", "thamai", "puluwan", "denna", "ba"}

COUNTRIES = {
    "Qatar": ["qatar", "doha", "කටාර්", "கத்தார்"],
    "United Arab Emirates": ["uae", "dubai", "abu dhabi", "sharjah", "ඩුබායි", "டுபாய்", "ஐக்கிய அரபு"],
    "Saudi Arabia": ["saudi", "riyadh", "jeddah", "සෞදි", "சவுதி"],
    "Kuwait": ["kuwait", "කුවේට්", "குவைத்"],
    "Oman": ["oman", "muscat", "ඕමාන්", "ஓமான்"],
    "Bahrain": ["bahrain"],
    "Israel": ["israel", "ඊශ්‍රායල", "இஸ்ரேல்"],
    "Jordan": ["jordan"],
    "Maldives": ["maldives", "malé", "මාලදිවයින"],
    "Japan": ["japan", "tokyo", "ජපාන", "ஜப்பான்"],
    "South Korea": ["korea", "seoul", "කොරියා", "கொரியா"],
    "Malaysia": ["malaysia", "kuala lumpur", "මැලේසියා", "மலேசியா"],
    "Singapore": ["singapore"],
    "Thailand": ["thailand", "bangkok", "තායිලන්ත", "தாய்லாந்து"],
    "Myanmar": ["myanmar", "myawaddy", "burma"],
    "Cambodia": ["cambodia"],
    "Laos": ["laos"],
    "Romania": ["romania"], "Poland": ["poland"], "Malta": ["malta"], "Cyprus": ["cyprus"], "Croatia": ["croatia"],
    "Serbia": ["serbia"], "Portugal": ["portugal"], "Italy": ["italy", "ඉතාලි"], "Germany": ["germany", "ජර්මනි"],
    "Ireland": ["ireland"], "Latvia": ["latvia"], "Lithuania": ["lithuania"], "Belarus": ["belarus"],
    "Georgia": ["georgia", "tbilisi"], "Russia": ["russia", "moscow", "රුසියා", "ரஷ்யா"],
    "Uzbekistan": ["uzbekistan"], "Seychelles": ["seychelles"], "Vanuatu": ["vanuatu"],
    "United Kingdom": ["the uk", "united kingdom", "england", "london", "එංගලන්ත", "பிரித்தானியா"],
    "Canada": ["canada", "toronto", "කැනඩා", "கனடா"],
    "Australia": ["australia", "sydney", "melbourne", "ඕස්ට්‍රේලියා", "அவுஸ்திரேலியா"],
    "New Zealand": ["new zealand", "නවසීලන්ත", "நியூசிலாந்து"],
    "United States": ["usa", "united states", "america"],
}
SL_PLACES = ["colombo", "kandy", "gampaha", "kurunegala", "galle", "negombo", "matara", "jaffna", "batticaloa",
             "trincomalee", "anuradhapura", "ratnapura", "badulla", "kalutara", "sri lanka", "කොළඹ", "මහනුවර"]

# Agency names: Capitalised words ending in a typical company suffix, e.g. "Lanka Bright Overseas (Pvt) Ltd".
_AGENCY_SUFFIX = (r"(?:\(Pvt\)|Pvt\.?|Private|Ltd\.?|Limited|Agency|Manpower|Recruiters?|Recruitment|Consultants?|"
                  r"Consultancy|Foreign Employment|Careers|Services|Education|Overseas|International|Holdings|"
                  r"Migration|Placements?)")
_AGENCY_RX = re.compile(r"\b((?:[A-Z][\w&'-]*\s+){1,4}?" + _AGENCY_SUFFIX + r"(?:\s+" + _AGENCY_SUFFIX + r")*)")
_AGENCY_STOP = {"the", "contact", "call", "visit", "our", "we", "registered", "approved", "by", "with", "from", "and",
                "beware", "selected", "job", "apply"}
_LICENCE_RX = re.compile(r"(?:licen[cs]e|lic\.?|බලපත්‍ර|உரிமம்)\s*(?:no\.?|number|අංක|இல\.?)?\s*[:#]?\s*(\d{3,5})",
                         re.IGNORECASE)


@dataclass
class Money:
    amount: float
    currency: str
    lkr: float
    kind: str  # fee | salary | unknown
    raw: str


@dataclass
class Details:
    content: Content
    licence_no: str | None = None
    fees: list[Money] = field(default_factory=list)
    salaries: list[Money] = field(default_factory=list)

    @property
    def max_fee_lkr(self) -> float | None:
        return max((m.lkr for m in self.fees), default=None)


# ---------------------------------------------------------------- language
def detect_language(text: str) -> str:
    si, ta, la = len(_SINHALA.findall(text)), len(_TAMIL.findall(text)), len(_LATIN.findall(text))
    total = si + ta + la
    if total == 0:
        return "unknown"
    if si / total > 0.25 and ta / total > 0.25:
        return "mixed"
    if si / total > 0.25:
        return "si"
    if ta / total > 0.25:
        return "ta"
    words = re.findall(r"[a-z]+", text.lower())
    hits = sum(w in _SINGLISH_WORDS for w in words)
    return "singlish" if hits >= 2 and hits / max(len(words), 1) > 0.05 else "en"


# ---------------------------------------------------------------- category
@lru_cache(maxsize=1)
def _category_model():
    return joblib.load(CATEGORY_PATH) if CATEGORY_PATH.exists() else None


_CATEGORY_KEYWORDS = {
    "student_visa": r"student visa|university|intake|ielts|scholarship|tuition|degree|diploma|study|ශිෂ්‍ය|மாணவர்",
    "visa_pr": r"\bpr\b|permanent residen|express entry|migrat|golden visa|sponsorship|ස්ථිර පදිංචිය|குடியுரிமை",
    "local_job": r"work from home|part[- ]time|epf|etf|o/l|gedara|ගෙදර සිට|வீட்டிலிருந்து",
    "foreign_job": r"overseas|abroad|foreign|slbfe|visa|විදේශ|வெளிநாட்டு",
}


def classify_category(text: str) -> str:
    model = _category_model()
    if model is not None:
        probs = model.predict_proba([text])[0]
        best = probs.argmax()
        if probs[best] >= CATEGORY_MIN_PROB:
            return str(model.classes_[best])
    for cat, rx in _CATEGORY_KEYWORDS.items():
        if re.search(rx, text, re.IGNORECASE):
            return cat
    return "unknown"


# ---------------------------------------------------------------- country / agency / licence
def find_country(text: str) -> str | None:
    low = text.lower()
    hits = []
    for country, aliases in COUNTRIES.items():
        for a in aliases:
            pos = low.find(a) if not a.isascii() else _word_pos(low, a)
            if pos >= 0:
                hits.append((pos, country))
                break
    if hits:
        return min(hits)[1]
    if any(_word_pos(low, p) >= 0 if p.isascii() else p in low for p in SL_PLACES):
        return "Sri Lanka"
    return None


def _word_pos(low: str, word: str) -> int:
    m = re.search(r"(?<![a-z])" + re.escape(word) + r"(?![a-z])", low)
    return m.start() if m else -1


def find_agency(text: str) -> str | None:
    for m in _AGENCY_RX.finditer(text):
        words = m.group(1).split()
        while words and words[0].lower() in _AGENCY_STOP:
            words = words[1:]
        name = " ".join(words).rstrip(".,")
        if len(words) >= 2 and not re.fullmatch(_AGENCY_SUFFIX + r"(?:\s+" + _AGENCY_SUFFIX + r")*", name):
            return name
    return None


def find_licence(text: str) -> str | None:
    m = _LICENCE_RX.search(text)
    return m.group(1) if m else None


# ---------------------------------------------------------------- money
def _to_number(num: str, mult: str | None) -> float:
    value = float(num.replace(",", ""))
    return value * _MULTIPLIERS.get((mult or "").lower(), 1)


def find_money(text: str) -> list[Money]:
    found, taken = [], []
    for i, rx in enumerate(_MONEY_RX):
        for m in rx.finditer(text):
            if any(s <= m.start() < e for s, e in taken):
                continue
            if i == 0:
                cur, num, mult = m.group(1), m.group(2), m.group(3)
                currency = _CURRENCY_ALIASES.get(cur.lower(), cur.upper())
            else:
                num, mult, currency = m.group(1), m.group(2), "LKR"
            try:
                amount = _to_number(num, mult)
            except ValueError:
                continue
            taken.append((m.start(), m.end()))
            found.append(Money(amount, currency, amount * LKR_PER_UNIT.get(currency, 1), _money_kind(text, m), m.group(0)))
    return found


def _money_kind(text: str, m: re.Match) -> str:
    """Fee or salary, decided by the closest keyword in the same sentence."""
    start = max(text.rfind(c, 0, m.start()) for c in ".!?\n|") + 1
    ends = [i for i in (text.find(c, m.end()) for c in ".!?\n|") if i >= 0]
    sentence_end = min(ends) if ends else len(text)
    best = ("unknown", 10_000)
    for kind, rx in (("fee", _FEE_WORDS), ("salary", _SALARY_WORDS)):
        for k in rx.finditer(text, start, sentence_end):
            dist = m.start() - k.end() if k.end() <= m.start() else k.start() - m.end()
            if abs(dist) < best[1]:
                best = (kind, abs(dist))
    return best[0]


# ---------------------------------------------------------------- public API
def extract_details(text: str) -> Details:
    money = find_money(text)
    content = Content(
        type=classify_category(text),
        language=detect_language(text),
        claimed_country=find_country(text),
        claimed_agency=find_agency(text),
    )
    return Details(
        content=content,
        licence_no=find_licence(text),
        fees=[m for m in money if m.kind == "fee"],
        salaries=[m for m in money if m.kind == "salary"],
    )


def extract_content(text: str) -> Content:
    return extract_details(text).content
