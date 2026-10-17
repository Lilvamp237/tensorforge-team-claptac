"""Verification against official data: agency registry + fee rules. Owner: Person A.

Deterministic checks only (no LLM), each claim cites its source:
  1. Agency: claimed name or licence number matched against the SLBFE registry (fuzzy name match).
     data/kb/slbfe_registry.csv is built from SLBFE's public valid / suspended / cancelled agency lists
     (scripts/scrape_slbfe.py --jobbank); the fictional data/kb/registry_demo.csv is only a last fallback.
  2. Fees: every fee in the text is compared with
     - the agency's own SLBFE-approved job order (data/kb/slbfe_job_orders.csv) when the agency matched and a job order
       with a similar title (and the same country, if one was extracted) is still valid: above its
       "Maximum Amount Chargeable by Sri Lanka Agency" is contradicted, at or below it is consistent;
     - the SLBFE rules in data/kb/fee_rules.json (no fee for Middle East domestic work, SLBFE registration fee,
       professional jobs) and the highest maximum fee among all currently valid job orders.
  3. Rule claims from red flags that contradict SLBFE rules (e.g. "no need to register with SLBFE").
"""

import json
import re
from datetime import date
from difflib import SequenceMatcher
from functools import lru_cache

import pandas as pd

from .config import KB_DIR, REGISTRY_PATH
from .extract import extract_details
from .features import find_red_flags
from .schema import Claim, Content, Verification

SLBFE_REGISTRY = KB_DIR / "slbfe_registry.csv"
JOB_ORDERS = KB_DIR / "slbfe_job_orders.csv"
DEMO_REGISTRY = KB_DIR / "registry_demo.csv"
FEE_RULES = KB_DIR / "fee_rules.json"
CLASSIFICATION = KB_DIR / "slbfe_job_classification.csv"
NAME_MATCH = 0.88
TITLE_MATCH = 0.5  # share of a job order's distinctive title words that must appear in the offer
_DOMESTIC = re.compile(r"house ?maid|domestic (worker|helper|housekeep)|ගෘහ සේවික|வீட்டுப் பணிப்பெண்", re.IGNORECASE)
_SUFFIXES = re.compile(r"\(?\b(pvt|private|ltd|limited|co)\b\.?\)?", re.IGNORECASE)
# Job-bank country names -> the names extract.find_country() returns.
_COUNTRY_ALIASES = {"u a e": "United Arab Emirates", "uae": "United Arab Emirates", "united state": "United States",
                    "republic of korea": "South Korea", "korea": "South Korea"}
# Words that say nothing about which job it is ("Specified Skilled Worker (SSW) - Jobs (Bus Drivers)").
_TITLE_STOP = {"specified", "skilled", "worker", "workers", "ssw", "jobs", "job", "general", "and", "the", "for",
               "with", "of", "in", "industry", "field", "business", "male", "female", "junior", "senior", "assistant",
               "helper", "labour", "labor", "staff", "other", "others", "related", "work", "works", "cum"}


def _read_registry(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    df["key"] = df["name"].map(normalise)
    return df


@lru_cache(maxsize=1)
def load_registry() -> pd.DataFrame:
    """data/processed/registry.csv (e.g. an --agencies import) unless it is only a copy of the demo list, then the
    crawled SLBFE registry, then the fictional demo registry."""
    if REGISTRY_PATH.exists():
        df = _read_registry(REGISTRY_PATH)
        if len(df) and not df["source"].str.startswith("DEMO").all():
            return df
    return _read_registry(SLBFE_REGISTRY if SLBFE_REGISTRY.exists() else DEMO_REGISTRY)


@lru_cache(maxsize=1)
def load_job_orders() -> pd.DataFrame:
    cols = ["licence_no", "country", "job_title", "max_fee_lkr", "approved_date", "valid_until", "source"]
    if not JOB_ORDERS.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(JOB_ORDERS, dtype=str).fillna("")
    df["max_fee_lkr"] = pd.to_numeric(df["max_fee_lkr"], errors="coerce")
    return df.dropna(subset=["max_fee_lkr"])


def _valid_orders(orders: pd.DataFrame, today: str | None = None) -> pd.DataFrame:
    today = today or date.today().isoformat()
    return orders[orders["valid_until"] >= today]


def highest_valid_cap(orders: pd.DataFrame | None = None) -> tuple[float, str] | None:
    """(highest max_fee_lkr among currently valid job orders, its source URL), or None without job-order data."""
    valid = _valid_orders(load_job_orders() if orders is None else orders)
    if valid.empty:
        return None
    top = valid.loc[valid["max_fee_lkr"].idxmax()]
    return float(top["max_fee_lkr"]), top["source"]


def _country(name: str) -> str:
    return _COUNTRY_ALIASES.get(name.strip().lower(), name.strip())


def _title_words(title: str) -> set[str]:
    words = {w.rstrip("s") if len(w) > 4 else w for w in re.findall(r"[a-z]{3,}", title.lower())}
    return {w for w in words if w not in _TITLE_STOP}


def find_job_order(text: str, licence_no: str, country: str | None) -> pd.Series | None:
    """The agency's valid job order whose title best matches the offer (same country if one was extracted).
    When several orders match equally well, the most generous cap is used, so we never under-state what is allowed."""
    orders = _valid_orders(load_job_orders())
    orders = orders[orders["licence_no"].str.lstrip("0") == licence_no.lstrip("0")]
    if country:
        orders = orders[orders["country"].map(_country) == country]
    if orders.empty:
        return None
    offer = _title_words(text)
    best, best_score = None, 0.0
    for _, row in orders.iterrows():
        words = _title_words(row["job_title"])
        if not words:
            continue
        score = len(words & offer) / len(words)
        if score > best_score or (score == best_score and best is not None and row["max_fee_lkr"] > best["max_fee_lkr"]):
            best, best_score = row, score
    return best if best_score >= TITLE_MATCH else None


@lru_cache(maxsize=1)
def load_rules() -> dict:
    return json.loads(FEE_RULES.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _professional_titles() -> re.Pattern | None:
    if not CLASSIFICATION.exists():
        return None
    jobs = pd.read_csv(CLASSIFICATION)["job"].astype(str)
    # "Accountant - Budget" -> "accountant"; keep distinctive titles only.
    bases = {re.split(r"\s*[-(]", j)[0].strip().lower() for j in jobs}
    bases = sorted((b for b in bases if len(b) >= 5), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(map(re.escape, bases)) + r")s?\b", re.IGNORECASE)


def normalise(name: str) -> str:
    name = _SUFFIXES.sub(" ", name.lower())
    return re.sub(r"[^\w\s]", " ", re.sub(r"\s+", " ", name)).strip()


def match_agency(name: str | None, licence_no: str | None) -> tuple[str, Claim | None]:
    status, claim, _ = _match_agency(name, licence_no)
    return status, claim


def _match_agency(name: str | None, licence_no: str | None) -> tuple[str, Claim | None, str | None]:
    """(status, claim, licence number of the matched licensed agency or None)."""
    if not name and not licence_no:
        return "unknown", None, None
    reg = load_registry()
    sources = ", ".join(dict.fromkeys(reg["source"])) if len(reg) else None
    by_name = None
    if name:
        key = normalise(name)
        scores = reg["key"].map(lambda k: SequenceMatcher(None, k, key).ratio())
        if len(scores) and scores.max() >= NAME_MATCH:
            by_name = reg.loc[scores.idxmax()]
    by_lic = None
    if licence_no:
        rows = reg[reg["licence_no"].str.lstrip("0") == licence_no.lstrip("0")]
        by_lic = rows.iloc[0] if len(rows) else None

    label = name or f"licence {licence_no}"
    # A real licence number quoted under a different agency's name is a classic impersonation pattern.
    if name and by_lic is not None and (by_name is None or by_name["licence_no"] != by_lic["licence_no"]):
        return "not_matched", Claim(claim=f"Licence {licence_no} belongs to '{name}'", status="contradicted",
                                    source=f"Registry has licence {by_lic['licence_no']} for '{by_lic['name']}' "
                                           f"({by_lic['source']})"), None
    hit = by_name if by_name is not None else by_lic
    if hit is None:
        return "not_matched", Claim(claim=f"Agency '{label}' is licensed", status="unverifiable",
                                    source=f"Not found in registry ({sources})"), None
    if hit["status"] != "licensed":
        return "not_matched", Claim(claim=f"Agency '{label}' is licensed", status="contradicted",
                                    source=f"Registry lists '{hit['name']}' as {hit['status']} ({hit['source']})"), None
    if licence_no and hit["licence_no"].lstrip("0") != licence_no.lstrip("0"):
        return "not_matched", Claim(claim=f"Licence {licence_no} belongs to '{label}'", status="contradicted",
                                    source=f"Registry has licence {hit['licence_no']} for '{hit['name']}'"), None
    return "matched", Claim(claim=f"Agency '{label}' is licensed", status="consistent",
                            source=f"Registry: '{hit['name']}', licence {hit['licence_no']} ({hit['source']})"), \
        hit["licence_no"]


def global_cap(rules: dict) -> tuple[float, str]:
    """Highest maximum fee among currently valid SLBFE job orders; fee_rules.json's recorded value without the CSV."""
    top = highest_valid_cap()
    if top is not None:
        value, source = top
        return value, (f"Above the highest maximum fee in any currently valid SLBFE-approved job order, "
                       f"Rs. {value:,.0f} ({source}).")
    cap = rules["highest_published_cap_lkr"]
    return cap["value"], f"Above the highest SLBFE-approved maximum on record, Rs. {cap['value']:,}. {cap['source']}"


def check_fees(text: str, content: Content, fees, flags: list[str], licence_no: str | None = None) -> list[Claim]:
    """licence_no: licence of the matched, licensed agency (enables the per-job-order fee check)."""
    rules = load_rules()
    claims = []
    domestic_me = bool(_DOMESTIC.search(text)) and content.claimed_country in rules["middle_east_countries"]
    professional = _professional_titles()
    pro_job = professional.search(text) if professional is not None and content.type == "foreign_job" else None
    reg_fee = rules["slbfe_registration_fee_lkr"]["value"]
    cap, cap_source = global_cap(rules) if fees else (None, None)
    order = find_job_order(text, licence_no, content.claimed_country) if fees and licence_no else None
    if order is not None:
        where = f" in {order['country']}" if order["country"] else ""
        order_desc = (f"SLBFE job order '{order['job_title']}'{where} (approved {order['approved_date']}, valid until "
                      f"{order['valid_until']}): maximum chargeable by the agency Rs. {order['max_fee_lkr']:,.0f}. "
                      f"{order['source']}")

    for fee in fees:
        what = f"Fee of {fee.raw.strip()}"
        approx = "" if fee.currency == "LKR" else f" (about Rs. {fee.lkr:,.0f})"
        if domestic_me:
            r = rules["domestic_middle_east_agency_fee_lkr"]
            claims.append(Claim(claim=f"{what} for domestic work in {content.claimed_country}", status="contradicted",
                                source=f"{r['note']} {r['source']}"))
        elif order is not None and fee.lkr > order["max_fee_lkr"]:
            claims.append(Claim(claim=f"{what}{approx} is within the approved maximum", status="contradicted",
                                source=f"Above the {order_desc}"))
        elif fee.lkr > cap:
            claims.append(Claim(claim=f"{what}{approx}", status="contradicted", source=cap_source))
        elif abs(fee.lkr - reg_fee) <= 0.1 * reg_fee and "slbfe" in text.lower():
            claims.append(Claim(claim=f"{what} as the SLBFE registration fee", status="consistent",
                                source=rules["slbfe_registration_fee_lkr"]["source"]))
        elif order is not None:
            claims.append(Claim(claim=f"{what}{approx} is within the approved maximum", status="consistent",
                                source=f"At or below the {order_desc}"))
        elif pro_job:
            claims.append(Claim(claim=f"{what} for a professional job ({pro_job.group(0)})", status="unverifiable",
                                source=f"{rules['professional_jobs']['note']} {rules['professional_jobs']['source']}"))
        else:
            claims.append(Claim(claim=what, status="unverifiable",
                                source="Check the maximum fee on the SLBFE vacancy listing or call 1989."))
    if fees and "personal_account" in flags:
        claims.append(Claim(claim="Fee paid into a personal account", status="contradicted",
                            source="SLBFE fees are paid to SLBFE (online or at its offices). "
                                   + rules["slbfe_registration_fee_lkr"]["source"]))
    return claims


def check_rules(flags: list[str], content: Content) -> list[Claim]:
    rules = load_rules()
    claims = []
    if "no_slbfe" in flags and content.type in {"foreign_job", "unknown"}:
        claims.append(Claim(claim="SLBFE registration is not needed", status="contradicted",
                            source=rules["slbfe_registration_fee_lkr"]["note"] + " "
                                   + rules["slbfe_registration_fee_lkr"]["source"]))
    return claims


def verify(text: str, content: Content) -> Verification:
    details = extract_details(text)
    flags = find_red_flags(text)
    status, agency_claim, licence = _match_agency(content.claimed_agency, details.licence_no)
    claims = ([c for c in [agency_claim] if c] + check_fees(text, content, details.fees, flags, licence)
              + check_rules(flags, content))
    return Verification(agency_status=status, claims=claims)
