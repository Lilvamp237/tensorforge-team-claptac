"""Verification against official data: agency registry + fee rules. Owner: Person A.

Deterministic checks only (no LLM), each claim cites its source:
  1. Agency: claimed name or licence number matched against the registry (fuzzy name match).
  2. Fees: every fee in the text is compared with the SLBFE rules in data/kb/fee_rules.json
     (no fee for Middle East domestic work, highest published cap, SLBFE registration fee, professional jobs).
  3. Rule claims from red flags that contradict SLBFE rules (e.g. "no need to register with SLBFE").
"""

import json
import re
from difflib import SequenceMatcher
from functools import lru_cache

import pandas as pd

from .config import KB_DIR, REGISTRY_PATH
from .extract import extract_details
from .features import find_red_flags
from .schema import Claim, Content, Verification

DEMO_REGISTRY = KB_DIR / "registry_demo.csv"
FEE_RULES = KB_DIR / "fee_rules.json"
CLASSIFICATION = KB_DIR / "slbfe_job_classification.csv"
NAME_MATCH = 0.88
_DOMESTIC = re.compile(r"house ?maid|domestic (worker|helper|housekeep)|ගෘහ සේවික|வீட்டுப் பணிப்பெண்", re.IGNORECASE)
_SUFFIXES = re.compile(r"\(?\b(pvt|private|ltd|limited|co)\b\.?\)?", re.IGNORECASE)


@lru_cache(maxsize=1)
def load_registry() -> pd.DataFrame:
    path = REGISTRY_PATH if REGISTRY_PATH.exists() else DEMO_REGISTRY
    df = pd.read_csv(path, dtype=str).fillna("")
    df["key"] = df["name"].map(normalise)
    return df


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
    if not name and not licence_no:
        return "unknown", None
    reg = load_registry()
    source = reg["source"].iloc[0] if len(reg) else None
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
                                           f"({by_lic['source']})")
    hit = by_name if by_name is not None else by_lic
    if hit is None:
        return "not_matched", Claim(claim=f"Agency '{label}' is licensed", status="unverifiable",
                                    source=f"Not found in registry ({source})")
    if hit["status"] != "licensed":
        return "not_matched", Claim(claim=f"Agency '{label}' is licensed", status="contradicted",
                                    source=f"Registry lists '{hit['name']}' as {hit['status']} ({hit['source']})")
    if licence_no and hit["licence_no"].lstrip("0") != licence_no.lstrip("0"):
        return "not_matched", Claim(claim=f"Licence {licence_no} belongs to '{label}'", status="contradicted",
                                    source=f"Registry has licence {hit['licence_no']} for '{hit['name']}'")
    return "matched", Claim(claim=f"Agency '{label}' is licensed", status="consistent",
                            source=f"Registry: '{hit['name']}', licence {hit['licence_no']} ({hit['source']})")


def check_fees(text: str, content: Content, fees, flags: list[str]) -> list[Claim]:
    rules = load_rules()
    claims = []
    domestic_me = bool(_DOMESTIC.search(text)) and content.claimed_country in rules["middle_east_countries"]
    professional = _professional_titles()
    pro_job = professional.search(text) if professional is not None and content.type == "foreign_job" else None
    reg_fee = rules["slbfe_registration_fee_lkr"]["value"]
    cap = rules["highest_published_cap_lkr"]

    for fee in fees:
        what = f"Fee of {fee.raw.strip()}"
        if domestic_me:
            r = rules["domestic_middle_east_agency_fee_lkr"]
            claims.append(Claim(claim=f"{what} for domestic work in {content.claimed_country}", status="contradicted",
                                source=f"{r['note']} {r['source']}"))
        elif fee.lkr > cap["value"]:
            approx = "" if fee.currency == "LKR" else f" (about Rs. {fee.lkr:,.0f})"
            claims.append(Claim(claim=f"{what}{approx}", status="contradicted",
                                source=f"Above the highest SLBFE-published maximum on record, Rs. {cap['value']:,}. "
                                       f"{cap['source']}"))
        elif abs(fee.lkr - reg_fee) <= 0.1 * reg_fee and "slbfe" in text.lower():
            claims.append(Claim(claim=f"{what} as the SLBFE registration fee", status="consistent",
                                source=rules["slbfe_registration_fee_lkr"]["source"]))
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
    status, agency_claim = match_agency(content.claimed_agency, details.licence_no)
    claims = [c for c in [agency_claim] if c] + check_fees(text, content, details.fees, flags) + check_rules(flags, content)
    return Verification(agency_status=status, claims=claims)
