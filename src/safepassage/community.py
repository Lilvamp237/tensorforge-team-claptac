"""Community signal from user ratings of a site: trust-weighted, capped, and never decisive. Owner: Person B.

- Rater weight: verified raters (granted by an analyst: NGOs, job-centre officers, SLBFE-registered workers)
  count 1.0, everyone else 0.25. It is then scaled by how often the rater's past reports agreed with analyst
  decisions on other sites (Laplace-smoothed), so reliable raters gain influence and careless ones lose it.
- Score: Bayesian average with a neutral prior, so a handful of ratings barely moves it.
- Effect on the risk score is capped at +/-MAX_SHIFT and frozen during a burst of reports (possible brigading).
- Lists still change only through an analyst (reputation.apply_review). Rater ids are stored as salted hashes.
"""

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone

from . import config, reputation
from .monitoring import pending_reviews
from .schema import Verification

MAX_SHIFT = 0.15
PRIOR = 2.0                    # pseudo-ratings on each side
VERIFIED_WEIGHT, UNVERIFIED_WEIGHT = 1.0, 0.25
BURST_REPORTS, BURST_MINUTES = 10, 10
_SALT = os.environ.get("SAFEPASSAGE_RATER_SALT", "safepassage-demo-salt")


def rater_hash(rater_id: str) -> str:
    return hashlib.sha256((_SALT + rater_id).encode()).hexdigest()[:16]


def _raters_path():
    return config.LOGS_DIR / "raters.json"


def _raters() -> dict:
    path = _raters_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def verify_rater(rater_id: str, by: str = "analyst") -> dict:
    """Analyst grants verified-rater status."""
    raters = _raters()
    raters[rater_hash(rater_id)] = {"verified": True, "by": by}
    _raters_path().parent.mkdir(parents=True, exist_ok=True)
    _raters_path().write_text(json.dumps(raters, indent=1), encoding="utf-8")
    return {"rater": rater_hash(rater_id), "verified": True}


def rater_weight(rater: str | None, reports: list[dict], decisions: dict[str, str]) -> float:
    if not rater:
        return UNVERIFIED_WEIGHT
    base = VERIFIED_WEIGHT if _raters().get(rater, {}).get("verified") else UNVERIFIED_WEIGHT
    judged = [r for r in reports if r.get("rater") == rater and r["domain"] in decisions]
    agree = sum(r["verdict"] == decisions[r["domain"]] for r in judged)
    return base * 2 * (agree + 1) / (len(judged) + 2)  # 1x with no history, up to 2x when always right


def _decisions() -> dict[str, str]:
    lists = reputation.lists()
    return {**{d: "legit" for d in lists["allow"]}, **{d: "scam" for d in lists["block"]}}


def signal(url: str | None) -> tuple[float, str | None]:
    """Score shift and an explanation for a site, or (0, None) when there is nothing to say."""
    if not url:
        return 0.0, None
    domain, all_reports = reputation.domain_of(url), reputation.reports()
    site = [r for r in all_reports if r["domain"] == domain]
    if not site:
        return 0.0, None
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=BURST_MINUTES)
    if sum(datetime.fromisoformat(r["ts"]) >= cutoff for r in site) >= BURST_REPORTS:
        return 0.0, f"Community: {len(site)} reports, frozen for review (unusual burst)"
    decisions = _decisions()
    scam = sum(rater_weight(r.get("rater"), all_reports, decisions) for r in site if r["verdict"] == "scam")
    legit = sum(rater_weight(r.get("rater"), all_reports, decisions) for r in site if r["verdict"] == "legit")
    score = (scam + PRIOR) / (scam + legit + 2 * PRIOR)
    shift = max(-MAX_SHIFT, min(MAX_SHIFT, (score - 0.5) * 0.6))
    return shift, f"Community: {len(site)} reports, trust-weighted scam share {score:.2f} (score {shift:+.2f})"


def standing(url: str | None, verification: Verification) -> list[str]:
    """Graded, defensible labels: official facts and analyst decisions only, never "fake" from votes."""
    notes = []
    for c in verification.claims:
        src = (c.source or "").lower()
        if c.status == "contradicted" and ("cancelled" in src or "suspended" in src):
            notes.append("SLBFE records: licence cancelled or suspended")
            break
    else:
        if verification.agency_status == "matched":
            notes.append("SLBFE records: licensed agency")
    if url:
        listing, domain = reputation.status(url), reputation.domain_of(url)
        if listing == "blocklisted":
            notes.append("Confirmed scam (analyst decision; the agency can appeal)")
        elif listing == "allowlisted":
            notes.append("Verified by analyst")
        elif any(p["review_id"] == f"domain:{domain}" for p in pending_reviews()):
            notes.append("Reported, under review")
    return notes
