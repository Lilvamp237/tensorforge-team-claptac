"""Community reports and the site allow/block lists. Owner: Person B.

Reports are only a signal: a site moves onto a list solely through an analyst decision in the review queue,
so a burst of fake reports from scammers or competitors cannot whitelist or blacklist anyone.
"""

import json
from urllib.parse import urlparse

from . import config
from .monitoring import _append, _read, enqueue_review, pending_reviews, resolve_review


def domain_of(url: str) -> str:
    host = (urlparse(url if "://" in url else f"https://{url}").hostname or "").lower()
    return host.removeprefix("www.")


def _load() -> dict:
    if config.REPUTATION_PATH.exists():
        return json.loads(config.REPUTATION_PATH.read_text(encoding="utf-8"))
    return {"allow": {}, "block": {}}


def lists() -> dict:
    return _load()


def status(url: str) -> str | None:
    domain, data = domain_of(url), _load()
    if domain in data["block"]:
        return "blocklisted"
    if domain in data["allow"]:
        return "allowlisted"
    return None


def set_status(domain: str, listing: str | None, reason: str, by: str) -> None:
    """listing: 'allow', 'block' or None to remove the site from both lists."""
    data = _load()
    data["allow"].pop(domain, None)
    data["block"].pop(domain, None)
    if listing:
        data[listing][domain] = {"reason": reason, "by": by}
    config.REPUTATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.REPUTATION_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def add_report(url: str, verdict: str, comment: str = "", rater_id: str | None = None) -> dict:
    """Record a user report ('scam' or 'legit'); escalate the site to an analyst once reports pile up.
    `rater_id` is stored only as a salted hash and weights the report in community.signal()."""
    from .community import rater_hash

    domain = domain_of(url)
    rater = rater_hash(rater_id) if rater_id else None
    record = _append(config.REPORTS_LOG, {"domain": domain, "url": url, "verdict": verdict, "comment": comment[:500],
                                          "rater": rater})
    scam_reports = sum(r["domain"] == domain and r["verdict"] == "scam" for r in _read(config.REPORTS_LOG))
    review_id = f"domain:{domain}"
    already_queued = any(p["review_id"] == review_id for p in pending_reviews())
    if scam_reports >= config.REPORT_THRESHOLD and status(url) is None and not already_queued:
        enqueue_review({"review_id": review_id, "kind": "domain", "domain": domain, "scam_reports": scam_reports})
    return {**record, "scam_reports": scam_reports}


def apply_review(review_id: str, decision: str, reviewer: str = "analyst") -> dict:
    """Resolve a queued item; for reported sites, 'block' / 'allow' updates the lists."""
    record = resolve_review(review_id, decision, reviewer)
    if record.get("kind") == "domain" and decision in ("block", "allow"):
        set_status(record["domain"], decision, f"{record.get('scam_reports', 0)} user reports, analyst decision", reviewer)
    return record


def reports(domain: str | None = None) -> list[dict]:
    return [r for r in _read(config.REPORTS_LOG) if domain is None or r["domain"] == domain]
