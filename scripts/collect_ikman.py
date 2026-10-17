"""Collect public ikman.lk job listings and scan them through the shared SafePassage pipeline.

    PYTHONPATH=src python scripts/collect_ikman.py --once --max-ads 5
    PYTHONPATH=src python scripts/collect_ikman.py --category overseas-jobs --category jobs --interval-min 30
    python scripts/collect_ikman.py --once --api http://localhost:8000            # through the REST API
    PYTHONPATH=src python scripts/collect_ikman.py --once --fixtures tests/fixtures/ikman   # offline demo

Politeness and scope
- robots.txt is fetched at the start of every run and checked for every URL, with urllib.robotparser plus a
  wildcard-aware matcher (robotparser ignores the `*` patterns ikman uses, e.g. `/*?*sort=*`). Anything
  either check disallows is skipped. Pagination uses `?page=N` only when robots allows it.
- One request at a time, at least 1.5 s apart, honest User-Agent, retries with backoff on 5xx and network
  errors, and a clean stop on 403/429.
- ikman's terms (reviewed 17 Oct 2026) contain no clause on automated access, but say site material may not
  be "copied, reproduced, ... stored or distributed" without permission. So raw HTML is NOT kept unless
  --cache-raw is passed (and then only in the gitignored data/raw/ikman/), and runs should stay small.

Privacy (competition Rule 2.4)
- Contact cards, phone lists and shop emails in the page data are never read.
- Phone numbers, emails, NIC numbers and names in "contact <name>" patterns are replaced with tokens
  (<PHONE>, <EMAIL>, <NIC>, <NAME>) before anything is scanned or stored.
- logs/collector_seen.json keeps only ad id, URL, cleaned text and timestamps (used to avoid rescans).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.robotparser
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://ikman.lk"
USER_AGENT = "SafePassageLK-collector/0.1 (hackathon research)"
MIN_DELAY = 1.5
RAW_DIR = ROOT / "data" / "raw" / "ikman"
SEEN_PATH = ROOT / "logs" / "collector_seen.json"
# overseas-jobs: foreign employment ads; jobs: the parent jobs category (local vacancies, from the site nav).
KNOWN_CATEGORIES = ("overseas-jobs", "jobs")
MAX_TEXT = 20_000


class StopCollection(Exception):
    """The site asked us to back off (403/429) or robots.txt could not be read: stop this run cleanly."""


# --------------------------------------------------------------------------------------------- privacy

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Old NIC: 9 digits + V/X. New NIC: 12 digits starting with the birth year (19xx/20xx).
_NIC = re.compile(r"(?<![\w])(?:\d{9}[VvXx]|(?:19|20)\d{10})(?![\w])")
# Sri Lankan numbers: optional +94 / 0094 / 94 (with an optional "(0)"), or a leading 0, then a 2-digit
# mobile/area code and 7 more digits, each optionally separated by a space, dot or dash.
# Covers 0771234567, 077 123 4567, 077-123-4567, +94 77 123 4567, 011-2345678, 011 2 345 678, wa.me/9477...
_LK_PHONE = re.compile(
    r"(?<![\w+])(?:(?:\+|00)?\s?94[\s.-]?(?:\(0\)[\s.-]?)?|\(?0)\d{2}\)?(?:[\s.-]?\d){7}(?!\d)"
)
# Any other international number (Gulf agents often post +971 / +966 numbers).
_INTL_PHONE = re.compile(r"(?<![\w])\+\d{1,3}[\s.-]?\(?\d{1,4}\)?(?:[\s.-]?\d){5,10}(?!\d)")
_CONTACT_NAME = re.compile(
    r"\b((?i:contact|call|ask\s+for|speak\s+to|talk\s+to|whatsapp|viber|inquire\s+with|reach)"
    r"(?:\s+(?i:us|me))?\s*[:\-–]?\s*)"
    r"((?:(?:Mr|Mrs|Ms|Miss|Dr|Rev)\.?\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"
)
_HONORIFIC_NAME = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?")
# Capitalised words after "contact" that are not names.
_NOT_NAMES = {
    "us", "me", "now", "today", "our", "the", "office", "agency", "whatsapp", "viber", "phone", "number",
    "details", "via", "for", "on", "at", "hotline", "immediately", "directly", "head", "company", "hr",
    "manager", "branch", "telegram", "imo", "email", "mobile", "this", "above", "below", "soon", "free",
    "only", "or", "and", "any", "all", "with", "more", "information", "info", "interview", "center", "centre",
    "inquiries", "enquiries", "back", "anytime", "after", "before", "from", "to", "us:", "line",
}


def _name_repl(m: re.Match) -> str:
    words = m.group(2).split()
    if words[0].rstrip(".").lower() in _NOT_NAMES:
        return m.group(0)
    return f"{m.group(1)}<NAME>"


def scrub_pii(text: str) -> str:
    """Replace emails, NIC numbers, phone numbers and contact names with tokens."""
    if not text:
        return ""
    text = _EMAIL.sub("<EMAIL>", text)
    text = _NIC.sub("<NIC>", text)
    text = _LK_PHONE.sub("<PHONE>", text)
    text = _INTL_PHONE.sub("<PHONE>", text)
    text = _CONTACT_NAME.sub(_name_repl, text)
    text = _HONORIFIC_NAME.sub("<NAME>", text)
    return text


# ---------------------------------------------------------------------------------------------- robots


def _rule_regex(pattern: str) -> re.Pattern:
    anchored = pattern.endswith("$")
    body = re.escape(pattern.rstrip("$")).replace(r"\*", ".*")
    return re.compile(body + ("$" if anchored else ""))


class RobotsPolicy:
    """robots.txt check: urllib.robotparser AND a Google-style wildcard matcher must both allow a URL."""

    def __init__(self, text: str, user_agent: str = USER_AGENT, parser=None):
        self.user_agent = user_agent
        self.parser = parser or urllib.robotparser.RobotFileParser()
        if parser is None:
            self.parser.parse(text.splitlines())
        self.rules = self._rules_for_agent(text, user_agent)

    @staticmethod
    def _rules_for_agent(text: str, user_agent: str) -> list[tuple[bool, str, re.Pattern]]:
        token = user_agent.split("/")[0].lower()
        groups: list[tuple[list[str], list[tuple[bool, str]]]] = []
        agents: list[str] = []
        rules: list[tuple[bool, str]] = []
        in_rules = False
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            key, value = (s.strip() for s in line.split(":", 1))
            key = key.lower()
            if key == "user-agent":
                if in_rules:
                    groups.append((agents, rules))
                    agents, rules, in_rules = [], [], False
                agents.append(value.lower())
            elif key in ("allow", "disallow"):
                in_rules = True
                if value:
                    rules.append((key == "allow", value))
        if agents:
            groups.append((agents, rules))
        specific = [r for a, rs in groups if any(x != "*" and token.startswith(x) for x in a) for r in rs]
        chosen = specific or [r for a, rs in groups if "*" in a for r in rs]
        return [(allow, pat, _rule_regex(pat)) for allow, pat in chosen]

    def _wildcard_allows(self, url: str) -> bool:
        parts = urlsplit(url)
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        best: tuple[int, bool] | None = None
        for allow, pat, rx in self.rules:
            if rx.match(path):
                cand = (len(pat), allow)
                if best is None or cand[0] > best[0] or (cand[0] == best[0] and allow):
                    best = cand
        return True if best is None else best[1]

    def allowed(self, url: str) -> bool:
        return bool(self.parser.can_fetch(self.user_agent, url)) and self._wildcard_allows(url)


# --------------------------------------------------------------------------------------------- sources


class LiveSource:
    """Polite HTTP fetcher: robots-checked, rate-limited, retries with backoff, stops on 403/429."""

    def __init__(self, delay: float = 2.0, cache_raw: bool = False, retries: int = 3, session=None):
        self.delay = max(MIN_DELAY, delay)
        self.cache_raw = cache_raw
        self.retries = retries
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en"})
        self._last = 0.0
        self.fetched = 0
        self.robots: RobotsPolicy | None = None

    def _wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self.delay:
            time.sleep(self.delay - gap)
        self._last = time.monotonic()

    def _get(self, url: str) -> str:
        for attempt in range(self.retries + 1):
            self._wait()
            try:
                resp = self.session.get(url, timeout=20)
            except requests.RequestException as exc:
                err = f"network error: {exc.__class__.__name__}"
            else:
                self.fetched += 1
                if resp.status_code in (403, 429):
                    raise StopCollection(f"HTTP {resp.status_code} from {url}; stopping as asked")
                if resp.status_code == 200:
                    return resp.text
                if resp.status_code == 404:
                    raise LookupError(f"HTTP 404 for {url}")
                err = f"HTTP {resp.status_code}"
            if attempt < self.retries:
                time.sleep(min(60, 2 ** (attempt + 1)))
        raise LookupError(f"{err} for {url} after {self.retries + 1} attempts")

    def load_robots(self) -> RobotsPolicy:
        try:
            text = self._get(urljoin(BASE_URL, "/robots.txt"))
        except (LookupError, StopCollection) as exc:
            raise StopCollection(f"could not read robots.txt ({exc}); not collecting") from exc
        self.robots = RobotsPolicy(text)
        return self.robots

    def _cache(self, name: str, html: str) -> None:
        if self.cache_raw:
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            (RAW_DIR / f"{name}.html").write_text(html, encoding="utf-8")

    def listing(self, category: str, page: int) -> str:
        url = listing_url(category, page)
        html = self._get(url)
        self._cache(f"{category}-page-{page}", html)
        return html

    def ad(self, slug: str) -> str:
        html = self._get(ad_url(slug))
        self._cache(f"ad-{slug}", html)
        return html


class FixtureSource:
    """Reads saved HTML instead of fetching: <category>.html, <category>-page-<N>.html, ad-<slug>.html,
    and an optional robots.txt (all URLs are allowed when it is missing)."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.fetched = 0
        self.robots: RobotsPolicy | None = None

    def _read(self, name: str) -> str:
        path = self.dir / name
        if not path.exists():
            raise LookupError(f"no fixture {path}")
        self.fetched += 1
        return path.read_text(encoding="utf-8")

    def load_robots(self) -> RobotsPolicy:
        path = self.dir / "robots.txt"
        self.robots = RobotsPolicy(path.read_text(encoding="utf-8") if path.exists() else "")
        return self.robots

    def listing(self, category: str, page: int) -> str:
        return self._read(f"{category}.html" if page == 1 else f"{category}-page-{page}.html")

    def ad(self, slug: str) -> str:
        return self._read(f"ad-{slug}.html")


def listing_url(category: str, page: int = 1) -> str:
    url = f"{BASE_URL}/en/ads/sri-lanka/{category}"
    return url if page == 1 else f"{url}?page={page}"


def ad_url(slug: str) -> str:
    return f"{BASE_URL}/en/ad/{slug}"


# --------------------------------------------------------------------------------------------- parsing

_INITIAL_DATA = re.compile(r"window\.initialData\s*=\s*(\{.*?\})\s*;?\s*</script>", re.S)
_TAG = re.compile(r"<[^>]+>")


def _initial_data(html: str) -> dict | None:
    m = _INITIAL_DATA.search(html)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _text(fragment: str) -> str:
    from html import unescape

    return re.sub(r"\s+", " ", unescape(_TAG.sub(" ", fragment))).strip()


def _poster_type(auth_agent: bool, member: bool) -> str:
    if auth_agent:
        return "Authorised agent (AUTH AGENT badge)"
    return "Member (MEMBER badge)" if member else "Non-member"


def parse_listing(html: str) -> tuple[list[dict], dict]:
    """Ad cards on a category page, and pagination info {total, pageSize, activePage} when available."""
    data = _initial_data(html)
    cards: list[dict] = []
    pagination: dict = {}
    if data:
        serp = (((data.get("serp") or {}).get("ads") or {}).get("data")) or {}
        pagination = serp.get("paginationData") or {}
        for ad in (serp.get("topAds") or []) + (serp.get("ads") or []):
            if not ad.get("slug"):
                continue
            cards.append({
                "id": ad.get("id") or f"slug:{ad['slug']}",
                "slug": ad["slug"],
                "url": ad_url(ad["slug"]),
                "title": ad.get("title", ""),
                "agency": ad.get("details") or ad.get("shopName") or "",
                "salary": ad.get("price", ""),
                "location": ad.get("location", ""),
                "auth_agent": bool(ad.get("isAuthDealer")),
                "member": bool(ad.get("isMember")),
            })
    else:  # Fallback: server-rendered cards.
        for block in re.split(r'(?=<a [^>]*data-testid="ad-card-link")', html)[1:]:
            href = re.search(r'href="/en/ad/([^"?#]+)"', block)
            if not href:
                continue
            block = block[:4000]
            title = re.search(r"<h2[^>]*>(.*?)</h2>", block, re.S)
            agency = re.search(r"</h2>\s*<div>\s*<div>(.*?)</div>", block, re.S)
            desc = re.search(r'class="description[^"]*">(.*?)</div>', block, re.S)
            price = re.search(r'class="price[^"]*">(.*?)</div>', block, re.S)
            slug = href.group(1)
            cards.append({
                "id": f"slug:{slug}",
                "slug": slug,
                "url": ad_url(slug),
                "title": _text(title.group(1)) if title else "",
                "agency": _text(agency.group(1)) if agency else "",
                "salary": _text(price.group(1)) if price else "",
                "location": _text(desc.group(1)).split(",")[0] if desc else "",
                "auth_agent": "AUTH AGENT" in block,
                "member": "MEMBER" in block,
            })
    seen, unique = set(), []
    for c in cards:
        if c["slug"] not in seen:
            seen.add(c["slug"])
            unique.append(c)
    return unique, pagination


_KEEP_PROPS = {
    "country": "Country",
    "company_name": "Employer",
    "slbfe_license_number": "SLBFE licence number",
    "salary": "Salary",
    "employment": "Job type",
    "item_type": "Industry",
    "company_website": "Employer website",
    "deadline": "Application deadline",
}


def parse_ad(html: str, card: dict | None = None) -> dict:
    """Fields needed to judge an ad. Contact cards, phone lists and shop emails are deliberately ignored."""
    card = card or {}
    data = _initial_data(html)
    ad = ((((data or {}).get("adDetail") or {}).get("data") or {}).get("ad")) or {}
    if ad:
        props = {p.get("key"): str(p.get("value", "")) for p in ad.get("properties") or [] if p.get("key")}
        loc = ad.get("location") or {}
        area = ad.get("area") or {}
        names = [n for n in (loc.get("name"), area.get("name")) if n]
        if len(names) == 2 and names[1] in names[0]:
            names = names[:1]
        location = ", ".join(names)
        auth, member = bool(ad.get("isAuthDealer")), bool(ad.get("isMember"))
        out = {
            "id": ad.get("id") or card.get("id"),
            "slug": ad.get("slug") or card.get("slug"),
            "title": ad.get("title") or card.get("title", ""),
            "description": ad.get("description", ""),
            "salary": props.get("salary") or card.get("salary", ""),
            "location": location or card.get("location", ""),
            "agency": ad.get("details") or card.get("agency") or props.get("company_name", ""),
            "auth_agent": auth,
            "member": member,
            "member_since": ad.get("memberSince", ""),
            "category": (ad.get("category") or {}).get("name", ""),
            "properties": {k: props[k] for k in _KEEP_PROPS if props.get(k)},
        }
    else:  # Fallback: meta tags only.
        og_title = re.search(r'<meta[^>]+property="og:title"[^>]+content="([^"]*)"', html)
        og_desc = re.search(r'<meta[^>]+(?:property="og:description"|name="description")[^>]+content="([^"]*)"', html)
        out = {
            "id": card.get("id"),
            "slug": card.get("slug"),
            "title": _text(og_title.group(1)) if og_title else card.get("title", ""),
            "description": _text(og_desc.group(1)) if og_desc else "",
            "salary": card.get("salary", ""),
            "location": card.get("location", ""),
            "agency": card.get("agency", ""),
            "auth_agent": card.get("auth_agent", False),
            "member": card.get("member", False),
            "member_since": "",
            "category": "",
            "properties": {},
        }
    out["poster_type"] = _poster_type(out["auth_agent"], out["member"])
    return out


def build_scan_text(ad: dict) -> str:
    """One plain-text block per ad, PII scrubbed, in the shape the model saw in training (title + body)."""
    lines = [ad.get("title", "")]
    meta = [
        ("Category", ad.get("category")),
        ("Posted by", ad.get("agency")),
        ("Poster type", ad.get("poster_type")),
        ("Member since", ad.get("member_since")),
        ("Location", ad.get("location")),
    ]
    props = dict(ad.get("properties") or {})
    if ad.get("salary"):
        props.setdefault("salary", ad["salary"])
    meta += [(label, props.get(key)) for key, label in _KEEP_PROPS.items()]
    lines += [f"{label}: {value}" for label, value in meta if value]
    lines += ["", ad.get("description", "")]
    return scrub_pii("\n".join(lines).strip())[:MAX_TEXT]


# ------------------------------------------------------------------------------------------- seen store


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SeenStore:
    """logs/collector_seen.json: {ad_id: {url, text, first_seen, last_seen, scanned_at}}. Nothing else."""

    def __init__(self, path: Path = SEEN_PATH):
        self.path = Path(path)
        try:
            self.ads: dict = json.loads(self.path.read_text(encoding="utf-8")).get("ads", {})
        except (FileNotFoundError, json.JSONDecodeError):
            self.ads = {}

    def __contains__(self, ad_id: str) -> bool:
        return ad_id in self.ads

    def touch(self, ad_id: str) -> None:
        self.ads[ad_id]["last_seen"] = _now()

    def add(self, ad_id: str, url: str, text: str) -> None:
        now = _now()
        self.ads[ad_id] = {"url": url, "text": text, "first_seen": now, "last_seen": now, "scanned_at": now}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"ads": self.ads}, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)


# ------------------------------------------------------------------------------------------- scanning


def in_process_scanner():
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from safepassage import service

    def scan(text: str, url: str) -> dict:
        r = service.scan_text(text, source="collector", url=url)
        return {"risk": r.risk.label, "needs_review": r.needs_review}

    return scan


def api_scanner(base_url: str, session=None):
    session = session or requests.Session()
    endpoint = base_url.rstrip("/") + "/api/v1/scans/text"

    def scan(text: str, url: str) -> dict:
        resp = session.post(endpoint, json={"text": text, "url": url}, timeout=60)
        resp.raise_for_status()
        body = resp.json()
        return {"risk": body["risk"]["label"], "needs_review": body.get("needs_review", False)}

    return scan


# ------------------------------------------------------------------------------------------- the run


@dataclass
class RunStats:
    listed: int = 0
    new: int = 0
    scanned: int = 0
    review: int = 0
    already_seen: int = 0
    robots_skipped: int = 0
    errors: int = 0
    fetched: int = 0
    risk: Counter = field(default_factory=Counter)
    stopped: str = ""

    def summary(self) -> str:
        mix = ", ".join(f"{k} {self.risk.get(k, 0)}" for k in ("low", "medium", "high"))
        line = (
            f"[collector {datetime.now().strftime('%H:%M:%S')}] fetched {self.fetched} pages | "
            f"listed {self.listed} | new {self.new} | scanned {self.scanned} | risk: {mix} | "
            f"to human review {self.review}"
        )
        extra = []
        if self.already_seen:
            extra.append(f"already seen {self.already_seen}")
        if self.robots_skipped:
            extra.append(f"skipped by robots.txt {self.robots_skipped}")
        if self.errors:
            extra.append(f"errors {self.errors}")
        if self.stopped:
            extra.append(f"STOPPED: {self.stopped}")
        return line + (f" ({'; '.join(extra)})" if extra else "")


def run_once(source, scan, seen: SeenStore, categories: list[str], max_ads: int = 50, max_pages: int = 3,
             log=print) -> RunStats:
    stats = RunStats()
    try:
        robots = source.load_robots()
        for category in categories:
            page = 1
            while stats.new < max_ads and page <= max_pages:
                url = listing_url(category, page)
                if not robots.allowed(url):
                    stats.robots_skipped += 1
                    log(f"  robots.txt disallows {url}; skipping")
                    break
                try:
                    cards, pagination = parse_listing(source.listing(category, page))
                except LookupError as exc:
                    stats.errors += page == 1
                    if page == 1:
                        log(f"  {exc}")
                    break
                if not cards:
                    break
                stats.listed += len(cards)
                for card in cards:
                    if stats.new >= max_ads:
                        break
                    if card["id"] in seen:
                        seen.touch(card["id"])
                        stats.already_seen += 1
                        continue
                    if not robots.allowed(card["url"]):
                        stats.robots_skipped += 1
                        continue
                    try:
                        ad = parse_ad(source.ad(card["slug"]), card)
                    except LookupError as exc:
                        stats.errors += 1
                        log(f"  {exc}")
                        continue
                    text = build_scan_text(ad)
                    stats.new += 1
                    try:
                        result = scan(text, card["url"])
                    except Exception as exc:  # noqa: BLE001 - one bad scan should not end the run
                        stats.errors += 1
                        log(f"  scan failed for {card['url']}: {exc}")
                        continue
                    seen.add(card["id"], card["url"], text)
                    stats.scanned += 1
                    stats.risk[result["risk"]] += 1
                    stats.review += bool(result["needs_review"])
                total, size = pagination.get("total"), pagination.get("pageSize")
                if total and size and page * size >= total:
                    break
                page += 1
    except StopCollection as exc:
        stats.stopped = str(exc)
    finally:
        seen.save()
        stats.fetched = source.fetched
    return stats


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Scan public ikman.lk job ads with SafePassage.")
    ap.add_argument("--category", action="append", metavar="SLUG",
                    help=f"category slug, repeatable (default overseas-jobs; local jobs: 'jobs'). Known: {KNOWN_CATEGORIES}")
    ap.add_argument("--max-ads", type=int, default=50, help="max new ads to scan per run (default 50)")
    ap.add_argument("--max-pages", type=int, default=3, help="max listing pages per category per run (default 3)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="run once and exit (default)")
    mode.add_argument("--interval-min", type=float, help="repeat every N minutes until stopped")
    ap.add_argument("--delay", type=float, default=2.0, help=f"seconds between requests (min {MIN_DELAY})")
    ap.add_argument("--api", metavar="URL", help="POST to URL/api/v1/scans/text instead of scanning in-process")
    ap.add_argument("--fixtures", metavar="DIR", type=Path, help="parse saved HTML from DIR instead of fetching")
    ap.add_argument("--seen", type=Path, default=SEEN_PATH, help="dedupe store (default logs/collector_seen.json)")
    ap.add_argument("--cache-raw", action="store_true", help="keep raw HTML in the gitignored data/raw/ikman/")
    args = ap.parse_args(argv)
    args.category = args.category or ["overseas-jobs"]
    for slug in args.category:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug):
            ap.error(f"bad category slug: {slug!r}")
    if args.delay < MIN_DELAY:
        print(f"--delay raised to the minimum {MIN_DELAY}s", file=sys.stderr)
        args.delay = MIN_DELAY
    if args.max_ads < 1:
        ap.error("--max-ads must be at least 1")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    scan = api_scanner(args.api) if args.api else in_process_scanner()
    while True:
        source = FixtureSource(args.fixtures) if args.fixtures else LiveSource(args.delay, args.cache_raw)
        stats = run_once(source, scan, SeenStore(args.seen), args.category, args.max_ads, args.max_pages)
        print(stats.summary(), flush=True)
        if stats.stopped and not args.fixtures:
            return 2
        if not args.interval_min:
            return 0
        try:
            time.sleep(args.interval_min * 60)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
