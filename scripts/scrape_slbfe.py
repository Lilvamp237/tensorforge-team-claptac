"""Build the verification data from public SLBFE sources (polite crawl: identified user agent, >= 1 request/second).

1. SLBFE job classification (public, no login): https://services.slbfe.lk/Info/JobClassification
   -> data/kb/slbfe_job_classification.csv  (code, job, manpower_level). Job titles only, no personal data.
2. --jobbank: the public SLBFE agency lists and Online Job Bank (no login needed):
     https://applications.slbfe.lk/feb/la/la_detailsvalid.asp     valid licensed agencies
     https://applications.slbfe.lk/feb/la/la_suspended.asp        suspended / cancelled agencies
     https://applications.slbfe.lk/feb/feo/newjob/joblist.asp     agencies with job orders approved in the last 90 days
     https://applications.slbfe.lk/jobbank/jsearchdisplay_an_m.asp?an=NNNN   approved job orders of one agency
   -> data/kb/slbfe_registry.csv    name, licence_no, status, source, retrieved
   -> data/kb/slbfe_job_orders.csv  licence_no, country, job_title, vacancies_approved, min_salary, max_fee_lkr,
                                    approved_date, valid_until, source
   Only official business information is kept: agency name, licence number, licence status and the job-order table.
   Addresses, telephone and fax numbers, emails and any personal names are never parsed into the CSVs.
   Raw HTML is cached under data/raw/slbfe_jobbank/ (gitignored) and reused on later runs, so a re-run costs no
   requests. --max-requests caps the number of network requests (default 580).
3. Agency registry -> data/processed/registry.csv (name, licence_no, status, source)
     --agencies FILE   a CSV you obtained legitimately. Only the agency name and licence number are kept; every other
                       column, and any phone number or email, is dropped.
     otherwise         data/kb/slbfe_registry.csv if present, else data/kb/registry_demo.csv (FICTIONAL agencies).
"""

import argparse
import re
import sys
import time
from datetime import date, datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "data" / "kb"
RAW = ROOT / "data" / "raw" / "slbfe_jobbank"
REGISTRY = ROOT / "data" / "processed" / "registry.csv"
CLASSIFICATION_URL = "https://services.slbfe.lk/Info/GetJobsByManpowerLevel"
HEADERS = {"User-Agent": "SafePassageLK-research/0.1 (hackathon; contact via GitHub repo)"}
_PHONE_EMAIL = re.compile(r"(\+?\d[\d\s()-]{7,}\d)|([\w.+-]+@[\w-]+\.[\w.]+)")

LA = "https://applications.slbfe.lk/feb/la/"
VALID_URL = LA + "la_detailsvalid.asp"
SUSPENDED_URL = LA + "la_suspended.asp"
RECENT_URL = "https://applications.slbfe.lk/feb/feo/newjob/joblist.asp?timeperiod=90&period=3"
AGENCY_URL = "https://applications.slbfe.lk/jobbank/jsearchdisplay_an_m.asp?an={an}"
DELAY_S = 1.2


class Blocked(RuntimeError):
    """The site asked for a login, refused us, or kept failing: stop instead of hammering it."""


# ---------------------------------------------------------------- polite fetching with a cache
class Fetcher:
    def __init__(self, max_requests: int, cache: Path = RAW):
        self.max_requests, self.cache, self.requests = max_requests, cache, 0
        self.cache.mkdir(parents=True, exist_ok=True)
        self._last = 0.0
        self._failures = 0

    def get(self, url: str, name: str) -> str | None:
        path = self.cache / f"{name}.html"
        if path.exists():
            return path.read_text(encoding="utf-8")
        if self.requests >= self.max_requests:
            return None
        for attempt in range(3):
            time.sleep(max(0.0, DELAY_S - (time.monotonic() - self._last)))
            self.requests += 1
            self._last = time.monotonic()
            try:
                resp = requests.get(url, headers=HEADERS, timeout=60)
            except requests.RequestException as e:
                err = str(e)
            else:
                if resp.status_code == 200:
                    html = resp.content.decode(resp.encoding or "iso-8859-1", errors="replace")
                    if re.search(r'type=["\']?password|logon|please log ?in', html, re.IGNORECASE):
                        raise Blocked(f"{url} asks for a login")
                    self._failures = 0
                    path.write_text(html, encoding="utf-8")
                    return html
                if resp.status_code in (401, 403, 429):
                    raise Blocked(f"{url} returned HTTP {resp.status_code}")
                err = f"HTTP {resp.status_code}"
            if self.requests >= self.max_requests:
                break
            time.sleep(2 ** (attempt + 1))
        self._failures += 1
        print(f"  failed: {url} ({err})", file=sys.stderr)
        if self._failures >= 5:
            raise Blocked(f"5 pages in a row failed, last {url} ({err})")
        return None


# ---------------------------------------------------------------- HTML table parsing (stdlib only)
class _Rows(HTMLParser):
    """Collect table rows as (tr class, [cell text]); copes with the unclosed <tr>/<td> tags these pages use."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self._cell = [], None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._close_cell()
            self.rows.append((dict(attrs).get("class", ""), []))
        elif tag in ("td", "th") and self.rows:
            self._close_cell()
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th", "tr", "table"):
            self._close_cell()

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def _close_cell(self):
        if self._cell is not None and self.rows:
            self.rows[-1][1].append(_clean("".join(self._cell)))
        self._cell = None


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", unescape(text).replace("\xa0", " ")).strip()


def table_rows(html: str) -> list[tuple[str, list[str]]]:
    parser = _Rows()
    parser.feed(html)
    return parser.rows


def _date(text: str) -> str:
    text = _clean(text)
    for fmt in ("%B %d, %Y", "%d-%B-%Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    return ""


def _lkr(text: str) -> int | None:
    m = re.search(r"Rs\.?\s*([\d,]+)", text)
    return int(m.group(1).replace(",", "")) if m else None


def parse_agency_list(html: str, url: str, retrieved: str) -> pd.DataFrame:
    """Valid / suspended agency lists: keep name, licence number and status only (address, phones, emails dropped)."""
    out = []
    for _, cells in table_rows(html):
        if len(cells) < 3 or not re.fullmatch(r"\d+\.", cells[0]) or not re.fullmatch(r"\d{3,5}", cells[2]):
            continue
        tail = " ".join(cells[3:]).lower()
        status = "cancelled" if "cancelled" in tail else "suspended" if "suspended" in tail else "licensed"
        out.append({"name": _PHONE_EMAIL.sub("", cells[1]).strip(), "licence_no": cells[2], "status": status,
                    "source": url, "retrieved": retrieved})
    return pd.DataFrame(out, columns=["name", "licence_no", "status", "source", "retrieved"])


def parse_recent_agencies(html: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"actjobsjsearchdisplay\.asp\?an=(\d{3,5})", html)))


def parse_agency_page(html: str, url: str) -> tuple[str, str, list[dict]]:
    """One agency's job bank page -> (name, licence_no, job orders). The Tel/Fax/Email line is never read."""
    m = re.search(r"Available vacancies for</i>(.*?)\[Reg\.#\s*(\d+)\s*\]", html, re.S | re.I)
    if not m:
        return "", "", []
    name, licence = _clean(re.sub(r"<[^>]+>", " ", m.group(1))), m.group(2)
    orders, country = [], ""
    for cls, cells in table_rows(html):
        if "country-row" in cls and cells:
            country = cells[0]
        elif "job-row" in cls and len(cells) >= 13:
            fee = _lkr(cells[10])
            if fee is None:
                continue
            orders.append({"licence_no": licence, "country": country, "job_title": cells[0],
                           "vacancies_approved": cells[1], "min_salary": cells[4], "max_fee_lkr": fee,
                           "approved_date": _date(cells[11]), "valid_until": _date(cells[12]), "source": url})
    return name, licence, orders


def crawl_jobbank(max_requests: int) -> None:
    fetch = Fetcher(max_requests)
    retrieved = date.today().isoformat()
    try:
        lists = []
        for url, name in [(VALID_URL, "la_detailsvalid"), (SUSPENDED_URL, "la_suspended")]:
            html = fetch.get(url, name)
            if html is None:
                raise Blocked(f"could not read {url}")
            lists.append(parse_agency_list(html, url, retrieved))
        registry = pd.concat(lists, ignore_index=True)
        # A licence number on the valid list wins over an older cancelled entry with the same number.
        registry["rank"] = registry["status"].map({"licensed": 0, "suspended": 1, "cancelled": 2})
        registry = (registry.sort_values(["rank", "name"]).drop_duplicates("licence_no")
                    .drop(columns="rank").sort_values("name"))

        recent = fetch.get(RECENT_URL, "joblist_90d") or ""
        ids = parse_recent_agencies(recent)
        print(f"Agencies: {len(registry)} ({(registry['status'] == 'licensed').sum()} licensed); "
              f"{len(ids)} with job orders approved in the last 90 days")

        orders, no_marker = [], 0
        for i, an in enumerate(ids, 1):
            url = AGENCY_URL.format(an=an)
            html = fetch.get(url, f"an_{an}")
            if html is None:
                if fetch.requests >= fetch.max_requests:
                    print(f"  request cap reached after {i - 1} agency pages")
                    break
                continue
            _, licence, rows = parse_agency_page(html, url)
            no_marker = 0 if licence else no_marker + 1
            if no_marker >= 20:
                raise Blocked("20 agency pages in a row had no agency header (layout change or block page)")
            orders += rows
            if i % 50 == 0:
                print(f"  {i}/{len(ids)} agency pages, {len(orders)} job orders, {fetch.requests} requests")
    except Blocked as e:
        print(f"STOPPED: {e}. Nothing written; keep using the demo registry.", file=sys.stderr)
        sys.exit(2)

    cols = ["licence_no", "country", "job_title", "vacancies_approved", "min_salary", "max_fee_lkr",
            "approved_date", "valid_until", "source"]
    jobs = pd.DataFrame(orders, columns=cols).drop_duplicates()
    jobs = jobs.sort_values(["licence_no", "country", "job_title", "approved_date"])
    registry.to_csv(KB / "slbfe_registry.csv", index=False)
    jobs.to_csv(KB / "slbfe_job_orders.csv", index=False)
    print(f"Registry: {len(registry)} agencies -> {KB / 'slbfe_registry.csv'}")
    print(f"Job orders: {len(jobs)} from {jobs['licence_no'].nunique()} agencies -> {KB / 'slbfe_job_orders.csv'}")
    print(f"Network requests this run: {fetch.requests} (cache: {RAW})")


# ---------------------------------------------------------------- job classification + legacy registry
def fetch_classification() -> pd.DataFrame:
    resp = requests.get(CLASSIFICATION_URL, params={"level": 1}, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    rows = [{"code": r["Code"].strip(), "job": r["JOB"].strip(), "manpower_level": r["MLevel"].strip()}
            for r in resp.json()]
    return pd.DataFrame(rows).drop_duplicates("code").sort_values("job")


def clean_agencies(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    name_col = next(cols[c] for c in cols if "name" in c or "agency" in c)
    lic_col = next((cols[c] for c in cols if "licen" in c or c in {"lic", "lic_no"}), None)
    out = pd.DataFrame({
        "name": df[name_col].astype(str).str.replace(_PHONE_EMAIL, "", regex=True).str.strip(),
        "licence_no": df[lic_col].astype(str).str.extract(r"(\d{3,5})")[0] if lic_col else None,
    })
    out["status"] = "licensed"
    out["source"] = f"imported:{path.name}"
    return out[out["name"] != ""].drop_duplicates()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agencies", type=Path, help="CSV of licensed agencies obtained from an official source")
    ap.add_argument("--jobbank", action="store_true", help="crawl the public SLBFE agency lists and job bank")
    ap.add_argument("--max-requests", type=int, default=580, help="cap on network requests for --jobbank")
    ap.add_argument("--skip-fetch", action="store_true", help="do not fetch the job classification")
    args = ap.parse_args()

    if not args.skip_fetch:
        classification = fetch_classification()
        classification.to_csv(KB / "slbfe_job_classification.csv", index=False)
        print(f"Job classification: {len(classification)} jobs -> {KB / 'slbfe_job_classification.csv'}")
        time.sleep(1)

    if args.jobbank:
        crawl_jobbank(args.max_requests)

    if args.agencies:
        registry = clean_agencies(args.agencies)
    elif (KB / "slbfe_registry.csv").exists():
        registry = pd.read_csv(KB / "slbfe_registry.csv", dtype=str)
    else:
        registry = pd.read_csv(KB / "registry_demo.csv", dtype=str)
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(REGISTRY, index=False)
    print(f"Registry: {len(registry)} agencies ({registry['source'].iloc[0]}) -> {REGISTRY}")


if __name__ == "__main__":
    main()
