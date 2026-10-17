"""Build the verification data from public SLBFE sources (polite crawl: identified user agent, 1 request/second).

1. SLBFE job classification (public, no login): https://services.slbfe.lk/Info/JobClassification
   -> data/kb/slbfe_job_classification.csv  (code, job, manpower_level). Job titles only, no personal data.
2. Agency registry -> data/processed/registry.csv (name, licence_no, status, source)
   SLBFE does not publish its licensed-agency list in a public, machine-readable form (the job bank and agency search
   need an account, which we do not use). So the registry is:
     --agencies FILE   a CSV you obtained legitimately (e.g. typed from an official notice). Only the agency name and
                       licence number are kept; every other column, and any phone number or email, is dropped.
     otherwise         data/kb/registry_demo.csv: FICTIONAL agencies for the demo, labelled as such.
"""

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "data" / "kb"
REGISTRY = ROOT / "data" / "processed" / "registry.csv"
CLASSIFICATION_URL = "https://services.slbfe.lk/Info/GetJobsByManpowerLevel"
HEADERS = {"User-Agent": "SafePassageLK-research/0.1 (TensorForge 2.0 student project; polite crawl)"}
_PHONE_EMAIL = re.compile(r"(\+?\d[\d\s()-]{7,}\d)|([\w.+-]+@[\w-]+\.[\w.]+)")


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
    ap.add_argument("--skip-fetch", action="store_true", help="do not call the SLBFE website")
    args = ap.parse_args()

    if not args.skip_fetch:
        classification = fetch_classification()
        classification.to_csv(KB / "slbfe_job_classification.csv", index=False)
        print(f"Job classification: {len(classification)} jobs -> {KB / 'slbfe_job_classification.csv'}")
        time.sleep(1)

    registry = clean_agencies(args.agencies) if args.agencies else pd.read_csv(KB / "registry_demo.csv")
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(REGISTRY, index=False)
    print(f"Registry: {len(registry)} agencies ({registry['source'].iloc[0]}) -> {REGISTRY}")


if __name__ == "__main__":
    main()
