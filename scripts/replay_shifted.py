"""Drift demo: stream familiar offers, then shifted ones, and watch the drift alert fire.

    PYTHONPATH=src python scripts/replay_shifted.py                 # in-process, writes to logs/
    PYTHONPATH=src python scripts/replay_shifted.py --fresh         # archive old logs first (clean demo)
    python scripts/replay_shifted.py --api http://localhost:8000    # through the running REST API

Uses data/processed/test.csv (reference) and data/processed/shifted.csv when they exist;
otherwise falls back to the built-in synthetic samples below. All samples are fictional.
"""

import argparse
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from safepassage import config

# Familiar patterns: clearly legit offers and classic Gulf-job scams.
DEMO_REFERENCE = [
    "Accountant vacancy at a Colombo head office. Apply through our careers page with your CV.",
    "SLBFE approved job order: hotel cook, Qatar. Salary QAR 1800, food and accommodation provided. Licence No. 1234.",
    "Graduate trainee programme at a Kandy bank branch. Interviews held at our office.",
    "Conditional offer letter from the university. Pay the tuition deposit through the university's official portal.",
    "URGENT housemaid jobs Kuwait! 100% visa guaranteed. Pay registration fee now, only 5 seats left. WhatsApp us.",
    "Driver jobs Saudi, no experience needed! Guaranteed visa. Pay processing fee first to my personal account. Telegram us.",
    "Part-time job: like YouTube videos, earn daily commission! Send your NIC copy and pay registration fee now on WhatsApp.",
]

# New tactics the model has not seen: new schemes, destinations and pretexts.
DEMO_SHIFTED = [
    "SL-UDI digital ID field officer vacancies. Government approved. Submit NIC copy to apply. Contact on Viber.",
    "Canada SDS is back for Sri Lankan students! Guaranteed admission through our partner colleges, contact us on WhatsApp.",
    "Romania factory work permits, no IELTS needed. Seats filling fast, message us on Telegram.",
    "Earn USDT daily with prepaid task bundles from home. Join our Telegram team.",
    "Japan SSW caregiver jobs. Deposit payment due before the agency interview, Viber only.",
    "Australia new skilled visa lottery 2026: apply with passport photo upload today via WhatsApp.",
]


def load(path: Path, fallback: list[str], label: str) -> list[str]:
    if path.exists():
        texts = pd.read_csv(path)["text"].dropna().astype(str).tolist()
        print(f"{label}: {len(texts)} rows from {path}")
        return texts
    print(f"{label}: {path} not found, using {len(fallback)} built-in demo samples")
    return fallback


def archive_logs() -> None:
    files = [p for p in config.LOGS_DIR.glob("*.json*") if p.is_file()]
    if not files:
        return
    dest = config.LOGS_DIR / f"archive-{datetime.now():%Y%m%d-%H%M%S}"
    dest.mkdir(parents=True)
    for p in files:
        shutil.move(str(p), dest / p.name)
    print(f"Archived {len(files)} log files to {dest}")


def make_scanner(api: str | None):
    if api:
        def scan(text):
            requests.post(f"{api}/api/v1/scans/text", json={"text": text}, timeout=30).raise_for_status()

        def status():
            return requests.get(f"{api}/api/v1/drift", timeout=30).json()
        return scan, status

    from safepassage.response import check_drift, drift_active
    from safepassage.service import scan_text

    def status():
        return {**check_drift(), "drift": drift_active()}
    return scan_text, status


def run_phase(name, texts, n, scan, status, every, delay):
    print(f"\n== {name}: {n} scans ==")
    for i in range(1, n + 1):
        scan(random.choice(texts))
        if i % every == 0 or i == n:
            s = status()
            flag = "DRIFT" if s["drift"] else "ok"
            print(f"  {i:>4}  {flag:<5}  {s['signals']}")
        time.sleep(delay)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, default=config.DATA_DIR / "processed" / "test.csv")
    ap.add_argument("--shifted", type=Path, default=config.DATA_DIR / "processed" / "shifted.csv")
    ap.add_argument("-n", type=int, default=40, help="scans per phase")
    ap.add_argument("--every", type=int, default=10, help="print drift status every N scans")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between scans (for a live dashboard)")
    ap.add_argument("--api", help="base URL of the running REST API; default runs in-process")
    ap.add_argument("--fresh", action="store_true", help="archive existing logs before replaying")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    if args.fresh and not args.api:
        archive_logs()
    reference = load(args.reference, DEMO_REFERENCE, "reference")
    shifted = load(args.shifted, DEMO_SHIFTED, "shifted")
    scan, status = make_scanner(args.api)

    run_phase("Phase 1 - familiar offers", reference, args.n, scan, status, args.every, args.delay)
    run_phase("Phase 2 - shifted tactics", shifted, args.n, scan, status, args.every, args.delay)

    final = status()
    print(f"\nResult: {'drift DETECTED' if final['drift'] else 'no drift detected'} - {final['signals']}")
    if not args.api:
        from safepassage.monitoring import read_alerts
        for a in read_alerts()[-3:]:
            print(f"  alert: {a['kind']} at {a['ts']}")


if __name__ == "__main__":
    main()
