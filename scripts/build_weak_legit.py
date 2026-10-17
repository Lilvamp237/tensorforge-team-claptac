"""Weakly label real Sri Lankan job ads collected from ikman.lk as legitimate training data.

    PYTHONPATH=src python scripts/collect_ikman.py --category overseas-jobs --category jobs --max-ads 240 \\
        --max-pages 8 --once --seen data/raw/ikman_seen.json
    PYTHONPATH=src python scripts/build_weak_legit.py            # -> data/processed/weak_legit.csv (gitignored)

Why: the model has seen almost no real Sri Lankan legitimate ads, so it flags many of them (60% on the
real-world test set). An ad is labelled legit only on strong outside evidence, never on the model's own score:
  overseas: SLBFE licence number that is licensed in the real registry, or ikman's AUTH AGENT badge;
  local:    posted by an ikman member (verified account);
  and in both cases no severe red flag (personal account, ID documents, task job, crypto, secrecy, no SLBFE).
Everything else stays unlabelled. Collector metadata lines (poster type, licence, member since...) are removed
from the training text so the model can't learn them as a shortcut. Ads that are near-copies of the real-world,
held-out or shifted test sets are dropped, so those sets stay clean. Real ad text is never committed.
"""

import argparse
import json
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from safepassage.config import DATA_DIR, KB_DIR
from safepassage.extract import classify_category, detect_language
from safepassage.features import find_red_flags

SEEN = DATA_DIR / "raw" / "ikman_seen.json"
OUT = DATA_DIR / "processed" / "weak_legit.csv"
SEVERE = {"personal_account", "id_document_request", "task_scam", "crypto_payment", "secrecy", "no_slbfe"}
META = re.compile(r"^(Category|Posted by|Poster type|Member since|Location|Country|Employer|SLBFE licence number|"
                  r"Salary|Job type|Industry|Employer website|Application deadline): ?(.*)$")
NEAR_DUP = 0.9


def parse(text: str) -> tuple[dict, str]:
    """Split collector text into metadata and the ad itself (title + description)."""
    meta, body = {}, []
    for line in text.splitlines():
        m = META.match(line.strip())
        if m:
            meta[m.group(1)] = m.group(2).strip()
        else:
            body.append(line)
    return meta, re.sub(r"\n{3,}", "\n\n", "\n".join(body)).strip()


def licensed_numbers() -> set[str]:
    reg = pd.read_csv(KB_DIR / "slbfe_registry.csv", dtype=str).fillna("")
    return {n.lstrip("0") for n in reg.loc[reg["status"] == "licensed", "licence_no"]}


def evidence(meta: dict, flags: list[str], licensed: set[str]) -> str | None:
    if SEVERE.intersection(flags):
        return None
    overseas = "overseas" in meta.get("Category", "").lower() or bool(meta.get("Country"))
    poster = meta.get("Poster type", "")
    lic = re.sub(r"\D", "", meta.get("SLBFE licence number", "")).lstrip("0")
    if overseas:
        if lic and lic in licensed:
            return f"licence {lic} licensed in SLBFE registry"
        if poster.startswith("Authorised agent"):
            return "ikman AUTH AGENT badge"
        return None
    return "ikman member account" if poster.startswith("Member") else None


def drop_near_copies(df: pd.DataFrame, protected: list[str]) -> tuple[pd.DataFrame, int]:
    if not protected or df.empty:
        return df, 0
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit(list(df["text"]) + protected)
    sim = cosine_similarity(vec.transform(df["text"]), vec.transform(protected)).max(axis=1)
    keep = sim <= NEAR_DUP
    return df[keep], int((~keep).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seen", default=str(SEEN))
    args = ap.parse_args()

    ads = json.loads(open(args.seen, encoding="utf-8").read())
    ads = ads.get("ads", ads)
    licensed, rows, reasons = licensed_numbers(), [], {}
    for ad in ads.values():
        meta, body = parse(ad["text"])
        why = evidence(meta, find_red_flags(ad["text"]), licensed)
        reasons[why or "unlabelled"] = reasons.get(why or "unlabelled", 0) + 1
        if why and len(body) >= 40:
            rows.append({"text": body, "label": "legit", "category": classify_category(body),
                         "language": detect_language(body), "source": "ikman_weak", "tactic": why})
    df = pd.DataFrame(rows).drop_duplicates("text")

    protected = []
    for path in [DATA_DIR / "handwritten.csv", DATA_DIR / "processed" / "test.csv", DATA_DIR / "processed" / "shifted.csv"]:
        if path.exists():
            protected += pd.read_csv(path)["text"].astype(str).tolist()
    df, dropped = drop_near_copies(df, protected)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"ads: {len(ads)} | evidence: {reasons}")
    print(f"weak legit rows: {len(df)} (dropped {dropped} near-copies of test sets) -> {OUT}")
    print(df.groupby(["category", "language"]).size().to_string())


if __name__ == "__main__":
    main()
