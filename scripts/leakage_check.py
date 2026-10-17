"""Report train/test leakage and label shortcuts in data/processed/.

  1. Near-duplicate pairs (char n-gram TF-IDF cosine > 0.9) between train and val / test / shifted
     (and data/handwritten.csv if present), plus rows that share a train text skeleton and the share of
     test lines (sentences) that also occur verbatim in train.
  2. Per-class frequency of placeholder tokens: phone placeholders (07X XXX XXXX, 011 XXX XXXX), fictional agency
     names, licence numbers, currency formats, example.lk e-mail addresses.
  3. Fictional agency names and licence numbers that appear in only one class (a perfect label shortcut).

Run after scripts/make_dataset.py:  python scripts/leakage_check.py [--json reports/leakage.json]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import generate_synthetic as gs  # noqa: E402
from make_dataset import NEAR_DUP, NearDupIndex, skeleton  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
HANDWRITTEN = ROOT / "data" / "handwritten.csv"

TOKENS = {
    "mobile_placeholder (07X XXX XXXX)": r"07X XXX XXXX",
    "landline_placeholder (011 XXX XXXX)": r"011 XXX XXXX",
    "any_fictional_agency": "|".join(re.escape(a) for a in gs.AGENCY),
    "licence_number": r"(?:licen[cs]e|බලපත්‍ර|உரிமம்)(?:\s*no\.?)?\s*\d{3,6}",
    "currency_Rs": r"\bRs\.\s?\d",
    "currency_foreign_code": r"\b(?:USD|EUR|SAR|QAR|AED|KWD|JPY|CAD|AUD|THB|RUB|GEL)\s?\d",
    "email_example.lk": r"@example\.lk",
    "emoji_prefix": r"^(?:✈️|📢|🔥|⭐|🇱🇰)",
}
LICENCE_NUM = re.compile(r"(?:licen[cs]e|බලපත්‍ර|உரிமம்)(?:\s*no\.?)?\s*(\d{3,6})", re.IGNORECASE)


def load(name: str) -> pd.DataFrame | None:
    path = HANDWRITTEN if name == "handwritten" else PROCESSED / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def line_overlap(train: pd.DataFrame, other: pd.DataFrame) -> float:
    """Share of `other`'s lines / sentences that occur verbatim (after skeleton normalisation) in train."""
    split = lambda s: [x for x in re.split(r"\n| \| |(?<=[.!?])\s+", s) if len(x.strip()) > 3]  # noqa: E731
    train_lines = {skeleton(x) for t in train["text"] for x in split(t)}
    lines = [skeleton(x) for t in other["text"] for x in split(t)]
    return round(sum(x in train_lines for x in lines) / max(len(lines), 1), 3)


def leakage(train: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> dict:
    """Near-duplicate and template overlap of each split against `train`, overall and per source
    (EMSCAD rows are compared with EMSCAD train rows, synthetic rows with synthetic train rows)."""
    train = train.reset_index(drop=True)
    index, train_skel = NearDupIndex(train["text"]), set(train["text"].map(skeleton))
    out = {}
    for name, df in splits.items():
        sec = {}
        sources = ["all"] + (sorted(df["source"].unique()) if "source" in df else [])
        for src in sources:
            part = df if src == "all" else df[df["source"] == src]
            fam = "synthetic" if src.startswith("synthetic") else src
            rows = None if src == "all" else np.flatnonzero(train["source"].str.startswith(fam))
            sim, pairs = index.query(part["text"], rows=rows)
            sec[src] = {
                "n": len(part),
                "near_duplicate_pairs": pairs,
                "rows_with_near_duplicate": int((sim > NEAR_DUP).sum()),
                "rows_sharing_train_skeleton": int(part["text"].map(skeleton).isin(train_skel).sum()),
                "max_similarity_median": round(float(np.median(sim)), 3) if len(sim) else None,
                "line_overlap_with_train": line_overlap(train if rows is None else train.iloc[rows], part),
            }
        out[name] = sec
    return out


def token_frequency(df: pd.DataFrame) -> dict:
    out = {}
    for src, part in [("all", df)] + [(s, g) for s, g in df.groupby("source")]:
        out[src] = {tok: {lab: round(float(g["text"].str.contains(rx, regex=True).mean()), 3)
                          for lab, g in part.groupby("label")} for tok, rx in TOKENS.items()}
    return out


def single_class_names(df: pd.DataFrame) -> dict:
    agencies = {}
    for a in gs.AGENCY:
        hit = df[df["text"].str.contains(re.escape(a), regex=True)]
        agencies[a] = hit["label"].value_counts().to_dict()
    licences = {}
    for lab, txt in zip(df["label"], df["text"]):
        for num in LICENCE_NUM.findall(txt):
            licences.setdefault(num, {}).setdefault(lab, 0)
            licences[num][lab] += 1
    one = lambda d: {k: v for k, v in d.items() if len(v) == 1}  # noqa: E731
    # Legit ads always quote their own agency's licence; a mismatch is a (generator-made) scam cue.
    syn = df[df["source"] == "synthetic"]
    mismatch = {}
    for lab, g in syn.groupby("label"):
        n = bad = 0
        for txt in g["text"]:
            ag = next((a for a in gs.AGENCY if a in txt), None)
            nums = LICENCE_NUM.findall(txt)
            if ag and nums:
                n += 1
                bad += any(x != gs.AGENCY_LICENCE[ag] for x in nums)
        mismatch[lab] = round(bad / n, 3) if n else None
    return {"agency_counts_by_label": agencies, "agencies_in_one_class_only": one(agencies),
            "licence_numbers_in_one_class_only": one(licences),
            "agency_licence_mismatch_rate_by_label": mismatch}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, help="also write the full report to this path")
    args = ap.parse_args()

    train = load("train")
    if train is None:
        sys.exit("data/processed/train.csv not found: run scripts/make_dataset.py first")
    val = load("val")
    pool = pd.concat([train, val]) if val is not None else train  # everything a model could have seen
    splits = {n: df for n in ["val", "test", "shifted", "handwritten"] if (df := load(n)) is not None}
    if "val" in splits:  # val is checked against train alone
        splits_pool = {k: v for k, v in splits.items() if k != "val"}
        report = {"near_duplicates": {"val": leakage(train, {"val": splits["val"]})["val"],
                                      **leakage(pool, splits_pool)}}
    else:
        report = {"near_duplicates": leakage(pool, splits)}
    report["token_frequency_train"] = token_frequency(train)
    report["single_class_shortcuts_train"] = single_class_names(pool)

    print(f"Near-duplicates (cosine > {NEAR_DUP}) against train:")
    for name, sec in report["near_duplicates"].items():
        for src, s in sec.items():
            if isinstance(s, dict):
                print(f"  {name:11s} {src:18s} n={s['n']:5d}  pairs={s['near_duplicate_pairs']:6d}  "
                      f"rows={s['rows_with_near_duplicate']:5d}  same_skeleton={s['rows_sharing_train_skeleton']:5d}  "
                      f"median_max_sim={s['max_similarity_median']}  line_overlap={s['line_overlap_with_train']}")
    print("\nPlaceholder token frequency in train (share of rows, scam vs legit):")
    for src, toks in report["token_frequency_train"].items():
        print(f"  [{src}]")
        for tok, by in toks.items():
            if any(by.values()):
                print(f"    {tok:38s} " + "  ".join(f"{k}={v:.3f}" for k, v in sorted(by.items())))
    sc = report["single_class_shortcuts_train"]
    print("\nFictional agencies seen in one class only:", sc["agencies_in_one_class_only"] or "none")
    print("Licence numbers seen in one class only:", sc["licence_numbers_in_one_class_only"] or "none")
    print("Agency/licence mismatch rate (synthetic):", sc["agency_licence_mismatch_rate_by_label"])
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
