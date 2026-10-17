"""Build the train / validation / held-out test / shifted splits.

Sources
  EMSCAD (Employment Scam Aegean Dataset, 17,880 real job ads, 866 fraudulent), CC0 mirror on Hugging Face:
    https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction
  Synthetic Sri Lankan offers from scripts/generate_synthetic.py (run automatically if missing).

Outputs (data/processed/)
  train.csv    EMSCAD 80% + synthetic train pool, minus the validation rows
  val.csv      ~10% of the train pool, stratified by label and source and grouped by text skeleton.
               Used to pick thresholds, never to report final quality.
  test.csv     EMSCAD 20% (stratified) + synthetic held-out pool (reserved phrasing). Never used for training.
  shifted.csv  drift demo set (Tamil, new tactics, new destinations). Never used for training.
Columns: text, label (scam|legit), category, language, source, tactic

Leakage control
  The synthetic CSVs carry no template id, so rows are grouped by a normalised text skeleton: agency names,
  licence numbers, phone placeholders, amounts, places, job titles and digits are masked, then punctuation,
  emoji and case are dropped. A test / shifted row is removed when its skeleton also occurs in the train pool,
  or when it is a near-duplicate of a train row (character n-gram TF-IDF cosine > NEAR_DUP). The validation
  split keeps whole skeleton groups together, and validation rows that are near-duplicates of the remaining
  train rows go back to train. scripts/leakage_check.py reports what is left.
"""

import argparse
import re
import subprocess
import sys
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import generate_synthetic as gs  # noqa: E402  (agency names and slot values used for masking)

RAW = ROOT / "data" / "raw" / "fake_job_postings.csv"
SYN = ROOT / "data" / "synthetic"
OUT = ROOT / "data" / "processed"
EMSCAD_URL = ("https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction"
              "/resolve/main/fake_job_postings.csv")
COLUMNS = ["text", "label", "category", "language", "source", "tactic"]
MAX_CHARS = 1500  # keep EMSCAD ads close to the length of the short offers we score in production
NEAR_DUP = 0.9    # char n-gram TF-IDF cosine above which two texts count as near-duplicates
VAL_FOLDS = 10    # 1 of 10 folds -> ~10% validation

# ---------------------------------------------------------------- shortcut masking

CURRENCY = r"(?:Rs\.?|LKR|USD|EUR|GBP|SAR|QAR|AED|KWD|OMR|BHD|JPY|KRW|CAD|AUD|NZD|THB|RUB|GEL|MYR|SGD|\$|€|£)"
_AGENCY_RE = re.compile("|".join(re.escape(a) for a in sorted(gs.AGENCY, key=len, reverse=True))
                        + r"|\b(?:[A-Z][\w&.'-]*\s){1,4}\((?:Pvt|Private)\)\s(?:Ltd|Limited)\.?")
_LICENCE_RE = re.compile(r"((?:licen[cs]e|lic\.|registration|reg\.|බලපත්‍ර|உரிமம்)(?:\s*(?:no\.?|number|#|:))?\s*)"
                         r"\d{3,6}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?:\+94[\s-]?|\b0)\d[\dX]\s?[\dX]{3}\s?[\dX]{4}\b|\b0\d[\dX] XXX XXXX\b")
_AMOUNT_RE = re.compile(rf"{CURRENCY}\s?\d[\d,.]*(?:\s?(?:k|lakhs?|mn|million))?\b"
                        rf"|\b\d[\d,.]*\s?(?:{CURRENCY[3:-1]}|rupees|ரூபாய்|රුපියල්)"
                        r"|\b\d[\d,.]*\s?(?:லட்சம்|ஆயிரம்|ලක්ෂ)", re.IGNORECASE)
_SLOT_WORDS = sorted({w for cat in gs.SLOTS.values() for k in ("country", "job") for w in cat[k]}
                     | {w for ws in gs.NEW_DESTINATIONS.values() for w in ws}, key=len, reverse=True)
_SLOT_RE = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in _SLOT_WORDS) + r")\b", re.IGNORECASE)


def mask_shortcuts(text: str) -> str:
    """Replace the surface tokens a model can memorise instead of reading the offer:
    agency names, licence numbers, phone numbers / placeholders and money amounts."""
    text = _AGENCY_RE.sub("<AGENCY>", text)
    text = _LICENCE_RE.sub(r"\1<LICNO>", text)
    text = _PHONE_RE.sub("<PHONE>", text)
    text = _AMOUNT_RE.sub("<AMOUNT>", text)
    return text


def skeleton(text: str) -> str:
    """Normalised template skeleton: shortcuts, places, job titles and digits masked; punctuation, emoji, case and
    separators dropped. Two offers rendered from the same phrase combination share a skeleton."""
    t = _SLOT_RE.sub("<SLOT>", mask_shortcuts(text))
    t = re.sub(r"\d", "0", t.lower())
    t = "".join(" " if unicodedata.category(c)[0] in "PSZC" and c not in "<>" else c for c in t)
    return " ".join(t.split())


# ---------------------------------------------------------------- near-duplicates

def char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True, dtype=np.float32)


class NearDupIndex:
    """Char n-gram TF-IDF index over reference texts; vectorising is the slow part, so fit once, query often."""

    def __init__(self, reference: pd.Series):
        self.vec = char_vectorizer()
        self.R = self.vec.fit_transform(reference).tocsr()

    def query(self, others: pd.Series, rows=None, threshold: float = NEAR_DUP,
              chunk: int = 500) -> tuple[np.ndarray, int]:
        """Max cosine of each text in `others` to the reference rows (all, or the positional subset `rows`),
        and the number of (other, reference) pairs above `threshold`."""
        R = self.R if rows is None else self.R[rows]
        if R.shape[0] == 0 or len(others) == 0:
            return np.zeros(len(others)), 0
        O, RT = self.vec.transform(others), R.T.tocsr()
        best, pairs = np.zeros(O.shape[0]), 0
        for i in range(0, O.shape[0], chunk):
            S = (O[i:i + chunk] @ RT).toarray()
            best[i:i + chunk] = S.max(axis=1)
            pairs += int((S > threshold).sum())
        return best, pairs


def near_duplicates(reference: pd.Series, others: pd.Series, threshold: float = NEAR_DUP) -> tuple[np.ndarray, int]:
    """For each text in `others`: max cosine similarity to any `reference` text, plus the pair count above
    `threshold`."""
    if len(reference) == 0:
        return np.zeros(len(others)), 0
    return NearDupIndex(reference).query(others, threshold=threshold)


def drop_leaks(train_skel: set, index: NearDupIndex, other: pd.DataFrame, name: str) -> pd.DataFrame:
    """Remove rows of `other` that repeat a train skeleton or are near-duplicates of a train row."""
    same_skel = other["text"].map(skeleton).isin(train_skel).to_numpy()
    sim, _ = index.query(other["text"])
    near = sim > NEAR_DUP
    keep = ~(same_skel | near)
    print(f"{name}: dropped {int((~keep).sum())} of {len(other)} rows "
          f"(shared skeleton {int(same_skel.sum())}, near-duplicate {int(near.sum())})")
    return other[keep]


def split_validation(train: pd.DataFrame, index: NearDupIndex, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve ~1/VAL_FOLDS of train into validation, stratified by label x source, grouped by skeleton.
    Validation rows that are still near-duplicates of the remaining train rows are moved back to train."""
    train = train.reset_index(drop=True)
    strata = train["label"] + "|" + train["source"]
    groups = train["text"].map(skeleton)
    sgkf = StratifiedGroupKFold(n_splits=VAL_FOLDS, shuffle=True, random_state=seed)
    tr_idx, val_idx = next(sgkf.split(train, strata, groups))
    tr, val = train.iloc[tr_idx], train.iloc[val_idx]
    sim, _ = index.query(val["text"], rows=tr_idx)
    back = sim > NEAR_DUP
    print(f"val: {len(val)} rows carved, {int(back.sum())} near-duplicates of train moved back")
    return pd.concat([tr, val[back]]), val[~back]


# ---------------------------------------------------------------- data loading

def download_emscad() -> None:
    if RAW.exists():
        return
    RAW.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading EMSCAD -> {RAW}")
    urllib.request.urlretrieve(EMSCAD_URL, RAW)


def load_emscad(legit_sample: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(RAW)
    fields = ["title", "company_profile", "description", "requirements", "benefits"]
    text = df[fields].fillna("").agg(" \n".join, axis=1).str.replace(r"\s+", " ", regex=True).str.strip()
    out = pd.DataFrame({
        "text": text.str[:MAX_CHARS],
        "label": df["fraudulent"].map({1: "scam", 0: "legit"}),
        "category": "foreign_job",  # overseas ads from a Sri Lankan point of view; not used to train the category model
        "language": "en",
        "source": "emscad",
        "tactic": "emscad",
    })
    out = out[out["text"].str.len() > 50].drop_duplicates("text")
    # Down-sample legit ads so EMSCAD does not drown the Sri Lankan synthetic data.
    scam, legit = out[out.label == "scam"], out[out.label == "legit"]
    legit = legit.sample(min(legit_sample, len(legit)), random_state=seed)
    return pd.concat([scam, legit])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emscad-legit", type=int, default=4000, help="number of legit EMSCAD ads to keep")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-weak", action="store_true", help="leave out data/processed/weak_legit.csv")
    args = ap.parse_args()

    download_emscad()
    if not (SYN / "synthetic_train.csv").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_synthetic.py")], check=True)

    emscad = load_emscad(args.emscad_legit, args.seed)
    em_train, em_test = train_test_split(emscad, test_size=0.2, stratify=emscad["label"], random_state=args.seed)

    train = pd.concat([em_train, pd.read_csv(SYN / "synthetic_train.csv")])
    weak = OUT / "weak_legit.csv"  # real ikman ads, weakly labelled legit (scripts/build_weak_legit.py); train only
    if weak.exists() and not args.no_weak:
        weak_df = pd.read_csv(weak)
        train = pd.concat([train, weak_df])
        print(f"added {len(weak_df)} weakly labelled real legit ads to train")
    test = pd.concat([em_test, pd.read_csv(SYN / "synthetic_test.csv")])
    shifted = pd.read_csv(SYN / "shifted.csv")
    train = train.reset_index(drop=True)
    index, train_skel = NearDupIndex(train["text"]), set(train["text"].map(skeleton))
    test = drop_leaks(train_skel, index, test[~test["text"].isin(train["text"])], "test")
    shifted = drop_leaks(train_skel, index, shifted[~shifted["text"].isin(train["text"])], "shifted")
    train, val = split_validation(train, index, args.seed)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in [("train", train), ("val", val), ("test", test), ("shifted", shifted)]:
        df = df[COLUMNS].sample(frac=1, random_state=args.seed).reset_index(drop=True)
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} rows | {df.groupby(['source', 'label']).size().to_dict()}")


if __name__ == "__main__":
    main()
