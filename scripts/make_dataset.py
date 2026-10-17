"""Build the train / held-out test / shifted splits.

Sources
  EMSCAD (Employment Scam Aegean Dataset, 17,880 real job ads, 866 fraudulent), CC0 mirror on Hugging Face:
    https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction
  Synthetic Sri Lankan offers from scripts/generate_synthetic.py (run automatically if missing).

Outputs (data/processed/)
  train.csv    EMSCAD 80% + synthetic train pool
  test.csv     EMSCAD 20% (stratified) + synthetic held-out pool (reserved phrasing). Never used for training.
  shifted.csv  drift demo set (Tamil, new tactics, new destinations). Never used for training.
Columns: text, label (scam|legit), category, language, source, tactic
"""

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "fake_job_postings.csv"
SYN = ROOT / "data" / "synthetic"
OUT = ROOT / "data" / "processed"
EMSCAD_URL = ("https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction"
              "/resolve/main/fake_job_postings.csv")
COLUMNS = ["text", "label", "category", "language", "source", "tactic"]
MAX_CHARS = 1500  # keep EMSCAD ads close to the length of the short offers we score in production


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
    args = ap.parse_args()

    download_emscad()
    if not (SYN / "synthetic_train.csv").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_synthetic.py")], check=True)

    emscad = load_emscad(args.emscad_legit, args.seed)
    em_train, em_test = train_test_split(emscad, test_size=0.2, stratify=emscad["label"], random_state=args.seed)

    train = pd.concat([em_train, pd.read_csv(SYN / "synthetic_train.csv")])
    test = pd.concat([em_test, pd.read_csv(SYN / "synthetic_test.csv")])
    shifted = pd.read_csv(SYN / "shifted.csv")
    test = test[~test["text"].isin(train["text"])]
    shifted = shifted[~shifted["text"].isin(train["text"])]

    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in [("train", train), ("test", test), ("shifted", shifted)]:
        df = df[COLUMNS].sample(frac=1, random_state=args.seed).reset_index(drop=True)
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} rows | {df.groupby(['source', 'label']).size().to_dict()}")


if __name__ == "__main__":
    main()
