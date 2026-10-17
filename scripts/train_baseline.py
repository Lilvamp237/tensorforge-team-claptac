"""Train the TF-IDF + logistic regression baseline.

Expects data/processed/train.csv (run scripts/make_dataset.py first) with columns: text, label (legit|scam), category, source.
Saves
  models/baseline.joblib   scam classifier: word 1-2 grams + char 2-5 grams (char n-grams carry Sinhala and Singlish)
  models/category.joblib   content-type classifier (local_job | foreign_job | student_visa | visa_pr), synthetic rows only
  models/meta.json         training metadata used by evaluate.py and the drift monitor
"""

import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

from safepassage.config import DATA_DIR, MODEL_PATH, MODEL_VERSION, MODELS_DIR

CATEGORY_PATH = MODELS_DIR / "category.joblib"
META_PATH = MODELS_DIR / "meta.json"


def text_features():
    return make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50_000),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=100_000),
    )


def main():
    train = pd.read_csv(DATA_DIR / "processed" / "train.csv")

    scam_model = make_pipeline(text_features(), LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
    scam_model.fit(train["text"], train["label"])

    synthetic = train[train["source"] == "synthetic"]
    category_model = make_pipeline(text_features(), LogisticRegression(max_iter=2000, C=4.0))
    category_model.fit(synthetic["text"], synthetic["category"])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scam_model, MODEL_PATH)
    joblib.dump(category_model, CATEGORY_PATH)
    META_PATH.write_text(json.dumps({
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": len(train),
        "label_counts": train["label"].value_counts().to_dict(),
        "source_counts": train["source"].value_counts().to_dict(),
        "language_counts": train["language"].value_counts().to_dict(),
    }, indent=2))
    print(f"Saved {MODEL_PATH.name}, {CATEGORY_PATH.name}, {META_PATH.name} to {MODELS_DIR}")


if __name__ == "__main__":
    main()
