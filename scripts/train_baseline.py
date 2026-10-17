"""Train the TF-IDF + logistic regression baseline.

Expects data/processed/train.csv (run scripts/make_dataset.py first) with columns: text, label (legit|scam), category, source.
Saves
  models/baseline.joblib   scam classifier: word 1-2 grams + char 2-5 grams (char n-grams carry Sinhala and Singlish)
  models/category.joblib   content-type classifier (local_job | foreign_job | student_visa | visa_pr), synthetic rows only
  models/meta.json         training metadata used by evaluate.py and the drift monitor
Analyst-confirmed labels in $SAFEPASSAGE_FEEDBACK (default data/processed/feedback.csv; columns text,label) are
appended to the training data. Set SAFEPASSAGE_MODEL_DIR to write the models somewhere other than models/.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

from safepassage.config import DATA_DIR, FEEDBACK_PATH, MODEL_PATH, MODEL_VERSION, MODELS_DIR

OUT_DIR = Path(os.environ.get("SAFEPASSAGE_MODEL_DIR") or MODELS_DIR)
MODEL_PATH = OUT_DIR / MODEL_PATH.name
CATEGORY_PATH = OUT_DIR / "category.joblib"
META_PATH = OUT_DIR / "meta.json"


def text_features():
    return make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50_000),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=100_000),
    )


def merge_feedback(train: pd.DataFrame, feedback: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Append analyst-confirmed rows. The analyst label wins over any training row with the same text, and the
    latest decision wins when the same text was confirmed twice."""
    fb = feedback.dropna(subset=["text", "label"])
    fb = fb[fb["label"].isin(["scam", "legit"])].drop_duplicates("text", keep="last")
    overridden = int(train["text"].isin(fb["text"]).sum())
    fb = fb[["text", "label"]].assign(category="unknown", language="unknown", source="feedback", tactic="feedback")
    merged = pd.concat([train[~train["text"].isin(fb["text"])], fb], ignore_index=True)
    return merged, {"feedback_rows": len(fb), "feedback_replaced_rows": overridden}


def main():
    train = pd.read_csv(DATA_DIR / "processed" / "train.csv")
    feedback_path = Path(os.environ.get("SAFEPASSAGE_FEEDBACK") or FEEDBACK_PATH)
    feedback_info = {"feedback_rows": 0, "feedback_replaced_rows": 0}
    if feedback_path.exists():
        train, feedback_info = merge_feedback(train, pd.read_csv(feedback_path))
        print(f"Added {feedback_info['feedback_rows']} analyst-confirmed rows from {feedback_path}")

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
        **feedback_info,
    }, indent=2))
    print(f"Saved {MODEL_PATH.name}, {CATEGORY_PATH.name}, {META_PATH.name} to {OUT_DIR}")


if __name__ == "__main__":
    main()
