"""Train the TF-IDF + logistic regression baseline.

Expects data/processed/train.csv and data/processed/test.csv with columns: text, label (legit|scam).
"""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline

from safepassage.config import DATA_DIR, MODEL_PATH


def main():
    train = pd.read_csv(DATA_DIR / "processed" / "train.csv")
    test = pd.read_csv(DATA_DIR / "processed" / "test.csv")

    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    pipeline.fit(train["text"], train["label"])
    print(classification_report(test["label"], pipeline.predict(test["text"]), digits=3))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
