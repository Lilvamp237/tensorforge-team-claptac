import joblib

from .config import MODEL_PATH, REVIEW_THRESHOLD, SCAM_THRESHOLD


class ScamModel:
    def __init__(self, path=MODEL_PATH):
        self.pipeline = joblib.load(path)

    def score(self, text: str) -> float:
        """Probability that the text is a scam."""
        classes = list(self.pipeline.classes_)
        return float(self.pipeline.predict_proba([text])[0][classes.index("scam")])


def verdict(score: float) -> str:
    if score >= SCAM_THRESHOLD:
        return "scam"
    if score >= REVIEW_THRESHOLD:
        return "suspicious"
    return "legit"
