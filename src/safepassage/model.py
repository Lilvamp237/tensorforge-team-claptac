"""Supervised classifier wrapper. Owner: Person A."""

import joblib

from .config import HIGH_RISK, MEDIUM_RISK, MODEL_PATH


class ScamModel:
    def __init__(self, path=MODEL_PATH):
        self.pipeline = joblib.load(path)

    def score(self, text: str) -> float:
        """Probability that the text is a scam."""
        classes = list(self.pipeline.classes_)
        return float(self.pipeline.predict_proba([text])[0][classes.index("scam")])


def risk_label(p: float) -> str:
    if p >= HIGH_RISK:
        return "high"
    if p >= MEDIUM_RISK:
        return "medium"
    return "low"
