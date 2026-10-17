"""Novelty (unfamiliar-pattern) score and campaign clustering on embeddings. Owner: Person A.

Embedding: char 2-5 gram TF-IDF -> 128-d LSA (TruncatedSVD), L2-normalised. Language-agnostic, so it works for
Sinhala, Tamil and Singlish without a multilingual transformer.
Score: mean cosine distance to the k nearest training items, rescaled with a calibration split that the index never
saw: 0 at the calibration median, 1 at twice the calibration p99 distance above the median. The novelty threshold
(config.NOVELTY_THRESHOLD) therefore sits a little above the p99 of normal traffic.
"""

from functools import lru_cache

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from .config import MODELS_DIR, NOVELTY_THRESHOLD
from .schema import Campaign, Novelty

NOVELTY_PATH = MODELS_DIR / "novelty.joblib"
K = 5


class NoveltyIndex:
    def __init__(self, dims: int = 128, k: int = K, seed: int = 42):
        self.embedder = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=100_000),
            TruncatedSVD(n_components=dims, random_state=seed),
            Normalizer(),
        )
        self.k = k
        self.nn: NearestNeighbors | None = None
        self.median = 0.0
        self.p99 = 1.0

    def fit(self, train_texts, calib_texts) -> "NoveltyIndex":
        emb = self.embedder.fit_transform(list(train_texts))
        self.nn = NearestNeighbors(n_neighbors=self.k, metric="cosine").fit(emb)
        d = self.distances(calib_texts)
        self.median, self.p99 = float(np.median(d)), float(np.percentile(d, 99))
        return self

    def embed(self, texts) -> np.ndarray:
        return self.embedder.transform(list(texts))

    def distances(self, texts) -> np.ndarray:
        dist, _ = self.nn.kneighbors(self.embed(texts))
        return dist.mean(axis=1)

    def scores(self, texts) -> np.ndarray:
        span = max(2 * (self.p99 - self.median), 1e-6)
        return np.clip((self.distances(texts) - self.median) / span, 0.0, 1.0)

    def save(self, path=NOVELTY_PATH) -> None:
        joblib.dump(self, path)


@lru_cache(maxsize=1)
def _index() -> NoveltyIndex | None:
    return joblib.load(NOVELTY_PATH) if NOVELTY_PATH.exists() else None


def novelty(text: str) -> Novelty:
    index = _index()
    if index is None:
        return Novelty()
    score = round(float(index.scores([text])[0]), 4)
    return Novelty(score=score, status="unfamiliar_pattern" if score >= NOVELTY_THRESHOLD else "familiar")


def campaign(text: str) -> Campaign | None:
    # TODO(A, stretch): nearest cluster id + cluster size
    return None
