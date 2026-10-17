"""Novelty (unfamiliar-pattern) score and campaign clustering on embeddings. Owner: Person A."""

from .schema import Campaign, Novelty


def novelty(text: str) -> Novelty:
    # TODO(A): kNN distance to training embeddings, normalised to 0-1; status from NOVELTY_THRESHOLD
    return Novelty()


def campaign(text: str) -> Campaign | None:
    # TODO(A, stretch): nearest cluster id + cluster size
    return None
