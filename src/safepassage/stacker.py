"""Stacking meta-model: learns how to combine every signal into one scam probability. Owner: Person B.

Inputs per offer: the text model's probability, red-flag counts and key flags, official-record verification
(agency matched, contradicted checks, fee within an approved job-order cap). Novelty is left out on purpose:
every training row is in the novelty index, so it would be near-constant in training (a leak). A logistic
regression on these few features is trained by scripts/train_stacker.py on out-of-fold text-model scores, so it
never sees a text-model score from a model that trained on the same row. It stays small and readable: every
weight can be printed and explained.
"""

from functools import lru_cache

import joblib
import numpy as np

from .config import MODELS_DIR
from .schema import Verification

STACKER_PATH = MODELS_DIR / "stacker.joblib"
KEY_FLAGS = ["upfront_fee", "guaranteed", "personal_account", "contact_offplatform", "task_scam",
             "id_document_request", "no_slbfe", "crypto_payment", "secrecy", "urgency"]
SEVERE = {"personal_account", "id_document_request", "task_scam", "no_slbfe", "crypto_payment", "secrecy"}
FEATURES = (["text_logit", "n_flags", "n_severe"] + [f"flag_{f}" for f in KEY_FLAGS]
            + ["agency_matched", "agency_not_matched", "n_contradicted", "fee_within_cap"])


def features(p: float, flags: list[str], verification: Verification) -> list[float]:
    p = min(max(p, 1e-4), 1 - 1e-4)
    contradicted = sum(c.status == "contradicted" for c in verification.claims)
    fee_ok = any(c.status == "consistent" and c.claim.lower().startswith("fee") for c in verification.claims)
    return ([float(np.log(p / (1 - p))), len(flags), len(SEVERE.intersection(flags))]
            + [float(f in flags) for f in KEY_FLAGS]
            + [float(verification.agency_status == "matched"), float(verification.agency_status == "not_matched"),
               contradicted, float(fee_ok)])


@lru_cache(maxsize=1)
def load():
    return joblib.load(STACKER_PATH) if STACKER_PATH.exists() else None


def score(p: float, flags: list[str], verification: Verification) -> float | None:
    """Combined scam probability, or None when no stacker is trained (the caller keeps the text-model score)."""
    bundle = load()
    if bundle is None:
        return None
    x = np.array([features(p, flags, verification)])
    return float(bundle["model"].predict_proba(x)[0, 1])
