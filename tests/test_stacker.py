import numpy as np
from sklearn.linear_model import LogisticRegression

from safepassage import stacker
from safepassage.schema import Claim, Verification

VERIFIED = Verification(agency_status="matched", claims=[Claim(claim="Fee of Rs. 1 is within the cap", status="consistent")])


def test_feature_vector_matches_feature_names():
    x = stacker.features(0.7, ["personal_account", "urgency"], VERIFIED)
    assert len(x) == len(stacker.FEATURES)
    named = dict(zip(stacker.FEATURES, x))
    assert named["n_severe"] == 1 and named["flag_personal_account"] == 1 and named["fee_within_cap"] == 1
    assert named["agency_matched"] == 1 and named["text_logit"] > 0


def test_score_is_none_without_a_trained_stacker(monkeypatch):
    monkeypatch.setattr(stacker, "load", lambda: None)
    assert stacker.score(0.9, [], Verification()) is None


def test_score_uses_the_trained_model(monkeypatch):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, len(stacker.FEATURES)))
    y = (X[:, 0] > 0).astype(int)  # learn "high text logit = scam"
    monkeypatch.setattr(stacker, "load", lambda: {"model": LogisticRegression().fit(X, y)})
    assert stacker.score(0.99, [], Verification()) > stacker.score(0.01, [], Verification())
