"""Fast tests for scripts/tune_hparams.py: synthetic objectives only, no real data."""

import importlib.util
import math
import random
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "tune_hparams", Path(__file__).resolve().parents[1] / "scripts" / "tune_hparams.py")
th = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(th)

# A small mixed space with a known optimum: C=10, k=4, choice "b", flag True.
SPACE = [
    th.LogUniform("C", 0.01, 100.0),
    th.IntRange("k", 1, 6),
    th.Choice("choice", ["a", "b", "c", "d"], ordered=True),
    th.Choice("flag", [False, True]),
]


def quadratic(cfg):
    """Maximum 0 at the optimum, negative elsewhere."""
    return -((math.log10(cfg["C"]) - 1.0) ** 2 + 0.3 * (cfg["k"] - 4) ** 2
             + 0.5 * (cfg["choice"] != "b") + 0.5 * (not cfg["flag"]))


class Counter:
    def __init__(self, f):
        self.f, self.calls = f, 0

    def __call__(self, cfg):
        self.calls += 1
        return self.f(cfg)


def test_acceptance_rule_high_and_low_temperature():
    assert th.acceptance_probability(0.02, 1e-6) == 1.0  # improvements always accepted
    assert th.acceptance_probability(0.0, 1e-6) == 1.0
    assert th.acceptance_probability(-0.01, 10.0) > 0.99  # hot: almost any drop accepted
    assert th.acceptance_probability(-0.01, 1e-4) < 1e-40  # cold: drops essentially never accepted
    assert th.acceptance_probability(-0.01, 0.0) == 0.0
    assert th.acceptance_probability(-0.01, 0.015) == pytest.approx(math.exp(-0.01 / 0.015))
    # monotone in temperature and in the size of the drop
    temps = [1e-3, 1e-2, 1e-1, 1.0]
    probs = [th.acceptance_probability(-0.05, t) for t in temps]
    assert probs == sorted(probs)
    assert th.acceptance_probability(-0.1, 0.05) < th.acceptance_probability(-0.01, 0.05)


def test_metropolis_empirical_rate():
    rng = random.Random(0)
    p = th.acceptance_probability(-0.01, 0.01)  # e^-1
    rate = np.mean([rng.random() < p for _ in range(20000)])
    assert rate == pytest.approx(math.exp(-1), abs=0.01)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sa_improves_and_spends_budget(seed):
    rng = random.Random(seed)
    start = {"C": 0.01, "k": 1, "choice": "d", "flag": False}
    f = Counter(quadratic)
    res = th.simulated_annealing(SPACE, f, 60, rng, t0=0.5, alpha=0.93, start=start)
    assert f.calls == 60 and len(res["trials"]) == 60
    best_cfg, best_score = res["best"]
    assert best_score > quadratic(start) + 1.0
    assert best_score == max(s for _, s in res["trials"])
    temps = res["temperatures"]
    assert temps[0] == 0.5 and all(b < a for a, b in zip(temps[1:], temps[2:]))  # geometric cooling


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ga_improves_and_spends_budget(seed):
    f = Counter(quadratic)
    res = th.genetic_algorithm(SPACE, f, 40, random.Random(seed), pop_size=8, n_elite=2)
    assert f.calls == 40 and len(res["trials"]) == 40
    initial_best = max(s for _, s in res["generations"][0])
    assert res["best"][1] >= initial_best
    assert res["best"][1] > np.mean([s for _, s in res["generations"][0]])
    assert res["best"][1] > -0.6


def test_ga_keeps_the_elite():
    res = th.genetic_algorithm(SPACE, quadratic, 60, random.Random(3), pop_size=8, n_elite=2)
    gens = res["generations"]
    assert len(gens) >= 5
    for prev, nxt in zip(gens, gens[1:]):
        assert len(nxt) == 8
        top2 = sorted(prev, key=lambda ind: ind[1], reverse=True)[:2]
        for cfg, score in top2:  # the two best individuals survive unchanged
            assert any(c == cfg and s == score for c, s in nxt)
        assert max(s for _, s in nxt) >= max(s for _, s in prev)


def test_random_search_spends_budget():
    f = Counter(quadratic)
    res = th.random_search(SPACE, f, 25, random.Random(0))
    assert f.calls == 25 and all(th.in_bounds(SPACE, c) for c, _ in res["trials"])


def test_moves_and_mutations_stay_in_bounds():
    rng = random.Random(0)
    for space in (SPACE, th.SPACE):
        cfg = th.sample_config(space, rng)
        for _ in range(3000):
            cfg = th.sa_neighbour(space, cfg, rng)
            assert th.in_bounds(space, cfg), cfg
        edge = {p.name: (p.high if hasattr(p, "high") else p.options[-1]) for p in space}
        for _ in range(500):
            assert th.in_bounds(space, th.mutate(edge, space, rng, p_mut=1.0))
            child = th.uniform_crossover(edge, th.sample_config(space, rng), space, rng)
            assert th.in_bounds(space, child)


def test_neighbour_moves_are_local():
    rng = random.Random(1)
    ints = th.IntRange("k", 1, 5)
    assert {ints.neighbour(1, rng) for _ in range(50)} == {2}
    assert {ints.neighbour(3, rng) for _ in range(50)} == {2, 4}
    ordered = th.Choice("o", ["a", "b", "c"], ordered=True)
    assert {ordered.neighbour("a", rng) for _ in range(50)} == {"b"}
    flag = th.Choice("f", [False, True])
    assert all(flag.neighbour(True, rng) is False for _ in range(20))
    c = th.LogUniform("C", 0.01, 100)
    steps = [abs(math.log10(c.neighbour(1.0, rng))) for _ in range(500)]
    assert np.median(steps) < 0.6  # sd 0.5 decades


def test_default_config_in_space():
    assert th.in_bounds(th.SPACE, th.DEFAULT_CONFIG)


@pytest.mark.parametrize("cfg", [
    dict(lo=2, hi=4, min_df=1, max_features=None, sublinear=True),
    dict(lo=2, hi=5, min_df=2, max_features=150, sublinear=False),
    dict(lo=3, hi=5, min_df=1, max_features=80, sublinear=True),
    dict(lo=3, hi=6, min_df=2, max_features=None, sublinear=True),
])
def test_cached_char_tfidf_matches_sklearn(cfg):
    """Rebuilding TF-IDF from cached 2-6 gram counts must equal a freshly fitted TfidfVectorizer."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    tr = ["Pay Rs 50,000 now a to I go", "Job in Qatar, visa free!! a b", "SLBFE approved agency, no fee",
          "kandavaganna ikmanin a i u", "WhatsApp 07X XXX now to go", "Dubai driver job a an the in"]
    va = ["pay now to qatar a", "new zz word x"]
    bank = th.CountBank("char", tr, va)
    got_tr, got_va = bank.block(cfg["lo"], cfg["hi"], cfg["min_df"], cfg["max_features"], cfg["sublinear"])
    ref = TfidfVectorizer(analyzer="char_wb", ngram_range=(cfg["lo"], cfg["hi"]), min_df=cfg["min_df"],
                          max_features=cfg["max_features"], sublinear_tf=cfg["sublinear"])
    exp_tr, exp_va = ref.fit_transform(tr), ref.transform(va)
    assert got_tr.shape == exp_tr.shape
    np.testing.assert_allclose(got_tr.toarray(), exp_tr.toarray(), atol=1e-12)
    np.testing.assert_allclose(got_va.toarray(), exp_va.toarray(), atol=1e-12)


@pytest.mark.parametrize("hi,min_df,max_features", [(1, 1, None), (2, 1, 20), (3, 2, None)])
def test_cached_word_tfidf_matches_sklearn(hi, min_df, max_features):
    from sklearn.feature_extraction.text import TfidfVectorizer

    tr = ["pay the fee now and fly", "the job is in qatar and the visa is free", "pay now fly now",
          "approved agency no fee", "the fee is the fee"]
    va = ["pay the fee", "unknown words only"]
    bank = th.CountBank("word", tr, va)
    got_tr, got_va = bank.block(1, hi, min_df, max_features, True)
    ref = TfidfVectorizer(ngram_range=(1, hi), min_df=min_df, max_features=max_features, sublinear_tf=True)
    np.testing.assert_allclose(got_tr.toarray(), ref.fit_transform(tr).toarray(), atol=1e-12)
    np.testing.assert_allclose(got_va.toarray(), ref.transform(va).toarray(), atol=1e-12)


def test_bootstrap_report_shapes():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p_good = np.clip(y * 0.6 + rng.random(300) * 0.5, 0, 1)
    rep = th.bootstrap_report(y, {"default": p_good, "random": p_good, "sa": rng.random(300)}, n_boot=200)
    for name in rep:
        lo, hi = rep[name]["f1"]["ci95"]
        assert lo <= rep[name]["f1"]["value"] <= hi
    assert rep["random"]["f1_minus_default"]["ci95"] == [0.0, 0.0]  # identical predictions
    assert rep["sa"]["f1_minus_default"]["ci95"][1] < 0  # a coin-flip model is clearly worse
