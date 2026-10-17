"""Hyperparameter search for the scam classifier: random search vs simulated annealing vs a genetic algorithm.

The optimisers are written out by hand (no tuning library) so every step can be explained.

Pipeline being tuned (same shape as scripts/train_baseline.py):
    FeatureUnion(word TF-IDF, char_wb TF-IDF) -> LogisticRegression

Objective: mean F1 (scam = positive) over 3 grouped, stratified folds of a fixed TRAIN subsample.
Near-duplicate texts (cosine >= 0.7 on digit-masked word TF-IDF) share a group, so copies of one synthetic
template or one EMSCAD company boilerplate never sit on both sides of a fold. test.csv, shifted.csv and
data/handwritten.csv are never read during the search; they are only used at the end, once, to report.

Speed: per fold we fit ONE word CountVectorizer (1-3 grams) and ONE char_wb CountVectorizer (2-6 grams).
Every TF-IDF configuration in the search space (n-gram range, min_df, max_features, sublinear_tf) is then
rebuilt from those counts with cheap column selection and arithmetic, giving exactly what TfidfVectorizer
would produce (checked in tests/test_tuning.py). Built blocks are cached, so changing only C or
class_weight costs one logistic regression fit per fold.

Usage:  PYTHONPATH=src python scripts/tune_hparams.py [--budget 40] [--seeds 0 1] [--subsample 4000]
Writes: reports/tuning.json, reports/tuning.md, reports/tuning_convergence.png (if matplotlib),
        models/best_params.json
"""

import argparse
import json
import math
import random
import re
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

# ----------------------------------------------------------------------------------------------------------
# 1. Search space
# ----------------------------------------------------------------------------------------------------------


class LogUniform:
    """Continuous parameter sampled uniformly in log10 space (e.g. C from 0.01 to 100)."""

    def __init__(self, name, low, high, step_sd=0.5):
        self.name, self.low, self.high, self.step_sd = name, low, high, step_sd
        self.lo, self.hi = math.log10(low), math.log10(high)

    def sample(self, rng):
        return 10 ** rng.uniform(self.lo, self.hi)

    def neighbour(self, value, rng):
        # Gaussian step of step_sd decades in log space, reflected back inside the bounds.
        x = math.log10(value) + rng.gauss(0.0, self.step_sd)
        span = self.hi - self.lo
        while x < self.lo or x > self.hi:
            x = 2 * self.lo - x if x < self.lo else 2 * self.hi - x
            if span == 0:
                x = self.lo
        return 10 ** x

    def contains(self, value):
        return self.low * (1 - 1e-9) <= value <= self.high * (1 + 1e-9)


class IntRange:
    """Integer parameter in [low, high]; a neighbour is one step up or down (bounced at the ends)."""

    def __init__(self, name, low, high):
        self.name, self.low, self.high = name, low, high

    def sample(self, rng):
        return rng.randint(self.low, self.high)

    def neighbour(self, value, rng):
        if self.low == self.high:
            return value
        step = rng.choice((-1, 1))
        if not self.low <= value + step <= self.high:
            step = -step
        return value + step

    def contains(self, value):
        return isinstance(value, int) and self.low <= value <= self.high


class Choice:
    """Categorical parameter. If ordered, a neighbour is an adjacent option; otherwise any other option."""

    def __init__(self, name, options, ordered=False):
        self.name, self.options, self.ordered = name, list(options), ordered

    def sample(self, rng):
        return rng.choice(self.options)

    def neighbour(self, value, rng):
        if len(self.options) == 1:
            return value
        i = self.options.index(value)
        if self.ordered:
            j = i + rng.choice((-1, 1))
            if not 0 <= j < len(self.options):
                j = i - (j - i)
            return self.options[j]
        return rng.choice([o for k, o in enumerate(self.options) if k != i])

    def contains(self, value):
        return value in self.options


SPACE = [
    LogUniform("C", 0.01, 100.0),
    IntRange("word_ngram_max", 1, 3),
    Choice("char_ngram", [(2, 4), (2, 5), (3, 5), (3, 6)], ordered=True),
    IntRange("min_df", 1, 5),
    Choice("sublinear_tf", [False, True]),
    Choice("class_weight", [None, "balanced"]),
    Choice("char_max_features", [50_000, 100_000, 200_000], ordered=True),
]
WORD_MAX_FEATURES = 50_000  # fixed, as in train_baseline.py

# What scripts/train_baseline.py trains today (it uses separate min_df for the two vectorizers).
DEFAULT_CONFIG = {"C": 4.0, "word_ngram_max": 2, "char_ngram": (2, 5), "min_df": 2, "word_min_df": 2,
                  "char_min_df": 3, "sublinear_tf": True, "class_weight": "balanced", "char_max_features": 100_000}


def sample_config(space, rng):
    return {p.name: p.sample(rng) for p in space}


def in_bounds(space, config):
    return all(p.contains(config[p.name]) for p in space)


# ----------------------------------------------------------------------------------------------------------
# 2. Optimisers. Each takes objective(config) -> score to MAXIMISE and spends exactly `budget` calls.
# ----------------------------------------------------------------------------------------------------------


def random_search(space, objective, budget, rng):
    trials = []
    for _ in range(budget):
        cfg = sample_config(space, rng)
        trials.append((cfg, objective(cfg)))
    return {"trials": trials}


def acceptance_probability(delta, temperature):
    """Metropolis rule for maximisation: always take an improvement, take a drop of |delta| with exp(delta/T)."""
    if delta >= 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(delta / temperature)


def sa_neighbour(space, config, rng):
    """Change exactly one randomly chosen parameter by that parameter's local move."""
    p = rng.choice(space)
    new = dict(config)
    new[p.name] = p.neighbour(config[p.name], rng)
    return new


def simulated_annealing(space, objective, budget, rng, t0=0.015, alpha=0.95, start=None):
    """Geometric cooling T_k = t0 * alpha**k. With F1 differences of ~0.01, t0=0.015 accepts a 0.01 drop with
    p ~ 0.51 at the start; after 39 steps T ~ 0.002 and the same drop is accepted with p ~ 0.007."""
    current = dict(start) if start is not None else sample_config(space, rng)
    current_score = objective(current)
    best, best_score = dict(current), current_score
    trials = [(dict(current), current_score)]
    temps, accepted = [t0], [True]
    temperature = t0
    for _ in range(budget - 1):
        cand = sa_neighbour(space, current, rng)
        score = objective(cand)
        trials.append((cand, score))
        ok = rng.random() < acceptance_probability(score - current_score, temperature)
        if ok:
            current, current_score = cand, score
            if score > best_score:
                best, best_score = dict(cand), score
        temps.append(temperature)
        accepted.append(ok)
        temperature *= alpha
    return {"trials": trials, "temperatures": temps, "accepted": accepted, "best": (best, best_score)}


def tournament(population, rng, k=3):
    """Pick k individuals at random, return the fittest."""
    return max(rng.sample(population, min(k, len(population))), key=lambda ind: ind[1])[0]


def uniform_crossover(a, b, space, rng):
    """Each gene comes from parent a or parent b with probability 1/2."""
    return {p.name: (a if rng.random() < 0.5 else b)[p.name] for p in space}


def mutate(config, space, rng, p_mut):
    """Each gene independently, with probability p_mut, takes one local move (same moves as SA)."""
    new = dict(config)
    for p in space:
        if rng.random() < p_mut:
            new[p.name] = p.neighbour(new[p.name], rng)
    return new


def genetic_algorithm(space, objective, budget, rng, pop_size=8, n_elite=2, k=3, p_mut=0.2):
    """Generational GA. Generation 0 is pop_size random configs; every next generation keeps the n_elite best
    unchanged (not re-evaluated) and fills the rest with mutated crossover children of tournament winners.
    Stops as soon as `budget` objective calls are spent (the last generation may be partial)."""
    trials, generations = [], []
    population = []
    for _ in range(min(pop_size, budget)):
        cfg = sample_config(space, rng)
        s = objective(cfg)
        trials.append((cfg, s))
        population.append((cfg, s))
    generations.append(list(population))
    while len(trials) < budget:
        population.sort(key=lambda ind: ind[1], reverse=True)
        nxt = population[:n_elite]
        while len(nxt) < pop_size and len(trials) < budget:
            child = mutate(uniform_crossover(tournament(population, rng, k), tournament(population, rng, k),
                                             space, rng), space, rng, p_mut)
            s = objective(child)
            trials.append((child, s))
            nxt.append((child, s))
        if len(nxt) < pop_size:  # budget ran out mid-generation: survivors fill the gap
            nxt += population[n_elite:n_elite + pop_size - len(nxt)]
        population = nxt
        generations.append(list(population))
    best = max(trials, key=lambda t: t[1])
    return {"trials": trials, "generations": generations, "best": best}


def best_so_far(scores):
    out, b = [], -math.inf
    for s in scores:
        b = max(b, s)
        out.append(b)
    return out


# ----------------------------------------------------------------------------------------------------------
# 3. Objective: grouped CV with cached feature matrices
# ----------------------------------------------------------------------------------------------------------


def near_duplicate_groups(texts, threshold=0.7):
    """Connected components of the 'cosine >= threshold' graph on digit-masked word TF-IDF."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    from sklearn.feature_extraction.text import TfidfVectorizer

    masked = [re.sub(r"x{2,}", "#", re.sub(r"\d+", "0", t.lower())) for t in texts]
    X = TfidfVectorizer().fit_transform(masked)
    rows, cols = [], []
    for s in range(0, X.shape[0], 2000):
        sim = (X[s:s + 2000] @ X.T).toarray()
        r, c = np.nonzero(sim >= threshold)
        rows.append(r + s)
        cols.append(c)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    graph = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(X.shape[0],) * 2)
    return connected_components(graph, directed=False)[1]


def _idf(counts_csc, n_docs):
    """Smooth IDF exactly as TfidfVectorizer: ln((1 + n) / (1 + df)) + 1, df = documents containing the term."""
    df = np.diff(counts_csc.indptr)
    return np.log((1 + n_docs) / (1 + df)) + 1.0


class CountBank:
    """Raw n-gram counts of one fold's train/validation texts, from which any TF-IDF config is rebuilt.

    kind='word': CountVectorizer(ngram_range=(1, 3))        -> word n-grams, sliced by number of tokens
    kind='char': CountVectorizer(analyzer='char_wb', (2, 6)) -> char n-grams, sliced by string length
    """

    def __init__(self, kind, train_texts, val_texts):
        from sklearn.feature_extraction.text import CountVectorizer

        self.kind = kind
        if kind == "word":
            cv = CountVectorizer(ngram_range=(1, 3), dtype=np.int64)
        else:
            cv = CountVectorizer(analyzer="char_wb", ngram_range=(2, 6), dtype=np.int64)
        self.Xtr = cv.fit_transform(train_texts).tocsc()
        self.Xva = cv.transform(val_texts).tocsc()
        names = cv.get_feature_names_out()  # alphabetical, as TfidfVectorizer's vocabulary
        # n-gram order of each column: number of tokens (word) or string length (char). Slicing the 2-6 char
        # counts by length is exact because char_wb emits a padded word shorter than n once, at n = its length,
        # and then stops; padded words are at least 3 characters long, so this holds whenever lo <= 3.
        self.length = np.array([n.count(" ") + 1 if kind == "word" else len(n) for n in names])
        self.df = np.diff(self.Xtr.indptr)
        self.n_train = self.Xtr.shape[0]

    def block(self, lo, hi, min_df, max_features, sublinear):
        """Return (train, val) TF-IDF matrices identical to TfidfVectorizer(ngram_range=(lo, hi), min_df,
        max_features, sublinear_tf) fitted on the train texts (smooth_idf=True, norm='l2')."""
        from sklearn.preprocessing import normalize

        if self.kind == "char" and lo > 3:
            raise ValueError("char n-gram slices are exact only for lo <= 3")
        cand = np.where((self.length >= lo) & (self.length <= hi) & (self.df >= min_df))[0]
        Xtr, Xva = self.Xtr[:, cand], self.Xva[:, cand]
        if max_features is not None and len(cand) > max_features:
            tfs = np.asarray(Xtr.sum(axis=0)).ravel()
            keep = np.sort((-tfs).argsort()[:max_features])
            Xtr, Xva = Xtr[:, keep], Xva[:, keep]
        idf = _idf(Xtr, self.n_train)
        out = []
        for X in (Xtr, Xva):
            X = X.tocsr().astype(np.float64)
            if sublinear:
                X.data = 1.0 + np.log(X.data)
            X = X.multiply(idf).tocsr()
            out.append(normalize(X))
        return out


def word_params(cfg):
    return (1, cfg["word_ngram_max"], cfg.get("word_min_df", cfg["min_df"]), WORD_MAX_FEATURES, cfg["sublinear_tf"])


def char_params(cfg):
    lo, hi = cfg["char_ngram"]
    return (lo, hi, cfg.get("char_min_df", cfg["min_df"]), cfg["char_max_features"], cfg["sublinear_tf"])


def make_classifier(cfg):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=cfg["C"], class_weight=cfg["class_weight"], max_iter=2000)


class CVObjective:
    """mean F1 over fixed folds; logs every call as a trial."""

    def __init__(self, texts, y, folds, cache_size=32, n_jobs=-1):
        from joblib import Parallel, delayed

        self.y, self.folds = y, folds
        t = time.time()
        jobs = [(kind, [texts[i] for i in tr], [texts[i] for i in va]) for tr, va in folds for kind in ("word", "char")]
        banks = Parallel(n_jobs=n_jobs)(delayed(CountBank)(*j) for j in jobs)  # processes: tokenising holds the GIL
        self.banks = [{"word": banks[2 * f], "char": banks[2 * f + 1]} for f in range(len(folds))]
        self.setup_seconds = time.time() - t
        self.cache = [OrderedDict() for _ in folds]  # one LRU cache per fold, so fold threads never share one
        self.cache_size = cache_size
        self.n_jobs = n_jobs
        self.memo = {}
        self.log = []

    def _block(self, fold, kind, params):
        cache, key = self.cache[fold], (kind, params)
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        val = self.banks[fold][kind].block(*params)
        cache[key] = val
        if len(cache) > self.cache_size:
            cache.popitem(last=False)
        return val

    def _fold_f1(self, cfg, f):
        from scipy.sparse import hstack
        from sklearn.metrics import f1_score

        wtr, wva = self._block(f, "word", word_params(cfg))
        ctr, cva = self._block(f, "char", char_params(cfg))
        tr, va = self.folds[f]
        clf = make_classifier(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(hstack([wtr, ctr]).tocsr(), self.y[tr])
        p = clf.predict_proba(hstack([wva, cva]).tocsr())[:, 1]
        return f1_score(self.y[va], (p >= 0.5).astype(int), zero_division=0)

    def __call__(self, cfg):
        from joblib import Parallel, delayed

        t = time.time()
        key = config_key(cfg)
        if key in self.memo:  # identical config already scored (the objective is deterministic)
            fold_f1 = self.memo[key]
        else:
            # one thread per fold: build (or fetch) its TF-IDF blocks, fit, score
            fold_f1 = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(self._fold_f1)(cfg, f) for f in range(len(self.folds)))
            self.memo[key] = fold_f1
        score = float(np.mean(fold_f1))
        self.log.append({"params": jsonable(cfg), "cv_f1": round(score, 5),
                         "fold_f1": [round(float(v), 4) for v in fold_f1], "seconds": round(time.time() - t, 3)})
        return score


def config_key(cfg):
    return tuple((k, round(v, 10) if isinstance(v, float) else v) for k, v in sorted(cfg.items()))


def jsonable(cfg):
    return {k: (list(v) if isinstance(v, tuple) else (round(v, 6) if isinstance(v, float) else v))
            for k, v in cfg.items()}


def build_pipeline(cfg):
    """The real sklearn pipeline for a config (used for the final refit and by the training script later)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline, make_union

    _, wmax, wmin, wfeat, sub = word_params(cfg)
    clo, chi, cmin, cfeat, _ = char_params(cfg)
    return make_pipeline(
        make_union(
            TfidfVectorizer(ngram_range=(1, wmax), min_df=wmin, sublinear_tf=sub, max_features=wfeat),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(clo, chi), min_df=cmin, sublinear_tf=sub,
                            max_features=cfeat),
        ),
        make_classifier(cfg),
    )
