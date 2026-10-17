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


# ----------------------------------------------------------------------------------------------------------
# 4. Final evaluation with bootstrap confidence intervals
# ----------------------------------------------------------------------------------------------------------


def _metrics(y, p, idx=None):
    from sklearn.metrics import roc_auc_score

    if idx is not None:
        y, p = y[idx], p[idx]
    pred = p >= 0.5
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    fn = np.sum(~pred & (y == 1))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    auc = roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan")
    return {"f1": f1, "precision": prec, "recall": rec, "roc_auc": auc}


def bootstrap_report(y, probs, n_boot=1000, seed=0, reference="default"):
    """Point metrics + 95% percentile CIs per method, and paired CIs of the F1 difference vs `reference`
    (the same resampled rows are used for every method, so the difference CI accounts for their correlation)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    out = {}
    boots = {}
    for name, p in probs.items():
        point = _metrics(y, p)
        b = [_metrics(y, p, i) for i in idx]
        boots[name] = np.array([r["f1"] for r in b])
        out[name] = {k: {"value": round(float(point[k]), 4),
                         "ci95": [round(float(np.nanpercentile([r[k] for r in b], q)), 4) for q in (2.5, 97.5)]}
                     for k in point}
    for name in probs:
        for ref in {reference, "random"} - {name}:
            if ref in boots:
                d = boots[name] - boots[ref]
                out[name][f"f1_minus_{ref}"] = {
                    "value": round(float(out[name]["f1"]["value"] - out[ref]["f1"]["value"]), 4),
                    "ci95": [round(float(np.percentile(d, q)), 4) for q in (2.5, 97.5)]}
    return out


# ----------------------------------------------------------------------------------------------------------
# 5. Driver
# ----------------------------------------------------------------------------------------------------------

OPTIMISERS = ("random", "sa", "ga")
LABELS = {"default": "Default (train_baseline.py)", "random": "Random search", "sa": "Simulated annealing",
          "ga": "Genetic algorithm"}


def run_optimiser(name, objective, budget, seed):
    rng = random.Random(f"{name}:{seed}")  # own stream per optimiser, fixed per seed
    if name == "random":
        return random_search(SPACE, objective, budget, rng)
    if name == "sa":
        return simulated_annealing(SPACE, objective, budget, rng)
    return genetic_algorithm(SPACE, objective, budget, rng)


def stratified_subsample(df, n, seed):
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    frac = n / len(df)
    parts = [g.sample(n=max(1, round(len(g) * frac)), random_state=seed) for _, g in df.groupby(["source", "label"])]
    import pandas as pd
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def fmt_cfg(cfg):
    c = dict(cfg)
    parts = [f"C={c['C']:.3g}", f"word 1-{c['word_ngram_max']}", f"char {tuple(c['char_ngram'])}",
             (f"min_df w{c['word_min_df']}/c{c['char_min_df']}" if "word_min_df" in c else f"min_df={c['min_df']}"),
             f"sublinear={c['sublinear_tf']}", f"class_weight={c['class_weight']}",
             f"char_max={c['char_max_features'] // 1000}k"]
    return ", ".join(parts)


def fit_and_predict(cfg, X, y, eval_texts):
    """Refit a config on all of train with the real sklearn pipeline; return P(scam) for each evaluation set."""
    model = build_pipeline(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    scam_idx = list(model.classes_).index("scam")
    return {name: model.predict_proba(texts)[:, scam_idx] for name, texts in eval_texts.items()}


def main():
    import pandas as pd
    from joblib import Parallel, delayed
    from sklearn.model_selection import StratifiedGroupKFold

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budget", type=int, default=40, help="objective evaluations per optimiser run")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="repeat every optimiser per seed")
    ap.add_argument("--subsample", type=int, default=4000, help="train rows used in the search (0 = all)")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    t_start = time.time()

    train = pd.read_csv(DATA_DIR / "processed" / "train.csv")
    sub = stratified_subsample(train, args.subsample or None, seed=0)
    texts = sub["text"].tolist()
    y = (sub["label"] == "scam").astype(int).to_numpy()
    groups = near_duplicate_groups(texts)
    folds = list(StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=0).split(texts, y, groups))
    print(f"search set: {len(sub)} train rows, {len(set(groups))} near-duplicate groups, "
          f"fold sizes {[len(v) for _, v in folds]}")

    objective = CVObjective(texts, y, folds)
    print(f"count matrices for {args.folds} folds built in {objective.setup_seconds:.1f}s")

    default_cv = objective(DEFAULT_CONFIG)
    default_log = objective.log.pop()
    print(f"default config CV F1 = {default_cv:.4f}")

    runs = {name: [] for name in OPTIMISERS}
    for seed in args.seeds:
        for name in OPTIMISERS:
            objective.log = []
            t = time.time()
            res = run_optimiser(name, objective, args.budget, seed)
            log = objective.log
            for i, (row, curve) in enumerate(zip(log, best_so_far([r["cv_f1"] for r in log]))):
                row["eval"] = i + 1
                row["best_so_far"] = curve
            if name == "sa":
                for row, temp, acc in zip(log, res["temperatures"], res["accepted"]):
                    row["temperature"] = round(temp, 6)
                    row["accepted"] = bool(acc)
            best = max(log, key=lambda r: r["cv_f1"])
            run = {"seed": seed, "seconds": round(time.time() - t, 1), "best_cv_f1": best["cv_f1"],
                   "best_params": best["params"], "trials": log}
            if name == "ga":
                run["generation_best"] = [round(max(s for _, s in g), 5) for g in res["generations"]]
            runs[name].append(run)
            print(f"seed {seed} {name:6s} best CV F1 {best['cv_f1']:.4f} in {run['seconds']:.0f}s  "
                  f"[{fmt_cfg(best['params'])}]")
    search_seconds = time.time() - t_start

    def to_cfg(params):
        c = dict(params)
        c["char_ngram"] = tuple(c["char_ngram"])
        return c

    summary = {}
    best_cfgs = {"default": DEFAULT_CONFIG}
    for name, rs in runs.items():
        scores = [r["best_cv_f1"] for r in rs]
        top = max(rs, key=lambda r: r["best_cv_f1"])
        best_cfgs[name] = to_cfg(top["best_params"])
        summary[name] = {"best_cv_f1_mean": round(float(np.mean(scores)), 4),
                         "best_cv_f1_std": round(float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0, 4),
                         "best_cv_f1_per_seed": scores, "best_overall_cv_f1": top["best_cv_f1"],
                         "best_seed": top["seed"], "best_params": top["best_params"]}
    summary["default"] = {"best_cv_f1_mean": round(default_cv, 4), "best_cv_f1_std": 0.0,
                          "best_params": jsonable(DEFAULT_CONFIG)}

    # Final, one-off refit on FULL train and evaluation on the held-out sets.
    test = pd.read_csv(DATA_DIR / "processed" / "test.csv")
    eval_sets = {"test": test}
    hand = DATA_DIR / "handwritten.csv"
    if hand.exists():
        eval_sets["handwritten"] = pd.read_csv(hand)
    t = time.time()
    unique = {config_key(cfg): cfg for cfg in best_cfgs.values()}  # identical winners are fitted once
    keys = list(unique)
    outs = Parallel(n_jobs=min(len(keys), 4))(
        delayed(fit_and_predict)(unique[k], train["text"], train["label"], {n: d["text"] for n, d in eval_sets.items()})
        for k in keys)
    by_key = dict(zip(keys, outs))
    probs = {set_name: {name: by_key[config_key(cfg)][set_name] for name, cfg in best_cfgs.items()}
             for set_name in eval_sets}
    refit_seconds = time.time() - t
    final = {}
    for set_name, df in eval_sets.items():
        yt = (df["label"] == "scam").astype(int).to_numpy()
        final[set_name] = {"n": int(len(df)), "metrics": bootstrap_report(yt, probs[set_name], n_boot=args.n_boot)}
    total_seconds = time.time() - t_start

    # Recommendation: the config with the best CV score; the test set is only reported, never used to choose.
    rec_name = max(OPTIMISERS, key=lambda n: summary[n]["best_overall_cv_f1"])
    if summary[rec_name]["best_overall_cv_f1"] <= default_cv:
        rec_name = "default"
    recommended = jsonable(best_cfgs[rec_name])

    result = {
        "setup": {"budget_per_run": args.budget, "seeds": args.seeds, "folds": args.folds,
                  "search_rows": len(sub), "train_rows": len(train), "n_groups": int(len(set(groups))),
                  "group_rule": "connected components of cosine >= 0.7 on digit-masked word TF-IDF",
                  "cv": "StratifiedGroupKFold(n_splits=%d, shuffle=True, random_state=0)" % args.folds,
                  "objective": "mean F1 (scam positive, threshold 0.5) over the folds",
                  "sa": {"t0": 0.015, "alpha": 0.95, "move": "change one random parameter"},
                  "ga": {"pop_size": 8, "n_elite": 2, "tournament_k": 3, "p_mut": 0.2,
                         "crossover": "uniform"},
                  "space": {p.name: (p.options if isinstance(p, Choice) else [p.low, p.high]) for p in SPACE},
                  "word_max_features": WORD_MAX_FEATURES},
        "runtime_seconds": {"setup_counts": round(objective.setup_seconds, 1), "search": round(search_seconds, 1),
                            "refit_and_eval": round(refit_seconds, 1), "total": round(total_seconds, 1)},
        "default": {"cv_f1": round(default_cv, 5), "fold_f1": default_log["fold_f1"],
                    "params": jsonable(DEFAULT_CONFIG)},
        "summary": summary,
        "runs": runs,
        "final": final,
        "recommended": {"from": rec_name, "params": recommended},
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "tuning.json").write_text(json.dumps(result, indent=1))
    MODELS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / "best_params.json").write_text(json.dumps(
        {"source": f"scripts/tune_hparams.py ({LABELS[rec_name]})", "params": recommended,
         "word_max_features": WORD_MAX_FEATURES, "cv_f1": summary[rec_name].get("best_overall_cv_f1", default_cv)},
        indent=2))
    png = plot_convergence(runs, default_cv)
    (REPORTS_DIR / "tuning.md").write_text(render_markdown(result, png), encoding="utf-8")

    print(f"\ntotal runtime {total_seconds / 60:.1f} min (search {search_seconds:.0f}s, refit {refit_seconds:.0f}s)")
    for name in ("default",) + OPTIMISERS:
        m = final["test"]["metrics"][name]
        print(f"{name:8s} CV {summary[name]['best_cv_f1_mean']:.4f}±{summary[name]['best_cv_f1_std']:.4f}  "
              f"test F1 {m['f1']['value']:.3f} {m['f1']['ci95']}  AUC {m['roc_auc']['value']:.3f}")
    print(f"recommended: {rec_name} -> models/best_params.json")


def mean_curve(runs, name):
    curves = np.array([[r["best_so_far"] for r in run["trials"]] for run in runs[name]])
    return curves.mean(axis=0), curves.std(axis=0)


def plot_convergence(runs, default_cv):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"random": "#7f7f7f", "sa": "#1f77b4", "ga": "#d62728"}
    for name in OPTIMISERS:
        m, s = mean_curve(runs, name)
        x = np.arange(1, len(m) + 1)
        ax.plot(x, m, label=LABELS[name], color=colors[name], lw=2)
        ax.fill_between(x, m - s, m + s, color=colors[name], alpha=0.15)
    ax.axhline(default_cv, color="black", ls="--", lw=1, label="Default config")
    ax.set_xlabel("objective evaluations")
    ax.set_ylabel("best-so-far CV F1 (mean ± sd over seeds)")
    ax.set_title("Convergence at equal budget")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = REPORTS_DIR / "tuning_convergence.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path.name


def render_markdown(r, png):
    """Numbers come from the run; the explanatory text is fixed."""
    s, f, setup = r["summary"], r["final"]["test"]["metrics"], r["setup"]
    budget, seeds = setup["budget_per_run"], setup["seeds"]
    names = ("default",) + OPTIMISERS
    L = []
    w = L.append
    w("# Hyperparameter search: random search vs simulated annealing vs genetic algorithm\n")
    w(f"Generated by `scripts/tune_hparams.py`. Total runtime {r['runtime_seconds']['total'] / 60:.1f} min on a "
      f"laptop CPU (search {r['runtime_seconds']['search']:.0f}s, final refit and evaluation "
      f"{r['runtime_seconds']['refit_and_eval']:.0f}s). Raw numbers, every trial included: `reports/tuning.json`.\n")

    w("## Results at a glance\n")
    w("| Method | Best CV F1 (mean ± sd over seeds) | Test F1 [95% CI] | Precision | Recall | ROC-AUC | "
      "Test F1 minus default [95% CI] |")
    w("|---|---|---|---|---|---|---|")
    for n in names:
        m = f[n]
        cv = f"{s[n]['best_cv_f1_mean']:.4f}" + (f" ± {s[n]['best_cv_f1_std']:.4f}" if n != "default" else " (fixed)")
        diff = m.get("f1_minus_default")
        dtxt = f"{diff['value']:+.3f} [{diff['ci95'][0]:+.3f}, {diff['ci95'][1]:+.3f}]" if diff else "-"
        w(f"| {LABELS[n]} | {cv} | {m['f1']['value']:.3f} [{m['f1']['ci95'][0]:.3f}, {m['f1']['ci95'][1]:.3f}] | "
          f"{m['precision']['value']:.3f} | {m['recall']['value']:.3f} | {m['roc_auc']['value']:.3f} | {dtxt} |")
    w("")
    w(conclusion(r))
    if "handwritten" in r["final"]:
        h = r["final"]["handwritten"]
        w(f"\n**Hand-written set** (n={h['n']}, test only):\n")
        w("| Method | F1 [95% CI] | Precision | Recall | ROC-AUC |")
        w("|---|---|---|---|---|")
        for n in names:
            m = h["metrics"][n]
            w(f"| {LABELS[n]} | {m['f1']['value']:.3f} [{m['f1']['ci95'][0]:.3f}, {m['f1']['ci95'][1]:.3f}] | "
              f"{m['precision']['value']:.3f} | {m['recall']['value']:.3f} | {m['roc_auc']['value']:.3f} |")
    w("")

    w("## Best configuration per method\n")
    w("| Method | Configuration | CV F1 |")
    w("|---|---|---|")
    for n in names:
        cv = s[n].get("best_overall_cv_f1", s[n]["best_cv_f1_mean"])
        w(f"| {LABELS[n]} | {fmt_cfg(s[n]['best_params'])} | {cv:.4f} |")
    rec = r["recommended"]
    w(f"\nRecommended (highest CV F1, chosen without looking at the test set): **{LABELS[rec['from']]}**, "
      f"written to `models/best_params.json`.\n")

    w("## What was searched\n")
    w("The pipeline is the one in `scripts/train_baseline.py`: word TF-IDF and character TF-IDF (`char_wb`, which "
      "copes with Sinhala script and Singlish spelling) joined side by side, then logistic regression. "
      "Seven knobs, mixed types:\n")
    w("| Parameter | Type | Range | What it changes |")
    w("|---|---|---|---|")
    w("| `C` | continuous, log scale | 0.01 to 100 | inverse regularisation strength: small C = simpler model |")
    w("| `word_ngram_max` | integer | 1 to 3 | longest word phrase used as a feature |")
    w("| `char_ngram` | ordered choice | (2,4), (2,5), (3,5), (3,6) | character fragment lengths |")
    w("| `min_df` | integer | 1 to 5 | drop n-grams seen in fewer documents (both vectorizers) |")
    w("| `sublinear_tf` | boolean | False / True | use 1 + log(count) instead of the raw count |")
    w("| `class_weight` | choice | None / balanced | up-weight the minority class in the loss |")
    w("| `char_max_features` | ordered choice | 50k, 100k, 200k | vocabulary cap for character n-grams |")
    w(f"\nWord vocabulary is capped at {setup['word_max_features']:,} as in the baseline. LinearSVC was left out: "
      "at large C it needed 10+ seconds per fit to converge and gives no probabilities without an extra "
      "calibration step, which would have broken the time budget.\n")

    w("## How a configuration is scored (the objective)\n")
    w(f"- **Train data only.** A fixed, stratified subsample of {setup['search_rows']:,} of the "
      f"{setup['train_rows']:,} training rows (same mix of EMSCAD/synthetic and scam/legit). "
      "`test.csv`, `shifted.csv` and the hand-written set are never read during the search.")
    w(f"- **Grouped, stratified {setup['folds']}-fold CV.** Before splitting, texts are linked when their cosine "
      "similarity is at least 0.7 after digits and `XXX` placeholders are masked; linked texts form one group "
      f"({setup['n_groups']:,} groups). `StratifiedGroupKFold` keeps every group inside one fold, so a model is "
      "never validated on a near-copy of a text it trained on (synthetic ads made from the same template "
      "lines; EMSCAD companies that reuse the same boilerplate). Without this, CV rewards memorising templates.")
    w("- **Score** = mean F1 over the folds, scam = positive class, threshold 0.5 (the same rule as "
      "`scripts/evaluate.py`). The folds are identical for every method and every seed, so differences "
      "come from the optimisers, not from luck in the split.")
    w("- **Speed trick.** For each fold, one word counter (1-3 grams) and one character counter (2-6 grams) "
      "are fitted once. Any TF-IDF setting in the space is then rebuilt from those counts by keeping the "
      "right columns (n-gram length, min_df, top-k by frequency), applying log and IDF weighting and "
      "normalising. A unit test checks that this matches scikit-learn's `TfidfVectorizer` exactly. Built "
      "matrices are cached, so a move that only changes `C` or `class_weight` costs only three logistic "
      "regression fits (run in parallel).\n")

    w("## The three optimisers (same budget for each)\n")
    w(f"Every run gets exactly **{budget} objective evaluations** and is repeated with seeds {seeds}. "
      "All three share the same moves, so the comparison is about the search strategy alone.\n")
    w("**Random search (baseline).** Draw each parameter independently from its range (C uniformly on the "
      "log scale) and keep the best. It has no memory, which makes it a fair and surprisingly strong baseline "
      "when only a few parameters matter.\n")
    w("**Simulated annealing (SA).** Start from a random configuration. Each step changes *one* randomly "
      "chosen parameter by a small local move:")
    w("- `C`: add Gaussian noise with sd 0.5 to log10(C), reflected back inside [0.01, 100];")
    w("- integers and ordered choices: one step up or down (bounced at the ends);")
    w("- booleans and unordered choices: switch to the other value.\n")
    w("The new configuration is scored, and the difference Δ = new F1 − current F1 decides whether we move "
      "there (the **Metropolis rule**): if Δ ≥ 0 always accept; if Δ < 0 accept with probability exp(Δ / T). "
      "T is the *temperature*. Early on T is high, so the search sometimes walks downhill and can escape a "
      "local optimum; as T falls it behaves more and more like greedy hill climbing. The schedule is "
      f"**geometric**: T_k = T0 · α^k with **T0 = {setup['sa']['t0']}** and **α = {setup['sa']['alpha']}**. "
      "Since F1 differences here are around 0.01, T0 = 0.015 accepts a 0.01 drop with probability "
      f"exp(−0.01/0.015) ≈ 0.51; after {budget - 1} steps T ≈ {setup['sa']['t0'] * setup['sa']['alpha'] ** (budget - 1):.4f} "
      "and the same drop is accepted with probability ≈ "
      f"{math.exp(-0.01 / (setup['sa']['t0'] * setup['sa']['alpha'] ** (budget - 1))):.3f}. "
      "The best configuration ever seen is tracked separately from the current one.\n")
    sa_acc = [np.mean([t["accepted"] for t in run["trials"][1:]]) for run in r["runs"]["sa"]]
    w(f"Observed: SA accepted {100 * np.mean(sa_acc):.0f}% of proposed moves on average.\n")
    ga = setup["ga"]
    w(f"**Genetic algorithm (GA).** A population of **{ga['pop_size']}** configurations evolves over generations. "
      f"Generation 0 is random. To build the next generation:")
    w(f"- **Elitism:** the best **{ga['n_elite']}** individuals are copied over unchanged (and not re-scored), so "
      "the best score in the population can never go down.")
    w(f"- **Tournament selection:** pick {ga['tournament_k']} individuals at random and take the fittest as a parent; "
      "do this twice for two parents. Fitter configurations become parents more often, but weak ones still get "
      "a chance, which keeps diversity.")
    w("- **Uniform crossover:** the child takes each parameter from one parent or the other with probability 1/2.")
    w(f"- **Mutation:** each parameter of the child, independently with probability {ga['p_mut']}, takes one local "
      "move (the same moves SA uses).")
    w(f"\nWith {budget} evaluations that is 8 random individuals, then {ga['pop_size'] - ga['n_elite']} new children per "
      f"generation for {(budget - ga['pop_size']) // (ga['pop_size'] - ga['n_elite'])} full generations"
      + (f" plus {(budget - ga['pop_size']) % (ga['pop_size'] - ga['n_elite'])} in a final partial one."
         if (budget - ga['pop_size']) % (ga['pop_size'] - ga['n_elite']) else "."))
    gb = [run["generation_best"] for run in r["runs"]["ga"]]
    w(f" Best score in the population per generation (seed {seeds[0]}): "
      + " → ".join(f"{v:.4f}" for v in gb[0]) + " (never decreases, thanks to elitism).\n")

    w("## Convergence: best-so-far CV F1 after n evaluations\n")
    w("Mean over seeds (sd in brackets). The default config scores "
      f"{r['default']['cv_f1']:.4f} on the same folds.\n")
    checkpoints = [n for n in (1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100) if n <= budget]
    if budget not in checkpoints:
        checkpoints.append(budget)
    w("| Evaluations | " + " | ".join(LABELS[n] for n in OPTIMISERS) + " |")
    w("|---|" + "---|" * len(OPTIMISERS))
    curves = {n: mean_curve(r["runs"], n) for n in OPTIMISERS}
    for c in checkpoints:
        w(f"| {c} | " + " | ".join(f"{curves[n][0][c - 1]:.4f} ({curves[n][1][c - 1]:.4f})" for n in OPTIMISERS) + " |")
    if png:
        w(f"\n![Convergence]({png})\n")
    else:
        w("")

    w("## Limitations\n")
    w("- **CV on partly synthetic data.** About half the training rows are synthetic ads built from template "
      "lines. Grouping near-duplicates removes the most obvious leakage, but different templates still share "
      "slot values and phrasing, so CV F1 is optimistic and the ranking of configurations may not carry over "
      "to real Sri Lankan ads.")
    w("- **Overfitting the CV.** Each optimiser looked at "
      f"{budget * len(seeds)} configurations and we keep the maximum, so the best CV score is biased upwards "
      "(the winner is partly the luckiest). That is why the honest comparison is the one-off held-out test "
      "score, not the CV score.")
    w(f"- **Small budget and few seeds.** {budget} evaluations × {len(seeds)} seeds is enough to compare "
      "strategies roughly, not to rank them with confidence; with more evaluations SA and GA usually pull ahead "
      "of random search on smooth spaces, and the gap can disappear when only one or two parameters matter.")
    w(f"- **Search on a {setup['search_rows']:,}-row subsample.** The winner is refitted on all "
      f"{setup['train_rows']:,} rows; the best `C` and `min_df` can shift with more data.")
    w("- **Threshold fixed at 0.5.** Tuning the decision threshold (for example to favour recall, since a "
      "missed scam costs more than a false alarm) is a separate, cheap step and is not part of this search.")
    w("- **Test set reused across four models.** Four refits are scored on one test set; the paired bootstrap "
      "CIs above do not correct for multiple comparisons.")
    return "\n".join(L) + "\n"


def conclusion(r):
    f = r["final"]["test"]["metrics"]
    s = r["summary"]
    lines = ["**Honest conclusion.** "]
    better_cv = [n for n in ("sa", "ga") if s[n]["best_cv_f1_mean"] - s[n]["best_cv_f1_std"]
                 > s["random"]["best_cv_f1_mean"] + s["random"]["best_cv_f1_std"]]
    if better_cv:
        lines.append("On cross-validation, " + " and ".join(LABELS[n] for n in better_cv)
                     + " beat random search by more than one seed-to-seed standard deviation. ")
    else:
        lines.append("On cross-validation the three optimisers end up within about one seed-to-seed standard "
                     "deviation of each other, so with this budget none is clearly better at searching. ")
    tuned = [n for n in OPTIMISERS]
    vs_def = {n: f[n]["f1_minus_default"] for n in tuned}
    sig_up = [n for n in tuned if vs_def[n]["ci95"][0] > 0]
    sig_down = [n for n in tuned if vs_def[n]["ci95"][1] < 0]
    inside = [n for n in tuned if n not in sig_up and n not in sig_down]
    if inside:
        lines.append("On the held-out test set, the F1 difference to the default config for "
                     + ", ".join(LABELS[n] for n in inside)
                     + " is inside the 95% bootstrap confidence interval (the paired CI of the difference "
                       "includes 0), so we cannot claim they are better than the default. ")
    if sig_up:
        lines.append(", ".join(LABELS[n] for n in sig_up)
                     + " improve on the default with a paired 95% CI that excludes 0. ")
    if sig_down:
        lines.append(", ".join(LABELS[n] for n in sig_down)
                     + " are worse than the default with a paired 95% CI that excludes 0. ")
    vs_rand = [n for n in ("sa", "ga") if f[n]["f1_minus_random"]["ci95"][0] > 0]
    if vs_rand:
        lines.append(" and ".join(LABELS[n] for n in vs_rand) + " also beat random search on test outside the CI. ")
    else:
        lines.append("Neither SA nor GA beats random search on the test set outside the confidence interval. ")
    lines.append("The search is still worth having: it shows which knobs matter, documents that the baseline is "
                 "not an accident, and gives a reproducible procedure to re-run when the data is fixed.")
    return "".join(lines)


if __name__ == "__main__":
    sys.exit(main())
