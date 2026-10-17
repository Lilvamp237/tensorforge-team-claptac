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
