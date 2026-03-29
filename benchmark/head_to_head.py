#!/usr/bin/env python3
"""
Head-to-head: StochTree-Ex (Elixir/Rust) vs StochTree Python 0.4.0
Runs matching BART configurations and reports wall time + RMSE.

Usage:
    ~/projects/learn_erl/python-env/bin/python benchmark/head_to_head.py
"""

import numpy as np
import time
import json
from stochtree import BARTModel


def friedman1(n, p_noise=5, seed=42):
    rng = np.random.RandomState(seed)
    p = 5 + p_noise
    X = rng.uniform(0, 1, (n, p))
    y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1])
         + 20 * (X[:, 2] - 0.5) ** 2
         + 10 * X[:, 3]
         + 5 * X[:, 4]
         + rng.normal(0, 1, n))
    return X, y


def friedman2(n, seed=42):
    rng = np.random.RandomState(seed)
    X = np.column_stack([
        rng.uniform(0, 100, n),
        rng.uniform(40 * np.pi, 560 * np.pi, n),
        rng.uniform(0, 1, n),
        rng.uniform(1, 11, n),
    ])
    y = np.sqrt(X[:, 0] ** 2 + (X[:, 1] * X[:, 2] - 1 / (X[:, 1] * X[:, 3])) ** 2)
    y += rng.normal(0, y.std() * 0.1, n)
    return X, y


def simple_linear(n, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 10, (n, 3))
    y = 3 * X[:, 0] + rng.normal(0, 0.5, n)
    return X, y


def high_p(n, p, seed=42):
    """y = 2*x1 + noise, p-1 noise features."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, (n, p))
    y = 2.0 * X[:, 0] + rng.normal(0, 0.5, n)
    return X, y


def run_python_bart(name, X, y, num_trees=200, num_gfr=10, num_mcmc=100):
    model = BARTModel()
    t0 = time.time()
    model.sample(
        X_train=X, y_train=y,
        X_test=X,
        num_gfr=num_gfr,
        num_mcmc=num_mcmc,
        mean_forest_params={"num_trees": num_trees},
    )
    elapsed = time.time() - t0

    y_hat = model.y_hat_test
    if y_hat.shape[0] == len(y):
        y_hat = y_hat.T
    y_hat = y_hat[-num_mcmc:]
    y_pred = y_hat.mean(axis=0)
    rmse = np.sqrt(np.mean((y_pred - y) ** 2))

    return {"name": name, "time_ms": int(elapsed * 1000), "rmse": round(float(rmse), 4)}


def main():
    results = []
    print("=" * 70)
    print("StochTree Python 0.4.0 — Head-to-Head Reference")
    print("=" * 70)
    print()

    tests = [
        ("smoke (n=1K, p=10)", lambda: friedman1(1000, p_noise=5), 200, 100, 100),
        ("friedman2 (n=500, p=4)", lambda: friedman2(500), 200, 10, 100),
        ("simple-linear (n=200, p=3)", lambda: simple_linear(200), 50, 5, 50),
        ("high-p-100 (n=1K, p=100)", lambda: high_p(1000, 100), 200, 100, 100),
        ("extreme-p-500 (n=1K, p=500)", lambda: high_p(1000, 500), 200, 100, 100),
        ("medium-n (n=5K, p=10)", lambda: friedman1(5000, p_noise=5), 200, 100, 100),
        ("large-n (n=10K, p=10)", lambda: friedman1(10000, p_noise=5), 200, 100, 100),
    ]

    print(f"{'Test':<35} {'Time':>10} {'RMSE':>10}")
    print("-" * 60)

    for name, gen_fn, trees, gfr, mcmc in tests:
        X, y = gen_fn()
        r = run_python_bart(name, X, y, trees, gfr, mcmc)
        results.append(r)
        print(f"  {name:<33} {r['time_ms']:>8}ms  {r['rmse']:>8}")

    print()
    print("=" * 70)

    with open("benchmark/python_head_to_head.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to benchmark/python_head_to_head.json")


if __name__ == "__main__":
    main()
