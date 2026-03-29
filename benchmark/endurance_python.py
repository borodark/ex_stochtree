#!/usr/bin/env python3
"""
BART Endurance — Python reference using StochTree 0.4.0.
Matches the exact test configs from endurance_bench.exs.

Usage:
    ~/projects/learn_erl/python-env/bin/python benchmark/endurance_python.py
    ~/projects/learn_erl/python-env/bin/python benchmark/endurance_python.py --quick
"""

import sys
import time
import numpy as np
from stochtree import BARTModel


# ── Data generators (matching Elixir) ────────────────────────────────

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


def smooth_sin(n, seed=50):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, (n, 5))
    y = (np.sin(2 * np.pi * X[:, 0]) + np.cos(2 * np.pi * X[:, 1])
         + 2 * X[:, 2] ** 2 + X[:, 3] + rng.normal(0, 0.2, n))
    return X, y


def sparse_linear(n, p, seed=60):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 10, (n, p))
    y = (3.0 * X[:, 0] + 2.0 * X[:, 1] - 1.5 * X[:, 2]
         + 0.8 * X[:, 3] + 0.3 * X[:, 4] + rng.normal(0, 0.5, n))
    return X, y


def california_housing(seed=70):
    rng = np.random.RandomState(seed)
    n = 20000
    inc = rng.uniform(1, 51, n)
    age = rng.uniform(1, 51, n)
    rooms = rng.uniform(1, 11, n)
    bedrms = rng.uniform(1, 6, n)
    pop = rng.uniform(1, 35001, n)
    occup = rng.uniform(1, 7, n)
    lat = rng.uniform(32, 37, n)
    lon = rng.uniform(-124, -114, n)
    X = np.column_stack([inc, age, rooms, bedrms, pop, occup, lat, lon])
    log_price = (0.4 * np.log(inc + 1) - 0.1 * np.log(pop + 1)
                 + 0.2 * np.log(rooms + 1) - 0.05 * np.abs(lat - 37)
                 - 0.03 * np.abs(lon + 120) + 0.01 * age - 0.1 * occup
                 + rng.normal(0, 0.3, n))
    y = np.exp(log_price) / 100
    return X, y


# ── Benchmark runner ─────────────────────────────────────────────────

def run_test(name, X, y, num_trees, num_gfr, num_mcmc, seed=42):
    n, p = X.shape

    t0 = time.time()
    model = BARTModel()
    model.sample(
        X_train=X, y_train=y, X_test=X[:min(100, n)],
        num_gfr=num_gfr, num_mcmc=num_mcmc,
        mean_forest_params={"num_trees": num_trees},
    )
    elapsed = time.time() - t0
    ms = int(elapsed * 1000)

    # RMSE on first 100 rows
    y_hat = model.y_hat_test
    if y_hat.shape[0] == min(100, n):
        y_hat = y_hat.T
    y_pred = y_hat[-num_mcmc:].mean(axis=0)
    rmse = np.sqrt(np.mean((y_pred - y[:min(100, n)]) ** 2))

    sigma = model.sigma_samples[-1] if hasattr(model, "sigma_samples") else 0.0

    print(f"  {name:<25} {ms:>8}ms  RMSE={rmse:.3f}  sigma={sigma:.4f}")
    return {"name": name, "time_ms": ms, "rmse": round(float(rmse), 4),
            "sigma": round(float(sigma), 4), "n": n, "p": p}


# ── Test suite matching endurance_bench.exs ──────────────────────────

def main():
    quick = "--quick" in sys.argv

    print("=" * 70)
    print("  StochTree Python 0.4.0 Endurance")
    print("=" * 70)
    print()

    tests = [
        ("smoke (1K×10)",
         lambda: (*friedman1(1000, 5), 200, 10, 100)),
        ("medium-n (10K×10)",
         lambda: (*friedman1(10000, 5), 200, 10, 100)),
        ("large-n (50K×10)",
         lambda: (*friedman1(50000, 5), 100, 5, 50)),
        ("high-p (1K×100)",
         lambda: (*sparse_linear(1000, 100), 200, 10, 100)),
        ("extreme-p (5K×500)",
         lambda: (*sparse_linear(5000, 500), 200, 10, 50)),
        ("smooth-sin (5K×5)",
         lambda: (*smooth_sin(5000), 200, 15, 200)),
        ("california (20K×8)",
         lambda: (*california_housing(), 200, 10, 100)),
    ]

    if quick:
        tests = tests[:1]

    header = f"{'Test':<25} {'Time':>10}  {'RMSE':>8}  {'Sigma':>8}"
    print(header)
    print("-" * 60)

    total_ms = 0
    results = []
    for name, gen_fn in tests:
        X, y, trees, gfr, mcmc = gen_fn()
        r = run_test(name, X, y, trees, gfr, mcmc)
        results.append(r)
        total_ms += r["time_ms"]

    print()
    print("=" * 70)
    print(f"  Total: {total_ms // 1000}s ({total_ms / 60000:.1f} min)")
    print("=" * 70)

    import json
    with open("benchmark/python_endurance.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to benchmark/python_endurance.json")


if __name__ == "__main__":
    main()
