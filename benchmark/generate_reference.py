#!/usr/bin/env python3
"""
Generate BART reference results from StochTree Python for validation.

Produces JSON files with data, fitted predictions, and variable importance
that the Elixir tests compare against.

Usage:
    pip install stochtree numpy
    python3 benchmark/generate_reference.py
"""

import json
import numpy as np
import time

try:
    from stochtree import BARTModel
    HAS_STOCHTREE = True
except ImportError:
    HAS_STOCHTREE = False
    print("WARNING: stochtree not installed. Generating data-only reference.")
    print("Install: pip install stochtree")


def friedman1(n, p_noise=5, seed=42):
    """Friedman #1: y = 10*sin(pi*x1*x2) + 20*(x3-0.5)^2 + 10*x4 + 5*x5 + noise
    5 active features + p_noise noise features."""
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
    """Friedman #2: y = sqrt(x1^2 + (x2*x3 - 1/(x2*x4))^2) + noise"""
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
    """y = 3*x1 + 0*x2 + 0*x3 + noise (variable selection test)"""
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 10, (n, 3))
    y = 3 * X[:, 0] + rng.normal(0, 0.5, n)
    return X, y


def generate_benchmark(name, X_train, y_train, X_test, y_test_true,
                       num_trees=200, num_gfr=10, num_mcmc=100, seed=42):
    """Run StochTree Python and save results."""
    result = {
        "name": name,
        "n_train": len(y_train),
        "n_test": len(y_test_true),
        "n_features": X_train.shape[1],
        "X_train": X_train.tolist(),
        "y_train": y_train.tolist(),
        "X_test": X_test.tolist(),
        "y_test_true": y_test_true.tolist(),
        "config": {
            "num_trees": num_trees,
            "num_gfr": num_gfr,
            "num_mcmc": num_mcmc,
            "seed": seed,
        },
    }

    if HAS_STOCHTREE:
        print(f"  Fitting {name} with StochTree Python...")
        model = BARTModel()
        t0 = time.time()
        model.sample(
            X_train=X_train, y_train=y_train,
            X_test=X_test,
            num_gfr=num_gfr,
            num_mcmc=num_mcmc,
            mean_forest_params={"num_trees": num_trees},
        )
        elapsed = time.time() - t0

        # Get predictions — shape depends on StochTree version
        y_pred_samples = model.y_hat_test  # (n_samples, n_test) or (n_test, n_samples)
        if y_pred_samples.shape[0] == len(y_test_true):
            # (n_test, n_samples) — transpose
            y_pred_samples = y_pred_samples.T
        # Now (n_samples, n_test) — take only MCMC samples (skip GFR)
        y_pred_samples = y_pred_samples[-num_mcmc:]
        y_pred_mean = y_pred_samples.mean(axis=0)
        y_pred_lower = np.percentile(y_pred_samples, 5, axis=0)
        y_pred_upper = np.percentile(y_pred_samples, 95, axis=0)

        rmse = np.sqrt(np.mean((y_pred_mean - y_test_true) ** 2))
        coverage = np.mean((y_test_true >= y_pred_lower) & (y_test_true <= y_pred_upper))

        result["stochtree_python"] = {
            "y_pred_mean": y_pred_mean.tolist(),
            "y_pred_lower": y_pred_lower.tolist(),
            "y_pred_upper": y_pred_upper.tolist(),
            "rmse": float(rmse),
            "coverage_90": float(coverage),
            "elapsed_s": round(elapsed, 2),
        }
        print(f"    RMSE={rmse:.4f}, coverage={coverage:.2%}, time={elapsed:.1f}s")
    else:
        # Without stochtree, just save data for Elixir-only testing
        result["stochtree_python"] = None

    return result


def main():
    results = {}

    # Friedman #1: the BART paper benchmark
    print("Friedman #1 (n=500, p=10):")
    X, y = friedman1(500, p_noise=5, seed=42)
    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]
    results["friedman1"] = generate_benchmark(
        "friedman1", X_train, y_train, X_test, y_test,
        num_trees=200, num_gfr=10, num_mcmc=100)

    # Friedman #2: harder nonlinearity
    print("Friedman #2 (n=500, p=4):")
    X, y = friedman2(500, seed=42)
    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]
    results["friedman2"] = generate_benchmark(
        "friedman2", X_train, y_train, X_test, y_test,
        num_trees=200, num_gfr=10, num_mcmc=100)

    # Simple linear: variable selection test
    print("Simple linear (n=200, p=3):")
    X, y = simple_linear(200, seed=42)
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]
    results["simple_linear"] = generate_benchmark(
        "simple_linear", X_train, y_train, X_test, y_test,
        num_trees=50, num_gfr=5, num_mcmc=50)

    # Save
    out_path = "benchmark/reference_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f)

    size_mb = len(json.dumps(results)) / 1_048_576
    print(f"\nSaved to {out_path} ({size_mb:.1f} MB)")
    print(f"Benchmarks: {list(results.keys())}")
    if HAS_STOCHTREE:
        for name, r in results.items():
            sp = r["stochtree_python"]
            if sp:
                print(f"  {name}: RMSE={sp['rmse']:.4f}, coverage={sp['coverage_90']:.2%}")


if __name__ == "__main__":
    main()
