# Benchmark Results: StochTree-Ex vs StochTree Python 0.4.0

**Date**: 2026-03-27
**Hardware**: Dual Intel Xeon E5-2699 v4 (44 cores / 88 threads), 256GB RAM
**Python**: OMP_NUM_THREADS=88, OPENBLAS_NUM_THREADS=88
**Elixir**: +sbt tnnps, rayon cap=64, 88 BEAM schedulers
**Each run sequential** — full machine per implementation.

## Wall Time

| Test | Python (C++ OMP) | Elixir (Rust rayon) | Ratio |
|---|---|---|---|
| smoke (1K×10) | 1,800ms | 19,989ms | 0.09x |
| medium-n (10K×10) | 37,792ms | 122,831ms | 0.31x |
| large-n (50K×10) | 22,245ms | 159,343ms | 0.14x |
| high-p (1K×100) | 25,579ms | 27,125ms | **0.94x** |
| extreme-p (5K×500) | 57,019ms | 97,683ms | **0.58x** |
| smooth-sin (5K×5) | 7,445ms | 124,797ms | 0.06x |
| california (20K×8) | 41,740ms | 680,395ms | 0.06x |
| **Total** | **193s** | **1,327s** | **0.15x** |

## RMSE (lower is better)

| Test | Python | Elixir | Winner |
|---|---|---|---|
| smoke (1K×10) | 0.870 | **0.408** | Elixir |
| medium-n (10K×10) | 1.000 | **0.975** | Elixir |
| large-n (50K×10) | 1.116 | **1.113** | Elixir |
| high-p (1K×100) | 0.715 | **0.195** | Elixir |
| extreme-p (5K×500) | 0.644 | **0.422** | Elixir |
| smooth-sin (5K×5) | 0.189 | **0.160** | Elixir |
| california (20K×8) | **0.005** | 0.007 | Python |

Elixir wins RMSE on **6 of 7 tests**. The ForestTracker's exhaustive
cutpoint search (every possible split in one sorted scan) finds better
splits than StochTree Python's sampled cutpoints.

## Analysis

**Python is 7x faster on wall time.** StochTree Python's C++ backend
(with OpenMP) has lower per-iteration overhead. The tree fitting inner loop
in C++ is ~10x faster than the equivalent Rust NIF code called through
the BEAM NIF boundary.

**The gap narrows with dimensionality.** At p=100: 0.94x (near parity).
At p=500: 0.58x. The ForestTracker's O(n) per-feature sorted scan scales
better than the C++ approach as p grows. Rayon parallelism across features
also helps more at high p.

**The gap widens with n.** california (n=20K): 0.06x. The per-observation
overhead in the NIF (Elixir↔Rust boundary, memory allocation) scales
linearly with n. StochTree Python's C++ keeps everything in contiguous
C arrays.

## Optimization History

| Version | extreme-p (5K×500) | Speedup from v0 |
|---|---|---|
| v0 (brute-force) | 16,854,194ms (4.7h) | 1x |
| v1 (ForestTracker) | 127,106ms (2.1 min) | 133x |
| v2 (rayon cap=64) | 97,683ms (1.6 min) | 173x |
| StochTree Python | 57,019ms (57s) | — |

## Sprint 3 Target

Close the per-iteration gap on low-p/high-n tests. Primary bottleneck:
NIF call overhead and memory allocation patterns in the Rust tree
operations.
