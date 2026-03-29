# BART Sprint 3: Close the Per-Iteration Gap

## Current State

ForestTracker (Sprint 2) fixed algorithmic scaling: extreme-p went from
4.7h to 1.6 min (173x). But StochTree Python is still 7x faster overall
on wall time. The gap is in the per-tree-per-iteration constant factor:

| | Elixir/Rust | Python/C++ | Gap |
|---|---|---|---|
| Per tree per iter | 0.88ms | ~0.09ms | 10x |

The 10x comes from three sources:
1. NIF call overhead + data marshalling
2. Memory allocation patterns in Rust (Vec, HashMap)
3. Cache efficiency (C++ uses contiguous arrays, Rust uses HashMap for leaf_stats)

## Sprint 3 Optimizations

### 3a. Replace HashMap with Dense Vec for Leaf Stats

Current: `leaf_stats: HashMap<u32, SuffStats>` — hash lookup per access.

Trees have at most ~64 leaves (depth 6). Use a dense `Vec<Option<SuffStats>>`
indexed by leaf_id. Leaf IDs are sequential integers from `next_leaf_id`.

```rust
// Before
leaf_stats: Vec<HashMap<u32, SuffStats>>

// After
leaf_stats: Vec<Vec<Option<SuffStats>>>  // [tree][leaf_id] = Some(stats)
```

**Expected impact**: Eliminate HashMap overhead (hashing, collision handling,
pointer chasing). Each leaf lookup becomes a single array index. ~2x speedup
on per-tree operations.

**Effort**: Low-medium. Mechanical replacement of HashMap ops.

### 3b. Batch NIF: Multiple Trees per NIF Call

Current: `fit_bart` is a single NIF call that runs the entire BART sampler.
This is already batched. The overhead is NOT per-NIF-call but per-tree
within the NIF.

The actual bottleneck is the backfitting loop:
```rust
for tree_idx in 0..num_trees {
    // refresh_leaf_stats: O(n) scan
    // grow_from_root or mcmc_step: O(n × p) split search
    // resample_leaves: O(n) scan
    // update predictions: O(n) scan
}
```

Four O(n) passes per tree × 200 trees = 800 × O(n) per iteration. At n=20K
(california): 800 × 20K = 16M array accesses per iteration.

### 3c. Fuse O(n) Passes

Combine the four per-tree O(n) passes into two:

Pass 1: `refresh_leaf_stats` + split search setup (pre-compute residuals
and update leaf stats in one scan)

Pass 2: `apply_split/prune` + `resample_leaves` + `update_predictions`
(single scan: for each observation, look up leaf_id, apply new mu, update
prediction)

**Expected impact**: 2x reduction in memory bandwidth. At n=20K with 200
trees, save 400 × 20K = 8M array accesses per iteration.

**Effort**: Medium. Requires restructuring the backfitting loop.

### 3d. SIMD-Friendly Sorted Scan

The sorted-scan split evaluation iterates through pre-sorted indices,
skipping observations not in the target leaf. With a dense leaf_assignment
array, this is a branch-heavy loop:

```rust
for &idx in sorted_indices[feat].iter() {
    if leaf_assignment[tree_idx][idx as usize] != target_leaf { continue; }
    // accumulate stats
}
```

At n=20K with ~32 leaves, ~97% of iterations are skips. Replace with:

```rust
// Pre-filter: collect indices for target leaf (O(n_leaf))
let leaf_indices: Vec<u32> = (0..n_obs)
    .filter(|&i| leaf_assignment[tree_idx][i] == target_leaf)
    .map(|i| i as u32)
    .collect();

// Sort leaf_indices by feature value (O(n_leaf log n_leaf))
leaf_indices.sort_by(|&a, &b| x_val(a, feat).partial_cmp(&x_val(b, feat)).unwrap());

// Dense scan: no skips, no branch mispredictions
for &idx in leaf_indices.iter() {
    // accumulate stats — every iteration does useful work
}
```

**Expected impact**: Eliminate branch mispredictions in the hot loop. At
n=20K with 32 leaves (avg 625 obs/leaf), the dense scan does 625 useful
iterations vs the current 20,000 iterations with 97% skips. **32x fewer
loop iterations** for the average leaf.

**Effort**: Medium. Need to maintain per-leaf sorted indices through
split/prune operations. On split: partition parent's sorted indices into
two children. On prune: merge (already sorted, merge-sort O(n_leaf)).

**Risk**: Memory. Per-leaf sorted indices for all features:
sum(n_leaf) × p × 4 bytes = n × p × 4 bytes per tree. For n=20K, p=10,
200 trees: 160MB. Acceptable on 256GB, but scales poorly. Consider
maintaining only for the current tree being updated (one set of per-leaf
indices, rebuilt on `refresh_leaf_stats`).

### 3e. Arena Allocator for Tree Nodes

Rust's default allocator (jemalloc or system) handles many small
allocations for TreeNode enum variants. An arena allocator (bumpalo)
would batch these into contiguous memory, improving cache locality:

```rust
use bumpalo::Bump;

let arena = Bump::new();
let node = arena.alloc(TreeNode::Internal { ... });
```

**Expected impact**: Better cache locality for tree traversal. Marginal
(~10-20%) but compounds across 200 trees × 200 iterations.

**Effort**: Low. Add bumpalo, allocate trees per-iteration in an arena.

## Implementation Order

1. **3a** (dense Vec for leaf_stats) — low effort, immediate 2x on per-tree ops
2. **3d** (per-leaf sorted indices) — medium effort, 32x fewer loop iterations
3. **3c** (fuse O(n) passes) — medium effort, 2x less memory bandwidth
4. **3e** (arena allocator) — low effort, marginal cache improvement
5. **3b** is already done (single NIF call)

## Target

| Test | Current | Sprint 3 Target | Python |
|---|---|---|---|
| smoke (1K×10) | 20s | 5s | 1.8s |
| extreme-p (5K×500) | 98s | 30s | 57s |
| california (20K×8) | 680s | 100s | 42s |

The per-leaf sorted indices (3d) is the highest-impact change. Combined
with dense Vec (3a), the per-tree cost should drop from 0.88ms to ~0.2ms,
closing the gap to ~2x of C++.
