use crate::tree::SuffStats;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256StarStar;
use std::collections::HashMap;

/// Initial capacity for the dense leaf_stats Vec per tree.
/// Trees have at most ~64 leaves (depth 6). 128 covers depth 7.
const LEAF_VEC_INITIAL_CAP: usize = 128;

/// Minimum n_obs to enable per-leaf sorted indices.
/// Below this threshold, the skip rate in the global scan is tolerable and
/// the O(n*p) rebuild cost per tree outweighs the scan savings.
const LEAF_SORTED_MIN_OBS: usize = 100_000;

/// Pre-sorted column indices: for each feature j, indices sorted by x[., j].
/// Built once per fit_bart call, reused across all iterations.
pub struct SortedColumnIndex {
    /// sorted_indices[j] = observation indices sorted by ascending x[., j]
    sorted_indices: Vec<Vec<u32>>,
}

impl SortedColumnIndex {
    /// Build sorted indices for all features. O(n * p * log(n)) one-time cost.
    pub fn new(x_flat: &[f64], n_obs: usize, n_features: usize) -> Self {
        let sorted_indices = (0..n_features)
            .map(|feat| {
                let mut idx_val: Vec<(u32, f64)> = (0..n_obs as u32)
                    .map(|i| (i, x_flat[i as usize * n_features + feat]))
                    .collect();
                idx_val.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                idx_val.into_iter().map(|(i, _)| i).collect()
            })
            .collect();
        SortedColumnIndex { sorted_indices }
    }

    pub fn get_sorted(&self, feature: usize) -> &[u32] {
        &self.sorted_indices[feature]
    }
}

/// Result of finding the best split for a leaf.
pub struct SplitSearchResult {
    pub feature: usize,
    pub cutpoint: f64,
    pub left_stats: SuffStats,
    pub right_stats: SuffStats,
    pub gain: f64,
}

/// Per-leaf sorted indices: for each feature, the observation indices belonging
/// to this leaf, maintained in sorted-by-feature-value order. Eliminates the
/// 97% skip rate in find_best_split.
struct LeafSortedIndices {
    /// indices_by_feature[feat] = Vec<u32> of obs indices in this leaf,
    /// sorted by ascending x[., feat]
    indices_by_feature: Vec<Vec<u32>>,
}

/// Tracks per-tree leaf assignments and sufficient statistics.
/// Avoids recomputing observation partitions during split evaluation.
pub struct ForestTracker {
    /// leaf_assignment[tree_idx][obs_idx] = leaf_id for that observation
    leaf_assignment: Vec<Vec<u32>>,
    /// Dense Vec leaf stats: leaf_stats[tree_idx][leaf_id] = Some(stats) for active leaves
    leaf_stats: Vec<Vec<Option<SuffStats>>>,
    /// Next available leaf_id per tree
    next_leaf_id: Vec<u32>,
    pub num_trees: usize,
    n_obs: usize,
    /// Per-leaf sorted indices for the CURRENT tree being processed.
    /// Rebuilt on reset_tree / refresh_leaf_stats, updated on apply_split / apply_prune.
    /// leaf_sorted[leaf_id] = Some(LeafSortedIndices) for active leaves.
    leaf_sorted: Vec<Option<LeafSortedIndices>>,
    /// Number of features (needed for per-leaf index maintenance)
    n_features: usize,
}

impl ForestTracker {
    /// Initialize: all observations in leaf 0 for every tree.
    pub fn new(num_trees: usize, n_obs: usize, n_features: usize, residuals: &[f64]) -> Self {
        let mut leaf_stats_vec = Vec::with_capacity(num_trees);
        let mut leaf_assignment_vec = Vec::with_capacity(num_trees);
        let mut next_leaf_id_vec = Vec::with_capacity(num_trees);

        let root_stats = {
            let mut s = SuffStats::new();
            for &r in residuals {
                s.add(r);
            }
            s
        };

        for _ in 0..num_trees {
            leaf_assignment_vec.push(vec![0u32; n_obs]);
            let mut stats_vec = vec![None; LEAF_VEC_INITIAL_CAP];
            stats_vec[0] = Some(root_stats.clone());
            leaf_stats_vec.push(stats_vec);
            next_leaf_id_vec.push(1);
        }

        ForestTracker {
            leaf_assignment: leaf_assignment_vec,
            leaf_stats: leaf_stats_vec,
            next_leaf_id: next_leaf_id_vec,
            num_trees,
            n_obs,
            leaf_sorted: Vec::new(), // built lazily on first reset_tree/refresh_leaf_stats
            n_features,
        }
    }

    /// Ensure the leaf_stats vec for a tree is large enough for leaf_id.
    #[inline]
    fn ensure_leaf_stats_capacity(&mut self, tree_idx: usize, leaf_id: u32) {
        let needed = leaf_id as usize + 1;
        let stats = &mut self.leaf_stats[tree_idx];
        if needed > stats.len() {
            stats.resize(needed, None);
        }
    }

    /// Ensure the leaf_sorted vec is large enough for leaf_id.
    #[inline]
    fn ensure_leaf_sorted_capacity(&mut self, leaf_id: u32) {
        let needed = leaf_id as usize + 1;
        if needed > self.leaf_sorted.len() {
            self.leaf_sorted.resize_with(needed, || None);
        }
    }

    /// Whether per-leaf sorted indices are enabled (n_obs >= threshold).
    #[inline]
    fn use_leaf_sorted(&self) -> bool {
        self.n_obs >= LEAF_SORTED_MIN_OBS
    }

    /// Build per-leaf sorted indices for a given tree from scratch.
    /// Uses the global SortedColumnIndex: for each feature, scan the global
    /// sorted order and bucket observations by their leaf assignment.
    /// O(n_obs * n_features) total.
    fn build_leaf_sorted_indices(
        &mut self,
        tree_idx: usize,
        sorted_cols: &SortedColumnIndex,
    ) {
        if !self.use_leaf_sorted() {
            self.leaf_sorted.clear();
            return;
        }

        let n_features = self.n_features;
        let next_id = self.next_leaf_id[tree_idx] as usize;
        let assignment = &self.leaf_assignment[tree_idx];

        // Allocate leaf_sorted up to next_id
        self.leaf_sorted.clear();
        self.leaf_sorted.resize_with(next_id, || None);

        // First pass: determine which leaf IDs are active and allocate their vecs
        // We can check leaf_stats to find active leaves
        for lid in 0..next_id {
            if let Some(ref stats) = self.leaf_stats[tree_idx].get(lid).and_then(|s| s.as_ref()) {
                if stats.n > 0 {
                    self.leaf_sorted[lid] = Some(LeafSortedIndices {
                        indices_by_feature: vec![Vec::with_capacity(stats.n); n_features],
                    });
                }
            }
        }

        // Second pass: for each feature, walk the global sorted order and distribute
        for feat in 0..n_features {
            let sorted = sorted_cols.get_sorted(feat);
            for &obs_idx in sorted {
                let lid = assignment[obs_idx as usize] as usize;
                if lid < self.leaf_sorted.len() {
                    if let Some(ref mut leaf_si) = self.leaf_sorted[lid] {
                        leaf_si.indices_by_feature[feat].push(obs_idx);
                    }
                }
            }
        }
    }

    /// Reset a tree: all observations assigned to leaf 0 with fresh stats.
    pub fn reset_tree(
        &mut self,
        tree_idx: usize,
        residuals: &[f64],
        sorted_cols: &SortedColumnIndex,
    ) {
        let assignment = &mut self.leaf_assignment[tree_idx];
        for a in assignment.iter_mut() {
            *a = 0;
        }

        let mut root_stats = SuffStats::new();
        for &r in residuals {
            root_stats.add(r);
        }

        // Clear all leaf stats, set leaf 0
        for slot in self.leaf_stats[tree_idx].iter_mut() {
            *slot = None;
        }
        self.ensure_leaf_stats_capacity(tree_idx, 0);
        self.leaf_stats[tree_idx][0] = Some(root_stats);
        self.next_leaf_id[tree_idx] = 1;

        // Build per-leaf sorted indices (all obs in leaf 0)
        self.build_leaf_sorted_indices(tree_idx, sorted_cols);
    }

    /// Recompute leaf_stats from current leaf_assignment and residuals.
    /// Also rebuilds per-leaf sorted indices.
    /// Used when residuals change but leaf assignment has not.
    pub fn refresh_leaf_stats(
        &mut self,
        tree_idx: usize,
        residuals: &[f64],
        sorted_cols: &SortedColumnIndex,
    ) {
        let assignment = &self.leaf_assignment[tree_idx];
        let next_id = self.next_leaf_id[tree_idx] as usize;

        // Clear existing stats
        let stats = &mut self.leaf_stats[tree_idx];
        if stats.len() < next_id {
            stats.resize(next_id, None);
        }
        for slot in stats.iter_mut() {
            *slot = None;
        }

        // Rebuild from assignments
        for (i, &leaf_id) in assignment.iter().enumerate() {
            let lid = leaf_id as usize;
            if lid >= stats.len() {
                stats.resize(lid + 1, None);
            }
            match &mut stats[lid] {
                Some(s) => s.add(residuals[i]),
                slot @ None => {
                    let mut s = SuffStats::new();
                    s.add(residuals[i]);
                    *slot = Some(s);
                }
            }
        }

        // Rebuild per-leaf sorted indices
        self.build_leaf_sorted_indices(tree_idx, sorted_cols);
    }

    /// Get the stats for a given leaf.
    pub fn get_leaf_stats(&self, tree_idx: usize, leaf_id: u32) -> Option<&SuffStats> {
        let lid = leaf_id as usize;
        self.leaf_stats[tree_idx].get(lid).and_then(|s| s.as_ref())
    }

    /// Get all leaf ids for a tree.
    pub fn leaf_ids(&self, tree_idx: usize) -> Vec<u32> {
        self.leaf_stats[tree_idx]
            .iter()
            .enumerate()
            .filter_map(|(i, s)| if s.is_some() { Some(i as u32) } else { None })
            .collect()
    }

    /// Get the observation indices belonging to a particular leaf.
    pub fn leaf_indices(&self, tree_idx: usize, leaf_id: u32) -> Vec<usize> {
        self.leaf_assignment[tree_idx]
            .iter()
            .enumerate()
            .filter_map(|(i, &lid)| if lid == leaf_id { Some(i) } else { None })
            .collect()
    }

    /// Get the leaf assignment for a specific observation.
    pub fn leaf_assignment_at(&self, tree_idx: usize, obs_idx: usize) -> u32 {
        self.leaf_assignment[tree_idx][obs_idx]
    }

    /// Find the best split for a given leaf using per-leaf sorted indices.
    /// Scans only the observations in this leaf (O(n_leaf * n_features)),
    /// eliminating the 97% skip rate of the old global-scan approach.
    pub fn find_best_split(
        &self,
        tree_idx: usize,
        leaf_id: u32,
        _sorted_cols: &SortedColumnIndex,
        x_flat: &[f64],
        residuals: &[f64],
        n_features: usize,
        sigma: f64,
        tau: f64,
    ) -> Option<SplitSearchResult> {
        let lid = leaf_id as usize;
        let total_stats = match self.leaf_stats[tree_idx].get(lid).and_then(|s| s.as_ref()) {
            Some(s) => s.clone(),
            None => return None,
        };

        if total_stats.n < 5 {
            return None;
        }

        // Use per-leaf sorted indices if available, fall back to assignment scan
        let leaf_si = if lid < self.leaf_sorted.len() {
            self.leaf_sorted[lid].as_ref()
        } else {
            None
        };

        let mut best_gain = f64::NEG_INFINITY;
        let mut best_result: Option<SplitSearchResult> = None;

        if let Some(lsi) = leaf_si {
            // Fast path: per-leaf sorted indices — zero skips
            for feat in 0..n_features {
                let leaf_indices = &lsi.indices_by_feature[feat];
                let mut left = SuffStats::new();
                let mut prev_x_val = f64::NEG_INFINITY;

                for &obs_idx in leaf_indices {
                    let obs = obs_idx as usize;
                    let x_val = x_flat[obs * n_features + feat];
                    let y_val = residuals[obs];

                    if left.n >= 2 && x_val > prev_x_val {
                        let right = total_stats.sub(&left);
                        if right.n >= 2 {
                            let cutpoint = (prev_x_val + x_val) / 2.0;
                            let gain = split_log_likelihood_gain(&left, &right, sigma, tau);
                            if gain > best_gain {
                                best_gain = gain;
                                best_result = Some(SplitSearchResult {
                                    feature: feat,
                                    cutpoint,
                                    left_stats: left.clone(),
                                    right_stats: right,
                                    gain,
                                });
                            }
                        }
                    }

                    left.add(y_val);
                    prev_x_val = x_val;
                }
            }
        } else {
            // Fallback: global scan with skip (shouldn't happen in normal flow)
            let assignment = &self.leaf_assignment[tree_idx];
            for feat in 0..n_features {
                let sorted = _sorted_cols.get_sorted(feat);
                let mut left = SuffStats::new();
                let mut prev_x_val = f64::NEG_INFINITY;

                for &obs_idx in sorted {
                    let obs = obs_idx as usize;
                    if assignment[obs] != leaf_id {
                        continue;
                    }

                    let x_val = x_flat[obs * n_features + feat];
                    let y_val = residuals[obs];

                    if left.n >= 2 && x_val > prev_x_val {
                        let right = total_stats.sub(&left);
                        if right.n >= 2 {
                            let cutpoint = (prev_x_val + x_val) / 2.0;
                            let gain = split_log_likelihood_gain(&left, &right, sigma, tau);
                            if gain > best_gain {
                                best_gain = gain;
                                best_result = Some(SplitSearchResult {
                                    feature: feat,
                                    cutpoint,
                                    left_stats: left.clone(),
                                    right_stats: right,
                                    gain,
                                });
                            }
                        }
                    }

                    left.add(y_val);
                    prev_x_val = x_val;
                }
            }
        }

        best_result
    }

    /// Find the best split using rayon parallelism across features.
    /// Uses per-leaf sorted indices for zero-skip scanning.
    pub fn find_best_split_parallel(
        &self,
        tree_idx: usize,
        leaf_id: u32,
        _sorted_cols: &SortedColumnIndex,
        x_flat: &[f64],
        residuals: &[f64],
        n_features: usize,
        sigma: f64,
        tau: f64,
        pool: &rayon::ThreadPool,
    ) -> Option<SplitSearchResult> {
        let lid = leaf_id as usize;
        let total_stats = match self.leaf_stats[tree_idx].get(lid).and_then(|s| s.as_ref()) {
            Some(s) => s.clone(),
            None => return None,
        };

        if total_stats.n < 5 {
            return None;
        }

        // Use per-leaf sorted indices if available
        let leaf_si = if lid < self.leaf_sorted.len() {
            self.leaf_sorted[lid].as_ref()
        } else {
            None
        };

        let assignment = &self.leaf_assignment[tree_idx];

        // Parallel per-feature scan
        let results: Vec<Option<SplitSearchResult>> = pool.install(|| {
            use rayon::prelude::*;
            (0..n_features)
                .into_par_iter()
                .map(|feat| {
                    let mut left = SuffStats::new();
                    let mut prev_x_val = f64::NEG_INFINITY;
                    let mut best_gain = f64::NEG_INFINITY;
                    let mut best: Option<SplitSearchResult> = None;

                    if let Some(lsi) = leaf_si {
                        // Fast path: per-leaf sorted indices
                        let leaf_indices = &lsi.indices_by_feature[feat];
                        for &obs_idx in leaf_indices {
                            let obs = obs_idx as usize;
                            let x_val = x_flat[obs * n_features + feat];
                            let y_val = residuals[obs];

                            if left.n >= 2 && x_val > prev_x_val {
                                let right = total_stats.sub(&left);
                                if right.n >= 2 {
                                    let cutpoint = (prev_x_val + x_val) / 2.0;
                                    let gain = split_log_likelihood_gain(&left, &right, sigma, tau);
                                    if gain > best_gain {
                                        best_gain = gain;
                                        best = Some(SplitSearchResult {
                                            feature: feat,
                                            cutpoint,
                                            left_stats: left.clone(),
                                            right_stats: right,
                                            gain,
                                        });
                                    }
                                }
                            }

                            left.add(y_val);
                            prev_x_val = x_val;
                        }
                    } else {
                        // Fallback: global scan with skip
                        let sorted = _sorted_cols.get_sorted(feat);
                        for &obs_idx in sorted {
                            let obs = obs_idx as usize;
                            if assignment[obs] != leaf_id {
                                continue;
                            }

                            let x_val = x_flat[obs * n_features + feat];
                            let y_val = residuals[obs];

                            if left.n >= 2 && x_val > prev_x_val {
                                let right = total_stats.sub(&left);
                                if right.n >= 2 {
                                    let cutpoint = (prev_x_val + x_val) / 2.0;
                                    let gain = split_log_likelihood_gain(&left, &right, sigma, tau);
                                    if gain > best_gain {
                                        best_gain = gain;
                                        best = Some(SplitSearchResult {
                                            feature: feat,
                                            cutpoint,
                                            left_stats: left.clone(),
                                            right_stats: right,
                                            gain,
                                        });
                                    }
                                }
                            }

                            left.add(y_val);
                            prev_x_val = x_val;
                        }
                    }

                    best
                })
                .collect()
        });

        // Pick the best across all features
        let mut best_gain = f64::NEG_INFINITY;
        let mut best_result: Option<SplitSearchResult> = None;
        for r in results.into_iter().flatten() {
            if r.gain > best_gain {
                best_gain = r.gain;
                best_result = Some(r);
            }
        }

        best_result
    }

    /// Apply a split: observations in leaf_id with x[feat] <= cutpoint go to left,
    /// others go to right. Returns (left_id, right_id).
    /// Also maintains per-leaf sorted indices incrementally.
    pub fn apply_split(
        &mut self,
        tree_idx: usize,
        leaf_id: u32,
        feature: usize,
        cutpoint: f64,
        x_flat: &[f64],
        n_features: usize,
        left_stats: SuffStats,
        right_stats: SuffStats,
    ) -> (u32, u32) {
        let left_id = self.next_leaf_id[tree_idx];
        let right_id = left_id + 1;
        self.next_leaf_id[tree_idx] = right_id + 1;

        let assignment = &mut self.leaf_assignment[tree_idx];
        for i in 0..self.n_obs {
            if assignment[i] == leaf_id {
                let x_val = x_flat[i * n_features + feature];
                assignment[i] = if x_val <= cutpoint { left_id } else { right_id };
            }
        }

        // Update dense leaf stats
        self.ensure_leaf_stats_capacity(tree_idx, right_id);
        self.leaf_stats[tree_idx][leaf_id as usize] = None;
        self.leaf_stats[tree_idx][left_id as usize] = Some(left_stats);
        self.leaf_stats[tree_idx][right_id as usize] = Some(right_stats);

        // Maintain per-leaf sorted indices incrementally (only when enabled)
        if !self.use_leaf_sorted() {
            return (left_id, right_id);
        }

        self.ensure_leaf_sorted_capacity(right_id);
        let lid = leaf_id as usize;

        if lid < self.leaf_sorted.len() {
            if let Some(parent_si) = self.leaf_sorted[lid].take() {
                let n_feats = parent_si.indices_by_feature.len();

                // For the split feature: partition is trivial since indices are sorted
                // by that feature's value. All obs with x <= cutpoint go left, rest right.
                // For other features: scan and distribute based on the new assignment.
                let mut left_si = LeafSortedIndices {
                    indices_by_feature: Vec::with_capacity(n_feats),
                };
                let mut right_si = LeafSortedIndices {
                    indices_by_feature: Vec::with_capacity(n_feats),
                };

                let assignment = &self.leaf_assignment[tree_idx];

                for feat in 0..n_feats {
                    let parent_indices = &parent_si.indices_by_feature[feat];
                    let mut left_vec = Vec::new();
                    let mut right_vec = Vec::new();

                    if feat == feature {
                        // Split feature: partition point is where x > cutpoint
                        // Since sorted by this feature, left is prefix, right is suffix
                        for &idx in parent_indices {
                            let x_val = x_flat[idx as usize * n_features + feat];
                            if x_val <= cutpoint {
                                left_vec.push(idx);
                            } else {
                                right_vec.push(idx);
                            }
                        }
                    } else {
                        // Other features: check assignment (already updated)
                        for &idx in parent_indices {
                            if assignment[idx as usize] == left_id {
                                left_vec.push(idx);
                            } else {
                                right_vec.push(idx);
                            }
                        }
                    }

                    left_si.indices_by_feature.push(left_vec);
                    right_si.indices_by_feature.push(right_vec);
                }

                self.leaf_sorted[left_id as usize] = Some(left_si);
                self.leaf_sorted[right_id as usize] = Some(right_si);
            }
        }

        (left_id, right_id)
    }

    /// Prune: merge left_id and right_id back into a single parent leaf.
    /// Returns the new parent leaf_id.
    /// Note: does NOT maintain per-leaf sorted indices (no x_flat available).
    /// Use apply_prune_with_x when x_flat is available for proper index maintenance.
    pub fn apply_prune(
        &mut self,
        tree_idx: usize,
        left_id: u32,
        right_id: u32,
        residuals: &[f64],
    ) -> u32 {
        let parent_id = self.next_leaf_id[tree_idx];
        self.next_leaf_id[tree_idx] = parent_id + 1;

        let mut parent_stats = SuffStats::new();
        let assignment = &mut self.leaf_assignment[tree_idx];

        for i in 0..self.n_obs {
            if assignment[i] == left_id || assignment[i] == right_id {
                assignment[i] = parent_id;
                parent_stats.add(residuals[i]);
            }
        }

        self.ensure_leaf_stats_capacity(tree_idx, parent_id);
        self.leaf_stats[tree_idx][left_id as usize] = None;
        self.leaf_stats[tree_idx][right_id as usize] = None;
        self.leaf_stats[tree_idx][parent_id as usize] = Some(parent_stats);

        // Clean up children's sorted indices if enabled
        if self.use_leaf_sorted() {
            self.ensure_leaf_sorted_capacity(parent_id);
            let lid_l = left_id as usize;
            let lid_r = right_id as usize;
            if lid_l < self.leaf_sorted.len() { self.leaf_sorted[lid_l] = None; }
            if lid_r < self.leaf_sorted.len() { self.leaf_sorted[lid_r] = None; }
        }

        parent_id
    }

    /// Prune with x_flat available for proper merge-sort of per-leaf indices.
    pub fn apply_prune_with_x(
        &mut self,
        tree_idx: usize,
        left_id: u32,
        right_id: u32,
        residuals: &[f64],
        x_flat: &[f64],
    ) -> u32 {
        let parent_id = self.next_leaf_id[tree_idx];
        self.next_leaf_id[tree_idx] = parent_id + 1;

        let mut parent_stats = SuffStats::new();
        let assignment = &mut self.leaf_assignment[tree_idx];

        for i in 0..self.n_obs {
            if assignment[i] == left_id || assignment[i] == right_id {
                assignment[i] = parent_id;
                parent_stats.add(residuals[i]);
            }
        }

        self.ensure_leaf_stats_capacity(tree_idx, parent_id);
        self.leaf_stats[tree_idx][left_id as usize] = None;
        self.leaf_stats[tree_idx][right_id as usize] = None;
        self.leaf_stats[tree_idx][parent_id as usize] = Some(parent_stats);

        // Merge per-leaf sorted indices with proper merge-sort (only when enabled)
        if !self.use_leaf_sorted() {
            return parent_id;
        }

        self.ensure_leaf_sorted_capacity(parent_id);
        let lid_l = left_id as usize;
        let lid_r = right_id as usize;

        let left_si = if lid_l < self.leaf_sorted.len() { self.leaf_sorted[lid_l].take() } else { None };
        let right_si = if lid_r < self.leaf_sorted.len() { self.leaf_sorted[lid_r].take() } else { None };

        if let (Some(lsi), Some(rsi)) = (left_si, right_si) {
            let n_features = self.n_features;
            let n_feats = lsi.indices_by_feature.len();
            let mut parent_si = LeafSortedIndices {
                indices_by_feature: Vec::with_capacity(n_feats),
            };

            for feat in 0..n_feats {
                let left_indices = &lsi.indices_by_feature[feat];
                let right_indices = &rsi.indices_by_feature[feat];
                let mut merged = Vec::with_capacity(left_indices.len() + right_indices.len());

                // Proper merge-sort: both sorted by x[., feat]
                let mut li = 0;
                let mut ri = 0;
                while li < left_indices.len() && ri < right_indices.len() {
                    let lx = x_flat[left_indices[li] as usize * n_features + feat];
                    let rx = x_flat[right_indices[ri] as usize * n_features + feat];
                    if lx <= rx {
                        merged.push(left_indices[li]);
                        li += 1;
                    } else {
                        merged.push(right_indices[ri]);
                        ri += 1;
                    }
                }
                merged.extend_from_slice(&left_indices[li..]);
                merged.extend_from_slice(&right_indices[ri..]);

                parent_si.indices_by_feature.push(merged);
            }

            self.leaf_sorted[parent_id as usize] = Some(parent_si);
        }

        parent_id
    }

    /// Resample leaf values from the posterior for all leaves of a tree.
    /// Returns leaf_id -> new mu value (using HashMap for compatibility).
    pub fn resample_leaf_values(
        &self,
        tree_idx: usize,
        sigma: f64,
        tau: f64,
        rng: &mut Xoshiro256StarStar,
    ) -> HashMap<u32, f64> {
        let sigma2 = sigma * sigma;
        let tau2 = tau * tau;
        let mut values = HashMap::new();

        for (leaf_id, slot) in self.leaf_stats[tree_idx].iter().enumerate() {
            if let Some(stats) = slot {
                if stats.n > 0 {
                    let n = stats.n as f64;
                    let post_var = 1.0 / (n / sigma2 + 1.0 / tau2);
                    let post_mean = post_var * (stats.sum_y / sigma2);
                    let value = post_mean + post_var.sqrt() * randn(rng);
                    values.insert(leaf_id as u32, value);
                }
            }
        }

        values
    }
}

/// Log marginal likelihood gain from splitting (same formula as sampler.rs).
fn split_log_likelihood_gain(left: &SuffStats, right: &SuffStats, sigma: f64, tau: f64) -> f64 {
    let sigma2 = sigma * sigma;
    let tau2 = tau * tau;

    let ll = |s: &SuffStats| -> f64 {
        let n = s.n as f64;
        let post_var = 1.0 / (n / sigma2 + 1.0 / tau2);
        let post_mean = post_var * s.sum_y / sigma2;
        0.5 * (post_var / tau2).ln() + 0.5 * post_mean * post_mean / post_var
    };

    ll(left) + ll(right)
}

/// Box-Muller normal sample (same as sampler.rs).
fn randn(rng: &mut Xoshiro256StarStar) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-30);
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
