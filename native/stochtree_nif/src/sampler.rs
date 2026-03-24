use crate::tree::{TreeNode, SuffStats};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256StarStar;

/// A trained BART forest: collection of tree ensembles (one per MCMC sample)
pub struct BartForest {
    /// ensembles[i] = the forest at MCMC iteration i (post-burnin only)
    pub ensembles: Vec<Vec<TreeNode>>,
    /// sigma samples (one per saved iteration)
    pub sigma_samples: Vec<f64>,
    pub n_features: usize,
}

impl BartForest {
    /// Predict for new data: returns [n_samples][n_obs]
    pub fn predict(&self, x_flat: &[f64], n_obs: usize, n_features: usize) -> Vec<Vec<f64>> {
        self.ensembles.iter().map(|trees| {
            (0..n_obs).map(|i| {
                let row = &x_flat[i * n_features..(i + 1) * n_features];
                trees.iter().map(|t| t.predict(row)).sum()
            }).collect()
        }).collect()
    }

    /// Variable importance: split count per feature across all saved ensembles
    pub fn variable_importance(&self) -> Vec<(usize, usize)> {
        let mut counts = vec![0usize; self.n_features];
        for ensemble in &self.ensembles {
            for tree in ensemble {
                tree.split_counts(&mut counts);
            }
        }
        let mut result: Vec<(usize, usize)> = counts.into_iter().enumerate().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }
}

/// Sample from standard normal using Box-Muller
fn randn(rng: &mut Xoshiro256StarStar) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-30);
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Sample from Gamma(shape, rate) using Marsaglia-Tsang
fn sample_gamma(shape: f64, rate: f64, rng: &mut Xoshiro256StarStar) -> f64 {
    if shape < 1.0 {
        let u: f64 = rng.gen();
        return sample_gamma(shape + 1.0, rate, rng) * u.powf(1.0 / shape);
    }

    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        let z = randn(rng);
        let v = (1.0 + c * z).powi(3);
        if v <= 0.0 { continue; }
        let u: f64 = rng.gen();
        if u < 1.0 - 0.0331 * z.powi(4) {
            return d * v / rate;
        }
        if u.ln() < 0.5 * z * z + d * (1.0 - v + v.ln()) {
            return d * v / rate;
        }
    }
}

/// Fit a BART model
pub fn fit_bart(
    x_flat: &[f64],
    y: &[f64],
    n_obs: usize,
    n_features: usize,
    num_trees: usize,
    num_gfr: usize,
    num_mcmc: usize,
    seed: u64,
) -> BartForest {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);

    // Hyperparameters
    let alpha = 0.95_f64;
    let beta = 2.0_f64;
    let y_mean: f64 = y.iter().sum::<f64>() / n_obs as f64;
    let y_var: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum::<f64>() / (n_obs as f64 - 1.0);
    let y_sd = y_var.sqrt();

    // BART leaf prior: each leaf ~ N(0, tau^2)
    // Chipman et al. (2010): tau = y_sd / (k * sqrt(num_trees))
    // k=1 is weak (flexible), k=3 is strong (regularized)
    let k = 2.0_f64;
    let tau = y_sd / (k * (num_trees as f64).sqrt());

    // sigma prior: nu=3, lambda = y_var (weakly informative)
    // Scale lambda by quantile of chi-squared so prior is centered near y_sd
    let nu = 3.0_f64;
    let q = 0.9_f64; // P(sigma < y_sd) = q
    let lambda = y_var * nu / 2.0; // approximate calibration

    // Initialize: each tree predicts y_mean / num_trees
    let init_val = y_mean / num_trees as f64;
    let mut trees: Vec<TreeNode> = (0..num_trees)
        .map(|_| TreeNode::new_leaf(init_val))
        .collect();

    let mut preds: Vec<Vec<f64>> = (0..num_trees)
        .map(|_| vec![init_val; n_obs])
        .collect();

    let mut sigma = y_sd;

    // Only save MCMC samples (skip GFR burn-in)
    let mut ensembles = Vec::with_capacity(num_mcmc);
    let mut sigma_samples = Vec::with_capacity(num_mcmc);

    let cutpoints = compute_cutpoints(x_flat, n_obs, n_features, 100);

    let total_iters = num_gfr + num_mcmc;
    for iter in 0..total_iters {
        let use_gfr = iter < num_gfr;

        // Backfitting: sweep through each tree
        for j in 0..num_trees {
            // Partial residual: r_i = y_i - sum_{k != j} preds[k][i]
            let residuals: Vec<f64> = (0..n_obs).map(|i| {
                let other_sum: f64 = (0..num_trees)
                    .filter(|&k| k != j)
                    .map(|k| preds[k][i])
                    .sum();
                y[i] - other_sum
            }).collect();

            if use_gfr {
                // GFR: greedy grow from root (structure + leaves)
                trees[j] = grow_from_root(
                    x_flat, &residuals, n_obs, n_features,
                    &cutpoints, sigma, tau, alpha, beta, 6, &mut rng,
                );
            } else {
                // MCMC: re-grow from root (XBART-style) then Gibbs-sample leaves
                trees[j] = grow_from_root(
                    x_flat, &residuals, n_obs, n_features,
                    &cutpoints, sigma, tau, alpha, beta, 5, &mut rng,
                );
            }

            // Always resample leaf values from posterior (Gibbs step)
            resample_leaves(&mut trees[j], x_flat, &residuals, n_features,
                            &(0..n_obs).collect::<Vec<_>>(), sigma, tau, &mut rng);

            // Update predictions
            for i in 0..n_obs {
                let row = &x_flat[i * n_features..(i + 1) * n_features];
                preds[j][i] = trees[j].predict(row);
            }
        }

        // Sample sigma^2 from inverse-gamma posterior
        let rss: f64 = (0..n_obs).map(|i| {
            let pred_sum: f64 = (0..num_trees).map(|k| preds[k][i]).sum();
            (y[i] - pred_sum).powi(2)
        }).sum();

        let post_shape = nu / 2.0 + n_obs as f64 / 2.0;
        let post_scale = nu * lambda / 2.0 + rss / 2.0;
        let gamma_sample = sample_gamma(post_shape, 1.0 / post_scale, &mut rng);
        sigma = (1.0 / gamma_sample).sqrt().max(1e-10);

        // Save only MCMC samples (after GFR burn-in)
        if !use_gfr {
            ensembles.push(trees.clone());
            sigma_samples.push(sigma);
        }
    }

    BartForest {
        ensembles,
        sigma_samples,
        n_features,
    }
}

/// Grow a tree from root greedily (GFR / XBART style)
fn grow_from_root(
    x_flat: &[f64],
    y: &[f64],
    n_obs: usize,
    n_features: usize,
    cutpoints: &[Vec<f64>],
    sigma: f64,
    tau: f64,
    alpha: f64,
    beta: f64,
    max_depth: usize,
    rng: &mut Xoshiro256StarStar,
) -> TreeNode {
    let indices: Vec<usize> = (0..n_obs).collect();
    grow_recursive(x_flat, y, n_features, cutpoints, sigma, tau, alpha, beta, max_depth, 0, &indices, rng)
}

fn grow_recursive(
    x_flat: &[f64],
    y: &[f64],
    n_features: usize,
    cutpoints: &[Vec<f64>],
    sigma: f64,
    tau: f64,
    alpha: f64,
    beta: f64,
    max_depth: usize,
    depth: usize,
    indices: &[usize],
    rng: &mut Xoshiro256StarStar,
) -> TreeNode {
    let n = indices.len();

    if n < 5 || depth >= max_depth {
        return sample_leaf(y, indices, sigma, tau, rng);
    }

    let p_split = alpha * (1.0 + depth as f64).powf(-beta);
    if rng.gen::<f64>() > p_split {
        return sample_leaf(y, indices, sigma, tau, rng);
    }

    // Find best split
    let mut best_gain = f64::NEG_INFINITY;
    let mut best_feature = 0;
    let mut best_cutpoint = 0.0;

    for feat in 0..n_features {
        for &cp in &cutpoints[feat] {
            let (left_stats, right_stats) = split_stats(x_flat, y, n_features, indices, feat, cp);
            if left_stats.n < 2 || right_stats.n < 2 { continue; }

            let gain = split_log_likelihood(&left_stats, &right_stats, sigma, tau);
            if gain > best_gain {
                best_gain = gain;
                best_feature = feat;
                best_cutpoint = cp;
            }
        }
    }

    if best_gain == f64::NEG_INFINITY {
        return sample_leaf(y, indices, sigma, tau, rng);
    }

    let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = indices.iter()
        .partition(|&&i| x_flat[i * n_features + best_feature] <= best_cutpoint);

    if left_idx.is_empty() || right_idx.is_empty() {
        return sample_leaf(y, indices, sigma, tau, rng);
    }

    let left = grow_recursive(x_flat, y, n_features, cutpoints, sigma, tau, alpha, beta, max_depth, depth + 1, &left_idx, rng);
    let right = grow_recursive(x_flat, y, n_features, cutpoints, sigma, tau, alpha, beta, max_depth, depth + 1, &right_idx, rng);

    TreeNode::Internal {
        feature: best_feature,
        cutpoint: best_cutpoint,
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// MCMC step: propose grow or prune via Metropolis-Hastings
fn mcmc_step(
    tree: &TreeNode,
    x_flat: &[f64],
    y: &[f64],
    n_obs: usize,
    n_features: usize,
    cutpoints: &[Vec<f64>],
    sigma: f64,
    tau: f64,
    alpha: f64,
    beta: f64,
    rng: &mut Xoshiro256StarStar,
) -> TreeNode {
    let n_leaves = tree.num_leaves();

    // Choose grow or prune (50/50 when possible)
    let can_grow = n_leaves > 0;
    let can_prune = n_leaves > 1 && tree.depth() > 0;

    let do_grow = if can_grow && can_prune {
        rng.gen::<f64>() < 0.5
    } else {
        can_grow
    };

    if do_grow {
        // GROW: pick a random leaf, propose a split
        // For simplicity: re-grow from root with GFR (approximation of proper MH)
        // This is the XBART strategy — each MCMC iteration re-grows the tree
        let indices: Vec<usize> = (0..n_obs).collect();
        grow_recursive(x_flat, y, n_features, cutpoints, sigma, tau, alpha, beta, 6, 0, &indices, rng)
    } else {
        // PRUNE: return a single-leaf tree (aggressive pruning)
        sample_leaf(y, &(0..n_obs).collect::<Vec<_>>(), sigma, tau, rng)
    }
}

/// Resample leaf values given current tree structure and residuals
fn resample_leaves(
    tree: &mut TreeNode,
    x_flat: &[f64],
    y: &[f64],
    n_features: usize,
    indices: &[usize],
    sigma: f64,
    tau: f64,
    rng: &mut Xoshiro256StarStar,
) {
    match tree {
        TreeNode::Leaf { ref mut value } => {
            // Resample leaf value from posterior
            let n = indices.len() as f64;
            let sum_y: f64 = indices.iter().map(|&i| y[i]).sum();
            let sigma2 = sigma * sigma;
            let tau2 = tau * tau;
            let post_var = 1.0 / (n / sigma2 + 1.0 / tau2);
            let post_mean = post_var * (sum_y / sigma2);
            *value = post_mean + post_var.sqrt() * randn(rng);
        }
        TreeNode::Internal { feature, cutpoint, ref mut left, ref mut right } => {
            let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = indices.iter()
                .partition(|&&i| x_flat[i * n_features + *feature] <= *cutpoint);
            resample_leaves(left, x_flat, y, n_features, &left_idx, sigma, tau, rng);
            resample_leaves(right, x_flat, y, n_features, &right_idx, sigma, tau, rng);
        }
    }
}

/// Sample a leaf value from the Gaussian posterior
fn sample_leaf(
    y: &[f64],
    indices: &[usize],
    sigma: f64,
    tau: f64,
    rng: &mut Xoshiro256StarStar,
) -> TreeNode {
    let n = indices.len() as f64;
    let sum_y: f64 = indices.iter().map(|&i| y[i]).sum();

    let sigma2 = sigma * sigma;
    let tau2 = tau * tau;
    let post_var = 1.0 / (n / sigma2 + 1.0 / tau2);
    let post_mean = post_var * (sum_y / sigma2);

    let value = post_mean + post_var.sqrt() * randn(rng);
    TreeNode::new_leaf(value)
}

/// Compute sufficient statistics for a candidate split
fn split_stats(
    x_flat: &[f64],
    y: &[f64],
    n_features: usize,
    indices: &[usize],
    feature: usize,
    cutpoint: f64,
) -> (SuffStats, SuffStats) {
    let mut left = SuffStats::new();
    let mut right = SuffStats::new();

    for &i in indices {
        let xi = x_flat[i * n_features + feature];
        let yi = y[i];
        if xi <= cutpoint {
            left.add(yi);
        } else {
            right.add(yi);
        }
    }
    (left, right)
}

/// Log marginal likelihood gain from splitting
fn split_log_likelihood(
    left: &SuffStats,
    right: &SuffStats,
    sigma: f64,
    tau: f64,
) -> f64 {
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

/// Compute cutpoint candidates (quantiles of data per feature)
fn compute_cutpoints(
    x_flat: &[f64],
    n_obs: usize,
    n_features: usize,
    max_cutpoints: usize,
) -> Vec<Vec<f64>> {
    (0..n_features).map(|feat| {
        let mut vals: Vec<f64> = (0..n_obs)
            .map(|i| x_flat[i * n_features + feat])
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        vals.dedup();

        if vals.len() <= max_cutpoints {
            vals.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect()
        } else {
            let step = vals.len() / max_cutpoints;
            (0..max_cutpoints)
                .map(|i| {
                    let idx = (i * step).min(vals.len() - 2);
                    (vals[idx] + vals[idx + 1]) / 2.0
                })
                .collect()
        }
    }).collect()
}
