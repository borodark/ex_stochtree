/// A single decision tree node
#[derive(Clone, Debug)]
pub enum TreeNode {
    Leaf {
        value: f64,
    },
    Internal {
        feature: usize,
        cutpoint: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

impl TreeNode {
    pub fn new_leaf(value: f64) -> Self {
        TreeNode::Leaf { value }
    }

    /// Predict for a single observation
    pub fn predict(&self, x: &[f64]) -> f64 {
        match self {
            TreeNode::Leaf { value } => *value,
            TreeNode::Internal { feature, cutpoint, left, right } => {
                if x[*feature] <= *cutpoint {
                    left.predict(x)
                } else {
                    right.predict(x)
                }
            }
        }
    }

    /// Count splits per feature (for variable importance)
    pub fn split_counts(&self, counts: &mut Vec<usize>) {
        match self {
            TreeNode::Leaf { .. } => {},
            TreeNode::Internal { feature, left, right, .. } => {
                if *feature < counts.len() {
                    counts[*feature] += 1;
                }
                left.split_counts(counts);
                right.split_counts(counts);
            }
        }
    }

    /// Depth of the tree
    pub fn depth(&self) -> usize {
        match self {
            TreeNode::Leaf { .. } => 0,
            TreeNode::Internal { left, right, .. } => {
                1 + left.depth().max(right.depth())
            }
        }
    }

    /// Number of leaf nodes
    pub fn num_leaves(&self) -> usize {
        match self {
            TreeNode::Leaf { .. } => 1,
            TreeNode::Internal { left, right, .. } => {
                left.num_leaves() + right.num_leaves()
            }
        }
    }
}

/// Sufficient statistics for a set of observations in a leaf
#[derive(Clone, Debug)]
pub struct SuffStats {
    pub n: usize,
    pub sum_y: f64,
    pub sum_y2: f64,
}

impl SuffStats {
    pub fn new() -> Self {
        SuffStats { n: 0, sum_y: 0.0, sum_y2: 0.0 }
    }

    pub fn add(&mut self, y: f64) {
        self.n += 1;
        self.sum_y += y;
        self.sum_y2 += y * y;
    }

    pub fn mean(&self) -> f64 {
        if self.n == 0 { 0.0 } else { self.sum_y / self.n as f64 }
    }
}
