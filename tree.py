# tree.py (final version with TreeNode and is_leaf_node)

import numpy as np

class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class RegressionTree:
    def __init__(self, depth_maximum=2, mismpl_n_split=2):
        self.depth_maximum = depth_maximum
        self.mismpl_n_split = mismpl_n_split
        self.root = None

    def fit_func(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if (depth >= self.depth_maximum) or (num_samples < self.mismpl_n_split):
            return TreeNode(value=np.mean(y))

        best_split = self._best_split(X, y, num_features)
        if best_split is None:
            return TreeNode(value=np.mean(y))

        left_indcs = X[:, best_split["feature"]] < best_split["threshold"]
        right_indcs = ~left_indcs

        left_child = self._build_tree(X[left_indcs], y[left_indcs], depth + 1)
        right_child = self._build_tree(X[right_indcs], y[right_indcs], depth + 1)

        return TreeNode(
            feature_index=best_split["feature"],
            threshold=best_split["threshold"],
            left=left_child,
            right=right_child
        )

    def _best_split(self, X, y, num_features):
        best_mse = float("inf")
        best_split = None

        for feature_index in range(num_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] < threshold
                right_mask = ~left_mask
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue

                mse = self._calculate_mse(y[left_mask], y[right_mask])

                if mse < best_mse:
                    best_mse = mse
                    best_split = {
                        "feature": feature_index,
                        "threshold": threshold,
                        "mse": mse
                    }

        return best_split

    def _calculate_mse(self, left_y, right_y):
        mse_left = np.var(left_y) * len(left_y)
        mse_right = np.var(right_y) * len(right_y)
        return (mse_left + mse_right) / (len(left_y) + len(right_y))

    def prdction(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature_index] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
