# gradient_boosting.py

import numpy as np
from tree import RegressionTree  # Custom regression tree implementation
from dataset import make_moons, train_test_split, StandardScaler  # Custom data utilities
from metrics import classification_report, cnfsn_mtrx, roc_curve, auc  # Custom evaluation metrics
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

class GradientBoostingClassifier:
    def __init__(self, n_estmtr=500, ln_rate=0.02, depth_maximum=3, sub_smpl=0.8):
        # Initialize boosting parameters
        self.n_estmtr = n_estmtr
        self.ln_rate = ln_rate
        self.depth_maximum = depth_maximum
        self.sub_smpl = sub_smpl
        self.trees = []
        self.F0 = 0
        self.ftr_imprtnc = None

    def _prbablty(self, x):
        # Sigmoid activation function to convert raw scores to probabilities
        return 1 / (1 + np.exp(-x))

    def fit_func(self, X, y):
        # Model training using gradient boosting with subsampling
        smpl_n, n_ftr = X.shape
        self.ftr_imprtnc = np.zeros(n_ftr)
        ratio_of_pos = np.clip(np.mean(y), 1e-5, 1 - 1e-5)
        self.F0 = np.log(ratio_of_pos / (1 - ratio_of_pos))
        Fm = np.full(y.shape, self.F0)

        for _ in range(self.n_estmtr):
            p = self._prbablty(Fm)
            rsdl = y - p

            # Subsampling of training data
            indcs = np.random.choice(smpl_n, int(self.sub_smpl * smpl_n), replace=False)
            X_sub = X[indcs]
            rsdl_sub = rsdl[indcs]

            tree = RegressionTree(depth_maximum=self.depth_maximum)
            tree.fit_func(X_sub, rsdl_sub)
            pred = tree.prdction(X)
            Fm += self.ln_rate * pred
            self.trees.append(tree)

            # Track feature usage for importance
            self._ftr_updt_imprtncs(tree.root)

        self.ftr_imprtnc /= np.sum(self.ftr_imprtnc)

    def _ftr_updt_imprtncs(self, node):
        # Recursive feature importance tracker
        if node is None or node.is_leaf_node():
            return
        self.ftr_imprtnc[node.feature_index] += 1
        self._ftr_updt_imprtncs(node.left)
        self._ftr_updt_imprtncs(node.right)

    def prdction_proba(self, X):
        # Predict class probabilities using sigmoid on aggregated outputs
        Fm = np.full((X.shape[0],), self.F0)
        for tree in self.trees:
            Fm += self.ln_rate * tree.prdction(X)
        return self._prbablty(Fm)

    def prdction(self, X):
        # Predict binary class labels
        proba = self.prdction_proba(X)
        return (proba >= 0.5).astype(int)


def cnfsn_mtrx_plt(y_true, y_pred):
    # Plot a confusion matrix heatmap
    cm = cnfsn_mtrx(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.xlabel("prdctioned")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def roc_curv_plt(y_true, y_scores):
    # Plot ROC curve with AUC
    fls_postv_rate, tru_postv_rate, _ = roc_curve(y_true, y_scores)
    auc_of_roc_curve = auc(fls_postv_rate, tru_postv_rate)
    plt.figure()
    plt.plot(fls_postv_rate, tru_postv_rate, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {auc_of_roc_curve:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.show()


def dcsn_bndry_plt(X, y, model):
    # Plot decision boundary for 2D data
    h = 0.02
    rang_min_x, rang_max_x = X[:, 0].min() - 1, X[:, 0].max() + 1
    rang_min_y, rang_max_y = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(rang_min_x, rang_max_x, h),
                         np.arange(rang_min_y, rang_max_y, h))
    Z = model.prdction(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    light_cmp = ListedColormap(["#FFAAAA", "#AAFFAA"])
    bold_cmp = ListedColormap(["#FF0000", "#00FF00"])

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=light_cmp)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=bold_cmp, edgecolor='k')
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("\nRunning Final Boosted Gradient Boosting with Subsampling & Feature Importance...\n")
    X, y = make_moons(smpl_n=1000, noise=0.01, random_state=42)

    print("Class distribution:", dict(zip(*np.unique(y, return_counts=True))))

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = GradientBoostingClassifier(n_estmtr=500, ln_rate=0.02, depth_maximum=3, sub_smpl=0.8)
    clf.fit_func(X_train, y_train)

    preds = clf.prdction(X_test)
    scores = clf.prdction_proba(X_test)

    print("\nClassification Report (on test data):\n")
    print(classification_report(y_test, preds))

    print("\nFeature Importances (normalized):\n")
    for i, val in enumerate(clf.ftr_imprtnc):
        print(f"Feature {i}: {val:.4f}")

    cnfsn_mtrx_plt(y_test, preds)
    roc_curv_plt(y_test, scores)
    dcsn_bndry_plt(X_test, y_test, clf)
