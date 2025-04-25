# dataset.py

import numpy as np

# Pure Python version of make_moons (simplified)
def make_moons(smpl_n=100, noise=0.1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    smpl_n_out = smpl_n // 2
    smpl_n_in = smpl_n - smpl_n_out

    outer_circ_x = np.cos(np.linspace(0, np.pi, smpl_n_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, smpl_n_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, smpl_n_in))
    inner_circ_y = 0.5 - np.sin(np.linspace(0, np.pi, smpl_n_in))

    X = np.vstack([
        np.stack([outer_circ_x, outer_circ_y], axis=1),
        np.stack([inner_circ_x, inner_circ_y], axis=1)
    ])
    y = np.hstack([
        np.zeros(smpl_n_out, dtype=int),
        np.ones(smpl_n_in, dtype=int)
    ])

    X += noise * np.random.randn(*X.shape)
    return X, y

# Pure version of train_test_split

def train_test_split(X, y, test_size=0.25, random_state=None, stratify=None):
    if random_state is not None:
        np.random.seed(random_state)

    indcs = np.arange(X.shape[0])
    if stratify is not None:
        classes, y_indcs = np.unique(y, return_inverse=True)
        stratified_indcs = []
        for cls in np.unique(y):
            cls_indcs = indcs[y == cls]
            np.random.shuffle(cls_indcs)
            n_cls_test = int(np.floor(test_size * len(cls_indcs)))
            stratified_indcs.extend(cls_indcs[:n_cls_test])
        test_idx = np.array(stratified_indcs)
    else:
        np.random.shuffle(indcs)
        n_test = int(test_size * len(indcs))
        test_idx = indcs[:n_test]

    train_idx = np.setdiff1d(indcs, test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# Simple scaler
class StandardScaler:
    def fit_func(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit_func(X).transform(X)
