# test_visual_classifier.py (multi-case version with 5 test cases)

import numpy as np
from gradient_boosting import GradientBoostingClassifier
from dataset import make_moons, train_test_split, StandardScaler
from metrics import classification_report, cnfsn_mtrx, roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def cnfsn_mtrx_plt(y_true, y_pred, title):
    cm = cnfsn_mtrx(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {title}")
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_true)))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('prdctioned label')
    plt.tight_layout()
    plt.show()


def plot_roc(y_true, y_scores, title):
    fls_postv_rate, tru_postv_rate, _ = roc_curve(y_true, y_scores)
    auc_score = auc(fls_postv_rate, tru_postv_rate)
    plt.figure()
    plt.plot(fls_postv_rate, tru_postv_rate, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {title}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.grid()
    plt.show()


def dcsn_bndry_plt(X, y, model, title):
    if X.shape[1] > 2:
        print(f"Skipping decision boundary for {title} (more than 2 features)")
        return
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
    plt.contourf(xx, yy, Z, cmap=light_cmp, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=bold_cmp, edgecolor='k')
    plt.title(f"Decision Boundary - {title}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.show()


def make_classification(smpl_n, n_ftr, informative=2, redundant=0, class_sep=1.0, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    X = np.random.randn(smpl_n, n_ftr)
    true_weights = np.random.randn(n_ftr)
    signal = X[:, :informative] @ true_weights[:informative]
    scale = max(0.0, 1 - class_sep)
    noise = np.random.normal(scale=scale, size=signal.shape)
    scores = signal + noise
    y = (scores > np.median(scores)).astype(int)
    return X, y


def run_case(title, X, y):
    print(f"\n🧪 Running Test Case: {title}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = GradientBoostingClassifier(n_estmtr=500, ln_rate=0.02, depth_maximum=3, sub_smpl=0.8)
    model.fit_func(X_train, y_train)
    y_pred = model.prdction(X_test)
    y_scores = model.prdction_proba(X_test)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\n📈 Feature Importances:")
    for i, score in enumerate(model.ftr_imprtnc):
        print(f"Feature {i}: {score:.4f}")

    cnfsn_mtrx_plt(y_test, y_pred, title)
    plot_roc(y_test, y_scores, title)
    dcsn_bndry_plt(X_test, y_test, model, title)


def run_all_tests():
    # 1. Clean moons
    X1, y1 = make_moons(smpl_n=1000, noise=0.01, random_state=42)
    run_case("Clean Moons", X1, y1)

    # 2. Noisy moons
    X2, y2 = make_moons(smpl_n=1000, noise=0.25, random_state=42)
    run_case("Noisy Moons", X2, y2)

    # 3. Easy separation (class_sep=2)
    X3, y3 = make_classification(1000, 4, informative=4, redundant=0, class_sep=2, random_state=42)
    run_case("Easy Classification", X3, y3)

    # 4. Overlapping classes (class_sep=0.5)
    X4, y4 = make_classification(1000, 4, informative=2, redundant=2, class_sep=0.5, random_state=42)
    run_case("Overlapping Classes", X4, y4)

    # 5. High dimensional (10 features)
    X5, y5 = make_classification(1000, 10, informative=6, redundant=2, class_sep=1.0, random_state=42)
    run_case("High Dimensional", X5, y5)


if __name__ == "__main__":
    run_all_tests()
