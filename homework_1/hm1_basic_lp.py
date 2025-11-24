import numpy as np
from scipy.optimize import linprog
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class RobustLPClassifier:
    """
        A = [X_pos, 1],  B = [X_neg, 1]

        min  (1/m) * sum_i e_Ai  +  (1/k) * sum_j e_Bj
        s.t.
              -A w + gamma + 1 <= e_A
               B w - gamma + 1 <= e_B
              e_A, e_B >= 0

    Karar kuralı:
        f(x) = [x, 1]·w - gamma
        f(x) >= 0  ->  +1
        f(x) <  0  ->  -1
    """

    def __init__(self):
        self.w_ = None   # (n_features + 1,)  
        self.gamma_ = None

    def fit(self, X_pos, X_neg):
        X_pos = np.asarray(X_pos)
        X_neg = np.asarray(X_neg)

        n_pos, n_features = X_pos.shape
        n_neg = X_neg.shape[0]

        # Bias için 1'ler ekle
        A = np.hstack([X_pos, np.ones((n_pos, 1))])
        B = np.hstack([X_neg, np.ones((n_neg, 1))])

        d = n_features + 1              # w boyutu (bias dahil)
        n_vars = d + 1 + n_pos + n_neg  # w, gamma, e_A, e_B

        # Amaç fonksiyonu: (1/m) sum e_A + (1/k) sum e_B
        c = np.zeros(n_vars)
        idx_eA_start = d + 1
        idx_eB_start = d + 1 + n_pos
        c[idx_eA_start:idx_eA_start + n_pos] = 1.0 / n_pos
        c[idx_eB_start:idx_eB_start + n_neg] = 1.0 / n_neg

        # Eşitsizlikler A_ub x <= b_ub
        # Pozitif sınıf: -A w + gamma + 1 <= e_A
        #  -> [-A, 1, -I, 0] x <= -1
        A_pos = np.hstack([
            -A,
            np.ones((n_pos, 1)),
            -np.eye(n_pos),
            np.zeros((n_pos, n_neg)),
        ])
        b_pos = -np.ones(n_pos)

        # Negatif sınıf: B w - gamma + 1 <= e_B
        #  -> [B, -1, 0, -I] x <= -1
        A_neg = np.hstack([
            B,
            -np.ones((n_neg, 1)),
            np.zeros((n_neg, n_pos)),
            -np.eye(n_neg),
        ])
        b_neg = -np.ones(n_neg)

        A_ub = np.vstack([A_pos, A_neg])
        b_ub = np.concatenate([b_pos, b_neg])

        # Sınırlar: w ve gamma serbest, e_A ve e_B >= 0
        bounds = [(None, None)] * (d + 1) + [(0, None)] * (n_pos + n_neg)

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if not res.success:
            raise RuntimeError(f"LP çözülmedi: {res.message}")

        self.w_ = res.x[:d]
        self.gamma_ = res.x[d]

        return self

    def _augment(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def decision_function(self, X):
        Xb = self._augment(X)
        return Xb @ self.w_ - self.gamma_

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)

    def score(self, X, y):
        y = np.asarray(y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def confusion_matrix_basic(y_true, y_pred):
    # labels: 1 (pozitif), -1 (negatif)
    cm = np.zeros((2, 2), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        i = 0 if yt == 1 else 1
        j = 0 if yp == 1 else 1
        cm[i, j] += 1
    print("Confusion Matrix:")
    print("Gerçek/Tahmin |   1   |  -1  |")
    print(f"      1       |  {cm[0,0]:3d}  |  {cm[0,1]:3d}  |")
    print(f"     -1       |  {cm[1,0]:3d}  |  {cm[1,1]:3d}  |")
    return cm


def plot_decision_boundary_2d(clf, X_pos, X_neg, feature_names=None, title="RLP Decision Boundary"):
    X_pos = np.asarray(X_pos)
    X_neg = np.asarray(X_neg)

    plt.figure(figsize=(8, 5))
    plt.scatter(X_pos[:, 0], X_pos[:, 1], c="blue", marker="o",
                label="Pozitif (+1)", alpha=0.7)
    plt.scatter(X_neg[:, 0], X_neg[:, 1], c="red", marker="x",
                label="Negatif (-1)", alpha=0.7)

    # Karar sınırı: [x, y, 1]·w = gamma
    w0, w1, wb = clf.w_
    x_min = min(X_pos[:, 0].min(), X_neg[:, 0].min()) - 0.5
    x_max = max(X_pos[:, 0].max(), X_neg[:, 0].max()) + 0.5
    xs = np.linspace(x_min, x_max, 200)

    if abs(w1) > 1e-8:
        ys = (clf.gamma_ - w0 * xs - wb) / w1
        plt.plot(xs, ys, "g-", label="Karar sınırı")

    if feature_names is not None:
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
    else:
        plt.xlabel("x1")
        plt.ylabel("x2")

    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()


def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    # ===== Deney 1: Setosa (+1) vs Versicolor (-1), ilk 2 özellik =====
    mask = y < 2
    X_sub = X[mask][:, :2]
    y_sub = y[mask]

    X_pos = X_sub[y_sub == 0]  # setosa
    X_neg = X_sub[y_sub == 1]  # versicolor

    # Train/test split
    X_pos_train, X_pos_test = train_test_split(X_pos, test_size=0.3, random_state=42)
    X_neg_train, X_neg_test = train_test_split(X_neg, test_size=0.3, random_state=42)

    clf1 = RobustLPClassifier().fit(X_pos_train, X_neg_train)

    # Test set
    X_test = np.vstack([X_pos_test, X_neg_test])
    y_test = np.r_[np.ones(len(X_pos_test)), -np.ones(len(X_neg_test))]
    y_pred = clf1.predict(X_test)
    acc_test = np.mean(y_pred == y_test)
    cm = confusion_matrix_basic(y_test, y_pred)

    print("Deney 1: Setosa vs Versicolor (2D)")
    print("-----------------------------------")
    print("w       :", clf1.w_)
    print("gamma   :", clf1.gamma_)
    print(f"Test doğruluk: {acc_test * 100:.2f}%")
    print()

    plot_decision_boundary_2d(
        clf1,
        X_pos_train,
        X_neg_train,
        feature_names=["Sepal length", "Sepal width"],
        title="RLP: Setosa vs Versicolor (2D)",
    )
    plt.savefig("rlp_iris_experiment1.png", dpi=300, bbox_inches="tight")
    print("Grafik kaydedildi: rlp_iris_experiment1.png\n")

    # ===== Deney 2: Versicolor (+1) vs Virginica (-1), ilk 2 özellik =====
    mask2 = y > 0
    X_sub2 = X[mask2][:, :2]
    y_sub2 = y[mask2]

    X_pos2 = X_sub2[y_sub2 == 1]  # versicolor
    X_neg2 = X_sub2[y_sub2 == 2]  # virginica

    clf2 = RobustLPClassifier().fit(X_pos2, X_neg2)

    X_all2 = np.vstack([X_pos2, X_neg2])
    y_all2 = np.r_[np.ones(len(X_pos2)), -np.ones(len(X_neg2))]
    acc2 = clf2.score(X_all2, y_all2)

    print("Deney 2: Versicolor vs Virginica (2D)")
    print("--------------------------------------")
    print("w       :", clf2.w_)
    print("gamma   :", clf2.gamma_)
    print(f"Doğruluk: {acc2 * 100:.2f}%\n")

    plot_decision_boundary_2d(
        clf2,
        X_pos2,
        X_neg2,
        feature_names=["Sepal length", "Sepal width"],
        title="RLP: Versicolor vs Virginica (2D)",
    )
    plt.savefig("rlp_iris_experiment2.png", dpi=300, bbox_inches="tight")
    print("Grafik kaydedildi: rlp_iris_experiment2.png\n")

    # ===== Deney 3: Versicolor vs Virginica, 4 özellik =====
    X_pos3 = X[y == 1]
    X_neg3 = X[y == 2]

    clf3 = RobustLPClassifier().fit(X_pos3, X_neg3)

    X_all3 = np.vstack([X_pos3, X_neg3])
    y_all3 = np.r_[np.ones(len(X_pos3)), -np.ones(len(X_neg3))]
    acc3 = clf3.score(X_all3, y_all3)

    print("Deney 3: Versicolor vs Virginica (4D, sadece doğruluk)")
    print("------------------------------------------------------")
    print(f"Doğruluk: {acc3 * 100:.2f}%")

    plt.show()


if __name__ == "__main__":
    main()
