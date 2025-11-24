import numpy as np
from scipy.optimize import linprog
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class RobustLPClassifier:
    """
    Bennett & Mangasarian (1992) robust linear programming classifier
    """
    def __init__(self):
        self.w_ = None
        self.gamma_ = None

    def fit(self, X_pos, X_neg):
        X_pos = np.asarray(X_pos)
        X_neg = np.asarray(X_neg)
        n_pos, n_features = X_pos.shape
        n_neg = X_neg.shape[0]
        A = np.hstack([X_pos, np.ones((n_pos, 1))])
        B = np.hstack([X_neg, np.ones((n_neg, 1))])
        d = n_features + 1
        n_vars = d + 1 + n_pos + n_neg
        c = np.zeros(n_vars)
        idx_eA_start = d + 1
        idx_eB_start = d + 1 + n_pos
        c[idx_eA_start:idx_eA_start + n_pos] = 1.0 / n_pos
        c[idx_eB_start:idx_eB_start + n_neg] = 1.0 / n_neg
        A_pos = np.hstack([
            -A,
            np.ones((n_pos, 1)),
            -np.eye(n_pos),
            np.zeros((n_pos, n_neg)),
        ])
        b_pos = -np.ones(n_pos)
        A_neg = np.hstack([
            B,
            -np.ones((n_neg, 1)),
            np.zeros((n_neg, n_pos)),
            -np.eye(n_neg),
        ])
        b_neg = -np.ones(n_neg)
        A_ub = np.vstack([A_pos, A_neg])
        b_ub = np.concatenate([b_pos, b_neg])
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
        return np.where(scores >= 0, 1, 0)

def confusion_matrix_basic(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        i = 0 if yt == 1 else 1
        j = 0 if yp == 1 else 1
        cm[i, j] += 1
    return cm

def classification_report_basic(y_true, y_pred):
    cm = confusion_matrix_basic(y_true, y_pred)
    TP = cm[0, 0]
    FN = cm[0, 1]
    FP = cm[1, 0]
    TN = cm[1, 1]
    accuracy = (TP + TN) / np.sum(cm)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print("Confusion Matrix:")
    print("Gerçek/Tahmin |   1   |   0   |")
    print(f"      1       |  {TP:3d}  |  {FN:3d}  |")
    print(f"      0       |  {FP:3d}  |  {TN:3d}  |")
    return accuracy, precision, recall, specificity

def roc_auc_basic(y_true, scores):
    # Sadece temel bir ROC/AUC (trapez kuralı)
    thresholds = np.sort(np.unique(scores))[::-1]
    TPR = []
    FPR = []
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    for thresh in thresholds:
        y_pred = (scores >= thresh).astype(int)
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        TPR.append(TP / P if P > 0 else 0)
        FPR.append(FP / N if N > 0 else 0)
    TPR = np.array(TPR)
    FPR = np.array(FPR)
    auc = np.trapz(TPR, FPR)
    print(f"AUC (trapez): {auc:.4f}")
    return auc

def print_basic_breast_cancer_info(data):
    print("Breast Cancer Dataset Basic Info:")
    print(f"  Num samples: {data.data.shape[0]}")
    print(f"  Num features: {data.data.shape[1]}")
    print(f"  Feature names: {', '.join(data.feature_names)}")
    unique, counts = np.unique(data.target, return_counts=True)
    target_names = [data.target_names[u] for u in unique]
    print("  Class distribution:")
    for name, count in zip(target_names, counts):
        print(f"    {name}: {count}")
    print()

def balanced_test_split(X, y, test_size=0.3, random_state=42):
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    # Her iki sınıftan eşit sayıda örnek seç
    n_min = min(np.sum(y_test == 0), np.sum(y_test == 1))
    idx_0 = np.where(y_test == 0)[0][:n_min]
    idx_1 = np.where(y_test == 1)[0][:n_min]
    idx_balanced = np.concatenate([idx_0, idx_1])
    X_test_bal = X_test[idx_balanced]
    y_test_bal = y_test[idx_balanced]
    return X_train, X_test_bal, y_train, y_test_bal

def main():
    data = load_breast_cancer()
    print_basic_breast_cancer_info(data)
    X = data.data
    y = data.target
    # Dengeli test seti
    X_train, X_test, y_train, y_test = balanced_test_split(X, y, test_size=0.3, random_state=42)
    X_pos = X_train[y_train == 1]
    X_neg = X_train[y_train == 0]
    clf = RobustLPClassifier().fit(X_pos, X_neg)
    y_pred = clf.predict(X_test)
    scores = clf.decision_function(X_test)
    print("RLP - Breast Cancer Test Sonuçları:")
    classification_report_basic(y_test, y_pred)
    roc_auc_basic(y_test, scores)
    print("\nRLP Decision Boundary (Matematiksel Form):")
    print("w (feature katsayıları):")
    for name, coef in zip(data.feature_names, clf.w_[:-1]):
        print(f"  {name:25}: {coef:.4f}")
    print(f"Bias (w_[-1]): {clf.w_[-1]:.4f}")
    print(f"Gamma        : {clf.gamma_:.4f}")
    print("Karar fonksiyonu: f(x) = w^T x + bias - gamma")
    print("Sınıf: f(x) >= 0 -> malignant, f(x) < 0 -> benign")
    # Sonuçları dosyaya yaz
    with open("output_breast_cancer.txt", "w") as f:
        f.write("Breast Cancer Dataset Basic Info:\n")
        f.write(f"  Num samples: {data.data.shape[0]}\n")
        f.write(f"  Num features: {data.data.shape[1]}\n")
        f.write(f"  Feature names: {', '.join(data.feature_names)}\n")
        unique, counts = np.unique(data.target, return_counts=True)
        target_names = [data.target_names[u] for u in unique]
        f.write("  Class distribution:\n")
        for name, count in zip(target_names, counts):
            f.write(f"    {name}: {count}\n")
        f.write("\nTest set class counts (balanced):\n")
        f.write(f"  Malignant (1): {np.sum(y_test == 1)}\n")
        f.write(f"  Benign    (0): {np.sum(y_test == 0)}\n")
        f.write("\nRLP Decision Boundary (Matematiksel Form)\n")
        f.write("Feature           | Katsayı\n")
        f.write("------------------|---------\n")
        for name, coef in zip(data.feature_names, clf.w_[:-1]):
            f.write(f"{name:18} | {coef: .4f}\n")
        f.write(f"{'Bias':18} | {clf.w_[-1]: .4f}\n")
        f.write(f"{'Gamma':18} | {clf.gamma_: .4f}\n")
        f.write("\nKarar fonksiyonu: f(x) = w^T x + bias - gamma\n")
        f.write("Sınıf: f(x) >= 0 -> malignant, f(x) < 0 -> benign\n\n")
        # Confusion matrix ve metrikler
        cm = confusion_matrix_basic(y_test, y_pred)
        TP = cm[0, 0]
        FN = cm[0, 1]
        FP = cm[1, 0]
        TN = cm[1, 1]
        accuracy = (TP + TN) / np.sum(cm)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        f.write("Confusion Matrix (Test Set):\n")
        f.write("Gerçek/Tahmin |   1   |   0   |\n")
        f.write(f"      1       |  {TP:3d}  |  {FN:3d}  |\n")
        f.write(f"      0       |  {FP:3d}  |  {TN:3d}  |\n")
        f.write(f"\nAccuracy   : {accuracy:.4f}\n")
        f.write(f"Precision  : {precision:.4f}\n")
        f.write(f"Recall     : {recall:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
    print("Sonuçlar output_breast_cancer.txt dosyasına yazıldı.")

if __name__ == "__main__":
    main()
