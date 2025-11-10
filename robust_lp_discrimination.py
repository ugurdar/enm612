"""
Robust Linear Programming Discrimination
Implementation of Bennett & Mangasarian's RLP method (1992)

Bu kod, lineer olarak ayrılamayan iki veri kümesini ayırmak için
robust linear programming yaklaşımını kullanır.
"""

import numpy as np
from scipy.optimize import linprog
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


class RobustLPDiscrimination:
    """
    Robust Linear Programming için sınıflandırıcı
    
    Bennett & Mangasarian (1992) makalesindeki formülasyonu kullanır:
    
    min (1/m) * sum((-A_i*w + gamma + 1)+) + (1/k) * sum((B_i*w - gamma + 1)+)
    
    Burada (x)+ = max(0, x) pozitif kısım fonksiyonudur.
    
    NOT: Gamma optimize edildiğinde, w'x formunda bias yok (w son eleman olarak dahil).
    Gamma sabit verildiğinde, w'x + b formunda kullanılır.
    
    A: Sınıf +1 için veri matrisi [A 1]
    B: Sınıf -1 için veri matrisi [B 1]
    w: Ağırlık vektörü (gamma optimize edilirse bias dahil değil)
    gamma: Eşik değeri
    
    LP formülasyonu:
    min (1/m) * sum(e_A) + (1/k) * sum(e_B)
    s.t. -A*w + gamma + 1 <= e_A
         B*w - gamma + 1 <= e_B
         e_A, e_B >= 0
    """
    
    def __init__(self, gamma=None):
        """
        Parameters:
        -----------
        gamma : float or None
            Eşik değeri (marjin için). None ise otomatik optimize edilir.
        """
        self.gamma = gamma
        self.gamma_optimized = None
        self.w = None
        self.b = None
        
    def fit(self, X_pos, X_neg):
        """
        İki sınıf için robust LP ayırıcı hiper düzlemi öğren
        
        Parameters:
        -----------
        X_pos : array-like, shape (n_pos, n_features)
            Pozitif sınıf örnekleri
        X_neg : array-like, shape (n_neg, n_features)
            Negatif sınıf örnekleri
        """
        X_pos = np.array(X_pos)
        X_neg = np.array(X_neg)
        
        n_pos = X_pos.shape[0]
        n_neg = X_neg.shape[0]
        n_features = X_pos.shape[1]
        
        # Bias terimi için 1'ler ekle
        A = np.hstack([X_pos, np.ones((n_pos, 1))])
        B = np.hstack([X_neg, np.ones((n_neg, 1))])
        
        # Gamma'yı optimize edip etmeyeceğimizi belirle
        optimize_gamma = (self.gamma is None)
        
        if optimize_gamma:
            # LP formülasyonu (gamma da optimize edilir)
            # Değişkenler: [w (n_features+1), gamma (1), e_A (n_pos), e_B (n_neg)]
            n_vars = (n_features + 1) + 1 + n_pos + n_neg
            
            # Amaç fonksiyonu: min (1/m)*sum(e_A) + (1/k)*sum(e_B)
            c = np.zeros(n_vars)
            # w ve gamma katsayıları 0
            c[n_features+2:n_features+2+n_pos] = 1.0 / n_pos  # e_A katsayıları
            c[n_features+2+n_pos:] = 1.0 / n_neg  # e_B katsayıları
            
            # Eşitsizlik kısıtları: A_ub @ x <= b_ub
            A_ub = []
            b_ub = []
            
            # 1. Pozitif sınıf için: -A*w + gamma + 1 <= e_A
            for i in range(n_pos):
                row = np.zeros(n_vars)
                row[:n_features+1] = -A[i]  # -A_i*w
                row[n_features+1] = 1  # gamma
                row[n_features+2+i] = -1  # -e_A[i]
                A_ub.append(row)
                b_ub.append(-1)  # -1
            
            # 2. Negatif sınıf için: B*w - gamma + 1 <= e_B
            for i in range(n_neg):
                row = np.zeros(n_vars)
                row[:n_features+1] = B[i]  # B_i*w
                row[n_features+1] = -1  # -gamma
                row[n_features+2+n_pos+i] = -1  # -e_B[i]
                A_ub.append(row)
                b_ub.append(-1)  # -1
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            
            # Sınırlar: w sınırsız, gamma sınırsız, e >= 0
            bounds = []
            for i in range(n_features + 1):
                bounds.append((None, None))  # w değişkenleri sınırsız
            bounds.append((None, None))  # gamma sınırsız
            for i in range(n_pos + n_neg):
                bounds.append((0, None))  # e değişkenleri >= 0
            
            # LP'yi çöz
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, 
                            method='highs', options={'disp': False})
            
            if not result.success:
                print("Uyarı: LP çözümü optimal olmayabilir")
            
            # Sonuçları çıkar
            self.w = result.x[:n_features]
            self.b = result.x[n_features]
            self.gamma_optimized = result.x[n_features+1]
            
            # Hata istatistikleri
            e_A = result.x[n_features+2:n_features+2+n_pos]
            e_B = result.x[n_features+2+n_pos:]
            
            print(f"Optimizasyon durumu: {result.message}")
            print(f"Optimize edilen gamma: {self.gamma_optimized:.4f}")
            print(f"Amaç fonksiyonu değeri: {result.fun:.4f}")
            print(f"Ortalama pozitif sınıf ihlali: {np.sum(e_A)/n_pos:.4f}")
            print(f"Ortalama negatif sınıf ihlali: {np.sum(e_B)/n_neg:.4f}")
            print(f"Pozitif sınıf hatalı sayısı: {np.sum(e_A > 0.01)}/{n_pos}")
            print(f"Negatif sınıf hatalı sayısı: {np.sum(e_B > 0.01)}/{n_neg}")
            
        else:
            # LP formülasyonu (gamma sabit)
            # Değişkenler: [w (n_features+1), e_A (n_pos), e_B (n_neg)]
            n_vars = (n_features + 1) + n_pos + n_neg
            
            # Amaç fonksiyonu: min (1/m)*sum(e_A) + (1/k)*sum(e_B)
            c = np.zeros(n_vars)
            c[n_features+1:n_features+1+n_pos] = 1.0 / n_pos  # e_A katsayıları
            c[n_features+1+n_pos:] = 1.0 / n_neg  # e_B katsayıları
            
            # Eşitsizlik kısıtları: A_ub @ x <= b_ub
            A_ub = []
            b_ub = []
            
            # 1. Pozitif sınıf için: -A*w + gamma + 1 <= e_A
            for i in range(n_pos):
                row = np.zeros(n_vars)
                row[:n_features+1] = -A[i]  # -A_i*w
                row[n_features+1+i] = -1  # -e_A[i]
                A_ub.append(row)
                b_ub.append(-(self.gamma + 1))  # -(gamma + 1)
            
            # 2. Negatif sınıf için: B*w - gamma + 1 <= e_B
            for i in range(n_neg):
                row = np.zeros(n_vars)
                row[:n_features+1] = B[i]  # B_i*w
                row[n_features+1+n_pos+i] = -1  # -e_B[i]
                A_ub.append(row)
                b_ub.append(self.gamma - 1)  # gamma - 1
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            
            # Sınırlar: w sınırsız, e >= 0
            bounds = []
            for i in range(n_features + 1):
                bounds.append((None, None))  # w değişkenleri sınırsız
            for i in range(n_pos + n_neg):
                bounds.append((0, None))  # e değişkenleri >= 0
            
            # LP'yi çöz
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, 
                            method='highs', options={'disp': False})
            
            if not result.success:
                print("Uyarı: LP çözümü optimal olmayabilir")
            
            # Sonuçları çıkar
            self.w = result.x[:n_features]
            self.b = result.x[n_features]
            self.gamma_optimized = self.gamma
            
            # Hata istatistikleri
            e_A = result.x[n_features+1:n_features+1+n_pos]
            e_B = result.x[n_features+1+n_pos:]
            
            print(f"Optimizasyon durumu: {result.message}")
            print(f"Kullanılan gamma: {self.gamma:.4f}")
            print(f"Amaç fonksiyonu değeri: {result.fun:.4f}")
            print(f"Ortalama pozitif sınıf ihlali: {np.sum(e_A)/n_pos:.4f}")
            print(f"Ortalama negatif sınıf ihlali: {np.sum(e_B)/n_neg:.4f}")
            print(f"Pozitif sınıf hatalı sayısı: {np.sum(e_A > 0.01)}/{n_pos}")
            print(f"Negatif sınıf hatalı sayısı: {np.sum(e_B > 0.01)}/{n_neg}")
        
        return self
    
    def predict(self, X):
        """
        Yeni örnekleri sınıflandır
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test örnekleri
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Tahmin edilen sınıf etiketleri (+1 veya -1)
        """
        X = np.array(X)
        # Karar kuralı: w'x + b > gamma ise +1, değilse -1
        # Eşdeğer: w'x + b - gamma > 0
        decision_values = X @ self.w + self.b - self.gamma_optimized
        return np.where(decision_values > 0, 1, -1)
    
    def decision_function(self, X):
        """
        Karar fonksiyonu değerlerini hesapla
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test örnekleri
            
        Returns:
        --------
        scores : array, shape (n_samples,)
            Karar fonksiyonu değerleri
        """
        X = np.array(X)
        return X @ self.w + self.b
    
    def score(self, X, y):
        """
        Sınıflandırma doğruluğunu hesapla
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test örnekleri
        y : array-like, shape (n_samples,)
            Gerçek etiketler (+1 veya -1)
            
        Returns:
        --------
        accuracy : float
            Doğruluk oranı
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def plot_decision_boundary_2d(clf, X_pos, X_neg, feature_names=None, title="RLP Decision Boundary"):
    """
    2D karar sınırını görselleştir
    """
    plt.figure(figsize=(10, 6))
    
    # Veri noktalarını çiz
    plt.scatter(X_pos[:, 0], X_pos[:, 1], c='blue', marker='o', 
                label='Pozitif Sınıf', s=50, alpha=0.7)
    plt.scatter(X_neg[:, 0], X_neg[:, 1], c='red', marker='x', 
                label='Negatif Sınıf', s=50, alpha=0.7)
    
    # Karar sınırını çiz
    x_min = min(X_pos[:, 0].min(), X_neg[:, 0].min()) - 0.5
    x_max = max(X_pos[:, 0].max(), X_neg[:, 0].max()) + 0.5
    
    if abs(clf.w[1]) > 1e-10:  # w[1] != 0
        x_line = np.array([x_min, x_max])
        y_line = -(clf.w[0] * x_line + clf.b) / clf.w[1]
        plt.plot(x_line, y_line, 'g-', linewidth=2, label='Karar Sınırı')
        
        # Marjin çizgileri (gamma kadar uzaklıkta)
        gamma_to_use = clf.gamma_optimized if clf.gamma_optimized is not None else 0.0
        if gamma_to_use != 0:
            y_line_plus = -(clf.w[0] * x_line + clf.b - gamma_to_use) / clf.w[1]
            y_line_minus = -(clf.w[0] * x_line + clf.b + gamma_to_use) / clf.w[1]
            plt.plot(x_line, y_line_plus, 'g--', linewidth=1, alpha=0.5, label='Marjin')
            plt.plot(x_line, y_line_minus, 'g--', linewidth=1, alpha=0.5)
    
    if feature_names:
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
    else:
        plt.xlabel('Özellik 1')
        plt.ylabel('Özellik 2')
    
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt


def main():
    """
    Iris veri seti üzerinde RLP uygulaması
    """
    print("=" * 70)
    print("Robust Linear Programming Discrimination")
    print("Bennett & Mangasarian (1992)")
    print("=" * 70)
    print()
    
    # Iris veri setini yükle
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # İlk iki sınıfı seç (setosa vs versicolor)
    # Bu sınıflar daha iyi ayrılabilir
    print("Deney 1: Setosa vs Versicolor (Lineer Ayrılabilir)")
    print("-" * 70)
    mask = y < 2
    X_subset = X[mask]
    y_subset = y[mask]
    
    # Pozitif ve negatif sınıflara ayır
    X_pos = X_subset[y_subset == 0]  # Setosa
    X_neg = X_subset[y_subset == 1]  # Versicolor
    
    # Sadece ilk 2 özelliği kullan (görselleştirme için)
    X_pos_2d = X_pos[:, :2]
    X_neg_2d = X_neg[:, :2]
    
    # RLP modelini eğit
    clf1 = RobustLPDiscrimination(gamma=0.0)
    clf1.fit(X_pos_2d, X_neg_2d)
    
    print(f"\nÖğrenilen ağırlıklar: w = {clf1.w}")
    print(f"Öğrenilen bias: b = {clf1.b:.4f}")
    
    # Doğruluğu hesapla
    X_all = np.vstack([X_pos_2d, X_neg_2d])
    y_all = np.array([1]*len(X_pos_2d) + [-1]*len(X_neg_2d))
    accuracy1 = clf1.score(X_all, y_all)
    print(f"Eğitim doğruluğu: {accuracy1*100:.2f}%")
    
    # Görselleştir
    plot_decision_boundary_2d(clf1, X_pos_2d, X_neg_2d, 
                             feature_names=['Sepal Length', 'Sepal Width'],
                             title='RLP: Setosa vs Versicolor')
    plt.savefig('/Users/ugurdar/Documents/doktora/enm612/rlp_iris_experiment1.png', dpi=300, bbox_inches='tight')
    print("\nGrafik kaydedildi: rlp_iris_experiment1.png")
    
    print("\n" + "=" * 70)
    print("Deney 2: Versicolor vs Virginica (Lineer Ayrılamaz)")
    print("-" * 70)
    
    # Versicolor vs Virginica (daha zor ayrım)
    mask2 = y > 0
    X_subset2 = X[mask2]
    y_subset2 = y[mask2]
    
    X_pos2 = X_subset2[y_subset2 == 1]  # Versicolor
    X_neg2 = X_subset2[y_subset2 == 2]  # Virginica
    
    # İlk 2 özellik
    X_pos_2d_2 = X_pos2[:, :2]
    X_neg_2d_2 = X_neg2[:, :2]
    
    # RLP modelini eğit (gamma otomatik optimize edilir)
    clf2 = RobustLPDiscrimination(gamma=None)
    clf2.fit(X_pos_2d_2, X_neg_2d_2)
    
    print(f"\nÖğrenilen ağırlıklar: w = {clf2.w}")
    print(f"Öğrenilen bias: b = {clf2.b:.4f}")
    
    # Doğruluğu hesapla
    X_all2 = np.vstack([X_pos_2d_2, X_neg_2d_2])
    y_all2 = np.array([1]*len(X_pos_2d_2) + [-1]*len(X_neg_2d_2))
    accuracy2 = clf2.score(X_all2, y_all2)
    print(f"Eğitim doğruluğu: {accuracy2*100:.2f}%")
    
    # Görselleştir
    plot_decision_boundary_2d(clf2, X_pos_2d_2, X_neg_2d_2,
                             feature_names=['Sepal Length', 'Sepal Width'],
                             title='RLP: Versicolor vs Virginica')
    plt.savefig('/Users/ugurdar/Documents/doktora/enm612/rlp_iris_experiment2.png', dpi=300, bbox_inches='tight')
    print("\nGrafik kaydedildi: rlp_iris_experiment2.png")
    
    print("\n" + "=" * 70)
    print("Deney 3: Tüm 4 Özellik ile (Versicolor vs Virginica)")
    print("-" * 70)
    
    # 4 özellikle daha iyi performans bekliyoruz (gamma otomatik)
    clf3 = RobustLPDiscrimination(gamma=None)
    clf3.fit(X_pos2, X_neg2)
    
    print(f"\nÖğrenilen ağırlıklar: w = {clf3.w}")
    print(f"Öğrenilen bias: b = {clf3.b:.4f}")
    
    # Doğruluğu hesapla
    X_all3 = np.vstack([X_pos2, X_neg2])
    y_all3 = np.array([1]*len(X_pos2) + [-1]*len(X_neg2))
    accuracy3 = clf3.score(X_all3, y_all3)
    print(f"Eğitim doğruluğu: {accuracy3*100:.2f}%")
    
    print("\n" + "=" * 70)
    print("Deney 4: Gamma Manuel vs Otomatik Karşılaştırması")
    print("-" * 70)
    
    # Farklı gamma değerleri test et
    gamma_values = [None, 0.0, 0.5, 1.0, 2.0]
    print("\nGamma parametresinin etkisi (Versicolor vs Virginica, 2D):")
    print(f"{'Gamma':<15} {'Kullanılan':<15} {'Doğruluk':<15} {'Amaç Değeri':<15}")
    print("-" * 60)
    
    for gamma in gamma_values:
        clf_temp = RobustLPDiscrimination(gamma=gamma)
        clf_temp.fit(X_pos_2d_2, X_neg_2d_2)
        acc = clf_temp.score(X_all2, y_all2)
        
        gamma_str = "Otomatik" if gamma is None else f"{gamma:.1f}"
        gamma_used = clf_temp.gamma_optimized
        
        # Amaç fonksiyonu değerini hesapla
        e_A = np.maximum(0, -X_pos_2d_2 @ clf_temp.w - clf_temp.b + gamma_used + 1)
        e_B = np.maximum(0, X_neg_2d_2 @ clf_temp.w + clf_temp.b - gamma_used + 1)
        obj_value = np.mean(e_A) + np.mean(e_B)
        
        print(f"{gamma_str:<15} {gamma_used:<14.4f} {acc*100:<14.2f}% {obj_value:<15.4f}")
    
    print("\n" + "=" * 70)
    print("Özet:")
    print(f"  - Deney 1 (Setosa vs Versicolor): {accuracy1*100:.2f}%")
    print(f"  - Deney 2 (Versicolor vs Virginica, 2D): {accuracy2*100:.2f}%")
    print(f"  - Deney 3 (Versicolor vs Virginica, 4D): {accuracy3*100:.2f}%")
    print("=" * 70)
    
    plt.show()


if __name__ == "__main__":
    main()
