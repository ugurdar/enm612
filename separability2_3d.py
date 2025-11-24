"""
Max-Min Separability ile Belirlenmiş Karar Fonksiyonu Görselleştirmesi
z = max(min(2x+3y+5, -x+y+5, x-y+4, -3x-y+5), 
        min(2(x-5)+3, (y-5)+5, -(-x-5)+(y-5)+5, (x-5)-(y-5)+4, -3(x-5)-(y-5)+5))

İki grup var:
  Grup 1 (merkez): 4 hiper-düzlem → min(2x+3y+5, -x+y+5, x-y+4, -3x-y+5)
  Grup 2 (kaydırılmış): 5 hiper-düzlem → min(2(x-5)+3, (y-5)+5, -(x-5)+(y-5)+5, (x-5)-(y-5)+4, -3(x-5)-(y-5)+5)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Karar fonksiyonu
def decision_function(x, y):
    """
    Verilen max-min karar fonksiyonu
    z > 0 → Sınıf A (Pozitif - Mavi)
    z ≤ 0 → Sınıf B (Negatif - Kırmızı)
    """
    # Grup 1 (merkez)
    g1_p1 = 2*x + 3*y + 5
    g1_p2 = -x + y + 5
    g1_p3 = x - y + 4
    g1_p4 = -3*x - y + 5
    group1 = min(g1_p1, g1_p2, g1_p3, g1_p4)
    
    # Grup 2 (x-5, y-5 kaydırılmış)
    g2_p1 = 2*(x-5) + 3
    g2_p2 = (y-5) + 5
    g2_p3 = -(x-5) + (y-5) + 5  # -(-x-5)+(y-5)+5 = (x+5)+(y-5)+5 = x+y+5 → Düzeltme: -(x-5)+(y-5)+5
    g2_p4 = (x-5) - (y-5) + 4
    g2_p5 = -3*(x-5) - (y-5) + 5
    group2 = min(g2_p1, g2_p2, g2_p3, g2_p4, g2_p5)
    
    return max(group1, group2)

# Veri noktaları oluşturma - karar fonksiyonuna göre etiketleme
np.random.seed(42)

# Daha dengeli örnekleme: Pozitif bölgeye yoğunlaştırılmış örnekler
n_positive_samples = 400  # Pozitif bölgeye odaklanmış örnekler
n_negative_samples = 400  # Negatif bölgeye odaklanmış örnekler

# Pozitif bölge örnekleri (gerçek karar fonksiyonunun > 0 olduğu bölgelerden)
X_pos_candidates = []
# Grup 1 pozitif bölgesi: merkez yakını
X_pos_candidates.append(np.random.randn(150, 2) * 1.0 + np.array([0, -1]))
# Grup 2 pozitif bölgesi: (5,5) civarı
X_pos_candidates.append(np.random.randn(150, 2) * 1.0 + np.array([5, 5]))
# Aradaki bölge
X_pos_candidates.append(np.random.uniform(low=[0, 0], high=[5, 5], size=(100, 2)))

X_pos_all = np.vstack(X_pos_candidates)

# Sadece gerçekten pozitif olanları al
A = []
for point in X_pos_all:
    z = decision_function(point[0], point[1])
    if z > 0:
        A.append(point)
        if len(A) >= n_positive_samples:
            break

A = np.array(A[:n_positive_samples])

# Negatif bölge örnekleri
X_neg_candidates = []
# Çevreden örnekler
X_neg_candidates.append(np.random.uniform(low=[-6, -6], high=[11, 11], size=(600, 2)))

X_neg_all = np.vstack(X_neg_candidates)

# Sadece gerçekten negatif olanları al
B = []
for point in X_neg_all:
    z = decision_function(point[0], point[1])
    if z <= 0:
        B.append(point)
        if len(B) >= n_negative_samples:
            break

B = np.array(B[:n_negative_samples])

print(f"\nVeri Noktaları:")
print(f"  Sınıf A (Pozitif): {len(A)} nokta")
print(f"  Sınıf B (Negatif): {len(B)} nokta")
print(f"  Denge oranı: {len(A)/(len(A)+len(B))*100:.1f}% pozitif")
print(f"\nGerçek Karar Fonksiyonu:")
print(f"  Grup 1: min(2x+3y+5, -x+y+5, x-y+4, -3x-y+5)")
print(f"  Grup 2: min(2(x-5)+3, (y-5)+5, -(x-5)+(y-5)+5, (x-5)-(y-5)+4, -3(x-5)-(y-5)+5)")
print(f"  Karar: z = max(Grup 1, Grup 2)")
print(f"  Sınıflandırma: z > 0 → A (mavi), z ≤ 0 → B (kırmızı)")

# H-Polyhedral Loss Fonksiyonu
def hpoly_loss(params, A, B, h):
    """
    H-Polyhedral loss fonksiyonu
    Karar fonksiyonu: psi(x) = max_j { w_j^T x - gamma_j }
    
    params: [w1_x, w1_y, gamma1, w2_x, w2_y, gamma2, ...]
    """
    m, k = len(A), len(B)
    w_list = []
    gamma_list = []
    
    for j in range(h):
        idx = j * 3
        w_list.append(np.array([params[idx], params[idx+1]]))
        gamma_list.append(params[idx+2])
    
    loss = 0.0
    
    # A için: TÜM düzlemler içinde olmalı (her j için)
    for a in A:
        for j in range(h):
            val = np.dot(w_list[j], a) - gamma_list[j]
            if val > -1:  # İhlal varsa cezalandır
                loss += max(0, val + 1)**2
    
    # B için: En az bir düzlem dışında olmalı (max üzerinden)
    for b in B:
        vals = [np.dot(w_list[j], b) - gamma_list[j] for j in range(h)]
        psi_b = max(vals)
        if psi_b < 1:  # İhlal varsa cezalandır
            loss += (1 - psi_b)**2
    
    return loss

# Max-Min Loss Fonksiyonu
def maxmin_loss(params, A, B, group_sizes):
    """
    Max-Min loss fonksiyonu (Bagirov 2005 makalesine göre)
    Karar fonksiyonu: psi(x) = max_i { min_j∈G_i { w_j^T x - gamma_j } }
    
    A için: max_i { min_j∈G_i { w_j^T a - gamma_j } } ≤ -1
    B için: max_i { min_j∈G_i { w_j^T b - gamma_j } } ≥ 1
    
    Dengelenmiş loss: A ve B'nin eşit ağırlıkta katkısı
    
    params: [w1_x, w1_y, gamma1, w2_x, w2_y, gamma2, ...]
    """
    m, k = len(A), len(B)
    r = len(group_sizes)
    total_planes = sum(group_sizes)
    
    w_list = []
    gamma_list = []
    
    for j in range(total_planes):
        idx = j * 3
        w_list.append(np.array([params[idx], params[idx+1]]))
        gamma_list.append(params[idx+2])
    
    loss_A = 0.0
    loss_B = 0.0
    
    # A için: psi(a) = max_i { min_j∈G_i { w_j^T a - gamma_j } } ≤ -1
    # Loss: max(0, psi(a) + 1)^2
    for a in A:
        group_mins = []
        plane_idx = 0
        for i in range(r):
            group_vals = []
            for _ in range(group_sizes[i]):
                val = np.dot(w_list[plane_idx], a) - gamma_list[plane_idx]
                group_vals.append(val)
                plane_idx += 1
            group_mins.append(min(group_vals))
        
        psi_a = max(group_mins)
        if psi_a > -1:
            loss_A += max(0, psi_a + 1)**2
    
    # B için: psi(b) = max_i { min_j∈G_i { w_j^T b - gamma_j } } ≥ 1
    # Loss: max(0, 1 - psi(b))^2
    for b in B:
        group_mins = []
        plane_idx = 0
        for i in range(r):
            group_vals = []
            for _ in range(group_sizes[i]):
                val = np.dot(w_list[plane_idx], b) - gamma_list[plane_idx]
                group_vals.append(val)
                plane_idx += 1
            group_mins.append(min(group_vals))
        
        psi_b = max(group_mins)
        if psi_b < 1:
            loss_B += (1 - psi_b)**2
    
    # Dengelenmiş loss - sınıf büyüklüklerini normalize et
    return (loss_A / m) + (loss_B / k)

# H-Polyhedral optimizasyonu
print("\n" + "="*70)
print("H-Polyhedral Separability")
print("="*70)
print("Optimizasyon başlıyor...")
h = 8  # Daha fazla düzlem kullan (karmaşık yapıyı yakalaması için)
initial_params = np.random.randn(h * 3) * 0.5

result = minimize(hpoly_loss, initial_params, args=(A, B, h), 
                 method='BFGS', options={'maxiter': 2000, 'disp': False})

hpoly_params = result.x
hpoly_loss_val = result.fun

w_hpoly = []
gamma_hpoly = []
for j in range(h):
    idx = j * 3
    w_hpoly.append(np.array([hpoly_params[idx], hpoly_params[idx+1]]))
    gamma_hpoly.append(hpoly_params[idx+2])

print(f"Loss: {hpoly_loss_val:.4f}")
print(f"Düzlem sayısı: {h}")

# Max-Min optimizasyonu
print("\n" + "="*70)
print("Max-Min Separability")
print("="*70)
print("Optimizasyon başlıyor...")
r = 2
group_sizes = [4, 5]  # Gerçek yapıya uygun: Grup 1 (4 düzlem), Grup 2 (5 düzlem)
total_planes = sum(group_sizes)

# Gerçek düzlemlere yakın başlangıç değerleri kullan
# Grup 1: 2x+3y+5=0, -x+y+5=0, x-y+4=0, -3x-y+5=0
# Grup 2: 2(x-5)+3=0, (y-5)+5=0, -(x-5)+(y-5)+5=0, (x-5)-(y-5)+4=0, -3(x-5)-(y-5)+5=0
initial_params = np.array([
    # Grup 1
    2, 3, -5,     # 2x + 3y - 5 = 0 (gamma = -5 çünkü w^T x - gamma = 0)
    -1, 1, -5,    # -x + y - 5 = 0
    1, -1, -4,    # x - y - 4 = 0
    -3, -1, -5,   # -3x - y - 5 = 0
    # Grup 2
    2, 0, 7,      # 2x - 7 = 0 (2(x-5)+3=0 → 2x=7)
    0, 1, 0,      # y = 0
    -1, 1, -5,    # -x + y - 5 = 0 (aynı Grup 1'deki gibi)
    1, -1, -4,    # x - y - 4 = 0
    -3, -1, -25,  # -3x - y + 25 = 0
]) * 0.3 + np.random.randn(total_planes * 3) * 0.1  # Küçük gürültü ekle

# Çoklu başlangıç noktası ile optimizasyon
best_result = None
best_loss = np.inf

print("Çoklu başlangıç noktası ile optimizasyon yapılıyor...")
for trial in range(5):
    if trial == 0:
        x0 = initial_params  # Gerçek değerlere yakın
    else:
        x0 = np.random.randn(total_planes * 3) * 0.5  # Rastgele
    
    result = minimize(maxmin_loss, x0, args=(A, B, group_sizes),
                     method='BFGS', options={'maxiter': 5000, 'disp': False})
    
    if result.fun < best_loss:
        best_loss = result.fun
        best_result = result
        print(f"  Deneme {trial+1}: Loss = {result.fun:.4f} ✓ (en iyi)")
    else:
        print(f"  Deneme {trial+1}: Loss = {result.fun:.4f}")

result = best_result

maxmin_params = result.x
maxmin_loss_val = result.fun

w_maxmin = []
gamma_maxmin = []
for j in range(total_planes):
    idx = j * 3
    w_maxmin.append(np.array([maxmin_params[idx], maxmin_params[idx+1]]))
    gamma_maxmin.append(maxmin_params[idx+2])

print(f"Loss: {maxmin_loss_val:.4f}")
print(f"Grup sayısı: {r}")
print(f"Her gruptaki düzlem sayıları: {group_sizes}")

# Görselleştirme
print("\n" + "="*70)
print("Görselleştirme")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(18, 16))

# Grid oluştur (karar bölgelerini göstermek için)
x_min, x_max = -6, 11
y_min, y_max = -6, 11
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

# 1. Gerçek Karar Fonksiyonu
ax = axes[0, 0]
Z_true = np.zeros_like(xx)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        Z_true[i, j] = decision_function(xx[i, j], yy[i, j])

ax.contourf(xx, yy, Z_true, levels=[-100, 0, 100], colors=['lightcoral', 'lightblue'], alpha=0.4)
ax.contour(xx, yy, Z_true, levels=[0], colors='black', linewidths=2, linestyles='--')

# Veri noktaları
ax.scatter(A[:, 0], A[:, 1], c='blue', marker='o', s=60, edgecolors='darkblue', 
          linewidths=1.5, alpha=0.8, label='Sınıf A (Pozitif)')
ax.scatter(B[:, 0], B[:, 1], c='red', marker='x', s=60, linewidths=2, 
          alpha=0.8, label='Sınıf B (Negatif)')

# Gerçek hiper-düzlemleri çiz
x_range = np.linspace(x_min, x_max, 100)

# Grup 1 düzlemleri (turuncu)
plane_equations_g1 = [
    (2, 3, 5),    # 2x + 3y + 5 = 0
    (-1, 1, 5),   # -x + y + 5 = 0
    (1, -1, 4),   # x - y + 4 = 0
    (-3, -1, 5),  # -3x - y + 5 = 0
]

for w1, w2, gamma in plane_equations_g1:
    if abs(w2) > 1e-6:
        y_line = (-w1 * x_range - gamma) / w2
        ax.plot(x_range, y_line, 'orange', linewidth=1.5, alpha=0.7)

# Grup 2 düzlemleri (mor)
plane_equations_g2 = [
    (2, 0, -10+3),      # 2(x-5) + 3 = 0 → 2x - 7 = 0
    (0, 1, -5+5),       # (y-5) + 5 = 0 → y = 0
    (-1, 1, 5),         # -(x-5)+(y-5)+5 = 0 → -x+y+5 = 0
    (1, -1, 0),         # (x-5)-(y-5)+4 = 0 → x-y+4 = 0
    (-3, -1, 15+5),     # -3(x-5)-(y-5)+5 = 0 → -3x-y+25 = 0
]

for w1, w2, gamma in plane_equations_g2:
    if abs(w2) > 1e-6:
        y_line = (-w1 * x_range - gamma) / w2
        ax.plot(x_range, y_line, 'purple', linewidth=1.5, alpha=0.7)
    elif abs(w1) > 1e-6:
        x_line = -gamma / w1
        ax.axvline(x=x_line, color='purple', linewidth=1.5, alpha=0.7)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel('$x_1$', fontsize=13)
ax.set_ylabel('$x_2$', fontsize=13)
ax.set_title('Gerçek Karar Fonksiyonu\n(Siyah kesikli: Karar sınırı)', 
            fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)

# 2. H-Polyhedral Separability
ax = axes[0, 1]

# H-Polyhedral karar bölgesi
Z_hpoly = np.zeros_like(xx)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        point = np.array([xx[i, j], yy[i, j]])
        vals = [np.dot(w, point) - gamma for w, gamma in zip(w_hpoly, gamma_hpoly)]
        Z_hpoly[i, j] = max(vals)

ax.contourf(xx, yy, Z_hpoly, levels=[-100, 0, 100], colors=['lightcoral', 'lightblue'], alpha=0.4)
ax.contour(xx, yy, Z_hpoly, levels=[0], colors='black', linewidths=2)

# Veri noktaları
ax.scatter(A[:, 0], A[:, 1], c='blue', marker='o', s=60, edgecolors='darkblue', 
          linewidths=1.5, alpha=0.8, label='Sınıf A')
ax.scatter(B[:, 0], B[:, 1], c='red', marker='x', s=60, linewidths=2, 
          alpha=0.8, label='Sınıf B')

# Hiper-düzlemleri çiz (farklı renklerle)
colors_hpoly = plt.cm.viridis(np.linspace(0, 1, h))
for j, (w, gamma) in enumerate(zip(w_hpoly, gamma_hpoly)):
    if abs(w[1]) > 1e-6:
        y_line = (-w[0] * x_range - (-gamma)) / w[1]
        ax.plot(x_range, y_line, color=colors_hpoly[j], linewidth=1.2, alpha=0.6,
               label=f'Düzlem {j+1}' if j < 3 else '')

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel('$x_1$', fontsize=13)
ax.set_ylabel('$x_2$', fontsize=13)
ax.set_title(f'H-Polyhedral Separability ({h} Düzlem)\nLoss: {hpoly_loss_val:.4f}', 
            fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

# 3. Max-Min Separability
ax = axes[1, 0]

# Max-Min karar bölgesi
Z_maxmin = np.zeros_like(xx)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        point = np.array([xx[i, j], yy[i, j]])
        
        # Grup bazında min hesapla
        group_mins = []
        plane_idx = 0
        for g in range(r):
            group_vals = []
            for _ in range(group_sizes[g]):
                val = np.dot(w_maxmin[plane_idx], point) - gamma_maxmin[plane_idx]
                group_vals.append(val)
                plane_idx += 1
            group_mins.append(min(group_vals))
        
        Z_maxmin[i, j] = max(group_mins)

ax.contourf(xx, yy, Z_maxmin, levels=[-100, 0, 100], colors=['lightcoral', 'lightblue'], alpha=0.4)
ax.contour(xx, yy, Z_maxmin, levels=[0], colors='black', linewidths=2)

# Veri noktaları
ax.scatter(A[:, 0], A[:, 1], c='blue', marker='o', s=60, edgecolors='darkblue', 
          linewidths=1.5, alpha=0.8, label='Sınıf A')
ax.scatter(B[:, 0], B[:, 1], c='red', marker='x', s=60, linewidths=2, 
          alpha=0.8, label='Sınıf B')

# Hiper-düzlemleri gruplara göre çiz
colors_maxmin = ['orange', 'purple']
plane_idx = 0
for i in range(r):
    for j in range(group_sizes[i]):
        w = w_maxmin[plane_idx]
        gamma = gamma_maxmin[plane_idx]
        if abs(w[1]) > 1e-6:
            y_line = (-w[0] * x_range - (-gamma)) / w[1]
            ax.plot(x_range, y_line, color=colors_maxmin[i], linewidth=1.5, alpha=0.7,
                   label=f'Grup {i+1}' if j == 0 else '')
        plane_idx += 1

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel('$x_1$', fontsize=13)
ax.set_ylabel('$x_2$', fontsize=13)
ax.set_title(f'Max-Min Separability ({r} Grup: {group_sizes})\nLoss: {maxmin_loss_val:.4f}', 
            fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)

# 4. Karşılaştırma - Karar Sınırları
ax = axes[1, 1]

# Veri noktaları
ax.scatter(A[:, 0], A[:, 1], c='blue', marker='o', s=60, edgecolors='darkblue', 
          linewidths=1.5, alpha=0.8, label='Sınıf A', zorder=3)
ax.scatter(B[:, 0], B[:, 1], c='red', marker='x', s=60, linewidths=2, 
          alpha=0.8, label='Sınıf B', zorder=3)

# Karar sınırlarını çiz
ax.contour(xx, yy, Z_true, levels=[0], colors='black', linewidths=3, 
          linestyles='--', label='Gerçek', zorder=2)
ax.contour(xx, yy, Z_hpoly, levels=[0], colors='green', linewidths=2.5, 
          linestyles='-.', label='H-Polyhedral', zorder=1)
ax.contour(xx, yy, Z_maxmin, levels=[0], colors='darkorange', linewidths=2.5, 
          linestyles=':', label='Max-Min', zorder=1)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel('$x_1$', fontsize=13)
ax.set_ylabel('$x_2$', fontsize=13)
ax.set_title('Karar Sınırları Karşılaştırması', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('separability2_comparison.png', dpi=300, bbox_inches='tight')
print("\nGörselleştirme 'separability2_comparison.png' olarak kaydedildi")
print("\n4 Grafik:")
print("  1. Sol üst: Gerçek karar fonksiyonu ve düzlemler")
print("  2. Sağ üst: H-Polyhedral yaklaşımı")
print("  3. Sol alt: Max-Min yaklaşımı")
print("  4. Sağ alt: Tüm karar sınırlarının karşılaştırması")
