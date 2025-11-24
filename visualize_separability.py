"""
H-Polyhedral ve Max-Min Separability Görselleştirme

Bu script, iki boyutlu bir veri kümesi üzerinde H-Polyhedral ve Max-Min
separability yöntemlerinin ayrıştırma biçimlerini görselleştirir.
Hem 2D hem de 3D görselleştirme sunar.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

# Veri kümesi oluşturma (lineer ayrılamayan)
np.random.seed(42)

# Sınıf A: İki ayrı mavi küme (merkez + sağ üst)
A1 = np.random.randn(25, 2) * 0.5  # Merkezdeki mavi grup
A2 = np.random.randn(15, 2) * 0.4 + np.array([4, 4])  # Sağ üst köşedeki mavi grup
A = np.vstack([A1, A2])  # Sınıf A (pozitif) - 2 mavi grup

# Sınıf B: İki ayrı kırmızı grup (merkez çevresi + (2,2) noktası yakını)
B1 = np.random.randn(12, 2) * 0.4 + np.array([3, 0])    # Merkez çevresi - Sağ
B2 = np.random.randn(12, 2) * 0.4 + np.array([-3, 0])   # Merkez çevresi - Sol  
B3 = np.random.randn(12, 2) * 0.4 + np.array([0, 3])    # Merkez çevresi - Üst
B4 = np.random.randn(12, 2) * 0.4 + np.array([0, -3])   # Merkez çevresi - Alt
B5 = np.random.randn(14, 2) * 0.35 + np.array([2.2, 2.2])  # (2.2, 2.2) noktası çevresi
B = np.vstack([B1, B2, B3, B4, B5])  # Sınıf B (negatif) - 2 ayrı kırmızı grup

# H-Polyhedral Loss Fonksiyonu
def hpoly_loss(params, A, B, h):
    """
    H-Polyhedral loss fonksiyonu
    Karar fonksiyonu: psi(x) = max_j { w_j^T x - gamma_j }
    
    Geometrik anlam: h adet hiper-düzlem bir konveks çokyüzlü tanımlar.
    - A için: TÜM düzlemler A'yı dışta tutmalı → her j için w_j^T a <= gamma_j - 1
    - B için: psi(b) >= 1 → En az bir düzlem B'yi "içeride" bırakır
    
    params: [w1_x, w1_y, gamma1, w2_x, w2_y, gamma2, ...]
    """
    m, k = len(A), len(B)
    w_list = []
    gamma_list = []
    
    for j in range(h):
        idx = j * 3
        w_list.append(np.array([params[idx], params[idx+1]]))
        gamma_list.append(params[idx+2])
    
    # Sınıf A için: TÜM düzlemler ihlal edilmemeli
    # Her a için: max_j [w_j^T a - gamma_j + 1]_+
    loss_A = 0
    for a in A:
        # Her düzlem için ihlal kontrolü
        violations = [max(0, w @ a - gamma + 1) for w, gamma in zip(w_list, gamma_list)]
        # En kötü ihlal
        loss_A += max(violations)
    
    # Sınıf B için: psi(b) >= 1 olmalı
    # violation = max(0, 1 - psi(b))
    loss_B = 0
    for b in B:
        psi_b = max([w @ b - gamma for w, gamma in zip(w_list, gamma_list)])
        loss_B += max(0, 1 - psi_b)
    
    return loss_A / m + loss_B / k

# Max-Min Loss Fonksiyonu
def maxmin_loss(params, A, B, r, group_sizes):
    """
    Max-Min loss fonksiyonu (Bagirov 2005 makalesine göre)
    Karar fonksiyonu: psi(x) = max_i { min_j∈G_i { w_j^T x - gamma_j } }
    
    A için: psi(a) ≤ -1
    B için: psi(b) ≥ 1
    
    params: düzleştirilmiş [w, gamma] parametreleri
    r: grup sayısı
    group_sizes: her gruptaki hiper-düzlem sayısı listesi
    """
    m, k = len(A), len(B)
    total_planes = sum(group_sizes)
    
    # Parametreleri çöz
    w_list = []
    gamma_list = []
    for j in range(total_planes):
        idx = j * 3
        w_list.append(np.array([params[idx], params[idx+1]]))
        gamma_list.append(params[idx+2])
    
    loss = 0.0
    
    # A için: psi(a) = max_i { min_j∈G_i { w_j^T a - gamma_j } } ≤ -1
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
            loss += max(0, psi_a + 1)**2
    
    # B için: psi(b) = max_i { min_j∈G_i { w_j^T b - gamma_j } } ≥ 1
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
            loss += (1 - psi_b)**2
    
    return loss / (m + k)

# H-Polyhedral optimizasyonu (6 hiper-düzlem - daha iyi çokyüzlü)
print("H-Polyhedral optimizasyonu başlıyor...")
h = 6
x0_hpoly = np.random.randn(h * 3) * 0.1
result_hpoly = minimize(hpoly_loss, x0_hpoly, args=(A, B, h), method='BFGS', 
                        options={'maxiter': 1000})

w_list_hpoly = []
gamma_list_hpoly = []
for j in range(h):
    idx = j * 3
    w_list_hpoly.append(np.array([result_hpoly.x[idx], result_hpoly.x[idx+1]]))
    gamma_list_hpoly.append(result_hpoly.x[idx+2])

print(f"H-Polyhedral Loss: {result_hpoly.fun:.4f}")

# Max-Min optimizasyonu (2 grup, her grupta 3-4 hiper-düzlem - her mavi küme için)
print("\nMax-Min optimizasyonu başlıyor...")
r = 2
group_sizes = [4, 3]  # Grup 1: 4 düzlem (merkez mavi), Grup 2: 3 düzlem (sağ üst mavi)
total_planes = sum(group_sizes)
x0_maxmin = np.random.randn(total_planes * 3) * 0.1
result_maxmin = minimize(maxmin_loss, x0_maxmin, args=(A, B, r, group_sizes), 
                         method='BFGS', options={'maxiter': 1000})

groups_maxmin = []
idx = 0
for g_size in group_sizes:
    group_w = []
    group_gamma = []
    for _ in range(g_size):
        group_w.append(np.array([result_maxmin.x[idx], result_maxmin.x[idx+1]]))
        group_gamma.append(result_maxmin.x[idx+2])
        idx += 3
    groups_maxmin.append((group_w, group_gamma))

print(f"Max-Min Loss: {result_maxmin.fun:.4f}")

# Görselleştirme
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# H-Polyhedral görselleştirme
ax1 = axes[0]
ax1.scatter(A[:, 0], A[:, 1], c='blue', marker='o', s=50, alpha=0.6, label='Sınıf A (+)')
ax1.scatter(B[:, 0], B[:, 1], c='red', marker='x', s=50, alpha=0.6, label='Sınıf B (-)')

x_range = np.linspace(-5, 5, 300)
y_range = np.linspace(-5, 5, 300)
X_grid, Y_grid = np.meshgrid(x_range, y_range)

# H-Polyhedral karar bölgesi
Z_hpoly = np.zeros_like(X_grid)
for i in range(X_grid.shape[0]):
    for j in range(X_grid.shape[1]):
        x_point = np.array([X_grid[i, j], Y_grid[i, j]])
        # psi(x) = max_j { w_j^T x - gamma_j }
        psi = max([w @ x_point - gamma for w, gamma in zip(w_list_hpoly, gamma_list_hpoly)])
        Z_hpoly[i, j] = psi

# Karar sınırı: psi(x) = 0
# A için: psi(x) < -1 (mavi bölge)
# B için: psi(x) > +1 (kırmızı bölge)
ax1.contour(X_grid, Y_grid, Z_hpoly, levels=[-1, 0, 1], colors=['blue', 'green', 'red'], 
           linewidths=2, linestyles=['--', '-', '--'])
ax1.contourf(X_grid, Y_grid, Z_hpoly, levels=[-10, -1, 1, 10], 
            colors=['lightblue', 'white', 'lightcoral'], alpha=0.3)

# Hiper-düzlemleri çiz
colors_hpoly = ['orange', 'purple', 'brown', 'pink', 'cyan', 'magenta']
for idx, (w, gamma) in enumerate(zip(w_list_hpoly, gamma_list_hpoly)):
    if abs(w[1]) > 1e-6:
        y_line = (gamma - w[0] * x_range) / w[1]
        ax1.plot(x_range, y_line, '--', color=colors_hpoly[idx], 
                linewidth=1.5, alpha=0.7)
    elif abs(w[0]) > 1e-6:
        x_line = gamma / w[0]
        ax1.axvline(x=x_line, linestyle='--', color=colors_hpoly[idx], 
                   linewidth=1.5, alpha=0.7)

# Legend için bölge açıklamaları
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='lightblue', alpha=0.5, label='Mavi bölge: A (dış)'),
    Patch(facecolor='lightcoral', alpha=0.5, label='Kırmızı bölge: B (iç)'),
    plt.Line2D([0], [0], color='green', linewidth=2, label='Karar sınırı (ψ=0)'),
    plt.Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Marjin (ψ=-1)'),
    plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Marjin (ψ=+1)')
]
ax1.legend(handles=legend_elements, fontsize=8)

ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_title('H-Polyhedral Separability')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-5, 5)
ax1.set_ylim(-5, 5)

# Max-Min görselleştirme
ax2 = axes[1]
ax2.scatter(A[:, 0], A[:, 1], c='blue', marker='o', s=50, alpha=0.6, label='Sınıf A (+)')
ax2.scatter(B[:, 0], B[:, 1], c='red', marker='x', s=50, alpha=0.6, label='Sınıf B (-)')

# Max-Min karar bölgesi
Z_maxmin = np.zeros_like(X_grid)
for i in range(X_grid.shape[0]):
    for j in range(X_grid.shape[1]):
        x_point = np.array([X_grid[i, j], Y_grid[i, j]])
        # phi(x) = max_i min_{j in J_i} { w_j^T x - gamma_j }
        max_val = -np.inf
        for w_list, gamma_list in groups_maxmin:
            min_val = min([w @ x_point - gamma for w, gamma in zip(w_list, gamma_list)])
            max_val = max(max_val, min_val)
        Z_maxmin[i, j] = max_val

# Karar sınırı: phi(x) = 0
# A için: phi(x) < -1 (mavi bölge)
# B için: phi(x) > +1 (kırmızı bölge)
ax2.contour(X_grid, Y_grid, Z_maxmin, levels=[-1, 0, 1], colors=['blue', 'purple', 'red'], 
           linewidths=2, linestyles=['--', '-', '--'])
ax2.contourf(X_grid, Y_grid, Z_maxmin, levels=[-10, -1, 1, 10], 
            colors=['lightblue', 'white', 'lightcoral'], alpha=0.3)

# Gruplardaki hiper-düzlemleri farklı renklerde çiz
# Her grup bir mavi kümeyi kırmızılardan ayırmaya çalışır
colors = ['orange', 'purple']
for g_idx, (w_list, gamma_list) in enumerate(groups_maxmin):
    for idx, (w, gamma) in enumerate(zip(w_list, gamma_list)):
        if abs(w[1]) > 1e-6:
            y_line = (gamma - w[0] * x_range) / w[1]
            ax2.plot(x_range, y_line, '--', color=colors[g_idx], linewidth=1.5, alpha=0.7)
        elif abs(w[0]) > 1e-6:
            x_line = gamma / w[0]
            ax2.axvline(x=x_line, linestyle='--', color=colors[g_idx], linewidth=1.5, alpha=0.7)

# Legend için bölge açıklamaları
legend_elements_mm = [
    Patch(facecolor='lightblue', alpha=0.5, label='Mavi bölge: A (dış)'),
    Patch(facecolor='lightcoral', alpha=0.5, label='Kırmızı bölge: B (iç)'),
    plt.Line2D([0], [0], color='purple', linewidth=2, label='Karar sınırı (φ=0)'),
    plt.Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Marjin (φ=-1)'),
    plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Marjin (φ=+1)')
]
ax2.legend(handles=legend_elements_mm, fontsize=8)

ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_title('Max–Min Separability')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-5, 5)
ax2.set_ylim(-5, 5)

plt.tight_layout()
plt.savefig('separability_comparison.png', dpi=300, bbox_inches='tight')
print("\nGörsel 'separability_comparison.png' olarak kaydedildi.")

# ============================================================
# 3D GÖRSELLEŞTİRME - Hiper-düzlemleri 3 boyutta göster
# ============================================================
print("\n3D görselleştirme hazırlanıyor...")

fig_3d = plt.figure(figsize=(16, 8))

# 1. H-Polyhedral 3D görselleştirme
ax_3d_1 = fig_3d.add_subplot(121, projection='3d')

# Veri noktalarını 3D'de göster (z=0 düzleminde)
ax_3d_1.scatter(A[:, 0], A[:, 1], np.zeros(len(A)), c='blue', marker='o', s=50, alpha=0.8, label='Sınıf A (+)')
ax_3d_1.scatter(B[:, 0], B[:, 1], np.zeros(len(B)), c='red', marker='x', s=50, alpha=0.8, label='Sınıf B (-)')

# Her hiper-düzlemi 3D'de çiz: w_1*x_1 + w_2*x_2 - gamma = 0
# Bu denklemi z = w_1*x_1 + w_2*x_2 - gamma şeklinde 3D yüzey olarak gösterebiliriz
x_3d = np.linspace(-5, 5, 30)
y_3d = np.linspace(-5, 5, 30)
X_3d, Y_3d = np.meshgrid(x_3d, y_3d)

for idx, (w, gamma) in enumerate(zip(w_list_hpoly, gamma_list_hpoly)):
    # Hiper-düzlem denklemi: w[0]*x + w[1]*y - gamma = 0
    # 3D yüzey olarak: Z = w[0]*X + w[1]*Y - gamma
    Z_plane = w[0] * X_3d + w[1] * Y_3d - gamma
    ax_3d_1.plot_surface(X_3d, Y_3d, Z_plane, alpha=0.2, cmap='viridis')
    
    # z=0 kesişim çizgisini belirgin göster
    if abs(w[1]) > 1e-6:
        y_line = (gamma - w[0] * x_range) / w[1]
        ax_3d_1.plot(x_range, y_line, np.zeros_like(x_range), 'g-', linewidth=2, alpha=0.7)

# z=0 referans düzlemini ekle
Z_zero = np.zeros_like(X_3d)
ax_3d_1.plot_surface(X_3d, Y_3d, Z_zero, alpha=0.05, color='gray')

ax_3d_1.set_xlabel('x1')
ax_3d_1.set_ylabel('x2')
ax_3d_1.set_zlabel('z = w^T x - γ')
ax_3d_1.set_title('H-Polyhedral: Hiper-düzlemler (3D)')
ax_3d_1.legend()
ax_3d_1.set_xlim(-5, 5)
ax_3d_1.set_ylim(-5, 5)
ax_3d_1.view_init(elev=20, azim=45)

# 2. Max-Min 3D görselleştirme
ax_3d_2 = fig_3d.add_subplot(122, projection='3d')

# Veri noktalarını 3D'de göster
ax_3d_2.scatter(A[:, 0], A[:, 1], np.zeros(len(A)), c='blue', marker='o', s=50, alpha=0.8, label='Sınıf A (+)')
ax_3d_2.scatter(B[:, 0], B[:, 1], np.zeros(len(B)), c='red', marker='x', s=50, alpha=0.8, label='Sınıf B (-)')

# Her gruptaki hiper-düzlemleri farklı renklerde göster
colors_3d = ['orange', 'purple']
for g_idx, (w_list, gamma_list) in enumerate(groups_maxmin):
    for w, gamma in zip(w_list, gamma_list):
        Z_plane = w[0] * X_3d + w[1] * Y_3d - gamma
        ax_3d_2.plot_surface(X_3d, Y_3d, Z_plane, alpha=0.15, color=colors_3d[g_idx])
        
        # z=0 kesişim çizgisini göster
        if abs(w[1]) > 1e-6:
            y_line = (gamma - w[0] * x_range) / w[1]
            ax_3d_2.plot(x_range, y_line, np.zeros_like(x_range), '--', 
                        color=colors_3d[g_idx], linewidth=2, alpha=0.8)

# z=0 referans düzlemini ekle
ax_3d_2.plot_surface(X_3d, Y_3d, Z_zero, alpha=0.05, color='gray')

ax_3d_2.set_xlabel('x1')
ax_3d_2.set_ylabel('x2')
ax_3d_2.set_zlabel('z = w^T x - γ')
ax_3d_2.set_title('Max-Min: Hiper-düzlemler (3D)')
ax_3d_2.legend()
ax_3d_2.set_xlim(-5, 5)
ax_3d_2.set_ylim(-5, 5)
ax_3d_2.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig('separability_3d.png', dpi=300, bbox_inches='tight')
print("3D görsel 'separability_3d.png' olarak kaydedildi.")

print("\nVeri Yapısı:")
print("  Sınıf A (Pozitif): İki ayrı mavi küme")
print("    - Merkez grubu (~25 nokta)")
print("    - Sağ üst köşe grubu (~15 nokta, x1≈4, x2≈4)")
print("  Sınıf B (Negatif): İki ayrı kırmızı grup")
print("    - Merkez çevresi grubu (~48 nokta - sağ, sol, üst, alt)")
print("    - (2.2, 2.2) noktası çevresi grubu (~14 nokta)")
print("\nGrup Açıklaması:")
print("  Max-Min yönteminde 'grup' kavramı:")
print("    - Her grup, bir mavi kümeyi kırmızılardan ayırmak için hiper-düzlemler kümesidir")
print("    - Grup 1: Merkezdeki mavi kümeyi ayırır (4 hiper-düzlem)")
print("    - Grup 2: Sağ üst köşedeki mavi kümeyi ayırır (3 hiper-düzlem)")
print("    - Her grup kendi içinde MIN (en kısıtlayıcı) alır")
print("    - Gruplar arasında MAX (en esnek) alınır")
print("    - Bu sayede her iki mavi kümeyi de kırmızılardan esnek şekilde ayırabilir")
plt.show()
