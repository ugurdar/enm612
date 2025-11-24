"""
3 Boyutlu H-Polyhedral ve Max-Min Separability Görselleştirmesi
Hiper-düzlemler 3D'de düzlem yüzeyleri olarak gösterilir
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import minimize

# Veri kümesi oluşturma (3 boyutlu, lineer ayrılamayan)
np.random.seed(42)

# Sınıf A: İki ayrı mavi küme (merkez + sağ üst ön)
A1 = np.random.randn(25, 3) * 0.5  # Merkezdeki mavi grup
A2 = np.random.randn(15, 3) * 0.4 + np.array([4, 4, 4])  # Sağ üst ön köşedeki mavi grup
A = np.vstack([A1, A2])  # Sınıf A (pozitif) - 2 mavi grup

# Sınıf B: İki ayrı kırmızı grup (merkez çevresi + (2,2,2) noktası yakını)
B1 = np.random.randn(12, 3) * 0.4 + np.array([3, 0, 0])    # Merkez çevresi - Sağ
B2 = np.random.randn(12, 3) * 0.4 + np.array([-3, 0, 0])   # Merkez çevresi - Sol  
B3 = np.random.randn(12, 3) * 0.4 + np.array([0, 3, 0])    # Merkez çevresi - Yukarı
B4 = np.random.randn(12, 3) * 0.4 + np.array([0, -3, 0])   # Merkez çevresi - Aşağı
B5 = np.random.randn(14, 3) * 0.35 + np.array([2.2, 2.2, 2.2])  # (2.2, 2.2, 2.2) noktası çevresi
B = np.vstack([B1, B2, B3, B4, B5])  # Sınıf B (negatif) - 2 ayrı kırmızı grup

# H-Polyhedral Loss Fonksiyonu (3D)
def hpoly_loss(params, A, B, h):
    """
    H-Polyhedral loss fonksiyonu (3 boyut)
    Karar fonksiyonu: psi(x) = max_j { w_j^T x - gamma_j }
    
    params: [w1_x, w1_y, w1_z, gamma1, w2_x, w2_y, w2_z, gamma2, ...]
    """
    m, k = len(A), len(B)
    w_list = []
    gamma_list = []
    
    for j in range(h):
        idx = j * 4  # 3D: 3 bileşen + 1 gamma
        w_list.append(np.array([params[idx], params[idx+1], params[idx+2]]))
        gamma_list.append(params[idx+3])
    
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

# Max-Min Loss Fonksiyonu (3D)
def maxmin_loss(params, A, B, group_sizes):
    """
    Max-Min loss fonksiyonu (3 boyut)
    Karar fonksiyonu: psi(x) = max_i { min_j∈G_i { w_j^T x - gamma_j } }
    
    params: [w1_x, w1_y, w1_z, gamma1, w2_x, w2_y, w2_z, gamma2, ...]
    """
    m, k = len(A), len(B)
    r = len(group_sizes)
    total_planes = sum(group_sizes)
    
    w_list = []
    gamma_list = []
    
    for j in range(total_planes):
        idx = j * 4
        w_list.append(np.array([params[idx], params[idx+1], params[idx+2]]))
        gamma_list.append(params[idx+3])
    
    loss = 0.0
    
    # A için: TÜM gruplardaki TÜM düzlemler içinde tutmalı
    for a in A:
        for j in range(total_planes):
            val = np.dot(w_list[j], a) - gamma_list[j]
            if val > -1:
                loss += max(0, val + 1)**2
    
    # B için: En az bir grupta dışarıda olmalı (max-min yapısı)
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
    
    return loss

def plot_plane_3d(ax, w, gamma, xlim, ylim, zlim, color, alpha=0.3, label=None):
    """
    3D düzlem çizimi: w^T x = gamma
    w[0]*x + w[1]*y + w[2]*z = gamma
    z = (gamma - w[0]*x - w[1]*y) / w[2]
    """
    if abs(w[2]) < 1e-6:  # z bileşeni çok küçükse düzlem dikeye yakındır
        return
    
    # Grid oluştur
    x_range = np.linspace(xlim[0], xlim[1], 10)
    y_range = np.linspace(ylim[0], ylim[1], 10)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (gamma - w[0]*X - w[1]*Y) / w[2]
    
    # Sadece z limitleri içinde olan kısmı çiz
    Z = np.clip(Z, zlim[0], zlim[1])
    
    ax.plot_surface(X, Y, Z, alpha=alpha, color=color, edgecolor='none', label=label)

def plot_decision_region_3d(ax, w_list, gamma_list, xlim, ylim, zlim, is_maxmin=False, group_sizes=None):
    """
    3D karar bölgesini göster (sadece noktalarla sampling yaparak)
    """
    # 3D grid oluştur (daha seyrek - performans için)
    x_range = np.linspace(xlim[0], xlim[1], 30)
    y_range = np.linspace(ylim[0], ylim[1], 30)
    z_range = np.linspace(zlim[0], zlim[1], 30)
    
    # Karar bölgesini hesapla
    points_inside = []
    points_outside = []
    
    for x in x_range[::3]:  # Daha da seyrekleştir
        for y in y_range[::3]:
            for z in z_range[::3]:
                point = np.array([x, y, z])
                
                if is_maxmin:
                    # Max-Min karar
                    group_mins = []
                    plane_idx = 0
                    for i in range(len(group_sizes)):
                        group_vals = []
                        for _ in range(group_sizes[i]):
                            val = np.dot(w_list[plane_idx], point) - gamma_list[plane_idx]
                            group_vals.append(val)
                            plane_idx += 1
                        group_mins.append(min(group_vals))
                    decision = max(group_mins)
                else:
                    # H-Polyhedral karar
                    vals = [np.dot(w, point) - gamma for w, gamma in zip(w_list, gamma_list)]
                    decision = max(vals)
                
                if decision <= 0:
                    points_inside.append([x, y, z])
                else:
                    points_outside.append([x, y, z])
    
    if points_inside:
        points_inside = np.array(points_inside)
        ax.scatter(points_inside[:, 0], points_inside[:, 1], points_inside[:, 2], 
                  c='lightblue', alpha=0.1, s=5, marker='.')
    
    if points_outside:
        points_outside = np.array(points_outside)
        ax.scatter(points_outside[:, 0], points_outside[:, 1], points_outside[:, 2], 
                  c='lightcoral', alpha=0.1, s=5, marker='.')

print("\nVeri Yapısı (3D):")
print("  Sınıf A (Pozitif): İki ayrı mavi küme")
print("    - Merkez grubu (~25 nokta)")
print("    - Sağ üst ön köşe grubu (~15 nokta, x≈4, y≈4, z≈4)")
print("  Sınıf B (Negatif): İki ayrı kırmızı grup")
print("    - Merkez çevresi grubu (~48 nokta)")
print("    - (2.2, 2.2, 2.2) noktası çevresi grubu (~14 nokta)")

# H-Polyhedral optimizasyonu (3D - 6 düzlem)
print("\n" + "="*60)
print("H-Polyhedral Separability (3D)")
print("="*60)
print("Optimizasyon başlıyor...")
h = 6
initial_params = np.random.randn(h * 4) * 0.1  # 4 parametre per düzlem (wx, wy, wz, gamma)

result = minimize(hpoly_loss, initial_params, args=(A, B, h), 
                 method='BFGS', options={'maxiter': 1000})

hpoly_params = result.x
hpoly_loss_val = result.fun

w_hpoly = []
gamma_hpoly = []
for j in range(h):
    idx = j * 4
    w_hpoly.append(np.array([hpoly_params[idx], hpoly_params[idx+1], hpoly_params[idx+2]]))
    gamma_hpoly.append(hpoly_params[idx+3])

print(f"Loss: {hpoly_loss_val:.4f}")
print(f"Düzlem sayısı: {h}")

# Max-Min optimizasyonu (3D - 2 grup, 4+3 düzlem)
print("\n" + "="*60)
print("Max-Min Separability (3D)")
print("="*60)
print("Optimizasyon başlıyor...")
r = 2
group_sizes = [4, 3]
total_planes = sum(group_sizes)

initial_params = np.random.randn(total_planes * 4) * 0.1

result = minimize(maxmin_loss, initial_params, args=(A, B, group_sizes),
                 method='BFGS', options={'maxiter': 1000})

maxmin_params = result.x
maxmin_loss_val = result.fun

w_maxmin = []
gamma_maxmin = []
for j in range(total_planes):
    idx = j * 4
    w_maxmin.append(np.array([maxmin_params[idx], maxmin_params[idx+1], maxmin_params[idx+2]]))
    gamma_maxmin.append(maxmin_params[idx+3])

print(f"Loss: {maxmin_loss_val:.4f}")
print(f"Grup sayısı: {r}")
print(f"Her gruptaki düzlem sayıları: {group_sizes}")

# Görselleştirme - 4 subplot
print("\nGörselleştirme oluşturuluyor...")
fig = plt.figure(figsize=(20, 10))

# Eksen limitleri
xlim, ylim, zlim = (-5, 6), (-5, 6), (-5, 6)

# 1. H-Polyhedral - Veri Noktaları + Düzlemler
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.scatter(A[:, 0], A[:, 1], A[:, 2], c='blue', marker='o', s=50, alpha=0.7, label='Sınıf A (Pozitif)')
ax1.scatter(B[:, 0], B[:, 1], B[:, 2], c='red', marker='x', s=50, alpha=0.7, label='Sınıf B (Negatif)')

# H-Polyhedral düzlemlerini çiz
colors_hpoly = plt.cm.viridis(np.linspace(0, 1, h))
for j in range(h):
    plot_plane_3d(ax1, w_hpoly[j], gamma_hpoly[j], xlim, ylim, zlim, 
                  colors_hpoly[j], alpha=0.2)

ax1.set_xlabel('$x_1$', fontsize=11)
ax1.set_ylabel('$x_2$', fontsize=11)
ax1.set_zlabel('$x_3$', fontsize=11)
ax1.set_title(f'H-Polyhedral ({h} Düzlem)\nLoss: {hpoly_loss_val:.4f}', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
ax1.set_zlim(zlim)
ax1.view_init(elev=20, azim=45)

# 2. H-Polyhedral - Karar Bölgesi
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.scatter(A[:, 0], A[:, 1], A[:, 2], c='blue', marker='o', s=50, alpha=0.8, label='Sınıf A')
ax2.scatter(B[:, 0], B[:, 1], B[:, 2], c='red', marker='x', s=50, alpha=0.8, label='Sınıf B')

# Karar bölgesini göster
plot_decision_region_3d(ax2, w_hpoly, gamma_hpoly, xlim, ylim, zlim, is_maxmin=False)

ax2.set_xlabel('$x_1$', fontsize=11)
ax2.set_ylabel('$x_2$', fontsize=11)
ax2.set_zlabel('$x_3$', fontsize=11)
ax2.set_title('H-Polyhedral Karar Bölgesi\n(Açık mavi: İçeride, Açık kırmızı: Dışarıda)', 
             fontsize=12, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.set_zlim(zlim)
ax2.view_init(elev=20, azim=45)

# 3. Max-Min - Veri Noktaları + Düzlemler
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.scatter(A[:, 0], A[:, 1], A[:, 2], c='blue', marker='o', s=50, alpha=0.7, label='Sınıf A (Pozitif)')
ax3.scatter(B[:, 0], B[:, 1], B[:, 2], c='red', marker='x', s=50, alpha=0.7, label='Sınıf B (Negatif)')

# Max-Min düzlemlerini çiz (gruplara göre renklendir)
colors_maxmin = ['orange', 'purple']
plane_idx = 0
for i in range(r):
    for j in range(group_sizes[i]):
        plot_plane_3d(ax3, w_maxmin[plane_idx], gamma_maxmin[plane_idx], 
                     xlim, ylim, zlim, colors_maxmin[i], alpha=0.25)
        plane_idx += 1

# Legend için dummy plotlar
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='blue', alpha=0.7, label='Sınıf A'),
    Patch(facecolor='red', alpha=0.7, label='Sınıf B'),
    Patch(facecolor='orange', alpha=0.3, label=f'Grup 1 ({group_sizes[0]} düzlem)'),
    Patch(facecolor='purple', alpha=0.3, label=f'Grup 2 ({group_sizes[1]} düzlem)')
]

ax3.set_xlabel('$x_1$', fontsize=11)
ax3.set_ylabel('$x_2$', fontsize=11)
ax3.set_zlabel('$x_3$', fontsize=11)
ax3.set_title(f'Max-Min ({r} Grup: {group_sizes})\nLoss: {maxmin_loss_val:.4f}', 
             fontsize=12, fontweight='bold')
ax3.legend(handles=legend_elements, loc='upper left', fontsize=9)
ax3.set_xlim(xlim)
ax3.set_ylim(ylim)
ax3.set_zlim(zlim)
ax3.view_init(elev=20, azim=45)

# 4. Max-Min - Karar Bölgesi
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.scatter(A[:, 0], A[:, 1], A[:, 2], c='blue', marker='o', s=50, alpha=0.8, label='Sınıf A')
ax4.scatter(B[:, 0], B[:, 1], B[:, 2], c='red', marker='x', s=50, alpha=0.8, label='Sınıf B')

# Karar bölgesini göster
plot_decision_region_3d(ax4, w_maxmin, gamma_maxmin, xlim, ylim, zlim, 
                       is_maxmin=True, group_sizes=group_sizes)

ax4.set_xlabel('$x_1$', fontsize=11)
ax4.set_ylabel('$x_2$', fontsize=11)
ax4.set_zlabel('$x_3$', fontsize=11)
ax4.set_title('Max-Min Karar Bölgesi\n(Açık mavi: İçeride, Açık kırmızı: Dışarıda)', 
             fontsize=12, fontweight='bold')
ax4.legend(loc='upper left', fontsize=9)
ax4.set_xlim(xlim)
ax4.set_ylim(ylim)
ax4.set_zlim(zlim)
ax4.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig('separability_comparison_3d.png', dpi=300, bbox_inches='tight')
print("\nGörselleştirme 'separability_comparison_3d.png' olarak kaydedildi")
print("\n4 Grafik:")
print("  1. Sol üst: H-Polyhedral - Düzlemler (renkli yüzeyler)")
print("  2. Sağ üst: H-Polyhedral - Karar bölgesi (hacim görselleştirmesi)")
print("  3. Sol alt: Max-Min - Düzlemler (grup renklerine göre)")
print("  4. Sağ alt: Max-Min - Karar bölgesi (hacim görselleştirmesi)")
print("\nNot: Grafikleri fareyle döndürerek farklı açılardan inceleyebilirsiniz")
