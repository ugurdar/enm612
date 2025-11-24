
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Türkçe karakter desteği
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ==================== VERİ SETİ OLUŞTURMA ====================
np.random.seed(42)

# Sınıf A (Kırmızı - Negatif bölge)
A = np.array([
    [1.0, 1.2], [1.5, 1.0], [0.8, 1.5], [1.2, 0.9],
    [0.5, 1.8], [1.8, 1.4], [1.3, 1.1]
])

# Sınıf B (Mavi - Pozitif bölge)
B = np.array([
    [3.5, 3.5], [4.0, 4.0], [3.2, 3.8], [4.2, 3.6],
    [3.8, 4.2], [3.3, 3.3], [4.5, 4.1]
])

# ==================== PCF PARAMETRELERİ ====================
a = np.array([2.5, 2.5])  # Koni tepesi
w = np.array([0.7, 0.7])  # Ağırlık vektörü
xi = 1.0                   # L1-norm katsayısı (önemli!)
gamma = 3.5                # Offset parametresi

# PCF fonksiyonu
def pcf(x, w, xi, gamma, a):
    """Polyhedral Conic Function"""
    return np.dot(w, x - a) + xi * np.sum(np.abs(x - a)) - gamma

# Her bölgede PCF'nin açık formu
def pcf_quadrant(x, w, xi, gamma, a, quadrant):
    """Her quadrant'ta açık doğrusal form"""
    dx = x[0] - a[0]
    dy = x[1] - a[1]
    
    if quadrant == 1:  # x > a[0], y > a[1]
        return w[0]*dx + w[1]*dy + xi*(dx + dy) - gamma
    elif quadrant == 2:  # x < a[0], y > a[1]
        return w[0]*dx + w[1]*dy + xi*(-dx + dy) - gamma
    elif quadrant == 3:  # x < a[0], y < a[1]
        return w[0]*dx + w[1]*dy + xi*(-dx - dy) - gamma
    else:  # quadrant == 4: x > a[0], y < a[1]
        return w[0]*dx + w[1]*dy + xi*(dx - dy) - gamma

# ==================== GRID OLUŞTUR ====================
x_range = np.linspace(-0.5, 5.5, 400)
y_range = np.linspace(-0.5, 5.5, 400)
X, Y = np.meshgrid(x_range, y_range)

# Her nokta için PCF değerini hesapla
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = np.array([X[i, j], Y[i, j]])
        Z[i, j] = pcf(point, w, xi, gamma, a)

# A ve B noktaları için Z değerleri
Z_A = np.array([pcf(point, w, xi, gamma, a) for point in A])
Z_B = np.array([pcf(point, w, xi, gamma, a) for point in B])

# ==================== FİGÜR 1: TEORİK YAPI ====================
fig1 = plt.figure(figsize=(20, 6))

# ===== SOL PANEL: 4 Bölge (Quadrants) + Farklı Eğimler =====
ax1 = fig1.add_subplot(131)

# 4 bölgeyi renklendir
quadrant_colors = ['#fff5e6', '#e6f5ff', '#ffe6f5', '#f5ffe6']
ax1.axhline(a[1], color='black', linewidth=2, linestyle='--', alpha=0.3)
ax1.axvline(a[0], color='black', linewidth=2, linestyle='--', alpha=0.3)

# Her quadrant için gradient göster
for quad_idx, (x_start, x_end, y_start, y_end, color) in enumerate([
    (a[0], 5.5, a[1], 5.5, quadrant_colors[0]),  # Q1
    (-0.5, a[0], a[1], 5.5, quadrant_colors[1]),  # Q2
    (-0.5, a[0], -0.5, a[1], quadrant_colors[2]),  # Q3
    (a[0], 5.5, -0.5, a[1], quadrant_colors[3]),  # Q4
]):
    ax1.fill([x_start, x_end, x_end, x_start], 
            [y_start, y_start, y_end, y_end], 
            color=color, alpha=0.4, zorder=0)

# Kontur çizgileri
contour_levels = np.linspace(-4, 4, 15)
ax1.contour(X, Y, Z, levels=contour_levels, colors='gray', 
           linewidths=0.5, alpha=0.5)

# Önemli seviyeler
ax1.contour(X, Y, Z, levels=[0], colors='black', linewidths=3, 
           linestyles='--', label='g(x) = 0')
ax1.contour(X, Y, Z, levels=[-1, 1], colors=['red', 'blue'], 
           linewidths=2.5)

# 4 ana yön okları (her quadrant'ın gradyenti)
arrow_props = dict(arrowstyle='->', lw=2.5, color='darkgreen')
directions = [
    ([a[0], a[0]+1.5], [a[1], a[1]+1.5], f'Q1: ∇g = [{w[0]+xi:.1f}, {w[1]+xi:.1f}]'),
    ([a[0], a[0]-1.5], [a[1], a[1]+1.5], f'Q2: ∇g = [{w[0]-xi:.1f}, {w[1]+xi:.1f}]'),
    ([a[0], a[0]-1.5], [a[1], a[1]-1.5], f'Q3: ∇g = [{w[0]-xi:.1f}, {w[1]-xi:.1f}]'),
    ([a[0], a[0]+1.5], [a[1], a[1]-1.5], f'Q4: ∇g = [{w[0]+xi:.1f}, {w[1]-xi:.1f}]'),
]

for x_arr, y_arr, label in directions:
    ax1.annotate('', xy=(x_arr[1], y_arr[1]), xytext=(x_arr[0], y_arr[0]),
                arrowprops=arrow_props)
    # Label ekle
    mid_x, mid_y = (x_arr[0] + x_arr[1])/2, (y_arr[0] + y_arr[1])/2
    ax1.text(mid_x + 0.3, mid_y + 0.3, label, fontsize=8, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Veri noktaları
ax1.scatter(A[:, 0], A[:, 1], c='red', s=150, marker='o',
           edgecolors='darkred', linewidths=2, label='Sınıf A', zorder=10)
ax1.scatter(B[:, 0], B[:, 1], c='blue', s=150, marker='s',
           edgecolors='darkblue', linewidths=2, label='Sınıf B', zorder=10)

# Koni tepesi
ax1.scatter(a[0], a[1], c='gold', s=600, marker='*',
           edgecolors='darkgreen', linewidths=3, 
           label=f'Koni Tepesi a', zorder=11)

ax1.set_xlabel('$x_1$', fontsize=13)
ax1.set_ylabel('$x_2$', fontsize=13)
ax1.set_title('PCF: 4 Bölge ve Farklı Gradyanlar\n(Her bölgede farklı doğrusal fonksiyon)', 
             fontsize=11, fontweight='bold')
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.5, 5.5)
ax1.set_ylim(-0.5, 5.5)
ax1.set_aspect('equal')

# ===== ORTA PANEL: Polyhedral Yapı (Piramit Kesitleri) =====
ax2 = fig1.add_subplot(132)

# L1-norm level sets (elmas şekilleri)
for radius, color, alpha, lw in [(0.8, 'green', 0.15, 1.5), 
                                  (1.6, 'green', 0.10, 1.5),
                                  (2.4, 'green', 0.07, 1.5)]:
    angles = np.linspace(0, 2*np.pi, 100)
    x_diamond = a[0] + radius * np.cos(angles) / (abs(np.cos(angles)) + abs(np.sin(angles)))
    y_diamond = a[1] + radius * np.sin(angles) / (abs(np.cos(angles)) + abs(np.sin(angles)))
    ax2.fill(x_diamond, y_diamond, color=color, alpha=alpha)
    ax2.plot(x_diamond, y_diamond, color='darkgreen', linewidth=lw, 
            linestyle=':', label=f'||x-a||₁ = {radius}' if radius == 0.8 else '')

# Kontur
ax2.contour(X, Y, Z, levels=[0], colors='black', linewidths=3, 
           linestyles='--', label='Karar Sınırı')
ax2.contour(X, Y, Z, levels=[-1, 1], colors=['red', 'blue'], linewidths=2.5)

# 4 ana kenar çizgisi (piramit kenarları)
for angle in [45, 135, 225, 315]:
    rad = np.deg2rad(angle)
    dx, dy = 3*np.cos(rad), 3*np.sin(rad)
    ax2.plot([a[0], a[0]+dx], [a[1], a[1]+dy], 
            'darkgreen', linewidth=2, linestyle='--', alpha=0.6)

# Veri
ax2.scatter(A[:, 0], A[:, 1], c='red', s=150, marker='o',
           edgecolors='darkred', linewidths=2, label='Sınıf A', zorder=10)
ax2.scatter(B[:, 0], B[:, 1], c='blue', s=150, marker='s',
           edgecolors='darkblue', linewidths=2, label='Sınıf B', zorder=10)

ax2.scatter(a[0], a[1], c='gold', s=600, marker='*',
           edgecolors='darkgreen', linewidths=3, zorder=11)

# Annotation
ax2.text(2.5, 4.8, 'L1-norm → Elmas Yapı\n(Polyhedral)', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
        ha='center')

ax2.set_xlabel('$x_1$', fontsize=13)
ax2.set_ylabel('$x_2$', fontsize=13)
ax2.set_title('Polyhedral (Çok-Yüzlü) Yapı\n(L1-norm level sets)', 
             fontsize=11, fontweight='bold')
ax2.legend(loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-0.5, 5.5)
ax2.set_ylim(-0.5, 5.5)
ax2.set_aspect('equal')

# ===== SAĞ PANEL: 3D Piramit Yapısı =====
ax3 = fig1.add_subplot(133, projection='3d')

# 3D yüzey
surf = ax3.plot_surface(X, Y, Z, cmap='RdYlBu_r', alpha=0.6, 
                        edgecolor='none', vmin=-4, vmax=4)

# z=0 düzlemi
X_plane, Y_plane = np.meshgrid(x_range, y_range)
Z_plane = np.zeros_like(X_plane)
ax3.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.15, color='yellow')

# 4 piramit kenarını 3D'de göster
for angle in [45, 135, 225, 315]:
    rad = np.deg2rad(angle)
    dx, dy = 3*np.cos(rad), 3*np.sin(rad)
    x_line = [a[0], a[0]+dx]
    y_line = [a[1], a[1]+dy]
    z_line = [-gamma, pcf(np.array([x_line[1], y_line[1]]), w, xi, gamma, a)]
    ax3.plot(x_line, y_line, z_line, 'g-', linewidth=3, alpha=0.8)

# Kontur
ax3.contour(X, Y, Z, levels=[0], colors='black', linewidths=3, 
           linestyles='--', offset=0, zdir='z')

# Veri noktaları
ax3.scatter(A[:, 0], A[:, 1], Z_A, c='red', s=150, marker='o',
           edgecolors='darkred', linewidths=2, zorder=10)
ax3.scatter(B[:, 0], B[:, 1], Z_B, c='blue', s=150, marker='s',
           edgecolors='darkblue', linewidths=2, zorder=10)

# Koni tepesi
ax3.scatter([a[0]], [a[1]], [-gamma], c='gold', s=500, marker='*',
           edgecolors='darkgreen', linewidths=3, zorder=11)

# Dikey çizgi (koni ekseni)
ax3.plot([a[0], a[0]], [a[1], a[1]], [-gamma, 3], 
        'g--', linewidth=2, alpha=0.5)

ax3.set_xlabel('$x_1$', fontsize=11, labelpad=8)
ax3.set_ylabel('$x_2$', fontsize=11, labelpad=8)
ax3.set_zlabel('g(x)', fontsize=11, labelpad=8)
ax3.set_title('3D Piramit (Polyhedral Conic)\n(Yeşil: 4 ana kenar)', 
             fontsize=11, fontweight='bold', pad=15)
ax3.view_init(elev=25, azim=45)
ax3.set_zlim(-4, 4)

cbar = fig1.colorbar(surf, ax=ax3, shrink=0.6, aspect=10)
cbar.set_label('g(x)', fontsize=10)

plt.tight_layout()
plt.savefig('pcf_theoretical_geometry.png', dpi=300, bbox_inches='tight')

# ==================== FİGÜR 2: 3D FARKLI AÇILAR ====================
fig2 = plt.figure(figsize=(20, 13))

views = [
    (25, 45, 'Açı 1: Standart'),
    (15, 135, 'Açı 2: Yan'),
    (60, 30, 'Açı 3: Üstten (Elmas Yapı)'),
    (10, 225, 'Açı 4: Arka'),
    (35, -45, 'Açı 5: Sol'),
    (5, 90, 'Açı 6: Yan Kesit')
]

for idx, (elev, azim, title) in enumerate(views, 1):
    ax = fig2.add_subplot(2, 3, idx, projection='3d')
    
    # Yüzey
    surf = ax.plot_surface(X, Y, Z, cmap='RdYlBu_r', alpha=0.65, 
                          edgecolor='none', vmin=-4, vmax=4)
    
    # Düzlem
    ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.15, color='yellow')
    
    # 4 piramit kenarı (önemli!)
    for angle in [45, 135, 225, 315]:
        rad = np.deg2rad(angle)
        dx, dy = 3*np.cos(rad), 3*np.sin(rad)
        x_line = [a[0], a[0]+dx]
        y_line = [a[1], a[1]+dy]
        z_line = [-gamma, pcf(np.array([x_line[1], y_line[1]]), w, xi, gamma, a)]
        ax.plot(x_line, y_line, z_line, 'lime', linewidth=2.5, alpha=0.9)
    
    # Veri
    ax.scatter(A[:, 0], A[:, 1], Z_A, c='red', s=100, marker='o',
              edgecolors='darkred', linewidths=1.5, zorder=10)
    ax.scatter(B[:, 0], B[:, 1], Z_B, c='blue', s=100, marker='s',
              edgecolors='darkblue', linewidths=1.5, zorder=10)
    
    # Tepe
    ax.scatter([a[0]], [a[1]], [-gamma], c='gold', s=300, marker='*',
              edgecolors='darkgreen', linewidths=2.5, zorder=11)
    
    ax.set_xlabel('$x_1$', fontsize=10, labelpad=5)
    ax.set_ylabel('$x_2$', fontsize=10, labelpad=5)
    ax.set_zlabel('g(x)', fontsize=10, labelpad=5)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
    ax.view_init(elev=elev, azim=azim)
    ax.set_zlim(-4, 4)

plt.tight_layout()
plt.savefig('pcf_3d_multiview.png', dpi=300, bbox_inches='tight')

plt.show()

print("✓ PCF TEORİK GEOMETRİ oluşturuldu:")
print("  - pcf_theoretical_geometry.png (4 bölge + polyhedral + 3D piramit)")
print("  - pcf_3d_multiview.png (3D 6 farklı açıdan)")
print(f"\n📐 TEORİK YAPI:")
print(f"  • 4 bölgede farklı gradyan: w + ξ·(±1, ±1)")
print(f"  • L1-norm → Elmas (diamond) seviye eğrileri")
print(f"  • 3D'de 4 kenarlı piramit (polyhedral cone)")
print(f"  • Tüm kenarlar 'a' noktasından çıkar")