# learn_hpoly_maxmin_fixed.py
# H-Polyhedral ve Max–Min ayırıcılarını DOĞRUDAN LP ile optimize eder.
# Max–Min'de "max(min(...))" için epigraf + küçük bağlama (epsilon) ile s = max_g z_{g} sağlanır.
# Çalıştır: python learn_hpoly_maxmin_fixed.py

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from scipy.optimize import linprog

# -------------------------------
# Ortam / renkler
# -------------------------------
os.makedirs("output", exist_ok=True)
rng = np.random.default_rng(2025)

B_FILL = "#eef2f6"    # B bölgesi
A_FILL = "#ffe3c2"    # A bölgesi
EDGE   = "#bf6b00"    # karar sınırı çizgisi
NEG_PT = "#2f3b52"    # B (negatif) noktalar
POS_PT = "#f59f00"    # A (pozitif) noktalar

# -------------------------------
# Veri (≈50 nokta, dağılım güncel)
# -------------------------------
A1 = rng.normal(loc=[-2.0,  0.2], scale=[0.23, 0.23], size=(14, 2))
A2 = rng.normal(loc=[ 1.1,  1.8], scale=[0.25, 0.22], size=(12, 2))
A  = np.vstack([A1, A2])                       # ~26 pozitif

B1 = rng.normal(loc=[-0.2, -0.4], scale=[1.05, 0.95], size=(14, 2))
B2 = rng.normal(loc=[ 1.4, -1.0], scale=[0.95, 1.05], size=(12, 2))
B  = np.vstack([B1, B2])                       # ~26 negatif

m, k = len(A), len(B)
d = 2

# -------------------------------
# Çizim yardımcıları
# -------------------------------
def grid_vals(f, xx, yy):
    G = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    v = f(G).reshape(xx.shape)
    return v

def draw_points(ax):
    ax.scatter(B[:,0], B[:,1], s=34, alpha=0.95, marker='s', color=NEG_PT, label="B (neg)")
    ax.scatter(A[:,0], A[:,1], s=42, alpha=0.95, edgecolor='black', linewidth=0.6, color=POS_PT, label="A (pos)")
    ax.grid(alpha=0.25)
    ax.set_xlim(-3.6, 3.3); ax.set_ylim(-2.6, 3.9)

def plot_region(ax, mask, xx, yy, boundary, title):
    cmap = ListedColormap([B_FILL, A_FILL])
    ax.contourf(xx, yy, mask, levels=[-0.5, 0.5, 1.5], cmap=cmap, alpha=1.0)
    ax.contour(xx, yy, boundary, levels=[0], linewidths=2, colors=EDGE)
    draw_points(ax)
    ax.set_title(title)
    pA = mpatches.Patch(color=A_FILL, label='A bölgesi')
    pB = mpatches.Patch(color=B_FILL, label='B bölgesi')
    ax.legend(handles=[pA, pB], loc='upper left', frameon=True)

# -------------------------------
# L1 eşitliklerini ekle
# -------------------------------
def add_l1_equalities(A_eq, b_eq, pW, nW, pUP, pUN, pG, nG, pGP, pGN, nvars):
    # w = u_pos - u_neg
    for t in range(nW):
        row = np.zeros(nvars)
        row[pW + t]  = 1.0
        row[pUP + t] = -1.0
        row[pUN + t] =  1.0
        A_eq.append(row); b_eq.append(0.0)
    # gamma = ug_pos - ug_neg
    for j in range(nG):
        row = np.zeros(nvars)
        row[pG + j]  = 1.0
        row[pGP + j] = -1.0
        row[pGN + j] =  1.0
        A_eq.append(row); b_eq.append(0.0)

# ==========================================================
# H-Polyhedral: ψ(x) = max_j (w_j·x − γ_j)
#   A-loss: [ψ(a)+1]_+  → eA ≥ ψ(a)+1
#   B-loss: [1−ψ(b)]_+ → eB ≥ 1−ψ(b)
# ψ epigraf: sA_i ≥ w_j·a_i−γ_j,  tB_l ≥ w_j·b_l−γ_j  (tüm j)
# ==========================================================
def learn_hpoly_LP(A, B, h=6, lam=2e-3):
    nfeat = A.shape[1]

    nW = h*nfeat
    nG = h
    nEA, nEB = m, k
    nSA, nTB = m, k
    nUP, nUN = nW, nW
    nGP, nGN = nG, nG

    nvars = nW + nG + nEA + nEB + nSA + nTB + nUP + nUN + nGP + nGN
    pW   = 0
    pG   = pW + nW
    pEA  = pG + nG
    pEB  = pEA + nEA
    pSA  = pEB + nEB
    pTB  = pSA + nSA
    pUP  = pTB + nTB
    pUN  = pUP + nUP
    pGP  = pUN + nUN
    pGN  = pGP + nGP

    c = np.zeros(nvars)
    c[pEA:pEA+nEA] = 1.0/m
    c[pEB:pEB+nEB] = 1.0/k
    c[pUP:pUP+nUP] = lam
    c[pUN:pUN+nUN] = lam
    c[pGP:pGP+nGP] = lam
    c[pGN:pGN+nGN] = lam

    A_ub, b_ub, A_eq, b_eq = [], [], [], []

    # A: sA_i ≥ w_j·a_i − γ_j  → −w_j·a_i + γ_j + sA_i ≤ 0
    for i in range(m):
        ai = A[i]
        for j in range(h):
            row = np.zeros(nvars)
            row[pW + j*nfeat : pW + (j+1)*nfeat] = -ai
            row[pG + j] = 1.0
            row[pSA + i] = 1.0
            A_ub.append(row); b_ub.append(0.0)
    # A hinge: eA_i ≥ sA_i + 1 → sA_i − eA_i ≤ −1
    for i in range(m):
        row = np.zeros(nvars)
        row[pSA + i] = 1.0
        row[pEA + i] = -1.0
        A_ub.append(row); b_ub.append(-1.0)

    # B: tB_l ≥ w_j·b_l − γ_j → −w_j·b_l + γ_j + tB_l ≤ 0
    for l in range(k):
        bl = B[l]
        for j in range(h):
            row = np.zeros(nvars)
            row[pW + j*nfeat : pW + (j+1)*nfeat] = -bl
            row[pG + j] = 1.0
            row[pTB + l] = 1.0
            A_ub.append(row); b_ub.append(0.0)
    # B hinge: eB_l ≥ 1 − tB_l → −tB_l − eB_l ≤ −1
    for l in range(k):
        row = np.zeros(nvars)
        row[pTB + l] = -1.0
        row[pEB + l] = -1.0
        A_ub.append(row); b_ub.append(-1.0)

    add_l1_equalities(A_eq, b_eq, pW, nW, pUP, pUN, pG, nG, pGP, pGN, nvars)

    bounds = []
    # w, gamma serbest
    for _ in range(nW + nG):
        bounds.append((None, None))
    # eA, eB >= 0
    for _ in range(nEA + nEB):
        bounds.append((0.0, None))
    # sA, tB serbest
    for _ in range(nSA + nTB):
        bounds.append((None, None))
    # L1 yardımcıları >= 0
    for _ in range(nUP + nUN + nGP + nGN):
        bounds.append((0.0, None))

    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"[H-Poly] LP başarısız: {res.message}")

    x = res.x
    W = x[pW:pW+nW].reshape(h, nfeat)
    gamma = x[pG:pG+nG]

    def psi_func(Z):
        Z = Z.reshape(-1,2)
        return (Z @ W.T - gamma[None,:]).max(axis=1)
    return W, gamma, psi_func

# ==========================================================
# Max–Min: φ(x) = max_g min_{j∈J_g} (w_j·x − γ_j)
#   A-loss: [φ(a)+1]_+  → eA ≥ sA + 1
#   B-loss: [1−φ(b)]_+ → eB ≥ 1 − sB
# Epigraf:
#   z_{i,g} ≤ w_j·a_i − γ_j (∀j∈J_g),   sA_i ≥ z_{i,g} (∀g),  sA_i ≤ z_{i,g} + r^A_{i,g}
#   u_{l,g} ≤ w_j·b_l − γ_j (∀j∈J_g),   sB_l ≥ u_{l,g} (∀g),  sB_l ≤ u_{l,g} + r^B_{l,g}
# Amaç: eA/eB + λ||w,γ||_1 + ε∑r   (ε çok küçük → s ≈ max_g z_g)
# ==========================================================
def learn_maxmin_LP(A, B, r=2, s=4, lam=2e-3, eps=1e-4):
    nfeat = A.shape[1]
    h = r*s
    groups = [list(range(i*s, (i+1)*s)) for i in range(r)]

    nW = h*nfeat
    nG = h
    nEA, nEB = m, k
    nSA, nSB = m, k
    nZA = m*r   # z_{i,g}
    nUB = k*r   # u_{l,g}
    nRA = m*r   # r^A_{i,g}
    nRB = k*r   # r^B_{l,g}
    nUP, nUN = nW, nW
    nGP, nGN = nG, nG

    nvars = nW + nG + nEA + nEB + nSA + nSB + nZA + nUB + nRA + nRB + nUP + nUN + nGP + nGN
    pW   = 0
    pG   = pW + nW
    pEA  = pG + nG
    pEB  = pEA + nEA
    pSA  = pEB + nEB
    pSB  = pSA + nSA
    pZA  = pSB + nSB
    pUB  = pZA + nZA
    pRA  = pUB + nUB
    pRB  = pRA + nRA
    pUP  = pRB + nRB
    pUN  = pUP + nUP
    pGP  = pUN + nUN
    pGN  = pGP + nGP

    c = np.zeros(nvars)
    c[pEA:pEA+nEA] = 1.0/m
    c[pEB:pEB+nEB] = 1.0/k
    c[pUP:pUP+nUP] = lam
    c[pUN:pUN+nUN] = lam
    c[pGP:pGP+nGP] = lam
    c[pGN:pGN+nGN] = lam
    # bağlama (s = max_g z) için küçük ceza
    c[pRA:pRA+nRA] = eps
    c[pRB:pRB+nRB] = eps

    A_ub, b_ub, A_eq, b_eq = [], [], [], []

    # ---- A noktaları: z_{i,g} ≤ w_j·a_i − γ_j ; sA_i ≥ z_{i,g} ; sA_i ≤ z_{i,g} + r^A_{i,g}
    for i in range(m):
        ai = A[i]
        for gi, gj in enumerate(groups):
            idx_z = pZA + i*r + gi
            idx_r = pRA + i*r + gi
            # z_{i,g} ≤ w_j·a_i − γ_j → −w_j·a + γ_j + z_{i,g} ≤ 0
            for j in gj:
                row = np.zeros(nvars)
                row[pW + j*nfeat : pW + (j+1)*nfeat] = -ai
                row[pG + j] = 1.0
                row[idx_z] = 1.0
                A_ub.append(row); b_ub.append(0.0)
            # sA_i ≥ z_{i,g} → z_{i,g} − sA_i ≤ 0
            row = np.zeros(nvars)
            row[idx_z] = 1.0
            row[pSA + i] = -1.0
            A_ub.append(row); b_ub.append(0.0)
            # sA_i ≤ z_{i,g} + r^A_{i,g} → sA_i − z_{i,g} − r_{i,g} ≤ 0
            row = np.zeros(nvars)
            row[pSA + i] = 1.0
            row[idx_z] = -1.0
            row[idx_r] = -1.0
            A_ub.append(row); b_ub.append(0.0)
    # hinge: eA_i ≥ sA_i + 1 → sA_i − eA_i ≤ −1
    for i in range(m):
        row = np.zeros(nvars)
        row[pSA + i] = 1.0
        row[pEA + i] = -1.0
        A_ub.append(row); b_ub.append(-1.0)

    # ---- B noktaları: u_{l,g} ≤ w_j·b_l − γ_j ; sB_l ≥ u_{l,g} ; sB_l ≤ u_{l,g} + r^B_{l,g}
    for l in range(k):
        bl = B[l]
        for gi, gj in enumerate(groups):
            idx_u = pUB + l*r + gi
            idx_r = pRB + l*r + gi
            for j in gj:
                row = np.zeros(nvars)
                row[pW + j*nfeat : pW + (j+1)*nfeat] = -bl
                row[pG + j] = 1.0
                row[idx_u] = 1.0
                A_ub.append(row); b_ub.append(0.0)
            # sB ≥ u
            row = np.zeros(nvars)
            row[idx_u] = 1.0
            row[pSB + l] = -1.0
            A_ub.append(row); b_ub.append(0.0)
            # sB ≤ u + r
            row = np.zeros(nvars)
            row[pSB + l] = 1.0
            row[idx_u] = -1.0
            row[idx_r] = -1.0
            A_ub.append(row); b_ub.append(0.0)
    # hinge: eB_l ≥ 1 − sB_l → −sB_l − eB_l ≤ −1
    for l in range(k):
        row = np.zeros(nvars)
        row[pSB + l] = -1.0
        row[pEB + l] = -1.0
        A_ub.append(row); b_ub.append(-1.0)

    # L1 eşitlikleri
    add_l1_equalities(A_eq, b_eq, pW, nW, pUP, pUN, pG, nG, pGP, pGN, nvars)

    # sınırlar
    bounds = []
    # w, gamma serbest
    for _ in range(nW + nG):
        bounds.append((None, None))
    # eA, eB >= 0
    for _ in range(nEA + nEB):
        bounds.append((0.0, None))
    # sA, sB, z, u serbest
    for _ in range(nSA + nSB + nZA + nUB):
        bounds.append((None, None))
    # rA, rB >= 0 (bağlama gevşekliği)
    for _ in range(nRA + nRB):
        bounds.append((0.0, None))
    # L1 yardımcıları >= 0
    for _ in range(nUP + nUN + nGP + nGN):
        bounds.append((0.0, None))

    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"[Max–Min] LP başarısız: {res.message}")

    x = res.x
    W = x[pW:pW+nW].reshape(h, nfeat)
    gamma = x[pG:pG+nG]

    def phi_func(Z):
        Z = Z.reshape(-1,2)
        H = Z @ W.T - gamma[None,:]         # (N,h)
        mins = []
        for gi, gj in enumerate(groups):
            mins.append(H[:, gj].min(axis=1))
        mins = np.stack(mins, axis=1)       # (N,r)
        return mins.max(axis=1)
    return W, gamma, groups, phi_func

# -------------------------------
# Öğren ve görselleştir
# -------------------------------
W_H, g_H, psi = learn_hpoly_LP(A, B, h=6, lam=2e-3)
W_M, g_M, groups_M, phi = learn_maxmin_LP(A, B, r=2, s=4, lam=2e-3, eps=1e-4)

xx, yy = np.meshgrid(np.linspace(-3.6,3.3,600), np.linspace(-2.6,3.9,600))
PSI = grid_vals(psi, xx, yy)   # ψ(x)
PHI = grid_vals(phi, xx, yy)   # φ(x)

mask_H = (PSI <= 0).astype(int)   # A = ψ≤0
mask_M = (PHI <= 0).astype(int)   # A = φ≤0

# Yan yana
fig, axes = plt.subplots(1,2, figsize=(12,4.8))
plot_region(axes[0], mask_H, xx, yy, PSI, "H-Polyhedral (LP)")
plot_region(axes[1], mask_M, xx, yy, PHI, "Max–Min (LP, bağlamalı)")
plt.suptitle("H-Polyhedral vs Max–Min — LP ile öğrenilmiş", y=1.02, fontsize=14)
plt.tight_layout()
fig.savefig("output/compare_lp.png", dpi=300)

# Tek tek
fig, ax = plt.subplots(figsize=(6,5))
plot_region(ax, mask_H, xx, yy, PSI, "H-Polyhedral (LP)")
plt.tight_layout(); fig.savefig("output/hpoly_lp.png", dpi=300)

fig, ax = plt.subplots(figsize=(6,5))
plot_region(ax, mask_M, xx, yy, PHI, "Max–Min (LP)")
plt.tight_layout(); fig.savefig("output/maxmin_lp.png", dpi=300)

# Eğitim doğrulukları
Xall = np.vstack([A,B])
ytrue = np.array([+1]*len(A) + [-1]*len(B))
pred_H = (psi(Xall) <= 0).astype(int)*2 - 1
pred_M = (phi(Xall) <= 0).astype(int)*2 - 1
print("Saved:", "output/hpoly_lp.png", "output/maxmin_lp.png", "output/compare_lp.png")
print(f"Accuracy  H-Poly={ (pred_H==ytrue).mean():.2f},  Max–Min={ (pred_M==ytrue).mean():.2f}")
