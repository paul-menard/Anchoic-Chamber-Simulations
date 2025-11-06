import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- Constantes
eps_r = 1.4
eps_0 = 8.854e-12
mu_0  = 4e-7*np.pi
c     = 3e8
eta_0 = np.sqrt(mu_0/eps_0)

# ---------- MODELE COUCHES (corrigé) ----------
# convention temporelle e^{+j ω t}
def eps_c(sigma, omega):
    return eps_0*eps_r - 1j*sigma/omega

def beta_eta(sigma, omega):
    epsc = eps_c(sigma, omega)
    beta = omega*np.sqrt(mu_0*epsc)     # constante de propagation
    eta  = np.sqrt(mu_0/epsc)           # impédance de couche
    return beta, eta

def M_layer(sigma, d, omega):
    beta, eta = beta_eta(sigma, omega)
    b = beta*d
    C = np.cos(b)
    S = np.sin(b)
    # Matrice ABCD standard (incidence normale)
    return np.array([[C, 1j*eta*S],
                     [1j*S/eta, C]], dtype=complex)

def total_M(omega, d, sigmas):
    # Produit dans l'ordre: couche 1 côté air -> ... -> côté charge
    M_tot = np.identity(2, dtype=complex)
    for i in range(len(sigmas)):
        M_tot = M_tot @ M_layer(sigmas[i], d, omega)
    return M_tot

def reflection_coeff(omega, d, sigmas, ZL=0.0, Z0=eta_0):
    # ZL=0 => PEC en fond
    M = total_M(omega, d, sigmas)
    A, B, C, D = M[0,0], M[0,1], M[1,0], M[1,1]
    Zin = (A*ZL + B) / (C*ZL + D)
    r   = (Zin - Z0) / (Zin + Z0)
    return r

def calculate_reflectivity(f, d, sigmas, ZL=0.0):
    omega = 2*np.pi*f
    r = reflection_coeff(omega, d, sigmas, ZL=ZL)
    return np.abs(r)**2

# ---------- OPTIM (tu peux garder ta logique) ----------
def minimize_reflectivity(f, d, initial_sigmas, bounds, ZL=0.0):
    def objective(sigmas):
        return calculate_reflectivity(f, d, sigmas, ZL=ZL)
    res = minimize(objective, x0=np.asarray(initial_sigmas),
                   bounds=bounds, method='L-BFGS-B')
    return res

# ============= DEMO =============
# Fréquence test
f = 2e9  # 2 GHz

# Trois épaisseurs (m) si tu veux comparer
d1, d2, d3 = 0.2/3, 0.3/3, 0.4/3

# Sigmas init + bornes (S/m)
initial_sigmas = [1e-4, 1e-4, 1e-4]
bounds = [(1e-8, 1), (1e-8, 1), (1e-8, 1)]

# Optim à 2 GHz pour chaque d (PEC)
res1 = minimize_reflectivity(f, d1, initial_sigmas, bounds)
res2 = minimize_reflectivity(f, d2, initial_sigmas, bounds)
res3 = minimize_reflectivity(f, d3, initial_sigmas, bounds)
print("sigmas* (d1,d2,d3) =", res1.x, res2.x, res3.x)
print("Rmin(dB):", 10*np.log10(res1.fun), 10*np.log10(res2.fun), 10*np.log10(res3.fun))

# Balayage en fréquence
frequencies = np.logspace(8, 11, 600)
def R_dB(f, d, sigmas): return 10*np.log10(calculate_reflectivity(f, d, sigmas))

R1 = np.array([R_dB(ff, d1, res1.x) for ff in frequencies])
R2 = np.array([R_dB(ff, d2, res2.x) for ff in frequencies])
R3 = np.array([R_dB(ff, d3, res3.x) for ff in frequencies])

plt.figure(figsize=(8,4))
plt.plot(frequencies, R1, label=f'd={d1:.3f} m')
plt.plot(frequencies, R2, label=f'd={d2:.3f} m')
plt.plot(frequencies, R3, label=f'd={d3:.3f} m')
plt.xscale("log")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Réflectivité (dB)")
plt.title("Réflectivité vs fréquence (3 couches optimisées à 2 GHz)")
plt.grid(True, which='both', ls='--', lw=0.5)
plt.legend()
plt.show()

# ---------- Surface 3D R(d, sigma3) à f fixé ----------
# ATTENTION : 3D Matplotlib ne supporte pas set_yscale('log').
# On travaille donc en Y = log10(sigma3) pour la surface,
# et on personnalise les ticks pour afficher les valeurs en log.

f_fixed = 2e9
d_vec = np.linspace(0.02, 0.20, 120)           # 120 points (raisonnable)
sigma3_vec = np.logspace(-4, 1, 160)           # 160 points
Y_log = np.log10(sigma3_vec)
D, Y = np.meshgrid(d_vec, Y_log, indexing='xy')

# On fixe sigma1 et sigma2 (par ex. issus d'un des optimums)
sigma1_fixed, sigma2_fixed = res1.x[0], res1.x[1]

Z = np.empty_like(D, dtype=float)
for i in range(D.shape[0]):
    for j in range(D.shape[1]):
        d_val = D[i, j]
        sigma3 = 10**Y[i, j]                   # repasse du log à linéaire
        sigmas = [sigma1_fixed, sigma2_fixed, sigma3]
        Z[i, j] = calculate_reflectivity(f_fixed, d_val, sigmas)

# Surface
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
fig = plt.figure(figsize=(7.5,5.5))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(D, Y, 10*np.log10(Z), cmap='viridis', edgecolor='none', antialiased=True)
ax.set_xlabel('Épaisseur d (m)')
ax.set_ylabel('σ (S/m)')
ax.set_zlabel('R(en dB)')
ax.set_title('Surface 3D de R(d, σ) à f=2 GHz')

# ticks jolis pour l’axe Y (log)
yticks = [-4,-3,-2,-1,0,1]
ax.set_yticks(yticks)
ax.set_yticklabels([f"1e{k}" for k in yticks])

fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12)
plt.show()
