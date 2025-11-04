import numpy as np
import matplotlib.pyplot as plt
<<<<<<< HEAD
import scipy.optimize as opt
=======
>>>>>>> alexandre
from scipy.optimize import minimize

eps_r=1.4
eps_0=8.854e-12
c=3e8
mu_0=4e-7*np.pi
eta_0=np.sqrt(mu_0/eps_0)

def gamma(sigma, omega):
    n_prime=np.sqrt(np.sqrt(eps_r**2 + (sigma/(omega*eps_0))**2) + eps_r)/np.sqrt(2)
    n_double_prime=-np.sqrt(np.sqrt(eps_r**2 + (sigma/(omega*eps_0))**2) - eps_r)/np.sqrt(2)
    return (n_prime + 1j*n_double_prime)*omega/c

def M_i(d,omega,sigma):
    M = np.array([[np.cos(gamma(sigma,omega)*d), eta_0/ (1j*eps_r*omega)*np.sin(gamma(sigma,omega)*d)],
                  [1j*eps_r*omega/eta_0*np.sin(gamma(sigma,omega)*d), np.cos(gamma(sigma,omega)*d)]])
    return M

def total_M(omega,d,sigma,n_layers):
    M_tot = np.identity(2)
    for i in range(n_layers):
        M_tot = np.matmul(M_tot, M_i(d,omega,sigma[i]))
    return M_tot

def reflection_coeff(omega,d,sigma):
    M_tot = M_i(d,omega,sigma)
    r = (M_tot[0,0] - eta_0*M_tot[1,0]) / (M_tot[0,0] + eta_0*M_tot[1,0])   
    return r

<<<<<<< HEAD
def calculate_reflectivity(f,d,sigma):
    omega = 2*np.pi*f
    r = reflection_coeff(omega,d,sigma)
    R = np.abs(r)**2
    return R


# Minimum Reflectivity for sigma values (only parameter varying is sigmas)_optimization(frequencies,d,sigma_ranges,num_layers):
    
f0 = 1e9                     # fréquence fixe (1 GHz)
n_layers = 1               # nombre de couches
d = 0.5/3                   # épaisseur moyenne 20 cm par couche

Sigmas=np.linspace(1e-4, 1e4, 10000)  # plage de conductivités à tester
R_values = []
for s in Sigmas:
    gamma_0 = gamma(s, 2*np.pi*f0)
    M= M_i(d, 2*np.pi*f0, s)
    r = reflection_coeff(2*np.pi*f0, d, s)
    R = np.abs(r)**2
    R_values.append(R)

R_values = np.array(R_values)
plt.plot(Sigmas, R_values)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Conductivité (S/m)')
plt.ylabel('Réflectivité R')
plt.title('Réflectivité en fonction de la conductivité pour une couche à 1 GHz')
plt.grid(True)
plt.show()

print("Minimum Reflectivity R_min:", np.min(R_values))
print("Conductivité correspondante σ:", Sigmas[np.argmin(R_values)])
=======
def calculate_reflectivity(f,d,sigmas):
    reflectivities = []
    
    omega = 2*np.pi*f
    r = reflection_coeff(omega,d,sigmas)
    R = np.abs(r)**2
    return R


def calculate_reflectivity(f, d, sigmas):
    omega = 2 * np.pi * f
    r = reflection_coeff(omega, d, sigmas)
    R = np.abs(r)**2
    return R

def minimize_reflectivity(f, d, initial_sigmas, bounds=None):
    """
    Minimise la réflectivité en fonction des conductivités.

    Paramètres
    ----------
    f : float
        Fréquence.
    d : float
        Épaisseur de la couche.
    initial_sigmas : list de taille 3
        Valeurs initiales pour la conductivité des 3 couches.
    bounds : list de tuples, optionnel
        Bornes pour chaque sigma, ex: [(1e-6, 1), (1e-6, 1), (1e-6, 1)]

    Retour
    ------
    result : OptimizeResult
        Résultat de l’optimisation (avec .x pour les sigmas optimaux et .fun pour la réflectivité minimale).
    """

    # Fonction objectif : la réflectivité à minimiser
    def objective(sigmas):
        return calculate_reflectivity(f, d, sigmas)

    # Appel à la fonction d'optimisation
    result = minimize(
        objective,
        x0=np.array(initial_sigmas),
        bounds=bounds,
        method='L-BFGS-B',  # méthode adaptée aux bornes et aux problèmes lisses
        options={'disp': True}
    )

    return result

print(minimize_reflectivity(10**9, 0.5/3, [0.01, 0.01, 0.01], bounds=[(1e-6, 10), (1e-6, 1), (1e-6, 1)]))
>>>>>>> alexandre
