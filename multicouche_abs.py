import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

eps_r=1.4
eps_0=8.854e-12
c=3e8
mu_0=4e-7*np.pi
eta_0=np.sqrt(mu_0/eps_0)

def gamma_i(i,sigmas, omega):
    n_prime=np.sqrt(np.sqrt(eps_r**2 + (sigmas[i]/(omega*eps_0))**2) + eps_r)/np.sqrt(2)
    n_double_prime=-np.sqrt(np.sqrt(eps_r**2 + (sigmas[i]/(omega*eps_0))**2) - eps_r)/np.sqrt(2)
    return (n_prime + 1j*n_double_prime)*omega/c

def M_i(i,d,omega,sigmas):
    n_prime=np.sqrt(np.sqrt(eps_r**2 + (sigmas[i]/(omega*eps_0))**2) + eps_r)/np.sqrt(2)
    n_double_prime=-np.sqrt(np.sqrt(eps_r**2 + (sigmas[i]/(omega*eps_0))**2) - eps_r)/np.sqrt(2)
    M = np.array([[np.cos(gamma_i(i,sigmas,omega)*d), eta_0/(n_prime+1j*n_double_prime) * 1j * np.sin(gamma_i(i,sigmas,omega)*d)],
                  [1j*(n_prime+1j*n_double_prime)/(eta_0) * np.sin(gamma_i(i,sigmas,omega)*d), np.cos(gamma_i(i,sigmas,omega)*d)]])
    return M

def total_M(omega,d,sigmas):
    M_tot = np.identity(2)
    for i in range(len(sigmas)):
        M_tot = np.matmul(M_tot, M_i(i,gamma_i,d,omega,sigmas))
    return M_tot

def reflection_coeff(omega,d,sigmas):
    M_tot = total_M(omega,d,sigmas)
    r = (M_tot[0,0] - eta_0*M_tot[1,0]) / (M_tot[0,0] + eta_0*M_tot[1,0])   
    return r




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
            )

    return result
f = 10**9  # Fréquence de 1 GHz
d = 0.5/3  # Épaisseur de la couche
initial_sigmas = [0.001, 0.001, 0.001]
resultat = minimize_reflectivity(f, d, initial_sigmas, bounds=[(1e-6, 10), (1e-6, 10), (1e-6, 10)])

print("Sigmas optimaux :", resultat.x)
print("Réflectivité minimale :", resultat.fun)
print(calculate_reflectivity(f, d, [10e-6,0.1,0.0001]))
