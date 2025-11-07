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
        M_tot = np.matmul(M_tot, M_i(i,d,omega,sigmas))
    return M_tot

def reflection_coeff(omega,d,sigmas):
    M_tot = M_i(0,d,omega,sigmas)
    r = (M_tot[0,1] - eta_0*M_tot[1,1]) / (M_tot[0,1] + eta_0*M_tot[1,1])   
    return r




def calculate_reflectivity(f, d, sigmas):
    omega = 2 * np.pi * f
    r = reflection_coeff(omega, d, sigmas)
    R = np.abs(r)**2
    return R

def minimize_reflectivity(f, d, initial_sigmas, bounds):
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
    x0 = np.log10(initial_sigmas)
    log_bounds = [(np.log10(lo), np.log10(hi)) for lo, hi in bounds]
    # Fonction objectif : la réflectivité à minimiser
    def objective(x):
        sigmas = 10**x
        return calculate_reflectivity(f, d, sigmas)

    # Appel à la fonction d'optimisation
    result = minimize(
        objective,
        x0=x0,
        bounds=log_bounds,
        method='L-BFGS-B', 
        options={'ftol': 1e-10, 'gtol': 1e-10}  # méthode adaptée aux bornes et aux problèmes lisses
            )
    result.x = 10**result.x  # Convertir les sigmas optimaux en échelle linéaire
    return result


f = 2e9  # Fréquence de 2 GHz
d1 = 0.17  # Épaisseur de la couche
d2 = 0.4
d3 = 0.5
initial_sigmas = [0.1, 0.1, 0.1]
bounds=[(1e-10, 1), (1e-10, 1), (1e-10, 1)]  # car sigma est compris entre 0 et 1

### on trouve sigma optimal pour une fréquence fixé à 2GHz

resultat1 = minimize_reflectivity(f, d1, initial_sigmas, bounds)
resultat2 = minimize_reflectivity(f, d2, initial_sigmas, bounds)
resultat3 = minimize_reflectivity(f, d3, initial_sigmas, bounds)

print("Sigmas1 optimaux :", resultat1.x)
print("Sigmas2 optimaux :", resultat2.x)
print("Sigmas3 optimaux :", resultat3.x)
print("Réflectivité 1 minimale :", 10*np.log10(resultat1.fun))
print("Réflectivité 2 minimale :", 10*np.log10(resultat2.fun))
print("Réflectivité 3 minimale :", 10*np.log10(resultat3.fun))  
print("Sigmas initiaux :", initial_sigmas)

frequencies = np.logspace(8, 11, 1000)
R1 = []
R2 = []
R3 = []
for f in frequencies:
    R1.append(10*np.log10(calculate_reflectivity(f, d1, resultat1.x)))
    R2.append(10*np.log10(calculate_reflectivity(f, d2, resultat2.x)))
    R3.append(10*np.log10(calculate_reflectivity(f, d3, resultat3.x)))

plt.figure(figsize=(8,4))
plt.plot(frequencies, np.array(R1), label='d1 = ' + str(d1))
plt.plot(frequencies, np.array([-15 for _ in frequencies]), label='-15 dB' , linestyle='--', color='grey')
plt.xscale("log")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Réflectivité (dB)")
plt.title("Réflectivité en fonction de la fréquence")
plt.legend()
plt.grid()

plt.figure(figsize=(8,4))
plt.plot(frequencies, np.array(R2), label='d2 = ' + str(d2))

plt.plot(frequencies, np.array([-15 for _ in frequencies]), label='-15 dB' , linestyle='--', color='grey')
plt.xscale("log")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Réflectivité (dB)")
plt.title("Réflectivité en fonction de la fréquence")
plt.legend()
plt.grid()

plt.figure(figsize=(8,4))
plt.plot(frequencies, np.array(R3), label='d3 = ' + str(d3))
plt.plot(frequencies, np.array([-15 for _ in frequencies]), label='-15 dB' , linestyle='--', color='grey')
plt.xscale("log")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Réflectivité (dB)")
plt.title("Réflectivité en fonction de la fréquence")
plt.legend()
plt.grid()
plt.show()




