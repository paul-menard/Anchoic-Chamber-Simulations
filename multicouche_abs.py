import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
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