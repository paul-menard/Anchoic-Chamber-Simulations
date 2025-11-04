import numpy as np
import matplotlib.pyplot as plt

eps_r=1.4
eps_0=8.854e-12
c=3e8
mu_0=4e-7*np.pi
eta_0=np.sqrt(mu_0/eps_0)

def gamma_i(i,sigmas, omega):
    n_prime=np.sqrt(np.sqrt(eps_r**2 + (sigmas[i]/(omega*eps_0))**2) + eps_r)/np.sqrt(2)
    n_double_prime=-np.sqrt(np.sqrt(eps_r**2 + (sigmas[i]/(omega*eps_0))**2) - eps_r)/np.sqrt(2)
    return (n_prime + 1j*n_double_prime)*omega/c

def M_i(i,gamma_i,d,omega,sigmas):
    M = np.array([[np.cos(gamma_i(i,sigmas,omega)*d), eta_0/ (1j*eps_r*omega)*np.sin(gamma_i(i,sigmas,omega)*d)],
                  [1j*eps_r*omega/eta_0*np.sin(gamma_i(i,sigmas,omega)*d), np.cos(gamma_i(i,sigmas,omega)*d)]])
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

def calculate_reflectivity(frequencies,d,sigmas):
    reflectivities = []
    for f in frequencies:
        omega = 2*np.pi*f
        r = reflection_coeff(omega,d,sigmas)
        R = np.abs(r)**2
        reflectivities.append(R)
    return np.array(reflectivities)


# Minimum Reflectivity for sigma values (only parameter varying is sigmas)_optimization(frequencies,d,sigma_ranges,num_layers):
    
