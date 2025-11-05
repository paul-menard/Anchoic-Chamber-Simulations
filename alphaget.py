import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.optimize import minimize

#parameters 
l = 1.0          # y ∈ [-l, l]
L = 1
Nx = 50
Ny = 100
dx = 2 * L / Nx


#Parameters of the air
rho0 = 1.2
c0 = 340.0
eta0 = 1.0 / rho0
xi0 = 1.0 / (rho0 * c0**2)


#Parameters for the porous medium

sigma = 0
alphah =0
phi = 0



materials = {
    #sigma = resistivity, alphah = tortuosity, phi = porosity
    "melamine": {"sigma": 14000.0,  "alphah": 1.02, "phi": 0.99},
    "isorel":   {"sigma": 142300.0, "alphah": 1.15, "phi": 0.70},
    "stonewool": {"sigma": 88400,  "alphah": 1.01, "phi": 0.97},
}

medium = "stonewool"

sigma = materials[medium]["sigma"]
alphah = materials[medium]["alphah"]
phi = materials[medium]["phi"]


eta1 = phi/alphah
xi1 = (phi * (7/5))/c0**2
a = (sigma*(7/5)*phi**2)/(alphah*rho0*c0**2)


#ferrite

epsilonr=10+1j*4
sigma = .50
mu0 = 1
eta1= 1
xi1 = epsilonr/c0**2
a = sigma*mu0

A = 1.0
B = 1.0

# spatial grid for y
ygrid = np.linspace(-l, l, Ny)
dy = ygrid[1] - ygrid[0]

#Gives the k grid for the summation over k
k_grid = np.fft.fftshift(np.fft.fftfreq(Ny, d=dy)) * 2.0 * np.pi





# Sources
def g1(y, omega):
    omega0 = 2.0 * np.pi * 300.0
    sigma = 2.0 * np.pi * 200.0
    spec = np.exp(-((omega - omega0)**2) / (2.0 * sigma**2))
    #np.ones_like ensure independance on y
    return spec * np.ones_like(y, dtype=np.complex128)

def g2(y, omega):
    omega0 = 2.0 * np.pi * 1000.0
    sigma = 2.0 * np.pi * 400.0
    spatial = np.sin(np.pi * y / (2.0 * l))
    spec = np.exp(-((omega - omega0)**2) / (2.0 * sigma**2))
    return (spatial * spec).astype(np.complex128)

def g3(y, omega):
    omega0 = 2.0 * np.pi * 1000.0
    sigma = 2.0 * np.pi * 300.0
    spatial = np.exp(-(y**2 / (2.0 * (l**2))))
    spec = np.exp(-((omega - omega0)**2) / (2.0 * sigma**2))
    return (spatial * spec).astype(np.complex128)


sources = {'g1': g1, 'g2': g2, 'g3': g3}


def lambda0_of(k, omega):
    # sqrt(k^2 - (xi0/eta0) * omega^2) with complex sqrt
    return np.sqrt(k**2 - (xi0 / eta0) * (omega**2) + 0j)

def lambda1_of(k, omega):
    Aterm = k**2 - (xi1 / eta1) * (omega**2)
    Bterm = (a * omega / eta1)
    inside = Aterm**2 + (Bterm**2)
    sqrt_inside = np.sqrt(inside + 0j)
    term1 = np.sqrt(Aterm + sqrt_inside + 0j)
    term2 = np.sqrt(-Aterm + sqrt_inside + 0j)
    lam1 = (term1 - 1j * term2) / np.sqrt(2.0)
    return lam1


def ratio(lam0, x):
    """
    Compute the values of :
      ratio_minus = (lam0*eta0 - x)/f(x)
      ratio_plus  = (lam0*eta0 + x)/f(x)
    with f(x) = (lam0*eta0 - x) e^{-lam0 L} + (lam0*eta0 + x) e^{lam0 L}.
    """

    y = (lam0*eta0 - x) - np.exp(-lam0 * L) + (lam0*eta0 + x) * np.exp(lam0 * L)
    ratio_minus = (lam0 * eta0 - x) /y
    ratio_plus = (lam0 * eta0 + x) /y
    return ratio_minus, ratio_plus


# Computation of chi & gamma
def compute_chi_gamma_from_gk(k, alpha, omega, gk_value):
    lam0 = lambda0_of(k, omega)
    lam1 = lambda1_of(k, omega)
    r1_minus, r1_plus = ratio(lam0, lam1 * eta1)
    r2_minus, r2_plus = ratio(lam0, alpha)
    chi = gk_value * (r1_minus - r2_minus)
    gamma = gk_value * (r1_plus - r2_plus)
    return chi, gamma, lam0

# e_k computation
def compute_ek(k, alpha, omega, gk_value):
    chi, gamma, lam0 = compute_chi_gamma_from_gk(k, alpha, omega, gk_value)
    if (k**2) >= (xi0 / eta0) * (omega**2):
        # we decompose all operations to avoid mistakes
        two_lam0_L = 2.0 * lam0 * L
        exp_neg = np.exp(-two_lam0_L)
        one_minus_exp_neg = 1.0 - exp_neg
        exp_pos_minus_one = one_minus_exp_neg / exp_neg
        common = (np.abs(chi)**2) * one_minus_exp_neg + (np.abs(gamma)**2) * exp_pos_minus_one
        term1 = (A + B * (k**2)) * ((1.0 / (2.0 * lam0)) * common + 2.0 * L * np.real(chi * np.conj(gamma)))
        term2 = (B / (lam0**2)) * common
        term3 = -2.0 * B * (lam0**2) * L * np.real(chi * np.conj(gamma))
        ek = np.real(term1 + term2 + term3)
    else:
        part_im = np.imag(chi * np.conj(gamma) * (1.0 - np.exp(-2.0 * lam0 * L)))
        term1 = (A + B * (k**2)) * (L * (np.abs(chi)**2 + np.abs(gamma)**2) + (1j / (lam0)) * part_im)
        term2 = B * L * (np.abs(lam0)**2) * (np.abs(chi)**2 + np.abs(gamma)**2)
        term3 = 1j * B * lam0 * part_im
        ek = np.real(term1 + term2 + term3)
    return float(ek)



def compute_gk_from_gy_omega(g_y_omega):
    a = np.fft.fft(g_y_omega) / Ny
    gk = np.fft.fftshift(a)
    return k_grid, gk

# compute E by summing (using gk and k_grid)
def compute_E(omega, g_func, alpha):
    gy = g_func(ygrid, omega)
    kgrid, gk = compute_gk_from_gy_omega(gy)
    E = 0.0
    for k_val, gk_val in zip(kgrid, gk):
        E += compute_ek(k_val, alpha, omega, gk_val)
    return float(E)


def find_alpha_for_frequency(omega, g_func):

    lam1_init = lambda1_of(0.0, omega)
    alpha_init = lam1_init * eta1

    def objective(x):
        alpha_candidate = x[0] + 1j * x[1]
        return compute_E(omega, g_func, alpha_candidate)

    x0 = [alpha_init.real, alpha_init.imag]

    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 10000})
    
    alpha_opt = result.x[0] + 1j * result.x[1]
    e_opt = result.fun
    
    return alpha_opt, e_opt
