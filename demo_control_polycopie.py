# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi

# Custom modules
import alphaget
from compute_alpha import compute_alpha
from preprocessing import set2zero
import _env
import preprocessing
import processing
import postprocessing

np.set_printoptions(threshold=sys.maxsize)


def is_in_interest_domain(node):
    if node < 0:
        return 1
    if node == 3:
        return 2
    return 0


def project(chi, domain, V_obj):
    """
    This function performs the projection of χⁿ - μ∇.

    To perform the optimization, we use a projected gradient algorithm.
    This function characterizes the projection of chi onto the admissible space
    (the space of L∞ functions whose volume is equal to V_obj and whose
    values are between 0 and 1).
    """
    M, N = np.shape(domain)
    S = np.sum(domain == _env.NODE_ROBIN)

    if S == 0:
        print("Erreur (projected_chi): Pas de frontière de Robin (S=0).")
        return chi

    B = chi.copy()
    max_abs_B = np.max(np.abs(B)) + 1.0

    debut = -max_abs_B
    fin = max_abs_B

    chi_proj = np.zeros_like(B)
    ecart = fin - debut

    while ecart > 1e-4:
        l = (debut + fin) / 2
        for i in range(M):
            for j in range(N):
                chi_proj[i, j] = np.maximum(0, np.minimum(B[i, j] + l, 1))

        chi_proj = set2zero(chi_proj, domain)
        V = np.sum(chi_proj) / S

        if V > V_obj:
            fin = l
        else:
            debut = l

        ecart = fin - debut

    l = (debut + fin) / 2
    for i in range(M):
        for j in range(N):
            chi_proj[i, j] = np.maximum(0, np.minimum(B[i, j] + l, 1))

    chi_proj = set2zero(chi_proj, domain)
    return chi_proj


def compute_gradient_descent(chi, grad, domain, mu):
    M, N = np.shape(domain)

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            a = is_in_interest_domain(domain[i + 1, j])
            b = is_in_interest_domain(domain[i - 1, j])
            c = is_in_interest_domain(domain[i, j + 1])
            d = is_in_interest_domain(domain[i, j - 1])

            if a == 2:
                chi[i + 1, j] -= mu * grad[i, j]
            if b == 2:
                chi[i - 1, j] -= mu * grad[i, j]
            if c == 2:
                chi[i, j + 1] -= mu * grad[i, j]
            if d == 2:
                chi[i, j - 1] -= mu * grad[i, j]

    return chi


def your_optimization_procedure(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,
                                alpha_rob, Alpha, mu, chi, V_obj, mu1, V_0):
    """
    Return the optimized density.
    """

    k = 0
    M, N = np.shape(domain_omega)
    numb_iter = 10
    energy = np.zeros((numb_iter + 1, 1), dtype=np.float64)
    grad = np.zeros((M, N), dtype=float)
    chiopti = chi.copy()

    i_start, i_end = int(142 * 10 / 10), int(184 * 10 / 10)
    j_start, j_end = int(30 * 10 / 10), int(70 * 10 / 10)

    quiet_zone_mask = np.zeros((M, N), dtype=np.float64)
    quiet_zone_mask[i_start:i_end, j_start:j_end] = 1.0

    alpha_rob = chi.copy() * Alpha
    u = processing.solve_helmholtz(domain_omega, spacestep, omega,
                                   f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir,
                                   beta_neu, beta_rob, alpha_rob)

    current_energy = np.real(compute_objective_function(domain_omega, u, spacestep,
                                                        mu1, V_0, quiet_zone_mask))
    energy[0] = current_energy

    while k < numb_iter and mu > 1e-5:
        print(f"---- iteration number = {k}")
        print("1. computing solution of Helmholtz problem, i.e., u")
        print("2. computing solution of adjoint problem, i.e., p")

        f_adj = -2 * u * (1.0 + quiet_zone_mask)

        p = processing.solve_helmholtz(domain_omega, spacestep, omega,
                                       f_adj, f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir,
                                       beta_neu, beta_rob, alpha_rob)

        print("3. computing objective function, i.e., energy")
        print("4. computing parametric gradient")

        grad = -1 * np.real(Alpha * u * p)

        chi_current_iter = chi.copy()
        energy_current_iter = current_energy
        good_step = False

        while mu > 1e-5:
            chi_new_test = compute_gradient_descent(chi_current_iter.copy(), grad, domain_omega, mu)
            chi_proj_test = project(chi_new_test, domain_omega, V_obj)
            current_alpha_rob_test = Alpha * chi_proj_test

            u_test = processing.solve_helmholtz(domain_omega, spacestep, omega,
                                                f, f_dir, f_neu, f_rob,
                                                beta_pde, alpha_pde, alpha_dir,
                                                beta_neu, beta_rob, current_alpha_rob_test)

            ene_test = compute_objective_function(domain_omega, u_test, spacestep, mu1, V_0, quiet_zone_mask)

            if ene_test < energy_current_iter + 1e-7:
                print("    -> Succès : Énergie diminuée. Étape acceptée.")
                chi = chi_proj_test
                u = u_test
                current_energy = ene_test
                current_alpha_rob = current_alpha_rob_test
                mu *= 1.1
                good_step = True
                break
            else:
                print("    -> Échec : Énergie augmentée. Réduction du pas.")
                mu /= 2

        if not good_step:
            print("Recherche échouée (mu trop petit). Arrêt de l'optimisation.")
            break

        energy[k + 1] = current_energy
        k += 1

    energy = energy[0:k + 1]

    # Final binarization of chi
    robin_mask = (domain_omega == _env.NODE_ROBIN)
    chi_robin = chi[robin_mask]
    S = len(chi_robin)

    if S > 0:
        num_ones = int(round(V_obj * S))
        num_ones = max(0, min(num_ones, S))

        sorted_indices = np.argsort(-chi_robin)
        chi_binaire = np.zeros_like(chi_robin)
        chi_binaire[sorted_indices[:num_ones]] = 1.0

        chi_final = chi.copy()
        chi_final[robin_mask] = chi_binaire
        chi = chi_final

        print(f"Binarisation effectuée : {num_ones}/{S} points à 1 "
              f"(volume ≈ {num_ones / S if S > 0 else 0:.2f}).")
    else:
        print("Aucun point Robin détecté pour la binarisation.")

    return chi, energy, u, grad


def compute_objective_function(domain_omega, u, spacestep, mu1, V_0, quiet_zone):
    """
    Compute objective function:
    J(u, domain_omega) = ∫ ||u||² + μ₁ (Vol(domain_omega) - V₀)
    """
    M, N = np.shape(domain_omega)
    energy_domain = np.sum((np.abs(u) ** 2) * (domain_omega == _env.NODE_INTERIOR))
    energy_quiet = np.sum((np.abs(u) ** 2) * quiet_zone)
    energy = (energy_domain + energy_quiet) * spacestep ** 2

    print(f"    Energy : {np.real(energy)}")
    return energy


def g(y, omega):
    return np.exp(-y ** 2)


if __name__ == "__main__":
    N = 150
    M = 2 * N
    level = 2
    spacestep = 1.0 / N

    kx, ky = -1.0, -1.0
    wavenumber = 5.55874e9 * 2 * np.pi / (3e8)

    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    f_dir[:, :] = 0.0
    for i in range(N):
        f_dir[0, i] = 5 * np.exp(-(i - N / 2) ** 2 / 2)

    alpha_rob[:, :] = -wavenumber * 1j
    V_0 = 1
    V_obj = 0.2

    chi = np.zeros((M, N), dtype=np.float64)
    robin_indices = np.where(domain_omega == _env.NODE_ROBIN)
    S = len(robin_indices[0])

    if S == 0:
        print("Aucun point sur la frontière de Robin (mur) n'a été trouvé.")
        S = 1
    else:
        print(f"Nombre total de points sur le mur (S): {S}")

    num_ones = int(V_obj * S)
    print(f"Nombre de points à 1 (0.8 * S): {num_ones}")

    shuffled_indices = np.arange(S)
    np.random.shuffle(shuffled_indices)
    indices_to_set = shuffled_indices[:num_ones]
    i_coords, j_coords = robin_indices[0][indices_to_set], robin_indices[1][indices_to_set]
    chi[i_coords, j_coords] = 1.0
    vol_check = np.sum(chi) / S
    print(f"Volume initial de chi0 vérifié : {vol_check:.4f}")

    Alpha = 10 - 1j * 10
    alpha_rob = Alpha * chi

    mu = 1e-3
    mu1 = 1e-5

    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber,
                                   f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir,
                                   beta_neu, beta_rob, alpha_rob)

    chi0, u0 = chi.copy(), u.copy()

    chi, energy, u, grad = your_optimization_procedure(domain_omega, spacestep, wavenumber,
                                                       f, f_dir, f_neu, f_rob,
                                                       beta_pde, alpha_pde, alpha_dir,
                                                       beta_neu, beta_rob, alpha_rob,
                                                       Alpha, mu, chi, V_obj, mu1, V_0)

    chin, un = chi.copy(), u.copy()
    err = un - u0

    postprocessing._plot_uncontroled_solution(u0, chi0)
    postprocessing._plot_controled_solution(un, chin)
    postprocessing._plot_error(err)
    postprocessing._plot_energy_history(energy)

    print("End.")
