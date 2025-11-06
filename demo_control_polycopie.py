# -*- coding: utf-8 -*-
import os
from datetime import datetime
import pandas as pd
import numpy as _np

# Python packages
import matplotlib.pyplot
import numpy
import os
from compute_alpha import compute_alpha
from math import pi

# MRG packages
import _env
import preprocessing
import processing
import postprocessing
#import solutions
import sys
numpy.set_printoptions(threshold=sys.maxsize)

def is_in_interest_domain(i, j):
    """
    This function checks if the node (i, j) is in the interest domain (interior + Robin frontier).

    Parameters:
        i: int, index along x-axis.
        j: int, index along y-axis.
    """
    x = i * spacestep
    y = j * spacestep
    if 0.38 <= x <= 0.8 and 0.38 <= y <= 0.8:
        return 1
    else:
        return 0
    
def project(l, chi_trial):
    """
    This function projects chi_trial onto the admissible set.

    Parameters:
        l: int, iteration number (not used in this simple projection).
        chi_trial: Matrix (NxP), trial density to be projected.
    """
    chi_projected = np.clip(chi_trial + l, 0.0, 1.0)
    return chi_projected

def your_optimization_procedure(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                           Alpha, mu, chi, V_obj, mu1, V_0):
    """This function return the optimized density.

    Parameter:
        cf solvehelmholtz's remarks
        Alpha: complex, it corresponds to the absorbtion coefficient;
        mu: float, it is the initial step of the gradient's descent;
        V_obj: float, it characterizes the volume constraint on the density chi;
        mu1: float, it characterizes the importance of the volume constraint on
        the domain (not really important for our case, you can set it up to 0);
        V_0: float, volume constraint on the domain (you can it up to 1).
    """

    k = 0
    (M, N) = numpy.shape(domain_omega)
    numb_iter = 100
    energy = numpy.zeros((numb_iter+1, 1), dtype=numpy.float64)
    while k < numb_iter and mu > 10 ** -5:
        print('---- iteration number = ', k)
        print('1. computing solution of Helmholtz problem, i.e., u')
        alpha_rob = chi.copy() * Alpha
        u=processing.solve_helmholtz(domain_omega, spacestep, omega,
                    f, f_dir, f_neu, f_rob,
                    beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        
        print('2. computing solution of adjoint problem, i.e., p')

        # create fadjoint with ones and a central zone set to 2
        fadjoint = np.ones_like(f, dtype=complex)
        for i,j in numpy.ndindex(fadjoint.shape):
            if is_in_interest_domain(i, j):
                fadjoint[i, j] = 2.0 + 0j
        
        uconj = np.conj(u)
        fadjoint = -2*fadjoint*uconj

        p =processing.solve_helmholtz(domain_omega, spacestep, omega,
                    fadjoint, f_dir, f_neu, f_rob,
                    beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        
        print('3. computing objective function, i.e., energy')
        energy[k] = np.real(compute_objective_function(domain_omega, u, spacestep, mu1, V_0))
        chiopti = chi.copy()
        print('4. computing parametric gradient')
        ene = energy[k]
        while ene >= energy[k] and mu > 10 ** -5:
            l=0
            print('    a. computing gradient descent')
            grad = -1*np.real(Alpha * u * p)
            chi_trial = processing.compute_gradient_descent(chi, grad, domain_omega, mu)

            print('    b. computing projected gradient')
            # project chi onto admissible set: P_ell[chi - mu * grad]
            chi_trial = project(l, chi_trial)

            counter = 0
            while np.abs(np.sum(chi_trial) - V_obj) >= 1 and counter < 10000:
                if np.sum(chi_trial) >= V_obj:
                    l = l-1e-5
                else:
                    l = l+ 1e-5
                chi_trial = project(l, chi_trial)
                counter += 1


            print('    c. computing solution of Helmholtz problem, i.e., u')
            alpha_rob = chi_trial.copy() * Alpha
            u=processing.solve_helmholtz(domain_omega, spacestep, omega,
                    f, f_dir, f_neu, f_rob,
                    beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)

            print('    d. computing objective function, i.e., energy (E)')
            ene = np.real(compute_objective_function(domain_omega, u, spacestep, mu1, V_0))
            
            if  ene < energy[k]:
                print('    e. energy decreased, increasing mu and trying again')
                # The step is increased if the energy decreased
                mu = mu*1.1
            else:
                print('    e. energy increased, decreasing mu and trying again')
                # The step is decreased is the energy increased
                mu = mu / 2
        chi = chi_trial.copy()
        k += 1
        print('Energy at iteration ', k, ': ', ene)
        

    return chiopti, energy, u, grad



def compute_objective_function(domain_omega, u, spacestep, mu1, V_0):
    """
    This function computes the objective function:
    J(u, domain_omega) = \int_{domain_omega} ||u||^2 + mu1 * (Vol(domain_omega) - V_0)

    Parameters:
        domain_omega: Matrix (NxP), defines the domain and the shape of the Robin frontier.
        u: Matrix (NxP), solution of the Helmholtz problem, we are computing its energy.
        spacestep: float, step used to solve the Helmholtz equation.
        mu1: float, constant defining the importance of the volume constraint.
        V_0: float, reference volume.
    """
    # Term 1: ‖u‖²_{L²(Ω)} - Energy over entire domain
    energy_total = numpy.sum(u*numpy.conj(u) * spacestep**2)
    
    # Term 2: ‖1_{Ω_interest}u‖²_{L²(Ω)} - Energy in interest zone
    interest_domain = numpy.array([[is_in_interest_domain(i, j) 
                                    for j in range(N)] 
                                    for i in range(M)])
    energy_interest = numpy.sum(interest_domain * u*numpy.conj(u) * spacestep**2)
    
    # Total objective function
    energy = energy_total + energy_interest
    
    print(f'    Energy : {np.real(energy)}')
    
    return np.real(energy)


import numpy as np

def g(y,omega):
    return np.exp(-y**2)



if __name__ == '__main__':

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- set parameters of the geometry
    N = 150  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 1 # level of the fractal
    spacestep = 1.0 / N  # mesh size

    # -- set parameters of the partial differential equation
    kx = -1.0
    ky = -1.0
    
    wavenumber = numpy.sqrt(kx**2 + ky**2)  # wavenumber
    wavenumber = 5.55874e9*2*pi/(3e8)

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # --- set coefficients of the partial differential equation
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # -- set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- define boundary conditions
    # planar wave defined on top
    f_dir[:, :] = 0.0
    for i in range(N):
        f_dir[0,i] = 5*numpy.exp(-(i-N/2)**2/2)

    # spherical wave defined on top
    #f_dir[:, :] = 0.0
    #f_dir[0, int(N/2)] = 10.0

    # -- initialize
    alpha_rob[:, :] = - wavenumber * 1j

    # -- define material density matrix
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)

    # -- define absorbing material
    Alpha = 10.0 - 10.0 * 1j
    # -- this is the function you have written during your project
    import alphaget
    from compute_alpha import compute_alpha

    materials =  {
        'charged foam': { #mousse de polyuréthane dopée avec 11.2% (en poids) de nanotubes de carbone, une configuration typique pour un absorbant diélectrique
            "eps_r" : 8.65 ,
            "mu_r" : 1.0,
            "sigma" : 0.205 ,
            "c_0" : 3e8 
        },
        'concrete': { 
            "eps_r" : 5.24 ,
            "mu_r" : 1.0,
            "sigma" : 0.00462 ,
            "c_0" : 3e8 
        },
        'ferrite': { #nickel-zinc ferrite
            "eps_r" : 10.5 ,
            "mu_r" : 10,
            "sigma" : 0.05 ,
            "c_0" : 3e8 
        }
    }
    material = materials['charged foam']
    eps_r = material['eps_r']
    mu_r = material['mu_r']
    sigma = material['sigma']
    c_0 = material['c_0']
    omega = wavenumber * c_0


    Alpha = compute_alpha(omega, eps_r, mu_r, sigma, c_0)
    alpha_rob = Alpha * chi

    # -- set parameters for optimization
    S = 0  # surface of the fractal
    for i in range(0, M):
        for j in range(0, N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1
    V_0 = 1  # initial volume of the domain

    beta = 0.5 # volume fraction
    V_obj =  S*beta  # desired volume 
    mu = 1  # initial gradient step
    mu1 = 10**(-5)  # parameter of the volume functional

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # -- compute finite difference solution
    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    chi0 = chi.copy()
    u0 = u.copy()

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- compute optimization
    energy = numpy.zeros((100+1, 1), dtype=numpy.float64)
    chi, energy, u, grad = your_optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                        Alpha, mu, chi, V_obj, mu1, V_0)
    #chi, energy, u, grad = solutions.optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
    #                    beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
    #                    Alpha, mu, chi, V_obj, mu1, V_0)
    # --- en of optimization






    
    # build a binary design by keeping the largest V_obj values of chi
    (M, N) = numpy.shape(domain_omega)
    total_cells = M * N
    n_select = int(max(0, min(total_cells, V_obj)))  # ensure integer and within bounds

    flat_chi = chi.flatten()
    if n_select == 0:
        chi = numpy.zeros_like(chi)
    else:
        # indices of the largest n_select entries
        idx = numpy.argsort(flat_chi)[-n_select:]
        new_flat = numpy.zeros_like(flat_chi)
        new_flat[idx] = 1.0
        chi = new_flat.reshape((M, N))

    # compute forward solution with the binarized chi
    alpha_rob = chi.copy() * Alpha
    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber,
                                   f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)

    # build adjoint right-hand side (same indicator pattern as used previously)
    fadj_indicator = np.ones_like(f, dtype=complex)
    for i, j in numpy.ndindex(fadj_indicator.shape):
        if is_in_interest_domain(i, j):
            fadj_indicator[i, j] = 2.0 + 0j
    fadj = -2 * fadj_indicator * np.conj(u)

    # solve adjoint
    p = processing.solve_helmholtz(domain_omega, spacestep, wavenumber,
                                   fadj, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)

    # gradient and energy for the final design
    grad = -1 * np.real(Alpha * u * p)

  






    chin = chi.copy()
    un = u.copy()

    # trim energy at the first zero entry (if any)
    _energy_flat = numpy.ravel(energy)
    _zero_idx = numpy.where(numpy.isclose(_energy_flat, 0.0))[0]
    if _zero_idx.size:
        energy = energy[:int(_zero_idx[0])].copy()
    print('Chi sum:', numpy.sum(numpy.sum(chin)))
    # -- plot chi, u, and energy
    print(f"Final energy of the $\chi$ projected")
    #energy = np.append(energy,np.real(compute_objective_function(domain_omega, u, spacestep, mu1, V_0)))
    postprocessing._plot_uncontroled_solution(u0, chi0)
    postprocessing._plot_controled_solution(un, chin)
    err = un - u0
    postprocessing._plot_error(err)
    postprocessing._plot_energy_history(energy)

    print('End.')


