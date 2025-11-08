# -*- coding: utf-8 -*-

# Python packages
import matplotlib.pyplot as plt
import numpy
import numpy as np
import os
from math import pi
from compute_alpha import compute_alpha

# MRG packages
import _env
import preprocessing
import processing
import postprocessing

def is_in_interest_domain(i, j, spacestep):
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
            if is_in_interest_domain(i, j, spacestep):
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
    N, M = domain_omega.shape
    energy_total = numpy.sum(u*numpy.conj(u) * spacestep**2)
    
    # Term 2: ‖1_{Ω_interest}u‖²_{L²(Ω)} - Energy in interest zone
    interest_domain = numpy.array([[is_in_interest_domain(i, j, spacestep) 
                                    for j in range(M)] 
                                    for i in range(N)])
    energy_interest = numpy.sum(interest_domain * u*numpy.conj(u) * spacestep**2)
    
    # Total objective function
    energy = energy_total + energy_interest
    
    print(f'    Energy : {np.real(energy)}')
    
    return np.real(energy)


def g(y, omega):
    return np.exp(-y**2)


def energy_for_frequency(freq, domain_omega, spacestep,
                         f, f_dir, f_neu, f_rob,
                         beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,
                         chi, material_params, mu1=0, V_0=0):
    """
    Compute the energy for a given frequency.
    
    Parameters:
        freq: float, frequency in Hz
        domain_omega: domain matrix
        spacestep: spatial step
        f, f_dir, f_neu, f_rob: right-hand sides
        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob: PDE coefficients
        chi: material density matrix
        material_params: dict with 'eps_r', 'mu_r', 'sigma', 'c_0'
        mu1, V_0: volume constraint parameters
    
    Returns:
        energy: float, computed energy
    """
    # Convert frequency to wavenumber
    k = 2 * np.pi * freq / material_params['c_0']
    
    # Compute Alpha for this frequency
    omega = 2 * np.pi * freq
    Alpha = compute_alpha(omega, material_params['eps_r'], 
                         material_params['mu_r'], 
                         material_params['sigma'], 
                         material_params['c_0'])
    
    # Compute alpha_rob
    alpha_rob = chi.copy() * Alpha
    
    # Solve Helmholtz equation
    u = processing.solve_helmholtz(domain_omega, spacestep, k,
                                   f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, 
                                   beta_neu, beta_rob, alpha_rob)
    
    # Compute energy
    ene = compute_objective_function(domain_omega, u, spacestep, mu1, V_0)
    
    return ene


def find_max_energy_frequency(freq_start, freq_end, domain_omega, spacestep,
                               f, f_dir, f_neu, f_rob,
                               beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,
                               chi, material_params, mu1=0, V_0=0, 
                               method='bounded', n_samples=20):
    """
    Find the frequency with maximum energy in the given interval.
    
    Parameters:
        freq_start, freq_end: float, frequency range in Hz
        domain_omega, spacestep: domain parameters
        f, f_dir, f_neu, f_rob: right-hand sides
        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob: PDE coefficients
        chi: material density matrix
        material_params: dict with material properties
        mu1, V_0: volume constraint parameters
        method: str, optimization method ('bounded' or 'grid')
        n_samples: int, number of grid samples for initial search
    
    Returns:
        optimal_freq: float, frequency with maximum energy
        max_energy: float, maximum energy value
    """
    import scipy.optimize
    
    # Define negative energy function for minimization
    def neg_energy(freq):
        return -energy_for_frequency(freq, domain_omega, spacestep,
                                     f, f_dir, f_neu, f_rob,
                                     beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,
                                     chi, material_params, mu1, V_0)
    
    if method == 'grid':
        # Grid search for global maximum
        freq_samples = np.linspace(freq_start, freq_end, n_samples)
        energies = []
        
        print(f"Searching for maximum energy in [{freq_start:.3e}, {freq_end:.3e}] Hz...")
        for i, freq in enumerate(freq_samples):
            print(f"  Sample {i+1}/{n_samples}: freq = {freq:.6e} Hz", end='\r')
            ene = energy_for_frequency(freq, domain_omega, spacestep,
                                       f, f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,
                                       chi, material_params, mu1, V_0)
            energies.append(ene)
        
        print()  # New line after progress
        max_idx = np.argmax(energies)
        optimal_freq = freq_samples[max_idx]
        max_energy = energies[max_idx]
        
    else:  # bounded optimization
        print(f"Optimizing to find maximum energy in [{freq_start:.3e}, {freq_end:.3e}] Hz...")
        result = scipy.optimize.minimize_scalar(neg_energy, 
                                                bounds=(freq_start, freq_end), 
                                                method='bounded')
        optimal_freq = result.x
        max_energy = -result.fun
    
    print(f"Maximum energy {max_energy:.6e} found at frequency {optimal_freq:.6e} Hz")
    
    return optimal_freq, max_energy


def plot_energy_around_optimal(freq_start, freq_end, n_points=100, 
                                window_ratio=0.1, method='bounded'):
    """
    Find the optimal frequency in [freq_start, freq_end] and plot energy around it.
    
    Parameters:
        freq_start, freq_end: float, initial search range in Hz
        n_points: int, number of points to plot
        window_ratio: float, ratio of the search range to use as plot window
        method: str, 'bounded' for scipy optimization or 'grid' for grid search
    """
    # Setup problem parameters
    N = 100
    M = 2 * N
    level = 2
    spacestep = 1.0 / N
    
    # Set coefficients
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = \
        preprocessing._set_coefficients_of_pde(M, N)
    
    # Set right hand sides
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
    
    # Set geometry
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)
    
    # Define boundary conditions (planar wave)
    f_dir[:, :] = 0.0
    for i in range(N):
        f_dir[0, i] = 5 * numpy.exp(-(i - N/2)**2 / 2)
    
    # Initialize material density
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)
    
    # Material parameters
    materials = {
        'charged foam': {
            "eps_r": 8.65,
            "mu_r": 1.0,
            "sigma": 0.205,
            "c_0": 3e8
        },
        'concrete': {
            "eps_r": 5.24,
            "mu_r": 1.0,
            "sigma": 0.00462,
            "c_0": 3e8
        },
        'ferrite': {
            "eps_r": 10.5,
            "mu_r": 10,
            "sigma": 0.05,
            "c_0": 3e8
        }
    }
    
    material_params = materials['charged foam']
    
    # Find optimal frequency
    optimal_freq, max_energy = find_max_energy_frequency(
        freq_start, freq_end, domain_omega, spacestep,
        f, f_dir, f_neu, f_rob,
        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,
        chi, material_params, method=method
    )
    
    # Define plot window around optimal frequency
    freq_range = freq_end - freq_start
    plot_window = freq_range * window_ratio
    plot_start = max(freq_start, optimal_freq - plot_window / 2)
    plot_end = min(freq_end, optimal_freq + plot_window / 2)
    
    # Compute energies in plot window
    frequencies = np.linspace(plot_start, plot_end, n_points)
    energies = []
    
    print(f"\nComputing energy profile around optimal frequency...")
    for i, freq in enumerate(frequencies):
        print(f"  Point {i+1}/{n_points}", end='\r')
        ene = energy_for_frequency(freq, domain_omega, spacestep,
                                   f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,
                                   chi, material_params)
        energies.append(ene)
    print()  # New line
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, energies, 'x', linewidth=2, label='Energy')
    plt.axvline(optimal_freq, color='r', linestyle='--', linewidth=2, 
                label=f'Optimal freq = {optimal_freq:.6e} Hz')
    plt.scatter([optimal_freq], [max_energy], color='r', s=100, zorder=5)
    
    plt.title(f"Energy vs Frequency (around optimal)\nSearch range: [{freq_start:.3e}, {freq_end:.3e}] Hz", 
              fontsize=14)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Energy', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
    
    return optimal_freq, max_energy

def plot_energy(start, end):
    N = 100  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 3  # level of the fractal
    spacestep = 1.0 / N  # mesh size

    # -- set parameters of the partial differential equation
    kx = -1.0
    ky = -1.0
    
    frequencies = numpy.linspace(start, end, num=100)
    wavenumber = numpy.sqrt(kx**2 + ky**2)  # wavenumber
    wavenumbers = 2*np.pi*frequencies/(3e8)

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
        f_dir[0,i] = numpy.exp(-((i-N/2)*spacestep)**2/0.02)

    # spherical wave defined on top
    #f_dir[:, :] = 0.0
    #f_dir[0, int(N/2)] = 10.0

    # -- initialize
    alpha_rob[:, :] = - wavenumber * 1j

    # -- define material density matrix
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)

    """
    for i in range(M):
        for j in range(N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                chi[i, j] = 1
    """

    # -- define absorbing material
    Alpha = 10.0 - 10.0 * 1j
    # -- this is the function you have written during your project
    import alphaget
    #Alpha = alphaget.find_alpha_for_frequency(wavenumber*3e8, g)[0]
    materials = {
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
    eps_r, mu_r, sigma,  c_0 = materials['charged foam'].values()
    omega = wavenumber * c_0
    Alpha = compute_alpha(omega, eps_r, mu_r, sigma, c_0)
    alpha_rob = Alpha * chi

    S = 0  # surface of the fractal
    for i in range(0, M):
        for j in range(0, N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1
    V_0 = 1  # initial volume of the domain
    #V_obj = numpy.sum(numpy.sum(chi)) / S  # constraint on the density
    V_obj =  200  # desired volume fraction
    mu = 1  # initial gradient step
    mu1 = 10**(-5)  # parameter of the volume functional

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # -- compute finite difference solution
    energies = []
    for k in wavenumbers:
        u = processing.solve_helmholtz(domain_omega, spacestep, k, f, f_dir, f_neu, f_rob,
                            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        chi0 = chi.copy()
        u0 = u.copy()
        ene = compute_objective_function(domain_omega, u, spacestep, mu1, V_0)
        energies.append(ene)

    # Find the maximum energy and its corresponding frequency
    max_energy = max(energies)
    max_index = energies.index(max_energy)
    max_frequency = frequencies[max_index]

    # Plot the energy vs frequency
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, energies, 'x', label='Energy')

    # Highlight the maximum energy point with a vertical line
    plt.axvline(x=max_frequency, color='red', linestyle='--', label=f'Max Energy: {max_energy:.2f} at {max_frequency:.2e} Hz')

    plt.title(f"Energy vs Frequency for fractal level {level}\nMax Energy at {max_frequency:.4e} Hz", fontsize=14)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Energy')
    plt.grid(which='both', linestyle='-', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plot_abs_heatmap(u)
    postprocessing._plot_uncontroled_solution(u, chi)
    return



def plot_solution(N, freq, L = 1):
    N = 100  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 0  # level of the fractal
    spacestep = 1.0 / N  # mesh size

    # -- set parameters of the partial differential equation
    kx = -1.0
    ky = -1.0
    
    wavenumber = numpy.sqrt(kx**2 + ky**2)  # wavenumber
    wavenumbers = 2*np.pi*freq/(3e8)

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
        f_dir[0,i] = numpy.exp(-((i-N/2)*spacestep)**2/0.02)

    # spherical wave defined on top
    #f_dir[:, :] = 0.0
    #f_dir[0, int(N/2)] = 10.0

    # -- initialize
    alpha_rob[:, :] = - wavenumber * 1j

    # -- define material density matrix
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)

    for i in range(M):
        for j in range(N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                chi[i, j] = 1

    # -- define absorbing material
    Alpha = 10.0 - 10.0 * 1j
    # -- this is the function you have written during your project

    #Alpha = alphaget.find_alpha_for_frequency(wavenumber*3e8, g)[0]
    materials = {
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
    eps_r, mu_r, sigma,  c_0 = materials['charged foam'].values()
    omega = wavenumber * c_0
    Alpha = compute_alpha(omega, eps_r, mu_r, sigma, c_0)
    alpha_rob = Alpha * chi

    S = 0  # surface of the fractal
    for i in range(0, M):
        for j in range(0, N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1
    V_0 = 1  # initial volume of the domain
    #V_obj = numpy.sum(numpy.sum(chi)) / S  # constraint on the density
    V_obj =  200  # desired volume fraction
    mu = 1  # initial gradient step
    mu1 = 10**(-5)  # parameter of the volume functional
    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber,
                                   f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, 
                                   beta_neu, beta_rob, alpha_rob)
    plot_abs_heatmap(u)
    return

def plot_abs_heatmap(u):
    import matplotlib.pyplot as plt
    plt.imshow(numpy.abs(u), cmap='jet', interpolation='nearest')
    plt.colorbar()
    plt.title('Heatmap of |u|')
    plt.show()

    return

def compare_chi_random_to_everwhere(start, end, n_points=100):
    N = 100  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 2  # level of the fractal
    spacestep = 1.0 / N  # mesh size

    # -- set parameters of the partial differential equation
    kx = -1.0
    ky = -1.0
    frequencies = numpy.linspace(start, end, num=n_points)
    wavenumbers = 2*np.pi*frequencies/(3e8)

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # --- set coefficients of the partial differential equation
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # -- set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    materials = {
        'charged foam': { #mousse de polyuréthane dopée avec 11.2% (en poids) de nanotubes de carbone, une configuration typique pour un absorbant diélectrique
            "eps_r" : 8.65 ,
            "mu_r" : 1.0,
            "sigma" : 0.205 ,
            "c_0" : 3e8 
        }
    }
    eps_r, mu_r, sigma,  c_0 = materials['charged foam'].values()
    omega = wavenumbers * c_0
    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- define boundary conditions
    # planar wave defined on top
    f_dir[:, :] = 0.0
    for i in range(N):
        f_dir[0,i] = numpy.exp(-((i-N/2)*spacestep)**2/0.02)

    # spherical wave defined on top
    #f_dir[:, :] = 0.0
    #f_dir[0, int(N/2)] = 10.0

    # -- initialize

    S = 0  # surface of the fractal
    for i in range(M):
        for j in range(N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1
    
    V_0 = 1  # initial volume of the domain
    V_obj = 0.5  # desired volume fraction (adjust as needed)
    mu = 1.0  # initial gradient step
    mu1 = 10**(-5)  # parameter of the volume functional
    
    print(f"Robin boundary points: {S}")
    print(f"Target volume constraint: {V_obj}")

    num_ones = int(V_obj * S)
    print(f"Nombre de points à 1 (0.8 * S): {num_ones}")

    # -- define material density matrix
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)
    chi0 = chi.copy()
    
    chi = np.zeros((M, N), dtype=np.float64)
    robin_indices = np.where(domain_omega == _env.NODE_ROBIN)
    S = len(robin_indices[0])
    shuffled_indices = np.arange(S)
    np.random.shuffle(shuffled_indices)
    indices_to_set = shuffled_indices[:num_ones]
    i_coords, j_coords = robin_indices[0][indices_to_set], robin_indices[1][indices_to_set]
    chi[i_coords, j_coords] = 1.0
    vol_check = np.sum(chi) / S
    print(f"Volume initial de chi0 vérifié : {vol_check:.4f}")
    
    
    chi_everywhere = np.array([[1 if domain_omega[i, j] == _env.NODE_ROBIN else 0
                               for j in range(N)] 
                              for i in range(M)])

    random_energies = []
    everywhere_energies = []
    chi0_energies = []

    for k in range(len(wavenumbers)):
        print(f'k = {k}') 
        omega = 2 * np.pi * frequencies[k]
        k = wavenumbers[k]
        Alpha = compute_alpha(omega, eps_r, mu_r, sigma, c_0)
        alpha_rob = Alpha * chi

        u_random = processing.solve_helmholtz(domain_omega, spacestep, k,
                            f, f_dir, f_neu, f_rob,
                            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        ene_random = compute_objective_function(domain_omega, u_random, spacestep, mu1=0, V_0=0)
        random_energies.append(ene_random)

        alpha_0 = Alpha*chi0

        u_0 = processing.solve_helmholtz(domain_omega, spacestep, k,
                            f, f_dir, f_neu, f_rob,
                            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_0)
        ene_0 = compute_objective_function(domain_omega, u_0, spacestep, mu1=0, V_0=0)
        chi0_energies.append(ene_0)

        alpha_rob_everywhere = Alpha * chi_everywhere

        u_everywhere = processing.solve_helmholtz(domain_omega, spacestep, k,
                            f, f_dir, f_neu, f_rob,
                            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob_everywhere)
        ene_everywhere = compute_objective_function(domain_omega, u_everywhere, spacestep, mu1=0, V_0=0)
        everywhere_energies.append(ene_everywhere)
    print('Plotting comparison of energies...')
    postprocessing._plot_uncontroled_solution(u_random, chi)

     # Plot the energy comparison
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, random_energies, 'r--', label='Random chi')
    plt.plot(frequencies, everywhere_energies, 'b-', label='Chi = 1')
    plt.plot(frequencies, chi0_energies, '-x', label = "chi_0 ", color = "green")
    plt.title("Energy Comparison for Different Chi Configurations")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid()
    plt.show()
    postprocessing._plot_uncontroled_solution(u_everywhere, chi_everywhere)

        
def optimize_local_maxima(start, end, n_samples=20):
    """
    This function finds the frequency with maximum energy in the given interval,
    then optimizes the material density chi using gradient descent.
    
    Parameters:
        start: float, start frequency in Hz
        end: float, end frequency in Hz
        n_samples: int, number of samples for finding the maximum
    
    Returns:
        chi_optimal: optimized material density matrix
        optimal_freq: frequency with maximum energy
        max_energy: maximum energy value
        energy_history: history of energies during optimization
    """
    N = 100  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 0  # level of the fractal
    spacestep = 1.0 / N  # mesh size

    print("="*60)
    print("STEP 1: Finding frequency with maximum energy")
    print("="*60)
    
    # --- set coefficients of the partial differential equation
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # -- set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    # -- define boundary conditions (planar wave)
    f_dir[:, :] = 0.0
    for i in range(N):
        f_dir[0, i] = numpy.exp(-((i-N/2)*spacestep)**2/0.02)

    # -- define initial material density matrix
    chi_initial = preprocessing._set_chi(M, N, x, y)
    chi_initial = preprocessing.set2zero(chi_initial, domain_omega)

    # Material parameters
    materials = {
        'charged foam': {
            "eps_r": 8.65,
            "mu_r": 1.0,
            "sigma": 0.205,
            "c_0": 3e8
        }
    }
    material_params = materials['charged foam']
    
    # Find optimal frequency using grid search or bounded optimization
    optimal_freq, max_energy = find_max_energy_frequency(
        start, end, domain_omega, spacestep,
        f, f_dir, f_neu, f_rob,
        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,
        chi_initial, material_params, 
        method='grid', n_samples=n_samples
    )
    
    print("\n" + "="*60)
    print("STEP 2: Running optimization at optimal frequency")
    print("="*60)
    print(f"Optimal frequency: {optimal_freq:.6e} Hz")
    print(f"Initial energy: {max_energy:.6e}")
    
    # Compute wavenumber and Alpha for optimal frequency
    wavenumber = 2 * np.pi * optimal_freq / material_params['c_0']
    omega = 2 * np.pi * optimal_freq
    Alpha = compute_alpha(omega, material_params['eps_r'], 
                         material_params['mu_r'], 
                         material_params['sigma'], 
                         material_params['c_0'])
    
    print(f"Alpha = {Alpha}")
    
    # Set optimization parameters
    S = 0  # surface of the fractal
    for i in range(M):
        for j in range(N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1
    
    V_0 = 1  # initial volume of the domain
    V_obj = 200  # desired volume fraction (adjust as needed)
    mu = 1.0  # initial gradient step
    mu1 = 10**(-5)  # parameter of the volume functional
    
    print(f"Robin boundary points: {S}")
    print(f"Target volume constraint: {V_obj}")
    
    # Run optimization procedure
    chi_optimal, energy_history, u_final, grad_final = your_optimization_procedure(
        domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
        Alpha, mu, chi_initial.copy(), V_obj, mu1, V_0
    )
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    
    # Extract final energy (handle 2D array indexing)
    final_energy = np.real(energy_history[-1, 0])  # Get scalar from 2D array
    print(f"Final energy: {final_energy:.6e}")
    print(f"Energy reduction: {max_energy - final_energy:.6e}")
    
    # Plot results
    print("\nPlotting results...")
    
    # Plot energy history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    # Flatten energy_history to 1D for plotting
    energy_flat = energy_history[:, 0]
    # Remove zero entries at the end (unfilled iterations)
    non_zero_idx = np.where(energy_flat != 0)[0]
    if len(non_zero_idx) > 0:
        energy_flat = energy_flat[:non_zero_idx[-1]+1]
    plt.plot(energy_flat, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title(f'Energy History during Optimization\nFrequency: {optimal_freq:.6e} Hz')
    plt.grid(True)
    
    # Plot optimized chi
    plt.subplot(1, 2, 2)
    plt.imshow(chi_optimal, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Chi density')
    plt.title('Optimized Material Density (Chi)')
    plt.tight_layout()
    plt.show()
    
    # Plot the solution with optimized chi
    postprocessing._plot_controled_solution(u_final, chi_optimal)
    
    return chi_optimal, optimal_freq, max_energy, energy_history


if __name__ == '__main__':
    """
    # Example usage: find and plot optimal frequency
    freq_start = 0.5e9  # Start frequency in Hz 5.222215e9 
    freq_end = 6e9     # End frequency in Hz 5.22225e
    #plot_energy(freq_start, freq_end)
    
    #plot_solution(N=200, freq=2.908375e9)
    compare_chi_random_to_everwhere(freq_start, freq_end, n_points=100)
    """
    print('End.')
    

if __name__ == '__main__':
    """
    # Optimize for frequencies between 0.5 GHz and 6 GHz
    chi_opt, freq_opt, energy_opt, history = optimize_local_maxima(
        start=0.5e9, 
        end=6e9, 
        n_samples=30  # Number of frequencies to sample when finding maximum
    )
    """
    print('End.')

if __name__ == '__main__':
    
    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- set parameters of the geometry
    N = 100  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 3# level of the fractal
    spacestep = 1.0 / N  # mesh size

    # -- set parameters of the partial differential equation
    kx = -1.0
    ky = -1.0
    
    freq = 1.611145e9
    wavenumber = numpy.sqrt(kx**2 + ky**2)  # wavenumber
    wavenumber = freq*2*np.pi/(3e8)

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

    beta = 0.3 # volume fraction
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

    # Create a mask for Robin boundary nodes
    robin_mask = (domain_omega == _env.NODE_ROBIN)

    # Apply optimization result only to Robin nodes
    chi_robin = chi * robin_mask

    # Flatten and get indices
    flat_chi_robin = chi_robin.flatten()
    robin_indices = numpy.where(robin_mask.flatten())[0]

    # Select top V_obj values ONLY from Robin boundary
    n_select = int(min(len(robin_indices), V_obj))

    flat_chi = chi.flatten()
    if n_select > 0:
        # Get chi values only at Robin nodes
        chi_values_at_robin = flat_chi_robin[robin_indices]
        # Sort and get top n_select
        top_indices_in_robin = numpy.argsort(chi_values_at_robin)[-n_select:]
        # Map back to global indices
        global_indices = robin_indices[top_indices_in_robin]
        
        # Create binary chi
        new_flat = numpy.zeros_like(flat_chi_robin)
        new_flat[global_indices] = 1.0
        chi = new_flat.reshape((M, N))
    else:
        chi = numpy.zeros_like(chi)

    # compute forward solution with the binarized chi
    alpha_rob = chi.copy() * Alpha
    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber,
                                   f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)

    # build adjoint right-hand side (same indicator pattern as used previously)
    fadj_indicator = np.ones_like(f, dtype=complex)
    for i, j in numpy.ndindex(fadj_indicator.shape):
        if is_in_interest_domain(i, j, spacestep):
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
    print('initial energy:', energy[0])
    print('final energy:', energy[-1])
    print("energy evolution:", (energy[-1]-energy[0])/energy[0]*100, '%')
    print("for Beta =", beta)
    plot_abs_heatmap(un)
    
    print('End.')




def plot_energy_vs_beta(beta_values, freq, N=100, level=3, numb_iter=100):
    """
    Plot the energy efficiency metric as a function of beta (volume fraction).
    
    Efficiency metric = (E_initial - E_final) / Beta
    This represents the energy reduction per unit of material used.
    Higher values indicate better efficiency.
    
    Parameters:
        beta_values: array-like, list of beta values (volume fractions) to test
        freq: float, frequency in Hz
        N: int, number of points along x-axis (default 100)
        level: int, fractal level (default 3)
        numb_iter: int, number of optimization iterations (default 100)
    
    Returns:
        results: dict with beta_values, efficiency_metric, initial_energies, final_energies
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    M = 2 * N
    spacestep = 1.0 / N
    wavenumber = 2 * np.pi * freq / (3e8)
    
    # Setup coefficients and geometry (same for all betas)
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = \
        preprocessing._set_coefficients_of_pde(M, N)
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)
    
    # Define boundary conditions (planar wave)
    f_dir[:, :] = 0.0
    for i in range(N):
        f_dir[0, i] = 5 * np.exp(-(i - N/2)**2 / 2)
    
    # Material parameters
    materials = {
        'charged foam': {
            "eps_r": 8.65,
            "mu_r": 1.0,
            "sigma": 0.205,
            "c_0": 3e8
        }
    }
    material = materials['charged foam']
    omega = wavenumber * material['c_0']
    Alpha = compute_alpha(omega, material['eps_r'], material['mu_r'], 
                         material['sigma'], material['c_0'])
    
    # Count Robin boundary cells
    S = 0
    for i in range(M):
        for j in range(N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1
    
    print(f"Total Robin boundary cells: {S}")
    print(f"Frequency: {freq:.6e} Hz")
    print(f"Alpha: {Alpha}")
    print("="*60)
    
    # Storage for results
    normalized_energies = []
    initial_energies = []
    final_energies = []
    
    # Loop over beta values
    for idx, beta in enumerate(beta_values):
        print(f"\n[{idx+1}/{len(beta_values)}] Testing beta = {beta:.3f}")
        print("-"*60)
        
        # Initialize chi
        chi = preprocessing._set_chi(M, N, x, y)
        chi = preprocessing.set2zero(chi, domain_omega)
        chi = np.zeros_like(chi)
        
        # Set volume constraint based on beta
        V_obj = S * beta
        V_0 = 1
        mu = 1.0
        mu1 = 10**(-5)
        
        print(f"  Target volume: {V_obj:.1f} cells ({beta*100:.1f}% of boundary)")
        
        # Run optimization
        chi_opt, energy_history, u_final, grad_final = your_optimization_procedure(
            domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
            Alpha, mu, chi.copy(), V_obj, mu1, V_0
        )
        
        # Extract energies (handle 2D array)
        energy_flat = energy_history[:, 0]
        non_zero_idx = np.where(energy_flat != 0)[0]
        if len(non_zero_idx) > 0:
            energy_flat = energy_flat[:non_zero_idx[-1]+1]
        
        E_initial = energy_flat[0]
        E_final = energy_flat[-1]
        
        # Compute energy efficiency metric: (E_initial - E_final) / Beta
        # Higher is better (more energy reduction per unit of material)
        if beta != 0:
            energy_efficiency = (E_initial - E_final) / beta
        else:
            energy_efficiency = 0
        
        normalized_energies.append(energy_efficiency)
        initial_energies.append(E_initial)
        final_energies.append(E_final)
        
        print(f"  Initial energy: {E_initial:.6e}")
        print(f"  Final energy: {E_final:.6e}")
        print(f"  Energy reduction: {E_initial - E_final:.6e}")
        print(f"  Efficiency metric: {energy_efficiency:.6e}")
    
    # Create simple plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(beta_values, normalized_energies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Beta (Volume Fraction)', fontsize=14)
    plt.ylabel('Energy Efficiency: (E_initial - E_final) / Beta', fontsize=14)
    plt.title(f'Energy Reduction Efficiency vs Volume Fraction\nFrequency: {freq:.6e} Hz', 
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    best_idx = np.argmax(normalized_energies)  # Now we want MAXIMUM efficiency
    print(f"Best beta: {beta_values[best_idx]:.3f}")
    print(f"Best efficiency: {normalized_energies[best_idx]:.6e}")
    print(f"Energy reduction: {initial_energies[best_idx] - final_energies[best_idx]:.6e}")
    print(f"Initial energy: {initial_energies[best_idx]:.6e}")
    print(f"Final energy: {final_energies[best_idx]:.6e}")
    
    results = {
        'beta_values': np.array(beta_values),
        'efficiency_metric': np.array(normalized_energies),
        'initial_energies': np.array(initial_energies),
        'final_energies': np.array(final_energies),
        'best_beta': beta_values[best_idx],
        'best_efficiency': normalized_energies[best_idx]
    }
    
    return results

def plot_total_energy_vs_beta(beta_values, opt_freq, freq_start, freq_end, 
                              n_points=100, N=100, level=3, numb_iter=100):
    """
    Optimizes chi at a single frequency (opt_freq) for various beta values,
    then evaluates the *total energy* (summed over [freq_start, freq_end]) 
    for each optimized chi.
    
    Plots the total integrated energy and an efficiency metric vs. beta.
    
    Parameters:
        beta_values: array-like, list of beta values (volume fractions) to test
        opt_freq: float, the single frequency (Hz) to use for optimization
        freq_start: float, start of the frequency range (Hz) for evaluation
        freq_end: float, end of the frequency range (Hz) for evaluation
        n_points: int, number of frequency points to sample for evaluation
        N: int, number of points along x-axis (default 100)
        level: int, fractal level (default 3)
        numb_iter: int, number of optimization iterations (default 100)
    
    Returns:
        results: dict with beta_values, total_optimized_energies, 
                 total_baseline_energy, efficiency_metric
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    M = 2 * N
    spacestep = 1.0 / N
    
    # --- Standard Setup ---
    # Setup coefficients and geometry (same for all betas)
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = \
        preprocessing._set_coefficients_of_pde(M, N)
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)
    
    # Define boundary conditions (planar wave)
    f_dir[:, :] = 0.0
    for i in range(N):
        f_dir[0, i] = 5 * np.exp(-(i - N/2)**2 / 2)
    
    # Material parameters
    materials = {
        'charged foam': {
            "eps_r": 8.65,
            "mu_r": 1.0,
            "sigma": 0.205,
            "c_0": 3e8
        }
    }
    material = materials['charged foam']
    eps_r, mu_r, sigma, c_0 = material.values()

    # Count Robin boundary cells
    S = 0
    for i in range(M):
        for j in range(N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1
    
    print(f"Total Robin boundary cells: {S}")
    print(f"Optimization Frequency: {opt_freq:.6e} Hz")
    print(f"Evaluation Range: [{freq_start:.6e}, {freq_end:.6e}] Hz")
    print("="*60)

    # --- Frequencies for Optimization ---
    wavenumber_opt = 2 * np.pi * opt_freq / c_0
    omega_opt = 2 * np.pi * opt_freq
    Alpha_opt = compute_alpha(omega_opt, eps_r, mu_r, sigma, c_0)
    
    # --- Frequencies for Evaluation ---
    frequencies_eval = np.linspace(freq_start, freq_end, num=n_points)
    wavenumbers_eval = 2 * np.pi * frequencies_eval / c_0


    # --- 1. Calculate Baseline Total Energy (Chi=0) ---
    print("Calculating baseline total energy (Chi=0)...")
    chi_zero = preprocessing.set2zero(preprocessing._set_chi(M, N, x, y), domain_omega)
    total_energy_zero = 0.0
    
    for i in range(n_points):
        k_eval = wavenumbers_eval[i]
        omega_eval = 2 * np.pi * frequencies_eval[i]
        Alpha_eval = compute_alpha(omega_eval, eps_r, mu_r, sigma, c_0)
        
        alpha_rob_zero = Alpha_eval * chi_zero
        u_zero = processing.solve_helmholtz(domain_omega, spacestep, k_eval, 
                                            f, f_dir, f_neu, f_rob,
                                            beta_pde, alpha_pde, alpha_dir, 
                                            beta_neu, beta_rob, alpha_rob_zero)
        ene_zero = compute_objective_function(domain_omega, u_zero, spacestep, 0, 0)
        total_energy_zero += ene_zero
        
    print(f"Baseline Total Energy (Chi=0): {total_energy_zero:.6e}")
    print("="*60)

    # --- 2. Loop over beta values ---
    total_optimized_energies = []
    efficiency_metrics = []
    energies = []
    
    for idx, beta in enumerate(beta_values):
        print(f"\n[{idx+1}/{len(beta_values)}] Testing beta = {beta:.3f}")
        print("-"*60)
        
        # --- a. Optimize Chi at opt_freq for this beta ---
        chi_initial = np.zeros_like(domain_omega, dtype=float)
        V_obj = S * beta
        V_0 = 1
        mu = 1.0
        mu1 = 10**(-5)
        
        print(f"  Target volume: {V_obj:.1f} cells ({beta*100:.1f}% of boundary)")
        print(f"  Running optimization at {opt_freq:.6e} Hz...")
        
        # We need a placeholder alpha_rob, your_optimization_procedure will update it
        alpha_rob_placeholder = Alpha_opt * chi_initial 
        
        chi_optimized, energy_history, _, _ = your_optimization_procedure(
            domain_omega, spacestep, wavenumber_opt, f, f_dir, f_neu, f_rob,
            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob_placeholder,
            Alpha_opt, mu, chi_initial.copy(), V_obj, mu1, V_0
        )
        
        # --- b. Binarize the optimized chi ---
        robin_mask = (domain_omega == _env.NODE_ROBIN)
        chi_robin = chi_optimized * robin_mask
        flat_chi_robin = chi_robin.flatten()
        robin_indices = numpy.where(robin_mask.flatten())[0]
        n_select = int(min(len(robin_indices), V_obj))
        
        chi_binarized = numpy.zeros_like(chi_optimized)
        if n_select > 0:
            chi_values_at_robin = flat_chi_robin[robin_indices]
            top_indices_in_robin = numpy.argsort(chi_values_at_robin)[-n_select:]
            global_indices = robin_indices[top_indices_in_robin]
            
            new_flat = numpy.zeros_like(flat_chi_robin)
            new_flat[global_indices] = 1.0
            chi_binarized = new_flat.reshape((M, N))
        
        print(f"  Binarized chi sum: {np.sum(chi_binarized):.1f}")

        # --- c. Evaluate this binarized chi over the frequency range ---
        print(f"  Evaluating optimized chi over [{freq_start:.6e}, {freq_end:.6e}] Hz...")
        current_total_energy = 0.0
        
        for i in range(n_points):
            k_eval = wavenumbers_eval[i]
            omega_eval = 2 * np.pi * frequencies_eval[i]
            Alpha_eval = compute_alpha(omega_eval, eps_r, mu_r, sigma, c_0)
            
            alpha_rob_opt = Alpha_eval * chi_binarized
            u_opt = processing.solve_helmholtz(domain_omega, spacestep, k_eval, 
                                                f, f_dir, f_neu, f_rob,
                                                beta_pde, alpha_pde, alpha_dir, 
                                                beta_neu, beta_rob, alpha_rob_opt)
            ene_opt = compute_objective_function(domain_omega, u_opt, spacestep, 0, 0)
            current_total_energy += ene_opt
            
        print(f"  Total Optimized Energy: {current_total_energy:.6e}")
        
        # --- d. Store results ---
        total_optimized_energies.append(current_total_energy)
        
        reduction = total_energy_zero - current_total_energy
        efficiency = reduction / beta if beta > 0 else 0
        efficiency_metrics.append(efficiency)
        
        print(f"  Total Energy Reduction: {reduction:.6e}")
        print(f"  Efficiency Metric: {efficiency:.6e}")

    # --- 3. Plotting ---
    plt.figure(figsize=(18, 7))
    
    # Plot 1: Total Energy vs. Beta
    plt.plot(beta_values, total_optimized_energies, 'bo-', linewidth=2, markersize=8,
             label='Optimized Total Energy')
    plt.axhline(total_energy_zero, color='r', linestyle='--', linewidth=2,
                label=f'Baseline Total Energy (Chi=0) = {total_energy_zero:.4e}')
    plt.xlabel('Beta (Volume Fraction)', fontsize=14)
    plt.ylabel('Total Integrated Energy', fontsize=14)
    plt.title(f'Total Energy vs Volume Fraction\n(Optimized at {opt_freq:.4e} Hz)', 
              fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # --- 4. Print Summary ---
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    best_idx = np.argmax(efficiency_metrics)
    print(f"Best beta (for efficiency): {beta_values[best_idx]:.3f}")
    print(f"Best efficiency: {efficiency_metrics[best_idx]:.6e}")
    print(f"  Total energy reduction: {total_energy_zero - total_optimized_energies[best_idx]:.6e}")
    print(f"  Optimized total energy: {total_optimized_energies[best_idx]:.6e}")
    print(f"  Baseline total energy: {total_energy_zero:.6e}")
    
    results = {
        'beta_values': np.array(beta_values),
        'total_optimized_energies': np.array(total_optimized_energies),
        'total_baseline_energy': total_energy_zero,
        'efficiency_metric': np.array(efficiency_metrics),
        'best_beta': beta_values[best_idx],
        'best_efficiency': efficiency_metrics[best_idx]
    }
    
    return results


# Example usage:
if __name__ == '__main__':
    """
    # Test different volume fractions
    #beta_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    beta_values = np.linspace(0, 0.4, 100)
    
    # Use the optimal frequency found earlier (or any frequency of interest)
    freq = freq = 1.611145e9  # Hz
    
    results = plot_energy_vs_beta(beta_values, freq, N=100, level=3, numb_iter=100)
    
    # The best beta is the one with highest efficiency (most energy reduction per material used)
    print(f"\nBest efficiency beta: {results['best_beta']}")
    print(f"Best efficiency value: {results['best_efficiency']:.6e}")
    """
if __name__ == '__main__':
    
    # Example: Plot total energy vs. beta
    
    # Define the beta values to test
    beta_values = [0.1, 0.15, 0.2, 0.25, 0.35, 0.4,0.5,0.6,0.7] #np.linspace(0.1, 0.5, 11) # Test 0% to 50% in 11 steps
    
    # Define the single frequency for optimization
    opt_freq = 1.611145e9  # Hz
    
    # Define the frequency range for evaluation
    freq_start = 0.5e9
    freq_end = 6e9
    
    # Run the analysis
    results = plot_total_energy_vs_beta(
        beta_values=beta_values,
        opt_freq=opt_freq,
        freq_start=freq_start,
        freq_end=freq_end,
        n_points=100,  # Number of points for evaluation
        level=3,
        numb_iter=100
    )
    
    print(f"\nAnalysis complete. Best beta for efficiency: {results['best_beta']:.4f}")

    print('End.')


def compare_optimized_chi_to_configurations(start, end, beta, opt_freq, n_points=100):
    """
    Compare energy profiles of optimized chi vs random chi, chi=0, and chi=1 everywhere.
    
    Parameters:
        start: float, start frequency in Hz
        end: float, end frequency in Hz
        beta: float, volume fraction for random and optimized chi
        opt_freq: float, frequency at which optimization was performed
        n_points: int, number of frequency points to sample
    
    Returns:
        results: dict with frequencies and energies for all configurations
    """
    """
    N = 100  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 3  # level of the fractal
    spacestep = 1.0 / N  # mesh size

    frequencies = numpy.linspace(start, end, num=n_points)
    wavenumbers = 2*np.pi*frequencies/(3e8)

    # Setup coefficients and geometry
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = \
        preprocessing._set_coefficients_of_pde(M, N)
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    # Material parameters
    materials = {
        'charged foam': {
            "eps_r": 8.65,
            "mu_r": 1.0,
            "sigma": 0.205,
            "c_0": 3e8
        }
    }
    eps_r, mu_r, sigma, c_0 = materials['charged foam'].values()

    # Define boundary conditions (planar wave)
    f_dir[:, :] = 0.0
    for i in range(N):
        f_dir[0, i] = numpy.exp(-((i-N/2)*spacestep)**2/0.02)

    # Count Robin boundary cells
    S = 0
    for i in range(M):
        for j in range(N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1
    
    print(f"Total Robin boundary cells: {S}")
    print(f"Optimization frequency: {opt_freq:.6e} Hz")
    print(f"Beta (volume fraction): {beta:.3f}")
    print("="*60)

    # ========== Configuration 1: Optimized Chi ==========
    print("\n[1/4] Computing optimized chi at target frequency...")
    
    # Run optimization at opt_freq
    wavenumber_opt = 2 * np.pi * opt_freq / c_0
    omega_opt = 2 * np.pi * opt_freq
    Alpha_opt = compute_alpha(omega_opt, eps_r, mu_r, sigma, c_0)
    
    chi_initial = preprocessing._set_chi(M, N, x, y)
    chi_initial = preprocessing.set2zero(chi_initial, domain_omega)
    chi_initial = np.zeros_like(chi_initial)
    plot_abs_heatmap(chi_initial)
    
    V_obj = S * beta
    V_0 = 1
    mu = 1.0
    mu1 = 10**(-5)
    
    print(f"  Running optimization at {opt_freq:.6e} Hz...")
    chi_optimized, energy_history, u_opt, grad_opt = your_optimization_procedure(
        domain_omega, spacestep, wavenumber_opt, f, f_dir, f_neu, f_rob,
        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
        Alpha_opt, mu, chi_initial.copy(), V_obj, mu1, V_0
    )
    
    # Binarize optimized chi
    robin_mask = (domain_omega == _env.NODE_ROBIN)
    chi_robin = chi_optimized * robin_mask
    flat_chi_robin = chi_robin.flatten()
    robin_indices = numpy.where(robin_mask.flatten())[0]
    n_select = int(min(len(robin_indices), V_obj))
    
    if n_select > 0:
        chi_values_at_robin = flat_chi_robin[robin_indices]
        top_indices_in_robin = numpy.argsort(chi_values_at_robin)[-n_select:]
        global_indices = robin_indices[top_indices_in_robin]
        
        new_flat = numpy.zeros_like(flat_chi_robin)
        new_flat[global_indices] = 1.0
        chi_optimized = new_flat.reshape((M, N))
    else:
        chi_optimized = numpy.zeros_like(chi_optimized)
    
    print(f"  Optimized chi sum: {numpy.sum(chi_optimized)}")
    plot_abs_heatmap(chi_optimized)
    
    # ========== Configuration 2: Random Chi ==========
    print("\n[2/4] Creating random chi configuration...")
    num_ones = int(beta * S)
    chi_random = np.zeros((M, N), dtype=np.float64)
    robin_indices_full = np.where(domain_omega == _env.NODE_ROBIN)
    shuffled_indices = np.arange(S)
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(shuffled_indices)
    indices_to_set = shuffled_indices[:num_ones]
    i_coords, j_coords = robin_indices_full[0][indices_to_set], robin_indices_full[1][indices_to_set]
    chi_random[i_coords, j_coords] = 1.0
    print(f"  Random chi sum: {numpy.sum(chi_random)}")

    # ========== Configuration 3: Chi_0 (given by the teacher)==========
    print("\n[3/4] Creating chi=0 configuration...")
    chi_zero = preprocessing._set_chi(M, N, x, y)
    chi_zero = preprocessing.set2zero(chi_zero, domain_omega)
    print(f"  Chi=0 sum: {numpy.sum(chi_zero)}")

    # ========== Configuration 4: Chi = 1 everywhere ==========
    print("\n[4/4] Creating chi=1 everywhere configuration...")
    chi_everywhere = np.array([[1 if domain_omega[i, j] == _env.NODE_ROBIN else 0
                               for j in range(N)] 
                              for i in range(M)])
    print(f"  Chi=1 sum: {numpy.sum(chi_everywhere)}")

    # ========== Compute energies for all frequencies ==========
    print("\n" + "="*60)
    print("Computing energy profiles across frequency range...")
    print("="*60)
    
    optimized_energies = []
    random_energies = []
    zero_energies = []
    everywhere_energies = []

    for idx, k in enumerate(wavenumbers):
        print(f'  Progress: {idx+1}/{len(wavenumbers)}', end='\r')
        
        omega = 2 * np.pi * frequencies[idx]
        Alpha = compute_alpha(omega, eps_r, mu_r, sigma, c_0)
        
        # Optimized chi
        alpha_rob_opt = Alpha * chi_optimized
        u_opt = processing.solve_helmholtz(domain_omega, spacestep, k,
                            f, f_dir, f_neu, f_rob,
                            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob_opt)
        ene_opt = compute_objective_function(domain_omega, u_opt, spacestep, mu1=0, V_0=0)
        optimized_energies.append(ene_opt)
        
        # Random chi
        alpha_rob_random = Alpha * chi_random
        u_random = processing.solve_helmholtz(domain_omega, spacestep, k,
                            f, f_dir, f_neu, f_rob,
                            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob_random)
        ene_random = compute_objective_function(domain_omega, u_random, spacestep, mu1=0, V_0=0)
        random_energies.append(ene_random)

        # Chi = 0
        alpha_rob_zero = Alpha * chi_zero
        u_zero = processing.solve_helmholtz(domain_omega, spacestep, k,
                            f, f_dir, f_neu, f_rob,
                            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob_zero)
        ene_zero = compute_objective_function(domain_omega, u_zero, spacestep, mu1=0, V_0=0)
        zero_energies.append(ene_zero)

        # Chi = 1 everywhere
        alpha_rob_everywhere = Alpha * chi_everywhere
        u_everywhere = processing.solve_helmholtz(domain_omega, spacestep, k,
                            f, f_dir, f_neu, f_rob,
                            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob_everywhere)
        ene_everywhere = compute_objective_function(domain_omega, u_everywhere, spacestep, mu1=0, V_0=0)
        everywhere_energies.append(ene_everywhere)
    
    print()  # New line after progress
    
    # ========== Plot Results ==========
    print("\nPlotting comparison...")
    
    plt.figure(figsize=(12, 7))
    
    plt.plot(frequencies, optimized_energies, 'b-', linewidth=2.5, label='Optimized Chi', alpha=0.9)
    plt.plot(frequencies, random_energies, 'r--', linewidth=2, label='Random Chi', alpha=0.8)
    plt.plot(frequencies, zero_energies, 'g:', linewidth=2, label='Chi = 0', alpha=0.8)
    plt.plot(frequencies, everywhere_energies, 'm-.', linewidth=2, label='Chi = 1', alpha=0.8)
    plt.axvline(opt_freq, color='k', linestyle='--', linewidth=1.5, alpha=0.5, label='Opt. Freq.')
    
    plt.xlabel('Frequency (Hz)', fontsize=14)
    plt.ylabel('Energy', fontsize=14)
    plt.title(f'Energy Comparison Across Frequencies\n(β={beta:.2f}, Optimized at {opt_freq:.4e} Hz)', 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # ========== Print Summary ==========
    print("\n" + "="*60)
    print("ENERGY SUMS ACROSS ALL FREQUENCIES")
    print("="*60)
    
    sum_optimized = np.sum(optimized_energies)
    sum_random = np.sum(random_energies)
    sum_zero = np.sum(zero_energies)
    sum_everywhere = np.sum(everywhere_energies)
    
    print(f"\nTotal energy sum (integrated over frequency range):")
    print(f"  Optimized chi:     {sum_optimized:.6e}")
    print(f"  Random chi:        {sum_random:.6e}")
    print(f"  Chi = 0:           {sum_zero:.6e}")
    print(f"  Chi = 1:           {sum_everywhere:.6e}")
    
    print(f"\nEnergy reduction vs Chi=0 (baseline):")
    print(f"  Optimized: {sum_zero - sum_optimized:.6e} ({(sum_zero - sum_optimized)/sum_zero*100:.2f}%)")
    print(f"  Random:    {sum_zero - sum_random:.6e} ({(sum_zero - sum_random)/sum_zero*100:.2f}%)")
    print(f"  Chi=1:     {sum_zero - sum_everywhere:.6e} ({(sum_zero - sum_everywhere)/sum_zero*100:.2f}%)")
    
    print(f"\nOptimized vs Random:")
    print(f"  Additional reduction: {sum_random - sum_optimized:.6e}")
    print(f"  Improvement:          {(sum_random - sum_optimized)/sum_random*100:.2f}%")

    
    postprocessing._plot_controled_solution(u_opt, chi_optimized)
    postprocessing._plot_energy_history(optimized_energies)
    
    results = {
        'frequencies': frequencies,
        'optimized_energies': np.array(optimized_energies),
        'random_energies': np.array(random_energies),
        'zero_energies': np.array(zero_energies),
        'everywhere_energies': np.array(everywhere_energies),
        'chi_optimized': chi_optimized,
        'chi_random': chi_random,
        'chi_zero': chi_zero,
        'chi_everywhere': chi_everywhere,
        'opt_freq': opt_freq,
        'beta': beta
    }
    """
    return results


# Example usage:
if __name__ == '__main__':
    # Parameters
    freq_start = 0.5e9
    freq_end = 6e9
    beta = 0.5  # Volume fraction
    opt_freq = 1.611145e9  # Frequency where optimization was performed
    
    results = compare_optimized_chi_to_configurations(
        start=freq_start,
        end=freq_end,
        beta=beta,
        opt_freq=opt_freq,
        n_points=100
    )
    
    print('\nEnd.')