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

    # -- define material density matrix
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)
    chi_everywhere = np.array([[1 if domain_omega[i, j] == _env.NODE_ROBIN else 0
                               for j in range(N)] 
                              for i in range(M)])

    random_energies = []
    everywhere_energies = []

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
    plt.plot(frequencies, random_energies, 'r-x', label='Random chi')
    plt.plot(frequencies, everywhere_energies, 'b-x', label='Chi = 1')
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
    
    # Example usage: find and plot optimal frequency
    freq_start = 1.67e9  # Start frequency in Hz 5.222215e9 
    freq_end = 1.69e9     # End frequency in Hz 5.22225e
    #plot_energy(freq_start, freq_end)
    
    #plot_solution(N=200, freq=2.908375e9)
    compare_chi_random_to_everwhere(freq_start, freq_end, n_points=100)
    
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
