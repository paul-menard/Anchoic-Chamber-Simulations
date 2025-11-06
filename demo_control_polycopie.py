# -*- coding: utf-8 -*-

# Python packages
import matplotlib.pyplot
import numpy
import os
import scipy.integrate as integrate
from scipy.interpolate import RectBivariateSpline

# MRG packages
import _env
import preprocessing
import processing
import postprocessing


def your_compute_objective_function(u, spacestep, mu1, V_0):
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
    energy_total = numpy.sum(numpy.abs(u)**2) * spacestep**2
    
    # Term 2: ‖1_{Ω_interest}u‖²_{L²(Ω)} - Energy in interest zone
    interest_domain = numpy.array([[is_in_interest_domain(i, j) 
                                    for j in range(N)] 
                                    for i in range(M)])
    energy_interest = numpy.sum(interest_domain * numpy.abs(u)**2) * spacestep**2
    
    # Total objective function
    energy = energy_total + energy_interest
    
    print(f'    Energy total: {energy_total:.6e}, Energy interest: {energy_interest:.6e}')
    
    return energy

def BelongsInteriorDomain(node):
	if (node < 0):
		return 1
	if node == 3:
		#print("Robin")
		return 2
	else:
		return 0
      

      


def compute_gradient_descent(chi, grad, domain, mu):
	"""This function makes the gradient descent.
	This function has to be used before the 'Projected' function that will project
	the new element onto the admissible space.
	:param chi: density of absorption define everywhere in the domain
	:param grad: parametric gradient associated to the problem
	:param domain: domain of definition of the equations
	:param mu: step of the descent
	:type chi: numpy.array((M,N), dtype=float64
	:type grad: numpy.array((M,N), dtype=float64)
	:type domain: numpy.array((M,N), dtype=int64)
	:type mu: float
	:return chi:
	:rtype chi: numpy.array((M,N), dtype=float64

	.. warnings also: It is important that the conditions be expressed with an "if",
			not with an "elif", as some points are neighbours to multiple points
			of the Robin frontier.
	"""

	(M, N) = numpy.shape(domain)
	# for i in range(0, M):
	# 	for j in range(0, N):
	# 		if domain_omega[i, j] != _env.NODE_ROBIN:
	# 			chi[i, j] = chi[i, j] - mu * grad[i, j]
	# # for i in range(0, M):
	# 	for j in range(0, N):
	# 		if preprocessing.is_on_boundary(domain[i , j]) == 'BOUNDARY':
	# 			chi[i,j] = chi[i,j] - mu*grad[i,j]
	# print(domain,'jesuisla')
	#chi[50,:] = chi[50,:] - mu*grad[50,:]
	for i in range(1, M - 1):
		for j in range(1, N - 1):
			#print(i,j)
			#chi[i,j] = chi[i,j] - mu * grad[i,j]
			a = BelongsInteriorDomain(domain[i + 1, j])
			b = BelongsInteriorDomain(domain[i - 1, j])
			c = BelongsInteriorDomain(domain[i, j + 1])
			d = BelongsInteriorDomain(domain[i, j - 1])
			if a == 2:
				#print(i+1,j, "-----", "i+1,j")
				chi[i + 1, j] = chi[i + 1, j] - mu * grad[i, j]
			if b == 2:
				#print(i - 1, j, "-----", "i - 1, j")
				chi[i - 1, j] = chi[i - 1, j] - mu * grad[i, j]
			if c == 2:
				#print(i, j + 1, "-----", "i , j + 1")
				chi[i, j + 1] = chi[i, j + 1] - mu * grad[i, j]
			if d == 2:
				#print(i, j - 1, "-----", "i , j - 1")
				chi[i, j - 1] = chi[i, j - 1] - mu * grad[i, j]

	return chi



def compute_gradient(domain_omega, u, p, Alpha):
    """
    This function computes the parametric gradient.

    Parameters:
        domain_omega: Matrix (NxP), it defines the domain and the shape of the Robin frontier.
        u: Matrix (NxP), it is the solution of the Helmholtz problem.
        p: Matrix (NxP), it is the solution of the adjoint Helmholtz problem.
        Alpha: Complex, absorption coefficient.
    """
    grad = Alpha * u @ p.T
    return -numpy.real(grad)


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


def solve_adjoint_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob, u):
    """
    This function solves the adjoint Helmholtz problem.

    Parameters:
        cf solve_helmholtz's remarks.
    """
    # Modify right-hand side for adjoint problem
    interest_domain = numpy.array([
        [is_in_interest_domain(i, j) for j in range(numpy.shape(domain_omega)[1])]
        for i in range(numpy.shape(domain_omega)[0])
    ])
    f_adj = -2 * numpy.conj(u) + numpy.where(interest_domain == 1, -2 * Alpha * numpy.conj(u), 0)
    f_dir_adj = numpy.zeros_like(f_dir)

    # Solve adjoint problem
    p = processing.solve_helmholtz(domain_omega, spacestep, omega, f_adj, f_dir_adj, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    return p

def _apply_projection(chi_unprojected, ell, domain_omega):
    """
    Applique le projecteur P_l[chi] = max(0, min(chi + l, 1))
    sur la frontière de Robin.
    """
    
    # Applique le décalage de Lagrange
    chi_projected = chi_unprojected + ell
    
    # Projette sur [0, 1]
    chi_projected = numpy.maximum(0.0, chi_projected)
    chi_projected = numpy.minimum(1.0, chi_projected)
    
    # Assure que la projection n'affecte que la frontière de Robin
    # (Le gradient doit déjà être nul ailleurs, mais c'est une sécurité)
    mask = (domain_omega == _env.NODE_ROBIN)
    chi_projected = chi_projected * mask
    
    return chi_projected



def your_optimization_procedure(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                 beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                                 Alpha, mu, chi, V_obj, mu1, V_0):
    """
    This function returns the optimized density.

    Parameters:
        cf solve_helmholtz's remarks.
        Alpha: Complex, it corresponds to the absorption coefficient.
        mu: float, it is the initial step of the gradient descent.
        V_obj: float, it characterizes the volume constraint on the density chi.
        mu1: float, it characterizes the importance of the volume constraint on the domain.
        V_0: float, volume constraint on the domain.
    """
    k = 0 # Iteration counter
    (M, N) = numpy.shape(domain_omega)
    numb_iter = 20  # Number of iterations
    energy = numpy.zeros((numb_iter + 1, 1), dtype=numpy.float64)

    while k < numb_iter and mu > 10**(-5):
        print('---- iteration number = ', k)
        print('1. computing solution of Helmholtz problem, i.e., u')
        u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        print('2. computing solution of adjoint problem, i.e., p')
        p = solve_adjoint_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                     beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob, u)
        print('3. computing objective function, i.e., energy')
        ene = your_compute_objective_function(u, spacestep, mu1, V_0)
        energy[0] = ene
        print('    energy = ', ene)
        print('4. computing parametric gradient')
        grad = compute_gradient(domain_omega, u, p, Alpha)
        print("grad = ",grad)

        while ene >= energy[k] and mu > 10**(-5):
            print('    a. computing gradient descent')
            chi = compute_gradient_descent(chi, grad, domain_omega, mu)
            print('    b. computing projected gradient')
            grad = compute_gradient(domain_omega, u, p, Alpha)
            print('    c. computing solution of Helmholtz problem, i.e., u')
            u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
            print('    d. computing objective function, i.e., energy (E)')
            ene = your_compute_objective_function(u, spacestep, mu1, V_0)
            energy[k + 1] = ene
            if ene < energy[k]:
                # The step is increased if the energy decreased
                mu = mu * 1.1
            else:
                # The step is decreased if the energy increased
                mu = mu / 2

        k += 1

    print('end. computing solution of Helmholtz problem, i.e., u')
    return chi, energy, u, grad

"""
if __name__ == '__main__':
    # ----------------------------------------------------------------------
    # -- Feel free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    print('='*60)
    print('ANECHOIC CHAMBER OPTIMIZATION')
    print('='*60)
    # -- Set parameters of the geometry
    N = 50  # Number of points along x-axis
    M = 2 * N  # Number of points along y-axis
    level = 0  # Level of the fractal
    spacestep = 1.0 / N  # Mesh size

    # -- Set parameters of the partial differential equation
    kx = -1.0
    ky = -1.0
    wavenumber = numpy.sqrt(kx**2 + ky**2)  # Wavenumber
    wavenumber = 2*numpy.pi*4.53e9/3e8  # Wavenumber

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # --- Set coefficients of the partial differential equation
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

    # -- Set right-hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # -- Set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    # ----------------------------------------------------------------------
    # -- Feel free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- Define boundary conditions
    # Planar wave defined on top
    f_dir[:, :] = 0.0
    for i in range(N):
        f_dir[0,i] = 5*numpy.exp(-(i-N/2)**2/2)

    # -- Initialize
    alpha_rob[:, :] = -wavenumber * 1j

    # -- Define material density matrix
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)

    # -- Define absorbing material
    Alpha = 10.0 - 10.0 * 1j
    alpha_rob = Alpha * chi

    print(f'Absorption coefficient: α = {Alpha}')

    # -- Set parameters for optimization
    S = 0  # Surface of the fractal
    for i in range(0, M):
        for j in range(0, N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1
    V_0 = 1  # Initial volume of the domain
    V_obj = numpy.sum(numpy.sum(chi)) / S  # Constraint on the density
    mu = 5  # Initial gradient step
    mu1 = 1e-5  # Parameter of the volume functional

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # -- Compute finite difference solution
    print('\n' + '='*60)
    print('INITIAL SOLUTION (before optimization)')
    print('='*60)
    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                                    beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    chi0 = chi.copy()
    u0 = u.copy()

    # ----------------------------------------------------------------------
    # -- Feel free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- Compute optimization
    print('\n' + '='*60)
    print('STARTING OPTIMIZATION')
    print('='*60)

    energy = numpy.zeros((100 + 1, 1), dtype=numpy.float64)
    chi, energy, u, grad = your_optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                                                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                                                        Alpha, mu, chi, V_obj, mu1, V_0)

    print('\n' + '='*60)
    print('OPTIMIZATION RESULTS')
    print('='*60)
    print(energy)
    

    # -- Plot chi, u, and energy
    postprocessing._plot_uncontroled_solution(u0, chi0)
    postprocessing._plot_controled_solution(u, chi)
    err = u - u0
    postprocessing._plot_error(err)
    postprocessing._plot_energy_history(energy)

    print('End.')
"""



if __name__ == "__main__" :
    print('='*60)
    print('ANECHOIC CHAMBER OPTIMIZATION')
    print('='*60)
    # -- Set parameters of the geometry
    N = 150  # Number of points along x-axis
    M = 2 * N  # Number of points along y-axis
    level = 3  # Level of the fractal
    spacestep = 1.0 / N  # Mesh size

    # -- Set parameters of the partial differential equation
    kx = -1.0
    ky = -1.0
    frequencies  = numpy.linspace(1e8,6e9,30)   
    wavenumber = numpy.sqrt(kx**2 + ky**2)  # Wavenumber
    wavennumbers = 2*numpy.pi*frequencies/3e8  # Wavenumbers

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # --- Set coefficients of the partial differential equation
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

    # -- Set right-hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # -- Set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)
    f_dir[:, :] = 0.0
    for i in range(N):
        f_dir[0,i] = 5*numpy.exp(-(i-N/2)**2/2)

    energies = []

    # -- Define material density matrix
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)
    for i in range(M):
        for j in range(N):
            if domain_omega[i, j] != _env.NODE_ROBIN:
                chi[i, j] = 1.0

    # -- Define absorbing material
    Alpha = 0.01 - 0.01* 1j
    alpha_rob = Alpha * chi

    print(f'Absorption coefficient: α = {Alpha}')

    # -- Set parameters for optimization
    S = 0  # Surface of the fractal
    for i in range(0, M):
        for j in range(0, N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1
    V_0 = 1  # Initial volume of the domain
    V_obj = numpy.sum(numpy.sum(chi)) / S  # Constraint on the density
    mu = 5  # Initial gradient step
    mu1 = 1e-5  # Parameter of the volume functional

    print("computing objective function vs frequency")
    for wavenumber in wavennumbers:
        u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                                    beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        ene = your_compute_objective_function(u, spacestep, mu1, V_0)
        energies.append(ene)
    print("plotting objective function vs frequency...")
    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(frequencies, energies)
    matplotlib.pyplot.xlabel('Frequency (Hz)')
    matplotlib.pyplot.ylabel('Objective Function J')
    matplotlib.pyplot.title('Objective Function vs Frequency')
    matplotlib.pyplot.grid()
    matplotlib.pyplot.show()
    postprocessing._plot_uncontroled_solution(u, chi)
    print('End.')
