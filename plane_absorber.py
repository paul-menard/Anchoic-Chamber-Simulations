import matplotlib.pyplot as plt
import numpy as np
import scipy 

def plot_bode(G, sigmas, eps_r = 1.4, omega=2 * np.pi * 1e9, min_sigma=None, min_gain=None):
    """Plot the Bode diagram of the absorber's absorption coefficient.

    Parameters:
        G : transfer function of the absorber.
        sigmas: array_like, list of absorption coefficients corresponding to different flow resistivities.
    """
    print("Computing Gains for Bode diagram...")
    Gains = []
    for sigma in sigmas:
        gain = G(omega, eps_r, sigma)
        Gains.append(gain)
    print("Gains compute")
    print("Generating Bode diagram for sigmas:", sigmas[0], "to", sigmas[-1])
    plt.figure(figsize=(10, 6))

    # Magnitude plot
    plt.plot(sigmas, 20*np.log10(Gains), 'b-o')
    plt.scatter(min_sigma, 20*np.log10(min_gain), color='red', zorder=5)
    plt.title("Bode Diagram of the Plane Absorber")
    plt.ylabel('Magnitude (dB)')
    plt.xscale('log')
    plt.grid(which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

def calculate_reflectivity(omega, eps_r, sigma):
    """
    Calcule la réflectivité (R) d'une onde EM à incidence normale
    sur un matériau diélectrique avec pertes.

    Arguments :
    omega : Fréquence angulaire (rad/s)
    eps_r : Permittivité relative (partie réelle)
    sigma : Conductivité électrique (S/m)
    """
    
    # Constante epsilon_0 (Permittivité du vide)
    eps_0 = scipy.constants.epsilon_0  # 8.854...e-12 F/m
    mu_0 = scipy.constants.mu_0        # 4π x 10^-7 H/m
    
    # 1. Calculer le terme complexe sous la racine carrée :
    #    eps_r_complexe_normalisée = eps_r - i * (sigma / (eps_0 * omega))
    complex_term = eps_r*mu_0 - 1j * (sigma / (omega*eps_0))
    
    # 2. Calculer la racine carrée (X dans votre formule)
    #    C'est l'indice de réfraction complexe normalisé
    X = np.sqrt(complex_term)
    
    # 3. Calculer le coefficient de réflexion de champ (Gamma)
    #    Gamma = (1 - X) / (1 + X)
    numerator = 1 - X
    denominator = 1 + X
    gamma = numerator / denominator
    
    # 4. Calculer la réflectivité en puissance R
    #    R = |Gamma|^2
    R = np.abs(gamma)**2

    #R += np.abs((2/(1+X))) * np.exp(-2 * omega * n_prime_prime(omega, eps_r, sigma) / 3e8)
    #R = R**2
    
    return R

def minimize_reflectivity():
    """
    Minimize the reflectivity over sigma at 1 GHz and eps_r = 1.4.
    """
    omega = 2 * np.pi * 1e9  # 1 GHz
    eps_r = 1.4  # Relative permittivity

    # Minimize the reflectivity function
    result = scipy.optimize.minimize_scalar(
        lambda sigma: calculate_reflectivity(omega, eps_r, sigma),
        bounds=(1e-2, 1e6),  # Bounds for sigma
        method='bounded'
    )

    # Print the results
    print("Minimum reflectivity:", result.fun)
    print("Optimal sigma:", result.x)

    return result.x, result.fun


def n_prime_prime(omega, eps_r, sigma):
    """
    Calcule la partie imaginaire de l'indice de réfraction complexe
    pour un matériau diélectrique avec pertes.

    Arguments :
    omega : Fréquence angulaire (rad/s)
    eps_r : Permittivité relative (partie réelle)
    sigma : Conductivité électrique (S/m)
    """
    
    # Constante epsilon_0 (Permittivité du vide)
    eps_0 = scipy.constants.epsilon_0  # 8.854...e-12 F/m
    mu_0 = scipy.constants.mu_0        # 4π x 10^-7 H/m
    
    n = np.sqrt((eps_r*mu_0)**2 +(mu_0 * sigma / omega*eps_0)**2)-mu_0 * eps_r
    n = np.sqrt(n) / np.sqrt(2) 
    return n

def plot_bode_frequency_range(sigma, eps_r):
    """
    Plot the Bode diagram of the absorber's reflectivity over a frequency range (0.1 GHz to 100 GHz).

    Parameters:
        sigma : float
            Optimal conductivity (S/m).
        eps_r : float
            Relative permittivity.
    """
    # Define the frequency range (0.1 GHz to 100 GHz)
    frequencies = np.logspace(np.log10(0.1e9), np.log10(100e9), num=500)  # Frequencies in Hz
    omegas = 2 * np.pi * frequencies  # Convert to angular frequencies

    # Compute reflectivity for each frequency
    reflectivities = [calculate_reflectivity(omega, eps_r, sigma) for omega in omegas]

    # Plot the Bode diagram
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies / 1e9, 20 * np.log10(reflectivities), 'b-')  # Convert frequencies to GHz
    plt.title("Bode Diagram of Reflectivity vs Frequency")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Reflectivity (dB)")
    plt.xscale('log')
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sigma_opt, gain_min = minimize_reflectivity()
    print("Optimal sigma for minimum reflectivity at 1 GHz:", sigma_opt)
    plot_bode(calculate_reflectivity, sigmas=np.logspace(-7, 3, num=100), min_sigma = sigma_opt, min_gain = gain_min)
    plot_bode_frequency_range(sigma=sigma_opt, eps_r=1.4)
    print("End.")
    
