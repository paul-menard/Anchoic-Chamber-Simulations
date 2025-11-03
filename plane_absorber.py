import matplotlib.pyplot as plt
import numpy as np
import scipy 

def plot_bode(G, sigmas, eps_r = 1.4, omega=2 * np.pi * 1e9):
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
    
    # 1. Calculer le terme complexe sous la racine carrée :
    #    eps_r_complexe_normalisée = eps_r - i * (sigma / (eps_0 * omega))
    complex_term = eps_r - 1j * (sigma / (eps_0 * omega))
    
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
    
    return R


if __name__ == "__main__":
    plot_bode(calculate_reflectivity, sigmas=np.logspace(-4, 5, num=100))
    print("End.")
    
