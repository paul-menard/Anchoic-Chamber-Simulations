import numpy as np 
import scipy.io
import scipy.optimize
import matplotlib.pyplot as plt


def real_to_complex(z):
    return z[0] + 1j * z[1]


def complex_to_real(z):
    return np.array([np.real(z), np.imag(z)])

class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.f(*args)
        # .. todo: deepcopy here if returning objects
        return self.memo[args]

def compute_alpha(omega, eps_r, mu_r, sigma,  c_0):
    """
    .. warning: $w = 2 \pi f$
    w is called circular frequency
    f is called frequency
    """

    # parameters of the geometry
    L = 0.01 # length of the layer

    # parameters of the mesh
    resolution = 12  # := number of elements along L

    # parameters of the material (cont.)
    mu_0 = 4*np.pi*1e-7
    ksi_0 = 1.0 / (c_0 ** 2)
    mu_1 = mu_0 * mu_r
    ksi_1 = eps_r / (c_0 ** 2)
    a = sigma * mu_0

    """
    ksi_volume = phi * gamma_p / (c_0 ** 2)
    a_volume = sigma * (phi ** 2) * gamma_p / ((c_0 ** 2) * rho_0 * alpha_h)
    mu_volume = phi / alpha_h
    k2_volume = (1.0 / mu_volume) * ((omega ** 2) / (c_0 ** 2)) * (ksi_volume + 1j * a_volume / omega)
    #print(k2_volume)
    """

    # parameters of the objective function
    A = 1.0
    B = 1.0

    # defining k, omega and alpha dependant parameters' functions
    @Memoize
    def lambda_0(k, omega):
        if k ** 2 >= (omega ** 2) * ksi_0 / mu_0:
            return np.sqrt(k ** 2 - (omega ** 2) * ksi_0 / mu_0)
        else:
            return np.sqrt((omega ** 2) * ksi_0 / mu_0 - k ** 2) * 1j

    @Memoize
    def lambda_1(k, omega):
        temp1 = (omega ** 2) * ksi_1 / mu_1
        temp2 = np.sqrt((k ** 2 - temp1) ** 2 + (a * omega / mu_1) ** 2)
        real = (1.0 / np.sqrt(2.0)) * np.sqrt(k ** 2 - temp1 + temp2)
        im = (-1.0 / np.sqrt(2.0)) * np.sqrt(temp1 - k ** 2 + temp2)
        return complex(real, im)

    @Memoize
    def g(y):
        # ..warning: not validated ***********************
        return 1.0

    @Memoize
    def g_k(k):
        # ..warning: not validated ***********************
        if k == 0:
            return 1.0
        else:
            return 0.0

    @Memoize
    def f(x, k):
        return ((lambda_0(k, omega) * mu_0 - x) * np.exp(-lambda_0(k, omega) * L) \
                + (lambda_0(k, omega) * mu_0 + x) * np.exp(lambda_0(k, omega) * L))

    @Memoize
    def chi(k, alpha, omega):
        return (g_k(k) * ((lambda_0(k, omega) * mu_0 - lambda_1(k, omega) * mu_1) \
                          / f(lambda_1(k, omega) * mu_1, k) - (lambda_0(k, omega) * mu_0 - alpha) / f(alpha, k)))

    @Memoize
    def eta(k, alpha, omega):
        return (g_k(k) * ((lambda_0(k, omega) * mu_0 + lambda_1(k, omega) * mu_1) \
                          / f(lambda_1(k, omega) * mu_1, k) - (lambda_0(k, omega) * mu_0 + alpha) / f(alpha, k)))

    @Memoize
    def e_k(k, alpha, omega):
        expm = np.exp(-2.0 * lambda_0(k, omega) * L)
        expp = np.exp(+2.0 * lambda_0(k, omega) * L)

        if k ** 2 >= (omega ** 2) * ksi_0 / mu_0:
            return ((A + B * (np.abs(k) ** 2)) \
                    * ( \
                                (1.0 / (2.0 * lambda_0(k, omega))) \
                                * ((np.abs(chi(k, alpha, omega)) ** 2) * (1.0 - expm) \
                                   + (np.abs(eta(k, alpha, omega)) ** 2) * (expp - 1.0)) \
                                + 2 * L * np.real(chi(k, alpha, omega) * np.conj(eta(k, alpha, omega)))) \
                    + B * np.abs(lambda_0(k, omega)) / 2.0 * ((np.abs(chi(k, alpha, omega)) ** 2) * (1.0 - expm) \
                                                                 + (np.abs(eta(k, alpha, omega)) ** 2) * (
                                                                             expp - 1.0)) \
                    - 2 * B * (lambda_0(k, omega) ** 2) * L * np.real(
                        chi(k, alpha, omega) * np.conj(eta(k, alpha, omega))))
        else:
            return ((A + B * (np.abs(k) ** 2)) * (L \
                                                     * ((np.abs(chi(k, alpha, omega)) ** 2) + (
                                np.abs(eta(k, alpha, omega)) ** 2)) \
                                                     + complex(0.0, 1.0) * (1.0 / lambda_0(k, omega)) * np.imag(
                        chi(k, alpha, omega) * np.conj(eta(k, alpha, omega) \
                                                          * (1.0 - expm))))) + B * L * (
                               np.abs(lambda_0(k, omega)) ** 2) \
                   * ((np.abs(chi(k, alpha, omega)) ** 2) + (np.abs(eta(k, alpha, omega)) ** 2)) \
                   + complex(0.0, 1.0) * B * lambda_0(k, omega) * np.imag(
                chi(k, alpha, omega) * np.conj(eta(k, alpha, omega) \
                                                  * (1.0 - expm)))

    @Memoize
    def sum_e_k(omega):
        def sum_func(alpha):
            s = 0.0
            for n in range(-resolution, resolution + 1):
                k = n * np.pi / L
                s += e_k(k, alpha, omega)
            return s

        return sum_func

    @Memoize
    def alpha(omega):
        alpha_0 = np.array(complex(40.0, -40.0))
        temp = real_to_complex(scipy.optimize.minimize(lambda z: np.real(sum_e_k(omega)(real_to_complex(z))), complex_to_real(alpha_0), tol=1e-4).x)
        print(temp, "------", "je suis temp")
        return temp

    @Memoize
    def error(alpha, omega):
        temp = np.real(sum_e_k(omega)(alpha))
        return temp

    temp_alpha = alpha(omega)
    temp_error = error(temp_alpha, omega)

    return temp_alpha, temp_error


def run_compute_alpha(material):
    print('Computing alpha...')
    numb_omega = 1000  # 1000
    # omegas = np.logspace(np.log10(600), np.log10(30000), num=numb_omega)
    omegas = np.linspace(2.0 * np.pi, np.pi * 10000, num=numb_omega)
    temp = [compute_alpha(omega, material=material) for omega in omegas]
    print("temp:", "------", temp)
    alphas, errors = map(list, zip(*temp))
    alphas = np.array(alphas)
    errors = np.array(errors)

    print('Writing alpha...')
    output_filename = 'dta_omega_' + str(material) + '.mtx'
    scipy.io.mmwrite(output_filename, omegas.reshape(alphas.shape[0], 1), field='complex', symmetry='general')
    output_filename = 'dta_alpha_' + str(material) + '.mtx'
    scipy.io.mmwrite(output_filename, alphas.reshape(alphas.shape[0], 1), field='complex', symmetry='general')
    output_filename = 'dta_error_' + str(material) + '.mtx'
    scipy.io.mmwrite(output_filename, errors.reshape(errors.shape[0], 1), field='complex', symmetry='general')

    return


def run_plot_alpha(material):
    color = 'darkblue'

    print('Reading alpha...')
    input_filename = 'dta_omega_' + str(material) + '.mtx'
    omegas = scipy.io.mmread(input_filename)
    omegas = omegas.reshape(omegas.shape[0])
    input_filename = 'dta_alpha_' + str(material) + '.mtx'
    alphas = scipy.io.mmread(input_filename)
    alphas = alphas.reshape(alphas.shape[0])
    input_filename = 'dta_error_' + str(material) + '.mtx'
    errors = scipy.io.mmread(input_filename)
    errors = errors.reshape(errors.shape[0])

    print('Plotting alpha...')
    fig = plt.figure()
    plt.subplot(1, 1, 1)
    plt.plot(np.real(omegas), np.real(alphas), color=color)
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\operatorname{Re}(\alpha)$')
    # plt.show()
    plt.savefig('fig_alpha_real_' + str(material) + '.jpg')
    plt.close(fig)

    fig = plt.figure()
    plt.subplot(1, 1, 1)
    plt.plot(np.real(omegas), np.imag(alphas), color=color)
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\operatorname{Im}(\alpha)$')
    # plt.show()
    plt.savefig('fig_alpha_imag_' + str(material) + '.jpg')
    plt.close(fig)

    fig = plt.figure()
    ax = plt.axes()
    ax.fill_between(np.real(omegas), np.real(errors), color=color)
    plt.ylim(1.e-9, 1.e-4)
    plt.yscale('log')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$e(\alpha)$')
    # plt.show()
    plt.savefig('fig_error_' + str(material) + '.jpg')
    plt.close(fig)

    return


def run():
    print('Running compute_alpha...')
    omega = 30000 # angular frequency
    eps_r = 1.0    # relative permittivity
    mu_r = 1.0     # relative permeability
    sigma = 0.01   # conductivity
    c_0 = 3e8      # speed of light in vacuum
    print('Parameters set.')
    alpha, err_alpha = compute_alpha(omega, eps_r, mu_r , sigma=sigma, c_0=c_0)
    print('alpha = ', alpha)
    print('err_alpha = ', err_alpha)
    return 


if __name__ == '__main__':
    run()
    print('End.')