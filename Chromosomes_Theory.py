import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad, IntegrationWarning
from scipy.special import gamma
import warnings

N = 100


def f_tau_analytic(n, t, k):
    return k * math.factorial(N) / (np.math.factorial(n) * np.math.factorial(N-n-1))\
        * (np.exp(-(n+1)*k*t))*(1-np.exp(-k*t))**(N-n-1)


def f_tau(t, k, n, N):
    if t < 0:
        return 0.0

    try:
        comb_factor = gamma(N + 1.0) / (gamma(n + 1.0) * gamma(N - n))
        val = k * comb_factor * \
            np.exp(-(n + 1) * k * t) * (1 - np.exp(-k * t))**(N - n - 1)
    except (OverflowError, ZeroDivisionError, ValueError) as e:
        print(f"Error in f_tau: {e}, t={t}, k={k}, n={n}, N={N}")
        val = 0.0

    return val


def f_diff(z, k, n1, N1, n2, N2):
    """
    Compute the difference distribution f_X(z) for given parameters.
    """
    comb_factor = (N1**n1) * (N2**n2) / (gamma(n1+1) * gamma(n2+1))
    
    def integrand(t):
        try:
            exp_term = np.exp(-t) * np.exp(-k*t*z)
            if np.isinf(exp_term) or np.isnan(exp_term):
                return 0.0
            return t**(n1+n2-1) * exp_term
        except OverflowError:
            return 0.0
    
    lower_limit = 0
    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=IntegrationWarning)
        try:
            val, _ = quad(integrand, lower_limit, np.inf, epsabs=1e-8, epsrel=1e-8)
        except IntegrationWarning:
            val = np.nan
        except Exception as e:
            val = np.nan
    
    if np.isnan(val):
        return 0.0
    
    return k * comb_factor * val


def plot_chromosomes():
    t = np.linspace(0, 60, 100)
    k_values = [0.1, 0.2, 0.3, 0.4]  # Choose 4 values of k

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()  # Flatten the 2D array of axes for easy iteration
    for i, k in enumerate(k_values):
        for n in range(3, 10):
            axs[i].plot(t, f_tau_analytic(n, t, k), label=f'n={n}')
        axs[i].legend()
        axs[i].set_title(f'Plot for k={k}')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_chromosomes()
