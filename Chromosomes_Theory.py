import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad
from scipy.special import gamma

N = 100


def f_tau_analytic(n, t, k):
    return k * math.factorial(N) / (np.math.factorial(n) * np.math.factorial(N-n-1))\
        * (np.exp(-(n+1)*k*t))*(1-np.exp(-k*t))**(N-n-1)

def f_tau_gamma(t, k, n, N):
    if t < 0:
        return 0.0
    # Check domain for gamma arguments
    if (N < n) or (n < 0):
        return 0.0
    try:
        comb_factor = gamma(N + 1.0) / (gamma(n + 1.0) * gamma(N - n))
    except (ValueError, OverflowError):
        return 0.0
    
    val = k * comb_factor \
          * np.exp(-(n+1.0)*k*t) \
          * (1.0 - np.exp(-k*t))**( (N - n - 1.0) )
    if not np.isfinite(val) or (val < 0):
        return 0.0
    return val

def f_diff(z, k, n1, N1, n2, N2):
    """
    Difference PDF: X = tau1 - tau2
    f_X(z) = ∫ f_tau1(t)*f_tau2(t-z) dt, from t=max(0,z) to ∞.
    """
    lower_limit = max(0.0, z)
    
    def integrand(t):
        return f_tau_gamma(t, k, n1, N1) * f_tau_gamma(t - z, k, n2, N2)
    
    # Attempt integration; if it fails, we return a small number or 0
    try:
        val, _ = quad(integrand, lower_limit, np.inf, epsabs=1e-8, epsrel=1e-8)
    except (ValueError, OverflowError):
        return 0.0
    if not np.isfinite(val):
        return 0.0
    return val

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
