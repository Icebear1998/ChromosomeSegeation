import numpy as np
import matplotlib.pyplot as plt
import math
from math import exp, isfinite
from scipy.integrate import quad, IntegrationWarning
from scipy.special import gamma, gammaln
import warnings

N = 100


def f_tau_analytic(n, t, k):
    return k * math.factorial(N) / (np.math.factorial(n) * np.math.factorial(N-n-1))\
        * (np.exp(-(n+1)*k*t))*(1-np.exp(-k*t))**(N-n-1)


def f_tau_gamma(t, k, n, N):
    """
    Computes the PDF for the (n+1)-th event among N exponentials at rate k,
    using a log-domain computation for the combinatorial factor:
    
      f_tau(t) = k * exp( gammaln(N+1) - gammaln(n+1) - gammaln(N-n) )
                 * exp( -(n+1)*k*t )
                 * [1 - exp(-k*t)]^(N-n-1)
                 
    Returns 0.0 if any input is out of domain or if numerical issues arise.
    """
    if t < 0:
        return 0.0
    if (k <= 0) or (n < 0) or (N <= n):
        return 0.0
    try:
        log_comb_factor = gammaln(N + 1.0) - (gammaln(n + 1.0) + gammaln(N - n))
        comb_factor = np.exp(log_comb_factor)
    except Exception as e:
        # If there's an error in computing gammaln, return 0.
        return 0.0
    
    # Compute the base; clamp it to avoid raising zero to a positive exponent.
    base = 1.0 - np.exp(-k*t)
    if base < 1e-15:
        base = 1e-15
    exponent = (N - n - 1.0)
    if exponent < 0:
        return 0.0

    val = k * comb_factor * np.exp(-(n + 1.0)*k*t) * (base ** exponent)
    if (not isfinite(val)) or (val < 0):
        return 0.0
    return val

###############################################################################
# 2) Difference PDF: f_diff_gamma(x; k1,n1,N1, k2,n2,N2)
###############################################################################
def f_diff(x, k, n1, N1, n2, N2):
    """
    Computes the difference PDF f_X(x) via numerical convolution:
      f_X(x) = ∫ f_tau_gamma(t; k1, n1, N1) * f_tau_gamma(t-x; k2, n2, N2) dt,
    integrated from t = max(0, x) to ∞.
    """
    lower_bound = max(0.0, x)
    def integrand(t):
        return f_tau_gamma(t, k, n1, N1) * f_tau_gamma(t - x, k, n2, N2)
    try:
        val, _ = quad(integrand, lower_bound, np.inf, limit=300)
        if (not np.isfinite(val)) or (val < 0):
            return 0.0
        return val
    except (ValueError, OverflowError):
        return 0.0


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
