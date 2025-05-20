import numpy as np
import matplotlib.pyplot as plt
import math
from math import exp, isfinite
from scipy.integrate import quad, IntegrationWarning
from scipy.special import gamma, gammaln
import warnings

N = 100


def f_tau_analytic(n, t, k):
    return k * math.factorial(N) / (math.factorial(n) * math.factorial(N-n-1))\
        * (np.exp(-(n+1)*k*t))*(1-np.exp(-k*t))**(N-n-1)


def f_tau_gamma(t, k, n, N):
    if (k <= 0) or (n < 0) or (N <= n):
        print("Invalid input, k = ", k, " n = ", n, " N = ", N)
        return 0.0
    try:
        log_comb_factor = gammaln(
            N + 1.0) - (gammaln(n + 1.0) + gammaln(N - n))
        comb_factor = np.exp(log_comb_factor)
    except Exception as e:
        print("Error in computing comb_factor: ", e)
        print("Invalid input, t = ", t," k = ", k, " n = ", n, " N = ", N)
        return 0.0

    # Compute the base; clamp it to avoid raising zero to a positive exponent.
    base = 1.0 - np.exp(-k * t)
    base = max(base, 1e-15)  # Avoid zero or negative base
    exponent = (N - n - 1.0)
    if exponent < 0:
        return 0.0

    val = k * comb_factor * np.exp(-(n + 1.0) * k * t) * (base ** exponent)
    if (not isfinite(val)) or (val < 0):
        print("Invalid value: ", val)
        print("Invalid input, t = ", t," k = ", k, " n = ", n, " N = ", N)
        return 0.0
    return val

###############################################################################
# 2) Difference PDF: f_diff_gamma(x; k1,n1,N1, k2,n2,N2)
###############################################################################


def f_diff_gamma(x, k, n1, N1, n2, N2):
    lower_bound = max(0.0, x)

    def integrand(t):
        return f_tau_gamma(t, k, n1, N1) * f_tau_gamma(t - x, k, n2, N2)

    try:
        val, _ = quad(integrand, lower_bound, np.inf,
                      limit=300, epsabs=1e-8, epsrel=1e-8)
        if not np.isfinite(val) or val < 0:
            print("Invalid value in f_diff_gamma: ", val)
            print("Parameters: k =", k, "n1 =", n1, "N1 =", N1, "n2 =", n2, "N2 =", N2, "x =", x)
            return 0.0
        return val
    except (ValueError, OverflowError, IntegrationWarning) as e:
        print("Error in computing f_diff_gamma:", e)
        print("Parameters: k =", k, "n1 =", n1, "N1 =", N1, "n2 =", n2, "N2 =", N2, "x =", x)
        return 0.0


def f_diff_analytic(x, k, n1, N1, n2, N2):
    """
    Compute the difference PDF analytically using f_tau_analytic.
    """
    lower_bound = max(0.0, x)

    def integrand(t):
        return f_tau_analytic(n1, t, k) * f_tau_analytic(n2, t - x, k)

    try:
        val, _ = quad(integrand, lower_bound, np.inf,
                      limit=300, epsabs=1e-8, epsrel=1e-8)
        if not np.isfinite(val) or val < 0:
            print("Invalid value in f_diff_analytic: ", val)
            print("Invalid input, k = ", k, " n1 = ", n1, " N1 = ", N1,
                  " n2 = ", n2, " N2 = ", N2)
            return 0.0
        return val
    except (ValueError, OverflowError, IntegrationWarning):
        print("Error in computing f_diff_analytic")
        print("Invalid input, k = ", k, " n1 = ", n1, " N1 = ", N1,
              " n2 = ", n2, " N2 = ", N2)
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


def plot_compare_f_diff():
    """
    Plot and compare f_diff_gamma and f_diff_analytic.
    """
    x_values = np.linspace(-20, 20, 100)  # Range of x values
    k = 0.1  # Example degradation rate
    n1, N1 = 5, 100  # Parameters for Chromosome 1
    n2, N2 = 4, 100  # Parameters for Chromosome 2

    # Compute f_diff_gamma and f_diff_analytic
    f_diff_gamma_values = [f_diff_gamma(
        x, k, n1, N1, n2, N2) for x in x_values]
    f_diff_analytic_values = [f_diff_analytic(
        x, k, n1, N1, n2, N2) for x in x_values]

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, f_diff_gamma_values,
             label="f_diff_gamma", linestyle='-', color='blue')
    plt.plot(x_values, f_diff_analytic_values,
             label="f_diff_analytic", linestyle='--', color='red')
    plt.xlabel("x")
    plt.ylabel("f_diff")
    plt.title("Comparison of f_diff_gamma and f_diff_analytic")
    plt.legend()
    plt.grid()
    plt.show()

def generate_threshold_values(n0_mean, n0_total, num_simulations):
    # n01_list = np.random.normal(loc=n0_mean[0], scale=1, size=num_simulations)
    # n02_list = np.random.normal(loc=n0_mean[1], scale=1, size=num_simulations)
    # n03_list = n0_total - n01_list - n02_list


    # n01_list = np.floor(np.clip(n01_list, 0.01, n0_total))
    # n02_list = np.floor(np.clip(n02_list, 0.01, n0_total))
    # n03_list = np.floor(np.clip(n03_list, 0.01, n0_total))

    n01_list = n0_mean[0] * np.ones(num_simulations)
    n02_list = n0_mean[1] * np.ones(num_simulations)
    n03_list = n0_total - n01_list - n02_list
    n0_list = np.column_stack((n01_list, n02_list, n03_list))
    return n0_list


# if __name__ == "__main__":
#     plot_compare_f_diff()
