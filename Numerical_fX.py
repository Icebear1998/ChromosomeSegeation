import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import math
from scipy.special import digamma, polygamma
from scipy.stats import norm
from chromosome_Gillespie4 import ProteinDegradationSimulation, generate_threshold_values, run_simulations

# Define the PDF of separation time f_tau(t; k, n, N)


def f_tau(t, k, n, N):
    """
    Compute the PDF f_tau(t, theta) for a given t and parameters theta = (k, n, N).
    """
    if t < 0:
        return 0
    coeff = k * math.factorial(N) / (math.factorial(n)
                                     * math.factorial(N - n - 1))
    return coeff * np.exp(-(n + 1) * k * t) * (1 - np.exp(-k * t))**(N - n - 1)

# Compute the PDF of X = tau1 - tau2 using numerical convolution


def f_X_numerical(x, k1, n1, N1, k2, n2, N2):
    """
    Compute the PDF f_X(x) using numerical integration.
    """
    # Adjust the integration limits to ensure t - x >= 0
    def integrand(t): return f_tau(t, k1, n1, N1) * f_tau(t - x, k2, n2, N2)
    lower_limit = max(0, x)  # Ensure t >= x (since t - x >= 0)
    result, _ = integrate.quad(integrand, lower_limit, np.inf, limit=100)
    return result

# Define the moments of the gamma distribution


def mean_tau(k, n, N):
    return (digamma(N + 1) - digamma(n + 1)) / (k * (n + 1))


def var_tau(k, n, N):
    return (polygamma(1, n + 1) - polygamma(1, N + 1)) / (k**2 * (n + 1)**2)

# Theoretical PDF f_X(x)


def f_X_estimate(x, k1, n1, N1, k2, n2, N2):
    mu_tau1 = mean_tau(k1, n1, N1)
    mu_tau2 = mean_tau(k2, n2, N2)
    sigma_tau1_squared = var_tau(k1, n1, N1)
    sigma_tau2_squared = var_tau(k2, n2, N2)

    mu_X_theory = mu_tau1 - mu_tau2
    sigma_X_theory_squared = sigma_tau1_squared + sigma_tau2_squared

    coeff = 1 / np.sqrt(2 * np.pi * sigma_X_theory_squared)
    exponent = -0.5 * ((x - mu_X_theory)**2 / sigma_X_theory_squared)
    return coeff * np.exp(exponent)


# Define parameters for the two chromosomes
k1, n1, N1 = 0.1, 4, 100  # Parameters for tau1
k2, n2, N2 = 0.1, 3, 120  # Parameters for tau2

# Define the range of x values (difference in separation times)
x_values = np.linspace(-30, 30, 200)

# Compute f_X(x) using numerical integration over the range of x values
f_x_numerical_values = [f_X_numerical(
    x, k1, n1, N1, k2, n2, N2) for x in x_values]

# Normalize the PDF (ensure it integrates to 1)
f_x_numerical_values = np.array(f_x_numerical_values)
f_x_numerical_values /= np.trapz(f_x_numerical_values, x_values)

# Compute f_X(x) using the theoretical estimate over the range of x values
f_x_estimate_values = [f_X_estimate(
    x, k1, n1, N1, k2, n2, N2) for x in x_values]

# Normalize the PDF (ensure it integrates to 1)
f_x_estimate_values = np.array(f_x_estimate_values)
f_x_estimate_values /= np.trapz(f_x_estimate_values, x_values)

# Stochastic simulation using ProteinDegradationSimulation
initial_proteins1 = N1
initial_proteins2 = N2
initial_proteins3 = 350
initial_proteins = [initial_proteins1, initial_proteins2, initial_proteins3]
max_time = 150
num_simulations = 500

# Generate threshold values with Gaussian distribution
n0_total = 10
n01_mean = n1
n02_mean = n2
n03_mean = n0_total - n01_mean - n02_mean

n0_list = generate_threshold_values(
    [n01_mean, n02_mean], n0_total, num_simulations)

# Run simulations
simulations, seperate_times = run_simulations(
    initial_proteins, [k1, k2, k2], n0_list, max_time, num_simulations)

delta_t_list = np.array([[sep[0] - sep[1], sep[2] - sep[1]]
                        for sep in seperate_times])

# Extract the differences in separation times
delta_t1 = delta_t_list[:, 0]

# Plot the result
plt.figure(figsize=(10, 6))

# Plot the numerical integration result
plt.plot(x_values, f_x_numerical_values,
         label="Numerically Integrated PDF", color='blue')

# Plot the theoretical estimate result
plt.plot(x_values, f_x_estimate_values,
         label="Theoretical Estimate PDF", color='green', linestyle='--')

# Plot the stochastic simulation result
plt.hist(delta_t1, bins=30, density=True, alpha=0.5,
         label="Stochastic Simulation", color='orange')

# Add parameter annotations using axes coordinates
params_text = f"k1={k1:.2f}, n1={n1:.2f}, N1={N1:.2f}\n" \
              f"k2={k2:.2f}, n2={n2:.2f}, N2={N2:.2f}"
plt.text(0.05, 0.95, params_text, fontsize=10, transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

plt.xlabel("Difference in Separation Time (X = tau1 - tau2)")
plt.ylabel("Density")
plt.title("Comparison of Numerical Integration, Theoretical Estimate, and Stochastic Simulation")
plt.legend()
plt.grid()
plt.show()
