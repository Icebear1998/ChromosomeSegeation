import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import math
from scipy.special import gamma  # for the gamma-based coefficients
from Chromosome_Gillespie4 import ProteinDegradationSimulation, generate_threshold_values, run_simulations

###############################################################################
# 1) Factorial-based single-chromosome PDF f_tau(t; k, n, N)
###############################################################################
def f_tau_factorial(t, k, n, N):
    """
    PDF for the (n+1)-th event among N exponential processes, using factorial-based coefficient.
    Valid for integer n and N with N>n+1.
    """
    if t < 0:
        return 0.0
    # Factorial-based coefficient
    coeff = k * math.factorial(N) / (math.factorial(n) * math.factorial(N - n - 1))
    return coeff * np.exp(-(n + 1) * k * t) * (1 - np.exp(-k * t))**(N - n - 1)

###############################################################################
# 2) Gamma-based single-chromosome PDF f_tau_gamma(t; k, n, N)
###############################################################################
def f_tau_gamma(t, k, n, N):
    """
    PDF for the (n+1)-th event among N exponential processes, using Gamma-based coefficient.
    Matches the factorial version when n, N are integers, but also works for non-integer n, N.
    """
    if t < 0:
        return 0.0
    try:
        # Generalized combinatorial factor via Gamma
        comb_factor = gamma(N + 1.0) / (gamma(n + 1.0) * gamma(N - n))
    except ValueError:
        return 0.0
    
    val = k * comb_factor \
          * np.exp(-(n + 1.0) * k * t) \
          * (1.0 - np.exp(-k * t))**( (N - n - 1.0) )
    return val if val > 0 else 0.0

###############################################################################
# 3) Difference PDF f_X via numerical convolution (factorial-based single-chromosome PDFs)
###############################################################################
def f_X_numerical(x, k1, n1, N1, k2, n2, N2):
    """
    Convolution integral using the factorial-based PDF for each chromosome.
    f_X(x) = ∫ f_tau1(t) * f_tau2(t - x) dt, from t=max(0, x) to ∞.
    """
    def integrand(t):
        return f_tau_factorial(t, k1, n1, N1) * f_tau_factorial(t - x, k2, n2, N2)
    lower_limit = max(0.0, x)
    result, _ = integrate.quad(integrand, lower_limit, np.inf, limit=100)
    return result

###############################################################################
# 4) Difference PDF f_X via Gamma-based single-chromosome PDFs
###############################################################################
def f_X_gamma(x, k1, n1, N1, k2, n2, N2):
    """
    Convolution integral using the Gamma-based PDF for each chromosome.
    Matches factorial if n1, N1, n2, N2 are integer, but also handles real values.
    """
    def integrand(t):
        return f_tau_gamma(t, k1, n1, N1) * f_tau_gamma(t - x, k2, n2, N2)
    lower_limit = max(0.0, x)
    result, _ = integrate.quad(integrand, lower_limit, np.inf, limit=100)
    return result

###############################################################################
# 5) Parameters for two chromosomes
###############################################################################
k1, n1, N1 = 0.1, 4, 100   # Chromosome 1
k2, n2, N2 = 0.1, 3, 120   # Chromosome 2

###############################################################################
# 6) Create a range of x-values and compute f_X using:
#    (a) Factorial-based numerical integration
#    (b) Gamma-based numerical integration
###############################################################################
x_values = np.linspace(-30, 30, 200)

# (a) Factorial-based difference PDF
f_x_factorial_vals = [f_X_numerical(x, k1, n1, N1, k2, n2, N2) for x in x_values]
f_x_factorial_vals = np.array(f_x_factorial_vals)

# Normalize so that integral over x_values is 1
f_x_factorial_vals /= np.trapz(f_x_factorial_vals, x_values)

# (b) Gamma-based difference PDF
f_x_gamma_vals = [f_X_gamma(x, k1, n1, N1, k2, n2, N2) for x in x_values]
f_x_gamma_vals = np.array(f_x_gamma_vals)

# Normalize
f_x_gamma_vals /= np.trapz(f_x_gamma_vals, x_values)

###############################################################################
# 7) Stochastic simulation setup using the imported Gillespie-like module
###############################################################################
initial_proteins1 = N1
initial_proteins2 = N2
initial_proteins3 = 350
initial_proteins = [initial_proteins1, initial_proteins2, initial_proteins3]
max_time = 150
num_simulations = 1000

# Generate threshold values with Gaussian distribution
n0_total = 10
n01_mean = n1
n02_mean = n2
n03_mean = n0_total - n01_mean - n02_mean  # not used if only 2 thresholds?
# n0_list = generate_threshold_values([n01_mean, n02_mean], n0_total, num_simulations)
n0_list = np.array([[n01_mean, n02_mean, n03_mean] for _ in range(num_simulations)])

# Run simulations
simulations, separate_times = run_simulations(
    initial_proteins, [k1, k2, k2], n0_list, max_time, num_simulations
)

# Extract the differences: X = tau1 - tau2 from the result
delta_t_list = np.array([[sep[0] - sep[1], sep[2] - sep[1]] for sep in separate_times])
delta_t1 = delta_t_list[:, 0]  # differences for Chrom1 - Chrom2

###############################################################################
# 8) Plot all three results together:
#    - Factorial-based f_X
#    - Gamma-based f_X
#    - Stochastic simulation histogram
###############################################################################
plt.figure(figsize=(10, 6))

# (a) Factorial-based curve
plt.plot(x_values, f_x_factorial_vals, label="Factorial-based PDF", color='blue')

# (b) Gamma-based curve
plt.plot(x_values, f_x_gamma_vals, label="Gamma-based PDF", color='green', linestyle='--')

# (c) Stochastic simulation histogram
plt.hist(delta_t1, bins=30, density=True, alpha=0.5, label="Stochastic Simulation", color='orange')

# Add parameter annotations
params_text = (
    f"k1={k1:.2f}, n1={n1:.2f}, N1={N1:.2f}\n"
    f"k2={k2:.2f}, n2={n2:.2f}, N2={N2:.2f}"
)
plt.text(0.05, 0.95, params_text, fontsize=10, transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

plt.xlabel("Difference in Separation Time (X = tau1 - tau2)")
plt.ylabel("Density")
plt.title("Comparison: Factorial, Gamma-based, and Stochastic Simulation")
plt.legend()
plt.grid()
plt.show()
