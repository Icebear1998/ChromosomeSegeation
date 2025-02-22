import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm
from scipy import integrate
import math

# Define the PDF of separation time f_tau(t; k, n, N)


def f_tau(t, k, n, N):
    if t < 0 or N - n - 1 < 0:
        return 0
    coeff = k * math.factorial(int(N)) / (math.factorial(int(n))
                                          * math.factorial(int(N - n - 1)))
    return coeff * np.exp(-(n + 1) * k * t) * (1 - np.exp(-k * t))**(N - n - 1)

# Compute the PDF of X = tau1 - tau2 using numerical convolution


def f_X_numerical(x, k1, n1, N1, k2, n2, N2):
    def integrand(t): return f_tau(t, k1, n1, N1) * f_tau(t - x, k2, n2, N2)
    lower_limit = max(0, x)
    result, _ = integrate.quad(integrand, lower_limit, np.inf, limit=100)
    return result

# Objective function for optimization using numerical f_X


def objective_numerical(params, n1, N1, experimental_data12, experimental_data32):
    k, r1, R1, r2, R2 = params

    # Chromosome 1 vs 2
    n2 = r1 * n1
    N2 = R1 * N1
    x_values = np.linspace(-30, 30, 200)
    try:
        f_x_numerical_values12 = np.array(
            [f_X_numerical(x, k, n1, N1, k, n2, N2) for x in x_values])
        f_x_numerical_values12 /= np.trapz(f_x_numerical_values12, x_values)
    except ValueError:
        return np.inf

    # Chromosome 2 vs 3
    n3 = r2 * n1
    N3 = R2 * N1
    try:
        f_x_numerical_values32 = np.array(
            [f_X_numerical(x, k, n3, N3, k, n2, N2) for x in x_values])
        f_x_numerical_values32 /= np.trapz(f_x_numerical_values32, x_values)
    except ValueError:
        return np.inf

    # Compute the likelihood of the experimental data given the numerical f_X
    likelihood12 = np.sum(np.interp(experimental_data12,
                          x_values, f_x_numerical_values12))
    likelihood32 = np.sum(np.interp(experimental_data32,
                          x_values, f_x_numerical_values32))

    # Return the negative log-likelihood (to minimize)
    return -np.log(likelihood12 + likelihood32)


# Read experimental data
df = pd.read_excel('Chromosome_diff.xlsx')
experimental_data_wt12 = df['SCSdiff_Wildtype12'].dropna().tolist()
experimental_data_wt32 = df['SCSdiff_Wildtype23'].dropna().tolist()

# Fixed parameters
n1 = 4  # Fixed value for n1
N1 = 100  # Fixed value for N1

# Parameter bounds for k, r1, R1, r2, R2
bounds = [(0.01, 0.5), (0.1, 5), (0.1, 1.5), (0.1, 5), (0.1, 1.5)]

# Global optimization
result_de = differential_evolution(
    objective_numerical,
    bounds=bounds,
    args=(n1, N1, experimental_data_wt12, experimental_data_wt32),
    strategy='best1bin',
    maxiter=2000,
    popsize=20,
    tol=1e-7
)

if result_de.success:
    print("Global optimization succeeded.")
else:
    print("Global optimization failed. Proceeding with fallback.")

# Perform local refinement
result_refined = minimize(
    objective_numerical,
    result_de.x,
    args=(n1, N1, experimental_data_wt12, experimental_data_wt32),
    bounds=bounds,
    method='L-BFGS-B',
    options={'maxiter': 1000, 'ftol': 1e-9}
)

if result_refined.success:
    params_optimized = result_refined.x
    print("Local refinement succeeded.")
else:
    params_optimized = result_de.x  # Fallback to global result
    print("Local refinement failed. Using global optimization results.")

k_est, r1_est, R1_est, r2_est, R2_est = params_optimized
n2_est = r1_est * n1
N2_est = round(R1_est * N1)
n3_est = r2_est * n1
N3_est = round(R2_est * N1)
print("Optimized parameters:", params_optimized)


def plot_results(experimental_data_wt12, experimental_data_wt32, k_est, n1_est, N1_est, n2_est, N2_est, n3_est, N3_est):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Chromosome 1 vs 2
    mu_exp, std_exp = norm.fit(experimental_data_wt12)
    ax[0].hist(experimental_data_wt12, bins=20, density=True,
               alpha=0.2, color='blue', label='Experimental Data')
    xmin, xmax = ax[0].get_xlim()
    x_values = np.linspace(xmin, xmax, 100)
    p_exp = norm.pdf(x_values, mu_exp, std_exp)
    ax[0].plot(x_values, p_exp, 'blue', linewidth=2,
               label='Experimental Gaussian Fit')
    f_x_numerical_values12 = np.array([f_X_numerical(
        xi, k_est, n1_est, N1_est, k_est, n2_est, N2_est) for xi in x_values])
    # Normalize
    f_x_numerical_values12 /= integrate.trapezoid(
        f_x_numerical_values12, x_values)
    ax[0].plot(x_values, f_x_numerical_values12, 'red',
               alpha=0.5, linewidth=2, label='Numerical f_X(x)')
    ax[0].set_xlabel(
        "Difference in Separation Time (Chromosome 1 - Chromosome 2)")
    ax[0].set_ylabel("Density")
    ax[0].set_title("Chromosome 1 vs 2")
    params_text_12 = f"k1={k_est:.2f}, n1={n1_est:.2f}, N1={N1_est:.2f}\n" \
                     f"k2={k_est:.2f}, n2={n2_est:.2f}, N2={N2_est:.2f}"
    ax[0].text(0.05, 0.95, params_text_12, transform=ax[0].transAxes,
               fontsize=10, verticalalignment='top')
    ax[0].legend()
    ax[0].grid()

    # Chromosome 2 vs 3
    mu_exp, std_exp = norm.fit(experimental_data_wt32)
    ax[1].hist(experimental_data_wt32, bins=20, density=True,
               alpha=0.2, color='blue', label='Experimental Data')
    xmin, xmax = ax[1].get_xlim()
    x_values = np.linspace(xmin, xmax, 100)
    p_exp = norm.pdf(x_values, mu_exp, std_exp)
    ax[1].plot(x_values, p_exp, 'blue', linewidth=2,
               label='Experimental Gaussian Fit')
    f_x_numerical_values32 = np.array([f_X_numerical(
        xi, k_est, n3_est, N3_est, k_est, n2_est, N2_est) for xi in x_values])
    # Normalize
    f_x_numerical_values32 /= integrate.trapezoid(
        f_x_numerical_values32, x_values)
    ax[1].plot(x_values, f_x_numerical_values32, 'red',
               alpha=0.5, linewidth=2, label='Numerical f_X(x)')
    ax[1].set_xlabel(
        "Difference in Separation Time (Chromosome 3 - Chromosome 2)")
    ax[1].set_ylabel("Density")
    ax[1].set_title("Chromosome 3 vs 2")
    params_text_32 = f"k2={k_est:.2f}, n2={n2_est:.2f}, N2={N2_est:.2f}\n" \
                     f"k3={k_est:.2f}, n3={n3_est:.2f}, N3={N3_est:.2f}"
    ax[1].text(0.05, 0.95, params_text_32, transform=ax[1].transAxes,
               fontsize=10, verticalalignment='top')
    ax[1].legend()
    ax[1].grid()

    plt.tight_layout()
    plt.show()


# Call the plot_results function
plot_results(experimental_data_wt12, experimental_data_wt32,
             k_est, n1, N1, n2_est, N2_est, n3_est, N3_est)
