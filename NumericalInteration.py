import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import digamma, polygamma
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm
from scipy import integrate

# Define the moments of the gamma distribution


def mean_tau(k, n, N):
    return (digamma(N + 1) - digamma(n + 1)) / (k * (n + 1))


def var_tau(k, n, N):
    return (polygamma(1, n + 1) - polygamma(1, N + 1)) / (k**2 * (n + 1)**2)

# Theoretical PDF f_X(x)


def f_X(x, k1, n1, N1, k2, n2, N2):
    mu_tau1 = mean_tau(k1, n1, N1)
    mu_tau2 = mean_tau(k2, n2, N2)
    sigma_tau1_squared = var_tau(k1, n1, N1)
    sigma_tau2_squared = var_tau(k2, n2, N2)

    mu_X_theory = mu_tau1 - mu_tau2
    sigma_X_theory_squared = sigma_tau1_squared + sigma_tau2_squared

    coeff = 1 / np.sqrt(2 * np.pi * sigma_X_theory_squared)
    exponent = -0.5 * ((x - mu_X_theory)**2 / sigma_X_theory_squared)
    return coeff * np.exp(exponent)

# Objective function for simultaneous optimization


def objective_weighted(params, n1, N1, mu_X_exp12, sigma_X_exp12, mu_X_exp32, sigma_X_exp32, w_mu, w_sigma):
    k, r1, R1, r2, R2 = params

    # Chromosome 1 vs 2
    n2 = r1 * n1
    N2 = R1 * N1
    mu_tau1 = mean_tau(k, n1, N1)
    mu_tau2 = mean_tau(k, n2, N2)
    sigma_tau1_squared = var_tau(k, n1, N1)
    sigma_tau2_squared = var_tau(k, n2, N2)
    mu_X_theory12 = mu_tau1 - mu_tau2
    sigma_X_theory12 = np.sqrt(sigma_tau1_squared + sigma_tau2_squared)

    # Chromosome 2 vs 3
    n3 = r2 * n1
    N3 = R2 * N1
    mu_tau3 = mean_tau(k, n3, N3)
    sigma_tau3_squared = var_tau(k, n3, N3)
    mu_X_theory32 = mu_tau3 - mu_tau2
    sigma_X_theory32 = np.sqrt(sigma_tau2_squared + sigma_tau3_squared)

    # Errors
    error_mu_12 = w_mu * (mu_X_theory12 - mu_X_exp12)**2
    error_sigma_12 = w_sigma * (sigma_X_theory12 - sigma_X_exp12)**2
    error_mu_32 = w_mu * (mu_X_theory32 - mu_X_exp32)**2
    error_sigma_32 = w_sigma * (sigma_X_theory32 - sigma_X_exp32)**2

    return error_mu_12 + error_sigma_12 + error_mu_32 + error_sigma_32


# Read experimental data
df = pd.read_excel('Chromosome_diff.xlsx')
experimental_data_wt12 = df['SCSdiff_Wildtype12'].dropna().tolist()
experimental_data_wt32 = df['SCSdiff_Wildtype23'].dropna().tolist()

# Experimental moments
mu_X_exp12 = np.mean(experimental_data_wt12)
sigma_X_exp12 = np.std(experimental_data_wt12)
mu_X_exp32 = np.mean(experimental_data_wt32)
sigma_X_exp32 = np.std(experimental_data_wt32)

# Fixed parameters
n1 = 4  # Fixed value for n1
N1 = 100  # Fixed value for N1

# Parameter bounds for k, r1, R1, r2, R2
bounds = [(0.01, 0.5), (0.1, 5), (0.1, 5), (0.1, 5), (0.1, 5)]

# Global optimization
result_de = differential_evolution(
    objective_weighted,
    bounds=bounds,
    args=(n1, N1, mu_X_exp12, sigma_X_exp12,
          mu_X_exp32, sigma_X_exp32, 1.0, 1.0),
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
    objective_weighted,
    result_de.x,
    args=(n1, N1, mu_X_exp12, sigma_X_exp12,
          mu_X_exp32, sigma_X_exp32, 1.0, 1.0),
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
    f_x_values12 = np.array(
        [f_X(xi, k_est, n1_est, N1_est, k_est, n2_est, N2_est) for xi in x_values])
    f_x_values12 /= integrate.trapezoid(f_x_values12, x_values)  # Normalize
    ax[0].plot(x_values, f_x_values12, 'red', alpha=0.5,
               linewidth=2, label='Estimated f_X(x)')
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
    f_x_values32 = np.array(
        [f_X(xi, k_est, n3_est, N3_est, k_est, n2_est, N2_est) for xi in x_values])
    f_x_values32 /= integrate.trapezoid(f_x_values32, x_values)  # Normalize
    ax[1].plot(x_values, f_x_values32, 'red', alpha=0.5,
               linewidth=2, label='Estimated f_X(x)')
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
