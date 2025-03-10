from Chromosome_Gillespie4 import run_simulations, generate_threshold_values
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import quad
from math import exp, isfinite
from scipy.special import gamma
from Chromosomes_Theory import f_diff

# If your Chrom3–Chrom2 data is actually Chrom2–Chrom3, set True:
FLIP_CHROM3_DATA = False

###############################################################################
# 3) Objective function (Chrom1–Chrom2 + Chrom3–Chrom2),
#    but (R1, R2) are externally fixed. We optimize over (k, r1, r2).
###############################################################################


def combined_objective(vars_, R21_fixed, R23_fixed, data12, data32, x_grid):
    n2, N2, k, r21, r23 = vars_

    # Ensure n2 and N2 are integers
    n2 = int(round(n2))
    N2 = int(round(N2))

    n1 = int(round(r21 * n2))
    N1 = int(round(R21_fixed * N2))
    n3 = int(round(r23 * n2))
    N3 = int(round(R23_fixed * N2))

    # Chrom1–Chrom2
    pdf12 = np.array([
        f_diff(x, k, n1, N1, n2, N2)
        for x in x_grid
    ])
    area12 = np.trapz(pdf12, x_grid)
    if area12 < 1e-15:
        return np.inf
    pdf12 /= area12
    vals12 = np.interp(data12, x_grid, pdf12, left=0, right=0)
    if np.any(vals12 <= 0):
        return np.inf

    # Chrom3–Chrom2
    pdf32 = np.array([
        f_diff(x, k, n3, N3, n2, N2)
        for x in x_grid
    ])
    area32 = np.trapz(pdf32, x_grid)
    if area32 < 1e-15:
        return np.inf
    pdf32 /= area32
    vals32 = np.interp(data32, x_grid, pdf32, left=0, right=0)
    if np.any(vals32 <= 0):
        return np.inf

    # Negative log-likelihood
    return -np.sum(np.log(vals12)) - np.sum(np.log(vals32))

###############################################################################
# 4) Main "Hybrid" approach
###############################################################################


def main():
    #  a) Read data
    df = pd.read_excel("Data/Chromosome_diff.xlsx")
    data12 = df['SCSdiff_Wildtype12'].dropna().values
    data32 = df['SCSdiff_Wildtype23'].dropna().values

    # If your data for Chrom3–Chrom2 is actually (Chrom2–Chrom3), flip sign:
    if FLIP_CHROM3_DATA:
        data32 = -data32

    # c) Prepare a broad x_grid
    x_grid = np.linspace(-100, 100, 301)

    # d) We'll define small discrete sets for R1, R2
    # e.g. 0.50, 0.75, 1.00, ...
    R21_candidates = np.round(np.arange(0.25, 2.51, 0.25), 2)
    # e.g. 0.50, 1.00, 1.50, ...
    R23_candidates = np.round(np.arange(0.25, 5.01, 0.25), 2)

    # e) We'll do local optimization over [k, r1, r2] for each grid cell (R1,R2)
    param_bounds = [
        (5, 20),     # n2
        (80, 200),   # N2
        (0.01, 0.3),  # k
        (0.5,  2.0),  # r1
        (0.5,  2.0),  # r2
    ]

    best_obj = np.inf
    best_solution = None

    for R21_fixed in R21_candidates:
        for R23_fixed in R23_candidates:
            # Define a local objective that only depends on [k, r1, r2],
            # with R1_fixed, R2_fixed closed over.
            def local_obj_k_r1_r2(vars_):
                return combined_objective(
                    vars_, R21_fixed, R23_fixed, data12, data32, x_grid
                )

            # pick an initial guess for [n2, N2, k, r1, r2]
            x0 = [4, 100, 0.1, 1.0, 1.0]

            # local optimization
            res = minimize(
                local_obj_k_r1_r2,
                x0,
                method='L-BFGS-B',
                bounds=param_bounds,
                options={'maxiter': 300, 'disp': False}
            )

            if res.success and res.fun < best_obj:
                best_obj = res.fun
                # store the combined solution: (n2, N2, k, r1, r2, R1_fixed, R2_fixed)
                best_solution = (res.x[0], res.x[1], res.x[2],
                                 res.x[3], res.x[4], R21_fixed, R23_fixed)

    if best_solution is None:
        print("No valid solution found in the (R1, R2) grid.")
        return

    n2_opt, N2_opt, k_opt, r21_opt, r23_opt, R21_opt, R23_opt = best_solution
    print("Best negative log-likelihood:", best_obj)
    print(f"Best parameters:\n"
          f"  n2={n2_opt:.2f}, N2={N2_opt:.2f}\n"
          f"  k={k_opt:.4f}, r1={r21_opt:.4f}, R1={R21_opt:.2f}\n"
          f"  r2={r23_opt:.4f}, R2={R23_opt:.2f}")

    # f) Evaluate final PDF & plot
    n1_opt = r21_opt * n2_opt
    N1_opt = R21_opt * N2_opt
    n3_opt = r23_opt * n2_opt
    N3_opt = R23_opt * N2_opt

    pdf12 = np.array([
        f_diff(x, k_opt, n1_opt, N1_opt, n2_opt, N2_opt)
        for x in x_grid
    ])
    area12 = np.trapz(pdf12, x_grid)
    if area12 > 1e-15:
        pdf12 /= area12

    pdf32 = np.array([
        f_diff(x, k_opt, n3_opt, N3_opt, n2_opt, N2_opt)
        for x in x_grid
    ])
    area32 = np.trapz(pdf32, x_grid)
    if area32 > 1e-15:
        pdf32 /= area32

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].hist(data12, bins=30, density=True, alpha=0.4, label='data12')
    ax[0].plot(x_grid, pdf12, 'r-', label='model pdf12')
    ax[0].set_title("Chrom1 - Chrom2")
    ax[0].legend()

    ax[1].hist(data32, bins=30, density=True, alpha=0.4, label='data32')
    ax[1].plot(x_grid, pdf32, 'r-', label='model pdf32')
    ax[1].set_title("Chrom3 - Chrom2")
    ax[1].legend()

    plt.tight_layout()
    plt.show()
    fig.savefig('Results/theoryfit.png', dpi=300, bbox_inches='tight')

    # g) Run the stochastic simulation with best parameters, compare
    run_stochastic_simulation_and_plot(
        k_opt, r21_opt, R21_opt,
        r23_opt, R23_opt,
        data12, data32,
        n2_opt, N2_opt
    )

###############################################################################
# Stochastic Simulation
###############################################################################


def run_stochastic_simulation_and_plot(k_opt, r21_opt, R21_opt, r23_opt, R23_opt,
                                       data12, data32,
                                       n2_opt, N2_opt,
                                       max_time=150, num_sim=500):
    """
    Use the best-fit (k_opt, r1_opt, R1_opt, r2_opt, R2_opt) to run a Gillespie-like 
    simulation. Then compare sim difference times to the experimental data hist.
    """
    n1_opt = r21_opt*n2_opt
    N1_opt = R21_opt*N2_opt
    n3_opt = r23_opt*n2_opt
    N3_opt = R23_opt*N2_opt

    initial_proteins1 = N1_opt
    initial_proteins2 = N2_opt
    initial_proteins3 = N3_opt  # or any known / used
    initial_proteins = [initial_proteins1,
                        initial_proteins2, initial_proteins3]

    # If your code requires rates for each chromosome:
    rates = [k_opt, k_opt, k_opt]

    n0_total = n1_opt + n2_opt + n3_opt
    n01_mean = n1_opt
    n02_mean = n2_opt
    # if you only track 2 thresholds, pass [n01_mean, n02_mean]
    n0_list = generate_threshold_values(
        [n01_mean, n02_mean], n0_total, num_sim)

    simulations, separate_times = run_simulations(
        initial_proteins, rates, n0_list, max_time, num_sim
    )

    # separate_times[i] => [sep1, sep2, sep3]
    # so Chrom1 - Chrom2 => sep[0] - sep[1], Chrom3 - Chrom2 => sep[2] - sep[1]
    delta_t12 = [sep[0] - sep[1] for sep in separate_times]
    delta_t32 = [sep[2] - sep[1] for sep in separate_times]

    # Plot sim vs data
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].hist(delta_t12, bins=12, density=True, alpha=0.4, label='Sim data12')
    ax[0].hist(data12, bins=12, density=True, alpha=0.4, label='Exp data12')
    ax[0].set_title("Stochastic Sim vs Exp (Chrom1–Chrom2)")
    ax[0].legend()

    ax[1].hist(delta_t32, bins=12, density=True, alpha=0.4, label='Sim data32')
    ax[1].hist(data32, bins=12, density=True, alpha=0.4, label='Exp data32')
    ax[1].set_title("Stochastic Sim vs Exp (Chrom3–Chrom2)")
    ax[1].legend()

    plt.tight_layout()
    plt.show()
    fig.savefig('Results/simfit.png', dpi=300, bbox_inches='tight')


###############################################################################
# Run
###############################################################################
if __name__ == "__main__":
    main()
