from Chromosome_Gillespie4 import run_simulations, generate_threshold_values
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import quad
from math import exp, isfinite
from scipy.special import gamma
from Chromosomes_Theory import f_diff_gamma

# If your Chrom3–Chrom2 data is actually Chrom2–Chrom3, set True:
FLIP_CHROM3_DATA = False

###############################################################################
# 3) Objective function (Chrom1–Chrom2 + Chrom3–Chrom2),
#    but (R1, R2) are externally fixed. We optimize over (k, r1, r2).
###############################################################################


def combined_objective(vars_, R21_fixed, R23_fixed, data12, data32, x_grid):

    n2, N2, k, r21, r23 = vars_

    n1 = r21 * n2
    N1 = R21_fixed * N2
    n3 = r23 * n2
    N3 = R23_fixed * N2

    # Chrom1–Chrom2
    pdf12 = np.array([
        f_diff_gamma(x, k, n1, N1, n2, N2)
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
        f_diff_gamma(x, k, n3, N3, n2, N2)
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
    data12 = df['Wildtype12'].dropna().values
    data32 = df['Wildtype32'].dropna().values

    # If your data for Chrom3–Chrom2 is actually (Chrom2–Chrom3), flip sign:
    if FLIP_CHROM3_DATA:
        data32 = -data32

    # c) Prepare a broad x_grid
    x_grid = np.linspace(-80, 80, 301)

    # d) We'll define small discrete sets for R1, R2
    # e.g. 0.50, 0.75, 1.00, ...
    R21_candidates = np.round(np.arange(0.5, 2.01, 0.25), 2)
    # e.g. 0.50, 1.00, 1.50, ...
    R23_candidates = np.round(np.arange(0.5, 2.51, 0.5), 2)

    # e) We'll do local optimization over [k, r1, r2] for each grid cell (R1,R2)
    param_bounds = [
        (5, 10),     # n2
        (80, 200),   # N2
        (0.01, 0.3),  # k
        (0.5,  2.0),  # r1
        (0.5,  2.0),  # r2
    ]

    best_obj = np.inf
    best_solution = None

    for R21_fixed in R21_candidates:
        for R23_fixed in R23_candidates:
            print(f"Optimizing for R1={R21_fixed:.2f}, R2={R23_fixed:.2f}")
            # Define a local objective that only depends on [k, r1, r2],
            # with R1_fixed, R2_fixed closed over.

            def local_obj_k_r1_r2(vars_):
                return combined_objective(
                    vars_, R21_fixed, R23_fixed, data12, data32, x_grid
                )

            # pick an initial guess for [n2, N2, k, r1, r2]
            x0 = [7, 100, 0.05, 1.0, 1.0]

            # local optimization
            res = minimize(
                local_obj_k_r1_r2,
                x0,
                method='L-BFGS-B',
                bounds=param_bounds,
                options={'maxiter': 30, 'disp': False}
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


if __name__ == "__main__":
    main()

# Best negative log-likelihood: 1134.341497412577
# Best parameters:
#   n2=10.00, N2=98.96
#   k=0.0278, r1=0.5251, R1=0.50
#   r2=1.1022, R2=1.50
