import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from Chromosomes_Theory import f_diff

# If your Chrom3–Chrom2 data is actually Chrom2–Chrom3, set True:
FLIP_CHROM3_DATA = False

###############################################################################
# 3) Objective function (Chrom1–Chrom2 + Chrom3–Chrom2),
#    now optimizing over (n2, N2, k, r1, r2, R21, R23).
###############################################################################


def combined_objective(vars_, data12, data32, x_grid):
    n2, N2, k, r21, r23, R21, R23 = vars_

    # Use continuous values for mean calculations
    n1 = max(r21 * n2, 1)
    N1 = max(R21 * N2, 1)
    n3 = max(r23 * n2, 1)
    N3 = max(R23 * N2, 1)

    # Chrom1–Chrom2
    pdf12 = np.array([f_diff(x, k, n1, N1, n2, N2) for x in x_grid])
    area12 = np.trapz(pdf12, x_grid)
    if area12 < 1e-15 or np.any(np.isnan(pdf12)):
        return np.inf
    pdf12 /= max(area12, 1e-15)  # Avoid division by zero
    vals12 = np.interp(data12, x_grid, pdf12, left=0, right=0)
    if np.any(vals12 <= 0) or np.any(np.isnan(vals12)):
        return np.inf

    # Chrom3–Chrom2
    pdf32 = np.array([f_diff(x, k, n3, N3, n2, N2) for x in x_grid])
    area32 = np.trapz(pdf32, x_grid)
    if area32 < 1e-15 or np.any(np.isnan(pdf32)):
        return np.inf
    pdf32 /= max(area32, 1e-15)  # Avoid division by zero
    vals32 = np.interp(data32, x_grid, pdf32, left=0, right=0)
    if np.any(vals32 <= 0) or np.any(np.isnan(vals32)):
        return np.inf

    return -np.sum(np.log(vals12)) - np.sum(np.log(vals32))


def main():
    # a) Read data
    df = pd.read_excel("Data/Chromosome_diff.xlsx")
    data12 = df['Wildtype12'].dropna().values
    data32 = df['Wildtype32'].dropna().values

    # Flip sign for Chrom3–Chrom2 if needed:
    if FLIP_CHROM3_DATA:
        data32 = -data32

    # b) Prepare a broad x_grid
    x_grid = np.linspace(-100, 100, 301)

    # d) Define parameter bounds for optimization
    param_bounds = [
        (4, 30),     # n2
        (80, 100),   # N2
        (0.02, 0.3),  # k
        (0.5,  2.0),  # r1
        (0.5,  2.0),  # r2
        (0.4, 2),  # R21
        (0.5, 5),  # R23
    ]

    # e) Initial guess for [n2, N2, k, r1, r2, R21, R23]
    x0 = [10, 100, 0.0278, 0.5251, 0.50, 1.1022, 1.50]

    # f) Perform global optimization
    result = differential_evolution(
        combined_objective,
        bounds=param_bounds,
        args=(data12, data32, x_grid),
        strategy='best1bin',
        maxiter=1000,
        popsize=15,
        tol=1e-6,
        disp=True
    )

    if result.success:
        print("Best negative log-likelihood:", result.fun)
        print("Best parameters:", result.x)
    else:
        print("Global optimization failed.")


if __name__ == "__main__":
    main()
