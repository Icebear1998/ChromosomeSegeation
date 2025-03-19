import numpy as np
import pandas as pd
from scipy.optimize import minimize
from Chromosomes_Theory import f_diff

# If your Chrom3–Chrom2 data is actually Chrom2–Chrom3, set True:
FLIP_CHROM3_DATA = False

###############################################################################
# 3) Objective function (Chrom1–Chrom2 + Chrom3–Chrom2),
#    now optimizing over (n2, N2, k, r1, r2, R21, R23).
###############################################################################


def combined_objective(vars_, data12, data32, x_grid):
    n2, N2, k, r21, r23, R21, R23 = vars_

    n2 = int(round(n2))
    N2 = int(round(N2))

    n1 = int(round(r21 * n2))
    N1 = int(round(R21 * N2))
    n3 = int(round(r23 * n2))
    N3 = int(round(R23 * N2))

    # Chrom1–Chrom2
    pdf12 = np.array([
        f_diff(x, k, n1, N1, n2, N2)
        for x in x_grid
    ])
    area12 = np.trapz(pdf12, x_grid)
    if area12 < 1e-15 or np.any(np.isnan(pdf12)):
        print(
            f"Invalid area12 or NaN in pdf12: area12={area12}, pdf12={pdf12}")
        return np.inf
    pdf12 /= area12
    vals12 = np.interp(data12, x_grid, pdf12, left=0, right=0)
    if np.any(vals12 <= 0) or np.any(np.isnan(vals12)):
        print(f"Invalid vals12: vals12={vals12}")
        return np.inf

    # Chrom3–Chrom2
    pdf32 = np.array([
        f_diff(x, k, n3, N3, n2, N2)
        for x in x_grid
    ])
    area32 = np.trapz(pdf32, x_grid)
    if area32 < 1e-15 or np.any(np.isnan(pdf32)):
        print(
            f"Invalid area32 or NaN in pdf32: area32={area32}, pdf32={pdf32}")
        return np.inf
    pdf32 /= area32
    vals32 = np.interp(data32, x_grid, pdf32, left=0, right=0)
    if np.any(vals32 <= 0) or np.any(np.isnan(vals32)):
        print(f"Invalid vals32: vals32={vals32}")
        return np.inf

    # Negative log-likelihood
    return -np.sum(np.log(vals12)) - np.sum(np.log(vals32))


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

    # d) Define parameter bounds for optimization
    param_bounds = [
        (5, 20),     # n2
        (80, 100),   # N2
        (0.02, 0.3),  # k
        (0.5,  2.0),  # r1
        (0.5,  2.0),  # r2
        (0.5, 2.5),  # R21
        (0.5, 2.5),  # R23
    ]

    # e) Initial guess for [n2, N2, k, r1, r2, R21, R23]
    x0 = [10, 100, 0.1, 1.0, 1.0, 1.0, 1.0]

    # f) Perform global optimization
    res = minimize(
        combined_objective,
        x0,
        args=(data12, data32, x_grid),
        method='L-BFGS-B',
        bounds=param_bounds,
        options={'maxiter': 500, 'disp': True}
    )

    if not res.success:
        print("Optimization failed.")
        return

    # Extract optimized parameters
    n2_opt, N2_opt, k_opt, r21_opt, r23_opt, R21_opt, R23_opt = res.x
    n2_opt = int(round(n2_opt))
    N2_opt = int(round(N2_opt))
    print("Best negative log-likelihood:", res.fun)
    print(f"Best parameters:\n"
          f"  n2={n2_opt}, N2={N2_opt}\n"
          f"  k={k_opt:.4f}, r1={r21_opt:.4f}, R1={R21_opt:.2f}\n"
          f"  r2={r23_opt:.4f}, R2={R23_opt:.2f}")


if __name__ == "__main__":
    main()
