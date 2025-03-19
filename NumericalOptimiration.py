import numpy as np
import pandas as pd
from scipy.optimize import minimize
from Chromosomes_Theory import f_diff

# If your Chrom3–Chrom2 data is actually Chrom2–Chrom3, set True:
FLIP_CHROM3_DATA = False

###############################################################################
# 3) Objective function (Chrom1–Chrom2 + Chrom3–Chrom2),
#    with (R21, R23) externally fixed.
#    We optimize over (n2, N2, k, r21, r23).
###############################################################################
def combined_objective(vars_, R21_fixed, R23_fixed, data12, data32, x_grid):
    n2, N2, k, r21, r23 = vars_

    # Ensure n2 and N2 are integers
    if np.isnan(n2) or np.isnan(N2):
        print(f"NaN detected in n2 or N2: n2={n2}, N2={N2}")
        return np.inf
    n2 = int(round(n2))
    N2 = int(round(N2))

    n1 = int(round(r21 * n2))
    N1 = int(round(R21_fixed * N2))
    n3 = int(round(r23 * n2))
    N3 = int(round(R23_fixed * N2))

    # Chrom1–Chrom2
    pdf12 = np.array([f_diff(x, k, n1, N1, n2, N2) for x in x_grid])
    area12 = np.trapz(pdf12, x_grid)
    if area12 < 1e-15 or np.any(np.isnan(pdf12)):
        print(f"Invalid area12 or NaN in pdf12: area12={area12}")
        return np.inf
    pdf12 /= area12
    vals12 = np.interp(data12, x_grid, pdf12, left=0, right=0)
    if np.any(vals12 <= 0) or np.any(np.isnan(vals12)):
        print(f"Invalid vals12: {vals12}")
        return np.inf

    # Chrom3–Chrom2
    pdf32 = np.array([f_diff(x, k, n3, N3, n2, N2) for x in x_grid])
    area32 = np.trapz(pdf32, x_grid)
    if area32 < 1e-15 or np.any(np.isnan(pdf32)):
        print(f"Invalid area32 or NaN in pdf32: area32={area32}")
        return np.inf
    pdf32 /= area32
    vals32 = np.interp(data32, x_grid, pdf32, left=0, right=0)
    if np.any(vals32 <= 0) or np.any(np.isnan(vals32)):
        print(f"Invalid vals32: {vals32}")
        return np.inf

    return -np.sum(np.log(vals12)) - np.sum(np.log(vals32))

###############################################################################
# 4) Main "Hybrid" approach (no plotting code; progress tracking added)
###############################################################################
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

    # c) Define small discrete sets for R21 and R23 (less-sensitive parameters)
    R21_candidates = np.round(np.arange(0.5, 2.51, 0.25), 2)
    R23_candidates = np.round(np.arange(0.5, 2.51, 0.25), 2)

    # d) Define bounds for sensitive parameters [n2, N2, k, r1, r2]
    param_bounds = [
        (5, 12),     # n2
        (90, 150),   # N2
        (0.02, 0.3), # k
        (0.5,  2.0), # r1
        (0.5,  2.0), # r2
    ]

    best_obj = np.inf
    best_solution = None
    total_iterations = len(R21_candidates) * len(R23_candidates)
    iter_count = 0

    # e) Loop over grid cells for R21 and R23 and run local optimization for [n2, N2, k, r1, r2]
    for R21_fixed in R21_candidates:
        for R23_fixed in R23_candidates:
            iter_count += 1
            def local_obj_k_r1_r2(vars_):
                return combined_objective(vars_, R21_fixed, R23_fixed, data12, data32, x_grid)
            x0 = [4, 100, 0.1, 1.0, 1.0]

            res = minimize(
                local_obj_k_r1_r2,
                x0,
                method='L-BFGS-B',
                bounds=param_bounds,
                options={'maxiter': 50, 'disp': False}
            )

            # Track progress: print grid values, iteration count, and objective value.
            print(f"Iteration {iter_count}/{total_iterations} | R21_fixed={R21_fixed}, R23_fixed={R23_fixed} | Obj={res.fun:.4e}")
            if res.success:
                print(f"  Local solution: {res.x}")
            else:
                print("  Local optimization failed.")

            if res.success and res.fun < best_obj:
                best_obj = res.fun
                best_solution = (res.x[0], res.x[1], res.x[2],
                                 res.x[3], res.x[4], R21_fixed, R23_fixed)

    if best_solution is None:
        print("No valid solution found in the (R21, R23) grid.")
        return

    n2_opt, N2_opt, k_opt, r21_opt, r23_opt, R21_opt, R23_opt = best_solution
    print("Best negative log-likelihood:", best_obj)
    print(f"Best parameters:\n"
          f"  n2={n2_opt:.2f}, N2={N2_opt:.2f}\n"
          f"  k={k_opt:.4f}, r1={r21_opt:.4f}, R1={R21_opt:.2f}\n"
          f"  r2={r23_opt:.4f}, R2={R23_opt:.2f}")

if __name__ == "__main__":
    main()
