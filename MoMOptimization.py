import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm

# If your Chrom3–Chrom2 data is actually Chrom2–Chrom3, set True:
FLIP_CHROM3_DATA = False

###############################################################################
# 1) Compute MoM moments for f_X
###############################################################################
def compute_moments_mom(n1, N1, n2, N2, k):
    sum1_T1 = sum(1/m for m in range(int(n1) + 1, int(N1) + 1))
    sum1_T2 = sum(1/m for m in range(int(n2) + 1, int(N2) + 1))
    sum2_T1 = sum(1/(m**2) for m in range(int(n1) + 1, int(N1) + 1))
    sum2_T2 = sum(1/(m**2) for m in range(int(n2) + 1, int(N2) + 1))
    
    mean_T1 = sum1_T1 / k
    mean_T2 = sum1_T2 / k
    var_T1 = sum2_T1 / (k**2)
    var_T2 = sum2_T2 / (k**2)
    
    mean_X = mean_T1 - mean_T2
    var_X = var_T1 + var_T2
    return mean_X, var_X

###############################################################################
# 2) Objective function (Chrom1–Chrom2 + Chrom3–Chrom2),
#    now optimizing over (n2, N2, k, r21, r23, R21, R23) using MoM normal
###############################################################################
def combined_objective(vars_, data12, data32):
    n2, N2, k, r21, r23, R21, R23 = vars_

    # Use continuous values for mean calculations
    n1 = max(r21 * n2, 1)
    N1 = max(R21 * N2, 1)
    n3 = max(r23 * n2, 1)
    N3 = max(R23 * N2, 1)

    # Chrom1–Chrom2: Use MoM normal approximation
    mean12, var12 = compute_moments_mom(n1, N1, n2, N2, k)
    pdf12 = norm.pdf(data12, loc=mean12, scale=np.sqrt(var12))
    # if np.any(pdf12 <= 0) or np.any(np.isnan(pdf12)):
    #     print("Invalid pdf12. Parameters:")
    #     print("n1 =", n1, "N1 =", N1, "n2 =", n2, "N2 =", N2, "k =", k)
    #     return np.inf
    log_likelihood12 = np.sum(np.log(pdf12))

    # Chrom3–Chrom2: Use MoM normal approximation
    mean32, var32 = compute_moments_mom(n3, N3, n2, N2, k)
    pdf32 = norm.pdf(data32, loc=mean32, scale=np.sqrt(var32))
    # if np.any(pdf32 <= 0) or np.any(np.isnan(pdf32)):
    #     print("Invalid pdf32. Parameters:")
    #     print("n3 =", n3, "N3 =", N3, "n2 =", n2, "N2 =", N2, "k =", k)
    #     return np.inf
    log_likelihood32 = np.sum(np.log(pdf32))

    # Return negative log-likelihood (to minimize)
    return -(log_likelihood12 + log_likelihood32)

###############################################################################
# 3) Main function for optimization
###############################################################################
def main():
    # a) Read data
    df = pd.read_excel("Data/Chromosome_diff.xlsx")
    data12 = df['Wildtype12'].dropna().values
    data32 = df['Wildtype32'].dropna().values

    # Flip sign for Chrom3–Chrom2 if needed
    if FLIP_CHROM3_DATA:
        data32 = -data32

    # b) Define parameter bounds for global optimization (loose constraints)
    param_bounds = [
        (1, 30),     # n2
        (80, 400),   # N2
        (0.02, 0.4), # k
        (0.3, 2.5),  # r21
        (0.3, 2.5),  # r23
        (0.4, 2.5),  # R21
        (0.4, 5.0),  # R23
    ]

    # c) Global optimization with differential evolution to find top 10 solutions
    population_solutions = []
    def callback(xk, convergence):
        # Store the current population's best solutions
        population_solutions.append((combined_objective(xk, data12, data32), xk.copy()))

    result = differential_evolution(
        combined_objective,
        bounds=param_bounds,
        args=(data12, data32),
        strategy='best1bin',
        maxiter=2000,
        popsize=15,
        tol=1e-6,
        disp=True,
        callback=callback
    )

    # Collect all solutions from the callback
    population_solutions.sort(key=lambda x: x[0])  # Sort by negative log-likelihood
    top_10_solutions = population_solutions[:10]  # Take top 10

    print("\nTop 10 Solutions from Differential Evolution:")
    for i, (nll, params) in enumerate(top_10_solutions):
        print(f"Solution {i+1}: Negative Log-Likelihood = {nll:.4f}, Parameters = {params}")

    # d) Local optimization to refine the top 10 solutions
    refined_solutions = []
    for i, (_, params) in enumerate(top_10_solutions):
        result_local = minimize(
            combined_objective,
            x0=params,
            args=(data12, data32),
            method='L-BFGS-B',
            bounds=param_bounds,
            options={'disp': False}
        )
        if result_local.success:
            refined_solutions.append((result_local.fun, result_local.x))
        else:
            print(f"Local optimization failed for solution {i+1}")

    # e) Select the best solution
    refined_solutions.sort(key=lambda x: x[0])  # Sort by negative log-likelihood
    if refined_solutions:
        best_nll, best_params = refined_solutions[0]
        n2, N2, k, r21, r23, R21, R23 = best_params
        n1 = max(r21 * n2, 1)
        N1 = max(R21 * N2, 1)
        n3 = max(r23 * n2, 1)
        N3 = max(R23 * N2, 1)

        print("\nBest Solution After Local Optimization:")
        print(f"Negative Log-Likelihood: {best_nll:.4f}")
        print(f"Parameters: n2 = {n2:.2f}, N2 = {N2:.2f}, k = {k:.4f}, r21 = {r21:.2f}, r23 = {r23:.2f}, R21 = {R21:.2f}, R23 = {R23:.2f}")
        print(f"Derived: n1 = {n1:.2f}, N1 = {N1:.2f}, n3 = {n3:.2f}, N3 = {N3:.2f}")
    else:
        print("No successful local optimizations.")

if __name__ == "__main__":
    main()