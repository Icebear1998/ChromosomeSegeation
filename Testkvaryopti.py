import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm
from MoMCalculations import compute_moments_mom, compute_pdf_mom

def joint_objective(params, mechanism, data_wt12, data_wt32, data_threshold12, data_threshold32,
                    data_degrate12, data_degrate32, data_initial12, data_initial32):
    # Unpack parameters
    n2, N2, k, r21, r23, R21, R23, k_1, alpha, beta_k, gamma = params

    # Derived wild-type parameters
    n1 = max(r21 * n2, 1)
    n3 = max(r23 * n2, 1)
    N1 = max(R21 * N2, 1)
    N3 = max(R23 * N2, 1)

    total_nll = 0.0

    # Wild-Type (Chrom1–Chrom2 and Chrom3–Chrom2)
    pdf_wt12 = compute_pdf_mom(mechanism, data_wt12, n1, N1, n2, N2, k, k_1=k_1)
    if np.any(pdf_wt12 <= 0) or np.any(np.isnan(pdf_wt12)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_wt12)) / len(data_wt12)

    pdf_wt32 = compute_pdf_mom(mechanism, data_wt32, n3, N3, n2, N2, k, k_1=k_1)
    if np.any(pdf_wt32 <= 0) or np.any(np.isnan(pdf_wt32)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_wt32)) / len(data_wt32)

    # Threshold Mutant
    n1_th = max(n1 * alpha, 1)
    n2_th = max(n2 * alpha, 1)
    n3_th = max(n3 * alpha, 1)

    pdf_th12 = compute_pdf_mom(mechanism, data_threshold12, n1_th, N1, n2_th, N2, k, k_1=k_1)
    if np.any(pdf_th12 <= 0) or np.any(np.isnan(pdf_th12)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_th12)) / len(data_threshold12)

    pdf_th32 = compute_pdf_mom(mechanism, data_threshold32, n3_th, N3, n2_th, N2, k, k_1=k_1)
    if np.any(pdf_th32 <= 0) or np.any(np.isnan(pdf_th32)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_th32)) / len(data_threshold32)

    # Degradation Rate Mutant
    k_deg = max(beta_k * k, 0.001)
    if beta_k * k < 0.001:
        print("Warning: beta_k * k is less than 0.001, setting k_deg to 0.001")

    pdf_deg12 = compute_pdf_mom(mechanism, data_degrate12, n1, N1, n2, N2, k_deg, k_1=k_1)
    if np.any(pdf_deg12 <= 0) or np.any(np.isnan(pdf_deg12)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_deg12)) / len(data_degrate12)

    pdf_deg32 = compute_pdf_mom(mechanism, data_degrate32, n3, N3, n2, N2, k_deg, k_1=k_1)
    if np.any(pdf_deg32 <= 0) or np.any(np.isnan(pdf_deg32)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_deg32)) / len(data_degrate32)

    # Initial Proteins Mutant
    N1_init = max(N1 * gamma, 1)
    N2_init = max(N2 * gamma, 1)
    N3_init = max(N3 * gamma, 1)

    pdf_init12 = compute_pdf_mom(mechanism, data_initial12, n1, N1_init, n2, N2_init, k, k_1=k_1)
    if np.any(pdf_init12 <= 0) or np.any(np.isnan(pdf_init12)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_init12)) / len(data_initial12)

    pdf_init32 = compute_pdf_mom(mechanism, data_initial32, n3, N3_init, n2, N2_init, k, k_1=k_1)
    if np.any(pdf_init32 <= 0) or np.any(np.isnan(pdf_init32)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_init32)) / len(data_initial32)

    return total_nll

###############################################################################
# 3) Helper function to get rounded parameters
###############################################################################

def get_rounded_parameters(params):
    n2, N2, k, r21, r23, R21, R23, k_1, alpha, beta_k, gamma = params
    n1 = max(r21 * n2, 1)
    n3 = max(r23 * n2, 1)
    N1 = max(R21 * N2, 1)
    N3 = max(R23 * N2, 1)
    # Round n1, n2, n3, N1, N2, N3 to 1 decimal, k to 3 decimals, k_1 to 4 decimals, others to 2 decimals
    return (
        round(n1, 1),
        round(n2, 1),
        round(n3, 1),
        round(N1, 1),
        round(N2, 1),
        round(N3, 1),
        round(k, 3),
        round(k_1, 4),
        round(alpha, 2),
        round(beta_k, 2),
        round(gamma, 2)
    )

###############################################################################
# 4) Main function for optimization
###############################################################################

def main():
    # a) Read data
    df = pd.read_excel("Data/All_strains_SCStimes.xlsx")
    data_wt12 = df['wildtype12'].dropna().values
    data_wt32 = df['wildtype32'].dropna().values
    data_threshold12 = df['threshold12'].dropna().values
    data_threshold32 = df['threshold32'].dropna().values
    data_degrate12 = df['degRate12'].dropna().values
    data_degrate32 = df['degRate32'].dropna().values
    data_initial12 = df['initialProteins12'].dropna().values
    data_initial32 = df['initialProteins32'].dropna().values

    # b) Define mechanism and parameter bounds
    mechanism = 'time_varying_k'  # Updated: Moved mechanism here for easy adjustment
    bounds = [
        (3, 30),     # n2
        (80, 500),   # N2
        (0.005, 0.4),  # k
        (0.3, 2.5),  # r21
        (0.3, 2.5),  # r23
        (0.4, 2.5),  # R21
        (0.4, 5.0),  # R23
        (0.00001, 0.01),  # k_1 (for time_varying_k mechanism)
        (0.1, 1.0),    # alpha
        (0.1, 1.0),  # beta_k
        (0.01, 1.0),  # gamma
    ]

    # c) Global optimization to find top 5 solutions
    population_solutions = []

    def callback(xk, convergence):
        population_solutions.append((joint_objective(xk, mechanism, data_wt12, data_wt32,
                                                     data_threshold12, data_threshold32,
                                                     data_degrate12, data_degrate32,
                                                     data_initial12, data_initial32), xk.copy()))

    result = differential_evolution(
        joint_objective,
        bounds=bounds,
        args=(mechanism, data_wt12, data_wt32, data_threshold12, data_threshold32,
              data_degrate12, data_degrate32, data_initial12, data_initial32),
        strategy='best1bin',
        maxiter=2000,
        popsize=15,
        tol=1e-6,
        disp=True,
        callback=callback
    )

    # Collect top 5 solutions with distinct rounded parameters
    population_solutions.sort(key=lambda x: x[0])
    top_5_solutions = []
    seen_parameters = set()
    for nll, params in population_solutions:
        rounded_params = get_rounded_parameters(params)
        rounded_tuple = tuple(rounded_params)
        if rounded_tuple not in seen_parameters:
            top_5_solutions.append((nll, params))
            seen_parameters.add(rounded_tuple)
            if len(top_5_solutions) == 5:
                break

    if len(top_5_solutions) < 5:
        print(
            f"Warning: Only {len(top_5_solutions)} distinct solutions found after rounding.")

    print("\nTop 5 Solutions from Differential Evolution:")
    for i, (nll, params) in enumerate(top_5_solutions):
        n2, N2, k, r21, r23, R21, R23, k_1, alpha, beta_k, gamma = params
        n1 = max(r21 * n2, 1)
        n3 = max(r23 * n2, 1)
        N1 = max(R21 * N2, 1)
        N3 = max(R23 * N2, 1)
        print(f"Solution {i+1}: Negative Log-Likelihood = {nll:.4f}")
        print(f"Parameters: n2 = {n2:.2f}, N2 = {N2:.2f}, k = {k:.4f}, r21 = {r21:.2f}, "
              f"r23 = {r23:.2f}, R21 = {R21:.2f}, R23 = {R23:.2f}, k_1 = {k_1:.4f}, alpha = {alpha:.2f}, "
              f"beta_k = {beta_k:.2f}, gamma = {gamma:.2f}")
        print(
            f"Derived: n1 = {n1:.2f}, n3 = {n3:.2f}, N1 = {N1:.2f}, N3 = {N3:.2f}")

    # d) Local optimization to refine top 5 solutions
    refined_solutions = []
    for i, (_, params) in enumerate(top_5_solutions):
        result_local = minimize(
            joint_objective,
            x0=params,
            args=(mechanism, data_wt12, data_wt32, data_threshold12, data_threshold32,
                  data_degrate12, data_degrate32, data_initial12, data_initial32),
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': False}
        )
        if result_local.success:
            refined_solutions.append((result_local.fun, result_local.x))
        else:
            print(f"Local optimization failed for solution {i+1}")

    refined_solutions.sort(key=lambda x: x[0])
    if not refined_solutions:
        print("No successful optimizations.")
        return

    # e) Select the best solution
    best_nll, best_params = refined_solutions[0]
    n2, N2, k, r21, r23, R21, R23, k_1, alpha, beta_k, gamma = best_params
    n1 = max(r21 * n2, 1)
    n3 = max(r23 * n2, 1)
    N1 = max(R21 * N2, 1)
    N3 = max(R23 * N2, 1)

    # Compute individual negative log-likelihoods for reporting
    wt_nll = 0
    pdf_wt12 = compute_pdf_mom(mechanism, data_wt12, n1, N1, n2, N2, k, k_1 = k_1)
    if not (np.any(pdf_wt12 <= 0) or np.any(np.isnan(pdf_wt12))):
        wt_nll -= np.sum(np.log(pdf_wt12))
    pdf_wt32 = compute_pdf_mom(mechanism, data_wt32, n3, N3, n2, N2, k, k_1 = k_1)
    if not (np.any(pdf_wt32 <= 0) or np.any(np.isnan(pdf_wt32))):
        wt_nll -= np.sum(np.log(pdf_wt32))

    threshold_nll = 0
    n1_th = max(n1 * alpha, 1)
    n2_th = max(n2 * alpha, 1)
    n3_th = max(n3 * alpha, 1)
    pdf_th12 = compute_pdf_mom(mechanism, data_threshold12, n1_th, N1, n2_th, N2, k, k_1=k_1)
    if not (np.any(pdf_th12 <= 0) or np.any(np.isnan(pdf_th12))):
        threshold_nll -= np.sum(np.log(pdf_th12))
    pdf_th32 = compute_pdf_mom(mechanism, data_threshold32, n3_th, N3, n2_th, N2, k, k_1=k_1)
    if not (np.any(pdf_th32 <= 0) or np.any(np.isnan(pdf_th32))):
        threshold_nll -= np.sum(np.log(pdf_th32))

    degrate_nll = 0
    k_deg = max(beta_k * k, 0.001)
    pdf_deg12 = compute_pdf_mom(mechanism, data_degrate12, n1, N1, n2, N2, k_deg, k_1=k_1)
    if not (np.any(pdf_deg12 <= 0) or np.any(np.isnan(pdf_deg12))):
        degrate_nll -= np.sum(np.log(pdf_deg12))
    pdf_deg32 = compute_pdf_mom(mechanism, data_degrate32, n3, N3, n2, N2, k_deg, k_1=k_1)
    if not (np.any(pdf_deg32 <= 0) or np.any(np.isnan(pdf_deg32))):
        degrate_nll -= np.sum(np.log(pdf_deg32))

    initial_nll = 0
    N1_init = max(N1 * gamma, 1)
    N2_init = max(N2 * gamma, 1)
    N3_init = max(N3 * gamma, 1)
    pdf_init12 = compute_pdf_mom(mechanism, data_initial12, n1, N1_init, n2, N2_init, k, k_1=k_1)
    if not (np.any(pdf_init12 <= 0) or np.any(np.isnan(pdf_init12))):
        initial_nll -= np.sum(np.log(pdf_init12))
    pdf_init32 = compute_pdf_mom(mechanism, data_initial32, n3, N3_init, n2, N2_init, k, k_1=k_1)
    if not (np.any(pdf_init32 <= 0) or np.any(np.isnan(pdf_init32))):
        initial_nll -= np.sum(np.log(pdf_init32))

    # f) Print best solution
    print("\nBest Overall Solution:")
    print(f"Total Negative Log-Likelihood: {best_nll:.4f}")
    print(f"Wild-Type Negative Log-Likelihood: {wt_nll:.4f}")
    print(f"Threshold Mutant Negative Log-Likelihood: {threshold_nll:.4f}")
    print(
        f"Degradation Rate Mutant Negative Log-Likelihood: {degrate_nll:.4f}")
    print(
        f"Initial Proteins Mutant Negative Log-Likelihood: {initial_nll:.4f}")
    print(f"Wild-Type Parameters: n1 = {n1:.2f}, n2 = {n2:.2f}, n3 = {n3:.2f}, "
          f"N1 = {N1:.2f}, N2 = {N2:.2f}, N3 = {N3:.2f}, k = {k:.4f}, k_1 = {k_1:.4f}")
    print(f"Threshold Mutant: alpha = {alpha:.2f}")
    print(f"Degradation Rate Mutant: beta_k = {beta_k:.2f}")
    print(f"Initial Proteins Mutant: gamma = {gamma:.2f}")

    # g) Save optimized parameters to a text file
    with open("optimized_parameters_timeVaryingK.txt", "w") as f:
        f.write("# Wild-Type Parameters\n")
        f.write(f"n1: {n1:.6f}\n")
        f.write(f"n2: {n2:.6f}\n")
        f.write(f"n3: {n3:.6f}\n")
        f.write(f"N1: {N1:.6f}\n")
        f.write(f"N2: {N2:.6f}\n")
        f.write(f"N3: {N3:.6f}\n")
        f.write(f"k: {k:.6f}\n")
        f.write(f"k_1: {k_1:.6f}\n")
        f.write(f"wt_nll: {wt_nll:.6f}\n")
        f.write("# Mutant Parameters\n")
        f.write(f"alpha: {alpha:.6f}\n")
        f.write(f"beta_k: {beta_k:.6f}\n")
        f.write(f"gamma: {gamma:.6f}\n")
        f.write(f"threshold_nll: {threshold_nll:.6f}\n")
        f.write(f"degrate_nll: {degrate_nll:.6f}\n")
        f.write(f"initial_nll: {initial_nll:.6f}\n")
        f.write(f"total_nll: {best_nll:.6f}\n")
    print("Optimized parameters saved to optimized_parameters_timeVaryingK.txt")

if __name__ == "__main__":
    main()