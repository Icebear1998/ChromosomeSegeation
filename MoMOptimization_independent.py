import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize, basinhopping
from scipy.stats import norm
from MoMCalculations import compute_pdf_mom
import math

def wildtype_objective(vars_, mechanism_, data12, data32):
    n2, N2, k, r21, r23, R21, R23, burst_size = vars_
    n1 = max(r21 * n2, 1)
    N1 = max(R21 * N2, 1)
    n3 = max(r23 * n2, 1)
    N3 = max(R23 * N2, 1)

    # Validate inputs
    if burst_size <= 0 or N1 <= 0 or N2 <= 0 or N3 <= 0 or k <= 0:
        return np.inf

    # Sample sizes
    n12 = len(data12)
    n32 = len(data32)

    # Weight (adjustable, default 1.0)
    weight = 1.0

    # Chrom1–Chrom2
    pdf12 = compute_pdf_mom(mechanism_, data12, n1, N1, n2, N2, k, burst_size)
    if np.any(pdf12 <= 0) or np.any(np.isnan(pdf12)):
        return np.inf
    log_likelihood12 = np.sum(np.log(pdf12)) / n12

    # Chrom3–Chrom2
    pdf32 = compute_pdf_mom(mechanism_, data32, n3, N3, n2, N2, k, burst_size)
    if np.any(pdf32 <= 0) or np.any(np.isnan(pdf32)):
        return np.inf
    log_likelihood32 = np.sum(np.log(pdf32)) / n32

    return -weight * (log_likelihood12 + log_likelihood32)

###############################################################################
# 3) Objective functions for mutants
###############################################################################
# Threshold mutant (optimize alpha for n1, n2, n3)

def threshold_objective(vars_, mechanism_, data12, data32, params_baseline):
    n1_wt, n2_wt, n3_wt, N1_wt, N2_wt, N3_wt, k_wt, burst_size = params_baseline
    alpha = vars_[0]
    n1 = max(n1_wt * alpha, 1)
    n2 = max(n2_wt * alpha, 1)
    n3 = max(n3_wt * alpha, 1)

    # Validate inputs
    if burst_size <= 0 or N1_wt <= 0 or N2_wt <= 0 or N3_wt <= 0 or k_wt <= 0 or alpha <= 0:
        return np.inf

    # Sample sizes
    n12 = len(data12)
    n32 = len(data32)

    # Weight (adjustable, default 1.0)
    weight = 1.0

    # Chrom1–Chrom2
    pdf12 = compute_pdf_mom(mechanism_, data12, n1, N1_wt, n2, N2_wt, k_wt, burst_size)
    if np.any(pdf12 <= 0) or np.any(np.isnan(pdf12)):
        return np.inf
    log_likelihood12 = np.sum(np.log(pdf12)) / n12

    # Chrom3–Chrom2
    pdf32 = compute_pdf_mom(mechanism_, data32, n3, N3_wt, n2, N2_wt, k_wt, burst_size)
    if np.any(pdf32 <= 0) or np.any(np.isnan(pdf32)):
        return np.inf
    log_likelihood32 = np.sum(np.log(pdf32)) / n32

    return -weight * (log_likelihood12 + log_likelihood32)

# Degradation rate mutant (optimize beta_k for k)

def degrate_objective(vars_, mechanism_, data12, data32, params_baseline):
    n1_wt, n2_wt, n3_wt, N1_wt, N2_wt, N3_wt, k_wt, burst_size = params_baseline
    beta_k = vars_[0]
    k = max(beta_k * k_wt, 0.001)
    if (beta_k * k_wt < 0.001):
        print("Warning: k is too small, setting to 0.001")

    # Validate inputs
    if burst_size <= 0 or N1_wt <= 0 or N2_wt <= 0 or N3_wt <= 0 or k <= 0:
        return np.inf

    # Sample sizes
    n12 = len(data12)
    n32 = len(data32)

    # Weight (adjustable, default 1.0)
    weight = 1.0

    # Chrom1–Chrom2
    pdf12 = compute_pdf_mom(mechanism_, data12, n1_wt, N1_wt, n2_wt, N2_wt, k, burst_size)
    if np.any(pdf12 <= 0) or np.any(np.isnan(pdf12)):
        return np.inf
    log_likelihood12 = np.sum(np.log(pdf12)) / n12

    # Chrom3–Chrom2
    pdf32 = compute_pdf_mom(mechanism_, data32, n3_wt, N3_wt, n2_wt, N2_wt, k, burst_size)
    if np.any(pdf32 <= 0) or np.any(np.isnan(pdf32)):
        return np.inf
    log_likelihood32 = np.sum(np.log(pdf32)) / n32

    return -weight * (log_likelihood12 + log_likelihood32)

# Initial proteins mutant (optimize gamma for N1, N2, N3)

def initial_proteins_objective(vars_, mechanism_, data12, data32, params_baseline):
    n1_wt, n2_wt, n3_wt, N1_wt, N2_wt, N3_wt, k_wt, burst_size = params_baseline
    gamma = vars_[0]
    N1 = max(N1_wt * gamma, 1)
    N2 = max(N2_wt * gamma, 1)
    N3 = max(N3_wt * gamma, 1)

    # Validate inputs
    if burst_size <= 0 or n1_wt <= 0 or n2_wt <= 0 or n3_wt <= 0 or N1 <= 0 or N2 <= 0 or N3 <= 0 or k_wt <= 0 or gamma <= 0:
        return np.inf

    # Sample sizes
    n12 = len(data12)
    n32 = len(data32)

    # Weight (adjustable, default 1.0)
    weight = 1.0

    # Chrom1–Chrom2
    pdf12 = compute_pdf_mom(mechanism_, data12, n1_wt, N1, n2_wt, N2, k_wt, burst_size)
    if np.any(pdf12 <= 0) or np.any(np.isnan(pdf12)):
        return np.inf
    log_likelihood12 = np.sum(np.log(pdf12)) / n12

    # Chrom3–Chrom2
    pdf32 = compute_pdf_mom(mechanism_, data32, n3_wt, N3, n2_wt, N2, k_wt, burst_size)
    if np.any(pdf32 <= 0) or np.any(np.isnan(pdf32)):
        return np.inf
    log_likelihood32 = np.sum(np.log(pdf32)) / n32

    return -weight * (log_likelihood12 + log_likelihood32)

###############################################################################
# 4) Helper function to get rounded parameters
###############################################################################

def get_rounded_parameters(params):
    n2, N2, k, r21, r23, R21, R23, burst_size = params
    n1 = max(r21 * n2, 1)
    n3 = max(r23 * n2, 1)
    N1 = max(R21 * N2, 1)
    N3 = max(R23 * N2, 1)
    burst_size = max(burst_size, 1)
    # Round n1, n2, n3, N1, N2, N3 to 1 decimal, k to 3 decimals
    return (
        round(n1, 1),
        round(n2, 1),
        round(n3, 1),
        round(N1, 1),
        round(N2, 1),
        round(N3, 1),
        round(k, 3),
        round(burst_size, 1)
    )

###############################################################################
# 5) Helper class for bounded steps in basinhopping
###############################################################################

class BoundedStep:
    def __init__(self, bounds, stepsize=0.5):
        self.bounds = np.array(bounds)
        self.stepsize = stepsize

    def __call__(self, x):
        x_new = x + np.random.uniform(-self.stepsize,
                                      self.stepsize, size=x.shape)
        return np.clip(x_new, self.bounds[:, 0], self.bounds[:, 1])

###############################################################################
# 6) Main function for optimization
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

    # b) Define parameter bounds for wild-type optimization
    wt_bounds = [
        (1, 30),     # n2
        (80, 500),   # N2
        (0.005, 0.4),  # k
        (0.3, 2.5),  # r21
        (0.3, 2.5),  # r23
        (0.4, 2.5),  # R21
        (0.4, 5.0),  # R23
        (1, 10)      # burst_size
    ]

    # c) Global optimization for wild-type to find top 5 solutions
    population_solutions = []
    mechanism = 'fixed_burst'

    def callback(xk, convergence):
        population_solutions.append(
            (wildtype_objective(xk, mechanism, data_wt12, data_wt32), xk.copy()))

    result = differential_evolution(
        wildtype_objective,
        bounds=wt_bounds,
        args=(mechanism, data_wt12, data_wt32),
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
        if rounded_params not in seen_parameters:
            top_5_solutions.append((nll, params))
            seen_parameters.add(rounded_params)
            if len(top_5_solutions) == 5:
                break

    if len(top_5_solutions) < 5:
        print(
            f"Warning: Only {len(top_5_solutions)} distinct solutions found after rounding.")

    print("\nTop 5 Wild-Type Solutions from Differential Evolution:")
    for i, (nll, params) in enumerate(top_5_solutions):
        n2, N2, k, r21, r23, R21, R23, burst_size = params
        n1 = max(r21 * n2, 1)
        n3 = max(r23 * n2, 1)
        N1 = max(R21 * N2, 1)
        N3 = max(R23 * N2, 1)
        print(f"Solution {i+1}: Negative Log-Likelihood = {nll:.4f}")
        print(
            f"Parameters: n2 = {n2:.2f}, N2 = {N2:.2f}, k = {k:.4f}, r21 = {r21:.2f}, r23 = {r23:.2f}, R21 = {R21:.2f}, R23 = {R23:.2f}, burst_size = {burst_size:.2f}")
        print(
            f"Derived: n1 = {n1:.2f}, n3 = {n3:.2f}, N1 = {N1:.2f}, N3 = {N3:.2f}")

    # d) Local optimization to refine top 5 wild-type solutions
    refined_wt_solutions = []
    for i, (_, params) in enumerate(top_5_solutions):
        result_local = minimize(
            wildtype_objective,
            x0=params,
            args=(mechanism, data_wt12, data_wt32),
            method='L-BFGS-B',
            bounds=wt_bounds,
            options={'disp': False}
        )
        if result_local.success:
            refined_wt_solutions.append((result_local.fun, result_local.x))
        else:
            print(f"Wild-type local optimization failed for solution {i+1}")

    refined_wt_solutions.sort(key=lambda x: x[0])
    if not refined_wt_solutions:
        print("No successful wild-type optimizations.")
        return

    # e) Optimize mutants for each top 5 wild-type solution using basinhopping
    n_mutant_bound = [(0.1, 1.0)]       # For alpha
    degrate_bound = [(0.1, 1.0)]      # For beta_k
    N_mutant_bound = [(0.01, 1.0)]     # For gamma

    overall_results = []
    for wt_idx, (wt_nll, wt_params) in enumerate(refined_wt_solutions[:5]):
        n2_wt, N2_wt, k, r21, r23, R21, R23, burst_size = wt_params
        n1_wt = max(r21 * n2_wt, 1)
        N1_wt = max(R21 * N2_wt, 1)
        n3_wt = max(r23 * n2_wt, 1)
        N3_wt = max(R23 * N2_wt, 1)
        params_baseline = (n1_wt, n2_wt, n3_wt, N1_wt, N2_wt, N3_wt, k, burst_size)

        # Compute unweighted wild-type NLL for reporting
        wt_nll_unweighted = - \
            wildtype_objective(wt_params, mechanism, data_wt12, data_wt32)

        # Threshold mutant
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "args": (mechanism, data_threshold12, data_threshold32, params_baseline),
            "bounds": n_mutant_bound
        }
        result_threshold = basinhopping(
            threshold_objective,
            x0=np.array([1.0]),
            minimizer_kwargs=minimizer_kwargs,
            niter=100,
            T=1.0,
            stepsize=0.5,
            take_step=BoundedStep(n_mutant_bound),
            disp=False
        )
        if result_threshold.lowest_optimization_result.success:
            threshold_nll = result_threshold.lowest_optimization_result.fun
            alpha = result_threshold.lowest_optimization_result.x[0]
            # Compute unweighted NLL for reporting
            threshold_nll_unweighted = - \
                threshold_objective([alpha], mechanism, data_threshold12,
                                    data_threshold32, params_baseline)
        else:
            threshold_nll = np.inf
            alpha = np.nan
            threshold_nll_unweighted = np.inf

        # Degradation rate mutant
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "args": (mechanism, data_degrate12, data_degrate32, params_baseline),
            "bounds": degrate_bound
        }
        result_degrate = basinhopping(
            degrate_objective,
            x0=np.array([1.0]),
            minimizer_kwargs=minimizer_kwargs,
            niter=100,
            T=1.0,
            stepsize=0.5,
            take_step=BoundedStep(degrate_bound),
            disp=False
        )
        if result_degrate.lowest_optimization_result.success:
            degrate_nll = result_degrate.lowest_optimization_result.fun
            beta_k = result_degrate.lowest_optimization_result.x[0]
            # Compute unweighted NLL for reporting
            degrate_nll_unweighted = - \
                degrate_objective([beta_k], mechanism, data_degrate12,
                                  data_degrate32, params_baseline)
        else:
            degrate_nll = np.inf
            beta_k = np.nan
            degrate_nll_unweighted = np.inf

        # Initial proteins mutant
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "args": (mechanism, data_initial12, data_initial32, params_baseline),
            "bounds": N_mutant_bound
        }
        result_initial = basinhopping(
            initial_proteins_objective,
            x0=np.array([1.0]),
            minimizer_kwargs=minimizer_kwargs,
            niter=100,
            T=1.0,
            stepsize=0.5,
            take_step=BoundedStep(N_mutant_bound),
            disp=False
        )
        if result_initial.lowest_optimization_result.success:
            initial_nll = result_initial.lowest_optimization_result.fun
            gamma = result_initial.lowest_optimization_result.x[0]
            # Compute unweighted NLL for reporting
            initial_nll_unweighted = - \
                initial_proteins_objective(
                    [gamma], mechanism, data_initial12, data_initial32, params_baseline)
        else:
            initial_nll = np.inf
            gamma = np.nan
            initial_nll_unweighted = np.inf

        # Total negative log-likelihood (weighted for optimization)
        total_nll = wt_nll + threshold_nll + degrate_nll + initial_nll
        # Total unweighted NLL for reporting
        total_nll_unweighted = wt_nll_unweighted + threshold_nll_unweighted + \
            degrate_nll_unweighted + initial_nll_unweighted
        overall_results.append({
            'wt_idx': wt_idx,
            'total_nll': total_nll,
            'total_nll_unweighted': total_nll_unweighted,
            'wt_nll': wt_nll_unweighted,
            'wt_params': wt_params,
            'threshold_nll': threshold_nll_unweighted,
            'alpha': alpha,
            'degrate_nll': degrate_nll_unweighted,
            'beta_k': beta_k,
            'initial_nll': initial_nll_unweighted,
            'gamma': gamma,
        })

    # f) Select the best overall solution
    overall_results.sort(key=lambda x: x['total_nll'])
    best_result = overall_results[0]

    print("\nBest Overall Solution:")
    print(
        f"Total Negative Log-Likelihood (Weighted): {best_result['total_nll']:.4f}")
    print(
        f"Total Negative Log-Likelihood (Unweighted): {best_result['total_nll_unweighted']:.4f}")
    print(f"Wild-Type Negative Log-Likelihood: {best_result['wt_nll']:.4f}")
    n2, N2, k, r21, r23, R21, R23, burst_size = best_result['wt_params']
    n1 = max(r21 * n2, 1)
    N1 = max(R21 * N2, 1)
    n3 = max(r23 * n2, 1)
    N3 = max(R23 * N2, 1)
    print(f"Wild-Type Parameters: n1 = {n1:.2f}, n2 = {n2:.2f}, n3 = {n3:.2f}, "
          f"N1 = {N1:.2f}, N2 = {N2:.2f}, N3 = {N3:.2f}, k = {k:.4f}, burst_size = {burst_size:.2f}")
    print(f"Threshold Mutant: NLL = {best_result['threshold_nll']:.4f}, "
          f"alpha = {best_result['alpha']:.2f}")
    print(f"Degradation Rate Mutant: NLL = {best_result['degrate_nll']:.4f}, "
          f"beta_k = {best_result['beta_k']:.2f}")
    print(f"Initial Proteins Mutant: NLL = {best_result['initial_nll']:.4f}, "
          f"gamma = {best_result['gamma']:.2f}")

    # g) Save optimized parameters to a text file
    with open("optimized_parameters_IndeBurst.txt", "w") as f:
        f.write("# Wild-Type Parameters\n")
        f.write(f"n1: {n1:.6f}\n")
        f.write(f"n2: {n2:.6f}\n")
        f.write(f"n3: {n3:.6f}\n")
        f.write(f"N1: {N1:.6f}\n")
        f.write(f"N2: {N2:.6f}\n")
        f.write(f"N3: {N3:.6f}\n")
        f.write(f"k: {k:.6f}\n")
        f.write(f"burst_size: {burst_size:.6f}\n")
        f.write(f"wt_nll: {best_result['wt_nll']:.6f}\n")
        f.write("# Mutant Parameters\n")
        f.write(f"alpha: {best_result['alpha']:.6f}\n")
        f.write(f"beta_k: {best_result['beta_k']:.6f}\n")
        f.write(f"gamma: {best_result['gamma']:.6f}\n")
        f.write(f"threshold_nll: {best_result['threshold_nll']:.6f}\n")
        f.write(f"degrate_nll: {best_result['degrate_nll']:.6f}\n")
        f.write(f"initial_nll: {best_result['initial_nll']:.6f}\n")
        f.write(f"total_nll: {best_result['total_nll_unweighted']:.6f}\n")
    print("Optimized parameters saved to optimized_parameters_IndeUpdate.txt")

if __name__ == "__main__":
    main()