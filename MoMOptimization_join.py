import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm
from MoMCalculations import compute_moments_mom, compute_pdf_mom


def unpack_parameters(params, mechanism_info):
    """
    Unpack optimization parameters based on mechanism.

    Args:
        params (array): Parameter array from optimizer
        mechanism_info (dict): Mechanism information

    Returns:
        dict: Unpacked parameters
    """
    param_dict = {}
    param_names = mechanism_info['params']

    for i, name in enumerate(param_names):
        param_dict[name] = params[i]

    # Derived wild-type parameters
    param_dict['n1'] = max(param_dict['r21'] * param_dict['n2'], 1)
    param_dict['n3'] = max(param_dict['r23'] * param_dict['n2'], 1)
    param_dict['N1'] = max(param_dict['R21'] * param_dict['N2'], 1)
    param_dict['N3'] = max(param_dict['R23'] * param_dict['N2'], 1)

    return param_dict


def compute_pdf_for_mechanism(mechanism, data, n_i, N_i, n_j, N_j, k, mech_params):
    """
    Compute PDF for any mechanism with appropriate parameters.

    Args:
        mechanism (str): Mechanism name
        data (array): Data points
        n_i, N_i, n_j, N_j, k: Common parameters
        mech_params (dict): Mechanism-specific parameters

    Returns:
        array: PDF values
    """
    if mechanism == 'simple':
        return compute_pdf_mom(mechanism, data, n_i, N_i, n_j, N_j, k)
    elif mechanism == 'fixed_burst':
        return compute_pdf_mom(mechanism, data, n_i, N_i, n_j, N_j, k,
                               burst_size=mech_params['burst_size'])
    elif mechanism == 'time_varying_k':
        return compute_pdf_mom(mechanism, data, n_i, N_i, n_j, N_j, k,
                               k_1=mech_params['k_1'])
    elif mechanism == 'feedback':
        return compute_pdf_mom(mechanism, data, n_i, N_i, n_j, N_j, k,
                               feedbackSteepness=mech_params['feedbackSteepness'],
                               feedbackThreshold=mech_params['feedbackThreshold'])
    elif mechanism == 'feedback_linear':
        return compute_pdf_mom(mechanism, data, n_i, N_i, n_j, N_j, k,
                               w1=mech_params['w1'], w2=mech_params['w2'])
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")


def joint_objective(params, mechanism, mechanism_info, data_wt12, data_wt32, data_threshold12, data_threshold32,
                    data_degrate12, data_degrate32, data_initial12, data_initial32):
    """
    Joint objective function for all mechanisms.
    """
    # Unpack parameters
    param_dict = unpack_parameters(params, mechanism_info)

    # Validate constraints: n_i < N_i for all chromosomes
    if param_dict['n1'] >= param_dict['N1']:
        return np.inf
    if param_dict['n2'] >= param_dict['N2']:
        return np.inf
    if param_dict['n3'] >= param_dict['N3']:
        return np.inf

    # Additional validation for mutant scenarios
    # Threshold mutant: ensure n_i * alpha < N_i
    n1_th = max(param_dict['n1'] * param_dict['alpha'], 1)
    n2_th = max(param_dict['n2'] * param_dict['alpha'], 1)
    n3_th = max(param_dict['n3'] * param_dict['alpha'], 1)

    if n1_th >= param_dict['N1'] or n2_th >= param_dict['N2'] or n3_th >= param_dict['N3']:
        return np.inf

    # Initial proteins mutant: ensure n_i < N_i * gamma
    N1_init = max(param_dict['N1'] * param_dict['gamma'], 1)
    N2_init = max(param_dict['N2'] * param_dict['gamma'], 1)
    N3_init = max(param_dict['N3'] * param_dict['gamma'], 1)

    if param_dict['n1'] >= N1_init or param_dict['n2'] >= N2_init or param_dict['n3'] >= N3_init:
        return np.inf

    # Extract mechanism-specific parameters
    mech_params = {}
    if mechanism == 'fixed_burst':
        mech_params['burst_size'] = param_dict['burst_size']
    elif mechanism == 'time_varying_k':
        mech_params['k_1'] = param_dict['k_1']
    elif mechanism == 'feedback':
        mech_params['feedbackSteepness'] = param_dict['feedbackSteepness']
        mech_params['feedbackThreshold'] = param_dict['feedbackThreshold']
    elif mechanism == 'feedback_linear':
        mech_params['w1'] = param_dict['w1']
        mech_params['w2'] = param_dict['w2']

    total_nll = 0.0

    # Wild-Type (Chrom1–Chrom2 and Chrom3–Chrom2)
    pdf_wt12 = compute_pdf_for_mechanism(mechanism, data_wt12, param_dict['n1'], param_dict['N1'],
                                         param_dict['n2'], param_dict['N2'], param_dict['k'], mech_params)
    if np.any(pdf_wt12 <= 0) or np.any(np.isnan(pdf_wt12)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_wt12)) / len(data_wt12)

    pdf_wt32 = compute_pdf_for_mechanism(mechanism, data_wt32, param_dict['n3'], param_dict['N3'],
                                         param_dict['n2'], param_dict['N2'], param_dict['k'], mech_params)
    if np.any(pdf_wt32 <= 0) or np.any(np.isnan(pdf_wt32)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_wt32)) / len(data_wt32)

    # Threshold Mutant
    pdf_th12 = compute_pdf_for_mechanism(mechanism, data_threshold12, n1_th, param_dict['N1'],
                                         n2_th, param_dict['N2'], param_dict['k'], mech_params)
    if np.any(pdf_th12 <= 0) or np.any(np.isnan(pdf_th12)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_th12)) / len(data_threshold12)

    pdf_th32 = compute_pdf_for_mechanism(mechanism, data_threshold32, n3_th, param_dict['N3'],
                                         n2_th, param_dict['N2'], param_dict['k'], mech_params)
    if np.any(pdf_th32 <= 0) or np.any(np.isnan(pdf_th32)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_th32)) / len(data_threshold32)

    # Degradation Rate Mutant
    k_deg = max(param_dict['beta_k'] * param_dict['k'], 0.001)
    if param_dict['beta_k'] * param_dict['k'] < 0.001:
        print("Warning: beta_k * k is less than 0.001, setting k_deg to 0.001")

    pdf_deg12 = compute_pdf_for_mechanism(mechanism, data_degrate12, param_dict['n1'], param_dict['N1'],
                                          param_dict['n2'], param_dict['N2'], k_deg, mech_params)
    if np.any(pdf_deg12 <= 0) or np.any(np.isnan(pdf_deg12)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_deg12)) / len(data_degrate12)

    pdf_deg32 = compute_pdf_for_mechanism(mechanism, data_degrate32, param_dict['n3'], param_dict['N3'],
                                          param_dict['n2'], param_dict['N2'], k_deg, mech_params)
    if np.any(pdf_deg32 <= 0) or np.any(np.isnan(pdf_deg32)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_deg32)) / len(data_degrate32)

    # Initial Proteins Mutant
    pdf_init12 = compute_pdf_for_mechanism(mechanism, data_initial12, param_dict['n1'], N1_init,
                                           param_dict['n2'], N2_init, param_dict['k'], mech_params)
    if np.any(pdf_init12 <= 0) or np.any(np.isnan(pdf_init12)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_init12)) / len(data_initial12)

    pdf_init32 = compute_pdf_for_mechanism(mechanism, data_initial32, param_dict['n3'], N3_init,
                                           param_dict['n2'], N2_init, param_dict['k'], mech_params)
    if np.any(pdf_init32 <= 0) or np.any(np.isnan(pdf_init32)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_init32)) / len(data_initial32)

    return total_nll


def get_rounded_parameters(params, mechanism_info):
    """
    Get rounded parameters for display/comparison.
    """
    param_dict = unpack_parameters(params, mechanism_info)

    rounded = {}
    # Round n and N values to 1 decimal
    for key in ['n1', 'n2', 'n3', 'N1', 'N2', 'N3']:
        rounded[key] = round(param_dict[key], 1)

    # Round k to 3 decimals
    rounded['k'] = round(param_dict['k'], 3)

    # Round other parameters to 2 decimals
    for key in param_dict:
        if key not in rounded:
            rounded[key] = round(param_dict[key], 2)

    return rounded


def compute_moments_for_mechanism(mechanism, n_i, N_i, n_j, N_j, k, mech_params):
    """
    Compute moments for any mechanism with appropriate parameters.
    """
    if mechanism == 'simple':
        return compute_moments_mom(mechanism, n_i, N_i, n_j, N_j, k)
    elif mechanism == 'fixed_burst':
        return compute_moments_mom(mechanism, n_i, N_i, n_j, N_j, k,
                                   burst_size=mech_params['burst_size'])
    elif mechanism == 'time_varying_k':
        return compute_moments_mom(mechanism, n_i, N_i, n_j, N_j, k,
                                   k_1=mech_params['k_1'])
    elif mechanism == 'feedback':
        return compute_moments_mom(mechanism, n_i, N_i, n_j, N_j, k,
                                   feedbackSteepness=mech_params['feedbackSteepness'],
                                   feedbackThreshold=mech_params['feedbackThreshold'])
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")


def get_mechanism_info(mechanism):
    """
    Get mechanism-specific parameter information.

    Args:
        mechanism (str): 'simple', 'fixed_burst', 'time_varying_k', or 'feedback'

    Returns:
        dict: Contains parameter names, bounds, and default indices
    """
    # Common parameters for all mechanisms
    common_params = ['n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23']
    common_bounds = [
        (3, 25),      # n2 - reduced upper bound to ensure n < N constraints
        (100, 500),   # N2 - increased lower bound for better ratio
        (0.005, 0.4),  # k
        (0.3, 2.0),   # r21 - reduced upper bound to prevent n1 > N1
        (0.3, 2.0),   # r23 - reduced upper bound to prevent n3 > N3
        (0.5, 2.5),   # R21 - increased lower bound for better N1/n1 ratio
        (0.5, 5.0),   # R23 - increased lower bound for better N3/n3 ratio
    ]

    # Mutant parameters (always present)
    mutant_params = ['alpha', 'beta_k', 'gamma']
    mutant_bounds = [
        (0.2, 0.9),   # alpha - constrained to ensure n*alpha < N
        (0.2, 0.9),   # beta_k
        (0.1, 0.9),   # gamma - constrained to ensure n < N*gamma
    ]

    if mechanism == 'simple':
        mechanism_params = []
        mechanism_bounds = []
    elif mechanism == 'fixed_burst':
        mechanism_params = ['burst_size']
        mechanism_bounds = [(1, 15)]
    elif mechanism == 'time_varying_k':
        mechanism_params = ['k_1']
        mechanism_bounds = [(0.00001, 0.02)]
    elif mechanism == 'feedback':
        mechanism_params = ['feedbackSteepness', 'feedbackThreshold']
        mechanism_bounds = [(0.01, 0.1), (50, 150)]
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

    all_params = common_params + mechanism_params + mutant_params
    all_bounds = common_bounds + mechanism_bounds + mutant_bounds

    return {
        'params': all_params,
        'bounds': all_bounds,
        'common_count': len(common_params),
        'mechanism_count': len(mechanism_params),
        'mutant_count': len(mutant_params)
    }


def main():
    # ========== MECHANISM CONFIGURATION ==========
    # Choose mechanism: 'simple', 'fixed_burst', 'time_varying_k', 'feedback'
    mechanism = 'time_varying_k'  # Change this to test different mechanisms

    print(f"Optimizing for mechanism: {mechanism}")

    # Get mechanism-specific information
    mechanism_info = get_mechanism_info(mechanism)
    bounds = mechanism_info['bounds']

    print(f"Parameters to optimize: {mechanism_info['params']}")
    print(f"Number of parameters: {len(mechanism_info['params'])}")

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

    # c) Global optimization to find top 5 solutions
    population_solutions = []

    def callback(xk, convergence):
        population_solutions.append((joint_objective(xk, mechanism, mechanism_info, data_wt12, data_wt32,
                                                     data_threshold12, data_threshold32,
                                                     data_degrate12, data_degrate32,
                                                     data_initial12, data_initial32), xk.copy()))

    result = differential_evolution(
        joint_objective,
        bounds=bounds,
        args=(mechanism, mechanism_info, data_wt12, data_wt32, data_threshold12, data_threshold32,
              data_degrate12, data_degrate32, data_initial12, data_initial32),
        strategy='best1bin',
        maxiter=300,
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
        rounded_params = get_rounded_parameters(params, mechanism_info)
        rounded_tuple = tuple(rounded_params.values())
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
        param_dict = unpack_parameters(params, mechanism_info)
        print(f"Solution {i+1}: Negative Log-Likelihood = {nll:.4f}")

        # Print common parameters
        print(
            f"Parameters: n2 = {param_dict['n2']:.2f}, N2 = {param_dict['N2']:.2f}, k = {param_dict['k']:.4f}")
        print(
            f"Ratios: r21 = {param_dict['r21']:.2f}, r23 = {param_dict['r23']:.2f}, R21 = {param_dict['R21']:.2f}, R23 = {param_dict['R23']:.2f}")

        # Print mechanism-specific parameters
        if mechanism == 'fixed_burst':
            print(f"Burst size = {param_dict['burst_size']:.2f}")
        elif mechanism == 'time_varying_k':
            print(f"k_1 = {param_dict['k_1']:.4f}")
        elif mechanism == 'feedback':
            print(
                f"Feedback steepness = {param_dict['feedbackSteepness']:.3f}, threshold = {param_dict['feedbackThreshold']:.1f}")

        # Print mutant parameters
        print(
            f"Mutants: alpha = {param_dict['alpha']:.2f}, beta_k = {param_dict['beta_k']:.2f}, gamma = {param_dict['gamma']:.2f}")
        print(
            f"Derived: n1 = {param_dict['n1']:.2f}, n3 = {param_dict['n3']:.2f}, N1 = {param_dict['N1']:.2f}, N3 = {param_dict['N3']:.2f}")
        print()

    # d) Local optimization to refine top 5 solutions
    refined_solutions = []
    for i, (_, params) in enumerate(top_5_solutions):
        result_local = minimize(
            joint_objective,
            x0=params,
            args=(mechanism, mechanism_info, data_wt12, data_wt32, data_threshold12, data_threshold32,
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
    param_dict = unpack_parameters(best_params, mechanism_info)

    # Extract mechanism-specific parameters for computations
    mech_params = {}
    if mechanism == 'fixed_burst':
        mech_params['burst_size'] = param_dict['burst_size']
    elif mechanism == 'time_varying_k':
        mech_params['k_1'] = param_dict['k_1']
    elif mechanism == 'feedback':
        mech_params['feedbackSteepness'] = param_dict['feedbackSteepness']
        mech_params['feedbackThreshold'] = param_dict['feedbackThreshold']

    # Compute individual negative log-likelihoods for reporting
    wt_nll = 0
    pdf_wt12 = compute_pdf_for_mechanism(mechanism, data_wt12, param_dict['n1'], param_dict['N1'],
                                         param_dict['n2'], param_dict['N2'], param_dict['k'], mech_params)
    if not (np.any(pdf_wt12 <= 0) or np.any(np.isnan(pdf_wt12))):
        wt_nll -= np.sum(np.log(pdf_wt12))
    pdf_wt32 = compute_pdf_for_mechanism(mechanism, data_wt32, param_dict['n3'], param_dict['N3'],
                                         param_dict['n2'], param_dict['N2'], param_dict['k'], mech_params)
    if not (np.any(pdf_wt32 <= 0) or np.any(np.isnan(pdf_wt32))):
        wt_nll -= np.sum(np.log(pdf_wt32))

    threshold_nll = 0
    n1_th = max(param_dict['n1'] * param_dict['alpha'], 1)
    n2_th = max(param_dict['n2'] * param_dict['alpha'], 1)
    n3_th = max(param_dict['n3'] * param_dict['alpha'], 1)
    pdf_th12 = compute_pdf_for_mechanism(mechanism, data_threshold12, n1_th, param_dict['N1'],
                                         n2_th, param_dict['N2'], param_dict['k'], mech_params)
    if not (np.any(pdf_th12 <= 0) or np.any(np.isnan(pdf_th12))):
        threshold_nll -= np.sum(np.log(pdf_th12))
    pdf_th32 = compute_pdf_for_mechanism(mechanism, data_threshold32, n3_th, param_dict['N3'],
                                         n2_th, param_dict['N2'], param_dict['k'], mech_params)
    if not (np.any(pdf_th32 <= 0) or np.any(np.isnan(pdf_th32))):
        threshold_nll -= np.sum(np.log(pdf_th32))

    degrate_nll = 0
    k_deg = max(param_dict['beta_k'] * param_dict['k'], 0.001)
    pdf_deg12 = compute_pdf_for_mechanism(mechanism, data_degrate12, param_dict['n1'], param_dict['N1'],
                                          param_dict['n2'], param_dict['N2'], k_deg, mech_params)
    if not (np.any(pdf_deg12 <= 0) or np.any(np.isnan(pdf_deg12))):
        degrate_nll -= np.sum(np.log(pdf_deg12))
    pdf_deg32 = compute_pdf_for_mechanism(mechanism, data_degrate32, param_dict['n3'], param_dict['N3'],
                                          param_dict['n2'], param_dict['N2'], k_deg, mech_params)
    if not (np.any(pdf_deg32 <= 0) or np.any(np.isnan(pdf_deg32))):
        degrate_nll -= np.sum(np.log(pdf_deg32))

    initial_nll = 0
    N1_init = max(param_dict['N1'] * param_dict['gamma'], 1)
    N2_init = max(param_dict['N2'] * param_dict['gamma'], 1)
    N3_init = max(param_dict['N3'] * param_dict['gamma'], 1)
    pdf_init12 = compute_pdf_for_mechanism(mechanism, data_initial12, param_dict['n1'], N1_init,
                                           param_dict['n2'], N2_init, param_dict['k'], mech_params)
    if not (np.any(pdf_init12 <= 0) or np.any(np.isnan(pdf_init12))):
        initial_nll -= np.sum(np.log(pdf_init12))
    pdf_init32 = compute_pdf_for_mechanism(mechanism, data_initial32, param_dict['n3'], N3_init,
                                           param_dict['n2'], N2_init, param_dict['k'], mech_params)
    if not (np.any(pdf_init32 <= 0) or np.any(np.isnan(pdf_init32))):
        initial_nll -= np.sum(np.log(pdf_init32))

    # f) Print best solution
    print("\nBest Overall Solution:")
    print(f"Mechanism: {mechanism}")
    print(f"Total Negative Log-Likelihood: {best_nll:.4f}")
    print(f"Wild-Type Negative Log-Likelihood: {wt_nll:.4f}")
    print(f"Threshold Mutant Negative Log-Likelihood: {threshold_nll:.4f}")
    print(
        f"Degradation Rate Mutant Negative Log-Likelihood: {degrate_nll:.4f}")
    print(
        f"Initial Proteins Mutant Negative Log-Likelihood: {initial_nll:.4f}")

    # Print parameters
    print(
        f"Wild-Type Parameters: n1 = {param_dict['n1']:.2f}, n2 = {param_dict['n2']:.2f}, n3 = {param_dict['n3']:.2f}")
    print(
        f"                     N1 = {param_dict['N1']:.2f}, N2 = {param_dict['N2']:.2f}, N3 = {param_dict['N3']:.2f}, k = {param_dict['k']:.4f}")

    if mechanism == 'fixed_burst':
        print(
            f"                     burst_size = {param_dict['burst_size']:.2f}")
    elif mechanism == 'time_varying_k':
        print(f"                     k_1 = {param_dict['k_1']:.4f}")
    elif mechanism == 'feedback':
        print(
            f"                     feedbackSteepness = {param_dict['feedbackSteepness']:.3f}, feedbackThreshold = {param_dict['feedbackThreshold']:.1f}")

    print(f"Threshold Mutant: alpha = {param_dict['alpha']:.2f}")
    print(f"Degradation Rate Mutant: beta_k = {param_dict['beta_k']:.2f}")
    print(f"Initial Proteins Mutant: gamma = {param_dict['gamma']:.2f}")

    # g) Save optimized parameters to a text file
    filename = f"optimized_parameters_{mechanism}_join.txt"
    with open(filename, "w") as f:
        f.write(f"# Mechanism: {mechanism}\n")
        f.write("# Wild-Type Parameters\n")
        f.write(f"n1: {param_dict['n1']:.6f}\n")
        f.write(f"n2: {param_dict['n2']:.6f}\n")
        f.write(f"n3: {param_dict['n3']:.6f}\n")
        f.write(f"N1: {param_dict['N1']:.6f}\n")
        f.write(f"N2: {param_dict['N2']:.6f}\n")
        f.write(f"N3: {param_dict['N3']:.6f}\n")
        f.write(f"k: {param_dict['k']:.6f}\n")

        if mechanism == 'fixed_burst':
            f.write(f"burst_size: {param_dict['burst_size']:.6f}\n")
        elif mechanism == 'time_varying_k':
            f.write(f"k_1: {param_dict['k_1']:.6f}\n")
        elif mechanism == 'feedback':
            f.write(
                f"feedbackSteepness: {param_dict['feedbackSteepness']:.6f}\n")
            f.write(
                f"feedbackThreshold: {param_dict['feedbackThreshold']:.6f}\n")

        f.write(f"wt_nll: {wt_nll:.6f}\n")
        f.write("# Mutant Parameters\n")
        f.write(f"alpha: {param_dict['alpha']:.6f}\n")
        f.write(f"beta_k: {param_dict['beta_k']:.6f}\n")
        f.write(f"gamma: {param_dict['gamma']:.6f}\n")
        f.write(f"threshold_nll: {threshold_nll:.6f}\n")
        f.write(f"degrate_nll: {degrate_nll:.6f}\n")
        f.write(f"initial_nll: {initial_nll:.6f}\n")
        f.write(f"total_nll: {best_nll:.6f}\n")

    print(f"Optimized parameters saved to {filename}")


if __name__ == "__main__":
    main()
