import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm
from MoMCalculations import compute_pdf_for_mechanism


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


def joint_objective(params, mechanism, mechanism_info, data_wt12, data_wt32, data_threshold12, data_threshold32,
                    data_degrate12, data_degrate32, data_initial12, data_initial32, data_degrateAPC12, data_degrateAPC32):
    """
    Joint objective function for all mechanisms.

    Args:
        params: Parameters to optimize
        mechanism: Mechanism type
        mechanism_info: Mechanism information
        data_wt12, data_wt32: Wild-type data
        data_threshold12, data_threshold32: Threshold mutant data
        data_degrate12, data_degrate32: Degradation rate mutant data (separase)
        data_initial12, data_initial32: Initial proteins mutant data
        data_degrateAPC12, data_degrateAPC32: Degradation rate mutant data (APC)
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
    if 'gamma' in param_dict:  # unified mode
        N1_init = max(param_dict['N1'] * param_dict['gamma'], 1)
        N2_init = max(param_dict['N2'] * param_dict['gamma'], 1)
        N3_init = max(param_dict['N3'] * param_dict['gamma'], 1)
    else:  # separate mode
        N1_init = max(param_dict['N1'] * param_dict['gamma1'], 1)
        N2_init = max(param_dict['N2'] * param_dict['gamma2'], 1)
        N3_init = max(param_dict['N3'] * param_dict['gamma3'], 1)

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
        mech_params['w3'] = param_dict['w3']
    elif mechanism == 'feedback_onion':
        mech_params['n_inner'] = param_dict['n_inner']
    elif mechanism == 'feedback_zipper':
        mech_params['z1'] = param_dict['z1']
        mech_params['z2'] = param_dict['z2']
        mech_params['z3'] = param_dict['z3']
    elif mechanism == 'fixed_burst_feedback_linear':
        mech_params['burst_size'] = param_dict['burst_size']
        mech_params['w1'] = param_dict['w1']
        mech_params['w2'] = param_dict['w2']
        mech_params['w3'] = param_dict['w3']
    elif mechanism == 'fixed_burst_feedback_onion':
        mech_params['burst_size'] = param_dict['burst_size']
        mech_params['n_inner'] = param_dict['n_inner']

    total_nll = 0.0

    # Wild-Type (Chrom1–Chrom2 and Chrom3–Chrom2)
    pdf_wt12 = compute_pdf_for_mechanism(mechanism, data_wt12, param_dict['n1'], param_dict['N1'],
                                         param_dict['n2'], param_dict['N2'], param_dict['k'], mech_params, pair12=True)
    if np.any(pdf_wt12 <= 0) or np.any(np.isnan(pdf_wt12)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_wt12)) / len(data_wt12)

    pdf_wt32 = compute_pdf_for_mechanism(mechanism, data_wt32, param_dict['n3'], param_dict['N3'],
                                         param_dict['n2'], param_dict['N2'], param_dict['k'], mech_params, pair12=False)
    if np.any(pdf_wt32 <= 0) or np.any(np.isnan(pdf_wt32)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_wt32)) / len(data_wt32)

    # Threshold Mutant
    pdf_th12 = compute_pdf_for_mechanism(mechanism, data_threshold12, n1_th, param_dict['N1'],
                                         n2_th, param_dict['N2'], param_dict['k'], mech_params, pair12=True)
    if np.any(pdf_th12 <= 0) or np.any(np.isnan(pdf_th12)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_th12)) / len(data_threshold12)

    pdf_th32 = compute_pdf_for_mechanism(mechanism, data_threshold32, n3_th, param_dict['N3'],
                                         n2_th, param_dict['N2'], param_dict['k'], mech_params, pair12=False)
    if np.any(pdf_th32 <= 0) or np.any(np.isnan(pdf_th32)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_th32)) / len(data_threshold32)

    # Degradation Rate Mutant
    k_deg = max(param_dict['beta_k'] * param_dict['k'], 0.001)
    if param_dict['beta_k'] * param_dict['k'] < 0.001:
        print("Warning: beta_k * k is less than 0.001, setting k_deg to 0.001")

    pdf_deg12 = compute_pdf_for_mechanism(mechanism, data_degrate12, param_dict['n1'], param_dict['N1'],
                                          param_dict['n2'], param_dict['N2'], k_deg, mech_params, pair12=True)
    if np.any(pdf_deg12 <= 0) or np.any(np.isnan(pdf_deg12)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_deg12)) / len(data_degrate12)

    pdf_deg32 = compute_pdf_for_mechanism(mechanism, data_degrate32, param_dict['n3'], param_dict['N3'],
                                          param_dict['n2'], param_dict['N2'], k_deg, mech_params, pair12=False)
    if np.any(pdf_deg32 <= 0) or np.any(np.isnan(pdf_deg32)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_deg32)) / len(data_degrate32)

    # Degradation Rate APC Mutant
    k_degAPC = max(param_dict['beta2_k'] * param_dict['k'], 0.001)
    if param_dict['beta2_k'] * param_dict['k'] < 0.001:
        print("Warning: beta2_k * k is less than 0.001, setting k_degAPC to 0.001")

    pdf_degAPC12 = compute_pdf_for_mechanism(mechanism, data_degrateAPC12, param_dict['n1'], param_dict['N1'],
                                          param_dict['n2'], param_dict['N2'], k_degAPC, mech_params, pair12=True)
    if np.any(pdf_degAPC12 <= 0) or np.any(np.isnan(pdf_degAPC12)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_degAPC12)) / len(data_degrateAPC12)

    pdf_degAPC32 = compute_pdf_for_mechanism(mechanism, data_degrateAPC32, param_dict['n3'], param_dict['N3'],
                                          param_dict['n2'], param_dict['N2'], k_degAPC, mech_params, pair12=False)
    if np.any(pdf_degAPC32 <= 0) or np.any(np.isnan(pdf_degAPC32)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_degAPC32)) / len(data_degrateAPC32)

    # Initial Proteins Mutant
    pdf_init12 = compute_pdf_for_mechanism(mechanism, data_initial12, param_dict['n1'], N1_init,
                                           param_dict['n2'], N2_init, param_dict['k'], mech_params, pair12=True)
    if np.any(pdf_init12 <= 0) or np.any(np.isnan(pdf_init12)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_init12)) / len(data_initial12)

    pdf_init32 = compute_pdf_for_mechanism(mechanism, data_initial32, param_dict['n3'], N3_init,
                                           param_dict['n2'], N2_init, param_dict['k'], mech_params, pair12=False)
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


def get_mechanism_info(mechanism, gamma_mode):
    """
    Get mechanism-specific parameter information.

    Args:
        mechanism (str): 'simple', 'fixed_burst', 'time_varying_k', 'feedback', 'feedback_linear', 'feedback_onion', 'feedback_zipper', 'fixed_burst_feedback_linear', or 'fixed_burst_feedback_onion'
        gamma_mode (str): 'unified' for single gamma affecting all chromosomes, 'separate' for gamma1, gamma2, gamma3

    Returns:
        dict: Contains parameter names, bounds, and default indices
    """
    # Common parameters for all mechanisms
    common_params = ['n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23']
    common_bounds = [
        (3, 50),      # n2
        (100, 500),   # N2
        (0.005, 0.4),  # k
        (0.5, 3.0),   # r21
        (0.5, 3.0),   # r23
        (0.4, 2.0),   # R21
        (0.5, 5.0),   # R23
    ]

    # Mutant parameters (always present)
    if gamma_mode == 'unified':
        mutant_params = ['alpha', 'beta_k', 'beta2_k', 'gamma']
        mutant_bounds = [
            (0.1, 0.9),   # alpha
            (0.1, 0.9),   # beta_k
            (0.1, 0.9),   # beta2_k
            (0.1, 0.35),   # gamma
        ]
    elif gamma_mode == 'separate':
        mutant_params = ['alpha', 'beta_k', 'beta2_k', 'gamma1', 'gamma2', 'gamma3']
        mutant_bounds = [
            (0.1, 0.9),   # alpha
            (0.1, 0.9),   # beta_k
            (0.1, 0.9),   # beta2_k
            (0.1, 0.35),   # gamma1
            (0.1, 0.35),   # gamma2
            (0.1, 0.35),   # gamma3
        ]
    else:
        raise ValueError(f"Unknown gamma_mode: {gamma_mode}. Use 'unified' or 'separate'.")

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
    elif mechanism == 'feedback_linear':
        mechanism_params = ['w1', 'w2', 'w3']
        mechanism_bounds = [
            (0.0001, 0.02),  # w1
            (0.0001, 0.02),  # w2
            (0.0001, 0.02),  # w3
        ]
    elif mechanism == 'feedback_onion':
        mechanism_params = ['n_inner']
        mechanism_bounds = [
            (5, 50),   # n_inner
        ]
    elif mechanism == 'feedback_zipper':
        mechanism_params = ['z1', 'z2', 'z3']
        mechanism_bounds = [
            (10, 100),  # z1
            (10, 100),  # z2
            (10, 100),  # z3
        ]
    elif mechanism == 'fixed_burst_feedback_linear':
        mechanism_params = ['burst_size', 'w1', 'w2', 'w3']
        mechanism_bounds = [
            (1, 20),         # burst_size
            (0.0001, 0.02),  # w1
            (0.0001, 0.02),  # w2
            (0.0001, 0.02),  # w3
        ]
    elif mechanism == 'fixed_burst_feedback_onion':
        mechanism_params = ['burst_size', 'n_inner']
        mechanism_bounds = [
            (1, 20),   # burst_size
            (5, 50),   # n_inner
        ]
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
    # Choose mechanism: 'simple', 'fixed_burst', 'time_varying_k', 'feedback', 'feedback_linear', 'feedback_onion', 'feedback_zipper', 'fixed_burst_feedback_linear', 'fixed_burst_feedback_onion'
    mechanism = 'feedback_onion'  # Auto-set by RunAllMechanisms.py
    
    # ========== GAMMA CONFIGURATION ==========
    # Choose gamma mode: 'unified' for single gamma affecting all chromosomes, 'separate' for gamma1, gamma2, gamma3
    gamma_mode = 'separate'  # Change this to 'separate' for individual gamma per chromosome

    print(f"Optimizing for mechanism: {mechanism}")
    print(f"Gamma mode: {gamma_mode}")

    # Get mechanism-specific information
    mechanism_info = get_mechanism_info(mechanism, gamma_mode)
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
    data_degrateAPC12 = df['degRateAPC12'].dropna().values
    data_degrateAPC32 = df['degRateAPC32'].dropna().values

    # c) Global optimization to find top 5 solutions
    population_solutions = []

    def callback(xk, convergence):
        population_solutions.append((joint_objective(xk, mechanism, mechanism_info, data_wt12, data_wt32,
                                                     data_threshold12, data_threshold32,
                                                     data_degrate12, data_degrate32,
                                                     data_initial12, data_initial32,
                                                     data_degrateAPC12, data_degrateAPC32), xk.copy()))

    result = differential_evolution(
        joint_objective,
        bounds=bounds,
        args=(mechanism, mechanism_info, data_wt12, data_wt32, data_threshold12, data_threshold32,
              data_degrate12, data_degrate32, data_initial12, data_initial32,
              data_degrateAPC12, data_degrateAPC32),
        strategy='best1bin',
        maxiter=400,        # Increased from 300 to allow more iterations for complex mechanism
        popsize=30,         # Increased from 15 to maintain better population diversity
        tol=1e-8,          # Decreased from 1e-6 for more precise convergence
        mutation=(0.5, 1.0),  # Added mutation range for better exploration
        recombination=0.7,   # Added recombination rate
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
        elif mechanism == 'feedback_linear':
            print(
                f"Feedback linear w1 = {param_dict['w1']:.3f}, w2 = {param_dict['w2']:.3f}, w3 = {param_dict['w3']:.3f}")
        elif mechanism == 'feedback_onion':
            print(
                f"n_inner = {param_dict['n_inner']:.2f}")
        elif mechanism == 'feedback_zipper':
            print(
                f"z1 = {param_dict['z1']:.2f}, z2 = {param_dict['z2']:.2f}, z3 = {param_dict['z3']:.2f}")
        elif mechanism == 'fixed_burst_feedback_onion':
            print(
                f"burst_size = {param_dict['burst_size']:.2f}, n_inner = {param_dict['n_inner']:.2f}")

        # Print mutant parameters
        if 'gamma' in param_dict:  # unified mode
            print(
                f"Mutants: alpha = {param_dict['alpha']:.2f}, beta_k = {param_dict['beta_k']:.2f}, beta2_k = {param_dict['beta2_k']:.2f}, gamma = {param_dict['gamma']:.2f}")
        else:  # separate mode
            print(
                f"Mutants: alpha = {param_dict['alpha']:.2f}, beta_k = {param_dict['beta_k']:.2f}, beta2_k = {param_dict['beta2_k']:.2f}, gamma1 = {param_dict['gamma1']:.2f}, gamma2 = {param_dict['gamma2']:.2f}, gamma3 = {param_dict['gamma3']:.2f}")
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
                  data_degrate12, data_degrate32, data_initial12, data_initial32,
                  data_degrateAPC12, data_degrateAPC32),
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
    elif mechanism == 'feedback_linear':
        mech_params['w1'] = param_dict['w1']
        mech_params['w2'] = param_dict['w2']
        mech_params['w3'] = param_dict['w3']
    elif mechanism == 'feedback_onion':
        mech_params['n_inner'] = param_dict['n_inner']
    elif mechanism == 'feedback_zipper':
        mech_params['z1'] = param_dict['z1']
        mech_params['z2'] = param_dict['z2']
        mech_params['z3'] = param_dict['z3']
    elif mechanism == 'fixed_burst_feedback_linear':
        mech_params['burst_size'] = param_dict['burst_size']
        mech_params['w1'] = param_dict['w1']
        mech_params['w2'] = param_dict['w2']
        mech_params['w3'] = param_dict['w3']
    elif mechanism == 'fixed_burst_feedback_onion':
        mech_params['burst_size'] = param_dict['burst_size']
        mech_params['n_inner'] = param_dict['n_inner']

    # Compute individual negative log-likelihoods for reporting
    wt_nll = 0
    pdf_wt12 = compute_pdf_for_mechanism(mechanism, data_wt12, param_dict['n1'], param_dict['N1'],
                                         param_dict['n2'], param_dict['N2'], param_dict['k'], mech_params, pair12=True)
    if not (np.any(pdf_wt12 <= 0) or np.any(np.isnan(pdf_wt12))):
        wt_nll -= np.sum(np.log(pdf_wt12))
    pdf_wt32 = compute_pdf_for_mechanism(mechanism, data_wt32, param_dict['n3'], param_dict['N3'],
                                         param_dict['n2'], param_dict['N2'], param_dict['k'], mech_params, pair12=False)
    if not (np.any(pdf_wt32 <= 0) or np.any(np.isnan(pdf_wt32))):
        wt_nll -= np.sum(np.log(pdf_wt32))

    threshold_nll = 0
    n1_th = max(param_dict['n1'] * param_dict['alpha'], 1)
    n2_th = max(param_dict['n2'] * param_dict['alpha'], 1)
    n3_th = max(param_dict['n3'] * param_dict['alpha'], 1)
    pdf_th12 = compute_pdf_for_mechanism(mechanism, data_threshold12, n1_th, param_dict['N1'],
                                         n2_th, param_dict['N2'], param_dict['k'], mech_params, pair12=True)
    if not (np.any(pdf_th12 <= 0) or np.any(np.isnan(pdf_th12))):
        threshold_nll -= np.sum(np.log(pdf_th12))
    pdf_th32 = compute_pdf_for_mechanism(mechanism, data_threshold32, n3_th, param_dict['N3'],
                                         n2_th, param_dict['N2'], param_dict['k'], mech_params, pair12=False)
    if not (np.any(pdf_th32 <= 0) or np.any(np.isnan(pdf_th32))):
        threshold_nll -= np.sum(np.log(pdf_th32))

    degrate_nll = 0
    k_deg = max(param_dict['beta_k'] * param_dict['k'], 0.001)
    pdf_deg12 = compute_pdf_for_mechanism(mechanism, data_degrate12, param_dict['n1'], param_dict['N1'],
                                          param_dict['n2'], param_dict['N2'], k_deg, mech_params, pair12=True)
    if not (np.any(pdf_deg12 <= 0) or np.any(np.isnan(pdf_deg12))):
        degrate_nll -= np.sum(np.log(pdf_deg12))
    pdf_deg32 = compute_pdf_for_mechanism(mechanism, data_degrate32, param_dict['n3'], param_dict['N3'],
                                          param_dict['n2'], param_dict['N2'], k_deg, mech_params, pair12=False)
    if not (np.any(pdf_deg32 <= 0) or np.any(np.isnan(pdf_deg32))):
        degrate_nll -= np.sum(np.log(pdf_deg32))

    degrateAPC_nll = 0
    k_degAPC = max(param_dict['beta2_k'] * param_dict['k'], 0.001)
    pdf_degAPC12 = compute_pdf_for_mechanism(mechanism, data_degrateAPC12, param_dict['n1'], param_dict['N1'],
                                          param_dict['n2'], param_dict['N2'], k_degAPC, mech_params, pair12=True)
    if not (np.any(pdf_degAPC12 <= 0) or np.any(np.isnan(pdf_degAPC12))):
        degrateAPC_nll -= np.sum(np.log(pdf_degAPC12))
    pdf_degAPC32 = compute_pdf_for_mechanism(mechanism, data_degrateAPC32, param_dict['n3'], param_dict['N3'],
                                          param_dict['n2'], param_dict['N2'], k_degAPC, mech_params, pair12=False)
    if not (np.any(pdf_degAPC32 <= 0) or np.any(np.isnan(pdf_degAPC32))):
        degrateAPC_nll -= np.sum(np.log(pdf_degAPC32))

    initial_nll = 0
    if 'gamma' in param_dict:  # unified mode
        N1_init = max(param_dict['N1'] * param_dict['gamma'], 1)
        N2_init = max(param_dict['N2'] * param_dict['gamma'], 1)
        N3_init = max(param_dict['N3'] * param_dict['gamma'], 1)
    else:  # separate mode
        N1_init = max(param_dict['N1'] * param_dict['gamma1'], 1)
        N2_init = max(param_dict['N2'] * param_dict['gamma2'], 1)
        N3_init = max(param_dict['N3'] * param_dict['gamma3'], 1)
    pdf_init12 = compute_pdf_for_mechanism(mechanism, data_initial12, param_dict['n1'], N1_init,
                                           param_dict['n2'], N2_init, param_dict['k'], mech_params, pair12=True)
    if not (np.any(pdf_init12 <= 0) or np.any(np.isnan(pdf_init12))):
        initial_nll -= np.sum(np.log(pdf_init12))
    pdf_init32 = compute_pdf_for_mechanism(mechanism, data_initial32, param_dict['n3'], N3_init,
                                           param_dict['n2'], N2_init, param_dict['k'], mech_params, pair12=False)
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
        f"Degradation Rate APC Mutant Negative Log-Likelihood: {degrateAPC_nll:.4f}")
    print(
        f"Initial Proteins Mutant Negative Log-Likelihood: {initial_nll:.4f}")

    # Print parameters
    print(
        f"Wild-Type Parameters: n1 = {param_dict['n1']:.2f}, n2 = {param_dict['n2']:.2f}, n3 = {param_dict['n3']:.2f}")
    print(
        f"N1 = {param_dict['N1']:.2f}, N2 = {param_dict['N2']:.2f}, N3 = {param_dict['N3']:.2f}, k = {param_dict['k']:.4f}")

    if mechanism == 'fixed_burst':
        print(
            f"burst_size = {param_dict['burst_size']:.2f}")
    elif mechanism == 'time_varying_k':
        print(f"k_1 = {param_dict['k_1']:.4f}")
    elif mechanism == 'feedback':
        print(
            f"feedbackSteepness = {param_dict['feedbackSteepness']:.3f}, feedbackThreshold = {param_dict['feedbackThreshold']:.1f}")
    elif mechanism == 'feedback_linear':
        print(
            f"w1 = {param_dict['w1']:.3f}, w2 = {param_dict['w2']:.3f}, w3 = {param_dict['w3']:.3f}")
    elif mechanism == 'feedback_onion':
        print(
            f"n_inner = {param_dict['n_inner']:.2f}")
    elif mechanism == 'feedback_zipper':
        print(
            f"z1 = {param_dict['z1']:.2f}, z2 = {param_dict['z2']:.2f}, z3 = {param_dict['z3']:.2f}")
    elif mechanism == 'fixed_burst_feedback_linear':
        print(
            f"burst_size = {param_dict['burst_size']:.2f}, w1 = {param_dict['w1']:.3f}, w2 = {param_dict['w2']:.3f}, w3 = {param_dict['w3']:.3f}")
    elif mechanism == 'fixed_burst_feedback_onion':
        print(
            f"burst_size = {param_dict['burst_size']:.2f}, n_inner = {param_dict['n_inner']:.2f}")

    print(f"Threshold Mutant: alpha = {param_dict['alpha']:.2f}")
    print(f"Degradation Rate Mutant: beta_k = {param_dict['beta_k']:.2f}")
    print(f"Degradation Rate APC Mutant: beta2_k = {param_dict['beta2_k']:.2f}")
    if 'gamma' in param_dict:  # unified mode
        print(f"Initial Proteins Mutant: gamma = {param_dict['gamma']:.2f}")
    else:  # separate mode
        print(f"Initial Proteins Mutant: gamma1 = {param_dict['gamma1']:.2f}, gamma2 = {param_dict['gamma2']:.2f}, gamma3 = {param_dict['gamma3']:.2f}")

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
        elif mechanism == 'feedback_linear':
            f.write(f"w1: {param_dict['w1']:.6f}\n")
            f.write(f"w2: {param_dict['w2']:.6f}\n")
            f.write(f"w3: {param_dict['w3']:.6f}\n")
        elif mechanism == 'feedback_onion':
            f.write(f"n_inner: {param_dict['n_inner']:.6f}\n")
        elif mechanism == 'feedback_zipper':
            f.write(f"z1: {param_dict['z1']:.6f}\n")
            f.write(f"z2: {param_dict['z2']:.6f}\n")
            f.write(f"z3: {param_dict['z3']:.6f}\n")
        elif mechanism == 'fixed_burst_feedback_linear':
            f.write(f"burst_size: {param_dict['burst_size']:.6f}\n")
            f.write(f"w1: {param_dict['w1']:.6f}\n")
            f.write(f"w2: {param_dict['w2']:.6f}\n")
            f.write(f"w3: {param_dict['w3']:.6f}\n")
        elif mechanism == 'fixed_burst_feedback_onion':
            f.write(f"burst_size: {param_dict['burst_size']:.6f}\n")
            f.write(f"n_inner: {param_dict['n_inner']:.6f}\n")

        f.write(f"wt_nll: {wt_nll:.6f}\n")
        f.write("# Mutant Parameters\n")
        f.write(f"alpha: {param_dict['alpha']:.6f}\n")
        f.write(f"beta_k: {param_dict['beta_k']:.6f}\n")
        f.write(f"beta2_k: {param_dict['beta2_k']:.6f}\n")
        if 'gamma' in param_dict:  # unified mode
            f.write(f"gamma: {param_dict['gamma']:.6f}\n")
        else:  # separate mode
            f.write(f"gamma1: {param_dict['gamma1']:.6f}\n")
            f.write(f"gamma2: {param_dict['gamma2']:.6f}\n")
            f.write(f"gamma3: {param_dict['gamma3']:.6f}\n")
        f.write(f"threshold_nll: {threshold_nll:.6f}\n")
        f.write(f"degrate_nll: {degrate_nll:.6f}\n")
        f.write(f"degrateAPC_nll: {degrateAPC_nll:.6f}\n")
        f.write(f"initial_nll: {initial_nll:.6f}\n")
        f.write(f"total_nll: {best_nll:.6f}\n")

    print(f"Optimized parameters saved to {filename}")


if __name__ == "__main__":
    main()
