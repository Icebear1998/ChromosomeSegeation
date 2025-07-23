import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize, basinhopping
from scipy.stats import norm
from MoMCalculations import compute_pdf_for_mechanism


def unpack_wildtype_parameters(params, mechanism_info):
    """
    Unpack wild-type optimization parameters based on mechanism.
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


def wildtype_objective(vars_, mechanism, mechanism_info, data12, data32):
    """
    Wild-type objective function for any mechanism.

    Args:
        vars_: Parameters to optimize
        mechanism: Mechanism type
        mechanism_info: Mechanism information
        data12, data32: Wild-type data
    """
    # Unpack parameters
    param_dict = unpack_wildtype_parameters(vars_, mechanism_info)

    # Validate constraints: n_i < N_i for all chromosomes
    if param_dict['n1'] >= param_dict['N1']:
        return np.inf
    if param_dict['n2'] >= param_dict['N2']:
        return np.inf
    if param_dict['n3'] >= param_dict['N3']:
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

    # Validate mechanism-specific inputs
    if mechanism in ['fixed_burst', 'fixed_burst_feedback_linear', 'fixed_burst_feedback_onion'] and mech_params['burst_size'] <= 0:
        return np.inf
    if param_dict['k'] <= 0:
        return np.inf

    # Sample sizes
    n12 = len(data12)
    n32 = len(data32)

    # Weight (adjustable, default 1.0)
    weight = 1.0

    # Chrom1–Chrom2
    pdf12 = compute_pdf_for_mechanism(mechanism, data12, param_dict['n1'], param_dict['N1'],
                                      param_dict['n2'], param_dict['N2'], param_dict['k'], mech_params, pair12=True)
    if np.any(pdf12 <= 0) or np.any(np.isnan(pdf12)):
        return np.inf
    log_likelihood12 = np.sum(np.log(pdf12)) / n12

    # Chrom3–Chrom2
    pdf32 = compute_pdf_for_mechanism(mechanism, data32, param_dict['n3'], param_dict['N3'],
                                      param_dict['n2'], param_dict['N2'], param_dict['k'], mech_params, pair12=False)
    if np.any(pdf32 <= 0) or np.any(np.isnan(pdf32)):
        return np.inf
    log_likelihood32 = np.sum(np.log(pdf32)) / n32

    total_nll = -weight * (log_likelihood12 + log_likelihood32)

    return total_nll


def threshold_objective(vars_, mechanism, data12, data32, params_baseline):
    """
    Threshold mutant objective function for any mechanism.
    """
    n1_wt, n2_wt, n3_wt, N1_wt, N2_wt, N3_wt, k_wt, mech_params = params_baseline
    alpha = vars_[0]
    n1 = max(n1_wt * alpha, 1)
    n2 = max(n2_wt * alpha, 1)
    n3 = max(n3_wt * alpha, 1)

    # Validate constraints
    if n1 >= N1_wt or n2 >= N2_wt or n3 >= N3_wt:
        return np.inf
    if alpha <= 0 or k_wt <= 0:
        return np.inf

    # Sample sizes
    n12 = len(data12)
    n32 = len(data32)
    weight = 1.0

    # Chrom1–Chrom2
    pdf12 = compute_pdf_for_mechanism(
        mechanism, data12, n1, N1_wt, n2, N2_wt, k_wt, mech_params, pair12=True)
    if np.any(pdf12 <= 0) or np.any(np.isnan(pdf12)):
        return np.inf
    log_likelihood12 = np.sum(np.log(pdf12)) / n12

    # Chrom3–Chrom2
    pdf32 = compute_pdf_for_mechanism(
        mechanism, data32, n3, N3_wt, n2, N2_wt, k_wt, mech_params, pair12=False)
    if np.any(pdf32 <= 0) or np.any(np.isnan(pdf32)):
        return np.inf
    log_likelihood32 = np.sum(np.log(pdf32)) / n32

    return -weight * (log_likelihood12 + log_likelihood32)


def degrate_objective(vars_, mechanism, data12, data32, params_baseline):
    """
    Degradation rate mutant objective function for any mechanism.
    """
    n1_wt, n2_wt, n3_wt, N1_wt, N2_wt, N3_wt, k_wt, mech_params = params_baseline
    beta_k = vars_[0]
    k = max(beta_k * k_wt, 0.001)
    if (beta_k * k_wt < 0.001):
        print("Warning: k is too small, setting to 0.001")

    # Validate inputs
    if k <= 0:
        return np.inf

    # Sample sizes
    n12 = len(data12)
    n32 = len(data32)
    weight = 1.0

    # Chrom1–Chrom2
    pdf12 = compute_pdf_for_mechanism(
        mechanism, data12, n1_wt, N1_wt, n2_wt, N2_wt, k, mech_params, pair12=True)
    if np.any(pdf12 <= 0) or np.any(np.isnan(pdf12)):
        return np.inf
    log_likelihood12 = np.sum(np.log(pdf12)) / n12

    # Chrom3–Chrom2
    pdf32 = compute_pdf_for_mechanism(
        mechanism, data32, n3_wt, N3_wt, n2_wt, N2_wt, k, mech_params, pair12=False)
    if np.any(pdf32 <= 0) or np.any(np.isnan(pdf32)):
        return np.inf
    log_likelihood32 = np.sum(np.log(pdf32)) / n32

    return -weight * (log_likelihood12 + log_likelihood32)


def initial_proteins_objective(vars_, mechanism, data12, data32, params_baseline, gamma_mode='unified'):
    """
    Initial proteins mutant objective function for any mechanism.
    """
    n1_wt, n2_wt, n3_wt, N1_wt, N2_wt, N3_wt, k_wt, mech_params = params_baseline
    
    if gamma_mode == 'unified':
        gamma = vars_[0]
        N1 = max(N1_wt * gamma, 1)
        N2 = max(N2_wt * gamma, 1)
        N3 = max(N3_wt * gamma, 1)
        # Validate constraints
        if gamma <= 0 or k_wt <= 0:
            return np.inf
    elif gamma_mode == 'separate':
        gamma1, gamma2, gamma3 = vars_[0], vars_[1], vars_[2]
        N1 = max(N1_wt * gamma1, 1)
        N2 = max(N2_wt * gamma2, 1)
        N3 = max(N3_wt * gamma3, 1)
        # Validate constraints
        if gamma1 <= 0 or gamma2 <= 0 or gamma3 <= 0 or k_wt <= 0:
            return np.inf
    else:
        raise ValueError(f"Unknown gamma_mode: {gamma_mode}")

    # Validate constraints
    if n1_wt >= N1 or n2_wt >= N2 or n3_wt >= N3:
        return np.inf

    # Sample sizes
    n12 = len(data12)
    n32 = len(data32)
    weight = 1.0

    # Chrom1–Chrom2
    pdf12 = compute_pdf_for_mechanism(
        mechanism, data12, n1_wt, N1, n2_wt, N2, k_wt, mech_params, pair12=True)
    if np.any(pdf12 <= 0) or np.any(np.isnan(pdf12)):
        return np.inf
    log_likelihood12 = np.sum(np.log(pdf12)) / n12

    # Chrom3–Chrom2
    pdf32 = compute_pdf_for_mechanism(
        mechanism, data32, n3_wt, N3, n2_wt, N2, k_wt, mech_params, pair12=False)
    if np.any(pdf32 <= 0) or np.any(np.isnan(pdf32)):
        return np.inf
    log_likelihood32 = np.sum(np.log(pdf32)) / n32

    return -weight * (log_likelihood12 + log_likelihood32)


def get_rounded_parameters(params, mechanism_info):
    """
    Get rounded parameters for display/comparison.
    """
    param_dict = unpack_wildtype_parameters(params, mechanism_info)

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

    return tuple(rounded.values())


class BoundedStep:
    """Helper class for bounded steps in basinhopping."""

    def __init__(self, bounds, stepsize=0.5):
        self.bounds = np.array(bounds)
        self.stepsize = stepsize

    def __call__(self, x):
        x_new = x + np.random.uniform(-self.stepsize,
                                      self.stepsize, size=x.shape)
        return np.clip(x_new, self.bounds[:, 0], self.bounds[:, 1])


def get_mechanism_info(mechanism):
    """
    Get mechanism-specific parameter information.

    Args:
        mechanism (str): 'simple', 'fixed_burst', 'time_varying_k', 'feedback', 'feedback_linear', 'feedback_onion', 'feedback_zipper', 'fixed_burst_feedback_linear', or 'fixed_burst_feedback_onion'

    Returns:
        dict: Contains parameter names, bounds, and default indices
    """
    # Common wild-type parameters
    common_params = ['n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23']
    common_bounds = [
        (3, 50),      # n2
        (100, 500),   # N2
        (0.05, 0.4),  # k
        (0.5, 3.0),   # r21
        (0.5, 3.0),   # r23
        (0.4, 2.0),   # R21
        (0.5, 5.0),   # R23
    ]

    if mechanism == 'simple':
        mechanism_params = []
        mechanism_bounds = []
    elif mechanism == 'fixed_burst':
        mechanism_params = ['burst_size']
        mechanism_bounds = [(2, 10)]
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

    all_params = common_params + mechanism_params
    all_bounds = common_bounds + mechanism_bounds

    return {
        'params': all_params,
        'bounds': all_bounds,
        'common_count': len(common_params),
        'mechanism_count': len(mechanism_params)
    }


def main():
    # ========== MECHANISM CONFIGURATION ==========
    # Choose mechanism: 'simple', 'fixed_burst', 'time_varying_k', 'feedback', 'feedback_linear', 'feedback_onion', 'feedback_zipper', 'fixed_burst_feedback_linear', 'fixed_burst_feedback_onion'
    mechanism = 'fixed_burst_feedback_onion'  # Change this to test different mechanisms
    
    # ========== GAMMA CONFIGURATION ==========
    # Choose gamma mode: 'unified' for single gamma affecting all chromosomes, 'separate' for gamma1, gamma2, gamma3
    gamma_mode = 'unified'  # Change this to 'separate' for individual gamma per chromosome

    print(f"Independent optimization for mechanism: {mechanism}")
    print(f"Gamma mode: {gamma_mode}")

    # Get mechanism-specific information
    mechanism_info = get_mechanism_info(mechanism)
    wt_bounds = mechanism_info['bounds']

    print(f"Wild-type parameters to optimize: {mechanism_info['params']}")
    print(f"Number of wild-type parameters: {len(mechanism_info['params'])}")

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

    # c) Global optimization for wild-type to find top 5 solutions
    population_solutions = []

    def callback(xk, convergence):
        population_solutions.append(
            (wildtype_objective(xk, mechanism, mechanism_info, data_wt12, data_wt32), xk.copy()))

    result = differential_evolution(
        wildtype_objective,
        bounds=wt_bounds,
        args=(mechanism, mechanism_info, data_wt12, data_wt32),
        strategy='best1bin',
        maxiter=500,        # Increased from 300 to allow more iterations for complex mechanism
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
        param_dict = unpack_wildtype_parameters(params, mechanism_info)
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
                f"w1 = {param_dict['w1']:.4f}, w2 = {param_dict['w2']:.4f}, w3 = {param_dict['w3']:.4f}")
        elif mechanism == 'feedback_onion':
            print(
                f"n_inner = {param_dict['n_inner']:.2f}")
        elif mechanism == 'feedback_zipper':
            print(
                f"z1 = {param_dict['z1']:.2f}, z2 = {param_dict['z2']:.2f}, z3 = {param_dict['z3']:.2f}")
        elif mechanism == 'fixed_burst_feedback_linear':
            print(
                f"burst_size = {param_dict['burst_size']:.2f}, w1 = {param_dict['w1']:.4f}, w2 = {param_dict['w2']:.4f}, w3 = {param_dict['w3']:.4f}")
        elif mechanism == 'fixed_burst_feedback_onion':
            print(
                f"burst_size = {param_dict['burst_size']:.2f}, n_inner = {param_dict['n_inner']:.2f}")

        print(
            f"Derived: n1 = {param_dict['n1']:.2f}, n3 = {param_dict['n3']:.2f}, N1 = {param_dict['N1']:.2f}, N3 = {param_dict['N3']:.2f}")
        print()

    # d) Local optimization to refine top 5 wild-type solutions
    refined_wt_solutions = []
    for i, (_, params) in enumerate(top_5_solutions):
        result_local = minimize(
            wildtype_objective,
            x0=params,
            args=(mechanism, mechanism_info, data_wt12, data_wt32),
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
    n_mutant_bound = [(0.1, 0.9)]      # For alpha
    degrate_bound = [(0.1, 0.9)]       # For beta_k
    if gamma_mode == 'unified':
        N_mutant_bound = [(0.1, 0.35)]      # For gamma
    else:  # separate mode
        N_mutant_bound = [(0.1, 0.35), (0.1, 0.35), (0.1, 0.35)]  # For gamma1, gamma2, gamma3

    overall_results = []
    for wt_idx, (wt_nll, wt_params) in enumerate(refined_wt_solutions[:5]):
        param_dict = unpack_wildtype_parameters(wt_params, mechanism_info)

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

        params_baseline = (param_dict['n1'], param_dict['n2'], param_dict['n3'],
                           param_dict['N1'], param_dict['N2'], param_dict['N3'],
                           param_dict['k'], mech_params)

        # Compute unweighted wild-type NLL for reporting
        wt_nll_unweighted = - \
            wildtype_objective(wt_params, mechanism, mechanism_info, data_wt12, data_wt32)

        # Threshold mutant
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "args": (mechanism, data_threshold12, data_threshold32, params_baseline),
            "bounds": n_mutant_bound
        }
        result_threshold = basinhopping(
            threshold_objective,
            x0=np.array([0.5]),
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
            threshold_nll_unweighted = -threshold_objective([alpha], mechanism, data_threshold12,
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
            x0=np.array([0.5]),
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
            degrate_nll_unweighted = -degrate_objective([beta_k], mechanism, data_degrate12,
                                                        data_degrate32, params_baseline)
        else:
            degrate_nll = np.inf
            beta_k = np.nan
            degrate_nll_unweighted = np.inf

        # Initial proteins mutant
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "args": (mechanism, data_initial12, data_initial32, params_baseline, gamma_mode),
            "bounds": N_mutant_bound
        }
        if gamma_mode == 'unified':
            x0_initial = np.array([0.5])
        else:  # separate mode
            x0_initial = np.array([0.5, 0.5, 0.5])
        
        result_initial = basinhopping(
            initial_proteins_objective,
            x0=x0_initial,
            minimizer_kwargs=minimizer_kwargs,
            niter=100,
            T=1.0,
            stepsize=0.5,
            take_step=BoundedStep(N_mutant_bound),
            disp=False
        )
        if result_initial.lowest_optimization_result.success:
            initial_nll = result_initial.lowest_optimization_result.fun
            gamma_values = result_initial.lowest_optimization_result.x
            initial_nll_unweighted = -initial_proteins_objective(
                gamma_values, mechanism, data_initial12, data_initial32, params_baseline, gamma_mode)
            
            if gamma_mode == 'unified':
                gamma = gamma_values[0]
                gamma1 = gamma2 = gamma3 = np.nan
            else:  # separate mode
                gamma1, gamma2, gamma3 = gamma_values[0], gamma_values[1], gamma_values[2]
                gamma = np.nan
        else:
            initial_nll = np.inf
            if gamma_mode == 'unified':
                gamma = np.nan
                gamma1 = gamma2 = gamma3 = np.nan
            else:  # separate mode
                gamma1 = gamma2 = gamma3 = np.nan
                gamma = np.nan
            initial_nll_unweighted = np.inf

        # Total negative log-likelihood (weighted for optimization)
        total_nll = wt_nll + threshold_nll + degrate_nll + initial_nll
        # Total unweighted NLL for reporting
        total_nll_unweighted = wt_nll_unweighted + threshold_nll_unweighted + \
            degrate_nll_unweighted + initial_nll_unweighted
        result_dict = {
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
        }
        
        if gamma_mode == 'unified':
            result_dict['gamma'] = gamma
        else:  # separate mode
            result_dict['gamma1'] = gamma1
            result_dict['gamma2'] = gamma2
            result_dict['gamma3'] = gamma3
            
        overall_results.append(result_dict)

    # f) Select the best overall solution
    overall_results.sort(key=lambda x: x['total_nll'])
    best_result = overall_results[0]

    print("\nBest Overall Solution:")
    print(f"Mechanism: {mechanism}")
    print(
        f"Total Negative Log-Likelihood (Weighted): {best_result['total_nll']:.4f}")
    print(
        f"Total Negative Log-Likelihood (Unweighted): {best_result['total_nll_unweighted']:.4f}")
    print(f"Wild-Type Negative Log-Likelihood: {best_result['wt_nll']:.4f}")

    param_dict = unpack_wildtype_parameters(
        best_result['wt_params'], mechanism_info)
    print(
        f"Wild-Type Parameters: n1 = {param_dict['n1']:.2f}, n2 = {param_dict['n2']:.2f}, n3 = {param_dict['n3']:.2f}")
    print(
        f"                     N1 = {param_dict['N1']:.2f}, N2 = {param_dict['N2']:.2f}, N3 = {param_dict['N3']:.2f}, k = {param_dict['k']:.4f}")

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
            f"w1 = {param_dict['w1']:.4f}, w2 = {param_dict['w2']:.4f}, w3 = {param_dict['w3']:.4f}")
    elif mechanism == 'feedback_onion':
        print(
            f"n_inner = {param_dict['n_inner']:.2f}")
    elif mechanism == 'feedback_zipper':
        print(
            f"z1 = {param_dict['z1']:.2f}, z2 = {param_dict['z2']:.2f}, z3 = {param_dict['z3']:.2f}")
    elif mechanism == 'fixed_burst_feedback_linear':
        print(
            f"burst_size = {param_dict['burst_size']:.2f}, w1 = {param_dict['w1']:.4f}, w2 = {param_dict['w2']:.4f}, w3 = {param_dict['w3']:.4f}")
    elif mechanism == 'fixed_burst_feedback_onion':
        print(
            f"burst_size = {param_dict['burst_size']:.2f}, n_inner = {param_dict['n_inner']:.2f}")

    print(
        f"Threshold Mutant: NLL = {best_result['threshold_nll']:.4f}, alpha = {best_result['alpha']:.2f}")
    print(
        f"Degradation Rate Mutant: NLL = {best_result['degrate_nll']:.4f}, beta_k = {best_result['beta_k']:.2f}")
    if 'gamma' in best_result:  # unified mode
        print(
            f"Initial Proteins Mutant: NLL = {best_result['initial_nll']:.4f}, gamma = {best_result['gamma']:.2f}")
    else:  # separate mode
        print(
            f"Initial Proteins Mutant: NLL = {best_result['initial_nll']:.4f}, gamma1 = {best_result['gamma1']:.2f}, gamma2 = {best_result['gamma2']:.2f}, gamma3 = {best_result['gamma3']:.2f}")

    # g) Save optimized parameters to a text file
    filename = f"optimized_parameters_{mechanism}_independent.txt"
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

        f.write(f"wt_nll: {best_result['wt_nll']:.6f}\n")
        f.write("# Mutant Parameters\n")
        f.write(f"alpha: {best_result['alpha']:.6f}\n")
        f.write(f"beta_k: {best_result['beta_k']:.6f}\n")
        if 'gamma' in best_result:  # unified mode
            f.write(f"gamma: {best_result['gamma']:.6f}\n")
        else:  # separate mode
            f.write(f"gamma1: {best_result['gamma1']:.6f}\n")
            f.write(f"gamma2: {best_result['gamma2']:.6f}\n")
            f.write(f"gamma3: {best_result['gamma3']:.6f}\n")
        f.write(f"threshold_nll: {best_result['threshold_nll']:.6f}\n")
        f.write(f"degrate_nll: {best_result['degrate_nll']:.6f}\n")
        f.write(f"initial_nll: {best_result['initial_nll']:.6f}\n")
        f.write(f"total_nll: {best_result['total_nll_unweighted']:.6f}\n")

    print(f"Optimized parameters saved to {filename}")


if __name__ == "__main__":
    main()
