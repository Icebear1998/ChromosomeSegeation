"""
Independent optimization for chromosome segregation mechanisms.

TEMPORARY MODIFICATION: Initial proteins strain is excluded from fitting.
Currently fits only: wildtype, threshold, degrate, degrateAPC datasets.

To re-enable initial strain fitting, search for "TEMPORARILY EXCLUDED" 
and restore the original initial_proteins_objective optimization code.
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize, basinhopping
from scipy.stats import norm
from MoMCalculations import compute_pdf_for_mechanism
from Chromosomes_Theory import (
    calculate_bootstrap_likelihood,
    calculate_weighted_likelihood,
    BootstrappingFitnessCalculator,
    analyze_dataset_sizes
)


def robust_log_likelihood(pdf_values, epsilon=1e-10, clip_range=(-20, 0)):
    """
    Calculate robust log-likelihood with numerical stability.
    
    Args:
        pdf_values: Array of PDF values
        epsilon: Small value to prevent log(0)
        clip_range: Range to clip log values to prevent extreme values
    
    Returns:
        Robust log-likelihood value
    """
    # Protect against zero/negative values
    protected_pdf = np.maximum(pdf_values, epsilon)
    
    # Calculate log-likelihood
    log_pdf = np.log(protected_pdf)
    
    # Clip extreme values
    log_pdf_clipped = np.clip(log_pdf, clip_range[0], clip_range[1])
    
    return np.sum(log_pdf_clipped)


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
    if mechanism in ['fixed_burst', 'fixed_burst_feedback_linear', 'fixed_burst_feedback_onion']:
        if 'burst_size' not in mech_params or mech_params['burst_size'] <= 0:
            return np.inf
        # Additional validation: burst_size should be reasonable
        if mech_params['burst_size'] > min(param_dict['N1'], param_dict['N2'], param_dict['N3']):
            return np.inf
    if param_dict['k'] <= 0:
        return np.inf
    
    # Additional numerical validation
    for key, value in param_dict.items():
        if np.isnan(value) or np.isinf(value):
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
    log_likelihood12 = robust_log_likelihood(pdf12) / n12

    # Chrom3–Chrom2
    pdf32 = compute_pdf_for_mechanism(mechanism, data32, param_dict['n3'], param_dict['N3'],
                                      param_dict['n2'], param_dict['N2'], param_dict['k'], mech_params, pair12=False)
    if np.any(pdf32 <= 0) or np.any(np.isnan(pdf32)):
        return np.inf
    log_likelihood32 = robust_log_likelihood(pdf32) / n32

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
    log_likelihood12 = robust_log_likelihood(pdf12) / n12

    # Chrom3–Chrom2
    pdf32 = compute_pdf_for_mechanism(
        mechanism, data32, n3, N3_wt, n2, N2_wt, k_wt, mech_params, pair12=False)
    if np.any(pdf32 <= 0) or np.any(np.isnan(pdf32)):
        return np.inf
    log_likelihood32 = robust_log_likelihood(pdf32) / n32

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
    log_likelihood12 = robust_log_likelihood(pdf12) / n12

    # Chrom3–Chrom2
    pdf32 = compute_pdf_for_mechanism(
        mechanism, data32, n3_wt, N3_wt, n2_wt, N2_wt, k, mech_params, pair12=False)
    if np.any(pdf32 <= 0) or np.any(np.isnan(pdf32)):
        return np.inf
    log_likelihood32 = robust_log_likelihood(pdf32) / n32

    return -weight * (log_likelihood12 + log_likelihood32)


def degrateAPC_objective(vars_, mechanism, data12, data32, params_baseline):
    """
    Degradation rate APC mutant objective function for any mechanism.
    """
    n1_wt, n2_wt, n3_wt, N1_wt, N2_wt, N3_wt, k_wt, mech_params = params_baseline
    beta2_k = vars_[0]
    k = max(beta2_k * k_wt, 0.001)
    if (beta2_k * k_wt < 0.001):
        print("Warning: beta2_k * k is too small, setting to 0.001")

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
    log_likelihood12 = robust_log_likelihood(pdf12) / n12

    # Chrom3–Chrom2
    pdf32 = compute_pdf_for_mechanism(
        mechanism, data32, n3_wt, N3_wt, n2_wt, N2_wt, k, mech_params, pair12=False)
    if np.any(pdf32 <= 0) or np.any(np.isnan(pdf32)):
        return np.inf
    log_likelihood32 = robust_log_likelihood(pdf32) / n32

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
    log_likelihood12 = robust_log_likelihood(pdf12) / n12

    # Chrom3–Chrom2
    pdf32 = compute_pdf_for_mechanism(
        mechanism, data32, n3_wt, N3, n2_wt, N2, k_wt, mech_params, pair12=False)
    if np.any(pdf32 <= 0) or np.any(np.isnan(pdf32)):
        return np.inf
    log_likelihood32 = robust_log_likelihood(pdf32) / n32

    return -weight * (log_likelihood12 + log_likelihood32)


def wildtype_objective_with_bootstrapping(vars_, mechanism, mechanism_info, data12, data32,
                                        bootstrap_method='bootstrap', target_sample_size=50,
                                        num_bootstrap_samples=100, random_seed=None):
    """
    Wild-type objective function with bootstrapping support.
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

    # Initialize bootstrapping calculator if needed
    if bootstrap_method in ['bootstrap', 'weighted']:
        bootstrap_calc = BootstrappingFitnessCalculator(
            target_sample_size=target_sample_size,
            num_bootstrap_samples=num_bootstrap_samples,
            random_seed=random_seed
        )

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

    # Additional validations
    if mechanism in ['fixed_burst', 'fixed_burst_feedback_linear', 'fixed_burst_feedback_onion']:
        if 'burst_size' not in mech_params or mech_params['burst_size'] <= 0:
            return np.inf
        if mech_params['burst_size'] > min(param_dict['N1'], param_dict['N2'], param_dict['N3']):
            return np.inf
    if param_dict['k'] <= 0:
        return np.inf
    
    for key, value in param_dict.items():
        if np.isnan(value) or np.isinf(value):
            return np.inf

    # Helper function to calculate likelihood with bootstrapping
    def calculate_dataset_likelihood(data_exp, pdf_theory):
        """Calculate likelihood using specified bootstrap method."""
        if bootstrap_method == 'bootstrap':
            # Create simulated data from theoretical PDF
            try:
                pdf_normalized = pdf_theory / np.sum(pdf_theory)
                simulated_data = np.random.choice(data_exp, size=len(data_exp)*3, p=pdf_normalized, replace=True)
                return bootstrap_calc.calculate_bootstrap_likelihood(data_exp, simulated_data)
            except:
                return 1e6
        elif bootstrap_method == 'weighted':
            try:
                pdf_normalized = pdf_theory / np.sum(pdf_theory)
                simulated_data = np.random.choice(data_exp, size=len(data_exp)*3, p=pdf_normalized, replace=True)
                return bootstrap_calc.calculate_weighted_likelihood(data_exp, simulated_data)
            except:
                return 1e6
        else:  # standard method
            if np.any(pdf_theory <= 0) or np.any(np.isnan(pdf_theory)):
                return 1e6
            return -robust_log_likelihood(pdf_theory) / len(data_exp)

    weight = 1.0
    total_nll = 0.0

    # Chrom1–Chrom2
    pdf12 = compute_pdf_for_mechanism(mechanism, data12, param_dict['n1'], param_dict['N1'],
                                      param_dict['n2'], param_dict['N2'], param_dict['k'], mech_params, pair12=True)
    
    if bootstrap_method == 'standard':
        if np.any(pdf12 <= 0) or np.any(np.isnan(pdf12)):
            return np.inf
        log_likelihood12 = robust_log_likelihood(pdf12) / len(data12)
    else:
        nll_12 = calculate_dataset_likelihood(data12, pdf12)
        if nll_12 >= 1e6:
            return np.inf
        log_likelihood12 = -nll_12

    # Chrom3–Chrom2
    pdf32 = compute_pdf_for_mechanism(mechanism, data32, param_dict['n3'], param_dict['N3'],
                                      param_dict['n2'], param_dict['N2'], param_dict['k'], mech_params, pair12=False)
    
    if bootstrap_method == 'standard':
        if np.any(pdf32 <= 0) or np.any(np.isnan(pdf32)):
            return np.inf
        log_likelihood32 = robust_log_likelihood(pdf32) / len(data32)
    else:
        nll_32 = calculate_dataset_likelihood(data32, pdf32)
        if nll_32 >= 1e6:
            return np.inf
        log_likelihood32 = -nll_32

    total_nll = -weight * (log_likelihood12 + log_likelihood32)
    return total_nll


def threshold_objective_with_bootstrapping(vars_, mechanism, data12, data32, params_baseline,
                                         bootstrap_method='bootstrap', target_sample_size=50,
                                         num_bootstrap_samples=100, random_seed=None):
    """
    Threshold mutant objective function with bootstrapping support.
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

    # Initialize bootstrapping calculator if needed
    if bootstrap_method in ['bootstrap', 'weighted']:
        bootstrap_calc = BootstrappingFitnessCalculator(
            target_sample_size=target_sample_size,
            num_bootstrap_samples=num_bootstrap_samples,
            random_seed=random_seed
        )

    # Helper function to calculate likelihood with bootstrapping
    def calculate_dataset_likelihood(data_exp, pdf_theory):
        """Calculate likelihood using specified bootstrap method."""
        if bootstrap_method == 'bootstrap':
            try:
                pdf_normalized = pdf_theory / np.sum(pdf_theory)
                simulated_data = np.random.choice(data_exp, size=len(data_exp)*3, p=pdf_normalized, replace=True)
                return bootstrap_calc.calculate_bootstrap_likelihood(data_exp, simulated_data)
            except:
                return 1e6
        elif bootstrap_method == 'weighted':
            try:
                pdf_normalized = pdf_theory / np.sum(pdf_theory)
                simulated_data = np.random.choice(data_exp, size=len(data_exp)*3, p=pdf_normalized, replace=True)
                return bootstrap_calc.calculate_weighted_likelihood(data_exp, simulated_data)
            except:
                return 1e6
        else:  # standard method
            if np.any(pdf_theory <= 0) or np.any(np.isnan(pdf_theory)):
                return 1e6
            return -robust_log_likelihood(pdf_theory) / len(data_exp)

    weight = 1.0

    # Chrom1–Chrom2
    pdf12 = compute_pdf_for_mechanism(
        mechanism, data12, n1, N1_wt, n2, N2_wt, k_wt, mech_params, pair12=True)
    
    if bootstrap_method == 'standard':
        if np.any(pdf12 <= 0) or np.any(np.isnan(pdf12)):
            return np.inf
        log_likelihood12 = robust_log_likelihood(pdf12) / len(data12)
    else:
        nll_12 = calculate_dataset_likelihood(data12, pdf12)
        if nll_12 >= 1e6:
            return np.inf
        log_likelihood12 = -nll_12

    # Chrom3–Chrom2
    pdf32 = compute_pdf_for_mechanism(
        mechanism, data32, n3, N3_wt, n2, N2_wt, k_wt, mech_params, pair12=False)
    
    if bootstrap_method == 'standard':
        if np.any(pdf32 <= 0) or np.any(np.isnan(pdf32)):
            return np.inf
        log_likelihood32 = robust_log_likelihood(pdf32) / len(data32)
    else:
        nll_32 = calculate_dataset_likelihood(data32, pdf32)
        if nll_32 >= 1e6:
            return np.inf
        log_likelihood32 = -nll_32

    return -weight * (log_likelihood12 + log_likelihood32)


def degrate_objective_with_bootstrapping(vars_, mechanism, data12, data32, params_baseline,
                                       bootstrap_method='bootstrap', target_sample_size=50,
                                       num_bootstrap_samples=100, random_seed=None):
    """
    Degradation rate mutant objective function with bootstrapping support.
    """
    n1_wt, n2_wt, n3_wt, N1_wt, N2_wt, N3_wt, k_wt, mech_params = params_baseline
    beta_k = vars_[0]
    k = max(beta_k * k_wt, 0.001)
    if (beta_k * k_wt < 0.001):
        print("Warning: k is too small, setting to 0.001")

    # Validate inputs
    if k <= 0:
        return np.inf

    # Initialize bootstrapping calculator if needed
    if bootstrap_method in ['bootstrap', 'weighted']:
        bootstrap_calc = BootstrappingFitnessCalculator(
            target_sample_size=target_sample_size,
            num_bootstrap_samples=num_bootstrap_samples,
            random_seed=random_seed
        )

    # Helper function to calculate likelihood with bootstrapping
    def calculate_dataset_likelihood(data_exp, pdf_theory):
        """Calculate likelihood using specified bootstrap method."""
        if bootstrap_method == 'bootstrap':
            try:
                pdf_normalized = pdf_theory / np.sum(pdf_theory)
                simulated_data = np.random.choice(data_exp, size=len(data_exp)*3, p=pdf_normalized, replace=True)
                return bootstrap_calc.calculate_bootstrap_likelihood(data_exp, simulated_data)
            except:
                return 1e6
        elif bootstrap_method == 'weighted':
            try:
                pdf_normalized = pdf_theory / np.sum(pdf_theory)
                simulated_data = np.random.choice(data_exp, size=len(data_exp)*3, p=pdf_normalized, replace=True)
                return bootstrap_calc.calculate_weighted_likelihood(data_exp, simulated_data)
            except:
                return 1e6
        else:  # standard method
            if np.any(pdf_theory <= 0) or np.any(np.isnan(pdf_theory)):
                return 1e6
            return -robust_log_likelihood(pdf_theory) / len(data_exp)

    weight = 1.0

    # Chrom1–Chrom2
    pdf12 = compute_pdf_for_mechanism(
        mechanism, data12, n1_wt, N1_wt, n2_wt, N2_wt, k, mech_params, pair12=True)
    
    if bootstrap_method == 'standard':
        if np.any(pdf12 <= 0) or np.any(np.isnan(pdf12)):
            return np.inf
        log_likelihood12 = robust_log_likelihood(pdf12) / len(data12)
    else:
        nll_12 = calculate_dataset_likelihood(data12, pdf12)
        if nll_12 >= 1e6:
            return np.inf
        log_likelihood12 = -nll_12

    # Chrom3–Chrom2
    pdf32 = compute_pdf_for_mechanism(
        mechanism, data32, n3_wt, N3_wt, n2_wt, N2_wt, k, mech_params, pair12=False)
    
    if bootstrap_method == 'standard':
        if np.any(pdf32 <= 0) or np.any(np.isnan(pdf32)):
            return np.inf
        log_likelihood32 = robust_log_likelihood(pdf32) / len(data32)
    else:
        nll_32 = calculate_dataset_likelihood(data32, pdf32)
        if nll_32 >= 1e6:
            return np.inf
        log_likelihood32 = -nll_32

    return -weight * (log_likelihood12 + log_likelihood32)


def degrateAPC_objective_with_bootstrapping(vars_, mechanism, data12, data32, params_baseline,
                                          bootstrap_method='bootstrap', target_sample_size=50,
                                          num_bootstrap_samples=100, random_seed=None):
    """
    Degradation rate APC mutant objective function with bootstrapping support.
    """
    n1_wt, n2_wt, n3_wt, N1_wt, N2_wt, N3_wt, k_wt, mech_params = params_baseline
    beta2_k = vars_[0]
    k = max(beta2_k * k_wt, 0.001)
    if (beta2_k * k_wt < 0.001):
        print("Warning: beta2_k * k is too small, setting to 0.001")

    # Validate inputs
    if k <= 0:
        return np.inf

    # Initialize bootstrapping calculator if needed
    if bootstrap_method in ['bootstrap', 'weighted']:
        bootstrap_calc = BootstrappingFitnessCalculator(
            target_sample_size=target_sample_size,
            num_bootstrap_samples=num_bootstrap_samples,
            random_seed=random_seed
        )

    # Helper function to calculate likelihood with bootstrapping
    def calculate_dataset_likelihood(data_exp, pdf_theory):
        """Calculate likelihood using specified bootstrap method."""
        if bootstrap_method == 'bootstrap':
            try:
                pdf_normalized = pdf_theory / np.sum(pdf_theory)
                simulated_data = np.random.choice(data_exp, size=len(data_exp)*3, p=pdf_normalized, replace=True)
                return bootstrap_calc.calculate_bootstrap_likelihood(data_exp, simulated_data)
            except:
                return 1e6
        elif bootstrap_method == 'weighted':
            try:
                pdf_normalized = pdf_theory / np.sum(pdf_theory)
                simulated_data = np.random.choice(data_exp, size=len(data_exp)*3, p=pdf_normalized, replace=True)
                return bootstrap_calc.calculate_weighted_likelihood(data_exp, simulated_data)
            except:
                return 1e6
        else:  # standard method
            if np.any(pdf_theory <= 0) or np.any(np.isnan(pdf_theory)):
                return 1e6
            return -robust_log_likelihood(pdf_theory) / len(data_exp)

    weight = 1.0

    # Chrom1–Chrom2
    pdf12 = compute_pdf_for_mechanism(
        mechanism, data12, n1_wt, N1_wt, n2_wt, N2_wt, k, mech_params, pair12=True)
    
    if bootstrap_method == 'standard':
        if np.any(pdf12 <= 0) or np.any(np.isnan(pdf12)):
            return np.inf
        log_likelihood12 = robust_log_likelihood(pdf12) / len(data12)
    else:
        nll_12 = calculate_dataset_likelihood(data12, pdf12)
        if nll_12 >= 1e6:
            return np.inf
        log_likelihood12 = -nll_12

    # Chrom3–Chrom2
    pdf32 = compute_pdf_for_mechanism(
        mechanism, data32, n3_wt, N3_wt, n2_wt, N2_wt, k, mech_params, pair12=False)
    
    if bootstrap_method == 'standard':
        if np.any(pdf32 <= 0) or np.any(np.isnan(pdf32)):
            return np.inf
        log_likelihood32 = robust_log_likelihood(pdf32) / len(data32)
    else:
        nll_32 = calculate_dataset_likelihood(data32, pdf32)
        if nll_32 >= 1e6:
            return np.inf
        log_likelihood32 = -nll_32

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
        mechanism_bounds = [(1, 20)]  # Increased upper bound for more flexibility
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
    mechanism = 'fixed_burst_feedback_onion'  # Auto-set by RunAllMechanisms.py
    
    # ========== GAMMA CONFIGURATION ==========
    # Choose gamma mode: 'unified' for single gamma affecting all chromosomes, 'separate' for gamma1, gamma2, gamma3
    gamma_mode = 'separate'  # Change this to 'separate' for individual gamma per chromosome

    print(f"Independent optimization for mechanism: {mechanism}")
    print(f"Gamma mode: {gamma_mode}")
    print("NOTE: Initial proteins strain is TEMPORARILY EXCLUDED from fitting")
    print("Fitting datasets: wildtype, threshold, degrate, degrateAPC")

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
    data_degrateAPC12 = df['degRateAPC12'].dropna().values
    data_degrateAPC32 = df['degRateAPC32'].dropna().values

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
        maxiter=300,        # Restored original value
        popsize=30,         # Restored original value
        tol=1e-8,          # Restored original value
        mutation=(0.5, 1.0),
        recombination=0.7,
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
        print(f"\nAttempting local optimization for solution {i+1}...")
        
        # Test initial objective value
        initial_obj = wildtype_objective(params, mechanism, mechanism_info, data_wt12, data_wt32)
        print(f"Initial objective value: {initial_obj}")
        
        if np.isinf(initial_obj):
            print(f"Initial solution {i+1} has infinite objective value - skipping local optimization")
            continue
        
        result_local = minimize(
            wildtype_objective,
            x0=params,
            args=(mechanism, mechanism_info, data_wt12, data_wt32),
            method='L-BFGS-B',
            bounds=wt_bounds,
            options={'disp': True, 'maxiter': 1000}  # Enable display and increase iterations
        )
        
        if result_local.success:
            refined_wt_solutions.append((result_local.fun, result_local.x))
            print(f"✓ Local optimization succeeded for solution {i+1}")
        else:
            print(f"✗ Wild-type local optimization failed for solution {i+1}")
            print(f"  Termination message: {result_local.message}")
            print(f"  Final objective: {result_local.fun}")
            print(f"  Number of iterations: {result_local.nit}")

    refined_wt_solutions.sort(key=lambda x: x[0])
    if not refined_wt_solutions:
        print("Warning: No successful local optimizations. Using best differential evolution solutions instead.")
        # Use the top 5 differential evolution solutions as fallback
        refined_wt_solutions = top_5_solutions[:5]
        print(f"Proceeding with {len(refined_wt_solutions)} differential evolution solutions.")

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

        # Degradation rate APC mutant
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "args": (mechanism, data_degrateAPC12, data_degrateAPC32, params_baseline),
            "bounds": degrate_bound # Assuming the same bounds for beta2_k as beta_k
        }
        result_degrateAPC = basinhopping(
            degrateAPC_objective,
            x0=np.array([0.5]),
            minimizer_kwargs=minimizer_kwargs,
            niter=100,
            T=1.0,
            stepsize=0.5,
            take_step=BoundedStep(degrate_bound),
            disp=False
        )
        if result_degrateAPC.lowest_optimization_result.success:
            degrateAPC_nll = result_degrateAPC.lowest_optimization_result.fun
            beta2_k = result_degrateAPC.lowest_optimization_result.x[0]
            degrateAPC_nll_unweighted = -degrateAPC_objective([beta2_k], mechanism, data_degrateAPC12,
                                                              data_degrateAPC32, params_baseline)
        else:
            degrateAPC_nll = np.inf
            beta2_k = np.nan
            degrateAPC_nll_unweighted = np.inf

        # Initial proteins mutant - TEMPORARILY EXCLUDED FROM FITTING
        print("  Skipping initial proteins mutant optimization (temporarily excluded)")
        initial_nll = 0.0  # Set to 0 to exclude from total NLL
        initial_nll_unweighted = 0.0
        if gamma_mode == 'unified':
            gamma = np.nan  # Not fitted
            gamma1 = gamma2 = gamma3 = np.nan
        else:  # separate mode
            gamma1 = gamma2 = gamma3 = np.nan
            gamma = np.nan

        # Total negative log-likelihood (weighted for optimization)
        total_nll = wt_nll + threshold_nll + degrate_nll + degrateAPC_nll + initial_nll
        # Total unweighted NLL for reporting
        total_nll_unweighted = wt_nll_unweighted + threshold_nll_unweighted + \
            degrate_nll_unweighted + degrateAPC_nll_unweighted + initial_nll_unweighted
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
            'degrateAPC_nll': degrateAPC_nll_unweighted,
            'beta2_k': beta2_k,
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
    print(
        f"Degradation Rate APC Mutant: NLL = {best_result['degrateAPC_nll']:.4f}, beta2_k = {best_result['beta2_k']:.2f}")
    print("Initial Proteins Mutant: EXCLUDED FROM FITTING (temporarily)")

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
        f.write(f"beta2_k: {best_result['beta2_k']:.6f}\n")
        f.write("# Initial proteins mutant parameters - EXCLUDED FROM FITTING\n")
        f.write("# gamma: not_fitted\n")
        f.write("# gamma1: not_fitted\n")
        f.write("# gamma2: not_fitted\n")
        f.write("# gamma3: not_fitted\n")
        f.write(f"threshold_nll: {best_result['threshold_nll']:.6f}\n")
        f.write(f"degrate_nll: {best_result['degrate_nll']:.6f}\n")
        f.write(f"degrateAPC_nll: {best_result['degrateAPC_nll']:.6f}\n")
        f.write(f"initial_nll: {best_result['initial_nll']:.6f}  # EXCLUDED (set to 0.0)\n")
        f.write(f"total_nll: {best_result['total_nll_unweighted']:.6f}  # Excludes initial strain\n")

    print(f"Optimized parameters saved to {filename}")


def main_with_bootstrapping():
    """
    Main optimization routine with bootstrapping comparison for independent optimization.
    """
    # ========== MECHANISM CONFIGURATION ==========
    mechanism = 'fixed_burst_feedback_onion'  # Auto-set by RunAllMechanisms.py
    gamma_mode = 'separate'  # Change this to 'separate' for individual gamma per chromosome

    print(f"Independent bootstrapping optimization for mechanism: {mechanism}")
    print(f"Gamma mode: {gamma_mode}")
    print("NOTE: Initial proteins strain is TEMPORARILY EXCLUDED from fitting")
    print("Fitting datasets: wildtype, threshold, degrate, degrateAPC")

    # Get mechanism-specific information
    mechanism_info = get_mechanism_info(mechanism)
    wt_bounds = mechanism_info['bounds']

    print(f"Wild-type parameters to optimize: {mechanism_info['params']}")
    print(f"Number of wild-type parameters: {len(mechanism_info['params'])}")

    # Read data
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

    # Analyze dataset sizes
    datasets = {
        'wildtype': {'delta_t12': data_wt12, 'delta_t32': data_wt32},
        'threshold': {'delta_t12': data_threshold12, 'delta_t32': data_threshold32},
        'degrate': {'delta_t12': data_degrate12, 'delta_t32': data_degrate32},
        'degrateAPC': {'delta_t12': data_degrateAPC12, 'delta_t32': data_degrateAPC32}
    }
    
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    size_analysis = analyze_dataset_sizes(datasets)
    target_sample_size = size_analysis['recommended_target_size']

    print(f"\n{'='*60}")
    print("COMPARISON: Standard vs. Bootstrapping Methods (Independent)")
    print(f"{'='*60}")

    methods = [
        ('standard', 'Standard Independent Optimization'),
        ('weighted', 'Weighted Likelihood Optimization'),
        ('bootstrap', 'Bootstrap Optimization')
    ]

    wt_results = {}
    
    # Step 1: Optimize wildtype parameters with different methods
    for method, method_name in methods:
        print(f"\n{'-'*40}")
        print(f"WILDTYPE: {method_name.upper()}")
        print(f"{'-'*40}")
        
        try:
            # Choose objective function
            if method == 'standard':
                objective_func = wildtype_objective
                args = (mechanism, mechanism_info, data_wt12, data_wt32)
            else:
                objective_func = wildtype_objective_with_bootstrapping
                args = (mechanism, mechanism_info, data_wt12, data_wt32,
                       method, target_sample_size, 50, 42)  # Reduced bootstrap samples for testing

            # Run optimization
            result = differential_evolution(
                objective_func,
                bounds=wt_bounds,
                args=args,
                strategy='best1bin',
                maxiter=100,  # Reduced for testing
                popsize=15,   # Reduced for testing
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                disp=True,
                seed=42
            )

            if result.success:
                param_dict = unpack_wildtype_parameters(result.x, mechanism_info)
                wt_results[method] = {
                    'nll': result.fun,
                    'params': param_dict,
                    'wt_params_vector': result.x,
                    'success': True
                }
                
                print(f"✓ Wildtype {method_name} completed successfully")
                print(f"  Negative Log-Likelihood: {result.fun:.4f}")
                print(f"  Parameters: n2={param_dict['n2']:.1f}, N2={param_dict['N2']:.1f}, k={param_dict['k']:.4f}")
                
            else:
                wt_results[method] = {'success': False, 'nll': np.inf}
                print(f"✗ Wildtype {method_name} failed: {result.message}")
                
        except Exception as e:
            wt_results[method] = {'success': False, 'nll': np.inf}
            print(f"✗ Error in wildtype {method_name}: {e}")

    # Step 2: For successful wildtype optimizations, optimize mutants
    successful_wt_results = {k: v for k, v in wt_results.items() if v['success']}
    
    if not successful_wt_results:
        print("No successful wildtype optimizations. Cannot proceed with mutant optimization.")
        return

    final_results = {}
    
    for method, wt_result in successful_wt_results.items():
        print(f"\n{'-'*40}")
        print(f"MUTANTS: {method.upper()} METHOD")
        print(f"{'-'*40}")
        
        try:
            param_dict = wt_result['params']
            
            # Extract mechanism-specific parameters
            mech_params = {}
            if mechanism == 'fixed_burst_feedback_onion':
                mech_params['burst_size'] = param_dict['burst_size']
                mech_params['n_inner'] = param_dict['n_inner']

            params_baseline = (param_dict['n1'], param_dict['n2'], param_dict['n3'],
                              param_dict['N1'], param_dict['N2'], param_dict['N3'],
                              param_dict['k'], mech_params)

            # Mutant bounds
            n_mutant_bound = [(0.1, 0.9)]      # For alpha
            degrate_bound = [(0.1, 0.9)]       # For beta_k

            # Choose objective functions based on method
            if method == 'standard':
                threshold_obj = threshold_objective
                degrate_obj = degrate_objective
                degrateAPC_obj = degrateAPC_objective
                obj_args = (mechanism, )
            else:
                threshold_obj = threshold_objective_with_bootstrapping
                degrate_obj = degrate_objective_with_bootstrapping
                degrateAPC_obj = degrateAPC_objective_with_bootstrapping
                obj_args = (mechanism, )
                bootstrap_args = (method, target_sample_size, 50, 42)

            # Threshold mutant
            if method == 'standard':
                minimizer_kwargs = {
                    "method": "L-BFGS-B",
                    "args": obj_args + (data_threshold12, data_threshold32, params_baseline),
                    "bounds": n_mutant_bound
                }
            else:
                minimizer_kwargs = {
                    "method": "L-BFGS-B",
                    "args": obj_args + (data_threshold12, data_threshold32, params_baseline) + bootstrap_args,
                    "bounds": n_mutant_bound
                }
                
            result_threshold = basinhopping(
                threshold_obj,
                x0=np.array([0.5]),
                minimizer_kwargs=minimizer_kwargs,
                niter=50,  # Reduced for testing
                T=1.0,
                stepsize=0.5,
                take_step=BoundedStep(n_mutant_bound),
                disp=False
            )
            
            if result_threshold.lowest_optimization_result.success:
                alpha = result_threshold.lowest_optimization_result.x[0]
                threshold_nll = -result_threshold.lowest_optimization_result.fun
            else:
                alpha = np.nan
                threshold_nll = np.inf

            # Degradation rate mutant
            if method == 'standard':
                minimizer_kwargs = {
                    "method": "L-BFGS-B",
                    "args": obj_args + (data_degrate12, data_degrate32, params_baseline),
                    "bounds": degrate_bound
                }
            else:
                minimizer_kwargs = {
                    "method": "L-BFGS-B",
                    "args": obj_args + (data_degrate12, data_degrate32, params_baseline) + bootstrap_args,
                    "bounds": degrate_bound
                }
                
            result_degrate = basinhopping(
                degrate_obj,
                x0=np.array([0.5]),
                minimizer_kwargs=minimizer_kwargs,
                niter=50,  # Reduced for testing
                T=1.0,
                stepsize=0.5,
                take_step=BoundedStep(degrate_bound),
                disp=False
            )
            
            if result_degrate.lowest_optimization_result.success:
                beta_k = result_degrate.lowest_optimization_result.x[0]
                degrate_nll = -result_degrate.lowest_optimization_result.fun
            else:
                beta_k = np.nan
                degrate_nll = np.inf

            # Degradation rate APC mutant
            if method == 'standard':
                minimizer_kwargs = {
                    "method": "L-BFGS-B",
                    "args": obj_args + (data_degrateAPC12, data_degrateAPC32, params_baseline),
                    "bounds": degrate_bound
                }
            else:
                minimizer_kwargs = {
                    "method": "L-BFGS-B",
                    "args": obj_args + (data_degrateAPC12, data_degrateAPC32, params_baseline) + bootstrap_args,
                    "bounds": degrate_bound
                }
                
            result_degrateAPC = basinhopping(
                degrateAPC_obj,
                x0=np.array([0.5]),
                minimizer_kwargs=minimizer_kwargs,
                niter=50,  # Reduced for testing
                T=1.0,
                stepsize=0.5,
                take_step=BoundedStep(degrate_bound),
                disp=False
            )
            
            if result_degrateAPC.lowest_optimization_result.success:
                beta2_k = result_degrateAPC.lowest_optimization_result.x[0]
                degrateAPC_nll = -result_degrateAPC.lowest_optimization_result.fun
            else:
                beta2_k = np.nan
                degrateAPC_nll = np.inf

            # Calculate total NLL
            wt_nll = -wt_result['nll']
            total_nll = wt_nll + threshold_nll + degrate_nll + degrateAPC_nll
            
            final_results[method] = {
                'total_nll': total_nll,
                'wt_nll': wt_nll,
                'threshold_nll': threshold_nll,
                'degrate_nll': degrate_nll,
                'degrateAPC_nll': degrateAPC_nll,
                'params': param_dict,
                'alpha': alpha,
                'beta_k': beta_k,
                'beta2_k': beta2_k,
                'success': True
            }
            
            print(f"✓ {method.capitalize()} mutant optimization completed")
            print(f"  Total NLL: {total_nll:.4f}")
            print(f"  Mutant params: alpha={alpha:.3f}, beta_k={beta_k:.3f}, beta2_k={beta2_k:.3f}")
            
            # Save results
            suffix = f"_{method}" if method != 'standard' else "_standard"
            filename = f"optimized_parameters_{mechanism}_independent{suffix}.txt"
            
            with open(filename, "w") as f:
                f.write(f"# Mechanism: {mechanism}\n")
                f.write(f"# Method: Independent {method.capitalize()} Optimization\n")
                if method != 'standard':
                    f.write(f"# Target Sample Size: {target_sample_size}\n")
                f.write(f"# Total Negative Log-Likelihood: {total_nll:.6f}\n\n")
                
                f.write("# Wild-Type Parameters\n")
                for key in ['n1', 'n2', 'n3', 'N1', 'N2', 'N3', 'k']:
                    f.write(f"{key}: {param_dict[key]:.6f}\n")
                
                # Mechanism-specific parameters
                if mechanism == 'fixed_burst_feedback_onion':
                    f.write(f"burst_size: {param_dict['burst_size']:.6f}\n")
                    f.write(f"n_inner: {param_dict['n_inner']:.6f}\n")
                
                # Mutant parameters
                f.write("\n# Mutant Parameters\n")
                f.write(f"alpha: {alpha:.6f}\n")
                f.write(f"beta_k: {beta_k:.6f}\n")
                f.write(f"beta2_k: {beta2_k:.6f}\n")
                
                # Individual NLLs
                f.write(f"\n# Individual NLLs\n")
                f.write(f"wt_nll: {wt_nll:.6f}\n")
                f.write(f"threshold_nll: {threshold_nll:.6f}\n")
                f.write(f"degrate_nll: {degrate_nll:.6f}\n")
                f.write(f"degrateAPC_nll: {degrateAPC_nll:.6f}\n")
            
            print(f"  Results saved to: {filename}")
            
        except Exception as e:
            final_results[method] = {'success': False, 'total_nll': np.inf}
            print(f"✗ Error in mutant optimization for {method}: {e}")

    # Compare results
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON (Independent Optimization)")
    print(f"{'='*60}")
    
    successful_results = {k: v for k, v in final_results.items() if v['success']}
    
    if successful_results:
        for method, method_name in methods:
            if method in successful_results:
                total_nll = successful_results[method]['total_nll']
                print(f"{method_name:30}: {total_nll:.4f}")
        
        # Calculate improvements
        if 'standard' in successful_results:
            standard_nll = successful_results['standard']['total_nll']
            print(f"\nImprovement over standard:")
            
            for method in ['weighted', 'bootstrap']:
                if method in successful_results:
                    method_nll = successful_results[method]['total_nll']
                    if method_nll < standard_nll:
                        improvement = ((standard_nll - method_nll) / standard_nll) * 100
                        print(f"  {method.capitalize():10}: {improvement:.1f}% better")
                    else:
                        change = ((method_nll - standard_nll) / standard_nll) * 100
                        print(f"  {method.capitalize():10}: {change:.1f}% worse")
    else:
        print("No successful optimizations completed.")

    print(f"\n{'-' * 60}")
    print("Independent bootstrapping optimization comparison complete!")


if __name__ == "__main__":
    # Uncomment the line below to run bootstrapping comparison
    # main_with_bootstrapping()
    
    # Standard optimization
    main()
