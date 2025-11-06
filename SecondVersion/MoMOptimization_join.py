"""
Joint optimization for chromosome segregation mechanisms.

TEMPORARY MODIFICATION: Initial proteins strain is excluded from fitting.
Currently fits only: wildtype, threshold, degrate, degrateAPC datasets.

To re-enable initial strain fitting, search for "TEMPORARILY EXCLUDED" 
and restore the original initial proteins objective code.
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm
from MoMCalculations import compute_pdf_for_mechanism
from Chromosomes_Theory import (
    calculate_bootstrap_likelihood,
    calculate_weighted_likelihood,
    BootstrappingFitnessCalculator,
    analyze_dataset_sizes
)


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
                    data_degrate12, data_degrate32, data_initial12, data_initial32, data_degrateAPC12, data_degrateAPC32,
                    data_velcade12, data_velcade32):
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
        data_velcade12, data_velcade32: Velcade mutant data
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

    # Initial proteins mutant: TEMPORARILY EXCLUDED FROM FITTING
    # Constraints removed as initial strain is not being fitted

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
    total_nll -= np.sum(np.log(pdf_wt12))  # REMOVED: / len(data_wt12)

    pdf_wt32 = compute_pdf_for_mechanism(mechanism, data_wt32, param_dict['n3'], param_dict['N3'],
                                         param_dict['n2'], param_dict['N2'], param_dict['k'], mech_params, pair12=False)
    if np.any(pdf_wt32 <= 0) or np.any(np.isnan(pdf_wt32)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_wt32))  # REMOVED: / len(data_wt32)

    # Threshold Mutant
    pdf_th12 = compute_pdf_for_mechanism(mechanism, data_threshold12, n1_th, param_dict['N1'],
                                         n2_th, param_dict['N2'], param_dict['k'], mech_params, pair12=True)
    if np.any(pdf_th12 <= 0) or np.any(np.isnan(pdf_th12)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_th12))  # REMOVED: / len(data_threshold12)

    pdf_th32 = compute_pdf_for_mechanism(mechanism, data_threshold32, n3_th, param_dict['N3'],
                                         n2_th, param_dict['N2'], param_dict['k'], mech_params, pair12=False)
    if np.any(pdf_th32 <= 0) or np.any(np.isnan(pdf_th32)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_th32))  # REMOVED: / len(data_threshold32)

    # Degradation Rate Mutant
    k_deg = max(param_dict['beta_k'] * param_dict['k'], 0.0005)
    if param_dict['beta_k'] * param_dict['k'] < 0.00005:
        print("Warning: beta_k * k is less than 0.00005, setting k_deg to 0.00005")

    pdf_deg12 = compute_pdf_for_mechanism(mechanism, data_degrate12, param_dict['n1'], param_dict['N1'],
                                          param_dict['n2'], param_dict['N2'], k_deg, mech_params, pair12=True)
    if np.any(pdf_deg12 <= 0) or np.any(np.isnan(pdf_deg12)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_deg12))  # REMOVED: / len(data_degrate12)

    pdf_deg32 = compute_pdf_for_mechanism(mechanism, data_degrate32, param_dict['n3'], param_dict['N3'],
                                          param_dict['n2'], param_dict['N2'], k_deg, mech_params, pair12=False)
    if np.any(pdf_deg32 <= 0) or np.any(np.isnan(pdf_deg32)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_deg32))  # REMOVED: / len(data_degrate32)

    # Degradation Rate APC Mutant
    k_degAPC = max(param_dict['beta2_k'] * param_dict['k'], 0.0005)
    if param_dict['beta2_k'] * param_dict['k'] < 0.0005:
        print("Warning: beta2_k * k is less than 0.0005, setting k_degAPC to 0.0005")

    pdf_degAPC12 = compute_pdf_for_mechanism(mechanism, data_degrateAPC12, param_dict['n1'], param_dict['N1'],
                                             param_dict['n2'], param_dict['N2'], k_degAPC, mech_params, pair12=True)
    if np.any(pdf_degAPC12 <= 0) or np.any(np.isnan(pdf_degAPC12)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_degAPC12))  # REMOVED: / len(data_degrateAPC12)

    pdf_degAPC32 = compute_pdf_for_mechanism(mechanism, data_degrateAPC32, param_dict['n3'], param_dict['N3'],
                                             param_dict['n2'], param_dict['N2'], k_degAPC, mech_params, pair12=False)
    if np.any(pdf_degAPC32 <= 0) or np.any(np.isnan(pdf_degAPC32)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_degAPC32))  # REMOVED: / len(data_degrateAPC32)

    # Velcade Mutant
    k_velcade = max(param_dict['beta3_k'] * param_dict['k'], 0.0005)
    if param_dict['beta3_k'] * param_dict['k'] < 0.0005:
        print("Warning: beta3_k * k is less than 0.0005, setting k_velcade to 0.0005")

    pdf_velcade12 = compute_pdf_for_mechanism(mechanism, data_velcade12, param_dict['n1'], param_dict['N1'],
                                              param_dict['n2'], param_dict['N2'], k_velcade, mech_params, pair12=True)
    if np.any(pdf_velcade12 <= 0) or np.any(np.isnan(pdf_velcade12)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_velcade12))  # REMOVED: / len(data_velcade12)

    pdf_velcade32 = compute_pdf_for_mechanism(mechanism, data_velcade32, param_dict['n3'], param_dict['N3'],
                                              param_dict['n2'], param_dict['N2'], k_velcade, mech_params, pair12=False)
    if np.any(pdf_velcade32 <= 0) or np.any(np.isnan(pdf_velcade32)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_velcade32))  # REMOVED: / len(data_velcade32)

    # Initial Proteins Mutant - TEMPORARILY EXCLUDED FROM FITTING
    # PDF calculations and NLL contribution removed as initial strain is not being fitted

    return total_nll


def joint_objective_with_bootstrapping(params, mechanism, mechanism_info,
                                       data_wt12, data_wt32, data_threshold12, data_threshold32,
                                       data_degrate12, data_degrate32, data_initial12, data_initial32,
                                       data_degrateAPC12, data_degrateAPC32, data_velcade12, data_velcade32,
                                       bootstrap_method='bootstrap', target_sample_size=50,
                                       num_bootstrap_samples=100, random_seed=None):
    """
    Joint objective function with bootstrapping to handle unequal data points.

    Args:
        params: Parameters to optimize
        mechanism: Mechanism type
        mechanism_info: Mechanism information
        data_*: Experimental datasets
        bootstrap_method: 'bootstrap', 'weighted', or 'standard'
        target_sample_size: Target sample size for bootstrapping
        num_bootstrap_samples: Number of bootstrap samples
        random_seed: Random seed for reproducibility
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

    total_nll = 0.0

    # Helper function to calculate likelihood with bootstrapping
    def calculate_dataset_likelihood(data_exp, data_theory):
        """Calculate likelihood using specified bootstrap method."""
        if bootstrap_method == 'bootstrap':
            # Use theoretical PDF values as "simulated data" for bootstrapping
            return bootstrap_calc.calculate_bootstrap_likelihood(data_exp, data_theory)
        elif bootstrap_method == 'weighted':
            return bootstrap_calc.calculate_weighted_likelihood(data_exp, data_theory)
        else:  # standard method
            if np.any(data_theory <= 0) or np.any(np.isnan(data_theory)):
                return 1e6
            return -np.sum(np.log(data_theory)) / len(data_exp)

    # Wild-Type (Chrom1–Chrom2 and Chrom3–Chrom2)
    pdf_wt12 = compute_pdf_for_mechanism(mechanism, data_wt12, param_dict['n1'], param_dict['N1'],
                                         param_dict['n2'], param_dict['N2'], param_dict['k'], mech_params, pair12=True)

    if bootstrap_method == 'standard':
        if np.any(pdf_wt12 <= 0) or np.any(np.isnan(pdf_wt12)):
            return np.inf
        total_nll -= np.sum(np.log(pdf_wt12))  # REMOVED: / len(data_wt12)
    else:
        # For bootstrapping, we need to create "simulated" data from the theoretical PDF
        # Sample from the theoretical distribution represented by the PDF
        try:
            # Create a discrete distribution from the PDF
            pdf_normalized = pdf_wt12 / np.sum(pdf_wt12)
            simulated_wt12 = np.random.choice(data_wt12, size=len(
                data_wt12)*3, p=pdf_normalized, replace=True)
            nll_wt12 = calculate_dataset_likelihood(data_wt12, simulated_wt12)
            if nll_wt12 >= 1e6:
                return np.inf
            total_nll += nll_wt12
        except:
            return np.inf

    pdf_wt32 = compute_pdf_for_mechanism(mechanism, data_wt32, param_dict['n3'], param_dict['N3'],
                                         param_dict['n2'], param_dict['N2'], param_dict['k'], mech_params, pair12=False)

    if bootstrap_method == 'standard':
        if np.any(pdf_wt32 <= 0) or np.any(np.isnan(pdf_wt32)):
            return np.inf
        total_nll -= np.sum(np.log(pdf_wt32))  # REMOVED: / len(data_wt32)
    else:
        try:
            pdf_normalized = pdf_wt32 / np.sum(pdf_wt32)
            simulated_wt32 = np.random.choice(data_wt32, size=len(
                data_wt32)*3, p=pdf_normalized, replace=True)
            nll_wt32 = calculate_dataset_likelihood(data_wt32, simulated_wt32)
            if nll_wt32 >= 1e6:
                return np.inf
            total_nll += nll_wt32
        except:
            return np.inf

    # Threshold Mutant
    pdf_th12 = compute_pdf_for_mechanism(mechanism, data_threshold12, n1_th, param_dict['N1'],
                                         n2_th, param_dict['N2'], param_dict['k'], mech_params, pair12=True)

    if bootstrap_method == 'standard':
        if np.any(pdf_th12 <= 0) or np.any(np.isnan(pdf_th12)):
            return np.inf
        total_nll -= np.sum(np.log(pdf_th12))  # REMOVED: / len(data_threshold12)
    else:
        try:
            pdf_normalized = pdf_th12 / np.sum(pdf_th12)
            simulated_th12 = np.random.choice(data_threshold12, size=len(
                data_threshold12)*3, p=pdf_normalized, replace=True)
            nll_th12 = calculate_dataset_likelihood(
                data_threshold12, simulated_th12)
            if nll_th12 >= 1e6:
                return np.inf
            total_nll += nll_th12
        except:
            return np.inf

    pdf_th32 = compute_pdf_for_mechanism(mechanism, data_threshold32, n3_th, param_dict['N3'],
                                         n2_th, param_dict['N2'], param_dict['k'], mech_params, pair12=False)

    if bootstrap_method == 'standard':
        if np.any(pdf_th32 <= 0) or np.any(np.isnan(pdf_th32)):
            return np.inf
        total_nll -= np.sum(np.log(pdf_th32))  # REMOVED: / len(data_threshold32)
    else:
        try:
            pdf_normalized = pdf_th32 / np.sum(pdf_th32)
            simulated_th32 = np.random.choice(data_threshold32, size=len(
                data_threshold32)*3, p=pdf_normalized, replace=True)
            nll_th32 = calculate_dataset_likelihood(
                data_threshold32, simulated_th32)
            if nll_th32 >= 1e6:
                return np.inf
            total_nll += nll_th32
        except:
            return np.inf

    # Degradation Rate Mutant
    k_deg = max(param_dict['beta_k'] * param_dict['k'], 0.001)
    if param_dict['beta_k'] * param_dict['k'] < 0.001:
        print("Warning: beta_k * k is less than 0.001, setting k_deg to 0.001")

    pdf_deg12 = compute_pdf_for_mechanism(mechanism, data_degrate12, param_dict['n1'], param_dict['N1'],
                                          param_dict['n2'], param_dict['N2'], k_deg, mech_params, pair12=True)

    if bootstrap_method == 'standard':
        if np.any(pdf_deg12 <= 0) or np.any(np.isnan(pdf_deg12)):
            return np.inf
        total_nll -= np.sum(np.log(pdf_deg12))  # REMOVED: / len(data_degrate12)
    else:
        try:
            pdf_normalized = pdf_deg12 / np.sum(pdf_deg12)
            simulated_deg12 = np.random.choice(data_degrate12, size=len(
                data_degrate12)*3, p=pdf_normalized, replace=True)
            nll_deg12 = calculate_dataset_likelihood(
                data_degrate12, simulated_deg12)
            if nll_deg12 >= 1e6:
                return np.inf
            total_nll += nll_deg12
        except:
            return np.inf

    pdf_deg32 = compute_pdf_for_mechanism(mechanism, data_degrate32, param_dict['n3'], param_dict['N3'],
                                          param_dict['n2'], param_dict['N2'], k_deg, mech_params, pair12=False)

    if bootstrap_method == 'standard':
        if np.any(pdf_deg32 <= 0) or np.any(np.isnan(pdf_deg32)):
            return np.inf
        total_nll -= np.sum(np.log(pdf_deg32))  # REMOVED: / len(data_degrate32)
    else:
        try:
            pdf_normalized = pdf_deg32 / np.sum(pdf_deg32)
            simulated_deg32 = np.random.choice(data_degrate32, size=len(
                data_degrate32)*3, p=pdf_normalized, replace=True)
            nll_deg32 = calculate_dataset_likelihood(
                data_degrate32, simulated_deg32)
            if nll_deg32 >= 1e6:
                return np.inf
            total_nll += nll_deg32
        except:
            return np.inf

    # Degradation Rate APC Mutant
    k_degAPC = max(param_dict['beta2_k'] * param_dict['k'], 0.001)
    if param_dict['beta2_k'] * param_dict['k'] < 0.001:
        print("Warning: beta2_k * k is less than 0.001, setting k_degAPC to 0.001")

    pdf_degAPC12 = compute_pdf_for_mechanism(mechanism, data_degrateAPC12, param_dict['n1'], param_dict['N1'],
                                             param_dict['n2'], param_dict['N2'], k_degAPC, mech_params, pair12=True)

    if bootstrap_method == 'standard':
        if np.any(pdf_degAPC12 <= 0) or np.any(np.isnan(pdf_degAPC12)):
            return np.inf
        total_nll -= np.sum(np.log(pdf_degAPC12))  # REMOVED: / len(data_degrateAPC12)
    else:
        try:
            pdf_normalized = pdf_degAPC12 / np.sum(pdf_degAPC12)
            simulated_degAPC12 = np.random.choice(data_degrateAPC12, size=len(
                data_degrateAPC12)*3, p=pdf_normalized, replace=True)
            nll_degAPC12 = calculate_dataset_likelihood(
                data_degrateAPC12, simulated_degAPC12)
            if nll_degAPC12 >= 1e6:
                return np.inf
            total_nll += nll_degAPC12
        except:
            return np.inf

    pdf_degAPC32 = compute_pdf_for_mechanism(mechanism, data_degrateAPC32, param_dict['n3'], param_dict['N3'],
                                             param_dict['n2'], param_dict['N2'], k_degAPC, mech_params, pair12=False)

    if bootstrap_method == 'standard':
        if np.any(pdf_degAPC32 <= 0) or np.any(np.isnan(pdf_degAPC32)):
            return np.inf
        total_nll -= np.sum(np.log(pdf_degAPC32))  # REMOVED: / len(data_degrateAPC32)
    else:
        try:
            pdf_normalized = pdf_degAPC32 / np.sum(pdf_degAPC32)
            simulated_degAPC32 = np.random.choice(data_degrateAPC32, size=len(
                data_degrateAPC32)*3, p=pdf_normalized, replace=True)
            nll_degAPC32 = calculate_dataset_likelihood(
                data_degrateAPC32, simulated_degAPC32)
            if nll_degAPC32 >= 1e6:
                return np.inf
            total_nll += nll_degAPC32
        except:
            return np.inf

    # Velcade Mutant
    k_velcade = max(param_dict['beta3_k'] * param_dict['k'], 0.001)
    if param_dict['beta3_k'] * param_dict['k'] < 0.001:
        print("Warning: beta3_k * k is less than 0.001, setting k_velcade to 0.001")

    pdf_velcade12 = compute_pdf_for_mechanism(mechanism, data_velcade12, param_dict['n1'], param_dict['N1'],
                                              param_dict['n2'], param_dict['N2'], k_velcade, mech_params, pair12=True)

    if bootstrap_method == 'standard':
        if np.any(pdf_velcade12 <= 0) or np.any(np.isnan(pdf_velcade12)):
            return np.inf
        total_nll -= np.sum(np.log(pdf_velcade12))  # REMOVED: / len(data_velcade12)
    else:
        try:
            pdf_normalized = pdf_velcade12 / np.sum(pdf_velcade12)
            simulated_velcade12 = np.random.choice(data_velcade12, size=len(
                data_velcade12)*3, p=pdf_normalized, replace=True)
            nll_velcade12 = calculate_dataset_likelihood(
                data_velcade12, simulated_velcade12)
            if nll_velcade12 >= 1e6:
                return np.inf
            total_nll += nll_velcade12
        except:
            return np.inf

    pdf_velcade32 = compute_pdf_for_mechanism(mechanism, data_velcade32, param_dict['n3'], param_dict['N3'],
                                              param_dict['n2'], param_dict['N2'], k_velcade, mech_params, pair12=False)

    if bootstrap_method == 'standard':
        if np.any(pdf_velcade32 <= 0) or np.any(np.isnan(pdf_velcade32)):
            return np.inf
        total_nll -= np.sum(np.log(pdf_velcade32))  # REMOVED: / len(data_velcade32)
    else:
        try:
            pdf_normalized = pdf_velcade32 / np.sum(pdf_velcade32)
            simulated_velcade32 = np.random.choice(data_velcade32, size=len(
                data_velcade32)*3, p=pdf_normalized, replace=True)
            nll_velcade32 = calculate_dataset_likelihood(
                data_velcade32, simulated_velcade32)
            if nll_velcade32 >= 1e6:
                return np.inf
            total_nll += nll_velcade32
        except:
            return np.inf

    # Initial Proteins Mutant - TEMPORARILY EXCLUDED FROM FITTING
    # PDF calculations and NLL contribution removed as initial strain is not being fitted

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
    # Common parameters for all mechanisms - Updated to match simulation_utils.py bounds
    common_params = ['n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23']
    common_bounds = [
        (1.0, 50.0),      # n2 - Updated to match simulation bounds
        (50.0, 1000.0),   # N2 - Updated to match simulation bounds  
        (0.01, 0.1),      # k - Updated to match simulation k_max bounds
        (0.25, 4.0),      # r21 - Updated to match simulation bounds
        (0.25, 4.0),      # r23 - Updated to match simulation bounds
        (0.4, 2),       # R21 - Updated to match simulation bounds
        (0.5, 5.0),       # R23 - Kept same as both use (0.5, 5.0)
    ]

    # Mutant parameters - Updated to match simulation_utils.py bounds
    # Including alpha, beta_k, beta2_k, beta3_k (gamma parameters excluded)
    mutant_params = ['alpha', 'beta_k', 'beta2_k', 'beta3_k']
    mutant_bounds = [
        (0.1, 0.7),       # alpha - Updated to match simulation bounds
        (0.1, 1.0),       # beta_k - Updated to match simulation bounds
        (0.1, 1.0),        # beta2_k - Updated to match simulation beta_tau bounds
        (0.1, 1.0),        # beta3_k (Velcade) - Updated to match simulation beta_tau2 bounds
    ]

    if mechanism == 'simple':
        mechanism_params = []
        mechanism_bounds = []
    elif mechanism == 'fixed_burst':
        mechanism_params = ['burst_size']
        mechanism_bounds = [(1.0, 50.0)]  # Updated to match simulation bounds
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
            (1.0, 100.0),   # n_inner - Updated to match simulation bounds
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
            (1.0, 50.0),     # burst_size - Updated to match simulation bounds
            (0.0001, 0.02),  # w1
            (0.0001, 0.02),  # w2
            (0.0001, 0.02),  # w3
        ]
    elif mechanism == 'fixed_burst_feedback_onion':
        mechanism_params = ['burst_size', 'n_inner']
        mechanism_bounds = [
            (1.0, 50.0),     # burst_size - Updated to match simulation bounds
            (1.0, 100.0),    # n_inner - Updated to match simulation bounds
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


def run_mom_optimization_single(mechanism, data_arrays=None, max_iterations=200, seed=None, gamma_mode='separate'):
    """
    Run a single MoM optimization for a given mechanism.
    This function can be reused by other scripts for model comparison.
    
    Args:
        mechanism (str): Mechanism name
        data_arrays (dict): Dictionary containing data arrays, if None will load from file
        max_iterations (int): Maximum iterations for optimization
        seed (int): Random seed for reproducible results
        gamma_mode (str): 'unified' or 'separate' gamma mode
    
    Returns:
        dict: Results dictionary with success, nll, params, etc.
    """
    try:
        # Get mechanism-specific information
        mechanism_info = get_mechanism_info(mechanism, gamma_mode)
        bounds = mechanism_info['bounds']
        
        # Use provided data arrays or load from file
        if data_arrays is not None:
            # Use pre-loaded data arrays
            data_wt12 = data_arrays['data_wt12']
            data_wt32 = data_arrays['data_wt32']
            data_threshold12 = data_arrays['data_threshold12']
            data_threshold32 = data_arrays['data_threshold32']
            data_degrate12 = data_arrays['data_degrate12']
            data_degrate32 = data_arrays['data_degrate32']
            data_initial12 = data_arrays.get('data_initial12', np.array([]))
            data_initial32 = data_arrays.get('data_initial32', np.array([]))
            data_degrateAPC12 = data_arrays['data_degrateAPC12']
            data_degrateAPC32 = data_arrays['data_degrateAPC32']
            data_velcade12 = data_arrays['data_velcade12']
            data_velcade32 = data_arrays['data_velcade32']
        else:
            # Load data from file (for standalone use)
            df = pd.read_excel("Data/All_strains_SCStimes.xlsx")
            data_wt12 = df['wildtype12'].dropna().values
            data_wt32 = df['wildtype32'].dropna().values
            data_threshold12 = df['threshold12'].dropna().values
            data_threshold32 = df['threshold32'].dropna().values
            data_degrate12 = df['degRade12'].dropna().values
            data_degrate32 = df['degRade32'].dropna().values
            # Handle missing initialProteins columns
            data_initial12 = df['initialProteins12'].dropna().values if 'initialProteins12' in df.columns else np.array([])
            data_initial32 = df['initialProteins32'].dropna().values if 'initialProteins32' in df.columns else np.array([])
            data_degrateAPC12 = df['degRadeAPC12'].dropna().values
            data_degrateAPC32 = df['degRadeAPC32'].dropna().values
            data_velcade12 = df['degRadeVel12'].dropna().values
            data_velcade32 = df['degRadeVel32'].dropna().values
        
        # Use provided seed or default to 42
        optimization_seed = seed if seed is not None else 42
        
        # Run optimization using differential evolution
        result = differential_evolution(
            joint_objective,
            bounds=bounds,
            args=(mechanism, mechanism_info, data_wt12, data_wt32,
                  data_threshold12, data_threshold32,
                  data_degrate12, data_degrate32,
                  data_initial12, data_initial32,
                  data_degrateAPC12, data_degrateAPC32,
                  data_velcade12, data_velcade32),
            maxiter=max_iterations,
            popsize=15,
            seed=optimization_seed,
            disp=False
        )
        
        # Convert to standard format
        return {
            'success': True,
            'converged': result.success,
            'nll': result.fun,
            'params': dict(zip(mechanism_info['params'], result.x)),
            'result': result,
            'message': result.message if not result.success else "Converged successfully",
            'mechanism_info': mechanism_info
        }
        
    except Exception as e:
        return {
            'success': False,
            'converged': False,
            'nll': np.inf,
            'params': {},
            'result': None,
            'message': f"MoM optimization failed: {e}",
            'mechanism_info': None
        }


def main():
    # ========== MECHANISM CONFIGURATION ==========
    # Choose mechanism: 'simple', 'fixed_burst', 'feedback_onion', 'fixed_burst_feedback_onion'
    mechanism = 'fixed_burst_feedback_onion'  # Auto-set by RunAllMechanisms.py

    # ========== GAMMA CONFIGURATION ==========
    # Choose gamma mode: 'unified' for single gamma affecting all chromosomes, 'separate' for gamma1, gamma2, gamma3
    gamma_mode = 'unified'  # Change this to 'separate' for individual gamma per chromosome

    print(f"Optimizing for mechanism: {mechanism}")
    print(f"Gamma mode: {gamma_mode}")
    print("NOTE: Initial proteins strain is TEMPORARILY EXCLUDED from fitting")
    print("Fitting datasets: wildtype, threshold, degrate, degrateAPC")

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
    data_degrate12 = df['degRade12'].dropna().values
    data_degrate32 = df['degRade32'].dropna().values
    # Handle missing initialProteins columns
    data_initial12 = df['initialProteins12'].dropna().values if 'initialProteins12' in df.columns else np.array([])
    data_initial32 = df['initialProteins32'].dropna().values if 'initialProteins32' in df.columns else np.array([])
    data_degrateAPC12 = df['degRadeAPC12'].dropna().values
    data_degrateAPC32 = df['degRadeAPC32'].dropna().values
    data_velcade12 = df['degRadeVel12'].dropna().values
    data_velcade32 = df['degRadeVel32'].dropna().values

    # c) Global optimization to find top 5 solutions
    population_solutions = []

    def callback(xk, convergence):
        population_solutions.append((joint_objective(xk, mechanism, mechanism_info, data_wt12, data_wt32,
                                                     data_threshold12, data_threshold32,
                                                     data_degrate12, data_degrate32,
                                                     data_initial12, data_initial32,
                                                     data_degrateAPC12, data_degrateAPC32,
                                                     data_velcade12, data_velcade32), xk.copy()))

    result = differential_evolution(
        joint_objective,
        bounds=bounds,
        args=(mechanism, mechanism_info, data_wt12, data_wt32, data_threshold12, data_threshold32,
              data_degrate12, data_degrate32, data_initial12, data_initial32,
              data_degrateAPC12, data_degrateAPC32, data_velcade12, data_velcade32),
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

        # Print mutant parameters (initial strain excluded)
        print(
            f"Mutants: alpha = {param_dict['alpha']:.2f}, beta_k = {param_dict['beta_k']:.2f}, beta2_k = {param_dict['beta2_k']:.2f} (initial strain excluded)")
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
                  data_degrateAPC12, data_degrateAPC32, data_velcade12, data_velcade32),
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

    # Velcade Mutant
    velcade_nll = 0
    k_velcade = max(param_dict['beta3_k'] * param_dict['k'], 0.001)
    pdf_velcade12 = compute_pdf_for_mechanism(mechanism, data_velcade12, param_dict['n1'], param_dict['N1'],
                                              param_dict['n2'], param_dict['N2'], k_velcade, mech_params, pair12=True)
    if not (np.any(pdf_velcade12 <= 0) or np.any(np.isnan(pdf_velcade12))):
        velcade_nll -= np.sum(np.log(pdf_velcade12))
    pdf_velcade32 = compute_pdf_for_mechanism(mechanism, data_velcade32, param_dict['n3'], param_dict['N3'],
                                              param_dict['n2'], param_dict['N2'], k_velcade, mech_params, pair12=False)
    if not (np.any(pdf_velcade32 <= 0) or np.any(np.isnan(pdf_velcade32))):
        velcade_nll -= np.sum(np.log(pdf_velcade32))

    # Initial proteins mutant - TEMPORARILY EXCLUDED FROM FITTING
    initial_nll = 0.0  # Set to 0 to exclude from total NLL

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
    print(f"Velcade Mutant Negative Log-Likelihood: {velcade_nll:.4f}")
    print("Initial Proteins Mutant: EXCLUDED FROM FITTING (temporarily)")

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
    print(
        f"Degradation Rate APC Mutant: beta2_k = {param_dict['beta2_k']:.2f}")
    print(f"Velcade Mutant: beta3_k = {param_dict['beta3_k']:.2f}")
    print("Initial Proteins Mutant: EXCLUDED FROM FITTING (temporarily)")

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
        f.write(f"beta3_k: {param_dict['beta3_k']:.6f}\n")
        f.write("# Initial proteins mutant parameters - EXCLUDED FROM FITTING\n")
        f.write("# gamma: not_fitted\n")
        f.write("# gamma1: not_fitted\n")
        f.write("# gamma2: not_fitted\n")
        f.write("# gamma3: not_fitted\n")
        f.write(f"threshold_nll: {threshold_nll:.6f}\n")
        f.write(f"degrate_nll: {degrate_nll:.6f}\n")
        f.write(f"degrateAPC_nll: {degrateAPC_nll:.6f}\n")
        f.write(f"velcade_nll: {velcade_nll:.6f}\n")
        f.write(f"initial_nll: {initial_nll:.6f}  # EXCLUDED (set to 0.0)\n")
        f.write(f"total_nll: {best_nll:.6f}  # Includes all 5 datasets\n")

    print(f"Optimized parameters saved to {filename}")


def main_with_bootstrapping():
    """
    Main optimization routine with bootstrapping comparison.
    """
    # ========== MECHANISM CONFIGURATION ==========
    mechanism = 'fixed_burst_feedback_onion'  # Auto-set by RunAllMechanisms.py
    gamma_mode = 'separate'  # Change this to 'separate' for individual gamma per chromosome

    print(f"Bootstrapping optimization for mechanism: {mechanism}")
    print(f"Gamma mode: {gamma_mode}")
    print("NOTE: Initial proteins strain is TEMPORARILY EXCLUDED from fitting")
    print("Fitting datasets: wildtype, threshold, degrate, degrateAPC")

    # Get mechanism-specific information
    mechanism_info = get_mechanism_info(mechanism, gamma_mode)
    bounds = mechanism_info['bounds']

    print(f"Parameters to optimize: {mechanism_info['params']}")
    print(f"Number of parameters: {len(mechanism_info['params'])}")

    # Read data
    df = pd.read_excel("Data/All_strains_SCStimes.xlsx")
    data_wt12 = df['wildtype12'].dropna().values
    data_wt32 = df['wildtype32'].dropna().values
    data_threshold12 = df['threshold12'].dropna().values
    data_threshold32 = df['threshold32'].dropna().values
    data_degrate12 = df['degRade12'].dropna().values
    data_degrate32 = df['degRade32'].dropna().values
    data_initial12 = df['initialProteins12'].dropna().values
    data_initial32 = df['initialProteins32'].dropna().values
    data_degrateAPC12 = df['degRadeAPC12'].dropna().values
    data_degrateAPC32 = df['degRadeAPC32'].dropna().values
    data_velcade12 = df['degRadeVel12'].dropna().values
    data_velcade32 = df['degRadeVel32'].dropna().values

    # Analyze dataset sizes
    datasets = {
        'wildtype': {'delta_t12': data_wt12, 'delta_t32': data_wt32},
        'threshold': {'delta_t12': data_threshold12, 'delta_t32': data_threshold32},
        'degrate': {'delta_t12': data_degrate12, 'delta_t32': data_degrate32},
        'degrateAPC': {'delta_t12': data_degrateAPC12, 'delta_t32': data_degrateAPC32},
        'velcade': {'delta_t12': data_velcade12, 'delta_t32': data_velcade32}
    }

    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    size_analysis = analyze_dataset_sizes(datasets)
    target_sample_size = size_analysis['recommended_target_size']

    print(f"\n{'='*60}")
    print("COMPARISON: Standard vs. Bootstrapping Methods")
    print(f"{'='*60}")

    methods = [
        ('standard', 'Standard MoM Optimization'),
        ('weighted', 'Weighted Likelihood Optimization'),
        ('bootstrap', 'Bootstrap Optimization')
    ]

    results = {}

    for method, method_name in methods:
        print(f"\n{'-'*40}")
        print(f"{method_name.upper()}")
        print(f"{'-'*40}")

        try:
            # Choose objective function
            if method == 'standard':
                objective_func = joint_objective
                args = (mechanism, mechanism_info, data_wt12, data_wt32,
                        data_threshold12, data_threshold32, data_degrate12, data_degrate32,
                        data_initial12, data_initial32, data_degrateAPC12, data_degrateAPC32,
                        data_velcade12, data_velcade32)
            else:
                objective_func = joint_objective_with_bootstrapping
                args = (mechanism, mechanism_info, data_wt12, data_wt32,
                        data_threshold12, data_threshold32, data_degrate12, data_degrate32,
                        data_initial12, data_initial32, data_degrateAPC12, data_degrateAPC32,
                        data_velcade12, data_velcade32,
                        method, target_sample_size, 50, 42)  # Reduced bootstrap samples for testing

            # Run optimization
            result = differential_evolution(
                objective_func,
                bounds=bounds,
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
                param_dict = unpack_parameters(result.x, mechanism_info)
                results[method] = {
                    'nll': result.fun,
                    'params': param_dict,
                    'success': True
                }

                print(f"✓ {method_name} completed successfully")
                print(f"  Negative Log-Likelihood: {result.fun:.4f}")
                print(
                    f"  Parameters: n2={param_dict['n2']:.1f}, N2={param_dict['N2']:.1f}, k={param_dict['k']:.4f}")

                # Save results
                suffix = f"_{method}" if method != 'standard' else "_standard"
                filename = f"optimized_parameters_{mechanism}_join{suffix}.txt"

                with open(filename, "w") as f:
                    f.write(f"# Mechanism: {mechanism}\n")
                    f.write(f"# Method: {method_name}\n")
                    if method != 'standard':
                        f.write(
                            f"# Target Sample Size: {target_sample_size}\n")
                    f.write(f"# Negative Log-Likelihood: {result.fun:.6f}\n\n")

                    f.write("# Wild-Type Parameters\n")
                    for key in ['n1', 'n2', 'n3', 'N1', 'N2', 'N3', 'k']:
                        f.write(f"{key}: {param_dict[key]:.6f}\n")

                    # Mechanism-specific parameters
                    if mechanism == 'fixed_burst_feedback_onion':
                        f.write(
                            f"burst_size: {param_dict['burst_size']:.6f}\n")
                        f.write(f"n_inner: {param_dict['n_inner']:.6f}\n")

                    # Mutant parameters
                    f.write("\n# Mutant Parameters\n")
                    for key in ['alpha', 'beta_k', 'beta2_k', 'beta3_k']:
                        f.write(f"{key}: {param_dict[key]:.6f}\n")

                print(f"  Results saved to: {filename}")

            else:
                results[method] = {'success': False, 'nll': np.inf}
                print(f"✗ {method_name} failed: {result.message}")

        except Exception as e:
            results[method] = {'success': False, 'nll': np.inf}
            print(f"✗ Error in {method_name}: {e}")

    # Compare results
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON")
    print(f"{'='*60}")

    successful_results = {k: v for k, v in results.items() if v['success']}

    if successful_results:
        for method, method_name in methods:
            if method in successful_results:
                nll = successful_results[method]['nll']
                print(f"{method_name:30}: {nll:.4f}")

        # Calculate improvements
        if 'standard' in successful_results:
            standard_nll = successful_results['standard']['nll']
            print(f"\nImprovement over standard:")

            for method in ['weighted', 'bootstrap']:
                if method in successful_results:
                    method_nll = successful_results[method]['nll']
                    if method_nll < standard_nll:
                        improvement = (
                            (standard_nll - method_nll) / standard_nll) * 100
                        print(
                            f"  {method.capitalize():10}: {improvement:.1f}% better")
                    else:
                        change = ((method_nll - standard_nll) /
                                  standard_nll) * 100
                        print(
                            f"  {method.capitalize():10}: {change:.1f}% worse")
    else:
        print("No successful optimizations completed.")

    print(f"\n{'-' * 60}")
    print("Bootstrapping optimization comparison complete!")


if __name__ == "__main__":
    # Uncomment the line below to run bootstrapping comparison
    # main_with_bootstrapping()

    # Standard optimization
    main()
