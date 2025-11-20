"""
Joint optimization for chromosome segregation mechanisms.

TEMPORARY MODIFICATION: Initial proteins strain is excluded from fitting.
Currently fits only: wildtype, threshold, degrate, degrateAPC datasets.

To re-enable initial strain fitting, search for "TEMPORARILY EXCLUDED" 
and restore the original initial proteins objective code.
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from MoMCalculations import compute_pdf_for_mechanism


def validate_constraints(param_dict):
    """
    Validate parameter constraints: n_i < N_i for all chromosomes and threshold mutant.
    
    Args:
        param_dict (dict): Unpacked parameter dictionary
    
    Returns:
        bool: True if all constraints satisfied, False otherwise
    """
    # Wild-type constraints: n_i < N_i
    if (param_dict['n1'] >= param_dict['N1'] or 
        param_dict['n2'] >= param_dict['N2'] or 
        param_dict['n3'] >= param_dict['N3']):
        return False
    
    # Threshold mutant constraints: n_i * alpha < N_i
    alpha = param_dict['alpha']
    if (max(param_dict['n1'] * alpha, 1) >= param_dict['N1'] or
        max(param_dict['n2'] * alpha, 1) >= param_dict['N2'] or
        max(param_dict['n3'] * alpha, 1) >= param_dict['N3']):
        return False
    
    return True


def compute_strain_nll(mechanism, data, n_i, N_i, n_j, N_j, k, mech_params, pair12):
    """
    Compute negative log-likelihood for a single strain pair.
    
    Args:
        mechanism (str): Mechanism type
        data (ndarray): Experimental data (time differences between chromosomes)
        n_i, N_i (float): Threshold and initial count for chromosome i
        n_j, N_j (float): Threshold and initial count for chromosome j
        k (float): Degradation rate
        mech_params (dict): Mechanism-specific parameters
        pair12 (bool): Unused, kept for backward compatibility
    
    Returns:
        float: Negative log-likelihood, or np.inf if invalid
    """
    pdf = compute_pdf_for_mechanism(mechanism, data, n_i, N_i, n_j, N_j, k, mech_params, pair12=pair12)
    
    if np.any(pdf <= 0) or np.any(np.isnan(pdf)):
        return np.inf
    
    return -np.sum(np.log(pdf))


def calculate_individual_nlls(mechanism, param_dict, mech_params, data_arrays):
    """
    Calculate individual NLL values for each strain for reporting and visualization.
    """
    nlls = {}
    
    # Wild-Type
    _, nlls['wt'] = _add_strain_nll(
        0, mechanism, data_arrays['data_wt12'], data_arrays['data_wt32'],
        param_dict['n1'], param_dict['N1'], param_dict['n2'], param_dict['N2'],
        param_dict['n3'], param_dict['N3'], param_dict['k'], mech_params
    )
    
    # Threshold Mutant
    alpha = param_dict['alpha']
    n1_th = max(param_dict['n1'] * alpha, 1)
    n2_th = max(param_dict['n2'] * alpha, 1)
    n3_th = max(param_dict['n3'] * alpha, 1)
    _, nlls['threshold'] = _add_strain_nll(
        0, mechanism, data_arrays['data_threshold12'], data_arrays['data_threshold32'],
        n1_th, param_dict['N1'], n2_th, param_dict['N2'],
        n3_th, param_dict['N3'], param_dict['k'], mech_params
    )
    
    # Degradation Rate Mutants
    mutant_configs = [
        ('beta_k', 'degrate', 'data_degrate12', 'data_degrate32'),
        ('beta2_k', 'degrateAPC', 'data_degrateAPC12', 'data_degrateAPC32'),
        ('beta3_k', 'velcade', 'data_velcade12', 'data_velcade32')
    ]
    for beta_key, strain_name, key_12, key_32 in mutant_configs:
        k_mutant = max(param_dict[beta_key] * param_dict['k'], 0.001)
        _, nlls[strain_name] = _add_strain_nll(
            0, mechanism, data_arrays[key_12], data_arrays[key_32],
            param_dict['n1'], param_dict['N1'], param_dict['n2'], param_dict['N2'],
            param_dict['n3'], param_dict['N3'], k_mutant, mech_params
        )
    
    return nlls


def unpack_parameters(params, mechanism_info):
    """
    Unpack optimization parameters based on mechanism.

    Args:
        params (array): Parameter array from optimizer
        mechanism_info (dict): Mechanism information

    Returns:
        dict: Unpacked parameters
    """
    param_dict = dict(zip(mechanism_info['params'], params))
    
    # Derived wild-type parameters
    param_dict.update({
        'n1': max(param_dict['r21'] * param_dict['n2'], 1),
        'n3': max(param_dict['r23'] * param_dict['n2'], 1),
        'N1': max(param_dict['R21'] * param_dict['N2'], 1),
        'N3': max(param_dict['R23'] * param_dict['N2'], 1)
    })
    
    return param_dict


def _get_mech_params(mechanism, param_dict):
    """Extract mechanism-specific parameters."""
    mech_params = {}
    if mechanism == 'fixed_burst':
        mech_params['burst_size'] = param_dict['burst_size']
    elif mechanism == 'feedback_onion':
        mech_params['n_inner'] = param_dict['n_inner']
    elif mechanism == 'fixed_burst_feedback_onion':
        mech_params['burst_size'] = param_dict['burst_size']
        mech_params['n_inner'] = param_dict['n_inner']
    return mech_params


def _add_strain_nll(total_nll, mechanism, data_12, data_32, n1, N1, n2, N2, n3, N3, k, mech_params):
    """Add NLL for a strain pair (12 and 32)."""
    nll_12 = compute_strain_nll(mechanism, data_12, n1, N1, n2, N2, k, mech_params, pair12=True)
    if nll_12 == np.inf:
        return np.inf, np.inf
    nll_32 = compute_strain_nll(mechanism, data_32, n3, N3, n2, N2, k, mech_params, pair12=False)
    if nll_32 == np.inf:
        return np.inf, np.inf
    return total_nll + nll_12 + nll_32, nll_12 + nll_32


def joint_objective(params, mechanism, mechanism_info, data_wt12, data_wt32, data_threshold12, data_threshold32,
                    data_degrate12, data_degrate32, data_initial12, data_initial32, data_degrateAPC12, data_degrateAPC32,
                    data_velcade12, data_velcade32):
    """
    Joint objective function for all mechanisms.
    """
    param_dict = unpack_parameters(params, mechanism_info)
    
    if not validate_constraints(param_dict):
        return np.inf
    
    mech_params = _get_mech_params(mechanism, param_dict)
    total_nll = 0.0
    
    # Wild-Type
    total_nll, _ = _add_strain_nll(
        total_nll, mechanism, data_wt12, data_wt32,
        param_dict['n1'], param_dict['N1'], param_dict['n2'], param_dict['N2'],
        param_dict['n3'], param_dict['N3'], param_dict['k'], mech_params
    )
    if total_nll == np.inf:
        return np.inf
    
    # Threshold Mutant
    alpha = param_dict['alpha']
    n1_th = max(param_dict['n1'] * alpha, 1)
    n2_th = max(param_dict['n2'] * alpha, 1)
    n3_th = max(param_dict['n3'] * alpha, 1)
    total_nll, _ = _add_strain_nll(
        total_nll, mechanism, data_threshold12, data_threshold32,
        n1_th, param_dict['N1'], n2_th, param_dict['N2'],
        n3_th, param_dict['N3'], param_dict['k'], mech_params
    )
    if total_nll == np.inf:
        return np.inf
    
    # Degradation Rate Mutants
    for beta_key, data_12, data_32 in [
        ('beta_k', data_degrate12, data_degrate32),
        ('beta2_k', data_degrateAPC12, data_degrateAPC32),
        ('beta3_k', data_velcade12, data_velcade32)
    ]:
        k_mutant = max(param_dict[beta_key] * param_dict['k'], 0.0005)
        total_nll, _ = _add_strain_nll(
            total_nll, mechanism, data_12, data_32,
            param_dict['n1'], param_dict['N1'], param_dict['n2'], param_dict['N2'],
            param_dict['n3'], param_dict['N3'], k_mutant, mech_params
        )
        if total_nll == np.inf:
            return np.inf
    
    return total_nll


def get_rounded_parameters(params, mechanism_info):
    """Get rounded parameters for display/comparison."""
    param_dict = unpack_parameters(params, mechanism_info)
    rounded = {}
    
    for key in ['n1', 'n2', 'n3', 'N1', 'N2', 'N3']:
        rounded[key] = round(param_dict[key], 1)
    rounded['k'] = round(param_dict['k'], 3)
    
    for key in param_dict:
        if key not in rounded:
            rounded[key] = round(param_dict[key], 2)
    
    return rounded


def get_mechanism_info(mechanism, gamma_mode):
    """
    Get mechanism-specific parameter information.

    Args:
        mechanism (str): 'simple', 'fixed_burst', 'feedback_onion', or 'fixed_burst_feedback_onion'
        gamma_mode (str): 'unified' for single gamma affecting all chromosomes, 'separate' for gamma1, gamma2, gamma3

    Returns:
        dict: Contains parameter names, bounds, and default indices
    
    Bounds Notes:
    - All bounds are synchronized with simulation_utils.py for consistency
    - Biological constraints (n < N) are enforced in validate_constraints()
    - Numerical bounds prevent optimization instabilities and unrealistic parameter values
    """
    common_params = ['n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23']
    common_bounds = [
        (1.0, 50.0), (50.0, 1000.0), (0.01, 0.1),
        (0.25, 4.0), (0.25, 4.0), (0.4, 2.0), (0.5, 5.0)
    ]
    common_integrality = [True, True, False, False, False, False, False]

    mutant_params = ['alpha', 'beta_k', 'beta2_k', 'beta3_k']
    mutant_bounds = [(0.1, 0.7), (0.1, 1.0), (0.1, 1.0), (0.1, 1.0)]
    mutant_integrality = [False, False, False, False]

    mechanism_configs = {
        'simple': ([], [], []),
        'fixed_burst': (['burst_size'], [(1.0, 20.0)], [True]),
        'feedback_onion': (['n_inner'], [(1.0, 4000.0)], [True]),
        'fixed_burst_feedback_onion': (['burst_size', 'n_inner'], [(1.0, 20.0), (1.0, 4000.0)], [True, True])
    }
    
    if mechanism not in mechanism_configs:
        raise ValueError(f"Unknown mechanism: {mechanism}")
    
    mechanism_params, mechanism_bounds, mechanism_integrality = mechanism_configs[mechanism]

    all_params = common_params + mechanism_params + mutant_params
    all_bounds = common_bounds + mechanism_bounds + mutant_bounds
    all_integrality = common_integrality + mechanism_integrality + mutant_integrality

    return {
        'params': all_params,
        'bounds': all_bounds,
        'integrality': all_integrality,
        'common_count': len(common_params),
        'mechanism_count': len(mechanism_params),
        'mutant_count': len(mutant_params)
    }


def run_mom_optimization_single(mechanism, data_arrays=None, max_iterations=500, seed=None, gamma_mode='separate'):
    """
    Run a single MoM (Method of Moments) optimization for a given mechanism.
    
    This is a reusable optimization function that can be called from main() or from other scripts
    (e.g., model_comparison_aic_bic.py) for model selection and comparison workflows.
    
    Args:
        mechanism (str): Mechanism type. Valid options:
            - 'simple': Baseline harmonic degradation
            - 'fixed_burst': Degradation in fixed-size bursts
            - 'feedback_onion': Rate modified by onion-like feedback
            - 'fixed_burst_feedback_onion': Combined burst + onion feedback
        
        data_arrays (dict, optional): Pre-loaded data arrays. If None, will load from file.
            Expected keys: 'data_wt12', 'data_wt32', 'data_threshold12', 'data_threshold32',
            'data_degrate12', 'data_degrate32', 'data_initial12', 'data_initial32',
            'data_degrateAPC12', 'data_degrateAPC32', 'data_velcade12', 'data_velcade32'
        
        max_iterations (int): Maximum iterations for differential evolution (default: 500).
            Higher values give more thorough search but take longer.
        
        seed (int, optional): Random seed for reproducible results. Defaults to 42 if not provided.
        
        gamma_mode (str): Parameter mode (default: 'separate').
            - 'unified': Single gamma affects all chromosomes
            - 'separate': Individual gamma for each chromosome (currently excluded from fitting)
    
    Returns:
        dict: Results dictionary containing:
            - 'success' (bool): Whether optimization completed without exceptions
            - 'converged' (bool): Whether scipy reports convergence
            - 'nll' (float): Best negative log-likelihood found
            - 'params' (dict): Dictionary of optimized parameter names and values
            - 'result' (scipy result): Full scipy.optimize.differential_evolution result object
            - 'message' (str): Status message (success or error description)
            - 'mechanism_info' (dict): Mechanism metadata including parameter names and bounds
    
    Examples:
        # Direct call with data arrays (e.g., from model_comparison_aic_bic.py)
        result = run_mom_optimization_single(
            mechanism='fixed_burst',
            data_arrays=data_dict,
            max_iterations=500,
            seed=42
        )
        
        # Call without data arrays (loads from file)
        result = run_mom_optimization_single(
            mechanism='simple',
            max_iterations=400
        )
        
        # Access results
        if result['success']:
            print(f"NLL: {result['nll']:.4f}")
            print(f"Parameters: {result['params']}")
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
            popsize=20,
            strategy='best1bin',
            mutation=(0.5, 1.0),
            recombination=0.7,
            tol=1e-8,
            seed=optimization_seed,
            polish=True,
            workers=-1,
            integrality=mechanism_info['integrality']
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
    """
    Main optimization routine - uses run_mom_optimization_single() for the actual optimization.
    This function adds extra features like finding top 5 solutions and local refinement.
    """
    # ========== MECHANISM CONFIGURATION ==========
    # Choose mechanism: 'simple', 'fixed_burst', 'feedback_onion', 'fixed_burst_feedback_onion'
    mechanism = 'simple'  # Auto-set by RunAllMechanisms.py

    # ========== GAMMA CONFIGURATION ==========
    # Choose gamma mode: 'unified' for single gamma affecting all chromosomes, 'separate' for gamma1, gamma2, gamma3
    gamma_mode = 'unified'  # Change this to 'separate' for individual gamma per chromosome


    # Load data once
    df = pd.read_excel("Data/All_strains_SCStimes.xlsx")
    data_arrays = {
        'data_wt12': df['wildtype12'].dropna().values,
        'data_wt32': df['wildtype32'].dropna().values,
        'data_threshold12': df['threshold12'].dropna().values,
        'data_threshold32': df['threshold32'].dropna().values,
        'data_degrate12': df['degRade12'].dropna().values,
        'data_degrate32': df['degRade32'].dropna().values,
        'data_initial12': df['initialProteins12'].dropna().values if 'initialProteins12' in df.columns else np.array([]),
        'data_initial32': df['initialProteins32'].dropna().values if 'initialProteins32' in df.columns else np.array([]),
        'data_degrateAPC12': df['degRadeAPC12'].dropna().values,
        'data_degrateAPC32': df['degRadeAPC32'].dropna().values,
        'data_velcade12': df['degRadeVel12'].dropna().values,
        'data_velcade32': df['degRadeVel32'].dropna().values
    }
    
    result = run_mom_optimization_single(
        mechanism=mechanism,
        data_arrays=data_arrays,
        max_iterations=400,  # More iterations for thorough search
        seed=42,
        gamma_mode=gamma_mode
    )
    
    if not result['success']:
        print(f"Optimization failed: {result['message']}")
        return
    
    param_dict = result['params']
    best_nll = result['nll']
    mech_params = _get_mech_params(mechanism, param_dict)
    strain_nlls = calculate_individual_nlls(mechanism, param_dict, mech_params, data_arrays)
    
    print(f"Optimization completed: NLL = {best_nll:.4f}")

    filename = f"optimized_parameters_{mechanism}_join.txt"
    with open(filename, "w") as f:
        f.write(f"# Mechanism: {mechanism}\n")
        f.write("# Wild-Type Parameters\n")
        for key in ['n1', 'n2', 'n3', 'N1', 'N2', 'N3', 'k']:
            f.write(f"{key}: {param_dict[key]:.6f}\n")
        
        if 'burst_size' in param_dict:
            f.write(f"burst_size: {param_dict['burst_size']:.6f}\n")
        if 'n_inner' in param_dict:
            f.write(f"n_inner: {param_dict['n_inner']:.6f}\n")
        
        f.write(f"wt_nll: {strain_nlls['wt']:.6f}\n")
        f.write("# Mutant Parameters\n")
        for key in ['alpha', 'beta_k', 'beta2_k', 'beta3_k']:
            f.write(f"{key}: {param_dict[key]:.6f}\n")
        
        f.write("# Initial proteins mutant parameters - EXCLUDED FROM FITTING\n")
        for key in ['threshold', 'degrate', 'degrateAPC', 'velcade']:
            f.write(f"{key}_nll: {strain_nlls[key]:.6f}\n")
        f.write(f"total_nll: {best_nll:.6f}\n")


if __name__ == "__main__":
    main()
