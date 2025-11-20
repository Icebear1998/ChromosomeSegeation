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


def validate_constraints(param_dict):
    """
    Validate parameter constraints for a parameter dictionary.
    
    Ensures:
    - n_i < N_i for all chromosomes (baseline requirement)
    - n_i * alpha < N_i for threshold mutant
    
    Args:
        param_dict (dict): Unpacked parameter dictionary
    
    Returns:
        bool: True if all constraints satisfied, False otherwise
    """
    # Wild-type constraints: n_i < N_i
    if param_dict['n1'] >= param_dict['N1']:
        return False
    if param_dict['n2'] >= param_dict['N2']:
        return False
    if param_dict['n3'] >= param_dict['N3']:
        return False
    
    # Threshold mutant constraints: n_i * alpha < N_i
    n1_th = max(param_dict['n1'] * param_dict['alpha'], 1)
    n2_th = max(param_dict['n2'] * param_dict['alpha'], 1)
    n3_th = max(param_dict['n3'] * param_dict['alpha'], 1)
    
    if n1_th >= param_dict['N1']:
        return False
    if n2_th >= param_dict['N2']:
        return False
    if n3_th >= param_dict['N3']:
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
    
    Args:
        mechanism (str): Mechanism type
        param_dict (dict): Unpacked parameters with all derived values
        mech_params (dict): Mechanism-specific parameters
        data_arrays (dict): Dictionary with all data arrays
    
    Returns:
        dict: Dictionary with strain names as keys and NLL values
    """
    nlls = {}
    
    # Wild-Type
    nlls['wt'] = (
        compute_strain_nll(mechanism, data_arrays['data_wt12'],
                          param_dict['n1'], param_dict['N1'], param_dict['n2'], param_dict['N2'],
                          param_dict['k'], mech_params, pair12=True) +
        compute_strain_nll(mechanism, data_arrays['data_wt32'],
                          param_dict['n3'], param_dict['N3'], param_dict['n2'], param_dict['N2'],
                          param_dict['k'], mech_params, pair12=False)
    )
    
    # Threshold Mutant
    n1_th = max(param_dict['n1'] * param_dict['alpha'], 1)
    n2_th = max(param_dict['n2'] * param_dict['alpha'], 1)
    n3_th = max(param_dict['n3'] * param_dict['alpha'], 1)
    nlls['threshold'] = (
        compute_strain_nll(mechanism, data_arrays['data_threshold12'],
                          n1_th, param_dict['N1'], n2_th, param_dict['N2'],
                          param_dict['k'], mech_params, pair12=True) +
        compute_strain_nll(mechanism, data_arrays['data_threshold32'],
                          n3_th, param_dict['N3'], n2_th, param_dict['N2'],
                          param_dict['k'], mech_params, pair12=False)
    )
    
    # Degradation Rate Mutant
    k_deg = max(param_dict['beta_k'] * param_dict['k'], 0.001)
    nlls['degrate'] = (
        compute_strain_nll(mechanism, data_arrays['data_degrate12'],
                          param_dict['n1'], param_dict['N1'], param_dict['n2'], param_dict['N2'],
                          k_deg, mech_params, pair12=True) +
        compute_strain_nll(mechanism, data_arrays['data_degrate32'],
                          param_dict['n3'], param_dict['N3'], param_dict['n2'], param_dict['N2'],
                          k_deg, mech_params, pair12=False)
    )
    
    # Degradation Rate APC Mutant
    k_degAPC = max(param_dict['beta2_k'] * param_dict['k'], 0.001)
    nlls['degrateAPC'] = (
        compute_strain_nll(mechanism, data_arrays['data_degrateAPC12'],
                          param_dict['n1'], param_dict['N1'], param_dict['n2'], param_dict['N2'],
                          k_degAPC, mech_params, pair12=True) +
        compute_strain_nll(mechanism, data_arrays['data_degrateAPC32'],
                          param_dict['n3'], param_dict['N3'], param_dict['n2'], param_dict['N2'],
                          k_degAPC, mech_params, pair12=False)
    )
    
    # Velcade Mutant
    k_velcade = max(param_dict['beta3_k'] * param_dict['k'], 0.001)
    nlls['velcade'] = (
        compute_strain_nll(mechanism, data_arrays['data_velcade12'],
                          param_dict['n1'], param_dict['N1'], param_dict['n2'], param_dict['N2'],
                          k_velcade, mech_params, pair12=True) +
        compute_strain_nll(mechanism, data_arrays['data_velcade32'],
                          param_dict['n3'], param_dict['N3'], param_dict['n2'], param_dict['N2'],
                          k_velcade, mech_params, pair12=False)
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

    # Validate constraints
    if not validate_constraints(param_dict):
        return np.inf

    # Extract mechanism-specific parameters
    mech_params = {}
    if mechanism == 'fixed_burst':
        mech_params['burst_size'] = param_dict['burst_size']
    elif mechanism == 'feedback_onion':
        mech_params['n_inner'] = param_dict['n_inner']
    elif mechanism == 'fixed_burst_feedback_onion':
        mech_params['burst_size'] = param_dict['burst_size']
        mech_params['n_inner'] = param_dict['n_inner']

    total_nll = 0.0

    # Wild-Type (Chrom1–Chrom2 and Chrom3–Chrom2)
    nll_wt12 = compute_strain_nll(mechanism, data_wt12, param_dict['n1'], param_dict['N1'],
                                   param_dict['n2'], param_dict['N2'], param_dict['k'], mech_params, pair12=True)
    if nll_wt12 == np.inf:
        return np.inf
    total_nll += nll_wt12

    nll_wt32 = compute_strain_nll(mechanism, data_wt32, param_dict['n3'], param_dict['N3'],
                                   param_dict['n2'], param_dict['N2'], param_dict['k'], mech_params, pair12=False)
    if nll_wt32 == np.inf:
        return np.inf
    total_nll += nll_wt32

    # Threshold Mutant
    n1_th = max(param_dict['n1'] * param_dict['alpha'], 1)
    n2_th = max(param_dict['n2'] * param_dict['alpha'], 1)
    n3_th = max(param_dict['n3'] * param_dict['alpha'], 1)
    
    nll_th12 = compute_strain_nll(mechanism, data_threshold12, n1_th, param_dict['N1'],
                                   n2_th, param_dict['N2'], param_dict['k'], mech_params, pair12=True)
    if nll_th12 == np.inf:
        return np.inf
    total_nll += nll_th12

    nll_th32 = compute_strain_nll(mechanism, data_threshold32, n3_th, param_dict['N3'],
                                   n2_th, param_dict['N2'], param_dict['k'], mech_params, pair12=False)
    if nll_th32 == np.inf:
        return np.inf
    total_nll += nll_th32

    # Degradation Rate Mutant (Separase)
    k_deg = max(param_dict['beta_k'] * param_dict['k'], 0.0005)

    nll_deg12 = compute_strain_nll(mechanism, data_degrate12, param_dict['n1'], param_dict['N1'],
                                    param_dict['n2'], param_dict['N2'], k_deg, mech_params, pair12=True)
    if nll_deg12 == np.inf:
        return np.inf
    total_nll += nll_deg12

    nll_deg32 = compute_strain_nll(mechanism, data_degrate32, param_dict['n3'], param_dict['N3'],
                                    param_dict['n2'], param_dict['N2'], k_deg, mech_params, pair12=False)
    if nll_deg32 == np.inf:
        return np.inf
    total_nll += nll_deg32

    # Degradation Rate APC Mutant
    k_degAPC = max(param_dict['beta2_k'] * param_dict['k'], 0.0005)

    nll_degAPC12 = compute_strain_nll(mechanism, data_degrateAPC12, param_dict['n1'], param_dict['N1'],
                                       param_dict['n2'], param_dict['N2'], k_degAPC, mech_params, pair12=True)
    if nll_degAPC12 == np.inf:
        return np.inf
    total_nll += nll_degAPC12

    nll_degAPC32 = compute_strain_nll(mechanism, data_degrateAPC32, param_dict['n3'], param_dict['N3'],
                                       param_dict['n2'], param_dict['N2'], k_degAPC, mech_params, pair12=False)
    if nll_degAPC32 == np.inf:
        return np.inf
    total_nll += nll_degAPC32

    # Velcade Mutant
    k_velcade = max(param_dict['beta3_k'] * param_dict['k'], 0.0005)

    nll_velcade12 = compute_strain_nll(mechanism, data_velcade12, param_dict['n1'], param_dict['N1'],
                                        param_dict['n2'], param_dict['N2'], k_velcade, mech_params, pair12=True)
    if nll_velcade12 == np.inf:
        return np.inf
    total_nll += nll_velcade12

    nll_velcade32 = compute_strain_nll(mechanism, data_velcade32, param_dict['n3'], param_dict['N3'],
                                        param_dict['n2'], param_dict['N2'], k_velcade, mech_params, pair12=False)
    if nll_velcade32 == np.inf:
        return np.inf
    total_nll += nll_velcade32

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
        mechanism (str): 'simple', 'fixed_burst', 'feedback_onion', or 'fixed_burst_feedback_onion'
        gamma_mode (str): 'unified' for single gamma affecting all chromosomes, 'separate' for gamma1, gamma2, gamma3

    Returns:
        dict: Contains parameter names, bounds, and default indices
    
    Bounds Notes:
    - All bounds are synchronized with simulation_utils.py for consistency
    - Biological constraints (n < N) are enforced in validate_constraints()
    - Numerical bounds prevent optimization instabilities and unrealistic parameter values
    """
    # Common parameters for all mechanisms
    # These represent wild-type characteristics and are shared across all mechanism variants
    common_params = ['n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23']
    common_bounds = [
        (1.0, 50.0),       # n2: threshold cohesin count for chromosome 2 (reference)
        (50.0, 1000.0),    # N2: initial cohesin count for chromosome 2
        (0.01, 0.1),       # k: base degradation rate (per minute)
        (0.25, 4.0),       # r21: ratio n1/n2 (threshold ratio between chromosomes)
        (0.25, 4.0),       # r23: ratio n3/n2 (threshold ratio between chromosomes)
        (0.4, 2.0),        # R21: ratio N1/N2 (initial count ratio between chromosomes)
        (0.5, 5.0),        # R23: ratio N3/N2 (initial count ratio between chromosomes)
    ]
    
    # Integrality constraints: n2 and N2 must be integers (cohesin counts are discrete)
    common_integrality = [True, True, False, False, False, False, False]

    # Mutant parameters: represent biological effects of mutations
    mutant_params = ['alpha', 'beta_k', 'beta2_k', 'beta3_k']
    mutant_bounds = [
        (0.1, 0.7),        # alpha: threshold reduction factor (threshold mutants)
        (0.1, 1.0),        # beta_k: degradation rate reduction (separase mutants)
        (0.1, 1.0),        # beta2_k: degradation rate reduction (APC mutants)
        (0.1, 1.0),        # beta3_k: degradation rate reduction (Velcade mutants)
    ]
    mutant_integrality = [False, False, False, False]

    if mechanism == 'simple':
        mechanism_params = []
        mechanism_bounds = []
        mechanism_integrality = []
    elif mechanism == 'fixed_burst':
        mechanism_params = ['burst_size']
        mechanism_bounds = [(1.0, 20.0)]  # Size of cohesin bursts per degradation event
        mechanism_integrality = [True]  # burst_size must be integer
    elif mechanism == 'feedback_onion':
        mechanism_params = ['n_inner']
        mechanism_bounds = [(1.0, 4000.0)]  # Inner threshold for onion-like feedback effect
        mechanism_integrality = [True]  # n_inner must be integer
    elif mechanism == 'fixed_burst_feedback_onion':
        mechanism_params = ['burst_size', 'n_inner']
        mechanism_bounds = [
            (1.0, 20.0),     # burst_size: cohesin count per burst
            (1.0, 4000.0),   # n_inner: inner threshold for feedback
        ]
        mechanism_integrality = [True, True]  # both are integer parameters
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

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
        
        # Run optimization using differential evolution with integrality constraints
        # Recommended configuration for this problem:
        # - popsize=30: Larger population for better exploration (11-14 parameters)
        # - strategy='best1bin': Good balance of exploration/exploitation
        # - mutation=(0.5, 1.0): Adaptive mutation for diverse search
        # - recombination=0.7: Standard value, good for mixed integer problems
        # - tol=1e-6: Reasonable convergence tolerance
        # - polish=True: Local refinement of best solution
        print(f"max_iterations: {max_iterations}")
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
            strategy='best1bin',    # Changed: More robust against local minima than 'best1bin'
            mutation=(0.5, 1.0),    # Kept: High mutation helps jump over 'np.inf' cliffs
            recombination=0.7,      # Increased: Parameters (N, n, k) are highly correlated
            tol=1e-8,               # Relaxed slightly: adequate for NLL optimization
            seed=optimization_seed,
            polish=True,            # Crucial: L-BFGS-B performs the final cleanup
            workers=-1,
            integrality=mechanism_info['integrality']
            #disp=True               # Helpful to see progress
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

    print(f"Optimizing for mechanism: {mechanism}")
    print(f"Gamma mode: {gamma_mode}")
    print("NOTE: Initial proteins strain is TEMPORARILY EXCLUDED from fitting")
    print("Fitting datasets: wildtype, threshold, degrate, degrateAPC, velcade")
    print()

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
    
    # Use the reusable optimization function
    print("Running optimization using run_mom_optimization_single()...")
    result = run_mom_optimization_single(
        mechanism=mechanism,
        data_arrays=data_arrays,
        max_iterations=400,  # More iterations for thorough search
        seed=42,
        gamma_mode=gamma_mode
    )
    
    if not result['success']:
        print(f"❌ Optimization failed: {result['message']}")
        return
    
    # Get the optimized parameters
    mechanism_info = result['mechanism_info']
    best_nll = result['nll']
    best_params = result['result'].x
    param_dict = result['params']
    
    print(f"\n✅ Optimization completed successfully!")
    print(f"Best Negative Log-Likelihood: {best_nll:.4f}")
    print(f"\nOptimized Parameters:")
    print(f"  n2 = {param_dict['n2']:.2f}, N2 = {param_dict['N2']:.2f}, k = {param_dict['k']:.4f}")
    print(f"  Ratios: r21 = {param_dict['r21']:.2f}, r23 = {param_dict['r23']:.2f}, R21 = {param_dict['R21']:.2f}, R23 = {param_dict['R23']:.2f}")
    
    # Print mechanism-specific parameters
    if mechanism == 'fixed_burst':
        print(f"  Burst size = {param_dict['burst_size']:.2f}")
    elif mechanism == 'feedback_onion':
        print(f"  n_inner = {param_dict['n_inner']:.2f}")
    elif mechanism == 'fixed_burst_feedback_onion':
        print(f"  burst_size = {param_dict['burst_size']:.2f}, n_inner = {param_dict['n_inner']:.2f}")
    
    # Print mutant parameters
    print(f"  Mutants: alpha = {param_dict['alpha']:.2f}, beta_k = {param_dict['beta_k']:.2f}, beta2_k = {param_dict['beta2_k']:.2f}, beta3_k = {param_dict['beta3_k']:.2f}")
    print(f"  Derived: n1 = {param_dict['n1']:.2f}, n3 = {param_dict['n3']:.2f}, N1 = {param_dict['N1']:.2f}, N3 = {param_dict['N3']:.2f}")
    print()

    # Extract mechanism-specific parameters for computations
    mech_params = {}
    if mechanism == 'fixed_burst':
        mech_params['burst_size'] = param_dict['burst_size']
    elif mechanism == 'feedback_onion':
        mech_params['n_inner'] = param_dict['n_inner']
    elif mechanism == 'fixed_burst_feedback_onion':
        mech_params['burst_size'] = param_dict['burst_size']
        mech_params['n_inner'] = param_dict['n_inner']

    # Compute individual negative log-likelihoods for reporting
    strain_nlls = calculate_individual_nlls(mechanism, param_dict, mech_params, data_arrays)
    wt_nll = strain_nlls['wt']
    threshold_nll = strain_nlls['threshold']
    degrate_nll = strain_nlls['degrate']
    degrateAPC_nll = strain_nlls['degrateAPC']
    velcade_nll = strain_nlls['velcade']
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
    elif mechanism == 'feedback_onion':
        print(
            f"n_inner = {param_dict['n_inner']:.2f}")
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


if __name__ == "__main__":
    main()
