"""
Bayesian optimization for chromosome segregation mechanisms using scikit-optimize.

This is an alternative to differential_evolution for comparison purposes.
Uses Gaussian Process-based Bayesian optimization for parameter search.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import sys
import os

# Import from the existing MoM module
sys.path.append(os.path.dirname(__file__))
from MoMCalculations import compute_pdf_for_mechanism


def unpack_parameters(params, mechanism_info):
    """
    Unpack optimization parameters based on mechanism.
    (Same as original MoMOptimization_join.py)
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
    (Same as original MoMOptimization_join.py)
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
    n1_th = max(param_dict['n1'] * param_dict['alpha'], 1)
    n2_th = max(param_dict['n2'] * param_dict['alpha'], 1)
    n3_th = max(param_dict['n3'] * param_dict['alpha'], 1)

    if n1_th >= param_dict['N1'] or n2_th >= param_dict['N2'] or n3_th >= param_dict['N3']:
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

    # Wild-Type
    pdf_wt12 = compute_pdf_for_mechanism(mechanism, data_wt12, param_dict['n1'], param_dict['N1'],
                                         param_dict['n2'], param_dict['N2'], param_dict['k'], mech_params, pair12=True)
    if np.any(pdf_wt12 <= 0) or np.any(np.isnan(pdf_wt12)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_wt12))

    pdf_wt32 = compute_pdf_for_mechanism(mechanism, data_wt32, param_dict['n3'], param_dict['N3'],
                                         param_dict['n2'], param_dict['N2'], param_dict['k'], mech_params, pair12=False)
    if np.any(pdf_wt32 <= 0) or np.any(np.isnan(pdf_wt32)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_wt32))

    # Threshold Mutant
    pdf_th12 = compute_pdf_for_mechanism(mechanism, data_threshold12, n1_th, param_dict['N1'],
                                         n2_th, param_dict['N2'], param_dict['k'], mech_params, pair12=True)
    if np.any(pdf_th12 <= 0) or np.any(np.isnan(pdf_th12)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_th12))

    pdf_th32 = compute_pdf_for_mechanism(mechanism, data_threshold32, n3_th, param_dict['N3'],
                                         n2_th, param_dict['N2'], param_dict['k'], mech_params, pair12=False)
    if np.any(pdf_th32 <= 0) or np.any(np.isnan(pdf_th32)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_th32))

    # Degradation Rate Mutant
    k_deg = max(param_dict['beta_k'] * param_dict['k'], 0.0005)
    pdf_deg12 = compute_pdf_for_mechanism(mechanism, data_degrate12, param_dict['n1'], param_dict['N1'],
                                          param_dict['n2'], param_dict['N2'], k_deg, mech_params, pair12=True)
    if np.any(pdf_deg12 <= 0) or np.any(np.isnan(pdf_deg12)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_deg12))

    pdf_deg32 = compute_pdf_for_mechanism(mechanism, data_degrate32, param_dict['n3'], param_dict['N3'],
                                          param_dict['n2'], param_dict['N2'], k_deg, mech_params, pair12=False)
    if np.any(pdf_deg32 <= 0) or np.any(np.isnan(pdf_deg32)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_deg32))

    # Degradation Rate APC Mutant
    k_degAPC = max(param_dict['beta2_k'] * param_dict['k'], 0.0005)
    pdf_degAPC12 = compute_pdf_for_mechanism(mechanism, data_degrateAPC12, param_dict['n1'], param_dict['N1'],
                                             param_dict['n2'], param_dict['N2'], k_degAPC, mech_params, pair12=True)
    if np.any(pdf_degAPC12 <= 0) or np.any(np.isnan(pdf_degAPC12)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_degAPC12))

    pdf_degAPC32 = compute_pdf_for_mechanism(mechanism, data_degrateAPC32, param_dict['n3'], param_dict['N3'],
                                             param_dict['n2'], param_dict['N2'], k_degAPC, mech_params, pair12=False)
    if np.any(pdf_degAPC32 <= 0) or np.any(np.isnan(pdf_degAPC32)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_degAPC32))

    # Velcade Mutant
    k_velcade = max(param_dict['beta3_k'] * param_dict['k'], 0.0005)
    pdf_velcade12 = compute_pdf_for_mechanism(mechanism, data_velcade12, param_dict['n1'], param_dict['N1'],
                                              param_dict['n2'], param_dict['N2'], k_velcade, mech_params, pair12=True)
    if np.any(pdf_velcade12 <= 0) or np.any(np.isnan(pdf_velcade12)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_velcade12))

    pdf_velcade32 = compute_pdf_for_mechanism(mechanism, data_velcade32, param_dict['n3'], param_dict['N3'],
                                              param_dict['n2'], param_dict['N2'], k_velcade, mech_params, pair12=False)
    if np.any(pdf_velcade32 <= 0) or np.any(np.isnan(pdf_velcade32)):
        return np.inf
    total_nll -= np.sum(np.log(pdf_velcade32))

    return total_nll


def get_mechanism_info(mechanism, gamma_mode):
    """
    Get mechanism-specific parameter information.
    (Same as original MoMOptimization_join.py)
    """
    common_params = ['n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23']
    common_bounds = [
        (1.0, 50.0),      # n2
        (50.0, 1000.0),   # N2
        (0.01, 0.1),      # k
        (0.25, 4.0),      # r21
        (0.25, 4.0),      # r23
        (0.4, 2),         # R21
        (0.5, 5.0),       # R23
    ]

    mutant_params = ['alpha', 'beta_k', 'beta2_k', 'beta3_k']
    mutant_bounds = [
        (0.1, 0.7),       # alpha
        (0.1, 1.0),       # beta_k
        (0.1, 1.0),       # beta2_k
        (0.1, 1.0),       # beta3_k
    ]

    if mechanism == 'simple':
        mechanism_params = []
        mechanism_bounds = []
    elif mechanism == 'fixed_burst':
        mechanism_params = ['burst_size']
        mechanism_bounds = [(1.0, 20.0)]
    elif mechanism == 'feedback_onion':
        mechanism_params = ['n_inner']
        mechanism_bounds = [(1.0, 4000.0)]
    elif mechanism == 'fixed_burst_feedback_onion':
        mechanism_params = ['burst_size', 'n_inner']
        mechanism_bounds = [(1.0, 20.0), (1.0, 4000.0)]
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


def run_bayesian_optimization(mechanism, data_arrays=None, n_calls=150, n_random_starts=30, seed=None, gamma_mode='separate'):
    """
    Run Bayesian optimization for a given mechanism using Gaussian Processes.
    
    Args:
        mechanism (str): Mechanism name
        data_arrays (dict): Dictionary containing data arrays
        n_calls (int): Number of function evaluations (similar to maxiter * popsize for DE)
        n_random_starts (int): Number of random initialization points before GP modeling
        seed (int): Random seed for reproducibility
        gamma_mode (str): 'unified' or 'separate' gamma mode
    
    Returns:
        dict: Results dictionary with success, nll, params, etc.
    """
    try:
        print(f"🔬 Starting Bayesian optimization for {mechanism}")
        print(f"   n_calls={n_calls}, n_random_starts={n_random_starts}")
        
        # Get mechanism-specific information
        mechanism_info = get_mechanism_info(mechanism, gamma_mode)
        bounds = mechanism_info['bounds']
        param_names = mechanism_info['params']
        
        # Use provided data arrays or load from file
        if data_arrays is not None:
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
            df = pd.read_excel("Data/All_strains_SCStimes.xlsx")
            data_wt12 = df['wildtype12'].dropna().values
            data_wt32 = df['wildtype32'].dropna().values
            data_threshold12 = df['threshold12'].dropna().values
            data_threshold32 = df['threshold32'].dropna().values
            data_degrate12 = df['degRade12'].dropna().values
            data_degrate32 = df['degRade32'].dropna().values
            data_initial12 = df['initialProteins12'].dropna().values if 'initialProteins12' in df.columns else np.array([])
            data_initial32 = df['initialProteins32'].dropna().values if 'initialProteins32' in df.columns else np.array([])
            data_degrateAPC12 = df['degRadeAPC12'].dropna().values
            data_degrateAPC32 = df['degRadeAPC32'].dropna().values
            data_velcade12 = df['degRadeVel12'].dropna().values
            data_velcade32 = df['degRadeVel32'].dropna().values
        
        # Create search space for scikit-optimize
        search_space = [Real(low, high, name=name) for (low, high), name in zip(bounds, param_names)]
        
        # Track best valid NLL seen so far for adaptive penalty
        best_valid_nll = [np.inf]  # Use list to allow modification in nested function
        
        # Define objective function wrapper for scikit-optimize
        @use_named_args(search_space)
        def objective_wrapper(**params):
            # Convert named parameters to array
            params_array = np.array([params[name] for name in param_names])
            
            try:
                # Call original objective
                nll = joint_objective(
                    params_array, mechanism, mechanism_info,
                    data_wt12, data_wt32, data_threshold12, data_threshold32,
                    data_degrate12, data_degrate32, data_initial12, data_initial32,
                    data_degrateAPC12, data_degrateAPC32, data_velcade12, data_velcade32
                )
                
                # Bayesian optimization cannot handle np.inf or NaN
                # Replace with adaptive penalty based on best valid value seen
                if np.isinf(nll) or np.isnan(nll):
                    # Use adaptive penalty: 2x best valid NLL, or 1e8 if no valid seen yet
                    penalty = 2.0 * best_valid_nll[0] if np.isfinite(best_valid_nll[0]) else 1e8
                    return penalty
                
                # Update best valid NLL
                if nll < best_valid_nll[0]:
                    best_valid_nll[0] = nll
                
                return nll
                
            except Exception as e:
                # Handle any unexpected errors gracefully
                print(f"   ⚠️  Error in objective evaluation: {e}")
                penalty = 2.0 * best_valid_nll[0] if np.isfinite(best_valid_nll[0]) else 1e8
                return penalty
        
        # Run Bayesian optimization with progress tracking
        print(f"   Running GP optimization...")
        result = gp_minimize(
            objective_wrapper,
            search_space,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            acq_func='EI',  # Expected Improvement
            random_state=seed if seed is not None else 42,
            verbose=False,
            n_jobs=1,  # Sequential evaluation (GP doesn't parallelize well)
            n_initial_points=n_random_starts,  # Explicit initial points
            acq_optimizer='lbfgs'  # Use L-BFGS-B for acquisition function optimization
        )
        
        # Local refinement using L-BFGS-B
        print(f"   Bayesian optimization complete. Best NLL: {result.fun:.4f}")
        print(f"   Running local refinement...")
        
        result_local = minimize(
            lambda x: joint_objective(
                x, mechanism, mechanism_info,
                data_wt12, data_wt32, data_threshold12, data_threshold32,
                data_degrate12, data_degrate32, data_initial12, data_initial32,
                data_degrateAPC12, data_degrateAPC32, data_velcade12, data_velcade32
            ),
            x0=result.x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        
        # Use local result if better
        if result_local.fun < result.fun:
            print(f"   Local refinement improved: {result.fun:.4f} → {result_local.fun:.4f}")
            final_params = result_local.x
            final_nll = result_local.fun
            converged = result_local.success
        else:
            print(f"   Bayesian result was better: {result.fun:.4f}")
            final_params = result.x
            final_nll = result.fun
            converged = True
        
        # Convert to standard format
        return {
            'success': True,
            'converged': converged,
            'nll': final_nll,
            'params': dict(zip(param_names, final_params)),
            'result': result,
            'message': "Bayesian optimization completed successfully",
            'mechanism_info': mechanism_info,
            'n_calls': n_calls
        }
        
    except Exception as e:
        print(f"   ❌ Bayesian optimization failed: {e}")
        return {
            'success': False,
            'converged': False,
            'nll': np.inf,
            'params': {},
            'result': None,
            'message': f"Bayesian optimization failed: {e}",
            'mechanism_info': None
        }


if __name__ == "__main__":
    print("Bayesian Optimization for MoM-based mechanisms")
    print("=" * 60)
    print("Note: This uses scikit-optimize (skopt)")
    print("Install with: pip install scikit-optimize")
    print("=" * 60)
    
    # Test with simple mechanism
    mechanism = 'simple'
    result = run_bayesian_optimization(
        mechanism=mechanism,
        n_calls=100,
        n_random_starts=20,
        seed=42,
        gamma_mode='separate'
    )
    
    if result['success']:
        print(f"\n✅ Optimization successful!")
        print(f"NLL: {result['nll']:.4f}")
        print(f"Parameters: {result['params']}")
    else:
        print(f"\n❌ Optimization failed: {result['message']}")

