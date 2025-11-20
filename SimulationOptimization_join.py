#!/usr/bin/env python3
"""
Simulation-based optimization for chromosome segregation timing models.
Uses MultiMechanismSimulationTimevary for time-varying mechanisms.
Uses MultiMechanismSimulation (SecondVersion) for simple and fixed_burst mechanisms.
Joint optimization strategy: optimizes all parameters simultaneously across all datasets.

Key differences from MoM-based optimization:
- Uses simulation instead of analytical approximations
- Separase mutant affects k_max (beta * k_max)
- APC mutant affects k_1 (beta_APC * k_1)
- Other mutants follow the same pattern as before
"""

import numpy as np
import pandas as pd
import sys
import os
from scipy.optimize import differential_evolution, minimize
from simulation_utils import *
from Chromosomes_Theory import *

import warnings
warnings.filterwarnings('ignore')


MECHANISM_PARAM_NAMES = {
    'simple': ['n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23', 'alpha', 'beta_k'],
    'fixed_burst': ['n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'alpha', 'beta_k'],
    'feedback_onion': ['n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23', 'n_inner', 'alpha', 'beta_k'],
    'fixed_burst_feedback_onion': [
        'n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'n_inner', 'alpha', 'beta_k'
    ],
    'time_varying_k': [
        'n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2'
    ],
    'time_varying_k_fixed_burst': [
        'n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23',
        'burst_size', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2'
    ],
    'time_varying_k_feedback_onion': [
        'n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23',
        'n_inner', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2'
    ],
    'time_varying_k_combined': [
        'n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23',
        'burst_size', 'n_inner', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2'
    ],
    'time_varying_k_burst_onion': [
        'n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23',
        'burst_size', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2'
    ],
}


def unpack_mechanism_params(params_vector, mechanism):
    """
    Unpack parameter vector based on mechanism type.
    
    Returns:
        tuple: (base_params, alpha, beta_k, beta_tau, beta_tau2)
    """
    
    if mechanism not in MECHANISM_PARAM_NAMES:
        raise ValueError(f"Unknown mechanism: {mechanism}")
    
    param_names = MECHANISM_PARAM_NAMES[mechanism]
    if len(params_vector) < len(param_names):
        raise ValueError(
            f"Expected at least {len(param_names)} parameters for '{mechanism}', "
            f"received {len(params_vector)}."
        )
    
    params = dict(zip(param_names, params_vector))
    
    base_params = {
        'n2': params['n2'],
        'N2': params['N2'],
        'n1': max(params['r21'] * params['n2'], 1),
        'n3': max(params['r23'] * params['n2'], 1),
        'N1': max(params['R21'] * params['N2'], 1),
        'N3': max(params['R23'] * params['N2'], 1),
    }
    
    if 'k' in params:
        base_params['k'] = params['k']
    else:
        k_max = params['k_max']
        tau = params['tau']
        base_params.update({
            'k_max': k_max,
            'tau': tau,
            'k_1': k_max / tau,
        })
    
    for optional_key in ('burst_size', 'n_inner'):
        if optional_key in params:
            base_params[optional_key] = params[optional_key]
    
    alpha = params.get('alpha')
    beta_k = params.get('beta_k')
    beta_tau = params.get('beta_tau')
    beta_tau2 = params.get('beta_tau2')
    
    return base_params, alpha, beta_k, beta_tau, beta_tau2


def joint_objective(params_vector, mechanism, datasets, num_simulations=500, selected_strains=None):
    """
    Joint objective function that handles both simple and time-varying mechanisms.
    
    Args:
        params_vector: Parameter vector (mechanism-specific)
        mechanism: Mechanism name
        datasets: Experimental data dictionary
        num_simulations: Number of simulations per evaluation
        selected_strains: Optional list of strain names to include in fitting
        
    Returns:
        float: Total negative log-likelihood
    """
    try:
        # Unpack parameters based on mechanism
        base_params, alpha, beta_k, beta_tau, beta_tau2 = unpack_mechanism_params(params_vector, mechanism)
        
        # Check constraints
        if base_params['n1'] >= base_params['N1'] or \
            base_params['n2'] >= base_params['N2'] or \
            base_params['n3'] >= base_params['N3']:
            if hasattr(joint_objective, 'debug_count'):
                joint_objective.debug_count += 1
                if joint_objective.debug_count <= 5:
                    print(f"Constraint violation: n >= N")
                    print(f"  n1={base_params['n1']:.1f} >= N1={base_params['N1']:.1f}: {base_params['n1'] >= base_params['N1']}")
                    print(f"  n2={base_params['n2']:.1f} >= N2={base_params['N2']:.1f}: {base_params['n2'] >= base_params['N2']}")
                    print(f"  n3={base_params['n3']:.1f} >= N3={base_params['N3']:.1f}: {base_params['n3'] >= base_params['N3']}")
            else:
                joint_objective.debug_count = 1
            return 1e6

        total_nll = 0
        
        # Loop over all datasets
        for dataset_name, data_dict in datasets.items():
            # Apply mutant modifications using unified helper function (works for both mechanism types)
            params, n0_list = apply_mutant_params(
                base_params, dataset_name, alpha, beta_k, beta_tau, beta_tau2
            )
            
            # Run simulations
            sim_delta_t12, sim_delta_t32 = run_simulation_for_dataset(
                mechanism, params, n0_list, num_simulations
            )
            
            if sim_delta_t12 is None or sim_delta_t32 is None:
                return 1e6
            
            # Calculate likelihood using KDE
            exp_data = {'delta_t12': data_dict['delta_t12'], 'delta_t32': data_dict['delta_t32']}
            sim_data = {'delta_t12': sim_delta_t12, 'delta_t32': sim_delta_t32}
            
            nll = calculate_likelihood(exp_data, sim_data)
            
            if nll >= 1e6:
                return 1e6
            
            total_nll += nll
        
        return total_nll
    
    except Exception as e:
        print(f"Objective function error: {e}")
        return 1e6



def run_optimization(mechanism, datasets, max_iterations=500, num_simulations=500, selected_strains=None, use_parallel=False, initial_guess=None):
    """
    Run joint optimization for all mechanism types.
    
    Handles both simple mechanisms ('simple', 'fixed_burst', 'feedback_onion', 'fixed_burst_feedback_onion')
    and time-varying mechanisms ('time_varying_k' and variants).
    
    Args:
        mechanism (str): Mechanism name
        datasets (dict): Experimental datasets
        max_iterations (int): Maximum iterations for optimization
        num_simulations (int): Number of simulations per evaluation
        selected_strains (list): List of strain names to include in fitting (optional)
        use_parallel (bool): Whether to use parallel processing
        initial_guess (list, optional): Initial guess parameter vector. If None, uses random initialization.
                                       Example: [n2, N2, k_max, tau, r21, r23, R21, R23, alpha, beta_k, beta_tau, beta_tau2]
    
    Returns:
        dict: Optimization results
    """
    # Determine if this is a simple or time-varying mechanism
    is_simple = mechanism in ['simple', 'fixed_burst', 'feedback_onion', 'fixed_burst_feedback_onion']
    
    print(f"\n=== Simulation-based Optimization for {mechanism.upper()} ===")
    print(f"Using KDE with Scott's rule (automatic bandwidth)")
    
    if selected_strains is None:
        print(f"Datasets: {list(datasets.keys())} (all datasets)")
    else:
        print(f"Selected datasets: {selected_strains}")
        print(f"Available datasets: {list(datasets.keys())}")
    
    print(f"Max iterations: {max_iterations}")
    print(f"Simulations per evaluation: {num_simulations}")
    sys.stdout.flush()
    
    # Get parameter bounds
    bounds = get_parameter_bounds(mechanism)
    
    print(f"\nOptimizing {len(bounds)} parameters...")
    
    try:
        param_names = MECHANISM_PARAM_NAMES[mechanism]
    except KeyError:
        print(f"Unknown mechanism: {mechanism}")
        return {'success': False, 'message': f'Unknown mechanism: {mechanism}'}
    
    print("Parameter bounds:")
    for name, bound in zip(param_names, bounds):
        print(f"  {name}: {bound}")
    sys.stdout.flush()

    print("\nTesting parameter bounds with a quick simulation...")
    import random
    test_params = []
    for bound in bounds:
        test_params.append(random.uniform(bound[0], bound[1]))
    
    test_nll = joint_objective(test_params, mechanism, datasets, num_simulations=10, selected_strains=selected_strains)
    if test_nll < 1e6:
        print(f"✓ Parameter bounds are valid (test NLL = {test_nll:.1f})")
    else:
        print(f"⚠ Warning: Test failed with NLL = {test_nll:.1f}")
        print("  Continuing with optimization anyway...")
    
    print(f"🚀 Starting optimization with {num_simulations} simulations per evaluation...")
    if initial_guess is not None:
        print("   Using provided initial guess")
    else:
        print("   Using random initialization")
    
    # Global optimization with mechanism-specific args
    if is_simple:
        # Simple mechanisms don't use selected_strains
        opt_args = (mechanism, datasets, num_simulations)
        de_params = {
            'seed': 42,
            'disp': True,
            'tol': 1e-6,
        }
    else:
        # Time-varying mechanisms can use selected_strains
        opt_args = (mechanism, datasets, num_simulations, selected_strains)
        de_params = {
            'tol': 1e-8,
            'disp': False,
        }
    
    result = differential_evolution(
        joint_objective,
        bounds,
        args=opt_args,
        x0=initial_guess,  # Add initial guess parameter
        maxiter=max_iterations,
        popsize=20,
        workers=-1,           # Force parallel processing
        strategy='rand1bin',  # Safer than best1bin for stochastic functions
        mutation=(0.5, 1.0),  # High mutation to explore the landscape
        recombination=0.8,    # Correlated parameters (k_max and tau) need this
        polish=True,
        **de_params
    )
    
    # Always extract and display the best solution found, even if not converged
    convergence_status = "converged" if result.success else "did not converge"
    print(f"🔍 Optimization {convergence_status}!")
    print(f"Best negative log-likelihood: {result.fun:.4f}")
    
    if not result.success:
        print(f"Note: {result.message}")
    
    # Unpack and display results
    params = result.x
    param_dict = dict(zip(param_names, params))
    
    # Calculate derived parameters for display
    n1_derived = max(param_dict['r21'] * param_dict['n2'], 1)
    n3_derived = max(param_dict['r23'] * param_dict['n2'], 1)
    N1_derived = max(param_dict['R21'] * param_dict['N2'], 1)
    N3_derived = max(param_dict['R23'] * param_dict['N2'], 1)
    
    print("\nBest Parameters Found:")
    print(f"  Base: n2={param_dict['n2']:.1f}, N2={param_dict['N2']:.1f}")
    print(f"  Ratios: r21={param_dict['r21']:.2f}, r23={param_dict['r23']:.2f}, R21={param_dict['R21']:.2f}, R23={param_dict['R23']:.2f}")
    print(f"  Derived: n1={n1_derived:.1f}, n3={n3_derived:.1f}, N1={N1_derived:.1f}, N3={N3_derived:.1f}")
    
    # Print mechanism-specific parameters
    if is_simple:
        # Simple mechanisms have single 'k' parameter
        print(f"  Rate: k={param_dict['k']:.4f}")
    else:
        # Time-varying mechanisms have k_max and tau
        k_1_derived = param_dict['k_max'] / param_dict['tau']
        print(f"  Rates: k_max={param_dict['k_max']:.4f}, tau={param_dict['tau']:.1f} min, k_1={k_1_derived:.6f}")
    
    # Print optional mechanism features (burst, onion)
    if 'burst_size' in param_dict:
        print(f"  Burst size: {param_dict['burst_size']:.1f}")
    if 'n_inner' in param_dict:
        print(f"  Inner threshold: {param_dict['n_inner']:.1f}")
    
    # Print mutant parameters
    print(f"  Mutants: alpha={param_dict['alpha']:.3f}, beta_k={param_dict['beta_k']:.3f}", end='')
    if 'beta_tau' in param_dict:
        print(f", beta_tau={param_dict['beta_tau']:.3f}, beta_tau2={param_dict['beta_tau2']:.3f}")
    else:
        print()
    
    sys.stdout.flush()
    
    return {
        'success': True,  # Always treat as success to save results
        'converged': result.success,  # Track actual convergence status
        'params': param_dict,
        'nll': result.fun,
        'result': result,
        'message': result.message if not result.success else "Converged successfully"
    }


def save_results(mechanism, results, filename=None, selected_strains=None):
    """
    Save optimization results to file.
    
    Args:
        mechanism (str): Mechanism name
        results (dict): Optimization results
        filename (str): Output filename (optional)
        selected_strains (list): List of strains used in fitting
    """
    if not results['success']:
        print("Cannot save results - optimization failed")
        return
    
    # Determine filename based on selected strains
    if filename is None:
        # Create strain suffix if specific strains were selected
        strain_suffix = ""
        if selected_strains is not None:
            strain_suffix = f"_{'_'.join(selected_strains)}"
        
        filename = f"simulation_optimized_parameters_{mechanism}{strain_suffix}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"Simulation-based Optimization Results (KDE)\n")
        f.write(f"Mechanism: {mechanism}\n")
        
        # Add strain selection information
        if selected_strains is not None:
            f.write(f"Selected Strains: {', '.join(selected_strains)}\n")
        else:
            f.write(f"Selected Strains: all datasets\n")
        
        f.write(f"Negative Log-Likelihood: {results['nll']:.6f}\n")
        f.write(f"Converged: {results.get('converged', 'Unknown')}\n")
        f.write(f"Status: {results.get('message', 'No message')}\n")
        f.write(f"Available Datasets: wildtype, threshold, degrade, degradeAPC, velcade\n\n")
        
        f.write("Optimized Parameters (ratio-based):\n")
        for param, value in results['params'].items():
            f.write(f"{param} = {value:.6f}\n")
        
        # Calculate and save derived parameters
        if 'r21' in results['params']:
            n1_derived = max(results['params']['r21'] * results['params']['n2'], 1)
            n3_derived = max(results['params']['r23'] * results['params']['n2'], 1)
            N1_derived = max(results['params']['R21'] * results['params']['N2'], 1)
            N3_derived = max(results['params']['R23'] * results['params']['N2'], 1)
            
            f.write(f"\nDerived Parameters:\n")
            f.write(f"n1 = {n1_derived:.6f}\n")
            f.write(f"n3 = {n3_derived:.6f}\n")
            f.write(f"N1 = {N1_derived:.6f}\n")
            f.write(f"N3 = {N3_derived:.6f}\n")
    
    print(f"Results saved to: {filename}")


def main():
    """
    Main optimization routine - now supports both simple and time-varying mechanisms.
    """
    max_iterations = 1000  # number of iterations for testing
    num_simulations = 5  # Number of simulations per evaluation
    
    print("Simulation-based Optimization with KDE")
    print("=" * 60)
    sys.stdout.flush()
    
    # Load experimental data
    datasets = load_experimental_data()
    if not datasets:
        print("Error: No datasets loaded!")
        sys.stdout.flush()
        return
    
    sys.stdout.flush()
    
    # Choose mechanism type - can be overridden by command line argument
    # Simple mechanisms: 'simple', 'fixed_burst', 'feedback_onion', 'fixed_burst_feedback_onion'
    # Time-varying mechanisms: 'time_varying_k', 'time_varying_k_fixed_burst', 'time_varying_k_feedback_onion', 'time_varying_k_combined'
    
    mechanism = 'simple'  # default mechanism
    
    print(f"Optimizing {mechanism} for ALL strains")
    sys.stdout.flush()
    
    try:
        # Use unified optimization function that handles both simple and time-varying
        results = run_optimization(
            mechanism, datasets, 
            max_iterations=max_iterations, 
            num_simulations=num_simulations,
            selected_strains=None,
            use_parallel=True,
            initial_guess=None
        )
        
        # Save results
        save_results(mechanism, results, selected_strains=None)
        
        print(f"\n{'-' * 60}")
        sys.stdout.flush()
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
     
    print("Optimization complete!")
    sys.stdout.flush()


if __name__ == "__main__":
    main() 