#!/usr/bin/env python3
"""
Simulation-based optimization for chromosome segregation timing models.
Uses MultiMechanismSimulationTimevary for time-varying mechanisms.
Joint optimization strategy: optimizes all parameters simultaneously across all datasets.

Key differences from MoM-based optimization:
- Uses simulation instead of analytical approximations
- Separase mutant affects k_max (beta * k_max)
- APC mutant affects k_1 (beta_APC * k_1)
- Other mutants follow the same pattern as before
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from simulation_utils import *
from Chromosomes_Theory import *
import warnings
warnings.filterwarnings('ignore')


# load_experimental_data is now imported from simulation_utils


# apply_mutant_params is now imported from simulation_utils


# run_simulation_for_dataset is now imported from simulation_utils


# calculate_likelihood is now imported from simulation_utils


def joint_objective_with_bootstrapping(params_vector, mechanism, datasets, 
                                      num_simulations=500, 
                                      bootstrap_method='bootstrap',
                                      target_sample_size=50, 
                                      num_bootstrap_samples=100,
                                      random_seed=None,
                                      selected_strains=None):
    """
    Joint objective function with bootstrapping to handle unequal data points.
    
    Args:
        params_vector (array): Parameter vector to optimize
        mechanism (str): Mechanism name
        datasets (dict): Experimental datasets
        num_simulations (int): Number of simulations per evaluation
        bootstrap_method (str): 'bootstrap', 'weighted', or 'standard'
        target_sample_size (int): Target sample size for bootstrapping
        num_bootstrap_samples (int): Number of bootstrap samples
        random_seed (int, optional): Random seed for reproducibility
    
    Returns:
        float: Total negative log-likelihood across all datasets
    """
    # Add debugging to see parameter values being tried
    if hasattr(joint_objective_with_bootstrapping, 'call_count'):
        joint_objective_with_bootstrapping.call_count += 1
    else:
        joint_objective_with_bootstrapping.call_count = 1
    
    if joint_objective_with_bootstrapping.call_count <= 5:  # Print first 5 calls for debugging
        print(f"Bootstrap Call {joint_objective_with_bootstrapping.call_count}: Testing parameters {params_vector[:6]}")  # Show first 6 params
    
    try:
        # Initialize bootstrapping calculator if needed
        if bootstrap_method in ['bootstrap', 'weighted']:
            bootstrap_calc = BootstrappingFitnessCalculator(
                target_sample_size=target_sample_size,
                num_bootstrap_samples=num_bootstrap_samples,
                random_seed=random_seed
            )
        
        # Unpack parameters based on mechanism - using ratio-based approach
        if mechanism == 'time_varying_k':
            n2, N2, k_1, k_max, r21, r23, R21, R23, alpha, beta_k, beta_tau = params_vector
            # Calculate derived parameters from ratios
            n1 = max(r21 * n2, 1)
            n3 = max(r23 * n2, 1)
            N1 = max(R21 * N2, 1)
            N3 = max(R23 * N2, 1)
            
            # Add constraint checks similar to MoM optimization
            if n1 >= N1 or n2 >= N2 or n3 >= N3:
                print(f"Constraint violation: n >= N. n1={n1:.1f}, N1={N1:.1f}, n2={n2:.1f}, N2={N2:.1f}, n3={n3:.1f}, N3={N3:.1f}")
                return 1e6
            
            base_params = {
                'n1': n1, 'n2': n2, 'n3': n3,
                'N1': N1, 'N2': N2, 'N3': N3,
                'k_1': k_1, 'k_max': k_max
            }
        elif mechanism == 'time_varying_k_fixed_burst':
            n2, N2, k_1, k_max, r21, r23, R21, R23, burst_size, alpha, beta_k, beta_tau = params_vector
            # Calculate derived parameters from ratios
            n1 = max(r21 * n2, 1)
            n3 = max(r23 * n2, 1)
            N1 = max(R21 * N2, 1)
            N3 = max(R23 * N2, 1)
            base_params = {
                'n1': n1, 'n2': n2, 'n3': n3,
                'N1': N1, 'N2': N2, 'N3': N3,
                'k_1': k_1, 'k_max': k_max, 'burst_size': burst_size
            }
        elif mechanism == 'time_varying_k_feedback_onion':
            n2, N2, k_1, k_max, r21, r23, R21, R23, n_inner, alpha, beta_k, beta_tau = params_vector
            # Calculate derived parameters from ratios
            n1 = max(r21 * n2, 1)
            n3 = max(r23 * n2, 1)
            N1 = max(R21 * N2, 1)
            N3 = max(R23 * N2, 1)
            base_params = {
                'n1': n1, 'n2': n2, 'n3': n3,
                'N1': N1, 'N2': N2, 'N3': N3,
                'k_1': k_1, 'k_max': k_max, 'n_inner': n_inner
            }
        elif mechanism == 'time_varying_k_combined':
            n2, N2, k_1, k_max, r21, r23, R21, R23, burst_size, n_inner, alpha, beta_k, beta_tau = params_vector
            # Calculate derived parameters from ratios
            n1 = max(r21 * n2, 1)
            n3 = max(r23 * n2, 1)
            N1 = max(R21 * N2, 1)
            N3 = max(R23 * N2, 1)
            
            # Add constraint checks similar to MoM optimization
            if n1 >= N1 or n2 >= N2 or n3 >= N3:
                print(f"Constraint violation: n >= N. n1={n1:.1f}, N1={N1:.1f}, n2={n2:.1f}, N2={N2:.1f}, n3={n3:.1f}, N3={N3:.1f}")
                return 1e6
            
            base_params = {
                'n1': n1, 'n2': n2, 'n3': n3,
                'N1': N1, 'N2': N2, 'N3': N3,
                'k_1': k_1, 'k_max': k_max, 'burst_size': burst_size, 'n_inner': n_inner
            }
        elif mechanism == 'time_varying_k_burst_onion':
            n2, N2, k_1, k_max, r21, r23, R21, R23, burst_size, alpha, beta_k, beta_tau = params_vector
            # Calculate derived parameters from ratios
            n1 = max(r21 * n2, 1)
            n3 = max(r23 * n2, 1)
            N1 = max(R21 * N2, 1)
            N3 = max(R23 * N2, 1)
            
            # Add constraint checks similar to MoM optimization
            if n1 >= N1 or n2 >= N2 or n3 >= N3:
                print(f"Constraint violation: n >= N. n1={n1:.1f}, N1={N1:.1f}, n2={n2:.1f}, N2={N2:.1f}, n3={n3:.1f}, N3={N3:.1f}")
                return 1e6
            
            base_params = {
                'n1': n1, 'n2': n2, 'n3': n3,
                'N1': N1, 'N2': N2, 'N3': N3,
                'k_1': k_1, 'k_max': k_max, 'burst_size': burst_size
            }
        else:
            return 1e6
        
        total_nll = 0
        
        # Filter datasets based on selected strains
        if selected_strains is None:
            # Use all datasets if no selection specified
            datasets_to_use = datasets
        else:
            datasets_to_use = {name: data for name, data in datasets.items() if name in selected_strains}
        
        if not datasets_to_use:
            print("Error: No datasets selected for fitting!")
            return 1e6
        
        for dataset_name, data_dict in datasets_to_use.items():
            # Apply mutant-specific modifications
            params, n0_list = apply_mutant_params(
                base_params, dataset_name, alpha, beta_k, beta_tau
            )
            
            # Run simulations
            sim_delta_t12, sim_delta_t32 = run_simulation_for_dataset(
                mechanism, params, n0_list, num_simulations
            )
            
            if sim_delta_t12 is None or sim_delta_t32 is None:
                print(f"Simulation failed for dataset {dataset_name}")
                return 1e6
            
            # Extract experimental data
            exp_delta_t12 = data_dict['delta_t12']
            exp_delta_t32 = data_dict['delta_t32']
            
            # Calculate likelihoods using specified method
            if bootstrap_method == 'bootstrap':
                nll_12 = bootstrap_calc.calculate_bootstrap_likelihood(exp_delta_t12, sim_delta_t12)
                nll_32 = bootstrap_calc.calculate_bootstrap_likelihood(exp_delta_t32, sim_delta_t32)
            elif bootstrap_method == 'weighted':
                nll_12 = bootstrap_calc.calculate_weighted_likelihood(exp_delta_t12, sim_delta_t12)
                nll_32 = bootstrap_calc.calculate_weighted_likelihood(exp_delta_t32, sim_delta_t32)
            else:  # standard method
                nll_12 = calculate_likelihood(exp_delta_t12, sim_delta_t12)
                nll_32 = calculate_likelihood(exp_delta_t32, sim_delta_t32)
            
            # Check for penalty values in likelihood calculation
            if nll_12 >= 1e6 or nll_32 >= 1e6:
                print(f"High likelihood penalty for dataset {dataset_name}: nll_12={nll_12:.1f}, nll_32={nll_32:.1f}")
                return 1e6
            
            # Add to total (no additional weighting needed since bootstrapping handles it)
            total_nll += nll_12 + nll_32
        
        return total_nll
    
    except Exception as e:
        print(f"Bootstrap objective function error: {e}")
        return 1e6


# In SimulationOptimization_join.py

def joint_objective(params_vector, mechanism, datasets, num_simulations=500, selected_strains=None):
    """
    Joint objective function that correctly handles simulation failures.
    """
    try:
        # (Your existing code for parameter unpacking and monitoring)
        # ...
        # (All parameter unpacking logic remains the same)
        # ...
        if mechanism == 'time_varying_k':
            n2, N2, k_max, tau, r21, r23, R21, R23, alpha, beta_k, beta_tau = params_vector
            k_1 = k_max / tau
            base_params = {
                'n1': max(r21 * n2, 1), 'n2': n2, 'n3': max(r23 * n2, 1),
                'N1': max(R21 * N2, 1), 'N2': N2, 'N3': max(R23 * N2, 1),
                'k_1': k_1, 'k_max': k_max, 'tau': tau
            }
        elif mechanism == 'time_varying_k_fixed_burst':
            n2, N2, k_max, tau, r21, r23, R21, R23, burst_size, alpha, beta_k, beta_tau = params_vector
            k_1 = k_max / tau
            base_params = {
                'n1': max(r21 * n2, 1), 'n2': n2, 'n3': max(r23 * n2, 1),
                'N1': max(R21 * N2, 1), 'N2': N2, 'N3': max(R23 * N2, 1),
                'k_1': k_1, 'k_max': k_max, 'tau': tau, 'burst_size': burst_size
            }
        elif mechanism == 'time_varying_k_feedback_onion':
            n2, N2, k_max, tau, r21, r23, R21, R23, n_inner, alpha, beta_k, beta_tau = params_vector
            k_1 = k_max / tau
            base_params = {
                'n1': max(r21 * n2, 1), 'n2': n2, 'n3': max(r23 * n2, 1),
                'N1': max(R21 * N2, 1), 'N2': N2, 'N3': max(R23 * N2, 1),
                'k_1': k_1, 'k_max': k_max, 'tau': tau, 'n_inner': n_inner
            }
        elif mechanism == 'time_varying_k_combined':
            n2, N2, k_max, tau, r21, r23, R21, R23, burst_size, n_inner, alpha, beta_k, beta_tau = params_vector
            k_1 = k_max / tau
            base_params = {
                'n1': max(r21 * n2, 1), 'n2': n2, 'n3': max(r23 * n2, 1),
                'N1': max(R21 * N2, 1), 'N2': N2, 'N3': max(R23 * N2, 1),
                'k_1': k_1, 'k_max': k_max, 'tau': tau, 'burst_size': burst_size, 'n_inner': n_inner
            }
        elif mechanism == 'time_varying_k_burst_onion':
            n2, N2, k_max, tau, r21, r23, R21, R23, burst_size, alpha, beta_k, beta_tau = params_vector
            k_1 = k_max / tau
            base_params = {
                'n1': max(r21 * n2, 1), 'n2': n2, 'n3': max(r23 * n2, 1),
                'N1': max(R21 * N2, 1), 'N2': N2, 'N3': max(R23 * N2, 1),
                'k_1': k_1, 'k_max': k_max, 'tau': tau, 'burst_size': burst_size
            }
        else:
            return 1e6
        
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
        
        # Use all datasets
        datasets_to_use = datasets
        
        for dataset_name, data_dict in datasets_to_use.items():
            params, n0_list = apply_mutant_params(
                base_params, dataset_name, alpha, beta_k, beta_tau
            )
            
            sim_delta_t12, sim_delta_t32 = run_simulation_for_dataset(
                mechanism, params, n0_list, num_simulations
            )
            
            # --- THIS IS THE SECOND CRITICAL CHANGE ---
            # Check for the failure signal from the simulation function.
            if sim_delta_t12 is None or sim_delta_t32 is None:
                return 1e6

            exp_delta_t12 = data_dict['delta_t12']
            exp_delta_t32 = data_dict['delta_t32']
            
            # Create proper data dictionaries for likelihood calculation
            exp_data = {'delta_t12': exp_delta_t12, 'delta_t32': exp_delta_t32}
            sim_data = {'delta_t12': np.array(sim_delta_t12), 'delta_t32': np.array(sim_delta_t32)}
            
            nll_total_dataset = calculate_likelihood(exp_data, sim_data)
            
            if nll_total_dataset >= 1e6:
                return 1e6
            
            total_nll += nll_total_dataset
        
        # (Monitoring printout logic remains the same)
        # ...

        return total_nll
    
    except Exception as e:
        return 1e6


# get_parameter_bounds is now imported from simulation_utils


def run_optimization(mechanism, datasets, max_iterations=300, num_simulations=500, selected_strains=None, use_parallel=False):
    """
    Run joint optimization for selected datasets.
    
    Args:
        mechanism (str): Mechanism name
        datasets (dict): Experimental datasets
        max_iterations (int): Maximum iterations for optimization
        num_simulations (int): Number of simulations per evaluation
        selected_strains (list): List of strain names to include in fitting
    
    Returns:
        dict: Optimization results
    """
    print(f"\n=== Joint Optimization for {mechanism.upper()} ===")
    if selected_strains is None:
        print(f"Datasets: {list(datasets.keys())} (all datasets)")
    else:
        print(f"Selected datasets: {selected_strains}")
        print(f"Available datasets: {list(datasets.keys())}")
    print(f"Max iterations: {max_iterations}")
    
    # Get parameter bounds
    bounds = get_parameter_bounds(mechanism)
    
    print(f"Optimizing {len(bounds)} parameters...")
    print("Running differential evolution...")
    
    # Add debugging to see parameter bounds
    print("Parameter bounds:")
    if mechanism == 'time_varying_k_fixed_burst':
        param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'alpha', 'beta_k', 'beta_tau']
        for i, (name, bound) in enumerate(zip(param_names, bounds)):
            print(f"  {name}: {bound}")
    
    # Quick test to ensure optimization will work
    print("\nTesting parameter bounds with a quick simulation...")
    import random
    test_params = []
    for bound in bounds:
        test_params.append(random.uniform(bound[0], bound[1]))
    
    test_nll = joint_objective(test_params, mechanism, datasets, num_simulations=10, selected_strains=selected_strains)
    if test_nll < 1e6:
        print(f"✓ Parameter bounds are valid (test NLL = {test_nll:.1f})")
        print(f"🚀 Starting optimization with {num_simulations} simulations per evaluation...")
    else:
        print(f"⚠ Warning: Test failed with NLL = {test_nll:.1f}")
        print("  Continuing with optimization anyway...")
    
    # Global optimization
    result = differential_evolution(
        joint_objective,
        bounds,
        args=(mechanism, datasets, num_simulations, selected_strains),
        maxiter=max_iterations,
        popsize=15,
        seed=42,
        disp=True,
        workers=-1 if use_parallel else 1,  # Use parallel if requested
        atol=1e-6,
        tol=0.01
    )
    
    # Always extract and display the best solution found, even if not converged
    convergence_status = "converged" if result.success else "did not converge"
    print(f"🔍 Optimization {convergence_status}!")
    print(f"Best negative log-likelihood: {result.fun:.4f}")
    
    if not result.success:
        print(f"Note: {result.message}")
    
    # Unpack and display results
    params = result.x
    if mechanism == 'time_varying_k':
        param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'alpha', 'beta_k', 'beta_tau']
    elif mechanism == 'time_varying_k_fixed_burst':
        param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'alpha', 'beta_k', 'beta_tau']
    elif mechanism == 'time_varying_k_feedback_onion':
        param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'n_inner', 'alpha', 'beta_k', 'beta_tau']
    elif mechanism == 'time_varying_k_combined':
        param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'n_inner', 'alpha', 'beta_k', 'beta_tau']
    elif mechanism == 'time_varying_k_burst_onion':
        param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'alpha', 'beta_k', 'beta_tau']
    
    param_dict = dict(zip(param_names, params))
    
    # Calculate derived parameters for display
    n1_derived = max(param_dict['r21'] * param_dict['n2'], 1)
    n3_derived = max(param_dict['r23'] * param_dict['n2'], 1)
    N1_derived = max(param_dict['R21'] * param_dict['N2'], 1)
    N3_derived = max(param_dict['R23'] * param_dict['N2'], 1)
    
    # Calculate k_1 from k_max and tau for display
    k_1_derived = param_dict['k_max'] / param_dict['tau']
    
    print("\nBest Parameters Found:")
    print(f"  Base: n2={param_dict['n2']:.1f}, N2={param_dict['N2']:.1f}")
    print(f"  Ratios: r21={param_dict['r21']:.2f}, r23={param_dict['r23']:.2f}, R21={param_dict['R21']:.2f}, R23={param_dict['R23']:.2f}")
    print(f"  Derived: n1={n1_derived:.1f}, n3={n3_derived:.1f}, N1={N1_derived:.1f}, N3={N3_derived:.1f}")
    print(f"  Rates: k_max={param_dict['k_max']:.4f}, tau={param_dict['tau']:.1f} min, k_1={k_1_derived:.6f}")
    
    if 'burst_size' in param_dict:
        print(f"           burst_size={param_dict['burst_size']:.1f}")
    if 'n_inner' in param_dict:
        print(f"           n_inner={param_dict['n_inner']:.1f}")
    if 'burst_size' in param_dict and 'n_inner' in param_dict:
        print(f"           Combined mechanism: burst_size={param_dict['burst_size']:.1f}, n_inner={param_dict['n_inner']:.1f}")
    
    print(f"  Mutants: alpha={param_dict['alpha']:.3f}, beta_k={param_dict['beta_k']:.3f}, beta_tau={param_dict['beta_tau']:.3f}")
    
    return {
        'success': True,  # Always treat as success to save results
        'converged': result.success,  # Track actual convergence status
        'params': param_dict,
        'nll': result.fun,
        'result': result,
        'message': result.message if not result.success else "Converged successfully"
    }


def run_optimization_with_bootstrapping(mechanism, datasets, 
                                       max_iterations=300, 
                                       num_simulations=500,
                                       bootstrap_method='bootstrap',
                                       target_sample_size=None,
                                       num_bootstrap_samples=100,
                                       random_seed=42,
                                       selected_strains=None):
    """
    Run joint optimization with bootstrapping for all datasets.
    
    Args:
        mechanism (str): Mechanism name
        datasets (dict): Experimental datasets
        max_iterations (int): Maximum iterations for optimization
        num_simulations (int): Number of simulations per evaluation
        bootstrap_method (str): 'bootstrap', 'weighted', or 'standard'
        target_sample_size (int, optional): Target sample size for bootstrapping
        num_bootstrap_samples (int): Number of bootstrap samples
        random_seed (int): Random seed for reproducibility
    
    Returns:
        dict: Optimization results
    """
    print(f"\n=== Bootstrapping Optimization for {mechanism.upper()} ===")
    print(f"Bootstrap method: {bootstrap_method}")
    if selected_strains is None:
        print(f"Datasets: {list(datasets.keys())} (all datasets)")
    else:
        print(f"Selected datasets: {selected_strains}")
        print(f"Available datasets: {list(datasets.keys())}")
    print(f"Max iterations: {max_iterations}")
    
    # Analyze dataset sizes and determine target sample size
    if selected_strains is None:
        datasets_to_analyze = datasets
    else:
        datasets_to_analyze = {name: data for name, data in datasets.items() if name in selected_strains}
    
    size_analysis = analyze_dataset_sizes(datasets_to_analyze)
    if target_sample_size is None:
        target_sample_size = size_analysis['recommended_target_size']
    
    print(f"Target sample size for bootstrapping: {target_sample_size}")
    print(f"Number of bootstrap samples: {num_bootstrap_samples}")
    
    # Get parameter bounds
    bounds = get_parameter_bounds(mechanism)
    
    print(f"Optimizing {len(bounds)} parameters...")
    print("Running differential evolution with bootstrapping...")
    
    # Global optimization with bootstrapping
    result = differential_evolution(
        joint_objective_with_bootstrapping,
        bounds,
        args=(mechanism, datasets, num_simulations, bootstrap_method, 
              target_sample_size, num_bootstrap_samples, random_seed, selected_strains),
        maxiter=max_iterations,
        popsize=15,
        seed=random_seed,
        disp=True,
        workers=1  # Use single worker to avoid multiprocessing issues
    )
    
    # Always extract and display the best solution found, even if not converged
    convergence_status = "converged" if result.success else "did not converge"
    print(f"🔍 Bootstrap optimization {convergence_status}!")
    print(f"Best negative log-likelihood: {result.fun:.4f}")
    
    if not result.success:
        print(f"Note: {result.message}")
    
    # Unpack and display results
    params = result.x
    if mechanism == 'time_varying_k':
        param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'alpha', 'beta_k', 'beta_tau']
    elif mechanism == 'time_varying_k_fixed_burst':
        param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'alpha', 'beta_k', 'beta_tau']
    elif mechanism == 'time_varying_k_feedback_onion':
        param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'n_inner', 'alpha', 'beta_k', 'beta_tau']
    elif mechanism == 'time_varying_k_combined':
        param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'n_inner', 'alpha', 'beta_k', 'beta_tau']
    elif mechanism == 'time_varying_k_burst_onion':
        param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'alpha', 'beta_k', 'beta_tau']
    
    param_dict = dict(zip(param_names, params))
    
    # Calculate derived parameters for display
    n1_derived = max(param_dict['r21'] * param_dict['n2'], 1)
    n3_derived = max(param_dict['r23'] * param_dict['n2'], 1)
    N1_derived = max(param_dict['R21'] * param_dict['N2'], 1)
    N3_derived = max(param_dict['R23'] * param_dict['N2'], 1)
    
    print("\nBest Parameters Found (with Bootstrapping):")
    print(f"  Base: n2={param_dict['n2']:.1f}, N2={param_dict['N2']:.1f}")
    print(f"  Ratios: r21={param_dict['r21']:.2f}, r23={param_dict['r23']:.2f}, R21={param_dict['R21']:.2f}, R23={param_dict['R23']:.2f}")
    print(f"  Derived: n1={n1_derived:.1f}, n3={n3_derived:.1f}, N1={N1_derived:.1f}, N3={N3_derived:.1f}")
    print(f"  Rates: k_1={param_dict['k_1']:.6f}, k_max={param_dict['k_max']:.4f}")
    
    if 'burst_size' in param_dict:
        print(f"           burst_size={param_dict['burst_size']:.1f}")
    if 'n_inner' in param_dict:
        print(f"           n_inner={param_dict['n_inner']:.1f}")
    if 'burst_size' in param_dict and 'n_inner' in param_dict:
        print(f"           Combined mechanism: burst_size={param_dict['burst_size']:.1f}, n_inner={param_dict['n_inner']:.1f}")
    
    print(f"  Mutants: alpha={param_dict['alpha']:.3f}, beta_k={param_dict['beta_k']:.3f}, beta_tau={param_dict['beta_tau']:.3f}")
    
    return {
        'success': True,  # Always treat as success to save results
        'converged': result.success,  # Track actual convergence status
        'params': param_dict,
        'nll': result.fun,
        'result': result,
        'message': result.message if not result.success else "Converged successfully",
        'bootstrap_method': bootstrap_method,
        'target_sample_size': target_sample_size,
        'num_bootstrap_samples': num_bootstrap_samples
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
    
    # Determine filename based on bootstrap method and selected strains
    if filename is None:
        # Create strain suffix if specific strains were selected
        strain_suffix = ""
        if selected_strains is not None:
            strain_suffix = f"_{'_'.join(selected_strains)}"
        
        if 'bootstrap_method' in results:
            bootstrap_suffix = f"_{results['bootstrap_method']}"
            filename = f"simulation_optimized_parameters_{mechanism}{strain_suffix}{bootstrap_suffix}.txt"
        else:
            filename = f"simulation_optimized_parameters_{mechanism}{strain_suffix}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"Simulation-based Optimization Results\n")
        f.write(f"Mechanism: {mechanism}\n")
        
        # Add strain selection information
        if selected_strains is not None:
            f.write(f"Selected Strains: {', '.join(selected_strains)}\n")
        else:
            f.write(f"Selected Strains: all datasets\n")
        
        # Add bootstrap information if available
        if 'bootstrap_method' in results:
            f.write(f"Bootstrap Method: {results['bootstrap_method']}\n")
            f.write(f"Target Sample Size: {results['target_sample_size']}\n")
            f.write(f"Number of Bootstrap Samples: {results['num_bootstrap_samples']}\n")
        
        f.write(f"Negative Log-Likelihood: {results['nll']:.6f}\n")
        f.write(f"Converged: {results.get('converged', 'Unknown')}\n")
        f.write(f"Status: {results.get('message', 'No message')}\n")
        f.write(f"Available Datasets: wildtype, threshold, degrate, degrateAPC\n\n")
        
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
    Main optimization routine for all strains.
    """
    max_iterations = 100  # number of iterations for testing
    num_simulations = 250  # Number of simulations per evaluation
    
    print("Simulation-based Optimization for Time-Varying Mechanisms")
    print("=" * 60)
    
    # Load experimental data
    datasets = load_experimental_data()
    if not datasets:
        print("Error: No datasets loaded!")
        return
    
    # Test mechanisms
    mechanisms = ['time_varying_k', 'time_varying_k_fixed_burst', 'time_varying_k_feedback_onion', 'time_varying_k_combined', 'time_varying_k_burst_onion']
    mechanism = mechanisms[3]  # Test the burst_onion mechanism
    
    print(f"\nOptimizing {mechanism} for ALL strains")
    
    try:
        # Run optimization for all strains
        results = run_optimization(
            mechanism, datasets, 
            max_iterations=max_iterations, 
            num_simulations=num_simulations,
            selected_strains=None,  # Use all strains
            use_parallel=True  # Enable parallel processing now that it's working
        )
        
        # Save results
        save_results(mechanism, results, selected_strains=None)
        
        print(f"\n{'-' * 60}")
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
     
    print("Optimization complete!")





def main_simple():
    """
    Simple main function for testing just one method.
    """
    max_iterations = 20  # number of iterations for testing
    num_simulations = 30  # Number of simulations per evaluation
    
    print("Simulation-based Optimization with Bootstrapping")
    print("=" * 60)
    
    # Load experimental data
    datasets = load_experimental_data()
    if not datasets:
        print("Error: No datasets loaded!")
        return
    
    # Test mechanisms
    mechanisms = ['time_varying_k', 'time_varying_k_fixed_burst', 'time_varying_k_feedback_onion', 'time_varying_k_combined', 'time_varying_k_burst_onion']
    mechanism = mechanisms[0]  # Test time_varying_k
    
    try:
        # Bootstrap optimization
        results = run_optimization_with_bootstrapping(
            mechanism, datasets, 
            max_iterations=max_iterations, 
            num_simulations=num_simulations,
            bootstrap_method='bootstrap',
            num_bootstrap_samples=100,
            random_seed=42
        )
        
        # Save results
        save_results(mechanism, results)
        
        print(f"\n{'-' * 60}")
        
    except Exception as e:
        print(f"Error optimizing {mechanism}: {e}")
        import traceback
        traceback.print_exc()
    
    print("Optimization complete!")


if __name__ == "__main__":
    # Choose which function to run:
    
    # Option 1: Run main() - standard optimization for all strains
    main() 
    
    # Option 2: Run main_simple() - simple bootstrapping test
    # Uncomment the line below for simple bootstrapping:
    # main_simple() 