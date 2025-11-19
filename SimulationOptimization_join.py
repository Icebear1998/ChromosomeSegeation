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

# Add SecondVersion to path for simple/fixed_burst mechanisms
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SecondVersion'))
from MultiMechanismSimulation import MultiMechanismSimulation
from simulation_kde import build_kde_from_simulations, calculate_kde_likelihood

import warnings
warnings.filterwarnings('ignore')


def run_simple_simulation_for_dataset(mechanism, params, n0_list, num_simulations=500):
    """
    Run simulations for simple/fixed_burst mechanisms using SecondVersion simulator.
    
    Args:
        mechanism: 'simple' or 'fixed_burst' or 'feedback_onion' or 'fixed_burst_feedback_onion'
        params: Dictionary with n1, n2, n3, N1, N2, N3, k, and optional burst_size/n_inner
        n0_list: List of threshold values [n01, n02, n03]
        num_simulations: Number of simulations to run
    
    Returns:
        tuple: (delta_t12_list, delta_t32_list) as numpy arrays
    """
    try:
        # Prepare rate parameters based on mechanism
        if mechanism == 'simple':
            rate_params = {'k': params['k']}
        elif mechanism == 'fixed_burst':
            rate_params = {'k': params['k'], 'burst_size': params['burst_size']}
        elif mechanism == 'feedback_onion':
            rate_params = {'k': params['k'], 'n_inner': params['n_inner']}
        elif mechanism == 'fixed_burst_feedback_onion':
            rate_params = {'k': params['k'], 'burst_size': params['burst_size'], 'n_inner': params['n_inner']}
        else:
            print(f"Unknown mechanism: {mechanism}")
            return None, None
        
        # Initial states
        initial_state = [params['N1'], params['N2'], params['N3']]
        
        # Run simulations
        delta_t12_list = []
        delta_t32_list = []
        
        for _ in range(num_simulations):
            sim = MultiMechanismSimulation(
                mechanism=mechanism,
                initial_state_list=initial_state,
                rate_params=rate_params,
                n0_list=n0_list,
                max_time=1000.0
            )
            
            _, _, sep_times = sim.simulate()
            
            delta_t12 = sep_times[0] - sep_times[1]  # T1 - T2
            delta_t32 = sep_times[2] - sep_times[1]  # T3 - T2
            
            delta_t12_list.append(delta_t12)
            delta_t32_list.append(delta_t32)
        
        return np.array(delta_t12_list), np.array(delta_t32_list)
    
    except Exception as e:
        print(f"Simulation error in {mechanism}: {e}")
        return None, None


def calculate_likelihood_kde(exp_data, sim_data):
    """
    Calculate negative log-likelihood using KDE (no bandwidth specification).
    
    Args:
        exp_data: Dictionary or array of experimental data
        sim_data: Dictionary or array of simulation data
        
    Returns:
        float: Negative log-likelihood
    """
    try:
        # Handle dictionary input
        if isinstance(exp_data, dict) and isinstance(sim_data, dict):
            nll_total = 0
            
            for key in ['delta_t12', 'delta_t32']:
                if key in exp_data and key in sim_data:
                    exp_values = np.asarray(exp_data[key]).flatten()
                    sim_values = np.asarray(sim_data[key]).flatten()
                    
                    # Remove non-finite values
                    exp_values = exp_values[np.isfinite(exp_values)]
                    sim_values = sim_values[np.isfinite(sim_values)]
                    
                    if len(exp_values) == 0 or len(sim_values) < 10:
                        return 1e6
                    
                    # Build KDE using Scott's rule (no manual bandwidth)
                    kde = build_kde_from_simulations(sim_values, bandwidth=None)
                    
                    # Calculate likelihood
                    nll = calculate_kde_likelihood(kde, exp_values)
                    nll_total += nll
            
            return nll_total
        
        # Handle array input
        else:
            exp_values = np.asarray(exp_data).flatten()
            sim_values = np.asarray(sim_data).flatten()
            
            exp_values = exp_values[np.isfinite(exp_values)]
            sim_values = sim_values[np.isfinite(sim_values)]
            
            if len(exp_values) == 0 or len(sim_values) < 10:
                return 1e6
            
            kde = build_kde_from_simulations(sim_values, bandwidth=None)
            return calculate_kde_likelihood(kde, exp_values)
    
    except Exception as e:
        print(f"Likelihood calculation error: {e}")
        return 1e6


# Bootstrapping functions removed - now using direct KDE approach

def joint_objective_simple_mechanisms(params_vector, mechanism, datasets, num_simulations=500):
    """
    Joint objective function for simple and fixed_burst mechanisms using KDE.
    
    Args:
        params_vector: Parameter vector [n2, N2, k, r21, r23, R21, R23, burst_size (optional), alpha, beta_k]
        mechanism: 'simple' or 'fixed_burst' or 'feedback_onion' or 'fixed_burst_feedback_onion'
        datasets: Experimental data dictionary
        num_simulations: Number of simulations per evaluation
        
    Returns:
        float: Total negative log-likelihood
    """
    try:
        # Unpack parameters based on mechanism
        if mechanism == 'simple':
            n2, N2, k, r21, r23, R21, R23, alpha, beta_k = params_vector
            base_params = {
                'n1': max(r21 * n2, 1), 'n2': n2, 'n3': max(r23 * n2, 1),
                'N1': max(R21 * N2, 1), 'N2': N2, 'N3': max(R23 * N2, 1),
                'k': k
            }
        elif mechanism == 'fixed_burst':
            n2, N2, k, r21, r23, R21, R23, burst_size, alpha, beta_k = params_vector
            base_params = {
                'n1': max(r21 * n2, 1), 'n2': n2, 'n3': max(r23 * n2, 1),
                'N1': max(R21 * N2, 1), 'N2': N2, 'N3': max(R23 * N2, 1),
                'k': k, 'burst_size': burst_size
            }
        elif mechanism == 'feedback_onion':
            n2, N2, k, r21, r23, R21, R23, n_inner, alpha, beta_k = params_vector
            base_params = {
                'n1': max(r21 * n2, 1), 'n2': n2, 'n3': max(r23 * n2, 1),
                'N1': max(R21 * N2, 1), 'N2': N2, 'N3': max(R23 * N2, 1),
                'k': k, 'n_inner': n_inner
            }
        elif mechanism == 'fixed_burst_feedback_onion':
            n2, N2, k, r21, r23, R21, R23, burst_size, n_inner, alpha, beta_k = params_vector
            base_params = {
                'n1': max(r21 * n2, 1), 'n2': n2, 'n3': max(r23 * n2, 1),
                'N1': max(R21 * N2, 1), 'N2': N2, 'N3': max(R23 * N2, 1),
                'k': k, 'burst_size': burst_size, 'n_inner': n_inner
            }
        else:
            return 1e6
        
        # Check constraints
        if base_params['n1'] >= base_params['N1'] or \
           base_params['n2'] >= base_params['N2'] or \
           base_params['n3'] >= base_params['N3']:
            return 1e6
        
        total_nll = 0
        
        # Loop over all datasets
        for dataset_name, data_dict in datasets.items():
            # Apply mutant modifications (simple mechanisms use different mutant logic)
            params = base_params.copy()
            n0_list = [base_params['n1'], base_params['n2'], base_params['n3']]
            
            if dataset_name == 'wildtype':
                pass  # No modifications
            elif dataset_name == 'threshold':
                # Reduce thresholds
                n0_list = [alpha * base_params['n1'], alpha * base_params['n2'], alpha * base_params['n3']]
            elif dataset_name == 'degrade':
                # Reduce degradation rate
                params['k'] = beta_k * base_params['k']
            elif dataset_name in ['degradeAPC', 'velcade']:
                # These mutants affect tau, not applicable to simple mechanisms
                # For simple mechanisms, treat them like degrade mutant
                params['k'] = beta_k * base_params['k']
            
            # Run simulations
            sim_delta_t12, sim_delta_t32 = run_simple_simulation_for_dataset(
                mechanism, params, n0_list, num_simulations
            )
            
            if sim_delta_t12 is None or sim_delta_t32 is None:
                return 1e6
            
            # Calculate likelihood using KDE
            exp_data = {'delta_t12': data_dict['delta_t12'], 'delta_t32': data_dict['delta_t32']}
            sim_data = {'delta_t12': sim_delta_t12, 'delta_t32': sim_delta_t32}
            
            nll = calculate_likelihood_kde(exp_data, sim_data)
            
            if nll >= 1e6:
                return 1e6
            
            total_nll += nll
        
        return total_nll
    
    except Exception as e:
        print(f"Objective function error: {e}")
        return 1e6


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
            n2, N2, k_max, tau, r21, r23, R21, R23, alpha, beta_k, beta_tau, beta_tau2 = params_vector
            k_1 = k_max / tau
            base_params = {
                'n1': max(r21 * n2, 1), 'n2': n2, 'n3': max(r23 * n2, 1),
                'N1': max(R21 * N2, 1), 'N2': N2, 'N3': max(R23 * N2, 1),
                'k_1': k_1, 'k_max': k_max, 'tau': tau
            }
        elif mechanism == 'time_varying_k_fixed_burst':
            n2, N2, k_max, tau, r21, r23, R21, R23, burst_size, alpha, beta_k, beta_tau, beta_tau2 = params_vector
            k_1 = k_max / tau
            base_params = {
                'n1': max(r21 * n2, 1), 'n2': n2, 'n3': max(r23 * n2, 1),
                'N1': max(R21 * N2, 1), 'N2': N2, 'N3': max(R23 * N2, 1),
                'k_1': k_1, 'k_max': k_max, 'tau': tau, 'burst_size': burst_size
            }
        elif mechanism == 'time_varying_k_feedback_onion':
            n2, N2, k_max, tau, r21, r23, R21, R23, n_inner, alpha, beta_k, beta_tau, beta_tau2 = params_vector
            k_1 = k_max / tau
            base_params = {
                'n1': max(r21 * n2, 1), 'n2': n2, 'n3': max(r23 * n2, 1),
                'N1': max(R21 * N2, 1), 'N2': N2, 'N3': max(R23 * N2, 1),
                'k_1': k_1, 'k_max': k_max, 'tau': tau, 'n_inner': n_inner
            }
        elif mechanism == 'time_varying_k_combined':
            n2, N2, k_max, tau, r21, r23, R21, R23, burst_size, n_inner, alpha, beta_k, beta_tau, beta_tau2 = params_vector
            k_1 = k_max / tau
            base_params = {
                'n1': max(r21 * n2, 1), 'n2': n2, 'n3': max(r23 * n2, 1),
                'N1': max(R21 * N2, 1), 'N2': N2, 'N3': max(R23 * N2, 1),
                'k_1': k_1, 'k_max': k_max, 'tau': tau, 'burst_size': burst_size, 'n_inner': n_inner
            }
        elif mechanism == 'time_varying_k_burst_onion':
            n2, N2, k_max, tau, r21, r23, R21, R23, burst_size, alpha, beta_k, beta_tau, beta_tau2 = params_vector
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
                base_params, dataset_name, alpha, beta_k, beta_tau, beta_tau2
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


def get_parameter_bounds_simple_mechanisms(mechanism):
    """
    Get parameter bounds for simple/fixed_burst mechanisms.
    
    Args:
        mechanism: 'simple', 'fixed_burst', 'feedback_onion', or 'fixed_burst_feedback_onion'
        
    Returns:
        list: List of (min, max) tuples for each parameter
    """
    # Base bounds: [n2, N2, k, r21, r23, R21, R23]
    bounds = [
        (1.0, 50.0),       # n2
        (50.0, 1000.0),    # N2
        (0.01, 0.2),       # k (degradation rate)
        (0.25, 4.0),       # r21
        (0.25, 4.0),       # r23
        (0.4, 2.5),        # R21
        (0.5, 5.0),        # R23
    ]
    
    # Add mechanism-specific bounds
    if mechanism == 'fixed_burst':
        bounds.append((1.0, 50.0))    # burst_size
    elif mechanism == 'feedback_onion':
        bounds.append((1.0, 100.0))   # n_inner
    elif mechanism == 'fixed_burst_feedback_onion':
        bounds.append((1.0, 50.0))    # burst_size
        bounds.append((1.0, 100.0))   # n_inner
    
    # Add mutant parameter bounds (only alpha and beta_k for simple mechanisms)
    bounds.extend([
        (0.1, 0.7),        # alpha (threshold reduction)
        (0.1, 1.0),        # beta_k (rate reduction)
    ])
    
    return bounds


def run_optimization_simple_mechanisms(mechanism, datasets, max_iterations=300, num_simulations=500):
    """
    Run optimization for simple/fixed_burst mechanisms using KDE.
    
    Args:
        mechanism: 'simple', 'fixed_burst', 'feedback_onion', or 'fixed_burst_feedback_onion'
        datasets: Experimental data dictionary
        max_iterations: Maximum iterations for optimization
        num_simulations: Number of simulations per evaluation
        
    Returns:
        dict: Optimization results
    """
    print(f"\n=== Simulation-based Optimization for {mechanism.upper()} ===")
    print(f"Using KDE with Scott's rule (automatic bandwidth)")
    print(f"Datasets: {list(datasets.keys())}")
    print(f"Max iterations: {max_iterations}")
    print(f"Simulations per evaluation: {num_simulations}")
    sys.stdout.flush()
    
    # Get parameter bounds
    bounds = get_parameter_bounds_simple_mechanisms(mechanism)
    
    print(f"\nOptimizing {len(bounds)} parameters...")
    sys.stdout.flush()
    
    if mechanism == 'simple':
        param_names = ['n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23', 'alpha', 'beta_k']
    elif mechanism == 'fixed_burst':
        param_names = ['n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'alpha', 'beta_k']
    elif mechanism == 'feedback_onion':
        param_names = ['n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23', 'n_inner', 'alpha', 'beta_k']
    elif mechanism == 'fixed_burst_feedback_onion':
        param_names = ['n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'n_inner', 'alpha', 'beta_k']
    
    print("\nParameter bounds:")
    for name, bound in zip(param_names, bounds):
        print(f"  {name}: {bound}")
    sys.stdout.flush()
    
    print("\n🚀 Starting differential evolution optimization...")
    sys.stdout.flush()
    
    # Run optimization
    result = differential_evolution(
        joint_objective_simple_mechanisms,
        bounds,
        args=(mechanism, datasets, num_simulations),
        maxiter=max_iterations,
        popsize=15,           # Keep small: 15 * ~9 params = ~135 simulations per generation
        strategy='rand1bin',  # Robust against simulation noise (prevents false convergence)
        mutation=(0.5, 1.0),  # Dithering helps escape local minima
        recombination=0.9,    # High recombination for correlated parameters (N and k)
        seed=42,
        disp=True,
        workers=-1,           # CRITICAL: Use all CPU cores
        polish=True,          # Use L-BFGS-B for final cleanup
        tol=0.1,              # LOOSE TOLERANCE: You cannot get 1e-6 precision on noisy data
        atol=1e-3             # Absolute tolerance should also be loose
    )
    sys.stdout.flush()
    
    # Display results
    convergence_status = "converged" if result.success else "did not converge"
    print(f"\n🔍 Optimization {convergence_status}!")
    print(f"Best negative log-likelihood: {result.fun:.4f}")
    sys.stdout.flush()
    
    if not result.success:
        print(f"Note: {result.message}")
        sys.stdout.flush()
    
    # Unpack parameters
    params = result.x
    param_dict = dict(zip(param_names, params))
    
    # Calculate derived parameters
    n1_derived = max(param_dict['r21'] * param_dict['n2'], 1)
    n3_derived = max(param_dict['r23'] * param_dict['n2'], 1)
    N1_derived = max(param_dict['R21'] * param_dict['N2'], 1)
    N3_derived = max(param_dict['R23'] * param_dict['N2'], 1)
    
    print("\nBest Parameters Found:")
    print(f"  Base: n2={param_dict['n2']:.1f}, N2={param_dict['N2']:.1f}")
    print(f"  Ratios: r21={param_dict['r21']:.2f}, r23={param_dict['r23']:.2f}, R21={param_dict['R21']:.2f}, R23={param_dict['R23']:.2f}")
    print(f"  Derived: n1={n1_derived:.1f}, n3={n3_derived:.1f}, N1={N1_derived:.1f}, N3={N3_derived:.1f}")
    print(f"  Rate: k={param_dict['k']:.4f}")
    
    if 'burst_size' in param_dict:
        print(f"  Burst size: {param_dict['burst_size']:.1f}")
    if 'n_inner' in param_dict:
        print(f"  Inner threshold: {param_dict['n_inner']:.1f}")
    
    print(f"  Mutants: alpha={param_dict['alpha']:.3f}, beta_k={param_dict['beta_k']:.3f}")
    sys.stdout.flush()
    
    return {
        'success': True,
        'converged': result.success,
        'params': param_dict,
        'nll': result.fun,
        'result': result,
        'message': result.message if not result.success else "Converged successfully"
    }


def run_optimization(mechanism, datasets, max_iterations=500, num_simulations=500, selected_strains=None, use_parallel=False, initial_guess=None):
    """
    Run joint optimization for selected datasets.
    
    Args:
        mechanism (str): Mechanism name
        datasets (dict): Experimental datasets
        max_iterations (int): Maximum iterations for optimization
        num_simulations (int): Number of simulations per evaluation
        selected_strains (list): List of strain names to include in fitting
        use_parallel (bool): Whether to use parallel processing
        initial_guess (list, optional): Initial guess parameter vector. If None, uses random initialization.
                                       Example: [n2, N2, k_max, tau, r21, r23, R21, R23, alpha, beta_k, beta_tau, beta_tau2]
    
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
        param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2']
        for i, (name, bound) in enumerate(zip(param_names, bounds)):
            print(f"  {name}: {bound}")
    elif mechanism == 'time_varying_k_combined':
        param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'n_inner', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2']
        for i, (name, bound) in enumerate(zip(param_names, bounds)):
            print(f"  {name}: {bound}")
    
    # Handle initial guess
    if initial_guess is not None:
        print(f"\n🎯 Using provided initial guess with {len(initial_guess)} parameters")
        print(f"   Initial guess: {initial_guess}")
        
        # Validate initial guess is within bounds
        if len(initial_guess) != len(bounds):
            print(f"❌ Error: Initial guess has {len(initial_guess)} parameters, but mechanism needs {len(bounds)}")
            print("   Falling back to random initialization")
            initial_guess = None
        else:
            valid_guess = True
            for i, (guess_val, bound) in enumerate(zip(initial_guess, bounds)):
                if not (bound[0] <= guess_val <= bound[1]):
                    print(f"⚠️  Initial guess parameter {i} ({guess_val:.4f}) is outside bounds {bound}")
                    valid_guess = False
            
            if not valid_guess:
                print("❌ Initial guess contains out-of-bounds values, falling back to random initialization")
                initial_guess = None
            else:
                print("✅ Initial guess validated")
                # Test initial guess
                test_nll = joint_objective(initial_guess, mechanism, datasets, num_simulations=10, selected_strains=selected_strains)
                print(f"   Initial guess NLL: {test_nll:.2f}")
    
    # Quick test to ensure optimization will work (if no initial guess provided)
    if initial_guess is None:
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
    
    # Global optimization
    result = differential_evolution(
        joint_objective,
        bounds,
        args=(mechanism, datasets, num_simulations, selected_strains),
        x0=initial_guess,  # Add initial guess parameter
        maxiter=max_iterations,
        popsize=12,           # DRASTIC REDUCTION from 200. 
        disp=True,
        workers=-1,           # Force parallel processing
        strategy='rand1bin',  # Safer than best1bin for stochastic functions
        mutation=(0.5, 1.0),  # High mutation to explore the landscape
        recombination=0.9,    # Correlated parameters (k_max and tau) need this
        tol=0.1,              # High tolerance for noisy objective
        polish=True,
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
        param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2']
    elif mechanism == 'time_varying_k_fixed_burst':
        param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2']
    elif mechanism == 'time_varying_k_feedback_onion':
        param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'n_inner', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2']
    elif mechanism == 'time_varying_k_combined':
        param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'n_inner', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2']
    elif mechanism == 'time_varying_k_burst_onion':
        param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2']
    
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
    
    print(f"  Mutants: alpha={param_dict['alpha']:.3f}, beta_k={param_dict['beta_k']:.3f}, beta_tau={param_dict['beta_tau']:.3f}, beta_tau2={param_dict['beta_tau2']:.3f}")
    
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
    num_simulations = 300  # Number of simulations per evaluation
    
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
        # Determine which optimization function to use
        if mechanism in ['simple', 'fixed_burst', 'feedback_onion', 'fixed_burst_feedback_onion']:
            # Use simple mechanism optimization with KDE
            print(f"\n🔬 Using simple mechanism optimization path...")
            sys.stdout.flush()
            results = run_optimization_simple_mechanisms(
                mechanism, datasets,
                max_iterations=max_iterations,
                num_simulations=num_simulations
            )
        else:
            # Use time-varying mechanism optimization
            print(f"\n🔬 Using time-varying mechanism optimization path...")
            sys.stdout.flush()
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