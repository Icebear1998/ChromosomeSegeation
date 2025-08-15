#!/usr/bin/env python3
"""
Bayesian optimization for chromosome segregation timing models.
Uses scikit-optimize for efficient optimization of expensive simulation-based objective functions.

Key advantages:
- Sample efficient (fewer function evaluations needed)
- Handles noisy objective functions well
- Can use initial guesses from previous optimizations
- Automatic exploration vs exploitation balance
"""

import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
from SimulationOptimization_join import (
    load_experimental_data, joint_objective, get_parameter_bounds, 
    apply_mutant_params, save_results
)
import warnings
warnings.filterwarnings('ignore')

# Try to import scikit-optimize
try:
    from skopt import gp_minimize, forest_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei
    SKOPT_AVAILABLE = True
    print("scikit-optimize available - Bayesian optimization enabled")
except ImportError:
    SKOPT_AVAILABLE = False
    print("Warning: scikit-optimize not available. Install with: pip install scikit-optimize")


def get_default_initial_guess(mechanism):
    """
    Get default initial parameter guess.
    
    Args:
        mechanism (str): Mechanism name
    
    Returns:
        list: Default initial parameter values
    """
    # Common initial guesses based on biological knowledge
    initial_guess = [
        25.0,    # n2 - reasonable threshold
        300.0,   # N2 - reasonable initial count
        0.003,   # k_1 - reasonable initial rate
        0.05,    # k_max - reasonable maximum rate
        1.2,     # r21 - slightly higher than n2
        1.2,     # r23 - similar to n2
        1.2,     # R21 - slightly lower than N2
        3,     # R23 - higher than N2
    ]
    
    # Mechanism-specific parameters
    if mechanism == 'time_varying_k_fixed_burst':
        initial_guess.append(5.0)   # burst_size - moderate burst
    elif mechanism == 'time_varying_k_feedback_onion':
        initial_guess.append(25.0)  # n_inner - reasonable inner threshold
    elif mechanism == 'time_varying_k_combined':
        initial_guess.append(5.0)   # burst_size
        initial_guess.append(25.0)  # n_inner
    
    # Mutant parameters
    initial_guess.extend([
        0.7,     # alpha - moderate threshold reduction
        0.6,     # beta_k - moderate rate reduction
        0.5,     # beta2_k - moderate rate reduction
    ])
    
    return initial_guess


def run_bayesian_optimization(mechanism, datasets, n_calls=150, n_initial_points=20, num_simulations=500, 
                             acq_func='EI', random_state=42):
    """
    Run Bayesian optimization using Gaussian Process.
    
    Args:
        mechanism (str): Mechanism name
        datasets (dict): Experimental datasets
        n_calls (int): Total number of function evaluations
        n_initial_points (int): Number of initial random points
        num_simulations (int): Number of simulations per evaluation
        acq_func (str): Acquisition function ('EI', 'PI', 'LCB')
        random_state (int): Random seed
    
    Returns:
        dict: Optimization results
    """
    if not SKOPT_AVAILABLE:
        print("Error: scikit-optimize is required for Bayesian optimization")
        print("Please install it with: pip install scikit-optimize")
        return {'success': False, 'message': 'scikit-optimize not available'}
    
    print(f"\n=== Bayesian Optimization for {mechanism.upper()} ===")
    print(f"Datasets: {list(datasets.keys())}")
    print(f"Total function evaluations: {n_calls}")
    print(f"Initial random points: {n_initial_points}")
    print(f"Acquisition function: {acq_func}")
    
    # Get parameter bounds
    bounds = get_parameter_bounds(mechanism)
    
    # Convert bounds to skopt format
    space = []
    param_names = []
    if mechanism == 'time_varying_k':
        param_names = ['n2', 'N2', 'k_1', 'k_max', 'r21', 'r23', 'R21', 'R23', 'alpha', 'beta_k', 'beta2_k']
    elif mechanism == 'time_varying_k_fixed_burst':
        param_names = ['n2', 'N2', 'k_1', 'k_max', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'alpha', 'beta_k', 'beta2_k']
    elif mechanism == 'time_varying_k_feedback_onion':
        param_names = ['n2', 'N2', 'k_1', 'k_max', 'r21', 'r23', 'R21', 'R23', 'n_inner', 'alpha', 'beta_k', 'beta2_k']
    elif mechanism == 'time_varying_k_combined':
        param_names = ['n2', 'N2', 'k_1', 'k_max', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'n_inner', 'alpha', 'beta_k', 'beta2_k']
    
    for i, (low, high) in enumerate(bounds):
        space.append(Real(low, high, name=param_names[i]))
    
    # Use default initial guess
    initial_guess = get_default_initial_guess(mechanism)
    print("Using default initial guess")
    
    # Ensure initial guess is within bounds
    for i, (low, high) in enumerate(bounds):
        if initial_guess[i] < low:
            initial_guess[i] = low
        elif initial_guess[i] > high:
            initial_guess[i] = high
    
    print(f"Optimizing {len(bounds)} parameters...")
    print(f"Initial guess: {[f'{x:.4f}' for x in initial_guess]}")
    
    # Define objective function for skopt
    @use_named_args(space)
    def objective(**params):
        # Convert named parameters back to vector
        param_vector = [params[name] for name in param_names]
        return joint_objective(param_vector, mechanism, datasets, num_simulations)
    
    # Run Bayesian optimization
    try:
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            x0=[initial_guess],  # Start with our initial guess
            acq_func=acq_func,
            random_state=random_state,
            verbose=True,
            n_jobs=1  # Avoid multiprocessing issues
        )
        
        # Extract results
        convergence_status = "completed"
        print(f"🔍 Bayesian optimization {convergence_status}!")
        print(f"Best negative log-likelihood: {result.fun:.4f}")
        print(f"Found after {len(result.func_vals)} function evaluations")
        
        # Create parameter dictionary
        param_dict = dict(zip(param_names, result.x))
        
        # Calculate derived parameters for display
        n1_derived = max(param_dict['r21'] * param_dict['n2'], 1)
        n3_derived = max(param_dict['r23'] * param_dict['n2'], 1)
        N1_derived = max(param_dict['R21'] * param_dict['N2'], 1)
        N3_derived = max(param_dict['R23'] * param_dict['N2'], 1)
        
        print("\nBest Parameters Found:")
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
        
        print(f"  Mutants: alpha={param_dict['alpha']:.3f}, beta_k={param_dict['beta_k']:.3f}, beta2_k={param_dict['beta2_k']:.3f}")
        
        return {
            'success': True,
            'converged': True,
            'params': param_dict,
            'nll': result.fun,
            'result': result,
            'n_evaluations': len(result.func_vals),
            'message': f"Bayesian optimization completed successfully after {len(result.func_vals)} evaluations"
        }
        
    except Exception as e:
        print(f"Bayesian optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'message': f'Bayesian optimization failed: {str(e)}'}





def save_bayesian_results(mechanism, results, filename=None):
    """
    Save Bayesian optimization results to file.
    
    Args:
        mechanism (str): Mechanism name
        results (dict): Optimization results
        filename (str): Output filename (optional)
    """
    if filename is None:
        filename = f"bayesian_optimized_parameters_{mechanism}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"Bayesian Optimization Results\n")
        f.write(f"Mechanism: {mechanism}\n")
        f.write(f"Negative Log-Likelihood: {results['nll']:.6f}\n")
        f.write(f"Converged: {results.get('converged', 'Unknown')}\n")
        f.write(f"Status: {results.get('message', 'No message')}\n")
        f.write(f"Function Evaluations: {results.get('n_evaluations', 'Unknown')}\n")
        f.write(f"Datasets: wildtype, threshold, degrate, degrateAPC\n\n")
        
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
    Main Bayesian optimization routine.
    """
    # ========== CONFIGURATION - MODIFY THESE VALUES ==========
    num_simulations = 500    # Number of simulations per evaluation
                           # Recommended values:
                           # - Testing: 20-50 (faster)
                           # - Production: 100-500 (more accurate)
    
    n_calls = 100           # Total function evaluations for Bayesian optimization
                           # Recommended values:
                           # - Quick test: 50-100
                           # - Production: 150-300
    
    n_initial_points = 25   # Initial random points before Bayesian optimization starts
                           # Recommended: 15-30 (should be < n_calls/3)
    
    acq_func = 'LCB'        # Acquisition function: 'EI' (Expected Improvement), 'PI', 'LCB'
                           # 'EI' is usually best for most problems
    # ========================================================
    
    print("Bayesian Optimization for Time-Varying Mechanisms")
    print("=" * 60)
    
    # Load experimental data
    datasets = load_experimental_data()
    if not datasets:
        print("Error: No datasets loaded!")
        return
    
    # Test mechanisms
    mechanisms = ['time_varying_k', 'time_varying_k_fixed_burst', 'time_varying_k_feedback_onion', 'time_varying_k_combined']
    mechanism = mechanisms[3]  # Test combined mechanism
    
    try:
        results = run_bayesian_optimization(
            mechanism, datasets, n_calls=n_calls, n_initial_points=n_initial_points, 
            num_simulations=num_simulations, acq_func=acq_func
        )
        
        if results['success']:
            save_bayesian_results(mechanism, results)
        else:
            print("Bayesian optimization failed!")
        
        print(f"\n{'-' * 60}")
        
    except Exception as e:
        print(f"Error during Bayesian optimization: {e}")
        import traceback
        traceback.print_exc()
    
    print("Bayesian optimization complete!")


if __name__ == "__main__":
    main()
