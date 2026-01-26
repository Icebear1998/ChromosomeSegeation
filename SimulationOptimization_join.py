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
import sys
import time
from scipy.optimize import differential_evolution
from simulation_utils import *
from Chromosomes_Theory import *

import warnings
warnings.filterwarnings('ignore')



MECHANISM_PARAM_NAMES = {
    'simple': ['n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23', 'alpha', 'beta_k1', 'beta_k2', 'beta_k3'],
    'fixed_burst': ['n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'alpha', 'beta_k1', 'beta_k2', 'beta_k3'],
    'feedback_onion': ['n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23', 'n_inner', 'alpha', 'beta_k1', 'beta_k2', 'beta_k3'],
    'fixed_burst_feedback_onion': [
        'n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'n_inner', 'alpha', 'beta_k1', 'beta_k2', 'beta_k3'
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
    beta_k1 = params.get('beta_k1')
    beta_k2 = params.get('beta_k2')
    beta_k3 = params.get('beta_k3')
    beta_tau = params.get('beta_tau')
    beta_tau2 = params.get('beta_tau2')
    
    return base_params, alpha, beta_k, beta_k1, beta_k2, beta_k3, beta_tau, beta_tau2


def joint_objective(params_vector, mechanism, datasets, num_simulations=500, selected_strains=None, return_breakdown=False, objective_metric='emd'):
    """
    Joint objective function that handles both simple and time-varying mechanisms.
    Supports both NLL (Negative Log-Likelihood) and EMD (Earth Mover's Distance).
    
    Args:
        params_vector: Parameter vector (mechanism-specific)
        mechanism: Mechanism name
        datasets: Experimental data dictionary
        num_simulations: Number of simulations per evaluation
        selected_strains: Optional list of strain names to include in fitting
        return_breakdown: If True, return dict with per-dataset scores instead of total
        objective_metric: 'nll' or 'emd' (default: 'nll')
        
    Returns:
        float: Total score (NLL or EMD)
        dict: {'total': float, 'per_dataset': dict} (if return_breakdown=True)
    """
    try:
        # Unpack parameters based on mechanism
        base_params, alpha, beta_k, beta_k1, beta_k2, beta_k3, beta_tau, beta_tau2 = unpack_mechanism_params(params_vector, mechanism)
        
        # Check constraints
        if base_params['n1'] >= base_params['N1'] or \
            base_params['n2'] >= base_params['N2'] or \
            base_params['n3'] >= base_params['N3']:
            if return_breakdown:
                return {'total': 1e6, 'per_dataset': {}}
            return 1e6

        total_score = 0
        per_dataset_score = {}
        
        # Loop over all datasets
        for dataset_name, data_dict in datasets.items():
            # Apply mutant modifications using unified helper function (works for both mechanism types)
            params, n0_list = apply_mutant_params(
                base_params, dataset_name, alpha, beta_k, beta_k1, beta_k2, beta_k3, beta_tau, beta_tau2
            )
            
            # Run simulations
            sim_delta_t12, sim_delta_t32 = run_simulation_for_dataset(
                mechanism, params, n0_list, num_simulations
            )
            
            if sim_delta_t12 is None or sim_delta_t32 is None:
                if return_breakdown:
                    return {'total': 1e6, 'per_dataset': {}}
                return 1e6
            
            # Calculate objective metric
            exp_data = {'delta_t12': data_dict['delta_t12'], 'delta_t32': data_dict['delta_t32']}
            sim_data = {'delta_t12': sim_delta_t12, 'delta_t32': sim_delta_t32}
            
            if objective_metric == 'emd':
                 score = calculate_emd(exp_data, sim_data)
                 # No log transform for EMD, it's already a distance
            else:
                 # Default to NLL
                 score = calculate_likelihood(exp_data, sim_data)
            
            # Large penalty check
            if score >= 1e6:
                if return_breakdown:
                    return {'total': 1e6, 'per_dataset': {}}
                return 1e6
            
            per_dataset_score[dataset_name] = score
            total_score += score
        
        if return_breakdown:
            return {'total': total_score, 'per_dataset': per_dataset_score}
        return total_score
    
    except Exception:
        if return_breakdown:
            return {'total': 1e6, 'per_dataset': {}}
        return 1e6


def get_per_dataset_nll(params_vector, mechanism, datasets, num_simulations=500, selected_strains=None):
    """
    Calculate per-dataset NLL breakdown for given parameters.
    
    Args:
        params_vector: Parameter vector
        mechanism: Mechanism name
        datasets: Experimental datasets
        num_simulations: Number of simulations per evaluation
        selected_strains: Optional list of strain names
        
    Returns:
        dict: {'total': float, 'per_dataset': dict} with per-dataset NLLs
    """
    return joint_objective(params_vector, mechanism, datasets, num_simulations, selected_strains, return_breakdown=True)


def run_optimization(mechanism, datasets, max_iterations=500, num_simulations=500, selected_strains=None, objective_metric='emd'):
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
        objective_metric (str): 'nll' or 'emd' (default: 'nll')
        
    Returns:
        dict: Optimization results
    """
    try:
        param_names = MECHANISM_PARAM_NAMES[mechanism]
    except KeyError:
        return {'success': False, 'message': f'Unknown mechanism: {mechanism}'}
    
    bounds = get_parameter_bounds(mechanism)
    
    metric_label = "EMD" if objective_metric == 'emd' else "NLL"
    print(f"\nOptimizing {mechanism} using {metric_label} ({len(bounds)} parameters, {num_simulations} sims/eval)")
    
    # Define Differential Evolution settings
    de_settings = {
        'maxiter': max_iterations,
        'popsize': 10,
        'workers': -1,
        'strategy': 'best1bin',
        #'mutation': (0.7, 1.0),
        #'recombination': 0.7,
        'tol': 0.01,
        #'atol': 1e-2,
        'disp': True
    }

    print("\nDifferential Evolution Settings:")
    for key, value in de_settings.items():
        print(f"  {key}: {value}")
    sys.stdout.flush()
    
    # Now passing metric to objective
    opt_args = (mechanism, datasets, num_simulations, selected_strains, False, objective_metric)
    
    result = differential_evolution(
        joint_objective,
        bounds,
        args=opt_args,
        #x0=initial_guess,
        **de_settings
    )
    
    params = result.x
    param_dict = dict(zip(param_names, params))
    
    status = "converged" if result.success else "not converged"
    print(f"Optimization {status}: {metric_label} = {result.fun:.4f}")
    
    print("\nBest Fit Parameters:")
    for name, val in param_dict.items():
        print(f"  {name}: {val:.6f}")
    
    # Get per-dataset score breakdown at final parameters
    score_breakdown = joint_objective(params, mechanism, datasets, num_simulations, selected_strains, return_breakdown=True, objective_metric=objective_metric)
    per_dataset_score = score_breakdown.get('per_dataset', {})
    
    if per_dataset_score:
        print(f"\nPer-dataset {metric_label} breakdown:")
        for dataset_name, score_val in per_dataset_score.items():
            print(f"  {dataset_name}: {score_val:.4f}")
    
    print(f"\nOptimization converged: {metric_label} = {result.fun:.4f}")

    
    return {
        'success': True,  # Always treat as success to save results
        'converged': result.success,  # Track actual convergence status
        'params': param_dict,
        'nll': result.fun,
        'per_dataset_score': per_dataset_score,  # Add per-dataset breakdown
        'result': result,
        'message': result.message if not result.success else "Converged successfully"
    }


def save_results(mechanism, results, filename=None, selected_strains=None, objective_metric='emd'):
    """
    Save optimization results to file.
    
    Args:
        mechanism (str): Mechanism name
        results (dict): Optimization results
        filename (str): Output filename (optional)
        selected_strains (list): List of strains used in fitting
        objective_metric (str): 'nll' or 'emd'
    """
    if not results['success']:
        return
    
    # Determine filename based on selected strains
    if filename is None:
        # Create strain suffix if specific strains were selected
        strain_suffix = ""
        if selected_strains is not None:
            strain_suffix = f"_{'_'.join(selected_strains)}"
        
        filename = f"simulation_optimized_parameters_{mechanism}{strain_suffix}.txt"
    
    metric_label = "Earth Mover's Distance" if objective_metric == 'emd' else "Negative Log-Likelihood"
    val_key = 'nll' if 'nll' in results else 'emd' # Fallback check
    if 'emd' in results: val_key = 'emd' # Explicit check if added later
    
    # Use generic 'score' or fun from result object if available
    metric_value = results.get('nll', results['result'].fun)
    
    with open(filename, 'w') as f:
        f.write(f"Simulation-based Optimization Results ({metric_label})\n")
        f.write(f"Mechanism: {mechanism}\n")
        
        # Add strain selection information
        if selected_strains is not None:
            f.write(f"Selected Strains: {', '.join(selected_strains)}\n")
        else:
            f.write(f"Selected Strains: all datasets\n")
        
        f.write(f"{metric_label}: {metric_value:.6f}\n")
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


def main():
    """
    Main optimization routine - now supports both simple and time-varying mechanisms.
    """
    # Mechanism to optimize
    mechanism = 'simple'  # Default mechanism
    
    # Objective Metric: 'nll' or 'emd'
    objective_metric = 'emd'      # Switch to EMD by default

    max_iterations = 1000
    num_simulations = 10000
    
    # KDE Bandwidth configuration
    # Options: 'scott' (adaptive, h = std * n^(-1/5)) or 'fixed' (constant bandwidth)
    bandwidth_method = 'fixed'     # 'scott' or 'fixed'
    fixed_bandwidth = 10.0         # Used when bandwidth_method='fixed'
    
    # Set bandwidth configuration
    set_kde_bandwidth(method=bandwidth_method, fixed_value=fixed_bandwidth)
    
    datasets = load_experimental_data()
    if not datasets:
        print("Error: No datasets loaded!")
        return

    
    try:
        results = run_optimization(
            mechanism, datasets,
            max_iterations=max_iterations,
            num_simulations=num_simulations,
            selected_strains=None,
            objective_metric=objective_metric
        )

        save_results(mechanism, results, selected_strains=None, objective_metric=objective_metric)
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 