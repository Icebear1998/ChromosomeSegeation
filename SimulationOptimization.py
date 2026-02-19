#!/usr/bin/env python3

import numpy as np
import sys
import time
from scipy.optimize import differential_evolution
from simulation_utils import *

import warnings
warnings.filterwarnings('ignore')





MECHANISM_PARAM_NAMES = {
    # Time-varying mechanisms (wide bounds)
    'time_varying_k': [
        'n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2'
    ],
    'time_varying_k_fixed_burst': [
        'n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23',
        'burst_size', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2'
    ],
    'time_varying_k_steric_hindrance': [
        'n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23',
        'n_inner', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2'
    ],
    'time_varying_k_combined': [
        'n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23',
        'burst_size', 'n_inner', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2'
    ],
    # Feedback variants (tight bounds for tau/beta)
    'time_varying_k_wfeedback': [
        'n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2'
    ],
    'time_varying_k_fixed_burst_wfeedback': [
        'n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23',
        'burst_size', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2'
    ],
    'time_varying_k_steric_hindrance_wfeedback': [
        'n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23',
        'n_inner', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2'
    ],
    'time_varying_k_combined_wfeedback': [
        'n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23',
        'burst_size', 'n_inner', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2'
    ],
}




def unpack_mechanism_params(params_vector, mechanism):
    """
    Unpack parameter vector for time-varying mechanisms.
    
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
    
    # All mechanisms are time-varying, so always expect k_max and tau
    k_max = params['k_max']
    tau = params['tau']
    base_params.update({
        'k_max': k_max,
        'tau': tau,
        'k_1': k_max / tau,
    })
    
    # Add optional parameters (burst_size, n_inner)
    for optional_key in ('burst_size', 'n_inner'):
        if optional_key in params:
            base_params[optional_key] = params[optional_key]
    
    alpha = params.get('alpha')
    beta_k = params.get('beta_k')
    beta_tau = params.get('beta_tau')
    beta_tau2 = params.get('beta_tau2')
    
    return base_params, alpha, beta_k, beta_tau, beta_tau2


def joint_objective(params_vector, mechanism, datasets, num_simulations=500, selected_strains=None, return_breakdown=False, objective_metric='emd'):
    """
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
        base_params, alpha, beta_k, beta_tau, beta_tau2 = unpack_mechanism_params(params_vector, mechanism)
        
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
            # Apply mutant modifications for time-varying mechanisms
            params, n0_list = apply_mutant_params(
                base_params, dataset_name, alpha, beta_k, beta_tau, beta_tau2
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


def run_optimization(mechanism, datasets, max_iterations=500, num_simulations=500, selected_strains=None, objective_metric='emd'):
    """
    Run joint optimization for time-varying mechanism types.
    
    Supports time-varying mechanisms ('time_varying_k' and variants).
    
    Args:
        mechanism (str): Mechanism name
        datasets (dict): Experimental datasets
        max_iterations (int): Maximum iterations for optimization
        num_simulations (int): Number of simulations per evaluation
        selected_strains (list): List of strain names to include in fitting (optional)
        objective_metric (str): 'nll' or 'emd' (default: 'nll')
        
    Returns:
        dict: Optimization results with comprehensive metadata
    """
    start_time = time.time()
    
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
        #'mutation': (0.5, 1.0), #default is (0.5, 1.0)
        #'recombination': 0.7, #default is 0.7
        'tol': 0.01,
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
    
    optimization_time = time.time() - start_time
    
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
        'nll': result.fun,  # Keep for backwards compatibility
        'score': result.fun,  # Generic score value
        'per_dataset_score': per_dataset_score,  # Per-dataset breakdown
        'result': result,
        'message': result.message if not result.success else "Converged successfully",
        # Optimization settings metadata
        'optimization_settings': {
            'method': 'Differential Evolution',
            'objective_metric': objective_metric,
            'num_simulations': num_simulations,
            'selected_strains': selected_strains if selected_strains else 'all',
            'de_settings': de_settings,
            'bounds': dict(zip(param_names, bounds)),
            'optimization_time_seconds': optimization_time,
            'n_iterations': result.nit if hasattr(result, 'nit') else 'unknown',
        }
    }


def save_results(mechanism, results, filename=None, selected_strains=None, objective_metric='emd'):
    """
    Save optimization results to file with comprehensive metadata.
    
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
    
    metric_label = "EMD" if objective_metric == 'emd' else "NLL"
    metric_full_name = "Earth Mover's Distance" if objective_metric == 'emd' else "Negative Log-Likelihood"
    
    # Extract settings
    opt_settings = results.get('optimization_settings', {})
    per_dataset_score = results.get('per_dataset_score', {})
    
    with open(filename, 'w') as f:
        # ===== HEADER =====
        f.write("="*80 + "\n")
        f.write(f"SIMULATION-BASED OPTIMIZATION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        # ===== OPTIMIZATION OVERVIEW =====
        f.write("OPTIMIZATION OVERVIEW\n")
        f.write("-"*80 + "\n")
        f.write(f"Mechanism: {mechanism}\n")
        f.write(f"Optimization Method: {opt_settings.get('method', 'Differential Evolution')}\n")
        f.write(f"Objective Metric: {metric_full_name} ({metric_label})\n")
        f.write(f"Converged: {results.get('converged', 'Unknown')}\n")
        f.write(f"Status: {results.get('message', 'No message')}\n")
        
        # Timing info
        opt_time = opt_settings.get('optimization_time_seconds', 0)
        if opt_time > 0:
            hours = int(opt_time // 3600)
            minutes = int((opt_time % 3600) // 60)
            seconds = opt_time % 60
            f.write(f"Optimization Time: {hours:02d}:{minutes:02d}:{seconds:06.3f} ({opt_time:.2f} seconds)\n")
        
        n_iters = opt_settings.get('n_iterations', 'unknown')
        f.write(f"Iterations Used: {n_iters}\n")
        f.write("\n")
        
        # ===== OBJECTIVE SCORES =====
        f.write("OBJECTIVE SCORES\n")
        f.write("-"*80 + "\n")
        total_score = results.get('score', results.get('nll', 0))
        f.write(f"Total {metric_label}: {total_score:.6f}\n")
        f.write(f"\nPer-Dataset {metric_label} Breakdown:\n")
        
        if per_dataset_score:
            for dataset_name, score_val in sorted(per_dataset_score.items()):
                f.write(f"  {dataset_name}: {score_val:.4f}\n")
        else:
            f.write("  (No per-dataset breakdown available)\n")
        f.write("\n")
        
        # ===== OPTIMIZATION SETTINGS =====
        f.write("OPTIMIZATION SETTINGS\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of Simulations per Evaluation: {opt_settings.get('num_simulations', 'N/A')}\n")
        
        selected = opt_settings.get('selected_strains', 'all')
        if selected == 'all':
            f.write(f"Selected Strains: all datasets\n")
            f.write(f"Available Datasets: wildtype, threshold, degrade, degradeAPC, velcade\n")
        else:
            f.write(f"Selected Strains: {', '.join(selected) if isinstance(selected, list) else selected}\n")
        
        # DE Settings
        de_settings = opt_settings.get('de_settings', {})
        if de_settings:
            f.write(f"\nDifferential Evolution Settings:\n")
            for key, val in de_settings.items():
                f.write(f"  {key}: {val}\n")
        f.write("\n")
        
        # ===== PARAMETER BOUNDS =====
        f.write("PARAMETER BOUNDS\n")
        f.write("-"*80 + "\n")
        bounds_dict = opt_settings.get('bounds', {})
        if bounds_dict:
            for param_name, bound in bounds_dict.items():
                f.write(f"{param_name:15s}: [{bound[0]:12.6f}, {bound[1]:12.6f}]\n")
        else:
            f.write("(No bounds information available)\n")
        f.write("\n")
        
        # ===== OPTIMIZED PARAMETERS =====
        f.write("OPTIMIZED PARAMETERS\n")
        f.write("-"*80 + "\n")
        for param, value in results['params'].items():
            f.write(f"{param:15s} = {value:.6f}\n")
        
        # Calculate and save derived parameters
        if 'r21' in results['params']:
            n1_derived = max(results['params']['r21'] * results['params']['n2'], 1)
            n3_derived = max(results['params']['r23'] * results['params']['n2'], 1)
            N1_derived = max(results['params']['R21'] * results['params']['N2'], 1)
            N3_derived = max(results['params']['R23'] * results['params']['N2'], 1)
            
            f.write(f"\nDerived Parameters:\n")
            f.write(f"{'n1':15s} = {n1_derived:.6f}\n")
            f.write(f"{'n3':15s} = {n3_derived:.6f}\n")
            f.write(f"{'N1':15s} = {N1_derived:.6f}\n")
            f.write(f"{'N3':15s} = {N3_derived:.6f}\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
    
    print(f"\nResults saved to: {filename}")


def main():
    """
    Main optimization routine - now supports both simple and time-varying mechanisms.
    """
    # Mechanism to optimize
    mechanism = 'time_varying_k'  # Default mechanism

    max_iterations = 1000
    num_simulations = 10000
    
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
        )

        save_results(mechanism, results, selected_strains=None)
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 