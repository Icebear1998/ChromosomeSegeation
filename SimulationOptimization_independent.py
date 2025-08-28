#!/usr/bin/env python3
"""
Independent simulation-based optimization for chromosome segregation timing models.
Uses MultiMechanismSimulationTimevary for time-varying mechanisms.
Independent optimization strategy: optimizes each dataset separately.

Key differences from joint optimization:
- Optimizes wildtype parameters first using only wildtype data
- Then optimizes mutant parameters for each mutant dataset separately
- Uses fixed wildtype parameters when optimizing mutant parameters
- Allows comparison of parameter consistency across datasets
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from MultiMechanismSimulationTimevary import MultiMechanismSimulationTimevary
from simulation_utils import *
import warnings
warnings.filterwarnings('ignore')


def wildtype_objective(params_vector, mechanism, wildtype_data, num_simulations=500, selected_strains=None):
    """
    Objective function for wildtype dataset only.
    
    Args:
        params_vector (array): Wildtype parameter vector to optimize
        mechanism (str): Mechanism name
        wildtype_data (dict): Wildtype experimental data
        num_simulations (int): Number of simulations per evaluation
        selected_strains (list): List of strain names to include (for compatibility)
    
    Returns:
        float: Negative log-likelihood for wildtype data
    """
    try:
        # Unpack wildtype parameters based on mechanism - using tau = k_max/k_1
        if mechanism == 'time_varying_k':
            n2, N2, k_max, tau, r21, r23, R21, R23 = params_vector
            # Calculate k_1 from k_max and tau
            k_1 = k_max / tau
            # Calculate derived parameters from ratios
            n1 = max(r21 * n2, 1)
            n3 = max(r23 * n2, 1)
            N1 = max(R21 * N2, 1)
            N3 = max(R23 * N2, 1)
            
            # Add constraint checks
            if n1 >= N1 or n2 >= N2 or n3 >= N3:
                return 1e6
            
            base_params = {
                'n1': n1, 'n2': n2, 'n3': n3,
                'N1': N1, 'N2': N2, 'N3': N3,
                'k_1': k_1, 'k_max': k_max
            }
        elif mechanism == 'time_varying_k_fixed_burst':
            n2, N2, k_max, tau, r21, r23, R21, R23, burst_size = params_vector
            # Calculate k_1 from k_max and tau
            k_1 = k_max / tau
            # Calculate derived parameters from ratios
            n1 = max(r21 * n2, 1)
            n3 = max(r23 * n2, 1)
            N1 = max(R21 * N2, 1)
            N3 = max(R23 * N2, 1)
            
            if n1 >= N1 or n2 >= N2 or n3 >= N3:
                return 1e6
            
            base_params = {
                'n1': n1, 'n2': n2, 'n3': n3,
                'N1': N1, 'N2': N2, 'N3': N3,
                'k_1': k_1, 'k_max': k_max, 'burst_size': burst_size
            }
        elif mechanism == 'time_varying_k_feedback_onion':
            n2, N2, k_max, tau, r21, r23, R21, R23, n_inner = params_vector
            # Calculate k_1 from k_max and tau
            k_1 = k_max / tau
            # Calculate derived parameters from ratios
            n1 = max(r21 * n2, 1)
            n3 = max(r23 * n2, 1)
            N1 = max(R21 * N2, 1)
            N3 = max(R23 * N2, 1)
            
            if n1 >= N1 or n2 >= N2 or n3 >= N3:
                return 1e6
            
            base_params = {
                'n1': n1, 'n2': n2, 'n3': n3,
                'N1': N1, 'N2': N2, 'N3': N3,
                'k_1': k_1, 'k_max': k_max, 'n_inner': n_inner
            }
        elif mechanism == 'time_varying_k_combined':
            n2, N2, k_max, tau, r21, r23, R21, R23, burst_size, n_inner = params_vector
            # Calculate k_1 from k_max and tau
            k_1 = k_max / tau
            # Calculate derived parameters from ratios
            n1 = max(r21 * n2, 1)
            n3 = max(r23 * n2, 1)
            N1 = max(R21 * N2, 1)
            N3 = max(R23 * N2, 1)
            
            if n1 >= N1 or n2 >= N2 or n3 >= N3:
                return 1e6
            
            base_params = {
                'n1': n1, 'n2': n2, 'n3': n3,
                'N1': N1, 'N2': N2, 'N3': N3,
                'k_1': k_1, 'k_max': k_max, 'burst_size': burst_size, 'n_inner': n_inner
            }
        elif mechanism == 'time_varying_k_burst_onion':
            n2, N2, k_max, tau, r21, r23, R21, R23, burst_size = params_vector
            # Calculate k_1 from k_max and tau
            k_1 = k_max / tau
            # Calculate derived parameters from ratios
            n1 = max(r21 * n2, 1)
            n3 = max(r23 * n2, 1)
            N1 = max(R21 * N2, 1)
            N3 = max(R23 * N2, 1)
            
            if n1 >= N1 or n2 >= N2 or n3 >= N3:
                return 1e6
            
            base_params = {
                'n1': n1, 'n2': n2, 'n3': n3,
                'N1': N1, 'N2': N2, 'N3': N3,
                'k_1': k_1, 'k_max': k_max, 'burst_size': burst_size
            }
        else:
            return 1e6
        
        # No mutant modifications for wildtype
        params, n0_list = apply_mutant_params(base_params, 'wildtype', 1.0, 1.0, 1.0)
        
        # Run simulations
        sim_delta_t12, sim_delta_t32 = run_simulation_for_dataset(
            mechanism, params, n0_list, num_simulations
        )
        
        if sim_delta_t12 is None or sim_delta_t32 is None:
            return 1e6
        
        # Extract experimental data
        exp_delta_t12 = wildtype_data['delta_t12']
        exp_delta_t32 = wildtype_data['delta_t32']
        
        # Calculate likelihoods
        nll_12 = calculate_likelihood(exp_delta_t12, sim_delta_t12)
        nll_32 = calculate_likelihood(exp_delta_t32, sim_delta_t32)
        
        if nll_12 >= 1e6 or nll_32 >= 1e6:
            return 1e6
        
        return nll_12 + nll_32
    
    except Exception as e:
        return 1e6


def mutant_objective(mutant_params, mechanism, mutant_data, wildtype_params, mutant_type, num_simulations=500, selected_strains=None):
    """
    Objective function for a single mutant dataset with fixed wildtype parameters.
    
    Args:
        mutant_params (array): Mutant parameter vector to optimize
        mechanism (str): Mechanism name
        mutant_data (dict): Mutant experimental data
        wildtype_params (dict): Fixed wildtype parameters
        mutant_type (str): Type of mutant
        num_simulations (int): Number of simulations per evaluation
        selected_strains (list): List of strain names to include (for compatibility)
    
    Returns:
        float: Negative log-likelihood for mutant data
    """
    try:
        # Unpack mutant-specific parameters
        if mutant_type == 'threshold':
            alpha = mutant_params[0]
            beta_k, beta_tau = 1.0, 1.0  # Not used for threshold mutant
        elif mutant_type == 'degrate':
            beta_k = mutant_params[0]
            alpha, beta_tau = 1.0, 1.0  # Not used for degrate mutant
        elif mutant_type == 'degrateAPC':
            beta_tau = mutant_params[0]
            alpha, beta_k = 1.0, 1.0  # Not used for degrateAPC mutant
        else:
            return 1e6
        
        # Apply mutant modifications to wildtype parameters
        params, n0_list = apply_mutant_params(wildtype_params, mutant_type, alpha, beta_k, beta_tau)
        
        # Run simulations
        sim_delta_t12, sim_delta_t32 = run_simulation_for_dataset(
            mechanism, params, n0_list, num_simulations
        )
        
        if sim_delta_t12 is None or sim_delta_t32 is None:
            return 1e6
        
        # Extract experimental data
        exp_delta_t12 = mutant_data['delta_t12']
        exp_delta_t32 = mutant_data['delta_t32']
        
        # Calculate likelihoods
        nll_12 = calculate_likelihood(exp_delta_t12, sim_delta_t12)
        nll_32 = calculate_likelihood(exp_delta_t32, sim_delta_t32)
        
        if nll_12 >= 1e6 or nll_32 >= 1e6:
            return 1e6
        
        return nll_12 + nll_32
    
    except Exception as e:
        return 1e6


def get_wildtype_parameter_bounds(mechanism):
    """
    Get parameter bounds for wildtype optimization.
    
    Args:
        mechanism (str): Mechanism name
    
    Returns:
        list: List of (min, max) bounds for wildtype parameters
    """
    # Common wildtype bounds - using ratio-based approach with tau = k_max/k_1
    bounds = [
        (3, 50),      # n2
        (100, 500),   # N2
        (0.01, 0.2),     # k_max
        (2, 240),     # tau = k_max/k_1 (2 seconds to 4 minutes, time units are in minutes)
        (0.5, 3.0),   # r21 (n1/n2 ratio)
        (0.5, 3.0),   # r23 (n3/n2 ratio)
        (0.4, 2.0),   # R21 (N1/N2 ratio)
        (0.5, 5.0),   # R23 (N3/N2 ratio)
    ]
    
    # Mechanism-specific bounds
    if mechanism == 'time_varying_k_fixed_burst':
        bounds.append((1, 20))  # burst_size
    elif mechanism == 'time_varying_k_feedback_onion':
        bounds.append((10, 50))  # n_inner
    elif mechanism == 'time_varying_k_combined':
        bounds.append((1, 20))   # burst_size
        bounds.append((10, 50))  # n_inner
    elif mechanism == 'time_varying_k_burst_onion':
        bounds.append((1, 20))   # burst_size
    
    return bounds


def get_mutant_parameter_bounds(mutant_type):
    """
    Get parameter bounds for mutant optimization.
    
    Args:
        mutant_type (str): Type of mutant
    
    Returns:
        list: List of (min, max) bounds for mutant parameters
    """
    if mutant_type == 'threshold':
        return [(0.1, 1.0)]  # alpha
    elif mutant_type == 'degrate':
        return [(0.1, 1.0)]  # beta_k
    elif mutant_type == 'degrateAPC':
        return [(2.0, 3.0)]  # beta_tau (tau becomes 2-3 times larger for APC mutant)
    else:
        return []


def optimize_wildtype(mechanism, wildtype_data, max_iterations=200, num_simulations=500, selected_strains=None):
    """
    Optimize wildtype parameters.
    
    Args:
        mechanism (str): Mechanism name
        wildtype_data (dict): Wildtype experimental data
        max_iterations (int): Maximum iterations
        num_simulations (int): Number of simulations per evaluation
        selected_strains (list): List of strain names to include (for compatibility)
    
    Returns:
        dict: Optimization results
    """
    print(f"\n=== Wildtype Optimization for {mechanism.upper()} ===")
    
    bounds = get_wildtype_parameter_bounds(mechanism)
    print(f"Optimizing {len(bounds)} wildtype parameters...")
    
    result = differential_evolution(
        wildtype_objective,
        bounds,
        args=(mechanism, wildtype_data, num_simulations, selected_strains),
        maxiter=max_iterations,
        popsize=15,
        seed=42,
        disp=True,
        workers=-1
    )
    
    convergence_status = "converged" if result.success else "did not converge"
    print(f"🔍 Wildtype optimization {convergence_status}!")
    print(f"Best negative log-likelihood: {result.fun:.4f}")
    
    if not result.success:
        print(f"Note: {result.message}")
    
    # Unpack parameters
    params = result.x
    if mechanism == 'time_varying_k':
        param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23']
    elif mechanism == 'time_varying_k_fixed_burst':
        param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'burst_size']
    elif mechanism == 'time_varying_k_feedback_onion':
        param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'n_inner']
    elif mechanism == 'time_varying_k_combined':
        param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'n_inner']
    elif mechanism == 'time_varying_k_burst_onion':
        param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'burst_size']
    
    param_dict = dict(zip(param_names, params))
    
    # Calculate derived parameters
    n1_derived = max(param_dict['r21'] * param_dict['n2'], 1)
    n3_derived = max(param_dict['r23'] * param_dict['n2'], 1)
    N1_derived = max(param_dict['R21'] * param_dict['N2'], 1)
    N3_derived = max(param_dict['R23'] * param_dict['N2'], 1)
    
    # Calculate k_1 from k_max and tau for complete params
    k_1_derived = param_dict['k_max'] / param_dict['tau']
    
    # Create complete parameter dictionary for simulations
    complete_params = {
        'n1': n1_derived, 'n2': param_dict['n2'], 'n3': n3_derived,
        'N1': N1_derived, 'N2': param_dict['N2'], 'N3': N3_derived,
        'k_1': k_1_derived, 'k_max': param_dict['k_max']
    }
    
    if 'burst_size' in param_dict:
        complete_params['burst_size'] = param_dict['burst_size']
    if 'n_inner' in param_dict:
        complete_params['n_inner'] = param_dict['n_inner']
    
    print(f"\nWildtype Parameters:")
    print(f"  Base: n2={param_dict['n2']:.1f}, N2={param_dict['N2']:.1f}")
    print(f"  Ratios: r21={param_dict['r21']:.2f}, r23={param_dict['r23']:.2f}, R21={param_dict['R21']:.2f}, R23={param_dict['R23']:.2f}")
    print(f"  Derived: n1={n1_derived:.1f}, n3={n3_derived:.1f}, N1={N1_derived:.1f}, N3={N3_derived:.1f}")
    print(f"  Rates: k_max={param_dict['k_max']:.4f}, tau={param_dict['tau']:.1f} min, k_1={k_1_derived:.6f}")
    
    return {
        'success': True,
        'converged': result.success,
        'params': param_dict,
        'complete_params': complete_params,
        'nll': result.fun,
        'result': result,
        'message': result.message if not result.success else "Converged successfully"
    }


def optimize_mutant(mechanism, mutant_data, wildtype_params, mutant_type, max_iterations=100, num_simulations=500, selected_strains=None):
    """
    Optimize mutant parameters with fixed wildtype parameters.
    
    Args:
        mechanism (str): Mechanism name
        mutant_data (dict): Mutant experimental data
        wildtype_params (dict): Fixed wildtype parameters
        mutant_type (str): Type of mutant
        max_iterations (int): Maximum iterations
        num_simulations (int): Number of simulations per evaluation
        selected_strains (list): List of strain names to include (for compatibility)
    
    Returns:
        dict: Optimization results
    """
    print(f"\n=== {mutant_type.title()} Mutant Optimization ===")
    
    bounds = get_mutant_parameter_bounds(mutant_type)
    if not bounds:
        return {'success': False, 'message': f"Unknown mutant type: {mutant_type}"}
    
    print(f"Optimizing {len(bounds)} mutant parameters...")
    
    result = differential_evolution(
        mutant_objective,
        bounds,
        args=(mechanism, mutant_data, wildtype_params, mutant_type, num_simulations, selected_strains),
        maxiter=max_iterations,
        popsize=10,
        seed=42,
        disp=True,
        workers=1
    )
    
    convergence_status = "converged" if result.success else "did not converge"
    print(f"🔍 {mutant_type.title()} optimization {convergence_status}!")
    print(f"Best negative log-likelihood: {result.fun:.4f}")
    
    if not result.success:
        print(f"Note: {result.message}")
    
    # Unpack parameters
    param_value = result.x[0]
    if mutant_type == 'threshold':
        param_name = 'alpha'
    elif mutant_type == 'degrate':
        param_name = 'beta_k'
    elif mutant_type == 'degrateAPC':
        param_name = 'beta_tau'
    
    print(f"{mutant_type.title()} parameter: {param_name} = {param_value:.3f}")
    
    return {
        'success': True,
        'converged': result.success,
        'param_name': param_name,
        'param_value': param_value,
        'nll': result.fun,
        'result': result,
        'message': result.message if not result.success else "Converged successfully"
    }


def run_independent_optimization(mechanism, datasets, max_iterations_wt=200, max_iterations_mut=100, num_simulations=500, selected_strains=None):
    """
    Run independent optimization for selected datasets.
    
    Args:
        mechanism (str): Mechanism name
        datasets (dict): Experimental datasets
        max_iterations_wt (int): Maximum iterations for wildtype
        max_iterations_mut (int): Maximum iterations for mutants
        num_simulations (int): Number of simulations per evaluation
        selected_strains (list): List of strain names to include in fitting
    
    Returns:
        dict: Complete optimization results
    """
    print(f"\n{'='*60}")
    print(f"Independent Optimization for {mechanism.upper()}")
    if selected_strains is not None:
        print(f"Selected strains: {selected_strains}")
    print(f"{'='*60}")
    
    results = {'mechanism': mechanism}
    
    # Step 1: Optimize wildtype parameters
    if 'wildtype' not in datasets:
        print("Error: Wildtype data not found!")
        return {'success': False}
    
    wt_results = optimize_wildtype(mechanism, datasets['wildtype'], max_iterations_wt, num_simulations, selected_strains)
    if not wt_results['success']:
        print("Wildtype optimization failed!")
        return {'success': False}
    
    results['wildtype'] = wt_results
    wildtype_params = wt_results['complete_params']
    
    # Step 2: Optimize each mutant separately
    if selected_strains is None:
        # Use all available mutants
        mutant_types = ['threshold', 'degrate', 'degrateAPC']
    else:
        # Only optimize mutants that are in the selected strains
        mutant_types = [strain for strain in selected_strains if strain != 'wildtype']
    
    for mutant_type in mutant_types:
        if mutant_type not in datasets:
            print(f"Warning: {mutant_type} data not found, skipping...")
            continue
        
        mut_results = optimize_mutant(
            mechanism, datasets[mutant_type], wildtype_params, 
            mutant_type, max_iterations_mut, num_simulations, selected_strains
        )
        
        if mut_results['success']:
            results[mutant_type] = mut_results
        else:
            print(f"Warning: {mutant_type} optimization failed")
    
    # Step 3: Calculate total NLL and create summary
    total_nll = results['wildtype']['nll']
    summary = {
        'wildtype_nll': results['wildtype']['nll'],
        'mutant_nlls': {}
    }
    
    for mutant_type in mutant_types:
        if mutant_type in results:
            total_nll += results[mutant_type]['nll']
            summary['mutant_nlls'][mutant_type] = results[mutant_type]['nll']
    
    summary['total_nll'] = total_nll
    results['summary'] = summary
    results['success'] = True
    
    return results


def save_independent_results(mechanism, results, filename=None, selected_strains=None):
    """
    Save independent optimization results to file.
    
    Args:
        mechanism (str): Mechanism name
        results (dict): Optimization results
        filename (str): Output filename (optional)
        selected_strains (list): List of strains used in fitting
    """
    if not results.get('success', False):
        print("Cannot save results - optimization failed")
        return
    
    # Determine filename based on selected strains
    if filename is None:
        # Create strain suffix if specific strains were selected
        strain_suffix = ""
        if selected_strains is not None:
            strain_suffix = f"_{'_'.join(selected_strains)}"
        
        filename = f"simulation_optimized_parameters_{mechanism}_independent{strain_suffix}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"Simulation-based Independent Optimization Results\n")
        f.write(f"Mechanism: {mechanism}\n")
        
        # Add strain selection information
        if selected_strains is not None:
            f.write(f"Selected Strains: {', '.join(selected_strains)}\n")
        else:
            f.write(f"Selected Strains: all datasets\n")
        
        f.write(f"Total Negative Log-Likelihood: {results['summary']['total_nll']:.6f}\n")
        f.write(f"Strategy: Independent optimization (wildtype first, then mutants separately)\n\n")
        
        # Wildtype parameters
        wt_params = results['wildtype']['params']
        f.write("=== WILDTYPE PARAMETERS ===\n")
        f.write(f"Converged: {results['wildtype']['converged']}\n")
        f.write(f"Wildtype NLL: {results['wildtype']['nll']:.6f}\n")
        f.write(f"Status: {results['wildtype']['message']}\n\n")
        
        f.write("Wildtype Parameters (ratio-based):\n")
        for param, value in wt_params.items():
            f.write(f"{param} = {value:.6f}\n")
        
        # Calculate and save derived parameters
        if 'r21' in wt_params:
            n1_derived = max(wt_params['r21'] * wt_params['n2'], 1)
            n3_derived = max(wt_params['r23'] * wt_params['n2'], 1)
            N1_derived = max(wt_params['R21'] * wt_params['N2'], 1)
            N3_derived = max(wt_params['R23'] * wt_params['N2'], 1)
            
            f.write(f"\nDerived Wildtype Parameters:\n")
            f.write(f"n1 = {n1_derived:.6f}\n")
            f.write(f"n3 = {n3_derived:.6f}\n")
            f.write(f"N1 = {N1_derived:.6f}\n")
            f.write(f"N3 = {N3_derived:.6f}\n")
        
        # Mutant parameters
        mutant_types = ['threshold', 'degrate', 'degrateAPC']
        f.write(f"\n=== MUTANT PARAMETERS ===\n")
        
        for mutant_type in mutant_types:
            if mutant_type in results:
                mut_result = results[mutant_type]
                f.write(f"\n{mutant_type.title()} Mutant:\n")
                f.write(f"Converged: {mut_result['converged']}\n")
                f.write(f"{mutant_type.title()} NLL: {mut_result['nll']:.6f}\n")
                f.write(f"Status: {mut_result['message']}\n")
                f.write(f"{mut_result['param_name']} = {mut_result['param_value']:.6f}\n")
        
        # Summary
        f.write(f"\n=== SUMMARY ===\n")
        f.write(f"Wildtype NLL: {results['summary']['wildtype_nll']:.6f}\n")
        for mutant_type, nll in results['summary']['mutant_nlls'].items():
            f.write(f"{mutant_type.title()} NLL: {nll:.6f}\n")
        f.write(f"Total NLL: {results['summary']['total_nll']:.6f}\n")
    
    print(f"Results saved to: {filename}")


def print_independent_summary(mechanism, results):
    """
    Print a summary of independent optimization results.
    
    Args:
        mechanism (str): Mechanism name
        results (dict): Optimization results
    """
    if not results.get('success', False):
        print("No results to summarize - optimization failed")
        return
    
    print(f"\n{'='*60}")
    print(f"INDEPENDENT OPTIMIZATION SUMMARY: {mechanism.upper()}")
    print(f"{'='*60}")
    
    # Wildtype summary
    wt_params = results['wildtype']['params']
    wt_complete = results['wildtype']['complete_params']
    
    print(f"\nWILDTYPE PARAMETERS:")
    print(f"  Convergence: {'✅' if results['wildtype']['converged'] else '❌'}")
    print(f"  NLL: {results['wildtype']['nll']:.4f}")
    print(f"  Base: n2={wt_params['n2']:.1f}, N2={wt_params['N2']:.1f}")
    print(f"  Ratios: r21={wt_params['r21']:.2f}, r23={wt_params['r23']:.2f}, R21={wt_params['R21']:.2f}, R23={wt_params['R23']:.2f}")
    print(f"  Derived: n1={wt_complete['n1']:.1f}, n3={wt_complete['n3']:.1f}, N1={wt_complete['N1']:.1f}, N3={wt_complete['N3']:.1f}")
    # Calculate k_1 from k_max and tau for display
    k_1_display = wt_params['k_max'] / wt_params['tau'] if 'tau' in wt_params else wt_complete['k_1']
    tau_display = f"{wt_params['tau']:.1f}" if 'tau' in wt_params else "N/A"
    print(f"  Rates: k_max={wt_params['k_max']:.4f}, tau={tau_display} min, k_1={k_1_display:.6f}")
    
    if 'burst_size' in wt_params:
        print(f"  Burst size: {wt_params['burst_size']:.1f}")
    if 'n_inner' in wt_params:
        print(f"  Inner threshold: {wt_params['n_inner']:.1f}")
    
    # Mutant summary
    print(f"\nMUTANT PARAMETERS:")
    mutant_types = ['threshold', 'degrate', 'degrateAPC']
    for mutant_type in mutant_types:
        if mutant_type in results:
            mut_result = results[mutant_type]
            convergence = '✅' if mut_result['converged'] else '❌'
            print(f"  {mutant_type.title()}: {convergence} {mut_result['param_name']}={mut_result['param_value']:.3f}, NLL={mut_result['nll']:.4f}")
        else:
            print(f"  {mutant_type.title()}: ❌ Not optimized")
    
    # Total summary
    print(f"\nTOTAL SUMMARY:")
    print(f"  Total NLL: {results['summary']['total_nll']:.4f}")
    print(f"  Components: WT={results['summary']['wildtype_nll']:.4f}")
    for mutant_type, nll in results['summary']['mutant_nlls'].items():
        print(f"             {mutant_type.upper()}={nll:.4f}")


def main():
    """
    Main independent optimization routine for all strains.
    """
    max_iterations_wt = 100  # Wildtype iterations
    max_iterations_mut = 100  # Mutant iterations
    num_simulations = 200     # Simulations per evaluation
    
    print("Simulation-based Independent Optimization for Time-Varying Mechanisms")
    print("=" * 70)
    
    # Load experimental data
    datasets = load_experimental_data()
    if not datasets:
        print("Error: No datasets loaded!")
        return
    
    # Test mechanisms
    mechanisms = ['time_varying_k', 'time_varying_k_fixed_burst', 'time_varying_k_feedback_onion', 'time_varying_k_combined', 'time_varying_k_burst_onion']
    mechanism = mechanisms[1]  # Test fixed_burst mechanism
    
    print(f"\nRunning independent optimization for {mechanism} with ALL strains")
    
    try:
        results = run_independent_optimization(
            mechanism, datasets, max_iterations_wt, max_iterations_mut, num_simulations,
            selected_strains=None  # Use all strains
        )
        
        if results['success']:
            print_independent_summary(mechanism, results)
            save_independent_results(mechanism, results, selected_strains=None)
        else:
            print("Independent optimization failed!")
        
        print(f"\n{'-' * 70}")
        
    except Exception as e:
        print(f"Error during independent optimization: {e}")
        import traceback
        traceback.print_exc()
    
    print("Independent optimization complete!")





if __name__ == "__main__":
    # Run main() - independent optimization for all strains
    main() 