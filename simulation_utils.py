#!/usr/bin/env python3
"""
Shared utilities for simulation-based optimization of chromosome segregation timing models.
Contains common functions used by both joint and independent optimization strategies.
"""

import numpy as np
import pandas as pd
from MultiMechanismSimulationTimevary import MultiMechanismSimulationTimevary
from sklearn.neighbors import KernelDensity
import warnings
warnings.filterwarnings('ignore')
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SecondVersion'))
from MultiMechanismSimulation import MultiMechanismSimulation
from MultiMechanismSimulationTimevary import MultiMechanismSimulationTimevary


def load_experimental_data():
    """
    Load experimental data from Excel file.
    
    Returns:
        dict: Dictionary with dataset names as keys and timing difference arrays as values
    """
    try:
        file_path = "Data/All_strains_SCStimes.xlsx"
        
        # Load the single sheet
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        print(f"Loaded data with shape: {df.shape}")
        
        datasets = {}
        
        # Map column names to dataset names
        dataset_mapping = {
            'wildtype': ('wildtype12', 'wildtype32'),
            'threshold': ('threshold12', 'threshold32'),
            'degrade': ('degRade12', 'degRade32'),
            'degradeAPC': ('degRadeAPC12', 'degRadeAPC32'),
            'velcade': ('degRadeVel12', 'degRadeVel32')
        }
        
        for dataset_name, (col_12, col_32) in dataset_mapping.items():
            if col_12 in df.columns and col_32 in df.columns:
                # Extract non-NaN values
                delta_t12 = df[col_12].dropna().values
                delta_t32 = df[col_32].dropna().values
                
                datasets[dataset_name] = {
                    'delta_t12': delta_t12,
                    'delta_t32': delta_t32
                }
                
                print(f"Loaded {dataset_name}: {len(delta_t12)} T1-T2 points, {len(delta_t32)} T3-T2 points")
            else:
                print(f"Warning: Could not find columns for {dataset_name}: {col_12}, {col_32}")
        
        return datasets
    
    except Exception as e:
        print(f"Error loading experimental data: {e}")
        return {}


def apply_mutant_params(base_params, mutant_type, alpha, beta_k, beta_tau=None, beta_tau2=None):
    """
    Apply mutant-specific parameter modifications for both simple and time-varying mechanisms.
    
    Args:
        base_params (dict): Base wildtype parameters
        mutant_type (str): Type of mutant ('wildtype', 'threshold', 'degrade', 'degradeAPC', 'velcade')
        alpha (float): Multiplier for threshold counts (threshold mutant)
        beta_k (float): Multiplier for k (simple) or k_max (time-varying) - degradation rate
        beta_tau (float): Multiplier for tau (APC mutant) - tau becomes 2-3 times larger (time-varying only)
        beta_tau2 (float): Multiplier for tau (velcade mutant) - similar to APC but separate parameter (time-varying only)
    
    Returns:
        tuple: (modified_params, modified_n0_list)
    """
    params = base_params.copy()
    
    # Base parameters - always present
    n1, n2, n3 = params['n1'], params['n2'], params['n3']
    N1, N2, N3 = params['N1'], params['N2'], params['N3']
    
    # Determine if this is a simple mechanism or time-varying
    is_simple = 'k' in params  # Simple mechanisms have 'k', time-varying have 'k_max' and 'tau'
    
    if mutant_type == 'wildtype':
        # No modifications for wildtype
        pass
    
    elif mutant_type == 'threshold':
        # Threshold mutant: reduce threshold counts (small n)
        n1, n2, n3 = alpha * n1, alpha * n2, alpha * n3
    
    elif mutant_type == 'degrade':
        # Degradation mutant: reduce degradation rate
        if is_simple:
            # Simple mechanisms: modify 'k' directly
            params['k'] = beta_k * params['k']
        else:
            # Time-varying mechanisms: modify 'k_max'
            params['k_max'] = beta_k * params['k_max']
    
    elif mutant_type == 'degradeAPC':
        # APC mutant: affects tau (time-varying) or treated as degrade (simple)
        if is_simple:
            # Simple mechanisms: treat like degrade mutant
            params['k'] = beta_k * params['k']
        else:
            # Time-varying mechanisms: affects tau
            if 'tau' in params:
                # Apply multiplier directly to tau
                params['tau'] = beta_tau * params['tau']
            elif 'k_1' in params and 'k_max' in params:
                # Calculate current tau, apply multiplier, then recalculate k_1
                current_tau = params['k_max'] / params['k_1']
                new_tau = beta_tau * current_tau
                params['k_1'] = params['k_max'] / new_tau
    
    elif mutant_type == 'velcade':
        # Velcade mutant: affects tau (time-varying) or treated as degrade (simple)
        if is_simple:
            # Simple mechanisms: treat like degrade mutant
            params['k'] = beta_k * params['k']
        else:
            # Time-varying mechanisms: affects tau
            if 'tau' in params:
                # Apply multiplier directly to tau
                params['tau'] = beta_tau2 * params['tau']
            elif 'k_1' in params and 'k_max' in params:
                # Calculate current tau, apply multiplier, then recalculate k_1
                current_tau = params['k_max'] / params['k_1']
                new_tau = beta_tau2 * current_tau
                params['k_1'] = params['k_max'] / new_tau
    
    # Update the modified parameters
    params['n1'], params['n2'], params['n3'] = n1, n2, n3
    params['N1'], params['N2'], params['N3'] = N1, N2, N3
    
    # Ensure k_1 is calculated and available for simulations (time-varying mechanisms only)
    if 'tau' in params and 'k_max' in params and 'k_1' not in params:
        params['k_1'] = calculate_k1_from_params(params)
    
    # Create n0_list (thresholds for simulation)
    n0_list = [n1, n2, n3]
    
    return params, n0_list


def calculate_k1_from_params(params):
    """
    Calculate k_1 from k_max and tau parameters.
    
    Args:
        params (dict): Parameters containing k_max and tau
    
    Returns:
        float: Calculated k_1 value
    """
    if 'k_max' in params and 'tau' in params:
        return params['k_max'] / params['tau']
    elif 'k_1' in params:
        return params['k_1']
    else:
        raise ValueError("Cannot calculate k_1: missing k_max/tau or k_1 parameters")


# In simulation_utils.py

def run_simulation_for_dataset(mechanism, params, n0_list, num_simulations=500):
    """
    Run simulations for a given mechanism and parameters.
    Handles both simple mechanisms (simple, fixed_burst, feedback_onion, fixed_burst_feedback_onion)
    and time-varying mechanisms (time_varying_k, time_varying_k_fixed_burst, etc.)
    
    Args:
        mechanism (str): Mechanism name
        params (dict): Parameter dictionary
        n0_list (list): List of threshold values [n01, n02, n03]
        num_simulations (int): Number of simulations to run
    
    Returns:
        tuple: (delta_t12_array, delta_t32_array) as numpy arrays, or (None, None) on failure
    """
    # Check if simple mechanism
    # Use simple mechanism simulator (SecondVersion)

    
    # Prepare rate parameters based on mechanism
    if mechanism == 'simple':
        rate_params = {'k': params['k']}
    elif mechanism == 'fixed_burst':
        rate_params = {'k': params['k'], 'burst_size': params['burst_size']}
    elif mechanism == 'feedback_onion':
        rate_params = {'k': params['k'], 'n_inner': params['n_inner']}
    elif mechanism == 'fixed_burst_feedback_onion':
        rate_params = {'k': params['k'], 'burst_size': params['burst_size'], 'n_inner': params['n_inner']}
    else: # time-varying mechanisms
        k_1 = calculate_k1_from_params(params)
        rate_params = {
            'k_1': k_1,
            'k_max': params['k_max']
        }
        if 'burst_size' in params:
            rate_params['burst_size'] = params['burst_size']
        if 'n_inner' in params:
            rate_params['n_inner'] = params['n_inner']
    # Initial states
    initial_state = [params['N1'], params['N2'], params['N3']]
    
    # Run simulations
    delta_t12_list = []
    delta_t32_list = []
    
    for _ in range(num_simulations):
        if mechanism in ['simple', 'fixed_burst', 'feedback_onion', 'fixed_burst_feedback_onion']:
            sim = MultiMechanismSimulation(
                mechanism=mechanism,
                initial_state_list=initial_state,
                rate_params=rate_params,
                n0_list=n0_list,
                max_time=1000.0
            )
        else:
            sim = MultiMechanismSimulationTimevary(
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


def calculate_likelihood(exp_data, sim_data):
    """
    Calculate negative log-likelihood using KDE with Scott's rule bandwidth.
    
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
                    
                    # Build KDE using Scott's rule (automatic bandwidth)
                    kde = KernelDensity(kernel='gaussian', bandwidth='scott')
                    kde.fit(sim_values.reshape(-1, 1))
                    
                    # Calculate likelihood
                    log_densities = kde.score_samples(exp_values.reshape(-1, 1))
                    nll = -np.sum(log_densities)
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
            
            kde = KernelDensity(kernel='gaussian', bandwidth='scott')
            kde.fit(sim_values.reshape(-1, 1))
            log_densities = kde.score_samples(exp_values.reshape(-1, 1))
            return -np.sum(log_densities)
    
    except Exception as e:
        print(f"Likelihood calculation error: {e}")
        return 1e6


def get_parameter_bounds(mechanism):
    """
    Get parameter bounds for optimization.
    
    Args:
        mechanism (str): Mechanism name
    
    Returns:
        list: List of (min, max) tuples for each parameter
    """
    
    bounds = [
        (1.0, 50.0),      # n2
        (50.0, 1000.0),    # N2
        (0.01, 0.1),       # k_max
        (0.1, 0.2),        # tau
        (0.25, 4.0),       # r21
        (0.25, 4.0),       # r23
        (0.4, 2),          # R21
        (0.5, 5.0),        # R23
    ]
    
    # Add mechanism-specific bounds
    if mechanism == 'time_varying_k_fixed_burst':
        bounds.append((1.0, 50.0))    # burst_size
    elif mechanism == 'time_varying_k_feedback_onion':
        bounds.append((1.0, 100.0))   # n_inner
    elif mechanism == 'time_varying_k_combined':
        bounds.append((1.0, 50.0))    # burst_size
        bounds.append((1.0, 100.0))   # n_inner
    elif mechanism == 'time_varying_k_burst_onion':
        bounds.append((1.0, 50.0))    # burst_size
    
    # Add mutant parameter bounds
    bounds.extend([
        (0.1, 0.7),       # alpha
        (0.1, 1.0),       # beta_k
        (2, 10.0),        # beta_tau
        (2, 40.0),        # beta_tau2
    ])
    
    return bounds


def get_parameter_names(mechanism):
    """
    Get parameter names for a given mechanism.
    
    Args:
        mechanism (str): Mechanism name
    
    Returns:
        list: List of parameter names
    """
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
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")
    
    return param_names


def parse_parameters(params_vector, mechanism):
    """
    Parse parameter vector into a dictionary.
    
    Args:
        params_vector (array): Parameter values
        mechanism (str): Mechanism name
    
    Returns:
        dict: Parameter dictionary
    """
    param_names = get_parameter_names(mechanism)
    param_dict = dict(zip(param_names, params_vector))
    
    # Calculate derived parameters for display
    n1_derived = max(param_dict['r21'] * param_dict['n2'], 1)
    n3_derived = max(param_dict['r23'] * param_dict['n2'], 1)
    N1_derived = max(param_dict['R21'] * param_dict['N2'], 1)
    N3_derived = max(param_dict['R23'] * param_dict['N2'], 1)
    
    # Calculate k_1 from k_max and tau for display
    k_1_derived = calculate_k1_from_params(param_dict)
    
    # Add derived parameters to the dictionary
    param_dict.update({
        'n1': n1_derived,
        'n3': n3_derived,
        'N1': N1_derived,
        'N3': N3_derived,
        'k_1': k_1_derived
    })
    
    return param_dict


def save_optimization_results(mechanism, results, filename=None, selected_strains=None):
    """
    Save optimization results to a text file.
    
    Args:
        mechanism (str): Mechanism name
        results (dict): Optimization results
        filename (str): Output filename (optional)
        selected_strains (list): List of selected strain names (optional)
    """
    if filename is None:
        filename = f"simulation_optimized_parameters_{mechanism}.txt"
    
    try:
        with open(filename, 'w') as f:
            f.write("Simulation-based Optimization Results\n")
            f.write(f"Mechanism: {mechanism}\n")
            f.write(f"Negative Log-Likelihood: {results['fun']:.6f}\n")
            f.write(f"Converged: {results['success']}\n")
            f.write(f"Status: {results.get('message', 'Unknown')}\n")
            
            if selected_strains:
                f.write(f"Datasets: {', '.join(selected_strains)}\n")
            else:
                f.write("Datasets: wildtype, threshold, degrade, degradeAPC\n")
            f.write("\n")
            
            # Parse parameters
            param_dict = parse_parameters(results['x'], mechanism)
            
            # Write ratio-based parameters (what was optimized)
            f.write("Optimized Parameters (ratio-based):\n")
            param_names = get_parameter_names(mechanism)
            for name in param_names:
                if name in param_dict:
                    f.write(f"{name} = {param_dict[name]:.6f}\n")
            f.write("\n")
            
            # Write derived parameters
            f.write("Derived Parameters:\n")
            f.write(f"n1 = {param_dict['n1']:.6f}\n")
            f.write(f"n3 = {param_dict['n3']:.6f}\n")
            f.write(f"N1 = {param_dict['N1']:.6f}\n")
            f.write(f"N3 = {param_dict['N3']:.6f}\n")
            f.write(f"k_1 = {param_dict['k_1']:.6f}\n")
            f.write("\n")
        
        print(f"Results saved to: {filename}")
    
    except Exception as e:
        print(f"Error saving results: {e}")
