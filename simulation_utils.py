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
    Apply mutant-specific parameter modifications.
    
    Args:
        base_params (dict): Base wildtype parameters
        mutant_type (str): Type of mutant ('wildtype', 'threshold', 'degrade', 'degradeAPC', 'velcade')
        alpha (float): Multiplier for threshold counts (threshold mutant)
        beta_k (float): Multiplier for k_max (separase mutant)
        beta_tau (float): Multiplier for tau (APC mutant) - tau becomes 2-3 times larger
        beta_tau2 (float): Multiplier for tau (velcade mutant) - similar to APC but separate parameter
    
    Returns:
        tuple: (modified_params, modified_n0_list)
    """
    params = base_params.copy()
    
    # Base parameters
    n1, n2, n3 = params['n1'], params['n2'], params['n3']
    N1, N2, N3 = params['N1'], params['N2'], params['N3']
    
    if mutant_type == 'wildtype':
        # No modifications for wildtype
        pass
    elif mutant_type == 'threshold':
        # Threshold mutant: reduce threshold counts (small n)
        n1, n2, n3 = alpha * n1, alpha * n2, alpha * n3
    elif mutant_type == 'degrade':
        # Separase mutant: reduce k_max (slower degradation)
        params['k_max'] = beta_k * params['k_max']
    elif mutant_type == 'degradeAPC':
        # APC mutant: affects tau (makes it 2-3 times larger), which affects k_1
        if 'tau' in params:
            # Apply multiplier directly to tau
            params['tau'] = beta_tau * params['tau']
        elif 'k_1' in params and 'k_max' in params:
            # Calculate current tau, apply multiplier, then recalculate k_1
            current_tau = params['k_max'] / params['k_1']
            new_tau = beta_tau * current_tau
            params['k_1'] = params['k_max'] / new_tau
    elif mutant_type == 'velcade':
        # Velcade mutant: affects tau (similar to APC but separate parameter), which affects k_1
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
    
    # Ensure k_1 is calculated and available for simulations
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
    """
    try:
        # (Your existing code for setting up the simulation)
        # ...
        initial_state = [int(round(params['N1'])), int(round(params['N2'])), int(round(params['N3']))]
        k_1 = calculate_k1_from_params(params)
        rate_params = { # Simplified for brevity, your full logic is correct
            'k_1': k_1,
            'k_max': params['k_max']
        }
        if 'burst_size' in params:
            rate_params['burst_size'] = params['burst_size']
        if 'n_inner' in params:
            rate_params['n_inner'] = params['n_inner']

        delta_t12_list = []
        delta_t32_list = []
        
        for _ in range(num_simulations):
            try:
                sim = MultiMechanismSimulationTimevary(
                    mechanism=mechanism,
                    initial_state_list=initial_state,
                    rate_params=rate_params,
                    n0_list=n0_list,
                    max_time=1000
                )
                _, _, sep_times = sim.simulate()
                delta_t12 = sep_times[0] - sep_times[1]
                delta_t32 = sep_times[2] - sep_times[1]
                delta_t12_list.append(delta_t12)
                delta_t32_list.append(delta_t32)
            except Exception as sim_error:
                continue

        # --- THIS IS THE CRITICAL CHANGE ---
        # If all simulations failed and the lists are empty, signal failure.
        if not delta_t12_list or not delta_t32_list:
            return None, None
        
        return delta_t12_list, delta_t32_list
    
    except Exception as e:
        # Also signal failure if there's a setup error
        return None, None


def calculate_likelihood(experimental_data, simulated_data):
    """
    Calculate likelihood using Kernel Density Estimation.
    
    Args:
        experimental_data (dict): Dictionary with 'delta_t12' and 'delta_t32' arrays
        simulated_data (dict): Dictionary with 'delta_t12' and 'delta_t32' arrays
    
    Returns:
        float: Negative log-likelihood
    """
    try:
        nll_total = 0.0
        
        for key in ['delta_t12', 'delta_t32']:
            if key not in experimental_data or key not in simulated_data:
                continue
            
            exp_data = experimental_data[key]
            sim_data = simulated_data[key]
            
            if len(exp_data) == 0 or len(sim_data) == 0:
                continue
            
            # Use KDE to estimate density from simulated data
            kde = KernelDensity(kernel='gaussian', bandwidth=10.0)
            kde.fit(sim_data.reshape(-1, 1))
            
            # Evaluate likelihood of experimental data under this density
            log_densities = kde.score_samples(exp_data.reshape(-1, 1))
            
            # Sum negative log-likelihoods
            nll = -np.sum(log_densities)
            
            # Add small penalty for extreme outliers
            if np.any(log_densities < -20):
                nll += 100 * np.sum(log_densities < -20)

            # NORMALIZATION REMOVED: Use raw NLL to show true model differences
            # nll = nll / len(exp_data)  # COMMENTED OUT for better discrimination

            nll_total += nll
        
        return nll_total
    
    except Exception as e:
        print(f"Error in likelihood calculation: {e}")
        return 1e6  # Return large penalty for errors


def get_parameter_bounds(mechanism):
    """
    Get parameter bounds for optimization.
    
    Args:
        mechanism (str): Mechanism name
    
    Returns:
        list: List of (min, max) tuples for each parameter
    """
    # Base bounds for all mechanisms
    bounds = [
        (1.0, 50.0),      # n2
        (50.0, 1000.0),    # N2
        (0.01, 0.1),     # k_max
        (2.0, 240.0),    # tau
        (0.25, 4.0),       # r21
        (0.25, 4.0),       # r23
        (0.4, 2),       # R21
        (0.5, 5.0),       # R23
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
        (2, 10.0),         # beta_tau
        (2, 40.0),         # beta_tau2
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
