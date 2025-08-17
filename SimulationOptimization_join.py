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
from MultiMechanismSimulationTimevary import MultiMechanismSimulationTimevary
from Chromosomes_Theory import (
    calculate_bootstrap_likelihood, 
    calculate_weighted_likelihood,
    BootstrappingFitnessCalculator,
    analyze_dataset_sizes
)
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
            'degrate': ('degRate12', 'degRate32'),
            'degrateAPC': ('degRateAPC12', 'degRateAPC32')
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


def apply_mutant_params(base_params, mutant_type, alpha, beta_k, beta2_k=None):
    """
    Apply mutant-specific parameter modifications.
    
    Args:
        base_params (dict): Base wildtype parameters
        mutant_type (str): Type of mutant ('wildtype', 'threshold', 'degrate', 'degrateAPC')
        alpha (float): Multiplier for threshold counts (threshold mutant)
        beta_k (float): Multiplier for k_max (separase mutant)
        beta2_k (float): Multiplier for k_1 (APC mutant)
    
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
    elif mutant_type == 'degrate':
        # Separase mutant: affects k_max
        if 'k_max' in params:
            params['k_max'] = beta_k * params['k_max']
    elif mutant_type == 'degrateAPC':
        # APC mutant: affects k_1
        if 'k_1' in params:
            params['k_1'] = beta2_k * params['k_1']
    
    # Update parameters (N values unchanged)
    params.update({'N1': N1, 'N2': N2, 'N3': N3})
    n0_list = [n1, n2, n3]  # Use modified threshold values
    
    return params, n0_list


def run_simulation_for_dataset(mechanism, params, n0_list, num_simulations=500):
    """
    Run multiple simulations for a given parameter set and calculate timing statistics.
    
    Args:
        mechanism (str): Mechanism name
        params (dict): Simulation parameters
        n0_list (list): Threshold counts
        num_simulations (int): Number of simulation runs
    
    Returns:
        tuple: (delta_t12_list, delta_t32_list) or (None, None) if failed
    """
    try:
        initial_state = [params['N1'], params['N2'], params['N3']]
        
        # Extract rate parameters based on mechanism
        if mechanism == 'time_varying_k':
            rate_params = {
                'k_1': params['k_1'],
                'k_max': params['k_max']
            }
        elif mechanism == 'time_varying_k_fixed_burst':
            rate_params = {
                'k_1': params['k_1'],
                'k_max': params['k_max'],
                'burst_size': params['burst_size']
            }
        elif mechanism == 'time_varying_k_feedback_onion':
            rate_params = {
                'k_1': params['k_1'],
                'k_max': params['k_max'],
                'n_inner': params['n_inner']
            }
        elif mechanism == 'time_varying_k_combined':
            rate_params = {
                'k_1': params['k_1'],
                'k_max': params['k_max'],
                'burst_size': params['burst_size'],
                'n_inner': params['n_inner']
            }
        elif mechanism == 'time_varying_k_burst_onion':
            rate_params = {
                'k_1': params['k_1'],
                'k_max': params['k_max'],
                'burst_size': params['burst_size']
            }
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")
        
        delta_t12_list = []
        delta_t32_list = []
        
        for _ in range(num_simulations):
            sim = MultiMechanismSimulationTimevary(
                mechanism=mechanism,
                initial_state_list=initial_state,
                rate_params=rate_params,
                n0_list=n0_list,
                max_time=1000
            )
            
            _, _, sep_times = sim.simulate()
            
            # Calculate time differences
            delta_t12 = sep_times[0] - sep_times[1]  # T1 - T2
            delta_t32 = sep_times[2] - sep_times[1]  # T3 - T2
            
            delta_t12_list.append(delta_t12)
            delta_t32_list.append(delta_t32)
        
        return delta_t12_list, delta_t32_list
    
    except Exception as e:
        print(f"Simulation error: {e}")
        return None, None


def calculate_likelihood(experimental_data, simulated_data):
    """
    Calculate negative log-likelihood between experimental and simulated data.
    Uses kernel density estimation for simulated data.
    
    Args:
        experimental_data (array): Experimental timing differences
        simulated_data (array): Simulated timing differences
    
    Returns:
        float: Negative log-likelihood
    """
    try:
        from scipy.stats import gaussian_kde
        
        if len(simulated_data) < 10:
            return 1e6  # Penalty for insufficient data
        
        # Create KDE from simulated data
        kde = gaussian_kde(simulated_data)
        
        # Calculate likelihood for experimental data points
        log_likelihoods = kde.logpdf(experimental_data)
        
        # Handle numerical issues
        log_likelihoods = np.clip(log_likelihoods, -50, 50)
        
        # Return negative log-likelihood (for minimization)
        return -np.sum(log_likelihoods)
    
    except Exception as e:
        print(f"Likelihood calculation error: {e}")
        return 1e6


def joint_objective_with_bootstrapping(params_vector, mechanism, datasets, 
                                     num_simulations=500, 
                                     bootstrap_method='bootstrap',
                                     target_sample_size=50, 
                                     num_bootstrap_samples=100,
                                     random_seed=None):
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
            n2, N2, k_1, k_max, r21, r23, R21, R23, alpha, beta_k, beta2_k = params_vector
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
            n2, N2, k_1, k_max, r21, r23, R21, R23, burst_size, alpha, beta_k, beta2_k = params_vector
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
            n2, N2, k_1, k_max, r21, r23, R21, R23, n_inner, alpha, beta_k, beta2_k = params_vector
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
            n2, N2, k_1, k_max, r21, r23, R21, R23, burst_size, n_inner, alpha, beta_k, beta2_k = params_vector
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
            n2, N2, k_1, k_max, r21, r23, R21, R23, burst_size, alpha, beta_k, beta2_k = params_vector
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
        
        for dataset_name, data_dict in datasets.items():
            # Apply mutant-specific modifications
            params, n0_list = apply_mutant_params(
                base_params, dataset_name, alpha, beta_k, beta2_k
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


def joint_objective(params_vector, mechanism, datasets, num_simulations=500):
    """
    Joint objective function for all datasets.
    
    Args:
        params_vector (array): Parameter vector to optimize
        mechanism (str): Mechanism name
        datasets (dict): Experimental datasets
        num_simulations (int): Number of simulations per evaluation
    
    Returns:
        float: Total negative log-likelihood across all datasets
    """
    # Add debugging to see parameter values being tried
    if hasattr(joint_objective, 'call_count'):
        joint_objective.call_count += 1
    else:
        joint_objective.call_count = 1
    
    if joint_objective.call_count <= 5:  # Print first 5 calls for debugging
        print(f"Call {joint_objective.call_count}: Testing parameters {params_vector[:6]}")  # Show first 6 params
    try:
        # Unpack parameters based on mechanism - using ratio-based approach
        if mechanism == 'time_varying_k':
            n2, N2, k_1, k_max, r21, r23, R21, R23, alpha, beta_k, beta2_k = params_vector
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
            n2, N2, k_1, k_max, r21, r23, R21, R23, burst_size, alpha, beta_k, beta2_k = params_vector
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
            n2, N2, k_1, k_max, r21, r23, R21, R23, n_inner, alpha, beta_k, beta2_k = params_vector
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
            n2, N2, k_1, k_max, r21, r23, R21, R23, burst_size, n_inner, alpha, beta_k, beta2_k = params_vector
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
            n2, N2, k_1, k_max, r21, r23, R21, R23, burst_size, alpha, beta_k, beta2_k = params_vector
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
        dataset_weights = {'wildtype': 1.0, 'threshold': 1.0, 'degrate': 1.0, 'degrateAPC': 1.0}
        
        for dataset_name, data_dict in datasets.items():
            # Apply mutant-specific modifications
            params, n0_list = apply_mutant_params(
                base_params, dataset_name, alpha, beta_k, beta2_k
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
            
            # Calculate likelihoods
            nll_12 = calculate_likelihood(exp_delta_t12, sim_delta_t12)
            nll_32 = calculate_likelihood(exp_delta_t32, sim_delta_t32)
            
            # Check for penalty values in likelihood calculation
            if nll_12 >= 1e6 or nll_32 >= 1e6:
                print(f"High likelihood penalty for dataset {dataset_name}: nll_12={nll_12:.1f}, nll_32={nll_32:.1f}")
                return 1e6
            
            # Weight and add to total
            weight = dataset_weights.get(dataset_name, 1.0)
            total_nll += weight * (nll_12 + nll_32)
        
        return total_nll
    
    except Exception as e:
        print(f"Objective function error: {e}")
        return 1e6


def get_parameter_bounds(mechanism):
    """
    Get parameter bounds for optimization.
    
    Args:
        mechanism (str): Mechanism name
    
    Returns:
        list: List of (min, max) bounds for each parameter
    """
    # Common bounds - using ratio-based approach like MoMOptimization_join.py
    bounds = [
        (3, 50),      # n2
        (100, 500),   # N2
        (0.0001, 0.01),  # k_1
        (0.01, 0.2),     # k_max
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
    
    # Mutant parameter bounds
    bounds.extend([
        (0.1, 1.0),   # alpha
        (0.1, 1.0),   # beta_k
        (0.1, 1.0),   # beta2_k
    ])
    
    return bounds


def run_optimization(mechanism, datasets, max_iterations=300, num_simulations=500):
    """
    Run joint optimization for all datasets.
    
    Args:
        mechanism (str): Mechanism name
        datasets (dict): Experimental datasets
        max_iterations (int): Maximum iterations for optimization
    
    Returns:
        dict: Optimization results
    """
    print(f"\n=== Joint Optimization for {mechanism.upper()} ===")
    print(f"Datasets: {list(datasets.keys())}")
    print(f"Max iterations: {max_iterations}")
    
    # Get parameter bounds
    bounds = get_parameter_bounds(mechanism)
    
    print(f"Optimizing {len(bounds)} parameters...")
    print("Running differential evolution...")
    
    # Global optimization
    result = differential_evolution(
        joint_objective,
        bounds,
        args=(mechanism, datasets, num_simulations),
        maxiter=max_iterations,
        popsize=15,
        seed=42,
        disp=True,
        workers=1  # Use single worker to avoid multiprocessing issues
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
        param_names = ['n2', 'N2', 'k_1', 'k_max', 'r21', 'r23', 'R21', 'R23', 'alpha', 'beta_k', 'beta2_k']
    elif mechanism == 'time_varying_k_fixed_burst':
        param_names = ['n2', 'N2', 'k_1', 'k_max', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'alpha', 'beta_k', 'beta2_k']
    elif mechanism == 'time_varying_k_feedback_onion':
        param_names = ['n2', 'N2', 'k_1', 'k_max', 'r21', 'r23', 'R21', 'R23', 'n_inner', 'alpha', 'beta_k', 'beta2_k']
    elif mechanism == 'time_varying_k_combined':
        param_names = ['n2', 'N2', 'k_1', 'k_max', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'n_inner', 'alpha', 'beta_k', 'beta2_k']
    elif mechanism == 'time_varying_k_burst_onion':
        param_names = ['n2', 'N2', 'k_1', 'k_max', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'alpha', 'beta_k', 'beta2_k']
    
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
    print(f"  Rates: k_1={param_dict['k_1']:.6f}, k_max={param_dict['k_max']:.4f}")
    
    if 'burst_size' in param_dict:
        print(f"           burst_size={param_dict['burst_size']:.1f}")
    if 'n_inner' in param_dict:
        print(f"           n_inner={param_dict['n_inner']:.1f}")
    if 'burst_size' in param_dict and 'n_inner' in param_dict:
        print(f"           Combined mechanism: burst_size={param_dict['burst_size']:.1f}, n_inner={param_dict['n_inner']:.1f}")
    
    print(f"  Mutants: alpha={param_dict['alpha']:.3f}, beta_k={param_dict['beta_k']:.3f}, beta2_k={param_dict['beta2_k']:.3f}")
    
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
                                      random_seed=42):
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
    print(f"Datasets: {list(datasets.keys())}")
    print(f"Max iterations: {max_iterations}")
    
    # Analyze dataset sizes and determine target sample size
    size_analysis = analyze_dataset_sizes(datasets)
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
              target_sample_size, num_bootstrap_samples, random_seed),
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
        param_names = ['n2', 'N2', 'k_1', 'k_max', 'r21', 'r23', 'R21', 'R23', 'alpha', 'beta_k', 'beta2_k']
    elif mechanism == 'time_varying_k_fixed_burst':
        param_names = ['n2', 'N2', 'k_1', 'k_max', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'alpha', 'beta_k', 'beta2_k']
    elif mechanism == 'time_varying_k_feedback_onion':
        param_names = ['n2', 'N2', 'k_1', 'k_max', 'r21', 'r23', 'R21', 'R23', 'n_inner', 'alpha', 'beta_k', 'beta2_k']
    elif mechanism == 'time_varying_k_combined':
        param_names = ['n2', 'N2', 'k_1', 'k_max', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'n_inner', 'alpha', 'beta_k', 'beta2_k']
    elif mechanism == 'time_varying_k_burst_onion':
        param_names = ['n2', 'N2', 'k_1', 'k_max', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'alpha', 'beta_k', 'beta2_k']
    
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
    
    print(f"  Mutants: alpha={param_dict['alpha']:.3f}, beta_k={param_dict['beta_k']:.3f}, beta2_k={param_dict['beta2_k']:.3f}")
    
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


def save_results(mechanism, results, filename=None):
    """
    Save optimization results to file.
    
    Args:
        mechanism (str): Mechanism name
        results (dict): Optimization results
        filename (str): Output filename (optional)
    """
    if not results['success']:
        print("Cannot save results - optimization failed")
        return
    
    # Determine filename based on bootstrap method if applicable
    if filename is None:
        if 'bootstrap_method' in results:
            bootstrap_suffix = f"_{results['bootstrap_method']}"
            filename = f"simulation_optimized_parameters_{mechanism}{bootstrap_suffix}.txt"
        else:
            filename = f"simulation_optimized_parameters_{mechanism}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"Simulation-based Optimization Results\n")
        f.write(f"Mechanism: {mechanism}\n")
        
        # Add bootstrap information if available
        if 'bootstrap_method' in results:
            f.write(f"Bootstrap Method: {results['bootstrap_method']}\n")
            f.write(f"Target Sample Size: {results['target_sample_size']}\n")
            f.write(f"Number of Bootstrap Samples: {results['num_bootstrap_samples']}\n")
        
        f.write(f"Negative Log-Likelihood: {results['nll']:.6f}\n")
        f.write(f"Converged: {results.get('converged', 'Unknown')}\n")
        f.write(f"Status: {results.get('message', 'No message')}\n")
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
    Main optimization routine with bootstrapping demonstration.
    """
    max_iterations = 20  # number of iterations for testing
    num_simulations = 30  # Number of simulations per evaluation
    
    print("Simulation-based Optimization for Time-Varying Mechanisms")
    print("=" * 60)
    
    # Load experimental data
    datasets = load_experimental_data()
    if not datasets:
        print("Error: No datasets loaded!")
        return
    
    # Test mechanisms
    mechanisms = ['time_varying_k', 'time_varying_k_fixed_burst', 'time_varying_k_feedback_onion', 'time_varying_k_combined', 'time_varying_k_burst_onion']
    mechanism = mechanisms[4]  # Test the new burst_onion mechanism
    
    print(f"\n{'='*60}")
    print("COMPARISON: Standard vs. Bootstrapping Methods")
    print(f"{'='*60}")
    
    try:
        # 1. Standard optimization (for comparison)
        print(f"\n{'-'*40}")
        print("1. STANDARD OPTIMIZATION")
        print(f"{'-'*40}")
        standard_results = run_optimization(mechanism, datasets, max_iterations, num_simulations)
        save_results(mechanism, standard_results, f"simulation_optimized_parameters_{mechanism}_standard.txt")
        
        # 2. Weighted likelihood optimization
        print(f"\n{'-'*40}")
        print("2. WEIGHTED LIKELIHOOD OPTIMIZATION")
        print(f"{'-'*40}")
        weighted_results = run_optimization_with_bootstrapping(
            mechanism, datasets, 
            max_iterations=max_iterations, 
            num_simulations=num_simulations,
            bootstrap_method='weighted',
            num_bootstrap_samples=50,  # Reduced for testing
            random_seed=42
        )
        save_results(mechanism, weighted_results)
        
        # 3. Bootstrap optimization
        print(f"\n{'-'*40}")
        print("3. BOOTSTRAP OPTIMIZATION")
        print(f"{'-'*40}")
        bootstrap_results = run_optimization_with_bootstrapping(
            mechanism, datasets, 
            max_iterations=max_iterations, 
            num_simulations=num_simulations,
            bootstrap_method='bootstrap',
            num_bootstrap_samples=50,  # Reduced for testing
            random_seed=42
        )
        save_results(mechanism, bootstrap_results)
        
        # Compare results
        print(f"\n{'='*60}")
        print("RESULTS COMPARISON")
        print(f"{'='*60}")
        print(f"Standard NLL:   {standard_results['nll']:.4f}")
        print(f"Weighted NLL:   {weighted_results['nll']:.4f}")
        print(f"Bootstrap NLL:  {bootstrap_results['nll']:.4f}")
        
        print(f"\nImprovement over standard:")
        if weighted_results['nll'] < standard_results['nll']:
            improvement = ((standard_results['nll'] - weighted_results['nll']) / standard_results['nll']) * 100
            print(f"Weighted method: {improvement:.2f}% better")
        else:
            print(f"Weighted method: {((weighted_results['nll'] - standard_results['nll']) / standard_results['nll']) * 100:.2f}% worse")
            
        if bootstrap_results['nll'] < standard_results['nll']:
            improvement = ((standard_results['nll'] - bootstrap_results['nll']) / standard_results['nll']) * 100
            print(f"Bootstrap method: {improvement:.2f}% better")
        else:
            print(f"Bootstrap method: {((bootstrap_results['nll'] - standard_results['nll']) / standard_results['nll']) * 100:.2f}% worse")
        
        print(f"\n{'-' * 60}")
        
    except Exception as e:
        print(f"Error during optimization comparison: {e}")
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
    main() 