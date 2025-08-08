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


def run_simulation_for_dataset(mechanism, params, n0_list, num_simulations=10):
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


def joint_objective(params_vector, mechanism, datasets, num_simulations=10):
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
    try:
        # Unpack parameters based on mechanism
        if mechanism == 'time_varying_k':
            n1, n2, n3, N1, N2, N3, k_1, k_max, alpha, beta_k, beta2_k = params_vector
            base_params = {
                'n1': n1, 'n2': n2, 'n3': n3,
                'N1': N1, 'N2': N2, 'N3': N3,
                'k_1': k_1, 'k_max': k_max
            }
        elif mechanism == 'time_varying_k_fixed_burst':
            n1, n2, n3, N1, N2, N3, k_1, k_max, burst_size, alpha, beta_k, beta2_k = params_vector
            base_params = {
                'n1': n1, 'n2': n2, 'n3': n3,
                'N1': N1, 'N2': N2, 'N3': N3,
                'k_1': k_1, 'k_max': k_max, 'burst_size': burst_size
            }
        elif mechanism == 'time_varying_k_feedback_onion':
            n1, n2, n3, N1, N2, N3, k_1, k_max, n_inner, alpha, beta_k, beta2_k = params_vector
            base_params = {
                'n1': n1, 'n2': n2, 'n3': n3,
                'N1': N1, 'N2': N2, 'N3': N3,
                'k_1': k_1, 'k_max': k_max, 'n_inner': n_inner
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
                return 1e6
            
            # Extract experimental data
            exp_delta_t12 = data_dict['delta_t12']
            exp_delta_t32 = data_dict['delta_t32']
            
            # Calculate likelihoods
            nll_12 = calculate_likelihood(exp_delta_t12, sim_delta_t12)
            nll_32 = calculate_likelihood(exp_delta_t32, sim_delta_t32)
            
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
    # Common bounds
    bounds = [
        (1, 20),      # n1
        (1, 30),      # n2  
        (1, 50),      # n3
        (50, 500),    # N1
        (100, 800),   # N2
        (200, 1200),  # N3
        (0.0001, 0.01),  # k_1
        (0.01, 0.2),     # k_max
    ]
    
    # Mechanism-specific bounds
    if mechanism == 'time_varying_k_fixed_burst':
        bounds.append((1, 20))  # burst_size
    elif mechanism == 'time_varying_k_feedback_onion':
        bounds.append((10, 50))  # n_inner
    
    # Mutant parameter bounds
    bounds.extend([
        (0.1, 1.0),   # alpha
        (0.1, 1.0),   # beta_k
        (0.1, 1.0),   # beta2_k
    ])
    
    return bounds


def run_optimization(mechanism, datasets, max_iterations=500):
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
        args=(mechanism, datasets),
        maxiter=max_iterations,
        popsize=15,
        seed=42,
        disp=True,
        workers=1  # Use single worker to avoid multiprocessing issues
    )
    
    if result.success:
        print(f"✅ Optimization converged!")
        print(f"Best negative log-likelihood: {result.fun:.4f}")
        
        # Unpack and display results
        params = result.x
        if mechanism == 'time_varying_k':
            param_names = ['n1', 'n2', 'n3', 'N1', 'N2', 'N3', 'k_1', 'k_max', 'alpha', 'beta_k', 'beta2_k']
        elif mechanism == 'time_varying_k_fixed_burst':
            param_names = ['n1', 'n2', 'n3', 'N1', 'N2', 'N3', 'k_1', 'k_max', 'burst_size', 'alpha', 'beta_k', 'beta2_k']
        elif mechanism == 'time_varying_k_feedback_onion':
            param_names = ['n1', 'n2', 'n3', 'N1', 'N2', 'N3', 'k_1', 'k_max', 'n_inner', 'alpha', 'beta_k', 'beta2_k']
        
        param_dict = dict(zip(param_names, params))
        
        print("\nOptimized Parameters:")
        print(f"  Wildtype: n1={param_dict['n1']:.1f}, n2={param_dict['n2']:.1f}, n3={param_dict['n3']:.1f}")
        print(f"           N1={param_dict['N1']:.1f}, N2={param_dict['N2']:.1f}, N3={param_dict['N3']:.1f}")
        print(f"           k_1={param_dict['k_1']:.6f}, k_max={param_dict['k_max']:.4f}")
        
        if 'burst_size' in param_dict:
            print(f"           burst_size={param_dict['burst_size']:.1f}")
        if 'n_inner' in param_dict:
            print(f"           n_inner={param_dict['n_inner']:.1f}")
        
        print(f"  Mutants: alpha={param_dict['alpha']:.3f}, beta_k={param_dict['beta_k']:.3f}, beta2_k={param_dict['beta2_k']:.3f}")
        
        return {
            'success': True,
            'params': param_dict,
            'nll': result.fun,
            'result': result
        }
    else:
        print(f"❌ Optimization failed: {result.message}")
        return {
            'success': False,
            'message': result.message,
            'result': result
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
    
    if filename is None:
        filename = f"simulation_optimized_parameters_{mechanism}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"Simulation-based Optimization Results\n")
        f.write(f"Mechanism: {mechanism}\n")
        f.write(f"Negative Log-Likelihood: {results['nll']:.6f}\n")
        f.write(f"Datasets: wildtype, threshold, degrate, degrateAPC\n\n")
        
        f.write("Optimized Parameters:\n")
        for param, value in results['params'].items():
            f.write(f"{param} = {value:.6f}\n")
    
    print(f"Results saved to: {filename}")


def main():
    """
    Main optimization routine.
    """
    print("Simulation-based Optimization for Time-Varying Mechanisms")
    print("=" * 60)
    
    # Load experimental data
    datasets = load_experimental_data()
    if not datasets:
        print("Error: No datasets loaded!")
        return
    
    # Test mechanisms
    mechanisms = ['time_varying_k', 'time_varying_k_fixed_burst', 'time_varying_k_feedback_onion']
    
    for mechanism in mechanisms:
        try:
            results = run_optimization(mechanism, datasets, max_iterations=500)  # Reduced for testing
            
            if results['success']:
                save_results(mechanism, results)
            
            print(f"\n{'-' * 60}")
            
        except Exception as e:
            print(f"Error optimizing {mechanism}: {e}")
            continue
    
    print("Optimization complete!")


if __name__ == "__main__":
    main() 