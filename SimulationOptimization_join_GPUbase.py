#!/usr/bin/env python3
"""
Simulation-based optimization for chromosome segregation timing models (GPU-Accelerated).
Uses MultiMechanismSimulationTimevary for time-varying mechanisms.
Uses simulation_utils_gpu for GPU-accelerated simulations.
Joint optimization strategy: optimizes all parameters simultaneously across all datasets.

Key differences from MoM-based optimization:
- Uses simulation instead of analytical approximations
- Separase mutant affects k_max (beta * k_max)
- APC mutant affects k_1 (beta_APC * k_1)
- Other mutants follow the same pattern as before
"""

import os
# Force single-thread for numpy/scipy to avoid oversubscription when using multiprocessing
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import sys
import time
from scipy.optimize import differential_evolution
from simulation_utils import *
from Chromosomes_Theory import *

import warnings
warnings.filterwarnings('ignore')

# Try to import GPU simulation function
try:
    from simulation_utils_gpu import run_simulation_for_dataset_gpu
    GPU_AVAILABLE = True
    print("GPU acceleration available and enabled.")
except ImportError:
    run_simulation_for_dataset_gpu = None
    GPU_AVAILABLE = False
    print("GPU acceleration NOT available. Falling back to CPU.")

USE_GPU = True

MECHANISM_PARAM_NAMES = {
    'simple': ['n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23', 'alpha', 'beta_k'],
    'fixed_burst': ['n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'alpha', 'beta_k'],
    'feedback_onion': ['n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23', 'n_inner', 'alpha', 'beta_k'],
    'fixed_burst_feedback_onion': [
        'n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'n_inner', 'alpha', 'beta_k'
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
    'time_varying_k_burst_onion': [
        'n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23',
        'burst_size', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2'
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
    beta_tau = params.get('beta_tau')
    beta_tau2 = params.get('beta_tau2')
    
    return base_params, alpha, beta_k, beta_tau, beta_tau2


def joint_objective(params_vector, mechanism, datasets, num_simulations=500, selected_strains=None):
    """
    Joint objective function that handles both simple and time-varying mechanisms.
    
    Args:
        params_vector: Parameter vector (mechanism-specific)
        mechanism: Mechanism name
        datasets: Experimental data dictionary
        num_simulations: Number of simulations per evaluation
        selected_strains: Optional list of strain names to include in fitting
        
    Returns:
        float: Total negative log-likelihood
    """
    try:
        # Unpack parameters based on mechanism
        base_params, alpha, beta_k, beta_tau, beta_tau2 = unpack_mechanism_params(params_vector, mechanism)
        
        # Check constraints
        if base_params['n1'] >= base_params['N1'] or \
            base_params['n2'] >= base_params['N2'] or \
            base_params['n3'] >= base_params['N3']:
            # print(f"Constraint violation: {base_params}")
            return 1e6

        total_nll = 0
        
        # Loop over all datasets
        for dataset_name, data_dict in datasets.items():
            # Apply mutant modifications using unified helper function (works for both mechanism types)
            params, n0_list = apply_mutant_params(
                base_params, dataset_name, alpha, beta_k, beta_tau, beta_tau2
            )
            
            # Run simulations - GPU ACCELERATED
            if USE_GPU and GPU_AVAILABLE:
                sim_delta_t12, sim_delta_t32 = run_simulation_for_dataset_gpu(
                    mechanism, params, n0_list, num_simulations=num_simulations, max_time=2000.0
                )
            else:
                sim_delta_t12, sim_delta_t32 = run_simulation_for_dataset(
                    mechanism, params, n0_list, num_simulations
                )
            
            if sim_delta_t12 is None or sim_delta_t32 is None:
                print(f"Simulation failed (None returned) for dataset {dataset_name}")
                return 1e6
            
            # Calculate likelihood using KDE
            exp_data = {'delta_t12': data_dict['delta_t12'], 'delta_t32': data_dict['delta_t32']}
            sim_data = {'delta_t12': sim_delta_t12, 'delta_t32': sim_delta_t32}
            
            nll = calculate_likelihood(exp_data, sim_data)
            
            if nll >= 1e6:
                print(f"Likelihood penalty for dataset {dataset_name}")
                return 1e6
            
            total_nll += nll
        
        return total_nll
    
    except Exception as e:
        print(f"Exception in joint_objective: {e}")
        import traceback
        traceback.print_exc()
        return 1e6


def run_optimization(mechanism, datasets, max_iterations=500, num_simulations=500, selected_strains=None):
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
        use_parallel (bool): Whether to use parallel processing
        initial_guess (list, optional): Initial guess parameter vector. If None, uses random initialization.
                                       Example: [n2, N2, k_max, tau, r21, r23, R21, R23, alpha, beta_k, beta_tau, beta_tau2]
    
    Returns:
        dict: Optimization results
    """
    try:
        param_names = MECHANISM_PARAM_NAMES[mechanism]
    except KeyError:
        return {'success': False, 'message': f'Unknown mechanism: {mechanism}'}
    
    bounds = get_parameter_bounds(mechanism)
    
    print(f"\nOptimizing {mechanism} ({len(bounds)} parameters, {num_simulations} sims/eval)")
    print(f"\nOptimizing {mechanism} ({len(bounds)} parameters, {num_simulations} sims/eval)")
    
    # Setup multiprocessing pool
    pool = None
    try:
        if USE_GPU and GPU_AVAILABLE:
            print("Using GPU acceleration.")
            try:
                # Use 'spawn' for CUDA compatibility
                import torch.multiprocessing as mp
                ctx = mp.get_context('spawn')
                
                # Determine number of workers from SLURM environment or default
                slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
                if slurm_cpus:
                    num_workers = int(slurm_cpus)
                    print(f"Using {num_workers} workers (from SLURM_CPUS_PER_TASK).")
                else:
                    # Default backup if not running in SLURM or env var missing
                    num_workers = 8 
                    print(f"Using {num_workers} workers (default backup).")
                
                print(f"Initializing multiprocessing pool with {num_workers} workers (spawn method)...")
                pool = ctx.Pool(num_workers)
                workers = pool.map
            except Exception as e:
                print(f"Failed to initialize multiprocessing pool: {e}")
                print("Falling back to sequential execution.")
                workers = 1
        if not (USE_GPU and GPU_AVAILABLE):
            print("Using CPU (standard).")
            workers = -1
        
        opt_args = (mechanism, datasets, num_simulations, selected_strains)
        
        # Define Differential Evolution settings
        de_settings = {
            'maxiter': max_iterations,
            'popsize': 15,
            'workers': workers,
            'strategy': 'best1bin',
            'tol': 1e-4,
            'polish': True,
            'disp': True
        }
        
        print("\nDifferential Evolution Settings:")
        for key, value in de_settings.items():
            if key == 'workers' and not isinstance(value, int):
                 print(f"  {key}: pool.map (using multiprocessing pool)")
            else:
                 print(f"  {key}: {value}")
        sys.stdout.flush()

        result = differential_evolution(
            joint_objective,
            bounds,
            args=opt_args,
            #x0=initial_guess,
            **de_settings
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    
    params = result.x
    param_dict = dict(zip(param_names, params))
    
    status = "converged" if result.success else "not converged"
    print(f"Optimization {status}: NLL = {result.fun:.4f}")
    
    print("\nBest Fit Parameters:")
    for name, val in param_dict.items():
        print(f"  {name}: {val:.6f}")
    
    sys.stdout.flush()
    
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


def main():
    """
    Main optimization routine - now supports both simple and time-varying mechanisms.
    """
    max_iterations = 200
    num_simulations = 500
    
    datasets = load_experimental_data()
    if not datasets:
        print("Error: No datasets loaded!")
        return
    
    # Default mechanism
    mechanism = 'simple'  # Change as needed for different mechanisms
    
    # Check for command line argument
    if len(sys.argv) > 1:
        mechanism = sys.argv[1]
        print(f"Mechanism set from command line: {mechanism}")
    else:
        print(f"Using default mechanism: {mechanism}")
    
    try:
        results = run_optimization(
            mechanism, datasets,
            max_iterations=max_iterations,
            num_simulations=num_simulations,
            selected_strains=None
        )
        save_results(mechanism, results, selected_strains=None)
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()