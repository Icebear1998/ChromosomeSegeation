#!/usr/bin/env python3
"""
Script to analyze the efficient number of simulations.
Runs N replicates for various simulation counts and plots:
1. Mean NLL vs Simulation Count
2. NLL Standard Deviation (Noise) vs Simulation Count

NOTE: This script automatically uses Fast simulation methods where available:
  - FastBetaSimulation for: simple, fixed_burst, time_varying_k, time_varying_k_fixed_burst
  - FastFeedbackSimulation for: feedback_onion, fixed_burst_feedback_onion, 
    time_varying_k_feedback_onion, time_varying_k_combined
  - Falls back to Gillespie for other mechanisms
This is handled automatically by run_simulation_for_dataset() in simulation_utils.

Usage: python AnalyzeSimulationEfficiency.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import time
from scipy.stats import sem
from multiprocessing import Pool, cpu_count
from functools import partial
from numpy.random import SeedSequence, default_rng

# Import simulation utilities (includes automatic Fast method dispatch)
from simulation_utils import (
    load_experimental_data, 
    apply_mutant_params,
    calculate_likelihood,
    run_simulation_for_dataset,  # Automatically uses Fast methods where available
    set_kde_bandwidth  # For bandwidth configuration
)


# Reuse load_parameters from SimulationNumberComparison.py logic
def load_parameters(filename):
    """Load parameters from optimization results file."""
    params = {}
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if ':' in line:
                parts = line.split(':', 1)
                try:
                    key = parts[0].strip()
                    val = float(parts[1].strip())
                    params[key] = val
                except ValueError:
                    pass
            elif '=' in line:
                parts = line.split('=', 1)
                try:
                    key = parts[0].strip()
                    val = float(parts[1].strip())
                    params[key] = val
                except ValueError:
                    pass
        
        if 'r21' in params and 'n1' not in params:
            params['n1'] = max(params['r21'] * params['n2'], 1)
        if 'r23' in params and 'n3' not in params:
             params['n3'] = max(params['r23'] * params['n2'], 1)
        if 'R21' in params and 'N1' not in params:
             params['N1'] = max(params['R21'] * params['N2'], 1)
        if 'R23' in params and 'N3' not in params:
             params['N3'] = max(params['R23'] * params['N2'], 1)
             
        return params
    except Exception as e:
        print(f"Error loading parameters: {e}")
        return None

def analyze_efficiency(mechanism, params_file, simulation_counts_list, replicates=20, n_workers=None):
    """
    Run analysis for list of simulation counts with parallel replicates.
    
    Args:
        mechanism: Mechanism name
        params_file: Path to parameter file
        simulation_counts_list: List of simulation counts to test
        replicates: Number of replicates per count
        n_workers: Number of parallel workers (default: CPU count)
    """
    print(f"Loading parameters from {params_file}...")
    params = load_parameters(params_file)
    if params is None:
        return

    # Prepare base params - include both simple and time-varying parameters
    base_keys = ['n1', 'n2', 'n3', 'N1', 'N2', 'N3', 'k', 'k_max', 'tau', 'k_1', 'burst_size', 'n_inner']
    base_params = {k: params.get(k) for k in base_keys if k in params}
    
    # Calculate k_1 if we have k_max and tau but not k_1 (for time-varying mechanisms)
    if 'k_max' in base_params and 'tau' in base_params and 'k_1' not in base_params:
        base_params['k_1'] = base_params['k_max'] / base_params['tau']
    
    exp_datasets = load_experimental_data()
    dataset_names = ['wildtype', 'threshold', 'degrade', 'degradeAPC', 'velcade']

    
    results = {
        'counts': [],
        'means': [],
        'stds': [],
        'sems': [],
        'times': []
    }
    
    # Set up parallel workers
    if n_workers is None:
        n_workers = min(cpu_count(), replicates)
    print(f"\nStarting analysis with {replicates} replicates per count using {n_workers} workers...")
    print(f"Counts to test: {simulation_counts_list}\n")
    
    for count in simulation_counts_list:
        print(f"Testing N={count}...", end='', flush=True)
        
        start_time = time.time()
        
        # Generate unique seeds using SeedSequence for proper parallel RNG
        # This guarantees statistically independent random streams
        ss = SeedSequence()
        child_seeds = ss.spawn(replicates)
        
        # Create worker arguments with unique seeds
        worker_args = [
            (i, int(child_seeds[i].generate_state(1)[0]), mechanism, base_params, 
             params, exp_datasets, dataset_names, count)
            for i in range(replicates)
        ]
        
        # Run replicates in parallel
        with Pool(processes=n_workers) as pool:
            nll_values = pool.starmap(_compute_single_replicate, worker_args)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / replicates
        
        nll_arr = np.array(nll_values)
        mean_val = np.mean(nll_arr)
        std_val = np.std(nll_arr)
        sem_val = sem(nll_arr)
        
        results['counts'].append(count)
        results['means'].append(mean_val)
        results['stds'].append(std_val)
        results['sems'].append(sem_val)
        results['times'].append(avg_time)
        
        print(f" Done. Mean={mean_val:.2f}, Std={std_val:.2f}, Time/Run={avg_time:.2f}s, Total={elapsed:.1f}s")

    # Plotting
    plot_efficiency_results(results, mechanism=mechanism)


def _compute_single_replicate(replicate_idx, seed, mechanism, base_params, params, exp_datasets, dataset_names, count):
    """
    Compute NLL for a single replicate. Used for parallel processing.
    
    Args:
        replicate_idx: Index of this replicate
        seed: Unique seed from SeedSequence for this worker
        mechanism: Mechanism name
        base_params: Base parameters dict
        params: Full parameters dict
        exp_datasets: Experimental datasets
        dataset_names: List of dataset names
        count: Number of simulations
    """
    # Set unique seed for this worker (from SeedSequence)
    np.random.seed(seed)
    
    total_nll = 0
    
    for dataset_name in dataset_names:
        if dataset_name not in exp_datasets:
            continue
        
        # Apply mutant parameters
        alpha = params.get('alpha', 1.0)
        beta_k = params.get('beta_k', 1.0)
        beta_tau = params.get('beta_tau', None)
        beta_tau2 = params.get('beta_tau2', None)
        
        mutant_params, n0_list = apply_mutant_params(
            base_params, dataset_name, alpha, beta_k, beta_tau, beta_tau2
        )
        
        # Threshold fix
        if dataset_name == 'threshold':
            n1_th = max(base_params['n1'] * alpha, 1)
            n2_th = max(base_params['n2'] * alpha, 1)
            n3_th = max(base_params['n3'] * alpha, 1)
            mutant_params['n1'], mutant_params['n2'], mutant_params['n3'] = n1_th, n2_th, n3_th
            n0_list = [n1_th, n2_th, n3_th]
        
        t12, t32 = run_simulation_for_dataset(mechanism, mutant_params, n0_list, num_simulations=count)
        
        if t12 is None:
            return float('inf')
        
        nll = calculate_likelihood(exp_datasets[dataset_name], {'delta_t12': t12, 'delta_t32': t32})
        total_nll += nll
    
    return total_nll

def plot_efficiency_results(results, mechanism='unknown', result_file=None):
    """Generate plots for efficiency analysis."""
    if result_file is not None:
        df = pd.read_csv(result_file)
        results = {
            'counts': df['counts'].tolist(),
            'means': df['means'].tolist(),
            'stds': df['stds'].tolist(),
            'sems': df['sems'].tolist(),
            'times': df['times'].tolist()
        }

    counts = results['counts']
    means = results['means']
    stds = results['stds']
    times = results['times']
    
    # Changed from 3 subplots to 2 subplots (removed computational cost)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Mean NLL vs Count
    ax1.errorbar(counts, means, yerr=stds, fmt='o-', capsize=5, ecolor='red')
    ax1.set_xlabel('Number of Simulations')
    ax1.set_ylabel('Mean NLL')
    ax1.set_title(f'NLL Mean vs Simulation Count ({mechanism})')
    ax1.grid(True, alpha=0.3)
    
    # 2. Std Dev vs Count (Noise Analysis) with value labels
    ax2.plot(counts, stds, 'o-', color='orange')
    ax2.set_xlabel('Number of Simulations')
    ax2.set_ylabel('NLL Standard Deviation')
    ax2.set_title(f'NLL Noise (Std Dev) vs Simulation Count ({mechanism})')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels to each point on the NLL noise plot
    for i, (count, std) in enumerate(zip(counts, stds)):
        ax2.annotate(f'{std:.1f}', 
                    xy=(count, std), 
                    xytext=(0, 8),  # 8 points vertical offset
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    color='darkblue')
    
    # Removed the 3rd subplot (Computational Cost)
    
    plt.tight_layout()
    output_file = 'simulation_efficiency_analysis.png'
    plt.savefig(output_file)
    print(f"\nPlots saved to {output_file}")
    
    # Save CSV
    df = pd.DataFrame(results)
    csv_file = 'simulation_efficiency_results.csv'
    df.to_csv(csv_file, index=False)
    print(f"Results saved to {csv_file}")

if __name__ == "__main__":
    mechanism = 'simple'  # Change as needed
    params_file = 'simulation_optimized_parameters_simple.txt'
    replicates = 200
    n_workers = 48  # Adjust based on your CPU cores
    
    # KDE Bandwidth configuration
    # Options: 'scott' (adaptive, h = std * n^(-1/5)) or 'fixed' (constant bandwidth)
    bandwidth_method = 'fixed'     # 'scott' or 'fixed'
    fixed_bandwidth = 10.0         # Used when bandwidth_method='fixed'
    
    # Set bandwidth configuration
    set_kde_bandwidth(method=bandwidth_method, fixed_value=fixed_bandwidth)
    
    # Extended counts to observe NLL plateau effect
    simulation_counts = [2000, 5000, 10000, 20000, 40000, 60000, 80000, 100000]#, 150000, 200000, 300000, 500000]
    
    if not os.path.exists(params_file):
        print(f"File not found: {params_file}")
    else:
        analyze_efficiency(mechanism, params_file, simulation_counts, replicates=replicates, n_workers=n_workers)
