#!/usr/bin/env python3
"""
Script to analyze the efficient number of simulations.
Runs N replicates for various simulation counts and plots:
1. Mean NLL vs Simulation Count
2. NLL Standard Deviation (Noise) vs Simulation Count

Usage: python AnalyzeSimulationEfficiency.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import time
from scipy.stats import sem

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SecondVersion'))
from simulation_utils import (
    load_experimental_data, 
    apply_mutant_params,
    calculate_likelihood,
    run_simulation_for_dataset
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

def analyze_efficiency(mechanism, params_file, simulation_counts_list, replicates=20):
    """
    Run analysis for list of simulation counts.
    """
    print(f"Loading parameters from {params_file}...")
    params = load_parameters(params_file)
    if params is None:
        return

    # Prepare base params
    base_keys = ['n1', 'n2', 'n3', 'N1', 'N2', 'N3', 'k', 'burst_size', 'n_inner']
    base_params = {k: params.get(k) for k in base_keys if k in params}
    
    exp_datasets = load_experimental_data()
    dataset_names = ['wildtype', 'threshold', 'degrade', 'degradeAPC', 'velcade']
    
    results = {
        'counts': [],
        'means': [],
        'stds': [],
        'sems': [],
        'times': []
    }
    
    print(f"\nStarting analysis with {replicates} replicates per count...")
    print(f"Counts to test: {simulation_counts_list}\n")
    
    for count in simulation_counts_list:
        print(f"Testing N={count}...", end='', flush=True)
        
        nll_values = []
        start_time = time.time()
        
        for i in range(replicates):
            total_nll = 0
            
            for dataset_name in dataset_names:
                if dataset_name not in exp_datasets: continue
                
                # Apply mutant parameters
                alpha = params.get('alpha', 1.0)
                beta_k = params.get('beta_k', 1.0)
                beta2_k = params.get('beta2_k', 1.0)
                beta3_k = params.get('beta3_k', 1.0)
                
                curr_beta = beta2_k if dataset_name == 'degradeAPC' else (beta3_k if dataset_name == 'velcade' else beta_k)
                
                mutant_params, n0_list = apply_mutant_params(base_params, dataset_name, alpha, curr_beta)
                
                # Threshold fix
                if dataset_name == 'threshold':
                    n1_th = max(base_params['n1'] * alpha, 1)
                    n2_th = max(base_params['n2'] * alpha, 1)
                    n3_th = max(base_params['n3'] * alpha, 1)
                    mutant_params['n1'], mutant_params['n2'], mutant_params['n3'] = n1_th, n2_th, n3_th
                    n0_list = [n1_th, n2_th, n3_th]
                
                t12, t32 = run_simulation_for_dataset(mechanism, mutant_params, n0_list, num_simulations=count)
                
                if t12 is None:
                    total_nll = float('inf')
                    break
                
                nll = calculate_likelihood(exp_datasets[dataset_name], {'delta_t12': t12, 'delta_t32': t32})
                total_nll += nll
            
            nll_values.append(total_nll)
            # print(f".", end='', flush=True)
        
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
        
        print(f" Done. Mean={mean_val:.2f}, Std={std_val:.2f}, Time/Run={avg_time:.2f}s")

    # Plotting
    plot_efficiency_results(results)

def plot_efficiency_results(results, result_file=None):
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
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Mean NLL vs Count
    ax1.errorbar(counts, means, yerr=stds, fmt='o-', capsize=5, ecolor='red')
    ax1.set_xlabel('Number of Simulations')
    ax1.set_ylabel('Mean NLL')
    ax1.set_title('NLL Mean vs Simulation Count')
    ax1.grid(True, alpha=0.3)
    
    # 2. Std Dev vs Count (Noise Analysis)
    ax2.plot(counts, stds, 'o-', color='orange')
    ax2.set_xlabel('Number of Simulations')
    ax2.set_ylabel('NLL Standard Deviation')
    ax2.set_title('NLL Noise (Std Dev) vs Simulation Count')
    ax2.grid(True, alpha=0.3)
    
    # # Add theoretical 1/sqrt(N) scaling line for comparison
    # # approximate scaling: std ~ k / sqrt(N)
    # # Fit to first point
    # if len(counts) > 0:
    #     k_scaling = stds[0] * np.sqrt(counts[0])
    #     theoretical_std = [k_scaling / np.sqrt(n) for n in counts]
    #     ax2.plot(counts, theoretical_std, 'k--', alpha=0.5, label='1/sqrt(N) scaling')
    #     ax2.legend()
    
    # 3. Time vs Count
    ax3.plot(counts, times, 'o-', color='green')
    ax3.set_xlabel('Number of Simulations')
    ax3.set_ylabel('Time per Run (s)')
    ax3.set_title('Computational Cost')
    ax3.grid(True, alpha=0.3)
    
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
    mechanism = 'simple'
    params_file = 'optimized_parameters_simple_join_toTest.txt'
    replicates = 10
    #result_file = 'simulation_efficiency_results.csv'  # 'simulation_efficiency_results.csv'  # Set to None to run analysis
    
    #plot_efficiency_results(results=None, result_file=result_file)
    # Define counts to test
    simulation_counts = [500, 1000, 2000, 3000] # Adjusted max for runtime
    
    if not os.path.exists(params_file):
        print(f"File not found: {params_file}")
    else:
        analyze_efficiency(mechanism, params_file, simulation_counts, replicates=replicates)
