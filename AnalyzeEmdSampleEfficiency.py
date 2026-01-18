#!/usr/bin/env python3
"""
Script to analyze the efficient number of simulations for EMD (Earth Mover's Distance).
Runs N replicates for various simulation counts and plots:
1. Mean EMD vs Simulation Count (for Good vs Bad parameters)
2. EMD Standard Deviation (Noise) vs Simulation Count

This helps determine the sample size required to:
- Distinguish between good and bad parameters (Signal-to-Noise Ratio).
- Achieve stable EMD estimates.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import time
from scipy.stats import sem
from multiprocessing import Pool, cpu_count
from numpy.random import SeedSequence

# Import simulation utilities
from simulation_utils import (
    load_experimental_data, 
    apply_mutant_params,
    calculate_emd,
    run_simulation_for_dataset,
    calculate_k1_from_params,
    load_parameters
)


def perturb_parameters(params, mechanism, perturbation_factor=0.2):
    """
    Create a 'bad' parameter set by perturbing values.
    
    Args:
        params: Original 'good' parameters
        mechanism: Mechanism name
        perturbation_factor: Fraction to perturb (0.2 = +/- 20%)
        
    Returns:
        dict: Perturbed parameters
    """
    bad_params = params.copy()
    
    # Keys to perturb (avoid perturbing integers like n_inner if possible, or round them)
    # We focus on the core rate/threshold parameters that drive the distribution
    keys_to_perturb = ['n2', 'N2', 'k', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23']
    
    np.random.seed(42) # Fixed seed for reproducible 'bad' params
    
    for key in keys_to_perturb:
        if key in bad_params:
            # Randomly perturb up or down
            change = 1.0 + np.random.uniform(-perturbation_factor, perturbation_factor)
            # Ensure we shift it away from 1.0 significantly to be "bad"
            if 0.95 < change < 1.05:
                change = 1.2 if np.random.random() > 0.5 else 0.8
            
            bad_params[key] *= change
    
    # Recalculate derived
    if 'r21' in bad_params:
        bad_params['n1'] = max(bad_params['r21'] * bad_params['n2'], 1)
        bad_params['n3'] = max(bad_params['r23'] * bad_params['n2'], 1)
        bad_params['N1'] = max(bad_params['R21'] * bad_params['N2'], 1)
        bad_params['N3'] = max(bad_params['R23'] * bad_params['N2'], 1)
    
    if 'k_max' in bad_params and 'tau' in bad_params:
         bad_params['k_1'] = calculate_k1_from_params(bad_params)
         
    return bad_params

def _compute_single_replicate(replicate_idx, seed, mechanism, base_params, params, exp_datasets, dataset_names, count):
    """Compute EMD for a single replicate."""
    np.random.seed(seed)
    total_emd = 0
    
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
        
        # Use EMD instead of Likelihood
        emd = calculate_emd(exp_datasets[dataset_name], {'delta_t12': t12, 'delta_t32': t32})
        total_emd += emd
    
    return total_emd

def analyze_emd_efficiency(mechanism, params_file, simulation_counts_list, replicates=20, n_workers=None):
    """
    Run analysis for list of simulation counts.
    """
    print(f"Loading parameters from {params_file}...")
    good_params = load_parameters(params_file)
    if good_params is None:
        return

    # Create Bad Parameters
    bad_params = perturb_parameters(good_params, mechanism)
    
    # Prepare base params for each
    base_keys = ['n1', 'n2', 'n3', 'N1', 'N2', 'N3', 'k', 'k_max', 'tau', 'k_1', 'burst_size', 'n_inner']
    
    good_base_params = {k: good_params.get(k) for k in base_keys if k in good_params}
    if 'k_max' in good_base_params and 'tau' in good_base_params and 'k_1' not in good_base_params:
        good_base_params['k_1'] = good_base_params['k_max'] / good_base_params['tau']

    bad_base_params = {k: bad_params.get(k) for k in base_keys if k in bad_params}
    if 'k_max' in bad_base_params and 'tau' in bad_base_params and 'k_1' not in bad_base_params:
        bad_base_params['k_1'] = bad_base_params['k_max'] / bad_base_params['tau']
    
    exp_datasets = load_experimental_data()
    dataset_names = ['wildtype', 'threshold', 'degrade', 'degradeAPC', 'velcade']
    
    results = {
        'counts': [],
        'good_means': [], 'good_stds': [], 'good_sems': [],
        'bad_means': [], 'bad_stds': [], 'bad_sems': [],
        'times': []
    }
    
    if n_workers is None:
        n_workers = min(cpu_count(), replicates)
        
    print(f"\nStarting EMD analysis with {replicates} replicates per count...")
    print(f"Counts: {simulation_counts_list}")
    
    for count in simulation_counts_list:
        print(f"Testing N={count}...", end='', flush=True)
        start_time = time.time()
        
        ss = SeedSequence()
        child_seeds = ss.spawn(replicates)
        
        # --- Run Good Params ---
        worker_args_good = [
            (i, int(child_seeds[i].generate_state(1)[0]), mechanism, good_base_params, 
             good_params, exp_datasets, dataset_names, count)
            for i in range(replicates)
        ]
        
        with Pool(processes=n_workers) as pool:
            good_emds = pool.starmap(_compute_single_replicate, worker_args_good)
        
        # --- Run Bad Params ---
        # Use new seeds for bad params to ensure independence
        child_seeds_bad = ss.spawn(replicates) 
        worker_args_bad = [
            (i, int(child_seeds_bad[i].generate_state(1)[0]), mechanism, bad_base_params, 
             bad_params, exp_datasets, dataset_names, count)
            for i in range(replicates)
        ]
        
        with Pool(processes=n_workers) as pool:
            bad_emds = pool.starmap(_compute_single_replicate, worker_args_bad)
            
        elapsed = time.time() - start_time
        avg_time = elapsed / (replicates * 2) # *2 because we ran good and bad
        
        # Stats
        results['counts'].append(count)
        results['times'].append(avg_time)
        
        good_arr = np.array(good_emds)
        results['good_means'].append(np.mean(good_arr))
        results['good_stds'].append(np.std(good_arr))
        results['good_sems'].append(sem(good_arr))
        
        bad_arr = np.array(bad_emds)
        results['bad_means'].append(np.mean(bad_arr))
        results['bad_stds'].append(np.std(bad_arr))
        results['bad_sems'].append(sem(bad_arr))
        
        print(f" Done. (Time/Run: {avg_time:.3f}s)")
        print(f"  Good EMD: {np.mean(good_arr):.2f} ± {np.std(good_arr):.2f}")
        print(f"  Bad EMD:  {np.mean(bad_arr):.2f} ± {np.std(bad_arr):.2f}")

    plot_results(results, mechanism)
    
    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv('emd_sample_efficiency.csv', index=False)
    print("Results saved to emd_sample_efficiency.csv")

def plot_results(results, mechanism):
    counts = results['counts']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Mean EMD (Good vs Bad)
    ax1.errorbar(counts, results['good_means'], yerr=results['good_stds'], 
                 fmt='o-', label='Optimized Params', capsize=5, color='blue')
    ax1.errorbar(counts, results['bad_means'], yerr=results['bad_stds'], 
                 fmt='s-', label='Perturbed Params', capsize=5, color='red')
    
    ax1.set_xlabel('Simulation Count (N)')
    ax1.set_ylabel('Total EMD (Sum of Wasserstein Distances)')
    ax1.set_title(f'EMD Stability: Good vs Bad Parameters ({mechanism})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Plot 2: Noise (Std Dev) vs N
    ax2.plot(counts, results['good_stds'], 'o-', label='Good Param Noise', color='blue')
    ax2.plot(counts, results['bad_stds'], 's-', label='Bad Param Noise', color='red')
    
    ax2.set_xlabel('Simulation Count (N)')
    ax2.set_ylabel('Standard Deviation of EMD')
    ax2.set_title('EMD Noise Level vs Sample Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    # Add values
    for i, (cnt, std) in enumerate(zip(counts, results['good_stds'])):
        ax2.annotate(f'{std:.2f}', (cnt, std), xytext=(0, -15), textcoords='offset points', ha='center', color='blue', fontsize=8)

    plt.tight_layout()
    plt.savefig('emd_sample_efficiency_analysis.png')
    print("Plot saved to emd_sample_efficiency_analysis.png")

if __name__ == "__main__":
    # Configuration
    mechanism = 'simple'
    params_file = 'simulation_optimized_parameters_simple.txt'
    
    # Test range
    counts = [500, 1000, 2000, 5000, 10000, 20000, 50000]
    replicates = 20
    n_workers = 8
    
    if not os.path.exists(params_file):
        print(f"Error: {params_file} not found.")
        # Try finding any parameter file
        print("Looking for other parameter files...")
        import glob
        files = glob.glob("simulation_optimized_parameters_*.txt")
        if files:
            params_file = files[0]
            if 'fixed_burst' in params_file: mechanism = 'fixed_burst'
            elif 'simple' in params_file: mechanism = 'simple'
            print(f"Using found file: {params_file} (mech: {mechanism})")
        else:
            sys.exit(1)
            
    analyze_emd_efficiency(mechanism, params_file, counts, replicates, n_workers)
