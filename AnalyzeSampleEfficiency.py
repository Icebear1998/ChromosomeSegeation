#!/usr/bin/env python3
"""
Script to analyze the efficient number of simulations for EMD (Earth Mover's Distance).
Uses k-fold cross-validation for various simulation counts and plots:
1. Train vs Val EMD to detect overfitting
2. Validation EMD Standard Deviation (Noise) vs Simulation Count

This helps determine the sample size required to:
- Achieve stable EMD estimates without overfitting (Train-Val gap).
- Optimize computational efficiency.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import time

# Import cross-validation and simulation utilities
from CrossValidation import create_folds, objective_function
from simulation_utils import (
    load_experimental_data,
    load_parameters,
    calculate_k1_from_params,
    get_parameter_names
)

plt.rcParams.update({'font.size': 14})

def convert_params_to_vector(params, mechanism):
    """Convert parameter dictionary to vector for objective_function."""
    # Strip _wfeedback suffix for base mechanism check
    base_mechanism = mechanism.replace('_wfeedback', '') if mechanism.endswith('_wfeedback') else mechanism
    
    if base_mechanism == 'time_varying_k':
        return [
            params['n2'], params['N2'], params['k_max'], params['tau'],
            params['r21'], params['r23'], params['R21'], params['R23'],
            params.get('alpha', 1.0), params.get('beta_k', 1.0),
            params.get('beta_tau', 1.0), params.get('beta_tau2', 0.0)
        ]
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}. Only time-varying mechanisms are supported.")

def analyze_emd_efficiency(mechanism, params_file, simulation_counts_list, k_folds=5):
    """
    Run analysis for list of simulation counts using cross-validation.
    
    Args:
        mechanism (str): Mechanism name
        params_file (str): Path to optimized parameters file
        simulation_counts_list (list): List of simulation counts to test
        k_folds (int): Number of cross-validation folds
    
    Returns:
        dict: Results containing train/val EMD statistics
    """
    print(f"\n{'='*60}")
    print(f"Analyzing mechanism: {mechanism}")
    print(f"Loading parameters from {params_file}...")
    params = load_parameters(params_file)
    if params is None:
        return None
    
    # Convert to parameter vector
    param_vector = convert_params_to_vector(params, mechanism)
    
    # Load experimental data and create folds
    exp_datasets = load_experimental_data()
    folds = create_folds(exp_datasets, k_folds=k_folds, seed=42)
    
    results = {
        'mechanism': mechanism,
        'counts': [],
        'train_means': [], 'train_stds': [],
        'val_means': [], 'val_stds': [],
        'times': []
    }
    
    print(f"Starting EMD analysis with {k_folds}-fold CV...")
    print(f"Counts: {simulation_counts_list}")
    
    for count in simulation_counts_list:
        print(f"Testing N={count}...", flush=True)
        start_time = time.time()
        
        fold_train_emds = []
        fold_val_emds = []
        
        for k in range(k_folds):
            print(f"  Fold {k+1}/{k_folds}: ", end="", flush=True)
            
            train_data = folds[k]['train']
            val_data = folds[k]['val']
            
            # Evaluate Parameters
            train_emd = objective_function(param_vector, mechanism, train_data, count)
            val_emd = objective_function(param_vector, mechanism, val_data, count)
            
            fold_train_emds.append(train_emd)
            fold_val_emds.append(val_emd)
            
            print(f"Train={train_emd:.1f}, Val={val_emd:.1f}")
            
        elapsed = time.time() - start_time
        avg_time = elapsed / k_folds
        
        # Calculate statistics across folds
        results['counts'].append(count)
        results['times'].append(avg_time)
        
        results['train_means'].append(np.mean(fold_train_emds))
        results['train_stds'].append(np.std(fold_train_emds))
        results['val_means'].append(np.mean(fold_val_emds))
        results['val_stds'].append(np.std(fold_val_emds))
        
        print(f"  Summary (Time/Fold: {avg_time:.3f}s):")
        print(f"    Train: {np.mean(fold_train_emds):.2f} ± {np.std(fold_train_emds):.2f}, "
              f"Val: {np.mean(fold_val_emds):.2f} ± {np.std(fold_val_emds):.2f}")

    return results

def plot_results(all_results):
    """
    Plot EMD efficiency showing Train vs Val across multiple mechanisms.
    
    Args:
        all_results: List of result dictionaries, one per mechanism
    """
    if not all_results:
        print("No results to plot.")
        return
    
    from matplotlib.ticker import ScalarFormatter
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Define colors and markers for different mechanisms
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', '^', 's', 'D', 'v', 'p']
    
    for idx, results in enumerate(all_results):
        mechanism = results['mechanism']
        counts = results['counts']
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        # Plot 1: Train vs Val EMD
        ax1.errorbar(counts, results['train_means'], yerr=results['train_stds'], 
                     fmt=f'{marker}-', label=f'{mechanism} (Train)', 
                     capsize=5, color=color, alpha=0.8, linewidth=2)
        ax1.errorbar(counts, results['val_means'], yerr=results['val_stds'], 
                     fmt=f'{marker}--', label=f'{mechanism} (Val)', 
                     capsize=5, color=color, alpha=0.6, linewidth=1.5)
        
        # Plot 2: Validation Noise (Std Dev) vs N
        ax2.plot(counts, results['val_stds'], f'{marker}-', 
                label=f'{mechanism}', color=color, alpha=0.8, linewidth=2)
    
    # Configure Plot 1
    ax1.set_xlabel('Simulation sample size')
    ax1.set_ylabel('Total EMD (sec)')
    ax1.set_title('EMD Stability: Train vs Validation', y=1.02)
    ax1.legend(loc='best', fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3, linestyle=':')
    
    # Configure Plot 2
    ax2.set_xlabel('Simulation sample size')
    ax2.set_ylabel('Std of Validation EMD (min)')
    ax2.set_title('Validation EMD Noise Level vs Sample Size', y=1.02)
    ax2.legend(loc='best', fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, linestyle=':')
    
    # Improve y-axis tick formatting for second plot
    ax2.yaxis.set_major_formatter(ScalarFormatter())
    ax2.ticklabel_format(style='plain', axis='y')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35)
    plt.savefig('emd_sample_efficiency_analysis_cv.png', dpi=300, bbox_inches='tight')
    plt.savefig('emd_sample_efficiency_analysis_cv.svg', format='svg', bbox_inches='tight')
    print("\nPlot saved to emd_sample_efficiency_analysis_cv.png")



if __name__ == "__main__":
    # Configuration - Multiple mechanisms
    mechanisms_to_analyze = [
        ('time_varying_k', 'ParameterFiles/simulation_optimized_parameters_time_varying_k.txt')
    ]
    
    # Test range
    counts = [500, 1000, 2000, 5000, 10000, 20000, 50000]
    k_folds = 5  # Number of cross-validation folds (was replicates=100)
    
    all_results = []
    
    for mechanism, params_file in mechanisms_to_analyze:
        if not os.path.exists(params_file):
            print(f"Warning: {params_file} not found. Skipping {mechanism}.")
            continue
        
        result = analyze_emd_efficiency(mechanism, params_file, counts, k_folds)
        if result is not None:
            all_results.append(result)
            
            # Save mechanism-specific CSV
            df = pd.DataFrame(result)
            csv_filename = f'emd_sample_efficiency_{mechanism}_cv.csv'
            df.to_csv(csv_filename, index=False)
            print(f"Results for {mechanism} saved to {csv_filename}")
    
    # Create combined plot
    if all_results:
        print(f"\n{'='*60}")
        print(f"Creating comparison plot for {len(all_results)} mechanism(s)...")
        plot_results(all_results)
    else:
        print("No results to plot. Please check parameter files.")

