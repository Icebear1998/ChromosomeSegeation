#!/usr/bin/env python3
"""
AnalyzeKDEBandwidth.py
Script to analyze the sensitivity of NLL to different KDE bandwidths.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import pandas as pd
import sys
import os
import time

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SecondVersion'))
try:
    from simulation_utils import (
        load_experimental_data, 
        run_simulation_for_dataset,
        apply_mutant_params,
        calculate_k1_from_params
    )
except ImportError:
    # Fallback if SecondVersion not in path
    sys.path.insert(0, os.path.dirname(__file__))
    from simulation_utils import (
        load_experimental_data, 
        run_simulation_for_dataset,
        apply_mutant_params,
        calculate_k1_from_params
    )

def calculate_nll_with_bandwidth(exp_data, sim_data, bandwidth):
    """
    Calculate NLL using a specific bandwidth.
    """
    try:
        # Check if we should use Scott's rule (default in sklearn)
        if bandwidth == 'scott':
            kde = KernelDensity(kernel='gaussian', bandwidth='scott')
        elif bandwidth == 'silverman':
            kde = KernelDensity(kernel='gaussian', bandwidth='silverman')
        else:
            kde = KernelDensity(kernel='gaussian', bandwidth=float(bandwidth))
            
        # Fit KDE
        sim_values = np.asarray(sim_data).flatten()
        sim_values = sim_values[np.isfinite(sim_values)]
        
        if len(sim_values) < 10:
            return 1e6
            
        kde.fit(sim_values.reshape(-1, 1))
        
        # Score samples
        exp_values = np.asarray(exp_data).flatten()
        exp_values = exp_values[np.isfinite(exp_values)]
        
        if len(exp_values) == 0:
            return 1e6
            
        log_densities = kde.score_samples(exp_values.reshape(-1, 1))
        
        # Robust mixture to prevent infinity
        epsilon = 1e-3
        background_density = 1e-6
        
        densities = np.exp(log_densities)
        robust_densities = (1 - epsilon) * densities + epsilon * background_density
        log_densities = np.log(robust_densities)
        
        return -np.sum(log_densities)
        
    except Exception as e:
        print(f"Error in KDE calc: {e}")
        return 1e6

def run_bandwidth_analysis():
    print("Starting Bandwidth Sensitivity Analysis...")
    
    # 1. Define Test Parameters (Reasonable starting point)
    # Using parameters close to a known good fit
    base_params = {
        'n2': 10.0,
        'N2': 100.0,
        'k_max': 0.05,
        'tau': 20.0,
        'r21': 1.0,
        'r23': 1.0,
        'R21': 1.0,
        'R23': 1.0,
        'alpha': 0.5,
        'beta_k': 0.5,
        'beta_tau': 2.0,
        'beta_tau2': 2.0
    }
    
    # Add derived parameters
    base_params.update({
        'n1': base_params['n2'] * base_params['r21'],
        'n3': base_params['n2'] * base_params['r23'],
        'N1': base_params['N2'] * base_params['R21'],
        'N3': base_params['N2'] * base_params['R23'],
        'k_1': base_params['k_max'] / base_params['tau']
    })
    
    mechanism = 'time_varying_k'
    
    # 2. Load Experimental Data
    exp_datasets = load_experimental_data()
    if not exp_datasets:
        print("Error: Could not load experimental data.")
        return

    # 3. Generate Simulation Data (Once, to isolate bandwidth effect)
    print("Generating simulation data (N=5000 simulations)...")
    sim_data_storage = {}
    
    dataset_names = ['wildtype', 'threshold', 'degrade', 'degradeAPC', 'velcade']
    
    for ds in dataset_names:
        if ds not in exp_datasets: continue
        
        print(f"  Simulating {ds}...")
        # Apply mutant logic
        curr_alpha = base_params['alpha']
        curr_beta = base_params['beta_k']
        # Simplified mutant mapping for this test
        if ds == 'degradeAPC': curr_beta = 1.0 # approximation
        if ds == 'velcade': curr_beta = 1.0 # approximation
        
        mutant_params, n0_list = apply_mutant_params(
            base_params, ds, curr_alpha, curr_beta, 
            beta_tau=base_params['beta_tau'], 
            beta_tau2=base_params['beta_tau2']
        )
        
        t12, t32 = run_simulation_for_dataset(
            mechanism, mutant_params, n0_list, num_simulations=5000
        )
        
        sim_data_storage[ds] = {'delta_t12': t12, 'delta_t32': t32}

    # 4. Test Bandwidths
    bandwidths = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    results = {'Bandwidth': bandwidths, 'Total_NLL': []}
    
    print("\nCalculating NLLs for different bandwidths...")
    for bw in bandwidths:
        total_nll = 0
        for ds in dataset_names:
            if ds not in exp_datasets or ds not in sim_data_storage: continue
            
            exp = exp_datasets[ds]
            sim = sim_data_storage[ds]
            
            nll12 = calculate_nll_with_bandwidth(exp['delta_t12'], sim['delta_t12'], bw)
            nll32 = calculate_nll_with_bandwidth(exp['delta_t32'], sim['delta_t32'], bw)
            
            total_nll += (nll12 + nll32)
            
        results['Total_NLL'].append(total_nll)
        print(f"  Bandwidth {bw:<5}: NLL = {total_nll:.2f}")

    # 5. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(bandwidths, results['Total_NLL'], 'bo-', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xlabel('KDE Bandwidth')
    plt.ylabel('Total Negative Log-Likelihood')
    plt.title('NLL Sensitivity to KDE Bandwidth')
    
    # Annotate points
    for bw, nll in zip(bandwidths, results['Total_NLL']):
        plt.annotate(f"{nll:.0f}", (bw, nll), textcoords="offset points", xytext=(0,10), ha='center')

    plt.tight_layout()
    plt.savefig('bandwidth_sensitivity.png')
    print(f"\nPlot saved to bandwidth_sensitivity.png")
    
    # Recommendation
    min_nll_idx = np.argmin(results['Total_NLL'])
    best_bw = bandwidths[min_nll_idx]
    print(f"\nLowest NLL observed at Bandwidth = {best_bw}")
    print("Note: Lower NLL isn't always 'better' if bandwidth is too narrow (overfitting).")
    print("A bandwidth of 5.0-10.0 usually provides a good balance for this timeframe data.")

if __name__ == "__main__":
    run_bandwidth_analysis()
