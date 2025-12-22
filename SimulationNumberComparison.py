#!/usr/bin/env python3
"""
Script to compare simulation efficiency (NLL and KDE) for different simulation counts.
Focuses on 500 vs 5000 simulations to determine if 500 is sufficient.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
import sys
import os
import time

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SecondVersion'))
from simulation_utils import (
    load_experimental_data, 
    apply_mutant_params,
    calculate_likelihood,
    run_simulation_for_dataset
)

def load_parameters(filename):
    """Load parameters from optimization results file."""
    params = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        
def load_parameters(filename):
    """Load parameters from optimization results file."""
    params = {}
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Try different parsing strategies
        # Strategy 1: Look for simple key: value pairs (common in some output files)
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # support "key: value" or "key = value"
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
        
        # Calculate derived parameters if not present
        if 'r21' in params and 'n1' not in params:
            params['n1'] = max(params['r21'] * params['n2'], 1)
        if 'r23' in params and 'n3' not in params:
            params['n3'] = max(params['r23'] * params['n2'], 1)
        if 'R21' in params and 'N1' not in params:
            params['N1'] = max(params['R21'] * params['N2'], 1)
        if 'R23' in params and 'N3' not in params:
            params['N3'] = max(params['R23'] * params['N2'], 1)
            
        # Ensure we have n1, n2, n3, N1, N2, N3 for the simulation
        required = ['n1', 'n2', 'n3', 'N1', 'N2', 'N3', 'k']
        missing = [p for p in required if p not in params and p != 'k'] # k might be beta_k modified later
        
        if missing:
             # Try to recover if we have r21/R21 style but map didn't work (which shouldn't happen with logic above)
             pass

        return params
    except Exception as e:
        print(f"Error loading parameters: {e}")
        return None
    except Exception as e:
        print(f"Error loading parameters: {e}")
        return None

def run_comparison(mechanism, params_file, simulation_counts=[500, 5000]):
    """
    Run comparison for different simulation counts.
    
    Args:
        mechanism (str): Mechanism name (e.g., 'simple')
        params_file (str): Path to parameter file
        simulation_counts (list): List of simulation counts to test
    """
    print(f"Loading parameters from {params_file}...")
    params = load_parameters(params_file)
    if params is None:
        return
    
    print("\nParameters:")
    for key, val in params.items():
        print(f"  {key}: {val}")
    
    print("\nLoading experimental data...")
    exp_datasets = load_experimental_data()
    
    results = {}
    
    # Base keys for simple mechanism
    base_keys = ['n1', 'n2', 'n3', 'N1', 'N2', 'N3', 'k']
    base_params = {k: params.get(k) for k in base_keys if k in params}
    
    # Store simulation data for plotting
    sim_data_storage = {}
    
    for count in simulation_counts:
        print(f"\n" + "="*60)
        print(f"Running for {count} simulations...")
        print("="*60)
        
        start_time = time.time()
        
        total_nll = 0
        individual_nlls = {}
        dataset_sim_data = {}
        
        # Iterate over datasets
        # Order: wildtype, threshold, degrade, degradeAPC, velcade
        dataset_names = ['wildtype', 'threshold', 'degrade', 'degradeAPC', 'velcade']
        
        for dataset_name in dataset_names:
            if dataset_name not in exp_datasets:
                continue
                
            # Apply mutant parameters
            alpha = params.get('alpha', 1.0)
            beta_k = params.get('beta_k', 1.0)
            beta2_k = params.get('beta2_k', 1.0)
            beta3_k = params.get('beta3_k', 1.0)
            
            # Determine which beta to use
            if dataset_name == 'degradeAPC':
                current_beta = beta2_k
            elif dataset_name == 'velcade':
                current_beta = beta3_k
            else:
                current_beta = beta_k
            
            mutant_params, n0_list = apply_mutant_params(
                base_params, dataset_name, alpha, current_beta
            )
            
            # Special handling for threshold mutant to match MoM consistency
            if dataset_name == 'threshold':
                 # Calculate threshold values the same way MoM does
                n1_th_mom = max(base_params['n1'] * alpha, 1)
                n2_th_mom = max(base_params['n2'] * alpha, 1)
                n3_th_mom = max(base_params['n3'] * alpha, 1)
                
                # Update to match
                mutant_params['n1'] = n1_th_mom
                mutant_params['n2'] = n2_th_mom
                mutant_params['n3'] = n3_th_mom
                n0_list = [n1_th_mom, n2_th_mom, n3_th_mom]

            
            # Run simulation
            # print(f"  Simulating {dataset_name}...")
            sim_t12, sim_t32 = run_simulation_for_dataset(
                mechanism, mutant_params, n0_list, num_simulations=count
            )
            
            if sim_t12 is None:
                print(f"    Failed for {dataset_name}")
                continue
            
            # Calculate NLL
            exp_data = exp_datasets[dataset_name]
            sim_data_dict = {'delta_t12': sim_t12, 'delta_t32': sim_t32}
            
            nll = calculate_likelihood(exp_data, sim_data_dict)
            individual_nlls[dataset_name] = nll
            total_nll += nll
            
            # Store data
            dataset_sim_data[dataset_name] = (sim_t12, sim_t32)
        
        elapsed = time.time() - start_time
        
        results[count] = {
            'total_nll': total_nll,
            'individual_nlls': individual_nlls,
            'time': elapsed
        }
        sim_data_storage[count] = dataset_sim_data
        
        print(f"  Total NLL: {total_nll:.4f}")
        print(f"  Time taken: {elapsed:.2f}s")
    
    # ================= Comparison and Plotting =================
    
    print(f"\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    # NLL Comparison Table
    print(f"{'Dataset':<15} | {'NLL (' + str(simulation_counts[0]) + ')':<15} | {'NLL (' + str(simulation_counts[1]) + ')':<15} | {'Diff':<10} | {'% Diff':<10}")
    print("-" * 80)
    
    c1 = simulation_counts[0]
    c2 = simulation_counts[1]
    
    for ds in dataset_names:
        if ds in results[c1]['individual_nlls']:
            nll1 = results[c1]['individual_nlls'][ds]
            nll2 = results[c2]['individual_nlls'][ds]
            diff = abs(nll1 - nll2)
            pct = (diff / abs(nll2)) * 100 if nll2 != 0 else 0
            
            print(f"{ds:<15} | {nll1:<15.4f} | {nll2:<15.4f} | {diff:<10.4f} | {pct:<10.2f}")
    
    print("-" * 80)
    t_nll1 = results[c1]['total_nll']
    t_nll2 = results[c2]['total_nll']
    t_diff = abs(t_nll1 - t_nll2)
    t_pct = (t_diff / abs(t_nll2)) * 100
    print(f"{'TOTAL':<15} | {t_nll1:<15.4f} | {t_nll2:<15.4f} | {t_diff:<10.4f} | {t_pct:<10.2f}")
    
    print(f"\nTime Efficiency:")
    print(f"  {c1} sims: {results[c1]['time']:.2f}s")
    print(f"  {c2} sims: {results[c2]['time']:.2f}s")
    
    # Plotting
    print("\nGenerating comparison plots...")
    plot_comparison(exp_datasets, sim_data_storage, simulation_counts)

def plot_comparison(exp_datasets, sim_data_storage, simulation_counts):
    """Generate plots comparing KDEs and experimental data."""
    c1, c2 = simulation_counts
    dataset_names = ['wildtype', 'threshold', 'degrade', 'degradeAPC', 'velcade']
    
    fig, axes = plt.subplots(len(dataset_names), 2, figsize=(15, 3 * len(dataset_names)))
    
    for i, ds in enumerate(dataset_names):
        if ds not in exp_datasets:
            continue
            
        exp_data = exp_datasets[ds]
        sim1 = sim_data_storage[c1].get(ds)
        sim2 = sim_data_storage[c2].get(ds)
        
        if not sim1 or not sim2:
            continue
            
        # Plot T1-T2 (Left Column)
        ax_left = axes[i, 0]
        plot_single_panel(ax_left, exp_data['delta_t12'], sim1[0], sim2[0], f"{ds} (T1-T2)", c1, c2)
        
        # Plot T3-T2 (Right Column)
        ax_right = axes[i, 1]
        plot_single_panel(ax_right, exp_data['delta_t32'], sim1[1], sim2[1], f"{ds} (T3-T2)", c1, c2)
        
    plt.tight_layout()
    plt.savefig(f"simulation_comparison_{c1}_vs_{c2}.png")
    print(f"Plot saved to simulation_comparison_{c1}_vs_{c2}.png")

def plot_single_panel(ax, exp, s1, s2, title, c1, c2):
    """Helper to plot a single panel with Hist + KDEs."""
    # Experimental Hist
    ax.hist(exp, bins=30, density=True, alpha=0.3, color='gray', label='Exp Data')
    
    # Limits for KDE evaluation
    data_min = min(exp.min(), s1.min(), s2.min())
    data_max = max(exp.max(), s1.max(), s2.max())
    pad = (data_max - data_min) * 0.1
    x_grid = np.linspace(data_min - pad, data_max + pad, 200)
    
    # KDE 1
    kde1 = gaussian_kde(s1)
    ax.plot(x_grid, kde1(x_grid), 'b--', lw=2, label=f'Sim {c1} (KDE)')
    
    # KDE 2
    kde2 = gaussian_kde(s2)
    ax.plot(x_grid, kde2(x_grid), 'r-', lw=2, alpha=0.7, label=f'Sim {c2} (KDE)')
    
    ax.set_title(title)
    if 'wildtype' in title and 'T1-T2' in title: # Only legend on first plot to reduce clutter
        ax.legend()

def run_stability_test(mechanism, params_file, count=500, replicates=20):
    """Run multiple replicates to measure NLL variance."""
    print(f"\n" + "="*60)
    print(f"Running STABILITY TEST: {replicates} replicates of {count} simulations")
    print("="*60)
    
    params = load_parameters(params_file)
    if params is None: return

    # Prepare params... (simplified version of run_comparison)
    base_keys = ['n1', 'n2', 'n3', 'N1', 'N2', 'N3', 'k']
    base_params = {k: params.get(k) for k in base_keys if k in params}
    exp_datasets = load_experimental_data()
    dataset_names = ['wildtype', 'threshold', 'degrade', 'degradeAPC', 'velcade']
    
    nll_values = []
    
    for i in range(replicates):
        total_nll = 0
        for dataset_name in dataset_names:
            if dataset_name not in exp_datasets: continue
            
            # Simplified param logic
            alpha = params.get('alpha', 1.0)
            beta_k = params.get('beta_k', 1.0)
            beta2_k = params.get('beta2_k', 1.0)
            beta3_k = params.get('beta3_k', 1.0)
            
            curr_beta = beta2_k if dataset_name == 'degradeAPC' else (beta3_k if dataset_name == 'velcade' else beta_k)
            mutant_params, n0_list = apply_mutant_params(base_params, dataset_name, alpha, curr_beta)
            
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
        print(f"  Run {i+1}/{replicates}: NLL = {total_nll:.4f}")

    nll_values = np.array(nll_values)
    mean_nll = np.mean(nll_values)
    std_nll = np.std(nll_values)
    range_nll = np.max(nll_values) - np.min(nll_values)
    
    print("-" * 60)
    print(f"Results for N={count}:")
    print(f"  Mean NLL: {mean_nll:.4f}")
    print(f"  Std Dev:  {std_nll:.4f}")
    print(f"  Range:    {range_nll:.4f} ({np.min(nll_values):.2f} - {np.max(nll_values):.2f})")
    print("-" * 60)
    
    if std_nll > 10.0:
        print("⚠ WARNING: High variance! Optimization may be unstable.")
    else:
        print("✓ Variance seems acceptable.")

if __name__ == "__main__":
    mechanism = 'simple'
    params_file = 'optimized_parameters_simple_join.txt'
    
    if len(sys.argv) > 1 and sys.argv[1] == 'stability':
        if not os.path.exists(params_file):
            print(f"File not found: {params_file}")
        else:
            run_stability_test(mechanism, params_file, count=5000, replicates=10)
    else:
        # Default behavior``
        simulation_counts = [500, 2000]
        if not os.path.exists(params_file):
            print(f"File not found: {params_file}")
        else:
            run_comparison(mechanism, params_file, simulation_counts)
