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
    
    # Base keys for simple and other mechanisms
    base_keys = ['n1', 'n2', 'n3', 'N1', 'N2', 'N3', 'k', 'burst_size', 'n_inner', 'k_max', 'tau', 'k_1']
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
    # Build header
    header = f"{'Dataset':<15}"
    for count in simulation_counts:
        header += f" | {'NLL (' + str(count) + ')':<15}"
    print(header)
    print("-" * len(header))
    
    for ds in dataset_names:
        row_str = f"{ds:<15}"
        for count in simulation_counts:
            val = results[count]['individual_nlls'].get(ds, 0.0)
            row_str += f" | {val:<15.4f}"
        print(row_str)
            
    print("-" * len(header))
    
    # Total row
    row_str = f"{'TOTAL':<15}"
    for count in simulation_counts:
        val = results[count]['total_nll']
        row_str += f" | {val:<15.4f}"
    print(row_str)
    
    print(f"\nTime Efficiency:")
    for count in simulation_counts:
        print(f"  {count} sims: {results[count]['time']:.2f}s")

    
    # Plotting
    print("\nGenerating comparison plots...")
    plot_comparison(exp_datasets, sim_data_storage, results, simulation_counts)

def plot_comparison(exp_datasets, sim_data_storage, results, simulation_counts):
    """Generate plots comparing KDEs and experimental data for multiple simulation counts."""
    # Setup colors for different counts
    # Use 'tab10' for distinct legible colors
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(simulation_counts))))
    # Or just select from the cycle directly
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    if len(simulation_counts) > len(colors):
        colors = plt.cm.tab20(np.linspace(0, 1, len(simulation_counts)))
    
    dataset_names = ['wildtype', 'threshold', 'degrade', 'degradeAPC', 'velcade']
    
    fig, axes = plt.subplots(len(dataset_names), 2, figsize=(15, 3.5 * len(dataset_names)))
    
    for i, ds in enumerate(dataset_names):
        if ds not in exp_datasets:
            continue
            
        exp_data = exp_datasets[ds]
        
        # Plot T1-T2 (Left Column)
        ax_left = axes[i, 0]
        plot_single_panel_multi(ax_left, exp_data['delta_t12'], ds, 0, sim_data_storage, results, simulation_counts, colors, f"{ds} (T1-T2)")
        
        # Plot T3-T2 (Right Column)
        ax_right = axes[i, 1]
        plot_single_panel_multi(ax_right, exp_data['delta_t32'], ds, 1, sim_data_storage, results, simulation_counts, colors, f"{ds} (T3-T2)")
        
    plt.tight_layout()
    output_filename = f"simulation_comparison_{'_'.join(map(str, simulation_counts))}.png"
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")

def plot_single_panel_multi(ax, exp, ds_name, time_idx, sim_data_storage, results, simulation_counts, colors, title):
    """Helper to plot a single panel with Hist + KDEs for multiple counts."""
    # Experimental Hist
    ax.hist(exp, bins=30, density=True, alpha=0.3, color='gray', label='Exp Data')
    
    # Check limits first to build grid
    all_data = [exp]
    for count in simulation_counts:
        sim_data = sim_data_storage.get(count, {}).get(ds_name)
        if sim_data:
            all_data.append(sim_data[time_idx])
            
    data_min = min(d.min() for d in all_data if len(d) > 0)
    data_max = max(d.max() for d in all_data if len(d) > 0)
    pad = (data_max - data_min) * 0.1
    x_grid = np.linspace(data_min - pad, data_max + pad, 200)
    
    # Plot KDEs for each count
    for i, count in enumerate(simulation_counts):
        sim_data = sim_data_storage.get(count, {}).get(ds_name)
        if not sim_data: continue
        
        s_vals = sim_data[time_idx]
        if len(s_vals) < 2: continue
        
        # Get NLL for label
        nll_val = results[count]['individual_nlls'].get(ds_name, float('nan'))
        
        # Use default bandwidth (1.0) explicitly here too if needed, but scipy defaults to scott.
        # However, simulation_utils calls fix bandwidth. Here we are just visualizing.
        # Let's use standard default for visualization or try to match utils?
        # Visualization usually uses scott.
        try:
            kde = gaussian_kde(s_vals) # Scipy uses scott by default.
            # Label with NLL
            label = f'N={count} (NLL={nll_val:.1f})'
            style = '--' if i % 2 == 0 else '-'
            ax.plot(x_grid, kde(x_grid), linestyle=style, color=colors[i], lw=2, alpha=0.8, label=label)
        except Exception:
            pass
            
    ax.set_title(title)
    ax.legend(fontsize=8)



def run_stability_test(mechanism, params_file, count=500, replicates=20):
    """Run multiple replicates to measure NLL variance."""
    print(f"\n" + "="*60)
    print(f"Running STABILITY TEST: {replicates} replicates of {count} simulations")
    print("="*60)
    
    params = load_parameters(params_file)
    if params is None: return

    # Prepare params... (simplified version of run_comparison)
    base_keys = ['n1', 'n2', 'n3', 'N1', 'N2', 'N3', 'k', 'burst_size', 'n_inner', 'k_max', 'tau', 'k_1']
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
    mechanism = 'fixed_burst'
    params_file = 'optimized_parameters_fixed_burst_join.txt'
    
    if len(sys.argv) > 1 and sys.argv[1] == 'stability':
        if not os.path.exists(params_file):
            print(f"File not found: {params_file}")
        else:
            run_stability_test(mechanism, params_file, count=5000, replicates=10)
    else:
        # Default behavior``
        simulation_counts = [500, 1000, 2000]
        if not os.path.exists(params_file):
            print(f"File not found: {params_file}")
        else:
            run_comparison(mechanism, params_file, simulation_counts)
