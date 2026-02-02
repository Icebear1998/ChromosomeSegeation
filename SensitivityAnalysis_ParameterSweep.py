#!/usr/bin/env python3
"""
SensitivityAnalysis_ParameterSweep.py

Performs parameter sweep sensitivity analysis by varying each parameter
across its full bounds and observing the effect on model outputs.

Method:
1. Load optimized parameters as the baseline.
2. For each parameter P:
   - Sweep P across its full parameter bounds (e.g., 20 values from min to max).
   - At each value, run simulations and calculate Mean(T12), Std(T12), Mean(T32), Std(T32).
3. Plot dual y-axis graphs: Mean and Std vs Parameter value.
4. Mark the optimized parameter value with a vertical line.
5. Save all data to CSV for later analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

import simulation_utils

def run_parameter_sweep(mechanism, num_simulations=5000, num_points=20, n_repeat=3):
    """
    Run parameter sweep sensitivity analysis.
    
    Args:
        mechanism (str): Name of the mechanism (e.g., 'time_varying_k').
        num_simulations (int): Number of simulations per evaluation.
        num_points (int): Number of points to sample across parameter bounds.
        n_repeat (int): Number of repeat simulations at each parameter value for confidence intervals.
    """
    print(f"--- Starting Parameter Sweep Analysis for '{mechanism}' ---")
    print(f"Simulations per evaluation: {num_simulations}")
    print(f"Sample points: {num_points} per parameter")
    print(f"Repeats per point: {n_repeat} (for confidence intervals)")
    print()
    # 1. Load Optimized Parameters
    try:
        params = simulation_utils.load_optimized_parameters(mechanism)
        if not params:
            print(f"Error: Could not load parameters for {mechanism}")
            return
    except Exception as e:
        print(f"Error loading parameters: {e}")
        return
    
    # 2. Get Parameter Bounds
    bounds_list = simulation_utils.get_parameter_bounds(mechanism)
    
    # Map bounds to parameter names (this is mechanism-dependent)
    # For time_varying_k: n2, N2, k_max, tau, r21, r23, R21, R23
    is_time_varying = mechanism.startswith('time_varying_k')
    
    if is_time_varying:
        param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23']
    else:
        param_names = ['n2', 'N2', 'k', 'r21', 'r23', 'R21', 'R23']
    
    # Add optional parameters if they exist
    if 'burst_size' in params:
        param_names.append('burst_size')
    if 'n_inner' in params:
        param_names.append('n_inner')
    
    # Ensure bounds align with param names
    param_bounds = dict(zip(param_names, bounds_list[:len(param_names)]))
    
    # Override k_max range to prevent extreme values from flattening other plots
    if 'k_max' in param_bounds:
        param_bounds['k_max'] = (0.02, 0.1)
        print("  k_max range narrowed to (0.02, 0.1) for better visualization")
    
    print(f"Parameters to sweep: {param_names}")
    
    # 3. Sweep Each Parameter
    output_dir = "SensitivityAnalysis"
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    for param_name in param_names:
        print(f"\n=== Sweeping {param_name} ===")
        
        if param_name not in param_bounds:
            print(f"Skipping {param_name} (no bounds defined)")
            continue
        
        min_val, max_val = param_bounds[param_name]
        optimized_val = params[param_name]
        
        # Generate sweep values
        sweep_values = np.linspace(min_val, max_val, num_points)

        # For n2, add extra points around optimized value to capture sharp transition
        if param_name == 'n2':
            #sweep_values = np.linspace(min_val, 10, num_points)
            # Add 8 extra points within ±50% of optimized value
            delta = 0.5 * optimized_val
            extra_points = np.linspace(max(min_val, optimized_val - delta),
                                      min(max_val, optimized_val + delta), 4)
            # Combine and sort
            sweep_values = np.sort(np.unique(np.concatenate([sweep_values, extra_points])))
            print(f"  Added {len(sweep_values) - num_points} extra points around n2={optimized_val:.2f}")
            
        results = []
        
        for i, value in enumerate(sweep_values):
            print(f"  {param_name} = {value:.4f} ({i+1}/{num_points})", end='\r')
            
            # Create perturbed parameters
            params_sweep = params.copy()
            params_sweep[param_name] = value
            update_derived_params(params_sweep)
            
            n0_list = [params_sweep['n1'], params_sweep['n2'], params_sweep['n3']]
            
            # Run simulation n_repeat times for confidence intervals
            mean_t12_repeats = []
            std_t12_repeats = []
            mean_t32_repeats = []
            std_t32_repeats = []
            
            for rep in range(n_repeat):
                t12_arr, t32_arr = simulation_utils.run_simulation_for_dataset(
                    mechanism, params_sweep, n0_list, num_simulations=num_simulations
                )
                
                # Data is already in seconds from simulation
                mean_t12_repeats.append(np.mean(t12_arr))
                std_t12_repeats.append(np.std(t12_arr))
                mean_t32_repeats.append(np.mean(t32_arr))
                std_t32_repeats.append(np.std(t32_arr))
            
            # Calculate mean and std across repeats
            results.append({
                'Parameter': param_name,
                'Value': value,
                # Mean of T12 across repeats (in seconds)
                'Mean_T12': np.mean(mean_t12_repeats),
                'Mean_T12_std': np.std(mean_t12_repeats),  # Confidence interval
                # Std of T12 across repeats (in seconds)
                'Std_T12': np.mean(std_t12_repeats),
                'Std_T12_std': np.std(std_t12_repeats),
                # Mean of T32 across repeats (in seconds)
                'Mean_T32': np.mean(mean_t32_repeats),
                'Mean_T32_std': np.std(mean_t32_repeats),
                # Std of T32 across repeats (in seconds)
                'Std_T32': np.mean(std_t32_repeats),
                'Std_T32_std': np.std(std_t32_repeats)
            })
        
        print(f"\n  Completed sweep for {param_name}")
        
        # Convert to DataFrame
        df_param = pd.DataFrame(results)
        all_results.append(df_param)
    
    # 3.5. Compute global min/max for unified axes
    df_all = pd.concat(all_results, ignore_index=True)
    global_limits = {
        'Mean_T12': (df_all['Mean_T12'].min(), df_all['Mean_T12'].max()),
        'Std_T12': (df_all['Std_T12'].min(), df_all['Std_T12'].max()),
        'Mean_T32': (df_all['Mean_T32'].min(), df_all['Mean_T32'].max()),
        'Std_T32': (df_all['Std_T32'].min(), df_all['Std_T32'].max())
    }
    
    # 3.6. Plot each parameter with unified axes
    for i, param_name in enumerate(param_names):
        df_param = all_results[i]
        optimized_val = params[param_name]
        plot_parameter_sweep(df_param, param_name, optimized_val, mechanism, 
                           output_dir, n_repeat, global_limits)
    
    # 4. Save All Data
    df_all = pd.concat(all_results, ignore_index=True)
    csv_path = f"{output_dir}/parameter_sweep_{mechanism}_full.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"\n\nAll results saved to {csv_path}")


def update_derived_params(params):
    """Update N1, N3, n1, n3 based on ratios if they exist."""
    if 'r21' in params and 'n2' in params:
        params['n1'] = params['r21'] * params['n2']
    if 'r23' in params and 'n2' in params:
        params['n3'] = params['r23'] * params['n2']
    if 'R21' in params and 'N2' in params:
        params['N1'] = params['R21'] * params['N2']
    if 'R23' in params and 'N2' in params:
        params['N3'] = params['R23'] * params['N2']
    
    if 'k_max' in params and 'tau' in params:
        params['k_1'] = params['k_max'] / params['tau']


def plot_parameter_sweep(df, param_name, optimized_val, mechanism, output_dir, n_repeat, global_limits=None):
    """
    Plot dual y-axis figure: Mean and Std vs Parameter value with error bars.
    Creates 2 subplots: one for T12, one for T32.
    
    Args:
        df: DataFrame with columns including Mean_T12, Mean_T12_std, etc.
        param_name: Parameter being swept
        optimized_val: Optimized parameter value
        mechanism: Mechanism name
        output_dir: Output directory
        global_limits: Dict with global min/max for unified axes (optional)
        n_repeat: Number of repeats (for documentation in title)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # T12 subplot
    ax1 = axes[0]
    ax2 = ax1.twinx()
    
    # Plot Mean_T12 with error bars
    l1 = ax1.errorbar(df['Value'], df['Mean_T12'], yerr=df['Mean_T12_std'],
                      fmt='o-', color='#1f77b4', label='Mean T12',
                      linewidth=2, markersize=5, capsize=3, alpha=0.8)
    
    # Plot Std_T12 with error bars
    l2 = ax2.errorbar(df['Value'], df['Std_T12'], yerr=df['Std_T12_std'],
                      fmt='s-', color='#ff7f0e', label='Std T12',
                      linewidth=2, markersize=5, capsize=3, alpha=0.8)
    
    # Mark optimized value
    ax1.axvline(optimized_val, color='red', linestyle='--', linewidth=2, 
                label=f'Optimized: {optimized_val:.3f}')
    
    ax1.set_xlabel(f'{param_name}', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean T12 (sec)', fontsize=11, color='#1f77b4', fontweight='bold')
    ax2.set_ylabel('Std T12 (sec)', fontsize=11, color='#ff7f0e', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax1.set_title('T12 Response', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Set unified y-axis limits if global_limits provided
    if global_limits:
        ax1.set_ylim(global_limits['Mean_T12'])
        ax2.set_ylim(global_limits['Std_T12'])
    
    # Combined legend
    lns = [l1, l2, ax1.get_lines()[0]]  # errorbar objects + vertical line
    labs = [l.get_label() if hasattr(l, 'get_label') else 'Mean T12' for l in lns]
    labs = ['Mean T12', 'Std T12', f'Optimized: {optimized_val:.3f}']
    ax1.legend(lns, labs, loc='best', framealpha=0.9)
    
    # T32 subplot
    ax3 = axes[1]
    ax4 = ax3.twinx()
    
    # Plot Mean_T32 with error bars
    l3 = ax3.errorbar(df['Value'], df['Mean_T32'], yerr=df['Mean_T32_std'],
                      fmt='o-', color='#2ca02c', label='Mean T32',
                      linewidth=2, markersize=5, capsize=3, alpha=0.8)
    
    # Plot Std_T32 with error bars
    l4 = ax4.errorbar(df['Value'], df['Std_T32'], yerr=df['Std_T32_std'],
                      fmt='s-', color='#d62728', label='Std T32',
                      linewidth=2, markersize=5, capsize=3, alpha=0.8)
    
    ax3.axvline(optimized_val, color='red', linestyle='--', linewidth=2, 
                label=f'Optimized: {optimized_val:.3f}')
    
    ax3.set_xlabel(f'{param_name}', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Mean T32 (sec)', fontsize=11, color='#2ca02c', fontweight='bold')
    ax4.set_ylabel('Std T32 (sec)', fontsize=11, color='#d62728', fontweight='bold')
    ax3.tick_params(axis='y', labelcolor='#2ca02c')
    ax4.tick_params(axis='y', labelcolor='#d62728')
    ax3.set_title('T32 Response', fontsize=13, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # Set unified y-axis limits if global_limits provided
    if global_limits:
        ax3.set_ylim(global_limits['Mean_T32'])
        ax4.set_ylim(global_limits['Std_T32'])
    
    lns2 = [l3, l4, ax3.get_lines()[0]]
    labs2 = ['Mean T32', 'Std T32', f'Optimized: {optimized_val:.3f}']
    ax3.legend(lns2, labs2, loc='best', framealpha=0.9)
    
    # Overall title with note about repeats
    fig.suptitle(f'Parameter Sweep: {param_name} ({mechanism}) - {n_repeat} repeats per point', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Create safe filename (handle case-insensitive filesystems)
    # Use 'lowercase_' prefix for lowercase params to avoid collision
    safe_param_name = param_name
    if param_name[0].islower() and len(param_name) >= 2:
        # Check if there's a corresponding uppercase version
        uppercase_version = param_name[0].upper() + param_name[1:]
        if uppercase_version in ['N2', 'N1', 'N3']:  # Common uppercase params
            safe_param_name = f"threshold_{param_name}"
    if param_name[0].isupper() and len(param_name) >= 2:
        # Uppercase params like N2, R21
        safe_param_name = f"initial_{param_name}"
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sweep_{mechanism}_{safe_param_name}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: sweep_{mechanism}_{safe_param_name}.png")


if __name__ == "__main__":
    # Configuration
    mechanism = "time_varying_k"
    num_simulations = 10000  # Simulations per point
    num_points = 30  # Points to sample across parameter range
    n_repeat = 10  # Number of repeats per point for confidence intervals
    
    run_parameter_sweep(mechanism, num_simulations, num_points, n_repeat)
