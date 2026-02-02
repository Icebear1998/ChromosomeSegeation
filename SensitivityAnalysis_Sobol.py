#!/usr/bin/env python3
"""
SensitivityAnalysis_Sobol.py

Performs global sensitivity analysis using Sobol indices to quantify:
1. First-order effects (S1): Isolated parameter impact
2. Total-order effects (ST): Parameter + interactions
3. Second-order effects (S2): Pairwise interactions

Uses SALib package with Saltelli sampling for variance-based sensitivity analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from multiprocessing import Pool, cpu_count
import time

# SALib imports
from SALib.sample import saltelli
from SALib.analyze import sobol

import simulation_utils


def define_sobol_problem(mechanism='time_varying_k'):
    """
    Define the Sobol problem for SALib.
    Uses log10-transformed parameter space for uniform sampling across orders of magnitude.
    
    Returns:
        problem (dict): SALib problem dictionary
        param_info (dict): Metadata about original bounds and optimized values
    """
    # Get bounds from simulation_utils
    bounds_list = simulation_utils.get_parameter_bounds(mechanism)
    
    # Parameter names for time_varying_k
    param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23']
    
    # Create bounds dictionary
    param_bounds = dict(zip(param_names, bounds_list[:len(param_names)]))
    
    # Transform to log10 space with safe lower bounds
    # Note: Use 0.1 for n2 to avoid log(0)
    log_bounds = []
    original_bounds = []
    
    bounds_adjustments = {
        'n2': (0.1, 50),      # Avoid n2=0
        'N2': (50, 1000),
        'k_max': (0.001, 0.1),
        'tau': (2, 240),
        'r21': (0.25, 4),
        'r23': (0.25, 4),
        'R21': (0.4, 2),
        'R23': (0.5, 5)
    }
    
    for param in param_names:
        lb, ub = bounds_adjustments.get(param, param_bounds[param])
        original_bounds.append([lb, ub])
        log_bounds.append([np.log10(lb), np.log10(ub)])
    
    # Define SALib problem
    problem = {
        'num_vars': len(param_names),
        'names': [f'log_{p}' for p in param_names],
        'bounds': log_bounds
    }
    
    param_info = {
        'param_names': param_names,
        'original_bounds': original_bounds,
        'log_bounds': log_bounds
    }
    
    return problem, param_info


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


def evaluate_model(param_values_log, param_names, mechanism, num_simulations):
    """
    Evaluate the model for a single parameter set.
    
    Args:
        param_values_log (array): Log10-transformed parameter values
        param_names (list): Parameter names
        mechanism (str): Mechanism name
        num_simulations (int): Number of simulations to run
        
    Returns:
        array: [Mean_T12, Std_T12, Mean_T32, Std_T32]
    """
    # Transform back to natural scale
    param_values = 10 ** param_values_log
    
    # Create parameter dictionary
    params = dict(zip(param_names, param_values))
    
    # Update derived parameters
    update_derived_params(params)
    
    # Get initial molecule counts
    n0_list = [params['n1'], params['n2'], params['n3']]
    
    # Run simulation
    try:
        t12_arr, t32_arr = simulation_utils.run_simulation_for_dataset(
            mechanism, params, n0_list, num_simulations=num_simulations
        )
        
        # Calculate outputs
        outputs = [
            np.mean(t12_arr),
            np.std(t12_arr),
            np.mean(t32_arr),
            np.std(t32_arr)
        ]
        
        return outputs
    
    except Exception as e:
        print(f"Error evaluating model: {e}")
        print(f"Parameters: {params}")
        return [np.nan, np.nan, np.nan, np.nan]


def evaluate_model_wrapper(args):
    """Wrapper for parallel evaluation."""
    return evaluate_model(*args)


def run_sobol_analysis(mechanism='time_varying_k', N=1000, num_simulations=10000, 
                       num_replicates=5, use_parallel=True):
    """
    Run Sobol global sensitivity analysis.
    
    Args:
        mechanism (str): Mechanism name
        N (int): Base sample size for Saltelli (total sets = N*(2k+2))
        num_simulations (int): Simulations per parameter set
        num_replicates (int): Number of replicates for convergence testing
        use_parallel (bool): Use parallel processing
    """
    print(f"=== Sobol Global Sensitivity Analysis ===")
    print(f"Mechanism: {mechanism}")
    print(f"Base sample size (N): {N}")
    print(f"Simulations per parameter set: {num_simulations:,}")
    print(f"Replicates: {num_replicates}")
    print()
    
    # Define problem
    problem, param_info = define_sobol_problem(mechanism)
    param_names = param_info['param_names']
    num_params = len(param_names)
    
    total_samples = N * (2 * num_params + 2)
    print(f"Total parameter sets per replicate: {total_samples:,}")
    print(f"Total evaluations (all replicates): {total_samples * num_replicates:,}")
    print()
    
    # Output directory
    output_dir = "SensitivityAnalysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Storage for all replicates
    all_indices = {
        'Mean_T12': {'S1': [], 'ST': [], 'S2': []},
        'Std_T12': {'S1': [], 'ST': [], 'S2': []},
        'Mean_T32': {'S1': [], 'ST': [], 'S2': []},
        'Std_T32': {'S1': [], 'ST': [], 'S2': []}
    }
    
    # Run replicates
    for rep in range(num_replicates):
        print(f"\n{'='*60}")
        print(f"Replicate {rep + 1}/{num_replicates}")
        print(f"{'='*60}")
        
        # Generate Saltelli sample
        print("Generating Saltelli sample...")
        param_values = saltelli.sample(problem, N, calc_second_order=True)
        print(f"Generated {param_values.shape[0]:,} parameter sets")
        
        # Save parameter sets
        df_params = pd.DataFrame(param_values, columns=problem['names'])
        # Add natural scale columns
        for i, pname in enumerate(param_names):
            df_params[pname] = 10 ** df_params[f'log_{pname}']
        
        csv_path = f"{output_dir}/sobol_parameters_{mechanism}_rep{rep+1}.csv"
        df_params.to_csv(csv_path, index=False)
        print(f"Saved parameters to {csv_path}")
        
        # Evaluate model for all parameter sets
        print(f"\nEvaluating {param_values.shape[0]:,} parameter sets...")
        start_time = time.time()
        
        if use_parallel:
            num_cores = min(cpu_count(), 8)
            print(f"Using {num_cores} cores for parallel execution")
            
            # Prepare arguments for parallel execution
            args_list = [
                (param_values[i], param_names, mechanism, num_simulations)
                for i in range(param_values.shape[0])
            ]
            
            with Pool(num_cores) as pool:
                Y = pool.map(evaluate_model_wrapper, args_list)
            
            Y = np.array(Y)
        else:
            Y = np.array([
                evaluate_model(param_values[i], param_names, mechanism, num_simulations)
                for i in range(param_values.shape[0])
            ])
        
        elapsed = time.time() - start_time
        print(f"Evaluation completed in {elapsed/60:.1f} minutes")
        print(f"Average time per parameter set: {elapsed/param_values.shape[0]:.2f} seconds")
        
        # Save outputs
        df_outputs = pd.DataFrame(Y, columns=['Mean_T12', 'Std_T12', 'Mean_T32', 'Std_T32'])
        csv_path = f"{output_dir}/sobol_outputs_{mechanism}_rep{rep+1}.csv"
        df_outputs.to_csv(csv_path, index=False)
        print(f"Saved outputs to {csv_path}")
        
        # Analyze Sobol indices for each output
        output_names = ['Mean_T12', 'Std_T12', 'Mean_T32', 'Std_T32']
        
        for j, output_name in enumerate(output_names):
            print(f"\n--- Analyzing {output_name} ---")
            
            Y_output = Y[:, j]
            
            # Check for NaNs
            if np.any(np.isnan(Y_output)):
                print(f"Warning: {np.sum(np.isnan(Y_output))} NaN values in {output_name}")
                # Replace NaNs with mean for analysis
                Y_output = np.nan_to_num(Y_output, nan=np.nanmean(Y_output))
            
            # Perform Sobol analysis
            Si = sobol.analyze(problem, Y_output, calc_second_order=True, print_to_console=False)
            
            # Store indices
            all_indices[output_name]['S1'].append(Si['S1'])
            all_indices[output_name]['ST'].append(Si['ST'])
            all_indices[output_name]['S2'].append(Si['S2'])
            
            # Print summary
            print(f"First-order indices (S1):")
            for i, pname in enumerate(param_names):
                print(f"  {pname:8s}: {Si['S1'][i]:7.4f} ± {Si['S1_conf'][i]:7.4f}")
            
            print(f"Total-order indices (ST):")
            for i, pname in enumerate(param_names):
                print(f"  {pname:8s}: {Si['ST'][i]:7.4f} ± {Si['ST_conf'][i]:7.4f}")
    
    print(f"\n{'='*60}")
    print("All replicates completed!")
    print(f"{'='*60}\n")
    
    # Aggregate indices across replicates
    print("Calculating aggregate statistics across replicates...")
    aggregate_indices = aggregate_sobol_indices(all_indices, param_names, output_dir, mechanism)
    
    # Visualization
    print("\nGenerating visualizations...")
    plot_sobol_indices(aggregate_indices, param_names, output_dir, mechanism)
    
    print(f"\nSobol analysis completed!")
    print(f"Results saved to {output_dir}/")
    
    return aggregate_indices


def aggregate_sobol_indices(all_indices, param_names, output_dir, mechanism):
    """
    Aggregate Sobol indices across replicates.
    """
    aggregate = {}
    
    output_names = ['Mean_T12', 'Std_T12', 'Mean_T32', 'Std_T32']
    
    for output_name in output_names:
        aggregate[output_name] = {}
        
        # S1 indices
        S1_array = np.array(all_indices[output_name]['S1'])  # (num_replicates, num_params)
        aggregate[output_name]['S1_mean'] = np.mean(S1_array, axis=0)
        aggregate[output_name]['S1_std'] = np.std(S1_array, axis=0)
        
        # ST indices
        ST_array = np.array(all_indices[output_name]['ST'])
        aggregate[output_name]['ST_mean'] = np.mean(ST_array, axis=0)
        aggregate[output_name]['ST_std'] = np.std(ST_array, axis=0)
        
        # S2 indices (num_replicates, num_params, num_params)
        S2_array = np.array(all_indices[output_name]['S2'])
        aggregate[output_name]['S2_mean'] = np.mean(S2_array, axis=0)
        aggregate[output_name]['S2_std'] = np.std(S2_array, axis=0)
    
    # Save aggregate indices to CSV
    for output_name in output_names:
        df_indices = pd.DataFrame({
            'Parameter': param_names,
            'S1_mean': aggregate[output_name]['S1_mean'],
            'S1_std': aggregate[output_name]['S1_std'],
            'ST_mean': aggregate[output_name]['ST_mean'],
            'ST_std': aggregate[output_name]['ST_std']
        })
        
        csv_path = f"{output_dir}/sobol_indices_{mechanism}_{output_name}.csv"
        df_indices.to_csv(csv_path, index=False)
        print(f"Saved {output_name} indices to {csv_path}")
    
    return aggregate


def plot_sobol_indices(aggregate_indices, param_names, output_dir, mechanism):
    """
    Create visualizations for Sobol indices.
    """
    output_names = ['Mean_T12', 'Std_T12', 'Mean_T32', 'Std_T32']
    
    # 1. Bar plots for S1 and ST
    for output_name in output_names:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        S1_mean = aggregate_indices[output_name]['S1_mean']
        S1_std = aggregate_indices[output_name]['S1_std']
        ST_mean = aggregate_indices[output_name]['ST_mean']
        ST_std = aggregate_indices[output_name]['ST_std']
        
        # S1 plot
        ax = axes[0]
        x_pos = np.arange(len(param_names))
        ax.bar(x_pos, S1_mean, yerr=S1_std, capsize=5, alpha=0.7, color='#2E86AB')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(param_names, rotation=45, ha='right')
        ax.set_ylabel('First-order Index (S1)', fontweight='bold')
        ax.set_title(f'First-order Sensitivity: {output_name}', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.8)
        
        # ST plot
        ax = axes[1]
        ax.bar(x_pos, ST_mean, yerr=ST_std, capsize=5, alpha=0.7, color='#A23B72')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(param_names, rotation=45, ha='right')
        ax.set_ylabel('Total-order Index (ST)', fontweight='bold')
        ax.set_title(f'Total-order Sensitivity: {output_name}', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.8)
        
        fig.suptitle(f'Sobol Indices: {output_name} ({mechanism})', 
                     fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        plot_path = f"{output_dir}/sobol_bar_{mechanism}_{output_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved bar plot: {plot_path}")
    
    # 2. Heatmaps for S2 interactions
    for output_name in output_names:
        S2_mean = aggregate_indices[output_name]['S2_mean']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Mask upper triangle (symmetric matrix)
        mask = np.triu(np.ones_like(S2_mean, dtype=bool), k=1)
        
        sns.heatmap(S2_mean, mask=mask, annot=True, fmt='.3f', 
                    cmap='RdYlBu_r', center=0, square=True,
                    xticklabels=param_names, yticklabels=param_names,
                    cbar_kws={'label': 'Second-order Index (S2)'},
                    ax=ax)
        
        ax.set_title(f'Parameter Interactions (S2): {output_name} ({mechanism})', 
                     fontweight='bold', fontsize=13)
        
        plt.tight_layout()
        plot_path = f"{output_dir}/sobol_heatmap_{mechanism}_{output_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved heatmap: {plot_path}")
    
    # 3. Combined comparison plot (all outputs, S1 and ST)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, output_name in enumerate(output_names):
        ax = axes[idx]
        
        S1_mean = aggregate_indices[output_name]['S1_mean']
        ST_mean = aggregate_indices[output_name]['ST_mean']
        
        x_pos = np.arange(len(param_names))
        width = 0.35
        
        ax.bar(x_pos - width/2, S1_mean, width, label='S1 (First-order)', 
               alpha=0.7, color='#2E86AB')
        ax.bar(x_pos + width/2, ST_mean, width, label='ST (Total-order)', 
               alpha=0.7, color='#A23B72')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(param_names, rotation=45, ha='right')
        ax.set_ylabel('Sobol Index', fontweight='bold')
        ax.set_title(output_name, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.8)
    
    fig.suptitle(f'Sobol Sensitivity Indices Comparison ({mechanism})', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    plot_path = f"{output_dir}/sobol_comparison_{mechanism}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {plot_path}")


if __name__ == "__main__":
    # Configuration
    mechanism = "time_varying_k"
    N = 1000  # Base sample size (total = N * (2*8 + 2) = 18,000)
    num_simulations = 10000  # Simulations per parameter set
    num_replicates = 5  # Replicates for convergence testing
    use_parallel = True  # Use parallel processing
    
    # Run Sobol analysis
    aggregate_indices = run_sobol_analysis(
        mechanism=mechanism,
        N=N,
        num_simulations=num_simulations,
        num_replicates=num_replicates,
        use_parallel=use_parallel
    )
