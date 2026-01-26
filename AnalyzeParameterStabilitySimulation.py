#!/usr/bin/env python3
"""
Parameter Stability Analysis for Fast Simulation Methods

This script runs multiple independent optimizations for a chosen mechanism
using the Fast simulation pipeline (FastBetaSimulation or FastFeedbackSimulation)
and analyzes the stability/consistency of the fitted parameters.

Similar to SecondVersion/AnalyzeParameterStability.py but for simulation-based optimization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from SimulationOptimization_join import run_optimization
from simulation_utils import load_experimental_data, get_parameter_bounds, set_kde_bandwidth

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.optimize")
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


def extract_params_from_result(result, mechanism):
    """Extract parameter dictionary from optimization result."""
    if not result.get('success', False):
        return None
    
    params = result.get('params', {})
    if not params:
        return None
    
    # Add NLL for reference
    params['nll'] = result.get('nll', np.nan)
    
    return params


def main():
    # ========== CONFIGURATION ==========
    mechanism = 'time_varying_k'  # Options: 'simple', 'fixed_burst', 'feedback_onion', 
                                   #          'fixed_burst_feedback_onion', 'time_varying_k',
                                   #          'time_varying_k_fixed_burst', 'time_varying_k_feedback_onion',
                                   #          'time_varying_k_combined'
    num_runs = 10                  # Number of repetitions
    num_simulations = 2000         # Simulations per evaluation
    max_iterations = 10000         # DE max iterations
    
    # KDE Bandwidth configuration
    # Options: 'scott' (adaptive, h = std * n^(-1/5)) or 'fixed' (constant bandwidth)
    bandwidth_method = 'scott'     # 'scott' or 'fixed'
    fixed_bandwidth = 10.0         # Used when bandwidth_method='fixed'
    # ===================================
    
    # Set KDE bandwidth configuration
    set_kde_bandwidth(method=bandwidth_method, fixed_value=fixed_bandwidth)

    print("="*70)
    print(f"Parameter Stability Analysis for Fast Simulation Methods")
    print("="*70)
    print(f"Mechanism: {mechanism}")
    print(f"Simulations per evaluation: {num_simulations}")
    print(f"Max iterations: {max_iterations}")
    print(f"Runs: {num_runs}")
    print(f"Bandwidth: {bandwidth_method}" + (f" (h={fixed_bandwidth})" if bandwidth_method == 'fixed' else " (adaptive)"))
    print()

    # Load experimental data
    print("Loading experimental data...")
    datasets = load_experimental_data()
    if not datasets:
        print("Error: Could not load experimental data!")
        return

    print(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}")
    
    # Get parameter bounds to determine parameter names
    bounds = get_parameter_bounds(mechanism)
    # For simulation-based mechanisms, the parameter order is defined in get_parameter_bounds
    # Typical order: n2, N2, k_max/k, tau (if applicable), r21, r23, R21, R23, [burst_size/n_inner], alpha, beta_k, [beta_tau, beta_tau2]
    
    # Define parameter names based on mechanism
    base_param_names = ['n2', 'N2', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23']
    
    # Adjust for different mechanisms
    if mechanism in ['simple', 'simple_simulation', 'fixed_burst', 'fixed_burst_simulation']:
        base_param_names[2] = 'k'  # Replace k_max with k
        base_param_names.pop(3)     # Remove tau
    
    # Add mechanism-specific params
    if mechanism in ['fixed_burst', 'fixed_burst_simulation', 'time_varying_k_fixed_burst', 
                     'time_varying_k_combined']:
        base_param_names.append('burst_size')
    
    if mechanism in ['feedback_onion', 'feedback_onion_simulation', 'fixed_burst_feedback_onion',
                     'fixed_burst_feedback_onion_simulation', 'time_varying_k_feedback_onion', 
                     'time_varying_k_combined']:
        base_param_names.append('n_inner')
    
    # Add mutant parameters
    # Check if mechanism is time-varying to determine which beta parameters to track
    is_time_varying = mechanism.startswith('time_varying_k')
    
    if is_time_varying:
        mutant_params = ['alpha', 'beta_k', 'beta_tau', 'beta_tau2']
    else:
        # Simple mechanisms
        mutant_params = ['alpha', 'beta_k1', 'beta_k2', 'beta_k3']
    
    base_param_names.extend(mutant_params)
    
    # Store results
    all_params = []  # List of dicts

    for i in range(num_runs):
        seed = 42 + i  # Different seed for each run
        print(f"\n{'='*70}")
        print(f"Run {i+1}/{num_runs} (seed={seed})")
        print(f"{'='*70}")
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        result = run_optimization(
            mechanism=mechanism,
            datasets=datasets,
            max_iterations=max_iterations,
            num_simulations=num_simulations,
            selected_strains=None
        )
        
        if result['success']:
            params = extract_params_from_result(result, mechanism)
            if params:
                # Add run ID for tracking/coloring
                params['Run'] = f"Run {i+1}"
                all_params.append(params)
                print(f"✅ Run {i+1} completed successfully. NLL: {params['nll']:.4f}")
            else:
                print(f"❌ Run {i+1} failed to extract parameters.")
        else:
            print(f"❌ Run {i+1} failed: {result.get('message', 'Unknown error')}")

    if not all_params:
        print("\n❌ No successful runs. Cannot analyze parameter stability.")
        return

    print(f"\n✅ Successfully completed {len(all_params)}/{num_runs} runs")
    
    # Convert to DataFrame
    df_results = pd.DataFrame(all_params)
    
    # Save raw results
    raw_file = f"parameter_stability_raw_{mechanism}.csv"
    df_results.to_csv(raw_file, index=False)
    print(f"\n📝 Raw results saved to {raw_file}")
    
    # Calculate and save statistics
    stats_file = f"parameter_stability_stats_{mechanism}.txt"
    with open(stats_file, 'w') as f:
        f.write(f"Parameter Stability Statistics ({len(all_params)} successful runs out of {num_runs})\\n")
        f.write(f"Mechanism: {mechanism}\\n")
        f.write(f"Simulations per evaluation: {num_simulations}\\n")
        f.write(f"Max iterations: {max_iterations}\\n")
        f.write("-" * 70 + "\\n")
        f.write(f"{'Parameter':<20} {'Mean':<12} {'Std':<12} {'CV(%)':<10} {'Min':<12} {'Max':<12}\\n")
        f.write("-" * 70 + "\\n")
        
        # Calculate stats for numeric columns only (exclude 'Run')
        numeric_cols = [c for c in df_results.columns if c not in ['Run']]
        
        for col in sorted(numeric_cols):
            vals = df_results[col].dropna()
            if len(vals) > 0:
                mean = vals.mean()
                std = vals.std()
                cv = (std / mean * 100) if mean != 0 else 0.0
                min_val = vals.min()
                max_val = vals.max()
                f.write(f"{col:<20} {mean:<12.6f} {std:<12.6f} {cv:<10.2f} {min_val:<12.6f} {max_val:<12.6f}\\n")
    
    print(f"📊 Statistics saved to {stats_file}")
    
    # Print summary table to console
    print(f"\n{'='*70}")
    print("PARAMETER STABILITY SUMMARY")
    print(f"{'='*70}")
    print(f"{'Parameter':<20} {'Mean':<12} {'Std':<12} {'CV(%)':<10}")
    print("-" * 70)
    
    numeric_cols = [c for c in df_results.columns if c not in ['Run']]
    for col in sorted(numeric_cols):
        vals = df_results[col].dropna()
        if len(vals) > 0:
            mean = vals.mean()
            std = vals.std()
            cv = (std / mean * 100) if mean != 0 else 0.0
            # Highlight high variability (CV > 10%)
            marker = "⚠️" if cv > 10 else "✅"
            print(f"{col:<20} {mean:<12.6f} {std:<12.6f} {cv:<10.2f} {marker}")
    
    # Plotting
    # Plot all numeric parameters except NLL
    plot_cols = [c for c in df_results.columns if c not in ['nll', 'Run']]
    n_params = len(plot_cols)
    
    if n_params == 0:
        print("\nNo parameters to plot.")
        return
    
    # Dynamic grid layout
    cols = 4
    rows = (n_params + cols - 1) // cols
    
    plt.figure(figsize=(16, 4 * rows))
    
    # Create a consistent palette for runs
    unique_runs = df_results['Run'].unique()
    palette = sns.color_palette("husl", len(unique_runs))
    
    for idx, param in enumerate(plot_cols):
        ax = plt.subplot(rows, cols, idx + 1)
        
        # Box plot
        sns.boxplot(y=df_results[param], ax=ax, color='lightblue', showfliers=True)
        # Strip plot (jitter) to show individual points, colored by Run
        sns.stripplot(data=df_results, y=param, hue='Run', palette=palette, ax=ax, 
                     alpha=0.7, jitter=True, legend=False, size=6)
        
        ax.set_title(param, fontsize=12, fontweight='bold')
        ax.set_ylabel('')
        
        # Add CV to xlabel
        vals = df_results[param].dropna()
        if len(vals) > 0:
            mean = vals.mean()
            std = vals.std()
            cv = (std / mean * 100) if mean != 0 else 0
            ax.set_xlabel(f"CV: {cv:.1f}%", fontsize=10)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f"Parameter Distribution Across {len(all_params)} Runs\\nMechanism: {mechanism}", 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plot_file = f"parameter_stability_distribution_{mechanism}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n📈 Distribution plot saved to {plot_file}")
    
    # Show plot
    # plt.show()  # Uncomment to display plot interactively


if __name__ == "__main__":
    main()
