#!/usr/bin/env python3
"""
SensitivityAnalysis_OAT.py

Performs One-At-a-Time (OAT) local sensitivity analysis around the optimized 
Wild-Type (WT) parameter set.

Method:
1. Load optimized parameters for a given mechanism.
2. Define base parameters (WT only).
3. For each parameter P:
   - Perturb P by +delta% and -delta%.
   - Run simulations (N times).
   - Calculate outputs: Mean(T12), Std(T12), Mean(T32), Std(T32).
   - Calculate Normalized Sensitivity Coefficient:
     S = (% Change in Output) / (% Change in Parameter)
       = ((Y_new - Y_base) / Y_base) / ((P_new - P_base) / P_base)
4. Visualize results as bar plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
try:
    import seaborn as sns
except ImportError:
    sns = None
import sys
import os
import argparse


import simulation_utils

def run_sensitivity_analysis(mechanism, num_simulations=1000, perturbation=0.01):
    """
    Run OAT sensitivity analysis.
    
    Args:
        mechanism (str): Name of the mechanism (e.g., 'time_varying_k').
        num_simulations (int): Number of simulations per evaluation.
        perturbation (float): Fractional perturbation (e.g., 0.01 for 1%).
    """
    print(f"--- Starting OAT Sensitivity Analysis for '{mechanism}' ---")
    print(f"Perturbation: {perturbation*100}%")
    print(f"Simulations per run: {num_simulations}")
    
    # 1. Load Parameters
    try:
        if mechanism in ['simple', 'fixed_burst', 'feedback_onion', 'fixed_burst_feedback_onion',
                       'steric_hindrance', 'fixed_burst_steric_hindrance']:
             # Use the ratio-based parameter file if available, or try standard
             params = simulation_utils.load_optimized_parameters(mechanism)
        else:
             params = simulation_utils.load_optimized_parameters(mechanism)
             
        if not params:
            print(f"Error: Could not load parameters for {mechanism}")
            return
            
    except Exception as e:
        print(f"Error loading parameters: {e}")
        return

    # 2. Identify Parameters to Vary (WT only)
    # We want to vary the underlying definition parameters, not necessarily the derived ones if they are coupled,
    # but the email implies "start with optimized parameters".
    # Usually we optimize: n2, N2, k (or k_max), r21, r23, R21, R23, tau (if time-varying), etc.
    
    # Filter out mutant parameters (alpha, beta_*) and derived params if we want to stick to the base set.
    # However, 'n1' is derived from 'n2' * 'r21'. 
    # If we perturb 'n2', 'n1' and 'n3' should also change if we maintain ratios?
    # OR do we perturb 'n2' and keep 'n1' fixed?
    # The "parameters" usually implies the degrees of freedom in the model. 
    # The optimized parameters are the degrees of freedom. So we should vary those.
    
    keys_to_vary = []
    
    # Base candidates
    candidates = ['n2', 'N2', 'k', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 'burst_size', 'n_inner']
    
    for key in candidates:
        if key in params:
            keys_to_vary.append(key)
            
    print(f"Parameters to analyze: {keys_to_vary}")
    
    # 3. Baseline Simulation
    print("\nRunning baseline simulation...")
    # Calculate n0_list from params
    n1 = params['n1']
    n2 = params['n2']
    n3 = params['n3']
    n0_list_base = [n1, n2, n3]
    
    t12_base_arr, t32_base_arr = simulation_utils.run_simulation_for_dataset(
        mechanism, params, n0_list_base, num_simulations=num_simulations
    )
    
    # Metrics: Mean, Std
    base_metrics = {
        'mean_t12': np.mean(t12_base_arr),
        'std_t12': np.std(t12_base_arr),
        'mean_t32': np.mean(t32_base_arr),
        'std_t32': np.std(t32_base_arr)
    }
    
    print(f"Baseline Metrics:")
    for k, v in base_metrics.items():
        print(f"  {k}: {v:.4f}")

    # 4. Iterate and Perturb
    results = []
    
    for param_name in keys_to_vary:
        original_value = params[param_name]
        
        # We will do +perturbation
        
        # --- Increase (+) ---
        perturbed_value_plus = original_value * (1 + perturbation)
        params_plus = params.copy()
        params_plus[param_name] = perturbed_value_plus
        
        # Re-calculate derived parameters if necessary
        # Note: 'simulation_utils.run_simulation_for_dataset' uses 'N1', 'n1' etc directly.
        # So if we change 'r21', we MUST update 'n1'.
        # Let's write a helper to update derived params.
        update_derived_params(params_plus)
        
        n0_list_plus = [params_plus['n1'], params_plus['n2'], params_plus['n3']]
        
        t12_p, t32_p = simulation_utils.run_simulation_for_dataset(
            mechanism, params_plus, n0_list_plus, num_simulations=num_simulations
        )
        
        metrics_plus = {
            'mean_t12': np.mean(t12_p),
            'std_t12': np.std(t12_p),
            'mean_t32': np.mean(t32_p),
            'std_t32': np.std(t32_p)
        }
        
        # --- Decrease (-) ---
        perturbed_value_minus = original_value * (1 - perturbation)
        params_minus = params.copy()
        params_minus[param_name] = perturbed_value_minus
        update_derived_params(params_minus)
        
        n0_list_minus = [params_minus['n1'], params_minus['n2'], params_minus['n3']]
        
        t12_m, t32_m = simulation_utils.run_simulation_for_dataset(
            mechanism, params_minus, n0_list_minus, num_simulations=num_simulations
        )
        
        metrics_minus = {
            'mean_t12': np.mean(t12_m),
            'std_t12': np.std(t12_m),
            'mean_t32': np.mean(t32_m),
            'std_t32': np.std(t32_m)
        }
        
        # --- Calculate Sensitivity Coefficients ---
        # S = ( (Y+ - Y-) / (2*Y_base) ) / perturbation
        # Or average of individual sensitivities:
        # S+ = ((Y+ - Yb)/Yb) / perturbation
        # S- = ((Y- - Yb)/Yb) / (-perturbation)
        # S_avg = (S+ + S-) / 2
        
        for metric_key in base_metrics.keys():
            y_base = base_metrics[metric_key]
            y_plus = metrics_plus[metric_key]
            y_minus = metrics_minus[metric_key]
            
            # Avoid divide by zero
            if abs(y_base) < 1e-9:
                s_avg = 0.0
            else:
                s_plus = ((y_plus - y_base) / y_base) / perturbation
                s_minus = ((y_minus - y_base) / y_base) / (-perturbation)
                s_avg = (s_plus + s_minus) / 2
            
            results.append({
                'Parameter': param_name,
                'Metric': metric_key,
                'Sensitivity': s_avg,
                'S_plus': s_plus,
                'S_minus': s_minus
            })

    # 5. Save and Visualize
    df_results = pd.DataFrame(results)
    
    # Save CSV
    output_dir = "SensitivityAnalysis"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = f"{output_dir}/OAT_sensitivity_{mechanism}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Plot
    plot_sensitivity(df_results, mechanism, output_dir, perturbation, num_simulations)

def update_derived_params(params):
    """
    Update N1, N3, n1, n3 based on ratios if they exist.
    """
    if 'r21' in params and 'n2' in params:
        params['n1'] = params['r21'] * params['n2']
    if 'r23' in params and 'n2' in params:
        params['n3'] = params['r23'] * params['n2']
    if 'R21' in params and 'N2' in params:
        params['N1'] = params['R21'] * params['N2']
    if 'R23' in params and 'N2' in params:
        params['N3'] = params['R23'] * params['N2']
    
    # For time varying, update k_1 if needed
    if 'k_max' in params and 'tau' in params:
        params['k_1'] = params['k_max'] / params['tau']

def plot_sensitivity(df, mechanism, output_dir, perturbation, num_simulations):
    """
    Generate sensitivity visualizations: heatmap and tornado diagram.
    """
    
    # Plot 1: Heatmap
    plot_heatmap(df, mechanism, output_dir, perturbation, num_simulations)
    
    # Plot 2: Tornado Diagram (standard for sensitivity analysis)
    plot_tornado(df, mechanism, output_dir, perturbation, num_simulations)


def plot_heatmap(df, mechanism, output_dir, perturbation, num_simulations):
    """Heatmap showing parameter sensitivity across all metrics."""
    # Pivot data for heatmap
    pivot = df.pivot(index='Parameter', columns='Metric', values='Sensitivity')
    
    # Reorder columns for clarity
    col_order = ['mean_t12', 'std_t12', 'mean_t32', 'std_t32']
    pivot = pivot[col_order]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(pivot.values, cmap='RdBu_r', aspect='auto', vmin=-8, vmax=8)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(['Mean T12', 'Std T12', 'Mean T32', 'Std T32'], fontsize=11)
    ax.set_yticklabels(pivot.index, fontsize=11)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar with better labels
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Normalized Sensitivity Coefficient\nS = (ΔY/Y₀) / (ΔP/P₀)', 
                   rotation=270, labelpad=25, fontsize=10)
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            text = ax.text(j, i, f'{val:.2f}', ha="center", va="center",
                          color="white" if abs(val) > 4 else "black", fontsize=9)
    
    # Add title with methodology note
    ax.set_title(f'OAT Sensitivity Analysis: {mechanism}', 
                 pad=15, fontweight='bold', fontsize=13)
    
    # Add explanatory note at bottom
    fig.text(0.5, 0.02, 
             f'Perturbation: ±{perturbation*100:.0f}% | Simulations per evaluation: {num_simulations:,} | '
             f'Positive (blue) = ↑parameter → ↑output | Negative (red) = ↑parameter → ↓output',
             ha='center', fontsize=9, style='italic', color='#333333')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(f"{output_dir}/OAT_sensitivity_{mechanism}_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {output_dir}/OAT_sensitivity_{mechanism}_heatmap.png")


def plot_grouped_bars(df, mechanism, output_dir):
    """Grouped bar chart showing all metrics side by side."""
    # Pivot for grouped bars
    pivot = df.pivot(index='Parameter', columns='Metric', values='Sensitivity')
    col_order = ['mean_t12', 'std_t12', 'mean_t32', 'std_t32']
    pivot = pivot[col_order]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(pivot.index))
    width = 0.2
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ['Mean T12', 'Std T12', 'Mean T32', 'Std T32']
    
    for i, (col, color, label) in enumerate(zip(col_order, colors, labels)):
        offset = (i - 1.5) * width
        ax.bar(x + offset, pivot[col], width, label=label, color=color, alpha=0.8)
    
    ax.set_xlabel('Parameter', fontweight='bold')
    ax.set_ylabel('Sensitivity Coefficient', fontweight='bold')
    ax.set_title(f'OAT Sensitivity Analysis: {mechanism}', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.legend(loc='best', framealpha=0.9)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/OAT_sensitivity_{mechanism}_grouped.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grouped bar chart saved to {output_dir}/OAT_sensitivity_{mechanism}_grouped.png")


def plot_tornado(df, mechanism, output_dir, perturbation, num_simulations):
    """
    Tornado diagram: horizontal bars showing absolute sensitivity,
    sorted by magnitude. One plot per metric.
    """
    metrics = ['mean_t12', 'std_t12', 'mean_t32', 'std_t32']
    labels = ['Mean T12', 'Std T12', 'Mean T32', 'Std T32']
    
    # Get perturbation percentage from the analysis (assume it's global or read from somewhere)
    # For now, we'll add it as an annotation
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()
    
    for ax, metric, label in zip(axes, metrics, labels):
        # Filter data for this metric
        subset = df[df['Metric'] == metric].copy()
        
        # Sort by absolute sensitivity
        subset['abs_sens'] = subset['Sensitivity'].abs()
        subset = subset.sort_values('abs_sens', ascending=True)
        
        # Create horizontal bars
        y_pos = np.arange(len(subset))
        colors = ['#d62728' if x < 0 else '#2ca02c' for x in subset['Sensitivity']]
        
        bars = ax.barh(y_pos, subset['Sensitivity'], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(subset['Parameter'], fontsize=10)
        ax.set_xlabel('Normalized Sensitivity Coefficient S', fontweight='bold', fontsize=10)
        ax.set_title(label, fontweight='bold', fontsize=12, pad=10)
        ax.axvline(0, color='black', linewidth=1.5, linestyle='-')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (idx, row) in enumerate(subset.iterrows()):
            val = row['Sensitivity']
            ax.text(val, i, f' {val:.2f}', va='center', 
                   ha='left' if val > 0 else 'right', fontsize=9, fontweight='bold')
    
    # Main title
    fig.suptitle(f'Tornado Diagram: {mechanism}', fontsize=15, fontweight='bold', y=0.98)
    
    # Add legend (manual patches)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', alpha=0.7, edgecolor='black', label='Positive: ↑Parameter → ↑Output'),
        Patch(facecolor='#d62728', alpha=0.7, edgecolor='black', label='Negative: ↑Parameter → ↓Output')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, 
               frameon=True, fontsize=10, bbox_to_anchor=(0.5, 0.96))
    
    # Add methodology note
    fig.text(0.5, 0.01, 
             f'Sensitivity S = (ΔY/Y₀)/(ΔP/P₀) | Perturbation: ±{perturbation*100:.0f}% | '
             f'Simulations: {num_simulations:,} per evaluation | Sorted by |S| (low to high)',
             ha='center', fontsize=9, style='italic', color='#333333')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_dir}/OAT_sensitivity_{mechanism}_tornado.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Tornado diagram saved to {output_dir}/OAT_sensitivity_{mechanism}_tornado.png")


def plot_seaborn_facets(df, mechanism, output_dir):
    """Original seaborn faceted plot (requires seaborn)."""
    sns.set_style("whitegrid")
    
    metric_map = {
        'mean_t12': 'Mean T12',
        'std_t12': 'Std T12', 
        'mean_t32': 'Mean T32', 
        'std_t32': 'Std T32'
    }
    df['Metric Label'] = df['Metric'].map(metric_map)
    
    g = sns.catplot(
        data=df, 
        x='Sensitivity', 
        y='Parameter', 
        col='Metric Label',
        kind='bar',
        col_wrap=2,
        height=4, 
        aspect=1.5,
        sharex=False
    )
    
    g.fig.suptitle(f'OAT Sensitivity Analysis: {mechanism}', y=1.02)
    g.set_axis_labels("Sensitivity Coefficient (%/%)", "Parameter")
    
    for ax in g.axes.flatten():
        ax.axvline(0, color='k', linewidth=0.8, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/OAT_sensitivity_{mechanism}_seaborn.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Seaborn plot saved to {output_dir}/OAT_sensitivity_{mechanism}_seaborn.png")


if __name__ == "__main__":
    # Hardcoded parameters as requested
    mechanism = "time_varying_k"
    num_simulations = 20000
    perturbation = 0.10  # 5% perturbation
    
    run_sensitivity_analysis(mechanism, num_simulations, perturbation)
