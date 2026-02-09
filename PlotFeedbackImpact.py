#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import json
from simulation_utils import get_parameter_names, get_parameter_bounds, load_optimized_parameters

# Mapping from mechanism+feedback status to all-fit parameter files
ALL_FIT_FILES = {
    ('time_varying_k', False): 'simulation_optimized_parameters_time_varying_k.txt',
    ('time_varying_k', True): 'simulation_optimized_parameters_time_varying_k_wfeedback.txt',
    ('time_varying_k_fixed_burst', False): 'simulation_optimized_parameters_time_varying_k_fixed_burst.txt',
    ('time_varying_k_fixed_burst', True): 'simulation_optimized_parameters_time_varying_k_fixed_burst_wfeedback.txt',
    ('time_varying_k_steric_hindrance', False): 'simulation_optimized_parameters_time_varying_k_steric_hindrance.txt',
    ('time_varying_k_steric_hindrance', True): 'simulation_optimized_parameters_time_varying_k_steric_hindrance_wfeedback.txt',
    ('time_varying_k_combined', False): 'simulation_optimized_parameters_time_varying_k_combined.txt',
    ('time_varying_k_combined', True): 'simulation_optimized_parameters_time_varying_k_combined_wfeedback.txt',
}

def load_data(feedback_files, no_feedback_files, load_all_fit=True):
    """Load data from defined files and structure it for plotting."""
    all_data = []

    # Helper to process files
    def process_file_list(file_list, group_label):
        for filepath in file_list:
            if not os.path.exists(filepath):
                print(f" Warning: File not found: {filepath}")
                continue

            try:
                df = pd.read_csv(filepath)
                if 'val_emd' not in df.columns:
                    print(f" Warning: 'val_emd' column missing in {filepath}")
                    continue

                # Derived mechanism name from filename
                basename = os.path.basename(filepath)
                
                # Logic to map filenames to official mechanism names and display names
                if 'time_varying_k_combined_wfeedback' in basename:
                    mechanism = 'time_varying_k_combined'
                    display_name = 'time_varying_k_combined (with feedback)'
                elif 'time_varying_k_combined_normal' in basename:
                    mechanism = 'time_varying_k_combined'
                    display_name = 'time_varying_k_combined'
                elif 'time_varying_k_fixed_burst_wfeedback' in basename:
                    mechanism = 'time_varying_k_combined' # Uses combined logic? Or specific? Old code said combined.
                    display_name = 'time_varying_k_fixed_burst (with feedback)'
                elif 'time_varying_k_steric_hindrance_normal' in basename:
                     # User put this in both groups, handling it as distinct
                     mechanism = 'time_varying_k_steric_hindrance'
                     display_name = 'time_varying_k_steric_hindrance'
                elif 'time_varying_k_steric_hindrance_wfeedback' in basename:
                    mechanism = 'time_varying_k_steric_hindrance'
                    display_name = 'time_varying_k_steric_hindrance (with feedback)'
                elif 'time_varying_k_wfeedback' in basename:
                    mechanism = 'time_varying_k_steric_hindrance'
                    display_name = 'time_varying_k (with feedback)'
                elif 'time_varying_k_fixed_burst_normal' in basename or 'time_varying_k_fixed_burst' in basename:
                    mechanism = 'time_varying_k_fixed_burst'
                    display_name = 'time_varying_k_fixed_burst'
                elif 'time_varying_k_normal' in basename or 'time_varying_k' in basename:
                    mechanism = 'time_varying_k'
                    display_name = 'time_varying_k'
                else:
                    mechanism = 'other'
                    display_name = basename

                # Parse parameter data if available
                params_list = []
                if 'params' in df.columns:
                    for params_str in df['params']:
                        try:
                            params_list.append(json.loads(params_str))
                        except:
                            params_list.append(None)
                else:
                    params_list = [None] * len(df)

                all_data.append({
                    'filepath': filepath,
                    'group': group_label,
                    'mechanism': mechanism,
                    'display_name': display_name,
                    'val_emds': df['val_emd'].values,
                    'mean_emd': df['val_emd'].mean(),
                    'sem_emd': df['val_emd'].std() / np.sqrt(len(df)),
                    'params': params_list
                })
                print(f" Loaded {mechanism}: Mean EMD = {df['val_emd'].mean():.2f}")

            except Exception as e:
                print(f" Error reading {filepath}: {e}")

    print("Loading Feedback Models...")
    process_file_list(feedback_files, 'With Feedback')
    
    print("Loading No-Feedback Models...")
    process_file_list(no_feedback_files, 'No Feedback')
    
    # Load all-fit parameters if requested
    if load_all_fit:
        print("\nLoading All-Fit Parameters...")
        for result in all_data:
            mechanism = result['mechanism']
            display_name = result['display_name']
            has_feedback = 'with feedback' in display_name.lower()
            
            # Get the appropriate all-fit file
            file_key = (mechanism, has_feedback)
            if file_key in ALL_FIT_FILES:
                filename = ALL_FIT_FILES[file_key]
                if os.path.exists(filename):
                    try:
                        all_fit_params = load_optimized_parameters(mechanism, filename)
                        if all_fit_params:
                            result['all_fit_params'] = all_fit_params
                            total_emd = all_fit_params.get('total_emd', 'N/A')
                            print(f"   Loaded all-fit for {display_name} (EMD: {total_emd})")
                        else:
                            result['all_fit_params'] = None
                            print(f"    Failed to load all-fit for {display_name}")
                    except Exception as e:
                        result['all_fit_params'] = None
                        print(f"   Error loading all-fit for {display_name}: {e}")
                else:
                    result['all_fit_params'] = None
                    print(f"    All-fit file not found for {display_name}: {filename}")
            else:
                result['all_fit_params'] = None

    return all_data

def plot_mean_comparison(data):
    """Plot mean EMD with error bars."""
    if not data:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Desired order: No Feedback models, then Feedback models
    # Use display_name for ordering to match what was loaded
    order_display_names = [
        'time_varying_k',
        'time_varying_k_fixed_burst',
        'time_varying_k_steric_hindrance',
        'time_varying_k_combined',
        'time_varying_k (with feedback)',
        'time_varying_k_fixed_burst (with feedback)',
        'time_varying_k_steric_hindrance (with feedback)',
        'time_varying_k_combined (with feedback)'
    ]
    
    # Create lookup by display_name
    data_map = {d['display_name']: d for d in data}
    
    x_positions = range(len(order_display_names))
    
    for i, display_name in enumerate(order_display_names):
        entry = data_map.get(display_name)
        if entry:
            # Color logic: Blue for No Feedback, Red for Feedback
            if 'with feedback' in display_name:
                color = 'firebrick'
                label = 'With Feedback'
            else:
                color = 'steelblue'
                label = 'No Feedback'
                
            # Only add label once for legend (logic: first of its group)
            # Group 1 indices: 0, 1. Group 2 indices: 2, 3.
            lbl = label if (i == 0 or i == 2) else ""
            
            ax.errorbar(i, entry['mean_emd'], 
                        yerr=entry['sem_emd'], fmt='o', 
                        color=color, label=lbl,
                        capsize=5, markeredgecolor='black', markersize=8)

    # Formatting
    ax.set_xticks(x_positions)
    
    # Use actual display names from data
    xtick_labels = [name.replace('_', ' ') for name in order_display_names]
    ax.set_xticklabels(xtick_labels, fontsize=10, fontweight='bold', rotation=15)
    
    ax.set_ylabel('Mean Validation EMD', fontsize=12)
    # ax.set_title('Impact of Feedback on Model Performance', fontsize=14)
    
    # Avoid duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_path = 'ModelComparisonEMDResults/feedback_impact_mean_emd.pdf'
    plt.savefig(save_path, dpi=300)
    print(f"\n Mean plot saved to: {save_path}")
    plt.close()

def plot_box_distribution(data):
    """Plot box plots of the distributions."""
    if not data:
        return

    # Specific order requested - use display_name
    order_display_names = [
        'time_varying_k',
        'time_varying_k_fixed_burst',
        'time_varying_k_steric_hindrance',
        'time_varying_k_combined',
        'time_varying_k (with feedback)',
        'time_varying_k_fixed_burst (with feedback)',
        'time_varying_k_steric_hindrance (with feedback)',
        'time_varying_k_combined (with feedback)'
    ]
    
    # Create lookup by display_name
    data_map = {d['display_name']: d for d in data}
    
    plot_vals = []
    plot_labels = []
    plot_colors = []
    
    for display_name in order_display_names:
        entry = data_map.get(display_name)
        if entry:
            plot_vals.append(entry['val_emds'])
            # Format label nicely
            label = display_name.replace('_', ' ')
            plot_labels.append(label)
            
            if 'with feedback' in display_name:
                plot_colors.append('salmon')
            else:
                plot_colors.append('lightblue')

    fig, ax = plt.subplots(figsize=(10, 6))
    
    bp = ax.boxplot(plot_vals, patch_artist=True, widths=0.5)
    
    # Color boxes
    for patch, color in zip(bp['boxes'], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    # Formatting
    ax.set_xticklabels(plot_labels, rotation=15, fontsize=10)
    ax.set_ylabel('Validation EMD Distribution', fontsize=12)
    ax.set_title('Distribution of EMD Scores', fontsize=14)
    
    # Manual Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', label='No Feedback'),
        Patch(facecolor='salmon', label='With Feedback')
    ]
    #ax.legend(handles=legend_elements, loc='upper right')
    
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_path = 'ModelComparisonEMDResults/feedback_impact_boxplot.pdf'
    plt.savefig(save_path, dpi=300)
    print(f" Box plot saved to: {save_path}")
    plt.close()

def plot_parameter_matrix(all_results, run_id=None, save_plots=True):
    """
    Create a matrix plot of parameter distributions across all mechanisms.
    Each row shows one parameter across all mechanisms.
    
    Args:
        all_results (list): List of results dictionaries for all mechanisms
        run_id (str): Run ID for filename (optional)
        save_plots (bool): Whether to save the plot
    """
    # Filter results with valid parameter data
    results_with_params = [
        r for r in all_results 
        if r.get('params') and any(p is not None for p in r['params'])
    ]
    
    if not results_with_params:
        print("  No parameter data available for any mechanism")
        return
    
    # Sort by desired display order (matching mean EMD plot)
    # Order: no-feedback models first, then with-feedback models
    display_name_order = [
        'time_varying_k',
        'time_varying_k_fixed_burst',
        'time_varying_k_steric_hindrance',
        'time_varying_k_combined',
        'time_varying_k (with feedback)',
        'time_varying_k_fixed_burst (with feedback)',
        'time_varying_k_steric_hindrance (with feedback)',
        'time_varying_k_combined (with feedback)'
    ]
    
    results_with_params.sort(
        key=lambda r: display_name_order.index(r['display_name']) 
        if r['display_name'] in display_name_order else 999
    )
    
    # Extract display names
    display_names = [r['display_name'] for r in results_with_params]
    n_mechanisms = len(display_names)
    
    # Define parameter display order
    # Note: Some parameters have different bounds for feedback vs non-feedback
    # so we split them into separate rows
    param_order = [
        'n2', 'N2', 'k/k_max',
        'tau (no feedback)', 'tau (with feedback)',
        'r21', 'r23', 'R21', 'R23',
        'burst_size', 'n_inner', 'alpha',
        'beta_k', 'beta_k1', 'beta_k2', 'beta_k3',
        'beta_tau (no feedback)', 'beta_tau (with feedback)',
        'beta_tau2 (no feedback)', 'beta_tau2 (with feedback)',
        'n1/N1', 'n2/N2', 'n3/N3'
    ]
    
    # Collect all parameters that exist across mechanisms
    all_params = set()
    for result in results_with_params:
        try:
            param_names = get_parameter_names(result['mechanism'])
            for name in param_names:
                # Map k/k_max
                display_name = 'k/k_max' if name in ['k', 'k_max'] else name
                all_params.add(display_name)
        except:
            continue
    
    # Add derived parameters
    all_params.update(['n1/N1', 'n2/N2', 'n3/N3'])
    
    # Sort by defined order
    # Filter params_to_plot to include only those whose BASE name exists in all_params
    # This ensures "tau (no feedback)" is included if "tau" exists in the data
    filtered_params = []
    for p in param_order:
        base_name = p.replace(' (no feedback)', '').replace(' (with feedback)', '')
        if base_name in all_params or p in all_params:
            filtered_params.append(p)
            
    params_to_plot = filtered_params
    n_params = len(params_to_plot)
    
    if n_params == 0:
        print("  No parameters found to plot")
        return
    
    # Create figure
    fig, axes = plt.subplots(
        n_params, 1,
        figsize=(max(12, 1.5 * n_mechanisms), 2.0 * n_params),
        sharex=False,
        squeeze=False
    )
    
    fig.suptitle('Parameter Distributions Across Mechanisms', 
                 fontsize=18, fontweight='bold', y=0.99)
    
    # Plot each parameter
    for param_idx, param_name in enumerate(params_to_plot):
        ax = axes[param_idx, 0]
        
        # Zebra striping
        if param_idx % 2 == 0:
            ax.set_facecolor('#f8f9fa')
        
        # Collect data for this parameter
        param_data = _collect_parameter_data(
            param_name, results_with_params, n_mechanisms
        )
        
        if param_data['has_data']:
            _plot_parameter_row(ax, param_data, n_mechanisms)
        else:
            _plot_empty_row(ax, param_name)
        
        # Common formatting
        ax.set_xlim(-0.5, n_mechanisms - 0.5)
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.set_xticklabels([])
    
    # Add mechanism names at top
    ax_top = axes[0, 0]
    ax_top.xaxis.set_label_position('top')
    ax_top.xaxis.tick_top()
    ax_top.tick_params(axis='x', which='both', top=True, labeltop=True)
    ax_top.set_xticks(range(n_mechanisms))
    ax_top.set_xticklabels(display_names, rotation=45, ha='left', 
                           fontsize=11, fontweight='bold')
    
    plt.subplots_adjust(hspace=0.1, left=0.1, right=0.95, top=0.88, bottom=0.05)
    
    # Save
    if save_plots:
        filename = 'ModelComparisonEMDResults/feedback_impact_matrix.pdf'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n Parameter matrix saved as: {filename}")
    
    plt.close()


def _extract_single_param_value(param_name, param_dict):
    """
    Extract a single parameter value from all-fit parameter dictionary.
    
    Args:
        param_name (str): Parameter name (may include feedback suffix)
        param_dict (dict): All-fit parameters dictionary
    
    Returns:
        float or None: Parameter value
    """
    # Strip feedback/no-feedback suffix
    base_param_name = param_name.replace(' (no feedback)', '').replace(' (with feedback)', '')
    
    # Handle derived parameters
    if base_param_name in ['n1/N1', 'n2/N2', 'n3/N3']:
        if base_param_name == 'n2/N2':
            if 'n2' in param_dict and 'N2' in param_dict:
                return param_dict['n2'] / param_dict['N2']
        elif base_param_name == 'n1/N1':
            if 'n1' in param_dict and 'N1' in param_dict:
                return param_dict['n1'] / param_dict['N1']
        elif base_param_name == 'n3/N3':
            if 'n3' in param_dict and 'N3' in param_dict:
                return param_dict['n3'] / param_dict['N3']
        return None
    
    # Handle k/k_max
    if base_param_name == 'k/k_max':
        if 'k' in param_dict:
            return param_dict['k']
        elif 'k_max' in param_dict:
            return param_dict['k_max']
        return None
    
    # Regular parameters
    if base_param_name in param_dict:
        return param_dict[base_param_name]
    
    return None


def _collect_parameter_data(param_name, results_with_params, n_mechanisms):
    """Helper to collect parameter values across mechanisms."""
    box_data = [None] * n_mechanisms
    point_data = [None] * n_mechanisms
    all_fit_values = [None] * n_mechanisms  # NEW: Store all-fit values
    param_bound = None
    
    # Check if this is a split parameter (feedback vs non-feedback)
    is_split_param = '(no feedback)' in param_name or '(with feedback)' in param_name
    base_param_name = param_name.replace(' (no feedback)', '').replace(' (with feedback)', '')
    
    for mech_idx, result in enumerate(results_with_params):
        mechanism = result['mechanism']
        display_name = result['display_name']
        params = result['params']
        
        if not params or all(p is None for p in params):
            continue
        
        # For split parameters, only process matching mechanisms
        if is_split_param:
            has_feedback = 'with feedback' in display_name
            if '(no feedback)' in param_name and has_feedback:
                continue  # Skip feedback mechanisms for no-feedback params
            if '(with feedback)' in param_name and not has_feedback:
                continue  # Skip non-feedback mechanisms for feedback params
        
        try:
            param_names = get_parameter_names(mechanism)
            param_bounds = get_parameter_bounds(mechanism)
        except:
            continue
        
        # Get parameter values (CV folds)
        param_values = _extract_param_values(
            param_name, param_names, params
        )
        
        if param_values is not None and len(param_values) > 0:
            box_data[mech_idx] = param_values
            point_data[mech_idx] = param_values
            
            # Get bounds for regular parameters
            if param_bound is None and param_name not in ['n1/N1', 'n2/N2', 'n3/N3']:
                # Strip the suffix for split parameters
                base_param_name = param_name.replace(' (no feedback)', '').replace(' (with feedback)', '')
                actual_name = 'k' if base_param_name == 'k/k_max' and 'k' in param_names else base_param_name
                if actual_name == 'k/k_max':
                    actual_name = 'k_max' if 'k_max' in param_names else None
                if actual_name and actual_name in param_names:
                    idx = param_names.index(actual_name)
                    if idx < len(param_bounds):
                        param_bound = param_bounds[idx]
            
            # Manual bound overrides for specific split parameters
            if 'tau' in base_param_name and 'beta' not in base_param_name:
                if '(no feedback)' in param_name:
                    param_bound = [2, 240]
                elif '(with feedback)' in param_name:
                    param_bound = [0.5, 5]
            
            elif 'beta_tau2' in base_param_name:
                if '(no feedback)' in param_name:
                    param_bound = [1, 20]
                elif '(with feedback)' in param_name:
                    param_bound = [1, 3]
                    
            elif 'beta_tau' in base_param_name: # Check this AFTER beta_tau2 to avoid substring match issues if not careful, though here names are distinct
                if '(no feedback)' in param_name:
                    param_bound = [1, 10]
                elif '(with feedback)' in param_name:
                    param_bound = [1, 3]
        
        # NEW: Extract all-fit value if available
        if 'all_fit_params' in result and result['all_fit_params']:
            all_fit_params = result['all_fit_params']
            all_fit_value = _extract_single_param_value(param_name, all_fit_params)
            if all_fit_value is not None:
                all_fit_values[mech_idx] = all_fit_value
    
    # Filter valid data
    valid_positions = [i for i, data in enumerate(box_data) if data is not None]
    valid_box_data = [box_data[i] for i in valid_positions]
    valid_point_data = [point_data[i] for i in valid_positions]
    valid_all_fit = [all_fit_values[i] for i in valid_positions]  # NEW
    
    return {
        'has_data': len(valid_box_data) > 0,
        'positions': valid_positions,
        'box_data': valid_box_data,
        'point_data': valid_point_data,
        'all_fit_values': valid_all_fit,  # NEW
        'bound': param_bound,
        'name': param_name
    }



def _extract_param_values(param_name, param_names, params):
    """Extract parameter values for a given parameter name."""
    # Strip feedback/no-feedback suffix for split parameters
    base_param_name = param_name.replace(' (no feedback)', '').replace(' (with feedback)', '')
    
    valid_params = [p for p in params if p is not None]
    if not valid_params:
        return None
        
    is_dict = isinstance(valid_params[0], dict)

    # Helper to get column values
    def get_col(name):
        if is_dict:
            # For dict, just look up key
            vals = [p.get(name) for p in valid_params]
            # Convert None to nan for array creation
            vals = [v if v is not None else np.nan for v in vals]
            arr = np.array(vals)
            # If all are nan, return None
            if np.all(np.isnan(arr)): return None
            return arr
        else:
            # Fallback to list index
            if name in param_names:
                idx = param_names.index(name)
                # Check bounds
                if len(valid_params[0]) > idx:
                     return np.array([p[idx] for p in valid_params])
            return None
    
    # Handle derived parameters
    if base_param_name in ['n1/N1', 'n2/N2', 'n3/N3']:
        n2_vals = get_col('n2')
        N2_vals = get_col('N2')
        
        if n2_vals is None or N2_vals is None:
            return None
        
        if base_param_name == 'n2/N2':
            return n2_vals / N2_vals
        elif base_param_name == 'n1/N1':
            r21 = get_col('r21')
            R21 = get_col('R21')
            if r21 is not None and R21 is not None:
                n1_vals = np.maximum(r21 * n2_vals, 1.0)
                N1_vals = np.maximum(R21 * N2_vals, 1.0)
                return n1_vals / N1_vals
        elif base_param_name == 'n3/N3':
            r23 = get_col('r23')
            R23 = get_col('R23')
            if r23 is not None and R23 is not None:
                n3_vals = np.maximum(r23 * n2_vals, 1.0)
                N3_vals = np.maximum(R23 * N2_vals, 1.0)
                return n3_vals / N3_vals
        return None
    
    # Handle regular parameters
    if base_param_name == 'k/k_max':
        # Try k then k_max
        vals = get_col('k')
        if vals is None:
            vals = get_col('k_max')
        return vals
    else:
        # Direct lookup (e.g. tau, beta_tau, n_inner)
        return get_col(base_param_name)


def _plot_parameter_row(ax, param_data, n_mechanisms):
    """Plot a single parameter row with box plots and scatter points."""
    # Box plots
    bp = ax.boxplot(
        param_data['box_data'],
        positions=param_data['positions'],
        widths=0.4,
        patch_artist=True,
        boxprops=dict(facecolor='#e0e0e0', alpha=0.5, edgecolor='#666666'),
        medianprops=dict(color='#d62728', linewidth=1.5),
        whiskerprops=dict(linewidth=1, color='#666666'),
        capprops=dict(linewidth=1, color='#666666'),
        flierprops=dict(marker='o', markersize=3, alpha=0.5)
    )
    
    # Scatter points with colors (CV fold data)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for pos, values in zip(param_data['positions'], param_data['point_data']):
        n_points = len(values)
        x_jitter = np.clip(
            np.random.normal(pos, 0.05, size=n_points),
            pos - 0.2, pos + 0.2
        )
        
        for point_idx, (x, y) in enumerate(zip(x_jitter, values)):
            color = colors[point_idx % len(colors)]
            ax.scatter(x, y, alpha=0.7, s=40, color=color,
                      edgecolors='white', linewidths=0.5, zorder=3)
    
    # NEW: Plot all-fit values with distinct markers
    if 'all_fit_values' in param_data:
        for pos, all_fit_val in zip(param_data['positions'], param_data['all_fit_values']):
            if all_fit_val is not None:
                ax.scatter(pos, all_fit_val, marker='D', s=120, color='black',
                          edgecolors='gold', linewidths=2, zorder=5, 
                          label='All-Fit' if pos == param_data['positions'][0] else '')
    
    # Set y-axis limits
    if param_data['bound']:
        if param_data['name'] in ['n1/N1', 'n2/N2', 'n3/N3']:
            max_val = max([np.max(d) for d in param_data['box_data']])
            ax.set_ylim(0, max(max_val * 1.2, 0.05))
        else:
            ax.set_ylim(param_data['bound'][0], param_data['bound'][1])
    
    # Labels and grid
    ax.set_ylabel(param_data['name'], fontsize=12, fontweight='bold', 
                  rotation=0, labelpad=40, ha='right', va='center')
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Vertical separators
    for i in range(n_mechanisms - 1):
        ax.axvline(i + 0.5, color='gray', linestyle=':', alpha=0.3)


def _plot_empty_row(ax, param_name):
    """Plot an empty row for parameters not used by any mechanism."""
    ax.text(0.5, 0.5, f'{param_name} (Not used)',
           ha='center', va='center', transform=ax.transAxes,
           fontsize=10, color='gray', style='italic')
    ax.set_yticks([])
    ax.set_ylabel(param_name, fontsize=12, fontweight='bold', color='gray')



def main():
    print("="*60)
    print("GENERATING FEEDBACK COMPARISON PLOTS")
    print("="*60)
    
    # Define file paths HERE in main
    FEEDBACK_FILES = [
        'ModelComparisonEMDResults/cv_results_time_varying_k_fixed_burst_wfeedback_1.csv',
        'ModelComparisonEMDResults/cv_results_time_varying_k_wfeedback_1.csv',
        'ModelComparisonEMDResults/cv_results_time_varying_k_combined_wfeedback_1.csv',
        'ModelComparisonEMDResults/cv_results_time_varying_k_steric_hindrance_wfeedback_1.csv'
    ]

    NO_FEEDBACK_FILES = [
        'ModelComparisonEMDResults/cv_results_time_varying_k_fixed_burst_normal_1.csv',
        'ModelComparisonEMDResults/cv_results_time_varying_k_normal_1.csv',
        'ModelComparisonEMDResults/cv_results_time_varying_k_combined_normal_1.csv',
        'ModelComparisonEMDResults/cv_results_time_varying_k_steric_hindrance_normal_1.csv'
    ]
    
    data = load_data(FEEDBACK_FILES, NO_FEEDBACK_FILES)
    
    if not data:
        print("❌ No data loaded. Check paths.")
        return

    plot_mean_comparison(data)
    plot_box_distribution(data)
    plot_parameter_matrix(data)
    
    print("\n Done.")

if __name__ == "__main__":
    main()