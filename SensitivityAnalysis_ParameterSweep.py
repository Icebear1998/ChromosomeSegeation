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
from scipy.interpolate import UnivariateSpline
import sys
import os

import simulation_utils

def run_parameter_sweep(mechanism, num_simulations=5000, num_points=20, n_repeat=3, oat_perturbation=0.10):
    """
    Run parameter sweep sensitivity analysis.
    
    Args:
        mechanism (str): Name of the mechanism (e.g., 'time_varying_k').
        num_simulations (int): Number of simulations per evaluation.
        num_points (int): Number of points to sample across parameter bounds.
        n_repeat (int): Number of repeat simulations at each parameter value for confidence intervals.
        oat_perturbation (float): Perturbation for OAT sensitivity (e.g., 0.10 for 10%).
    """
    print(f"--- Starting Parameter Sweep Analysis for '{mechanism}' ---")
    print(f"Simulations per evaluation: {num_simulations}")
    print(f"Sample points: {num_points} per parameter")
    print(f"Repeats per point: {n_repeat} (for confidence intervals)")
    print(f"OAT Perturbation: {oat_perturbation*100}%")
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
    all_slope_results = []

    
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
        else:
             # Ensure exact optimized value is included
             sweep_values = np.sort(np.unique(np.append(sweep_values, optimized_val)))

            
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
            mean_t12_avg = np.mean(mean_t12_repeats)
            mean_t12_std = np.std(mean_t12_repeats)
            mean_t12_sem = mean_t12_std / np.sqrt(n_repeat)
            
            std_t12_avg = np.mean(std_t12_repeats)
            std_t12_std = np.std(std_t12_repeats)
            std_t12_sem = std_t12_std / np.sqrt(n_repeat)
            
            mean_t32_avg = np.mean(mean_t32_repeats)
            mean_t32_std = np.std(mean_t32_repeats)
            mean_t32_sem = mean_t32_std / np.sqrt(n_repeat)
            
            std_t32_avg = np.mean(std_t32_repeats)
            std_t32_std = np.std(std_t32_repeats)
            std_t32_sem = std_t32_std / np.sqrt(n_repeat)
            
            results.append({
                'Parameter': param_name,
                'Value': value,
                # Mean of T12 across repeats (in seconds)
                'Mean_T12': mean_t12_avg,
                'Mean_T12_std': mean_t12_std,  # Variation across repeats
                'Mean_T12_sem': mean_t12_sem,  # Standard Error of Mean
                # Std of T12 across repeats (in seconds)
                'Std_T12': std_t12_avg,
                'Std_T12_std': std_t12_std,
                'Std_T12_sem': std_t12_sem,
                # Mean of T32 across repeats (in seconds)
                'Mean_T32': mean_t32_avg,
                'Mean_T32_std': mean_t32_std,
                'Mean_T32_sem': mean_t32_sem,
                # Std of T32 across repeats (in seconds)
                'Std_T32': std_t32_avg,
                'Std_T32_std': std_t32_std,
                'Std_T32_sem': std_t32_sem
            })
        
        print(f"\n  Completed sweep for {param_name}")
        
        # Convert to DataFrame
        df_param = pd.DataFrame(results)
        all_results.append(df_param)
        
        # 3.2. Calculate Fits and Slopes
        fits, slope_data = calculate_fits_and_slopes(
            df_param, optimized_val, param_name, mechanism, params, 
            num_simulations, oat_perturbation
        )
        
        # Update slope data with Parameter Name
        for item in slope_data:
            item['Parameter'] = param_name
        all_slope_results.extend(slope_data)
        
        # Save fit objects for plotting later (or plot directly)
        # We need to store fits somewhere if we separate plotting loop
        # But we can iterate over them in parallel because lists are ordered
        # Let's attach fits to the DataFrame as metadata attribute (not standard pandas but works for passing around)
        df_param.fits = fits 

    
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
        
        # Retrieve fits if attached
        fits = getattr(df_param, 'fits', None)
        
        # Filter slope data for this parameter from all_slope_results
        # Or just use the 'slope_data' we got from calculate_fits_and_slopes if we stored it?
        # We didn't store per-param slope_data list, but we can filter all_slope_results
        # Actually easier to just attach it to df_param like fits
        
        # Re-calculate or retrieve - wait, run_parameter_sweep is one big scope
        # In the loop: fits, slope_data = calculate_fits_and_slopes...
        # We need to access that 'slope_data' here. 
        # But 'slope_data' variable is overwritten in each loop iteration.
        # So we can't access it here easily unless we stored it.
        # Let's verify where plot_parameter_sweep is called. 
        # Ah, it's called in a separate loop AFTER all sweeps.
        
        # So we need to store slope_data associated with each param.
        # Let's attach it to df_param too.
        current_slopes = [s for s in all_slope_results if s['Parameter'] == param_name]
        
        plot_parameter_sweep(df_param, param_name, optimized_val, mechanism, 
                           output_dir, n_repeat, global_limits, fits, current_slopes)
    
    # 4. Save All Data
    df_all = pd.concat(all_results, ignore_index=True)
    csv_path = f"{output_dir}/parameter_sweep_{mechanism}_full.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"\n\nAll results saved to {csv_path}")
    
    # 5. Save Slopes Data
    if all_slope_results:
        df_slopes = pd.DataFrame(all_slope_results)
        # Reorder columns
        cols = ['Parameter', 'Metric', 'Method', 'Slope', 'Normalized_Slope', 'R2', 
                'Value_at_Optimum', 'Metric_at_Optimum']
        df_slopes = df_slopes[cols]
        slope_path = f"{output_dir}/sensitivity_slopes_{mechanism}.csv"
        df_slopes.to_csv(slope_path, index=False)
        print(f"Sensitivity slopes saved to {slope_path}")


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


def calculate_fits_and_slopes(df, optimized_val, param_name, mechanism, params, num_simulations, oat_perturbation):
    """
    Fit multiple curves (Poly1-3, Spline) to Mean/Std data and calculate slopes at optimum.
    
    Args:
        df: DataFrame with sweep data
        optimized_val: Optimized parameter value
        param_name: Name of parameter being varied
        mechanism: Mechanism name
        params: Full parameter dict
        num_simulations: Number of simulations for OAT
        oat_perturbation: Perturbation fraction for OAT (e.g., 0.10)
    
    Returns:
        fits: dict of fitted functions/data for plotting
        slopes: list of dicts with slope data (includes OAT, noFit, and fitted methods)
    """
    targets = ['Mean_T12', 'Std_T12', 'Mean_T32', 'Std_T32']
    x = df['Value'].values
    
    fits = {}
    slope_data = []
    
    # Find index of optimized value (or closest if float issues, but we added it explicitly)
    # Using small tolerance for float comparison
    idx_opt = np.argmin(np.abs(x - optimized_val))
    val_at_opt_check = x[idx_opt]
    
    # Store the parameter value at optimum (not the metric value!)
    param_val_at_opt = optimized_val
    
    # Get baseline metrics at optimum for OAT calculation
    # We need to run baseline simulation
    print(f"    Running OAT baseline simulation for {param_name}...")
    params_base = params.copy()
    update_derived_params(params_base)
    n0_list_base = [params_base['n1'], params_base['n2'], params_base['n3']]
    
    t12_base, t32_base = simulation_utils.run_simulation_for_dataset(
        mechanism, params_base, n0_list_base, num_simulations=num_simulations
    )
    
    base_metrics = {
        'Mean_T12': np.mean(t12_base),
        'Std_T12': np.std(t12_base),
        'Mean_T32': np.mean(t32_base),
        'Std_T32': np.std(t32_base)
    }
    
    for target in targets:
        y_sweep = df[target].values  # Get the metric values from sweep data
        
        # Get the metric value at optimum for normalized sensitivity calculation
        y_at_opt = base_metrics[target]
        
        # 0A. OAT Method: Run simulations at ±perturbation to calculate sensitivity
        # This matches the implementation from SensitivityAnalysis_OAT.py
        try:
            print(f"    Running OAT for {param_name} -> {target}...")
            
            # Perturb +
            params_plus = params.copy()
            params_plus[param_name] = optimized_val * (1 + oat_perturbation)
            update_derived_params(params_plus)
            n0_list_plus = [params_plus['n1'], params_plus['n2'], params_plus['n3']]
            
            t12_p, t32_p = simulation_utils.run_simulation_for_dataset(
                mechanism, params_plus, n0_list_plus, num_simulations=num_simulations
            )
            
            metrics_plus = {
                'Mean_T12': np.mean(t12_p),
                'Std_T12': np.std(t12_p),
                'Mean_T32': np.mean(t32_p),
                'Std_T32': np.std(t32_p)
            }
            
            # Perturb -
            params_minus = params.copy()
            params_minus[param_name] = optimized_val * (1 - oat_perturbation)
            update_derived_params(params_minus)
            n0_list_minus = [params_minus['n1'], params_minus['n2'], params_minus['n3']]
            
            t12_m, t32_m = simulation_utils.run_simulation_for_dataset(
                mechanism, params_minus, n0_list_minus, num_simulations=num_simulations
            )
            
            metrics_minus = {
                'Mean_T12': np.mean(t12_m),
                'Std_T12': np.std(t12_m),
                'Mean_T32': np.mean(t32_m),
                'Std_T32': np.std(t32_m)
            }
            
            # Calculate OAT sensitivity coefficient
            # Matches SensitivityAnalysis_OAT.py lines 170-181
            y_base = base_metrics[target]
            y_plus = metrics_plus[target]
            y_minus = metrics_minus[target]
            
            if abs(y_base) < 1e-9:
                oat_sensitivity = np.nan
                oat_slope = np.nan
            else:
                # S = average of S+ and S-
                # S+ = ((Y+ - Yb)/Yb) / perturbation
                # S- = ((Y- - Yb)/Yb) / (-perturbation)
                s_plus = ((y_plus - y_base) / y_base) / oat_perturbation
                s_minus = ((y_minus - y_base) / y_base) / (-oat_perturbation)
                oat_sensitivity = (s_plus + s_minus) / 2
                
                # Calculate equivalent absolute slope for comparison
                # slope = dY/dX ≈ (Y+ - Y-) / (2 * ΔX)
                delta_x = 2 * optimized_val * oat_perturbation
                oat_slope = (y_plus - y_minus) / delta_x
            
            slope_data.append({
                'Metric': target,
                'Method': 'OAT',
                'Slope': oat_slope,
                'Normalized_Slope': oat_sensitivity,
                'R2': np.nan,  # R² doesn't apply to OAT
                'Value_at_Optimum': param_val_at_opt,
                'Metric_at_Optimum': y_at_opt
            })
            
        except Exception as e:
            print(f"    OAT calculation failed for {target}: {e}")
        
        # 0B. No-Fit Method: Calculate empirical slope from raw data using finite differences
        y_sweep = df[target].values  # Get the metric values from sweep data
        try:
            # Use central difference if possible, otherwise forward/backward
            if idx_opt > 0 and idx_opt < len(x) - 1:
                # Central difference: (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
                dx = x[idx_opt + 1] - x[idx_opt - 1]
                dy = y_sweep[idx_opt + 1] - y_sweep[idx_opt - 1]
                nofit_slope = dy / dx if abs(dx) > 1e-10 else np.nan
            elif idx_opt == 0:
                # Forward difference: (y[i+1] - y[i]) / (x[i+1] - x[i])
                dx = x[1] - x[0]
                dy = y_sweep[1] - y_sweep[0]
                nofit_slope = dy / dx if abs(dx) > 1e-10 else np.nan
            else:
                # Backward difference: (y[i] - y[i-1]) / (x[i] - x[i-1])
                dx = x[-1] - x[-2]
                dy = y_sweep[-1] - y_sweep[-2]
                nofit_slope = dy / dx if abs(dx) > 1e-10 else np.nan
            
            # Calculate normalized sensitivity
            if abs(y_at_opt) > 1e-10:
                nofit_normalized_slope = nofit_slope * (param_val_at_opt / y_at_opt)
            else:
                nofit_normalized_slope = np.nan
            
            slope_data.append({
                'Metric': target,
                'Method': 'noFit',
                'Slope': nofit_slope,
                'Normalized_Slope': nofit_normalized_slope,
                'R2': np.nan,  # R² doesn't apply to raw data
                'Value_at_Optimum': param_val_at_opt,
                'Metric_at_Optimum': y_at_opt
            })
        except Exception as e:
            print(f"  noFit calculation failed for {target}: {e}")
        
        # 1. Polynomial Fits
        poly_fits = {}
        degrees = {1: 'Linear', 2: 'Quadratic', 3: 'Cubic'}
        
        for deg, name in degrees.items():
            try:
                # Fit polynomial
                p = np.poly1d(np.polyfit(x, y_sweep, deg))
                poly_fits[name] = p
                
                # Calculate slope at optimum (absolute)
                slope = p.deriv()(optimized_val)
                
                # Calculate normalized sensitivity: (dY/dX) * (X0/Y0)
                # This gives a dimensionless measure (elasticity)
                # Represents: % change in output per % change in input
                if abs(y_at_opt) > 1e-10:  # Avoid division by zero
                    normalized_slope = slope * (param_val_at_opt / y_at_opt)
                else:
                    normalized_slope = np.nan
                
                # Calculate R-squared
                y_pred = p(x)
                ss_res = np.sum((y_sweep - y_pred) ** 2)
                ss_tot = np.sum((y_sweep - np.mean(y_sweep)) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                
                slope_data.append({
                    'Metric': target,
                    'Method': name,
                    'Slope': slope,
                    'Normalized_Slope': normalized_slope,
                    'R2': r2,
                    'Value_at_Optimum': param_val_at_opt,
                    'Metric_at_Optimum': y_at_opt
                })
            except Exception as e:
                print(f"  Fit validation failed for {target} {name}: {e}")
        
        # 2. Spline Fit (Smoothing)
        try:
            # Sort x for spline (just in case, though df should be sorted)
            sort_idx = np.argsort(x)
            xs = x[sort_idx]
            ys = y_sweep[sort_idx]
            
            # Use UnivariateSpline with default smoothing (s=None)
            # or try a small s if data is noisy. 
            # Given we have Means from N=5000 simulations, data is relatively smooth but has stochasticity.
            # Let's use s=None which tries to find a good smoothing factor automatically
            spl = UnivariateSpline(xs, ys)
            
            slope = spl.derivative()(optimized_val)
            
            # Calculate normalized sensitivity for spline too
            if abs(y_at_opt) > 1e-10:
                normalized_slope = slope * (param_val_at_opt / y_at_opt)
            else:
                normalized_slope = np.nan
            
            # For plotting, we need to generate dense points
            x_dense = np.linspace(min(x), max(x), 200)
            y_dense = spl(x_dense)
            
            poly_fits['Spline'] = (x_dense, y_dense)
            
            slope_data.append({
                'Metric': target,
                'Method': 'Spline',
                'Slope': slope,
                'Normalized_Slope': normalized_slope,
                'R2': np.nan, # Harder to ensure consistent R2 for spline
                'Value_at_Optimum': param_val_at_opt,
                'Metric_at_Optimum': y_at_opt
            })
            
        except Exception as e:
            print(f"  Spline fit failed for {target}: {e}")
            
        fits[target] = poly_fits
        
    return fits, slope_data


def plot_parameter_sweep(df, param_name, optimized_val, mechanism, output_dir, n_repeat, global_limits=None, fits=None, slope_data=None):
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
        fits: Dict of fit functions/data calculated by calculate_fits_and_slopes
        slope_data: List of dicts with slope information
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
    labs = ['Mean T12', 'Std T12', f'Optimized: {optimized_val:.3f}']
    
    # PLOT FITS for T12
    if fits:
        x_plot = np.linspace(df['Value'].min(), df['Value'].max(), 200)
        colors_fit = {'Linear': 'gold', 'Quadratic': 'purple', 'Cubic': 'cyan', 'Spline': 'magenta'}
        linestyles = {'Linear': ':', 'Quadratic': '--', 'Cubic': '-.', 'Spline': '-'}
        
        # Mean T12 Fits
        for method, fit_obj in fits.get('Mean_T12', {}).items():
            if method == 'Spline':
                xs, ys = fit_obj
                l, = ax1.plot(xs, ys, color=colors_fit.get(method, 'black'), alpha=0.8, 
                              linestyle=linestyles.get(method, '-'), linewidth=1.5, label=method)
            else:
                l, = ax1.plot(x_plot, fit_obj(x_plot), color=colors_fit.get(method, 'black'), alpha=0.8, 
                              linestyle=linestyles.get(method, '-'), linewidth=1.5, label=method)
            lns.append(l)
            labs.append(method)
            
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
    
    # PLOT FITS for T32
    if fits:
        x_plot = np.linspace(df['Value'].min(), df['Value'].max(), 200)
        
        # Mean T32 Fits (same style logic)
        for method, fit_obj in fits.get('Mean_T32', {}).items():
            if method == 'Spline':
                xs, ys = fit_obj
                l32, = ax3.plot(xs, ys, color=colors_fit.get(method, 'black'), alpha=0.8, 
                                linestyle=linestyles.get(method, '-'), linewidth=1.5, label=method)
            else:
                l32, = ax3.plot(x_plot, fit_obj(x_plot), color=colors_fit.get(method, 'black'), alpha=0.8, 
                                linestyle=linestyles.get(method, '-'), linewidth=1.5, label=method)
            lns2.append(l32)
            labs2.append(method)

    ax3.legend(lns2, labs2, loc='best', framealpha=0.9)
    
    # Add Text Box with Slopes
    if slope_data:
        slope_text = "Slopes at Opt:\n"
        # Organization: Metric -> [Method:Slope]
        
        metrics_map = {'Mean_T12': 'M12', 'Std_T12': 'S12', 'Mean_T32': 'M32', 'Std_T32': 'S32'}
        metrics_map = {'Mean_T12': 'M12', 'Std_T12': 'S12', 'Mean_T32': 'M32', 'Std_T32': 'S32'}
        methods_to_show = ['Quadratic', 'Spline'] # Show these two as representatives
        
        # Group by metric
        from collections import defaultdict
        metric_slopes = defaultdict(dict)
        for item in slope_data:
            metric_slopes[item['Metric']][item['Method']] = item['Slope']
            
        for metric, label in metrics_map.items():
            if metric in metric_slopes:
                vals = []
                for m in methods_to_show:
                    if m in metric_slopes[metric]:
                        s = metric_slopes[metric][m]
                        vals.append(f"{m}={s:.2e}")
                
                if vals:
                    slope_text += f"{label}: {', '.join(vals)}\n"
        
        # Place text box in the first subplot (T12) or spread?
        # Let's put it on the T32 plot (right side) top right corner
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax3.text(1.1, 0.98, slope_text, transform=ax3.transAxes, fontsize=8, 
                verticalalignment='top', bbox=props)

    
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
    oat_perturbation = 0.10  # OAT perturbation (10%)
    
    run_parameter_sweep(mechanism, num_simulations, num_points, n_repeat, oat_perturbation)

