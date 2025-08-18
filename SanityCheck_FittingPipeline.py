#!/usr/bin/env python3
"""
Sanity Check for Fitting Pipeline

This script performs a comprehensive sanity check on the fitting pipeline:
1. Generate synthetic data using MultiMechanismSimulation.py with known parameters
2. Test if MoMOptimization_join.py can recover the parameters
3. Test if MoMOptimization_independent.py can recover the parameters
4. Compare the recovered parameters with the true parameters

This helps validate that the optimization methods are working correctly.
"""

import numpy as np
import pandas as pd
import os
import sys
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add the SecondVersion directory to the path
sys.path.append('SecondVersion')

from MultiMechanismSimulation import MultiMechanismSimulation
from MoMCalculations import compute_pdf_for_mechanism


def generate_synthetic_data(mechanism, true_params, num_simulations=1000, max_time=1000):
    """
    Generate synthetic data using MultiMechanismSimulation with known parameters.
    
    Args:
        mechanism (str): Mechanism name
        true_params (dict): True parameters used to generate data
        num_simulations (int): Number of simulations to run
        max_time (float): Maximum simulation time
    
    Returns:
        dict: Synthetic datasets for all strains
    """
    print(f"Generating synthetic data for mechanism: {mechanism}")
    print(f"True parameters: {true_params}")
    
    # Extract parameters
    n1, n2, n3 = true_params['n1'], true_params['n2'], true_params['n3']
    N1, N2, N3 = true_params['N1'], true_params['N2'], true_params['N3']
    k = true_params['k']
    
    # Mechanism-specific parameters
    mech_params = {}
    if mechanism == 'fixed_burst':
        mech_params['burst_size'] = true_params['burst_size']
    elif mechanism == 'time_varying_k':
        mech_params['k_1'] = true_params['k_1']
    elif mechanism == 'feedback':
        mech_params['feedbackSteepness'] = true_params['feedbackSteepness']
        mech_params['feedbackThreshold'] = true_params['feedbackThreshold']
    elif mechanism == 'feedback_linear':
        mech_params['w1'] = true_params['w1']
        mech_params['w2'] = true_params['w2']
        mech_params['w3'] = true_params['w3']
    elif mechanism == 'feedback_onion':
        mech_params['n_inner'] = true_params['n_inner']
    elif mechanism == 'feedback_zipper':
        mech_params['z1'] = true_params['z1']
        mech_params['z2'] = true_params['z2']
        mech_params['z3'] = true_params['z3']
    elif mechanism == 'fixed_burst_feedback_linear':
        mech_params['burst_size'] = true_params['burst_size']
        mech_params['w1'] = true_params['w1']
        mech_params['w2'] = true_params['w2']
        mech_params['w3'] = true_params['w3']
    elif mechanism == 'fixed_burst_feedback_onion':
        mech_params['burst_size'] = true_params['burst_size']
        mech_params['n_inner'] = true_params['n_inner']
    
    # Mutant parameters
    alpha = true_params['alpha']
    beta_k = true_params['beta_k']
    beta2_k = true_params['beta2_k']
    
    datasets = {}
    
    # Generate data for each strain
    strains = ['wildtype', 'threshold', 'degrate', 'degrateAPC']
    
    for strain in strains:
        print(f"  Generating {strain} data...")
        
        # Set parameters based on strain
        if strain == 'wildtype':
            strain_n1, strain_n2, strain_n3 = n1, n2, n3
            strain_N1, strain_N2, strain_N3 = N1, N2, N3
            strain_k = k
        elif strain == 'threshold':
            strain_n1, strain_n2, strain_n3 = n1 * alpha, n2 * alpha, n3 * alpha
            strain_N1, strain_N2, strain_N3 = N1, N2, N3
            strain_k = k
        elif strain == 'degrate':
            strain_n1, strain_n2, strain_n3 = n1, n2, n3
            strain_N1, strain_N2, strain_N3 = N1, N2, N3
            strain_k = k * beta_k
        elif strain == 'degrateAPC':
            strain_n1, strain_n2, strain_n3 = n1, n2, n3
            strain_N1, strain_N2, strain_N3 = N1, N2, N3
            strain_k = k * beta2_k
        
        # Ensure minimum values
        strain_n1, strain_n2, strain_n3 = max(strain_n1, 1), max(strain_n2, 1), max(strain_n3, 1)
        strain_N1, strain_N2, strain_N3 = max(strain_N1, 1), max(strain_N2, 1), max(strain_N3, 1)
        strain_k = max(strain_k, 0.001)
        
        # Run simulations
        delta_t12_list = []
        delta_t32_list = []
        
        for i in range(num_simulations):
            try:
                # Initialize simulation
                initial_state = [int(strain_N1), int(strain_N2), int(strain_N3)]
                n0_list = [int(strain_n1), int(strain_n2), int(strain_n3)]
                
                # Create rate parameters
                rate_params = {'k': strain_k}
                rate_params.update(mech_params)
                
                # Run simulation
                sim = MultiMechanismSimulation(
                    mechanism=mechanism,
                    initial_state_list=initial_state,
                    rate_params=rate_params,
                    n0_list=n0_list,
                    max_time=max_time
                )
                
                times, states, separate_times = sim.simulate()
                
                # Calculate delta times
                if separate_times[0] is not None and separate_times[1] is not None:
                    delta_t12 = separate_times[0] - separate_times[1]  # T1 - T2
                    delta_t12_list.append(delta_t12)
                
                if separate_times[2] is not None and separate_times[1] is not None:
                    delta_t32 = separate_times[2] - separate_times[1]  # T3 - T2
                    delta_t32_list.append(delta_t32)
                    
            except Exception as e:
                print(f"    Warning: Simulation {i} failed for {strain}: {e}")
                continue
        
        # Store results
        datasets[strain] = {
            'delta_t12': np.array(delta_t12_list),
            'delta_t32': np.array(delta_t32_list)
        }
        
        print(f"    Generated {len(delta_t12_list)} T1-T2 points, {len(delta_t32_list)} T3-T2 points")
    
    return datasets


def save_synthetic_data(datasets, filename="synthetic_data_sanity_check.xlsx"):
    """
    Save synthetic data to Excel file in the same format as the original data.
    
    Args:
        datasets (dict): Synthetic datasets
        filename (str): Output filename
    """
    # Create DataFrame with the same structure as the original data
    data_dict = {}
    
    for strain in ['wildtype', 'threshold', 'degrate', 'degrateAPC']:
        if strain in datasets:
            # Pad with NaN to make all columns the same length
            max_len = max(len(datasets[strain]['delta_t12']), len(datasets[strain]['delta_t32']))
            
            # Pad delta_t12
            delta_t12 = datasets[strain]['delta_t12']
            if len(delta_t12) < max_len:
                delta_t12 = np.concatenate([delta_t12, [np.nan] * (max_len - len(delta_t12))])
            
            # Pad delta_t32
            delta_t32 = datasets[strain]['delta_t32']
            if len(delta_t32) < max_len:
                delta_t32 = np.concatenate([delta_t32, [np.nan] * (max_len - len(delta_t32))])
            
            data_dict[f'{strain}12'] = delta_t12
            data_dict[f'{strain}32'] = delta_t32
    
    # Add other columns that might be expected
    data_dict['initialProteins12'] = [np.nan] * max_len
    data_dict['initialProteins32'] = [np.nan] * max_len
    
    df = pd.DataFrame(data_dict)
    df.to_excel(filename, index=False)
    print(f"Synthetic data saved to: {filename}")


def run_join_optimization(mechanism, datasets, true_params):
    """
    Run joint optimization using MoMOptimization_join.py.
    
    Args:
        mechanism (str): Mechanism name
        datasets (dict): Synthetic datasets
        true_params (dict): True parameters for comparison
    
    Returns:
        dict: Optimization results
    """
    print(f"\n{'='*60}")
    print("RUNNING JOINT OPTIMIZATION")
    print(f"{'='*60}")
    
    # Import the joint optimization module
    sys.path.append('SecondVersion')
    from MoMOptimization_join import (
        get_mechanism_info, unpack_parameters, joint_objective,
        differential_evolution, get_rounded_parameters
    )
    
    # Get mechanism info
    mechanism_info = get_mechanism_info(mechanism, 'separate')
    bounds = mechanism_info['bounds']
    
    print(f"Optimizing {len(bounds)} parameters...")
    
    # Prepare data in the format expected by joint optimization
    data_wt12 = datasets['wildtype']['delta_t12']
    data_wt32 = datasets['wildtype']['delta_t32']
    data_threshold12 = datasets['threshold']['delta_t12']
    data_threshold32 = datasets['threshold']['delta_t32']
    data_degrate12 = datasets['degrate']['delta_t12']
    data_degrate32 = datasets['degrate']['delta_t32']
    data_degrateAPC12 = datasets['degrateAPC']['delta_t12']
    data_degrateAPC32 = datasets['degrateAPC']['delta_t32']
    
    # Create dummy data for initial proteins (not used in current version)
    data_initial12 = np.array([np.nan])
    data_initial32 = np.array([np.nan])
    
    # Run optimization
    result = differential_evolution(
        joint_objective,
        bounds=bounds,
        args=(mechanism, mechanism_info, data_wt12, data_wt32,
              data_threshold12, data_threshold32, data_degrate12, data_degrate32,
              data_initial12, data_initial32, data_degrateAPC12, data_degrateAPC32),
        strategy='best1bin',
        maxiter=100,  # Reduced for sanity check
        popsize=15,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        disp=True,
        seed=42
    )
    
    if result.success:
        # Unpack results
        recovered_params = unpack_parameters(result.x, mechanism_info)
        
        print(f"✓ Joint optimization completed successfully!")
        print(f"  Negative Log-Likelihood: {result.fun:.4f}")
        
        return {
            'success': True,
            'nll': result.fun,
            'params': recovered_params,
            'result': result
        }
    else:
        print(f"✗ Joint optimization failed: {result.message}")
        return {
            'success': False,
            'nll': np.inf,
            'params': None,
            'result': result
        }


def run_independent_optimization(mechanism, datasets, true_params):
    """
    Run independent optimization using MoMOptimization_independent.py.
    
    Args:
        mechanism (str): Mechanism name
        datasets (dict): Synthetic datasets
        true_params (dict): True parameters for comparison
    
    Returns:
        dict: Optimization results
    """
    print(f"\n{'='*60}")
    print("RUNNING INDEPENDENT OPTIMIZATION")
    print(f"{'='*60}")
    
    # Import the independent optimization module
    sys.path.append('SecondVersion')
    from MoMOptimization_independent import (
        get_mechanism_info, unpack_wildtype_parameters, wildtype_objective,
        threshold_objective, degrate_objective, degrateAPC_objective,
        differential_evolution, basinhopping, BoundedStep
    )
    
    # Get mechanism info
    mechanism_info = get_mechanism_info(mechanism)
    wt_bounds = mechanism_info['bounds']
    
    print(f"Optimizing {len(wt_bounds)} wildtype parameters...")
    
    # Prepare data
    data_wt12 = datasets['wildtype']['delta_t12']
    data_wt32 = datasets['wildtype']['delta_t32']
    data_threshold12 = datasets['threshold']['delta_t12']
    data_threshold32 = datasets['threshold']['delta_t32']
    data_degrate12 = datasets['degrate']['delta_t12']
    data_degrate32 = datasets['degrate']['delta_t32']
    data_degrateAPC12 = datasets['degrateAPC']['delta_t12']
    data_degrateAPC32 = datasets['degrateAPC']['delta_t32']
    
    # Step 1: Optimize wildtype parameters
    result_wt = differential_evolution(
        wildtype_objective,
        bounds=wt_bounds,
        args=(mechanism, mechanism_info, data_wt12, data_wt32),
        strategy='best1bin',
        maxiter=100,  # Reduced for sanity check
        popsize=15,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        disp=True,
        seed=42
    )
    
    if not result_wt.success:
        print(f"✗ Wildtype optimization failed: {result_wt.message}")
        return {
            'success': False,
            'nll': np.inf,
            'params': None,
            'result': result_wt
        }
    
    # Unpack wildtype parameters
    wt_params = unpack_wildtype_parameters(result_wt.x, mechanism_info)
    
    # Extract mechanism-specific parameters
    mech_params = {}
    if mechanism == 'fixed_burst':
        mech_params['burst_size'] = wt_params['burst_size']
    elif mechanism == 'time_varying_k':
        mech_params['k_1'] = wt_params['k_1']
    elif mechanism == 'feedback':
        mech_params['feedbackSteepness'] = wt_params['feedbackSteepness']
        mech_params['feedbackThreshold'] = wt_params['feedbackThreshold']
    elif mechanism == 'feedback_linear':
        mech_params['w1'] = wt_params['w1']
        mech_params['w2'] = wt_params['w2']
        mech_params['w3'] = wt_params['w3']
    elif mechanism == 'feedback_onion':
        mech_params['n_inner'] = wt_params['n_inner']
    elif mechanism == 'feedback_zipper':
        mech_params['z1'] = wt_params['z1']
        mech_params['z2'] = wt_params['z2']
        mech_params['z3'] = wt_params['z3']
    elif mechanism == 'fixed_burst_feedback_linear':
        mech_params['burst_size'] = wt_params['burst_size']
        mech_params['w1'] = wt_params['w1']
        mech_params['w2'] = wt_params['w2']
        mech_params['w3'] = wt_params['w3']
    elif mechanism == 'fixed_burst_feedback_onion':
        mech_params['burst_size'] = wt_params['burst_size']
        mech_params['n_inner'] = wt_params['n_inner']
    
    params_baseline = (wt_params['n1'], wt_params['n2'], wt_params['n3'],
                      wt_params['N1'], wt_params['N2'], wt_params['N3'],
                      wt_params['k'], mech_params)
    
    # Step 2: Optimize mutant parameters
    mutant_bounds = [(0.1, 0.9)]
    
    # Threshold mutant
    result_threshold = basinhopping(
        threshold_objective,
        x0=np.array([0.5]),
        minimizer_kwargs={
            "method": "L-BFGS-B",
            "args": (mechanism, data_threshold12, data_threshold32, params_baseline),
            "bounds": mutant_bounds
        },
        niter=50,  # Reduced for sanity check
        T=1.0,
        stepsize=0.5,
        take_step=BoundedStep(mutant_bounds),
        disp=False
    )
    
    # Degradation rate mutant
    result_degrate = basinhopping(
        degrate_objective,
        x0=np.array([0.5]),
        minimizer_kwargs={
            "method": "L-BFGS-B",
            "args": (mechanism, data_degrate12, data_degrate32, params_baseline),
            "bounds": mutant_bounds
        },
        niter=50,  # Reduced for sanity check
        T=1.0,
        stepsize=0.5,
        take_step=BoundedStep(mutant_bounds),
        disp=False
    )
    
    # Degradation rate APC mutant
    result_degrateAPC = basinhopping(
        degrateAPC_objective,
        x0=np.array([0.5]),
        minimizer_kwargs={
            "method": "L-BFGS-B",
            "args": (mechanism, data_degrateAPC12, data_degrateAPC32, params_baseline),
            "bounds": mutant_bounds
        },
        niter=50,  # Reduced for sanity check
        T=1.0,
        stepsize=0.5,
        take_step=BoundedStep(mutant_bounds),
        disp=False
    )
    
    # Extract results
    alpha = result_threshold.lowest_optimization_result.x[0] if result_threshold.lowest_optimization_result.success else np.nan
    beta_k = result_degrate.lowest_optimization_result.x[0] if result_degrate.lowest_optimization_result.success else np.nan
    beta2_k = result_degrateAPC.lowest_optimization_result.x[0] if result_degrateAPC.lowest_optimization_result.success else np.nan
    
    # Combine parameters
    recovered_params = wt_params.copy()
    recovered_params.update({
        'alpha': alpha,
        'beta_k': beta_k,
        'beta2_k': beta2_k
    })
    
    print(f"✓ Independent optimization completed successfully!")
    print(f"  Wildtype NLL: {result_wt.fun:.4f}")
    print(f"  Alpha: {alpha:.3f}, Beta_k: {beta_k:.3f}, Beta2_k: {beta2_k:.3f}")
    
    return {
        'success': True,
        'nll': result_wt.fun,
        'params': recovered_params,
        'wt_result': result_wt,
        'threshold_result': result_threshold,
        'degrate_result': result_degrate,
        'degrateAPC_result': result_degrateAPC
    }


def compare_parameters(true_params, recovered_params, method_name):
    """
    Compare true parameters with recovered parameters.
    
    Args:
        true_params (dict): True parameters
        recovered_params (dict): Recovered parameters
        method_name (str): Name of the optimization method
    
    Returns:
        dict: Comparison results
    """
    print(f"\n{'='*60}")
    print(f"PARAMETER COMPARISON: {method_name.upper()}")
    print(f"{'='*60}")
    
    if recovered_params is None:
        print("✗ No parameters to compare - optimization failed")
        return None
    
    # Define parameter groups
    wildtype_params = ['n1', 'n2', 'n3', 'N1', 'N2', 'N3', 'k']
    mechanism_params = ['burst_size', 'k_1', 'feedbackSteepness', 'feedbackThreshold', 
                       'w1', 'w2', 'w3', 'n_inner', 'z1', 'z2', 'z3']
    mutant_params = ['alpha', 'beta_k', 'beta2_k']
    
    comparison = {
        'wildtype_errors': {},
        'mechanism_errors': {},
        'mutant_errors': {},
        'overall_rmse': 0.0,
        'success': True
    }
    
    total_squared_error = 0
    total_params = 0
    
    # Compare wildtype parameters
    print("Wildtype Parameters:")
    print(f"{'Parameter':<15} {'True':<10} {'Recovered':<12} {'Error':<10} {'% Error':<10}")
    print("-" * 60)
    
    for param in wildtype_params:
        if param in true_params and param in recovered_params:
            true_val = true_params[param]
            recovered_val = recovered_params[param]
            error = abs(recovered_val - true_val)
            percent_error = (error / true_val) * 100 if true_val != 0 else 0
            
            comparison['wildtype_errors'][param] = {
                'true': true_val,
                'recovered': recovered_val,
                'error': error,
                'percent_error': percent_error
            }
            
            total_squared_error += error ** 2
            total_params += 1
            
            print(f"{param:<15} {true_val:<10.3f} {recovered_val:<12.3f} {error:<10.3f} {percent_error:<10.1f}%")
    
    # Compare mechanism-specific parameters
    print(f"\nMechanism Parameters:")
    print(f"{'Parameter':<15} {'True':<10} {'Recovered':<12} {'Error':<10} {'% Error':<10}")
    print("-" * 60)
    
    for param in mechanism_params:
        if param in true_params and param in recovered_params:
            true_val = true_params[param]
            recovered_val = recovered_params[param]
            error = abs(recovered_val - true_val)
            percent_error = (error / true_val) * 100 if true_val != 0 else 0
            
            comparison['mechanism_errors'][param] = {
                'true': true_val,
                'recovered': recovered_val,
                'error': error,
                'percent_error': percent_error
            }
            
            total_squared_error += error ** 2
            total_params += 1
            
            print(f"{param:<15} {true_val:<10.3f} {recovered_val:<12.3f} {error:<10.3f} {percent_error:<10.1f}%")
    
    # Compare mutant parameters
    print(f"\nMutant Parameters:")
    print(f"{'Parameter':<15} {'True':<10} {'Recovered':<12} {'Error':<10} {'% Error':<10}")
    print("-" * 60)
    
    for param in mutant_params:
        if param in true_params and param in recovered_params:
            true_val = true_params[param]
            recovered_val = recovered_params[param]
            error = abs(recovered_val - true_val)
            percent_error = (error / true_val) * 100 if true_val != 0 else 0
            
            comparison['mutant_errors'][param] = {
                'true': true_val,
                'recovered': recovered_val,
                'error': error,
                'percent_error': percent_error
            }
            
            total_squared_error += error ** 2
            total_params += 1
            
            print(f"{param:<15} {true_val:<10.3f} {recovered_val:<12.3f} {error:<10.3f} {percent_error:<10.1f}%")
    
    # Calculate overall RMSE
    if total_params > 0:
        comparison['overall_rmse'] = np.sqrt(total_squared_error / total_params)
        print(f"\nOverall RMSE: {comparison['overall_rmse']:.4f}")
        
        # Determine if recovery was successful (RMSE < 20% of parameter values)
        avg_param_value = np.mean([abs(true_params.get(p, 0)) for p in wildtype_params + mechanism_params + mutant_params if p in true_params])
        if comparison['overall_rmse'] < 0.2 * avg_param_value:
            print("✅ Parameter recovery: SUCCESSFUL")
        else:
            print("❌ Parameter recovery: POOR")
            comparison['success'] = False
    else:
        print("❌ No parameters to compare")
        comparison['success'] = False
    
    return comparison


def _build_mech_params(param_dict, mechanism):
    """Helper: build mechanism-specific params dict from a parameter dictionary."""
    mech_params = {}
    if mechanism == 'fixed_burst':
        if 'burst_size' in param_dict:
            mech_params['burst_size'] = param_dict['burst_size']
    elif mechanism == 'time_varying_k':
        if 'k_1' in param_dict:
            mech_params['k_1'] = param_dict['k_1']
    elif mechanism == 'feedback':
        for key in ['feedbackSteepness', 'feedbackThreshold']:
            if key in param_dict:
                mech_params[key] = param_dict[key]
    elif mechanism == 'feedback_linear':
        for key in ['w1', 'w2', 'w3']:
            if key in param_dict:
                mech_params[key] = param_dict[key]
    elif mechanism == 'feedback_onion':
        if 'n_inner' in param_dict:
            mech_params['n_inner'] = param_dict['n_inner']
    elif mechanism == 'feedback_zipper':
        for key in ['z1', 'z2', 'z3']:
            if key in param_dict:
                mech_params[key] = param_dict[key]
    elif mechanism == 'fixed_burst_feedback_linear':
        for key in ['burst_size', 'w1', 'w2', 'w3']:
            if key in param_dict:
                mech_params[key] = param_dict[key]
    elif mechanism == 'fixed_burst_feedback_onion':
        for key in ['burst_size', 'n_inner']:
            if key in param_dict:
                mech_params[key] = param_dict[key]
    return mech_params


def _get_strain_adjusted_params(base_params, strain):
    """Return adjusted (n1,n2,n3,N1,N2,N3,k) for a given strain based on mutant multipliers."""
    n1 = base_params.get('n1'); n2 = base_params.get('n2'); n3 = base_params.get('n3')
    N1 = base_params.get('N1'); N2 = base_params.get('N2'); N3 = base_params.get('N3')
    k = base_params.get('k')
    alpha = base_params.get('alpha', 1.0)
    beta_k = base_params.get('beta_k', 1.0)
    beta2_k = base_params.get('beta2_k', 1.0)
    if any(v is None for v in [n1, n2, n3, N1, N2, N3, k]):
        return None
    if strain == 'threshold':
        n1, n2, n3 = max(n1 * alpha, 1), max(n2 * alpha, 1), max(n3 * alpha, 1)
    if strain == 'degrate':
        k = max(k * beta_k, 0.001)
    if strain == 'degrateAPC':
        k = max(k * beta2_k, 0.001)
    return n1, n2, n3, N1, N2, N3, k


def plot_histograms_with_pdfs(datasets, mechanism, true_params, joint_params, independent_params, out_dir):
    """
    Plot histograms of synthetic data with overlaid PDFs from recovered parameters.
    Draw both joint and independent PDFs when available.
    """
    os.makedirs(out_dir, exist_ok=True)

    strains = ['wildtype', 'threshold', 'degrate', 'degrateAPC']
    for strain in strains:
        for pair_key, pair12 in [('delta_t12', True), ('delta_t32', False)]:
            data = datasets[strain][pair_key]
            if data.size == 0:
                continue
            xmin = float(np.nanmin(data))
            xmax = float(np.nanmax(data))
            if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
                continue
            x_grid = np.linspace(xmin, xmax, 400)

            plt.figure(figsize=(7, 4))
            plt.hist(data, bins=50, density=True, alpha=0.35, color='gray', edgecolor='none', label='Empirical')

            # Plot theoretical PDFs using joint and independent recovered params
            for method_name, params, color in [
                ('Joint', joint_params, 'C0'),
                ('Independent', independent_params, 'C1')
            ]:
                if params is None:
                    continue
                adj = _get_strain_adjusted_params(params, strain)
                if adj is None:
                    continue
                n1, n2, n3, N1, N2, N3, k = adj
                mech_params = _build_mech_params(params, mechanism)
                try:
                    if pair12:
                        pdf_vals = compute_pdf_for_mechanism(mechanism, x_grid, n1, N1, n2, N2, k, mech_params, pair12=True)
                    else:
                        pdf_vals = compute_pdf_for_mechanism(mechanism, x_grid, n3, N3, n2, N2, k, mech_params, pair12=False)
                    plt.plot(x_grid, pdf_vals, color=color, lw=2, label=f'{method_name} PDF')
                except Exception as e:
                    print(f"Plot warning: could not compute {method_name} PDF for {strain} {pair_key}: {e}")

            title_pair = 'T1-T2' if pair12 else 'T3-T2'
            plt.title(f"{strain} – {title_pair}")
            plt.xlabel('Delta time')
            plt.ylabel('Density')
            plt.legend()
            fname = os.path.join(out_dir, f"hist_pdf_{strain}_{'12' if pair12 else '32'}.png")
            plt.tight_layout()
            plt.savefig(fname, dpi=150)
            plt.close()


def plot_parameter_bars(true_params, recovered_params, method_name, out_dir):
    """Bar plot comparing true vs recovered parameters."""
    os.makedirs(out_dir, exist_ok=True)
    keys_order = ['n1','n2','n3','N1','N2','N3','k','burst_size','k_1','feedbackSteepness','feedbackThreshold','w1','w2','w3','n_inner','z1','z2','z3','alpha','beta_k','beta2_k']
    keys = [k for k in keys_order if k in true_params and recovered_params and k in recovered_params]
    if not keys:
        return
    true_vals = [true_params[k] for k in keys]
    rec_vals = [recovered_params[k] for k in keys]
    x = np.arange(len(keys))
    width = 0.38
    plt.figure(figsize=(max(8, len(keys)*0.6), 4.5))
    plt.bar(x - width/2, true_vals, width, label='True', color='C2', alpha=0.7)
    plt.bar(x + width/2, rec_vals, width, label='Recovered', color='C0', alpha=0.7)
    plt.xticks(x, keys, rotation=45, ha='right')
    plt.ylabel('Value')
    plt.title(f'Parameters: True vs {method_name}')
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(out_dir, f"params_{method_name.lower().replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150)
    plt.close()


def get_test_parameters(mechanism):
    """
    Get test parameters for a given mechanism.
    
    Args:
        mechanism (str): Mechanism name
    
    Returns:
        dict: Test parameters
    """
    # Base parameters (same for all mechanisms)
    base_params = {
        'n1': 25.0,
        'n2': 20.0,
        'n3': 30.0,
        'N1': 200.0,
        'N2': 150.0,
        'N3': 250.0,
        'k': 0.1,
        'alpha': 0.7,
        'beta_k': 0.8,
        'beta2_k': 0.6
    }
    
    # Mechanism-specific parameters
    if mechanism == 'simple':
        return base_params
    elif mechanism == 'fixed_burst':
        base_params['burst_size'] = 5.0
        return base_params
    elif mechanism == 'time_varying_k':
        base_params['k_1'] = 0.005
        return base_params
    elif mechanism == 'feedback':
        base_params['feedbackSteepness'] = 0.05
        base_params['feedbackThreshold'] = 100.0
        return base_params
    elif mechanism == 'feedback_linear':
        base_params['w1'] = 0.01
        base_params['w2'] = 0.008
        base_params['w3'] = 0.012
        return base_params
    elif mechanism == 'feedback_onion':
        base_params['n_inner'] = 25.0
        return base_params
    elif mechanism == 'feedback_zipper':
        base_params['z1'] = 50.0
        base_params['z2'] = 40.0
        base_params['z3'] = 60.0
        return base_params
    elif mechanism == 'fixed_burst_feedback_linear':
        base_params['burst_size'] = 5.0
        base_params['w1'] = 0.01
        base_params['w2'] = 0.008
        base_params['w3'] = 0.012
        return base_params
    elif mechanism == 'fixed_burst_feedback_onion':
        base_params['burst_size'] = 5.0
        base_params['n_inner'] = 25.0
        return base_params
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")


def main():
    """
    Main sanity check routine.
    """
    print("SANITY CHECK FOR FITTING PIPELINE")
    print("=" * 60)
    
    # Configuration
    mechanism = 'fixed_burst_feedback_onion'  # Test mechanism
    num_simulations = 500  # Number of simulations for data generation
    max_time = 500  # Maximum simulation time
    
    print(f"Testing mechanism: {mechanism}")
    print(f"Number of simulations: {num_simulations}")
    print(f"Maximum simulation time: {max_time}")
    
    # Get test parameters
    true_params = get_test_parameters(mechanism)
    
    # Step 1: Generate synthetic data
    print(f"\n{'='*60}")
    print("STEP 1: GENERATING SYNTHETIC DATA")
    print(f"{'='*60}")
    
    datasets = generate_synthetic_data(mechanism, true_params, num_simulations, max_time)
    
    # Save synthetic data
    save_synthetic_data(datasets, "synthetic_data_sanity_check.xlsx")
    
    # Step 2: Run joint optimization
    print(f"\n{'='*60}")
    print("STEP 2: TESTING JOINT OPTIMIZATION")
    print(f"{'='*60}")
    
    joint_results = run_join_optimization(mechanism, datasets, true_params)
    
    # Step 3: Run independent optimization
    print(f"\n{'='*60}")
    print("STEP 3: TESTING INDEPENDENT OPTIMIZATION")
    print(f"{'='*60}")
    
    independent_results = run_independent_optimization(mechanism, datasets, true_params)
    
    # Step 4: Compare results
    print(f"\n{'='*60}")
    print("STEP 4: COMPARING RESULTS")
    print(f"{'='*60}")
    
    joint_comparison = None
    independent_comparison = None
    
    if joint_results['success']:
        joint_comparison = compare_parameters(true_params, joint_results['params'], "Joint Optimization")
    
    if independent_results['success']:
        independent_comparison = compare_parameters(true_params, independent_results['params'], "Independent Optimization")
    
    # Step 5: Plots
    print(f"\n{'='*60}")
    print("STEP 5: PLOTTING RESULTS")
    print(f"{'='*60}")
    out_dir = os.path.join('Results', 'SanityCheck')
    joint_params = joint_results['params'] if joint_results and joint_results.get('success') else None
    indep_params = independent_results['params'] if independent_results and independent_results.get('success') else None
    try:
        plot_histograms_with_pdfs(datasets, mechanism, true_params, joint_params, indep_params, out_dir)
    except Exception as e:
        print(f"Plotting error (hist+pdf): {e}")
    try:
        if joint_params:
            plot_parameter_bars(true_params, joint_params, 'Joint Optimization', out_dir)
        if indep_params:
            plot_parameter_bars(true_params, indep_params, 'Independent Optimization', out_dir)
    except Exception as e:
        print(f"Plotting error (params bars): {e}")

    # Step 6: Summary
    print(f"\n{'='*60}")
    print("SANITY CHECK SUMMARY")
    print(f"{'='*60}")
    
    print(f"Mechanism tested: {mechanism}")
    print(f"True parameters: {true_params}")
    
    if joint_comparison:
        print(f"Joint optimization: {'✅ SUCCESS' if joint_comparison['success'] else '❌ POOR RECOVERY'}")
        print(f"  RMSE: {joint_comparison['overall_rmse']:.4f}")
    else:
        print("Joint optimization: ❌ FAILED")
    
    if independent_comparison:
        print(f"Independent optimization: {'✅ SUCCESS' if independent_comparison['success'] else '❌ POOR RECOVERY'}")
        print(f"  RMSE: {independent_comparison['overall_rmse']:.4f}")
    else:
        print("Independent optimization: ❌ FAILED")
    
    # Determine overall success
    joint_success = joint_comparison and joint_comparison['success']
    independent_success = independent_comparison and independent_comparison['success']
    
    if joint_success and independent_success:
        print(f"\n🎉 SANITY CHECK PASSED: Both optimization methods successfully recovered parameters!")
    elif joint_success or independent_success:
        print(f"\n⚠️  SANITY CHECK PARTIAL: One optimization method succeeded, one failed.")
    else:
        print(f"\n❌ SANITY CHECK FAILED: Both optimization methods failed to recover parameters.")
    
    print(f"\nSynthetic data saved to: synthetic_data_sanity_check.xlsx")
    print("Sanity check complete!")


if __name__ == "__main__":
    main()
