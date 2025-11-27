#!/usr/bin/env python3
"""
Test script to compare MoM (Method of Moments) NLL vs Simulation KDE NLL.

This script:
1. Loads parameter sets from CSV file (includes expected NLL values)
2. Calculates MoM NLL using functions from MoMOptimization_join.py
3. Calculates Simulation KDE NLL using functions from simulation_utils.py
4. Compares both against the CSV NLL value for consistency

Supports non-time-varying mechanisms:
- simple
- fixed_burst
- feedback_onion
- fixed_burst_feedback_onion
"""

import numpy as np
import pandas as pd
import sys
import os
import warnings
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SecondVersion'))
from simulation_utils import (
    load_experimental_data, 
    apply_mutant_params,
    calculate_likelihood,
    run_simulation_for_dataset
)
from MoMOptimization_join import (
    unpack_parameters,
    get_mechanism_info,
    _get_mech_params,
    _add_strain_nll,
    joint_objective
)
from MoMCalculations import compute_pdf_for_mechanism

# Suppress warnings
warnings.filterwarnings('ignore')


def load_params_from_csv(csv_file, mechanism, run_number=None):
    """
    Load parameters from CSV file for a specific mechanism.
    
    Args:
        csv_file (str): Path to CSV file
        mechanism (str): Mechanism name ('simple', 'fixed_burst', etc.)
        run_number (int, optional): Specific run number. If None, picks first available.
    
    Returns:
        tuple: (params_dict, run_number, expected_nll)
    """
    df = pd.read_csv(csv_file)
    
    # Filter by mechanism
    mechanism_df = df[df['mechanism'] == mechanism].copy()
    
    if len(mechanism_df) == 0:
        raise ValueError(f"No data found for mechanism '{mechanism}' in CSV file")
    
    # Select run (first if not specified)
    if run_number is None:
        selected_row = mechanism_df.iloc[0]
    else:
        selected_row = mechanism_df[mechanism_df['run_number'] == run_number]
        if len(selected_row) == 0:
            raise ValueError(f"Run number {run_number} not found for mechanism '{mechanism}'")
        selected_row = selected_row.iloc[0]
    
    # Extract expected NLL from CSV
    expected_nll = selected_row['nll'] if 'nll' in selected_row and pd.notna(selected_row['nll']) else None
    
    # Extract parameters in the order expected by MoMOptimization_join
    # Order: n2, N2, k, r21, r23, R21, R23, [mechanism_params], alpha, beta_k, beta2_k, beta3_k
    params_list = [
        selected_row['n2'],
        selected_row['N2'],
        selected_row['k'],
        selected_row['r21'],
        selected_row['r23'],
        selected_row['R21'],
        selected_row['R23'],
    ]
    
    # Add mechanism-specific parameters
    if mechanism == 'fixed_burst':
        params_list.append(selected_row['burst_size'])
    elif mechanism == 'feedback_onion':
        params_list.append(selected_row['n_inner'])
    elif mechanism == 'fixed_burst_feedback_onion':
        params_list.append(selected_row['burst_size'])
        params_list.append(selected_row['n_inner'])
    
    # Add mutant parameters
    params_list.extend([
        selected_row['alpha'],
        selected_row['beta_k'],
    ])
    
    # Add beta2_k and beta3_k (for degradeAPC and velcade)
    if 'beta2_k' in selected_row and pd.notna(selected_row['beta2_k']):
        params_list.append(selected_row['beta2_k'])
    else:
        params_list.append(selected_row.get('beta_k', 1.0))  # Default to beta_k if missing
    
    if 'beta3_k' in selected_row and pd.notna(selected_row['beta3_k']):
        params_list.append(selected_row['beta3_k'])
    else:
        params_list.append(selected_row.get('beta_k', 1.0))  # Default to beta_k if missing
    
    return params_list, selected_row['run_number'], expected_nll


def calculate_mom_nll(mechanism, params_vector, datasets):
    """
    Calculate NLL using Method of Moments (using functions from MoMOptimization_join.py).
    
    Args:
        mechanism (str): Mechanism name
        params_vector (list): Parameter vector in MoM optimization format
        datasets (dict): Experimental datasets (from simulation_utils.load_experimental_data)
    
    Returns:
        tuple: (total_nll, individual_nlls_dict)
            - total_nll: Total NLL across all datasets
            - individual_nlls_dict: Dictionary with NLL for each dataset
    """
    # Get mechanism info (needed for unpack_parameters)
    mechanism_info = get_mechanism_info(mechanism, gamma_mode='unified')
    
    # Unpack parameters
    param_dict = unpack_parameters(params_vector, mechanism_info)
    
    # Get mechanism-specific parameters
    mech_params = _get_mech_params(mechanism, param_dict)
    
    # Convert datasets format to MoM format
    # MoM expects: data_wt12, data_wt32, data_threshold12, data_threshold32, etc.
    data_wt12 = datasets['wildtype']['delta_t12']
    data_wt32 = datasets['wildtype']['delta_t32']
    data_threshold12 = datasets['threshold']['delta_t12']
    data_threshold32 = datasets['threshold']['delta_t32']
    data_degrate12 = datasets['degrade']['delta_t12']
    data_degrate32 = datasets['degrade']['delta_t32']
    data_degrateAPC12 = datasets['degradeAPC']['delta_t12']
    data_degrateAPC32 = datasets['degradeAPC']['delta_t32']
    data_velcade12 = datasets['velcade']['delta_t12']
    data_velcade32 = datasets['velcade']['delta_t32']
    
    # Use empty arrays for initial proteins (not used in joint_objective)
    data_initial12 = np.array([])
    data_initial32 = np.array([])
    
    # Calculate individual NLLs using _add_strain_nll
    individual_nlls = {}
    total_nll = 0.0
    
    # Wild-Type
    total_nll, nll_wt = _add_strain_nll(
        total_nll, mechanism, data_wt12, data_wt32,
        param_dict['n1'], param_dict['N1'], param_dict['n2'], param_dict['N2'],
        param_dict['n3'], param_dict['N3'], param_dict['k'], mech_params
    )
    individual_nlls['wildtype'] = nll_wt
    
    # Threshold Mutant
    alpha = param_dict['alpha']
    n1_th = max(param_dict['n1'] * alpha, 1)
    n2_th = max(param_dict['n2'] * alpha, 1)
    n3_th = max(param_dict['n3'] * alpha, 1)
    
    # Diagnostic output for threshold mutant
    print(f"\n[DEBUG] Threshold Mutant Parameters:")
    print(f"  Base: n1={param_dict['n1']:.4f}, n2={param_dict['n2']:.4f}, n3={param_dict['n3']:.4f}")
    print(f"  Alpha: {alpha:.4f}")
    print(f"  MoM uses: n1={n1_th:.4f}, n2={n2_th:.4f}, n3={n3_th:.4f}")
    print(f"  (Note: MoM applies max(alpha * n, 1))")
    
    total_nll, nll_threshold = _add_strain_nll(
        total_nll, mechanism, data_threshold12, data_threshold32,
        n1_th, param_dict['N1'], n2_th, param_dict['N2'],
        n3_th, param_dict['N3'], param_dict['k'], mech_params
    )
    individual_nlls['threshold'] = nll_threshold
    
    # Degradation Rate Mutants
    # degrade (beta_k)
    k_mutant_degrate = max(param_dict['beta_k'] * param_dict['k'], 0.0005)
    total_nll, nll_degrate = _add_strain_nll(
        total_nll, mechanism, data_degrate12, data_degrate32,
        param_dict['n1'], param_dict['N1'], param_dict['n2'], param_dict['N2'],
        param_dict['n3'], param_dict['N3'], k_mutant_degrate, mech_params
    )
    individual_nlls['degrade'] = nll_degrate
    
    # degradeAPC (beta2_k)
    k_mutant_degrateAPC = max(param_dict['beta2_k'] * param_dict['k'], 0.0005)
    total_nll, nll_degrateAPC = _add_strain_nll(
        total_nll, mechanism, data_degrateAPC12, data_degrateAPC32,
        param_dict['n1'], param_dict['N1'], param_dict['n2'], param_dict['N2'],
        param_dict['n3'], param_dict['N3'], k_mutant_degrateAPC, mech_params
    )
    individual_nlls['degradeAPC'] = nll_degrateAPC
    
    # velcade (beta3_k)
    k_mutant_velcade = max(param_dict['beta3_k'] * param_dict['k'], 0.0005)
    total_nll, nll_velcade = _add_strain_nll(
        total_nll, mechanism, data_velcade12, data_velcade32,
        param_dict['n1'], param_dict['N1'], param_dict['n2'], param_dict['N2'],
        param_dict['n3'], param_dict['N3'], k_mutant_velcade, mech_params
    )
    individual_nlls['velcade'] = nll_velcade
    
    return total_nll, individual_nlls


def calculate_simulation_kde_nll(mechanism, params_vector, datasets, num_simulations=5000):
    """
    Calculate NLL using Simulation + KDE approach (using functions from simulation_utils.py).
    
    Args:
        mechanism (str): Mechanism name
        params_vector (list): Parameter vector in MoM optimization format
        datasets (dict): Experimental datasets (from simulation_utils.load_experimental_data)
        num_simulations (int): Number of simulations to run
    
    Returns:
        tuple: (total_nll, individual_nlls_dict)
            - total_nll: Total NLL across all datasets
            - individual_nlls_dict: Dictionary with NLL for each dataset
    """
    # Get mechanism info to unpack parameters
    mechanism_info = get_mechanism_info(mechanism, gamma_mode='unified')
    param_dict = unpack_parameters(params_vector, mechanism_info)
    
    # Extract base parameters
    base_params = {
        'n1': param_dict['n1'],
        'n2': param_dict['n2'],
        'n3': param_dict['n3'],
        'N1': param_dict['N1'],
        'N2': param_dict['N2'],
        'N3': param_dict['N3'],
        'k': param_dict['k'],
    }
    
    # Add mechanism-specific parameters
    if 'burst_size' in param_dict:
        base_params['burst_size'] = param_dict['burst_size']
    if 'n_inner' in param_dict:
        base_params['n_inner'] = param_dict['n_inner']
    
    # Extract mutant parameters
    alpha = param_dict.get('alpha', 1.0)
    beta_k = param_dict.get('beta_k', 1.0)
    beta2_k = param_dict.get('beta2_k', 1.0)  # For degradeAPC
    beta3_k = param_dict.get('beta3_k', 1.0)  # For velcade
    
    total_nll = 0.0
    individual_nlls = {}
    
    # Loop over datasets (same order as MoM: wildtype, threshold, degrade, degradeAPC, velcade)
    # Note: apply_mutant_params uses beta_k parameter for all degradation mutants
    # For simple mechanisms, degradeAPC and velcade should use beta2_k and beta3_k respectively
    # So we pass beta2_k as beta_k for degradeAPC, and beta3_k as beta_k for velcade
    for dataset_name, data_dict in datasets.items():
        # Apply mutant modifications
        if dataset_name == 'degradeAPC':
            # For degradeAPC, use beta2_k as the beta_k parameter
            mutant_params, n0_list = apply_mutant_params(
                base_params, 'degradeAPC', alpha, beta2_k, None, None
            )
        elif dataset_name == 'velcade':
            # For velcade, use beta3_k as the beta_k parameter
            mutant_params, n0_list = apply_mutant_params(
                base_params, 'velcade', alpha, beta3_k, None, None
            )
        else:
            # For wildtype, threshold, and degrade, use standard parameters
            mutant_params, n0_list = apply_mutant_params(
                base_params, dataset_name, alpha, beta_k, None, None
            )
        
        # FIX: For threshold mutant, ensure n0_list uses max(alpha * n, 1) to match MoM
        # This is critical because MoM uses max(alpha * n, 1) but apply_mutant_params uses alpha * n directly
        if dataset_name == 'threshold':
            # Calculate what MoM would use
            n1_th_mom = max(base_params['n1'] * alpha, 1)
            n2_th_mom = max(base_params['n2'] * alpha, 1)
            n3_th_mom = max(base_params['n3'] * alpha, 1)
            
            # Diagnostic output
            print(f"\n[DEBUG] Threshold Mutant Parameter Comparison:")
            print(f"  Base: n1={base_params['n1']:.4f}, n2={base_params['n2']:.4f}, n3={base_params['n3']:.4f}")
            print(f"  Alpha: {alpha:.4f}")
            print(f"  apply_mutant_params gives: n1={mutant_params['n1']:.4f}, n2={mutant_params['n2']:.4f}, n3={mutant_params['n3']:.4f}")
            print(f"  MoM uses: n1={n1_th_mom:.4f}, n2={n2_th_mom:.4f}, n3={n3_th_mom:.4f}")
            
            # Fix: Update mutant_params and n0_list to match MoM
            mutant_params['n1'] = n1_th_mom
            mutant_params['n2'] = n2_th_mom
            mutant_params['n3'] = n3_th_mom
            n0_list = [n1_th_mom, n2_th_mom, n3_th_mom]
            
            if abs(mutant_params['n1'] - n1_th_mom) > 0.01 or abs(mutant_params['n2'] - n2_th_mom) > 0.01 or abs(mutant_params['n3'] - n3_th_mom) > 0.01:
                print(f"  FIXED: Updated simulation parameters to match MoM")
        
        # Run simulations
        sim_delta_t12, sim_delta_t32 = run_simulation_for_dataset(
            mechanism, mutant_params, n0_list, num_simulations
        )
        
        if sim_delta_t12 is None or sim_delta_t32 is None:
            print(f"Warning: Simulation failed for {dataset_name}")
            return 1e6, {}
        
        # Calculate likelihood using KDE (from simulation_utils.py)
        exp_data = {
            'delta_t12': data_dict['delta_t12'],
            'delta_t32': data_dict['delta_t32']
        }
        sim_data = {
            'delta_t12': sim_delta_t12,
            'delta_t32': sim_delta_t32
        }
        
        nll = calculate_likelihood(exp_data, sim_data)
        
        if nll >= 1e6:
            print(f"Warning: KDE calculation failed for {dataset_name}")
            return 1e6, {}
        
        individual_nlls[dataset_name] = nll
        total_nll += nll
    
    return total_nll, individual_nlls


def plot_mom_vs_kde_distributions(mechanism, params_vector, datasets, num_simulations=5000):
    """
    Plot MoM PDF vs KDE distributions for comparison.
    
    Args:
        mechanism (str): Mechanism name
        params_vector (list): Parameter vector in MoM optimization format
        datasets (dict): Experimental datasets
        num_simulations (int): Number of simulations for KDE
    """
    # Get mechanism info to unpack parameters
    mechanism_info = get_mechanism_info(mechanism, gamma_mode='unified')
    param_dict = unpack_parameters(params_vector, mechanism_info)
    mech_params = _get_mech_params(mechanism, param_dict)
    
    # Extract base parameters
    base_params = {
        'n1': param_dict['n1'],
        'n2': param_dict['n2'],
        'n3': param_dict['n3'],
        'N1': param_dict['N1'],
        'N2': param_dict['N2'],
        'N3': param_dict['N3'],
        'k': param_dict['k'],
    }
    
    # Add mechanism-specific parameters
    if 'burst_size' in param_dict:
        base_params['burst_size'] = param_dict['burst_size']
    if 'n_inner' in param_dict:
        base_params['n_inner'] = param_dict['n_inner']
    
    # Extract mutant parameters
    alpha = param_dict.get('alpha', 1.0)
    beta_k = param_dict.get('beta_k', 1.0)
    beta2_k = param_dict.get('beta2_k', 1.0)
    beta3_k = param_dict.get('beta3_k', 1.0)
    
    # Create figure with subplots for each dataset
    num_datasets = len(datasets)
    fig, axes = plt.subplots(num_datasets, 2, figsize=(14, 4 * num_datasets))
    if num_datasets == 1:
        axes = axes.reshape(1, -1)
    
    dataset_names = list(datasets.keys())
    
    for idx, dataset_name in enumerate(dataset_names):
        data_dict = datasets[dataset_name]
        
        # Apply mutant modifications
        if dataset_name == 'degradeAPC':
            mutant_params, n0_list = apply_mutant_params(
                base_params, 'degradeAPC', alpha, beta2_k, None, None
            )
        elif dataset_name == 'velcade':
            mutant_params, n0_list = apply_mutant_params(
                base_params, 'velcade', alpha, beta3_k, None, None
            )
        else:
            mutant_params, n0_list = apply_mutant_params(
                base_params, dataset_name, alpha, beta_k, None, None
            )
        
        # Diagnostic output for threshold mutant in plotting
        if dataset_name == 'threshold':
            print(f"\n[DEBUG PLOT] Threshold Mutant Parameters for Plotting:")
            print(f"  Simulation params: n1={mutant_params['n1']:.4f}, n2={mutant_params['n2']:.4f}, n3={mutant_params['n3']:.4f}")
            print(f"  n0_list: {n0_list}")
            # Calculate what MoM would use
            n1_th_mom = max(base_params['n1'] * alpha, 1)
            n2_th_mom = max(base_params['n2'] * alpha, 1)
            n3_th_mom = max(base_params['n3'] * alpha, 1)
            print(f"  MoM would use: n1={n1_th_mom:.4f}, n2={n2_th_mom:.4f}, n3={n3_th_mom:.4f}")
            if abs(mutant_params['n1'] - n1_th_mom) > 0.01 or abs(mutant_params['n2'] - n2_th_mom) > 0.01 or abs(mutant_params['n3'] - n3_th_mom) > 0.01:
                print(f"  MISMATCH DETECTED: Simulation and MoM use different threshold values!")
        
        # Get experimental data
        exp_delta_t12 = data_dict['delta_t12']
        exp_delta_t32 = data_dict['delta_t32']
        
        # Run simulations for KDE
        sim_delta_t12, sim_delta_t32 = run_simulation_for_dataset(
            mechanism, mutant_params, n0_list, num_simulations
        )
        
        if sim_delta_t12 is None or sim_delta_t32 is None:
            print(f"Warning: Simulation failed for {dataset_name}, skipping plot")
            continue
        
        # Create evaluation points for smooth plots
        t12_min = min(exp_delta_t12.min(), sim_delta_t12.min())
        t12_max = max(exp_delta_t12.max(), sim_delta_t12.max())
        t32_min = min(exp_delta_t32.min(), sim_delta_t32.min())
        t32_max = max(exp_delta_t32.max(), sim_delta_t32.max())
        
        # Add some padding
        t12_range = np.linspace(t12_min - 0.1 * (t12_max - t12_min), 
                                t12_max + 0.1 * (t12_max - t12_min), 200)
        t32_range = np.linspace(t32_min - 0.1 * (t32_max - t32_min), 
                                t32_max + 0.1 * (t32_max - t32_min), 200)
        
        # For threshold mutant, MoM uses max(alpha * n, 1), but simulation uses alpha * n directly
        # Use the same parameters that MoM calculation uses for consistency
        if dataset_name == 'threshold':
            # Calculate threshold values the same way MoM does
            n1_th_mom = max(base_params['n1'] * alpha, 1)
            n2_th_mom = max(base_params['n2'] * alpha, 1)
            n3_th_mom = max(base_params['n3'] * alpha, 1)
            # Use these for MoM PDF calculation
            mom_n1 = n1_th_mom
            mom_n2 = n2_th_mom
            mom_n3 = n3_th_mom
        else:
            mom_n1 = mutant_params['n1']
            mom_n2 = mutant_params['n2']
            mom_n3 = mutant_params['n3']
        
        # Calculate MoM PDF at smooth evaluation points for delta_t12
        mom_pdf_12 = compute_pdf_for_mechanism(
            mechanism,
            t12_range,
            mom_n1, mutant_params['N1'],
            mom_n2, mutant_params['N2'],
            mutant_params['k'],
            mech_params,
            pair12=True
        )
        
        # Calculate MoM PDF at smooth evaluation points for delta_t32
        mom_pdf_32 = compute_pdf_for_mechanism(
            mechanism,
            t32_range,
            mom_n3, mutant_params['N3'],
            mom_n2, mutant_params['N2'],
            mutant_params['k'],
            mech_params,
            pair12=False
        )
        
        # Create KDE from simulation data
        kde_12 = gaussian_kde(sim_delta_t12)
        kde_32 = gaussian_kde(sim_delta_t32)
        
        # Evaluate KDE at these points
        kde_pdf_12 = kde_12(t12_range)
        kde_pdf_32 = kde_32(t32_range)
        
        # Plot delta_t12
        ax1 = axes[idx, 0]
        ax1.hist(exp_delta_t12, bins=30, density=True, alpha=0.4, label='Experimental Data', color='gray', edgecolor='black')
        ax1.hist(sim_delta_t12, bins=30, density=True, alpha=0.3, label='Simulation Data', color='orange', edgecolor='darkorange', histtype='step', linewidth=1.5)
        ax1.plot(t12_range, mom_pdf_12, '-', linewidth=2, label='MoM PDF', color='blue', alpha=0.8)
        ax1.plot(t12_range, kde_pdf_12, '--', linewidth=2, label='Simulation KDE', color='red', alpha=0.8)
        ax1.set_xlabel('ΔT12 (T1 - T2)')
        ax1.set_ylabel('Density')
        ax1.set_title(f'{dataset_name} - ΔT12')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot delta_t32
        ax2 = axes[idx, 1]
        ax2.hist(exp_delta_t32, bins=30, density=True, alpha=0.4, label='Experimental Data', color='gray', edgecolor='black')
        ax2.hist(sim_delta_t32, bins=30, density=True, alpha=0.3, label='Simulation Data', color='orange', edgecolor='darkorange', histtype='step', linewidth=1.5)
        ax2.plot(t32_range, mom_pdf_32, '-', linewidth=2, label='MoM PDF', color='blue', alpha=0.8)
        ax2.plot(t32_range, kde_pdf_32, '--', linewidth=2, label='Simulation KDE', color='red', alpha=0.8)
        ax2.set_xlabel('ΔT32 (T3 - T2)')
        ax2.set_ylabel('Density')
        ax2.set_title(f'{dataset_name} - ΔT32')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    #plt.savefig('comparison_plot.png')
    #print("Plot saved to comparison_plot.png")
    plt.show()


def compare_mom_vs_simulation(mechanism, csv_file, run_number=None, num_simulations=5000):
    """
    Compare MoM NLL vs Simulation KDE NLL for a given mechanism.
    Compares both against the expected NLL from CSV file.
    
    Args:
        mechanism (str): Mechanism name
        csv_file (str): Path to CSV file with optimized parameters
        run_number (int, optional): Specific run number to test
        num_simulations (int): Number of simulations for KDE approach
    
    Returns:
        dict: Comparison results
    """
    print("="*70)
    print(f"Comparing MoM vs Simulation KDE NLL for mechanism: {mechanism}")
    print("="*70)
    
    # Load parameters from CSV
    print(f"\nLoading parameters from {csv_file}...")
    params_vector, actual_run, expected_nll = load_params_from_csv(csv_file, mechanism, run_number)
    print(f"Using run number: {actual_run}")
    
    # Get mechanism info to display parameters
    mechanism_info = get_mechanism_info(mechanism, gamma_mode='unified')
    param_dict = unpack_parameters(params_vector, mechanism_info)
    
    print(f"\nParameters:")
    print(f"  n2={param_dict['n2']:.2f}, N2={param_dict['N2']:.2f}")
    print(f"  k={param_dict['k']:.6f}")
    print(f"  r21={param_dict['r21']:.3f}, r23={param_dict['r23']:.3f}")
    print(f"  R21={param_dict['R21']:.3f}, R23={param_dict['R23']:.3f}")
    print(f"  alpha={param_dict['alpha']:.3f}, beta_k={param_dict['beta_k']:.3f}")
    if 'burst_size' in param_dict:
        print(f"  burst_size={param_dict['burst_size']:.2f}")
    if 'n_inner' in param_dict:
        print(f"  n_inner={param_dict['n_inner']:.2f}")
    if 'beta2_k' in param_dict:
        print(f"  beta2_k={param_dict['beta2_k']:.3f}")
    if 'beta3_k' in param_dict:
        print(f"  beta3_k={param_dict['beta3_k']:.3f}")
    
    if expected_nll is not None:
        print(f"\nExpected NLL from CSV: {expected_nll:.4f}")
    
    # Load experimental data using simulation_utils function
    print("\nLoading experimental data...")
    datasets = load_experimental_data()
    print(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}")
    
    # Calculate MoM NLL using functions from MoMOptimization_join.py
    print("\n" + "-"*70)
    print("Calculating MoM NLL (using MoMOptimization_join.py functions)...")
    print("-"*70)
    mom_nll, mom_individual_nlls = calculate_mom_nll(mechanism, params_vector, datasets)
    print(f"MoM Total NLL: {mom_nll:.4f}")
    print("\nMoM Individual NLLs:")
    for dataset_name, nll_value in mom_individual_nlls.items():
        print(f"  {dataset_name:15s}: {nll_value:.4f}")
    
    # Calculate Simulation KDE NLL using functions from simulation_utils.py
    print("\n" + "-"*70)
    print(f"Calculating Simulation KDE NLL ({num_simulations} simulations)...")
    print("Using functions from simulation_utils.py")
    print("-"*70)
    sim_nll, sim_individual_nlls = calculate_simulation_kde_nll(mechanism, params_vector, datasets, num_simulations)
    print(f"Simulation KDE Total NLL: {sim_nll:.4f}")
    print("\nSimulation KDE Individual NLLs:")
    for dataset_name, nll_value in sim_individual_nlls.items():
        print(f"  {dataset_name:15s}: {nll_value:.4f}")
    
    # Print comparison of individual NLLs
    print("\n" + "-"*70)
    print("Individual NLL Comparison (MoM vs Simulation):")
    print("-"*70)
    print(f"{'Dataset':<15s} {'MoM NLL':>12s} {'Sim NLL':>12s} {'Difference':>12s} {'% Diff':>10s}")
    print("-" * 70)
    for dataset_name in mom_individual_nlls.keys():
        mom_val = mom_individual_nlls[dataset_name]
        sim_val = sim_individual_nlls.get(dataset_name, np.nan)
        diff = abs(mom_val - sim_val)
        pct_diff = (diff / max(abs(mom_val), 1e-10)) * 100 if not np.isnan(sim_val) else np.nan
        print(f"{dataset_name:<15s} {mom_val:>12.4f} {sim_val:>12.4f} {diff:>12.4f} {pct_diff:>9.2f}%")
    
    # Plot distributions for comparison
    print("\n" + "-"*70)
    print("Generating distribution plots (MoM vs KDE)...")
    print("-"*70)
    plot_mom_vs_kde_distributions(mechanism, params_vector, datasets, num_simulations)
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    if expected_nll is not None:
        print(f"Expected NLL (from CSV): {expected_nll:.4f}")
        print(f"MoM NLL:                  {mom_nll:.4f}")
        print(f"Simulation KDE NLL:       {sim_nll:.4f}")
        print()
        print("Comparison with Expected:")
        mom_diff = abs(mom_nll - expected_nll)
        sim_diff = abs(sim_nll - expected_nll)
        print(f"  MoM vs Expected:        {mom_diff:.4f} ({mom_diff/max(abs(expected_nll), 1e-10)*100:.2f}%)")
        print(f"  Simulation vs Expected: {sim_diff:.4f} ({sim_diff/max(abs(expected_nll), 1e-10)*100:.2f}%)")
    else:
        print(f"MoM NLL:              {mom_nll:.4f}")
        print(f"Simulation KDE NLL:   {sim_nll:.4f}")
    
    print()
    print("Direct Comparison:")
    direct_diff = abs(mom_nll - sim_nll)
    relative_diff = direct_diff / max(abs(mom_nll), 1e-10) * 100
    print(f"  MoM vs Simulation:     {direct_diff:.4f} ({relative_diff:.2f}%)")
    
    # Interpretation
    if expected_nll is not None:
        # Check agreement with expected
        mom_agreement = mom_diff / max(abs(expected_nll), 1e-10) * 100
        sim_agreement = sim_diff / max(abs(expected_nll), 1e-10) * 100
        
        print("\n" + "-"*70)
        print("AGREEMENT WITH EXPECTED NLL")
        print("-"*70)
        if mom_agreement < 0.1:
            print(f"✓ MoM: Excellent agreement with expected (< 0.1% difference)")
        elif mom_agreement < 1.0:
            print(f"✓ MoM: Good agreement with expected (< 1% difference)")
        elif mom_agreement < 5.0:
            print(f"⚠ MoM: Moderate agreement with expected (< 5% difference)")
        else:
            print(f"⚠ MoM: Significant difference from expected (> 5% difference)")
        
        if sim_agreement < 1.0:
            print(f"✓ Simulation: Good agreement with expected (< 1% difference)")
        elif sim_agreement < 5.0:
            print(f"⚠ Simulation: Moderate agreement with expected (< 5% difference)")
        elif sim_agreement < 10.0:
            print(f"⚠ Simulation: Significant difference from expected (< 10% difference)")
        else:
            print(f"⚠ Simulation: Large difference from expected (> 10% difference)")
            print("  This may indicate:")
            print("  - Insufficient simulations for KDE")
            print("  - KDE bandwidth issues")
            print("  - Numerical precision issues")
    
    print("\n" + "-"*70)
    print("MoM vs Simulation Direct Comparison")
    print("-"*70)
    if relative_diff < 1.0:
        print("✓ Excellent agreement between MoM and Simulation (< 1% difference)")
    elif relative_diff < 5.0:
        print("✓ Good agreement between MoM and Simulation (< 5% difference)")
    elif relative_diff < 10.0:
        print("⚠ Moderate agreement between MoM and Simulation (< 10% difference)")
    else:
        print("⚠ Significant difference between MoM and Simulation (> 10% difference)")
        print("  This may indicate:")
        print("  - MoM approximation limitations")
        print("  - Insufficient simulations for KDE")
        print("  - Numerical issues in either method")
    
    return {
        'mechanism': mechanism,
        'run_number': actual_run,
        'expected_nll': expected_nll,
        'mom_nll': mom_nll,
        'sim_nll': sim_nll,
        'mom_vs_expected_diff': mom_diff if expected_nll is not None else None,
        'sim_vs_expected_diff': sim_diff if expected_nll is not None else None,
        'mom_vs_sim_diff': direct_diff,
        'relative_difference_pct': relative_diff,
        'num_simulations': num_simulations
    }


def main():
    """
    Main function to run comparisons.
    
    Configure settings below to test different mechanisms and parameters.
    """
    # ========== CONFIGURATION SETTINGS ==========
    
    # Mechanisms to test (list of mechanism names)
    # Options: 'simple', 'fixed_burst', 'feedback_onion', 'fixed_burst_feedback_onion'
    mechanisms_to_test = ['simple']
    
    # Path to CSV file with optimized parameters
    csv_file = 'optimized_params_runs_20251120_150408.csv'
    #optimized_params_runs_20251121_072718.csv
    
    # Run number to test (None = use first available run for each mechanism)
    run_number = 1
    
    # Number of simulations for KDE approach
    num_simulations = 1000
    
    # ============================================
    
    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        print(f"Current directory: {os.getcwd()}")
        print(f"\nAvailable CSV files:")
        for f in os.listdir('.'):
            if f.endswith('.csv'):
                print(f"  - {f}")
        return 1
    
    # Validate mechanisms
    valid_mechanisms = ['simple', 'fixed_burst', 'feedback_onion', 'fixed_burst_feedback_onion']
    for mechanism in mechanisms_to_test:
        if mechanism not in valid_mechanisms:
            print(f"Error: Invalid mechanism '{mechanism}'. Valid options: {valid_mechanisms}")
            return 1
    
    # Run tests for each mechanism
    all_results = []
    
    try:
        for mechanism in mechanisms_to_test:
            print("\n" + "="*70)
            print(f"Testing mechanism: {mechanism}")
            print("="*70)
            
            results = compare_mom_vs_simulation(
                mechanism,
                csv_file,
                run_number,
                num_simulations
            )
            
            all_results.append(results)
            
            print("\n" + "="*70)
            print(f"Test completed for {mechanism}")
            print("="*70)
        
        # Summary
        if len(mechanisms_to_test) > 1:
            print("\n" + "="*70)
            print("SUMMARY - ALL MECHANISMS")
            print("="*70)
            for result in all_results:
                print(f"\n{result['mechanism']} (run {result['run_number']}):")
                if result['expected_nll'] is not None:
                    print(f"  Expected NLL:  {result['expected_nll']:.4f}")
                print(f"  MoM NLL:        {result['mom_nll']:.4f}")
                print(f"  Simulation NLL: {result['sim_nll']:.4f}")
                if result['expected_nll'] is not None:
                    mom_diff = result['mom_vs_expected_diff']
                    sim_diff = result['sim_vs_expected_diff']
                    print(f"  MoM vs Expected:    {mom_diff:.4f} ({mom_diff/max(abs(result['expected_nll']), 1e-10)*100:.2f}%)")
                    print(f"  Simulation vs Expected: {sim_diff:.4f} ({sim_diff/max(abs(result['expected_nll']), 1e-10)*100:.2f}%)")
                print(f"  MoM vs Simulation: {result['mom_vs_sim_diff']:.4f} ({result['relative_difference_pct']:.2f}%)")
        
        print("\n" + "="*70)
        print("All tests completed successfully!")
        print("="*70)
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

