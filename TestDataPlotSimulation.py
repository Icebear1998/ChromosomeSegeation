#!/usr/bin/env python3
"""
Test and plot simulation results using optimized parameters from simulation-based optimization.
Works with MultiMechanismSimulationTimevary and the new data structure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MultiMechanismSimulationTimevary import MultiMechanismSimulationTimevary
from SimulationOptimization_join import load_experimental_data, apply_mutant_params
import warnings
warnings.filterwarnings('ignore')


def load_optimized_parameters(mechanism, filename=None):
    """
    Load optimized parameters from file and calculate derived parameters.
    
    Args:
        mechanism (str): Mechanism name
        filename (str): Parameter file name (optional)
    
    Returns:
        dict: Optimized parameters including both ratio and derived parameters
    """
    if filename is None:
        filename = f"simulation_optimized_parameters_{mechanism}.txt"
    
    try:
        params = {}
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        # Find the parameters section
        param_section = False
        derived_section = False
        
        mutant_section = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # Handle different parameter section headers
            if ("Optimized Parameters (ratio-based):" in line or 
                "Wildtype Parameters (ratio-based):" in line):
                param_section = True
                derived_section = False
                mutant_section = False
                continue
            elif ("Derived Parameters:" in line or 
                  "Derived Wildtype Parameters:" in line):
                param_section = False
                derived_section = True
                mutant_section = False
                continue
            elif ("=== MUTANT PARAMETERS ===" in line or 
                  line_stripped.endswith("Mutant:") and not line_stripped.startswith("=")):
                param_section = False
                derived_section = False
                mutant_section = True
                continue
            elif line_stripped.startswith("===") or line_stripped.startswith("---"):
                param_section = False
                derived_section = False
                mutant_section = False
                continue
            
            # Parse parameter lines
            if "=" in line_stripped and not line_stripped.startswith("#"):
                if param_section or derived_section or mutant_section:
                    try:
                        key, value = line_stripped.split(" = ", 1)
                        params[key.strip()] = float(value.strip())
                    except ValueError:
                        continue
        
        # If we have ratio parameters but not derived ones, calculate them
        if 'r21' in params and 'n1' not in params:
            params['n1'] = max(params['r21'] * params['n2'], 1)
            params['n3'] = max(params['r23'] * params['n2'], 1)
            params['N1'] = max(params['R21'] * params['N2'], 1)
            params['N3'] = max(params['R23'] * params['N2'], 1)
            print("Calculated derived parameters from ratios")
        
        print(f"Loaded parameters from {filename}")
        print(f"Available parameters: {list(params.keys())}")
        return params
    
    except Exception as e:
        print(f"Error loading parameters: {e}")
        return {}


def run_simulation_with_params(mechanism, params, mutant_type, alpha, beta_k, beta2_k, num_simulations=500):
    """
    Run simulations with given parameters for a specific mutant type.
    
    Args:
        mechanism (str): Mechanism name
        params (dict): Base parameters
        mutant_type (str): Mutant type
        alpha, beta_k, beta2_k (float): Mutant parameters
        num_simulations (int): Number of simulations
    
    Returns:
        tuple: (delta_t12_list, delta_t32_list)
    """
    try:
        # Create base parameters dict
        base_params = {
            'n1': params['n1'], 'n2': params['n2'], 'n3': params['n3'],
            'N1': params['N1'], 'N2': params['N2'], 'N3': params['N3'],
            'k_1': params['k_1'], 'k_max': params['k_max']
        }
        
        # Add mechanism-specific parameters
        if mechanism == 'time_varying_k_fixed_burst':
            base_params['burst_size'] = params['burst_size']
        elif mechanism == 'time_varying_k_feedback_onion':
            base_params['n_inner'] = params['n_inner']
        elif mechanism == 'time_varying_k_combined':
            base_params['burst_size'] = params['burst_size']
            base_params['n_inner'] = params['n_inner']
        
        # Apply mutant modifications
        mutant_params, n0_list = apply_mutant_params(base_params, mutant_type, alpha, beta_k, beta2_k)
        
        # Extract rate parameters
        initial_state = [mutant_params['N1'], mutant_params['N2'], mutant_params['N3']]
        
        if mechanism == 'time_varying_k':
            rate_params = {
                'k_1': mutant_params['k_1'],
                'k_max': mutant_params['k_max']
            }
        elif mechanism == 'time_varying_k_fixed_burst':
            rate_params = {
                'k_1': mutant_params['k_1'],
                'k_max': mutant_params['k_max'],
                'burst_size': mutant_params['burst_size']
            }
        elif mechanism == 'time_varying_k_feedback_onion':
            rate_params = {
                'k_1': mutant_params['k_1'],
                'k_max': mutant_params['k_max'],
                'n_inner': mutant_params['n_inner']
            }
        elif mechanism == 'time_varying_k_combined':
            rate_params = {
                'k_1': mutant_params['k_1'],
                'k_max': mutant_params['k_max'],
                'burst_size': mutant_params['burst_size'],
                'n_inner': mutant_params['n_inner']
            }
        
        # Run simulations
        delta_t12_list = []
        delta_t32_list = []
        
        for _ in range(num_simulations):
            sim = MultiMechanismSimulationTimevary(
                mechanism=mechanism,
                initial_state_list=initial_state,
                rate_params=rate_params,
                n0_list=n0_list,
                max_time=1000
            )
            
            _, _, sep_times = sim.simulate()
            
            delta_t12 = sep_times[0] - sep_times[1]  # T1 - T2
            delta_t32 = sep_times[2] - sep_times[1]  # T3 - T2
            
            delta_t12_list.append(delta_t12)
            delta_t32_list.append(delta_t32)
        
        return delta_t12_list, delta_t32_list
    
    except Exception as e:
        print(f"Simulation error for {mutant_type}: {e}")
        return [], []


def create_comparison_plot(mechanism, params, experimental_data, num_simulations=500):
    """
    Create comparison plots between experimental data and simulation results.
    
    Args:
        mechanism (str): Mechanism name
        params (dict): Optimized parameters
        experimental_data (dict): Experimental datasets
        num_simulations (int): Number of simulations per dataset
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Simulation vs Experimental Data: {mechanism.upper()}', fontsize=16, y=0.98)
    
    dataset_names = ['wildtype', 'threshold', 'degrate', 'degrateAPC']
    dataset_titles = ['Wildtype', 'Threshold', 'Separase', 'APC']
    
    alpha = params['alpha']
    beta_k = params['beta_k']
    beta2_k = params['beta2_k']
    
    for i, (dataset_name, title) in enumerate(zip(dataset_names, dataset_titles)):
        if dataset_name not in experimental_data:
            continue
        
        # Get experimental data
        exp_delta_t12 = experimental_data[dataset_name]['delta_t12']
        exp_delta_t32 = experimental_data[dataset_name]['delta_t32']
        
        # Run simulations
        print(f"Running simulations for {dataset_name}...")
        sim_delta_t12, sim_delta_t32 = run_simulation_with_params(
            mechanism, params, dataset_name, alpha, beta_k, beta2_k, num_simulations
        )
        
        if not sim_delta_t12 or not sim_delta_t32:
            continue
        
        # Plot T1-T2 (top row) - matching TestDataPlot.py style
        ax_12 = axes[0, i]
        ax_12.hist(exp_delta_t12, bins=15, alpha=0.6, label='Experimental data', color='lightblue', density=True)
        ax_12.hist(sim_delta_t12, bins=15, alpha=0.6, label='Simulated data', color='orange', density=True)
        ax_12.set_xlim(-150, 150)
        ax_12.set_title(f'{title}\nChrom1-Chrom2')
        ax_12.set_xlabel('Time Difference')
        ax_12.set_ylabel('Density')
        ax_12.legend(fontsize=8)
        ax_12.grid(True, alpha=0.3)
        
        # Add statistics
        exp_mean_12 = np.mean(exp_delta_t12)
        sim_mean_12 = np.mean(sim_delta_t12)
        ax_12.axvline(exp_mean_12, color='blue', linestyle='--', alpha=0.8)
        ax_12.axvline(sim_mean_12, color='red', linestyle='--', alpha=0.8)
        stats_text12 = f'Exp: μ={exp_mean_12:.1f}, σ={np.std(exp_delta_t12):.1f}\nSim: μ={sim_mean_12:.1f}, σ={np.std(sim_delta_t12):.1f}'
        ax_12.text(0.02, 0.98, stats_text12, transform=ax_12.transAxes, 
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot T3-T2 (bottom row) - matching TestDataPlot.py style
        ax_32 = axes[1, i]
        ax_32.hist(exp_delta_t32, bins=15, alpha=0.6, label='Experimental data', color='lightblue', density=True)
        ax_32.hist(sim_delta_t32, bins=15, alpha=0.6, label='Simulated data', color='orange', density=True)
        ax_32.set_xlim(-150, 150)
        ax_32.set_title(f'Chrom3-Chrom2')
        ax_32.set_xlabel('Time Difference')
        ax_32.set_ylabel('Density')
        ax_32.legend(fontsize=8)
        ax_32.grid(True, alpha=0.3)
        
        # Add statistics
        exp_mean_32 = np.mean(exp_delta_t32)
        sim_mean_32 = np.mean(sim_delta_t32)
        ax_32.axvline(exp_mean_32, color='blue', linestyle='--', alpha=0.8)
        ax_32.axvline(sim_mean_32, color='red', linestyle='--', alpha=0.8)
        stats_text32 = f'Exp: μ={exp_mean_32:.1f}, σ={np.std(exp_delta_t32):.1f}\nSim: μ={sim_mean_32:.1f}, σ={np.std(sim_delta_t32):.1f}'
        ax_32.text(0.02, 0.98, stats_text32, transform=ax_32.transAxes, 
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Adjusted for title space
    
    # Save plot
    filename = f'simulation_fit_{mechanism}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {filename}")
    
    plt.show()


def create_single_dataset_plot(mechanism, params, experimental_data, dataset_name, num_simulations=500):
    """
    Create a single dataset comparison plot (2 subplots: T1-T2 and T3-T2).
    
    Args:
        mechanism (str): Mechanism name
        params (dict): Optimized parameters
        experimental_data (dict): Experimental datasets
        dataset_name (str): Dataset to plot
        num_simulations (int): Number of simulations
    """
    if dataset_name not in experimental_data:
        print(f"Dataset {dataset_name} not found in experimental data")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{dataset_name.title()} Dataset - {mechanism.replace("_", " ").title()} Mechanism', fontsize=14)
    
    # Get experimental data
    exp_delta_t12 = experimental_data[dataset_name]['delta_t12']
    exp_delta_t32 = experimental_data[dataset_name]['delta_t32']
    
    alpha = params['alpha']
    beta_k = params['beta_k']
    beta2_k = params['beta2_k']
    
    # Run simulations
    print(f"Running {num_simulations} simulations for {dataset_name}...")
    sim_delta_t12, sim_delta_t32 = run_simulation_with_params(
        mechanism, params, dataset_name, alpha, beta_k, beta2_k, num_simulations
    )
    
    if not sim_delta_t12 or not sim_delta_t32:
        print(f"Failed to generate simulation data for {dataset_name}")
        return
    
    # Plot Chrom1 - Chrom2
    ax1.hist(exp_delta_t12, bins=15, density=True, alpha=0.6, label='Experimental data', color='lightblue')
    ax1.hist(sim_delta_t12, bins=15, density=True, alpha=0.6, label='Simulated data', color='orange')
    ax1.set_xlim(-150, 150)
    ax1.set_xlabel('Time Difference')
    ax1.set_ylabel('Density')
    ax1.set_title(f"Chrom1 - Chrom2 ({dataset_name}, {mechanism.replace('_', ' ').title()})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics for Chrom1 - Chrom2
    stats_text12 = f'Exp: μ={np.mean(exp_delta_t12):.1f}, σ={np.std(exp_delta_t12):.1f}\nSim: μ={np.mean(sim_delta_t12):.1f}, σ={np.std(sim_delta_t12):.1f}'
    ax1.text(0.02, 0.98, stats_text12, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot Chrom3 - Chrom2
    ax2.hist(exp_delta_t32, bins=14, density=True, alpha=0.6, label='Experimental data', color='lightblue')
    ax2.hist(sim_delta_t32, bins=14, density=True, alpha=0.6, label='Simulated data', color='orange')
    ax2.set_xlim(-150, 150)
    ax2.set_xlabel('Time Difference')
    ax2.set_ylabel('Density')
    ax2.set_title(f"Chrom3 - Chrom2 ({dataset_name}, {mechanism.replace('_', ' ').title()})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics for Chrom3 - Chrom2
    stats_text32 = f'Exp: μ={np.mean(exp_delta_t32):.1f}, σ={np.std(exp_delta_t32):.1f}\nSim: μ={np.mean(sim_delta_t32):.1f}, σ={np.std(sim_delta_t32):.1f}'
    ax2.text(0.02, 0.98, stats_text32, transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    filename = f'simulation_fit_{mechanism}_{dataset_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {filename}")
    
    plt.show()
    
    # Print statistics
    print(f"\nStatistics for {dataset_name} dataset:")
    print(f"Experimental T1-T2 - Mean: {np.mean(exp_delta_t12):.2f}, Std: {np.std(exp_delta_t12):.2f}")
    print(f"Simulated T1-T2   - Mean: {np.mean(sim_delta_t12):.2f}, Std: {np.std(sim_delta_t12):.2f}")
    print(f"Experimental T3-T2 - Mean: {np.mean(exp_delta_t32):.2f}, Std: {np.std(exp_delta_t32):.2f}")
    print(f"Simulated T3-T2   - Mean: {np.mean(sim_delta_t32):.2f}, Std: {np.std(sim_delta_t32):.2f}")


def print_parameter_summary(mechanism, params):
    """
    Print a summary of the optimized parameters.
    
    Args:
        mechanism (str): Mechanism name
        params (dict): Optimized parameters
    """
    print(f"\n=== Parameter Summary for {mechanism.upper()} ===")
    
    # Show ratio parameters if available
    if 'r21' in params:
        print(f"Base Parameters:")
        print(f"  n2={params['n2']:.1f}, N2={params['N2']:.1f}")
        print(f"  Ratios: r21={params['r21']:.2f}, r23={params['r23']:.2f}, R21={params['R21']:.2f}, R23={params['R23']:.2f}")
    
    print(f"Wildtype Parameters:")
    print(f"  Thresholds: n1={params['n1']:.1f}, n2={params['n2']:.1f}, n3={params['n3']:.1f}")
    print(f"  Initial counts: N1={params['N1']:.1f}, N2={params['N2']:.1f}, N3={params['N3']:.1f}")
    print(f"  Rates: k_1={params['k_1']:.6f}, k_max={params['k_max']:.4f}")
    
    if 'burst_size' in params:
        print(f"  Burst size: {params['burst_size']:.1f}")
    if 'n_inner' in params:
        print(f"  Inner threshold: {params['n_inner']:.1f}")
    if 'burst_size' in params and 'n_inner' in params:
        print(f"  Combined mechanism: burst_size={params['burst_size']:.1f}, n_inner={params['n_inner']:.1f}")
    
    print(f"\nMutant Modifiers:")
    print(f"  Threshold mutant (alpha): {params['alpha']:.3f}")
    print(f"  Separase mutant (beta_k): {params['beta_k']:.3f}")
    print(f"  APC mutant (beta2_k): {params['beta2_k']:.3f}")
    
    # Show effective parameters for each mutant
    print(f"\nEffective Parameters by Mutant:")
    print(f"  Threshold: n1={params['alpha']*params['n1']:.1f}, n2={params['alpha']*params['n2']:.1f}, n3={params['alpha']*params['n3']:.1f}")
    print(f"  Separase: k_max={params['beta_k']*params['k_max']:.4f}")
    print(f"  APC: k_1={params['beta2_k']*params['k_1']:.6f}")


def main():
    """
    Main testing and plotting routine.
    """
    print("Testing Simulation-based Optimization Results")
    print("=" * 50)
    
    # Load experimental data
    experimental_data = load_experimental_data()
    if not experimental_data:
        print("Error: Could not load experimental data!")
        return
    
    # Test available mechanisms
    mechanisms = ['time_varying_k', 'time_varying_k_fixed_burst', 'time_varying_k_feedback_onion', 'time_varying_k_combined']
    
    for mechanism in mechanisms:
        print(f"\n{'-' * 50}")
        print(f"Testing {mechanism}")
        
        # Load optimized parameters
        params = load_optimized_parameters(mechanism)
        if not params:
            print(f"Skipping {mechanism} - no parameters found")
            continue
        
        # Print parameter summary
        print_parameter_summary(mechanism, params)
        
        # Create comparison plots
        try:
            create_comparison_plot(mechanism, params, experimental_data, num_simulations=200)
        except Exception as e:
            print(f"Error creating plots for {mechanism}: {e}")
            continue
    
    print(f"\n{'=' * 50}")
    print("Testing complete!")


if __name__ == "__main__":
    # ========== CONFIGURATION ==========
    # Choose what to run
    run_all_mechanisms = False  # Set to True to test all mechanisms
    
    # Single dataset configuration (only used if run_single_dataset = True)
    mechanism = 'time_varying_k_combined'
    #filename = 'simulation_optimized_parameters_time_varying_k_combined_independent.txt'
    filename = 'bayesian_optimized_parameters_time_varying_k_combined.txt'
    dataset = 'wildtype'  # Choose: 'wildtype', 'threshold', 'degrate', 'degrateAPC'
    
    if run_all_mechanisms:
        main()
    else:
        print(f"Testing single dataset: {dataset} with mechanism: {mechanism}")
        print("=" * 50)
        
        # Load experimental data
        experimental_data = load_experimental_data()
        if not experimental_data:
            print("Error: Could not load experimental data!")
        else:
            # Load optimized parameters
            params = load_optimized_parameters(mechanism, filename)
            if params:
                print_parameter_summary(mechanism, params)
                create_single_dataset_plot(mechanism, params, experimental_data, dataset, num_simulations=500)
            else:
                print(f"Could not load parameters for {mechanism}") 