#!/usr/bin/env python3
"""
Test and plot simulation results using optimized parameters from simulation-based optimization.
Works with MultiMechanismSimulationTimevary and the new data structure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from MultiMechanismSimulationTimevary import MultiMechanismSimulationTimevary
from simulation_utils import load_experimental_data, apply_mutant_params, calculate_emd, calculate_wasserstein_p_value
import warnings
warnings.filterwarnings('ignore')


def load_optimized_parameters(mechanism, filename=None):
    """
    Load optimized parameters from file and calculate derived parameters.
    Also extracts Total Validation EMD if available.
    
    Args:
        mechanism (str): Mechanism name
        filename (str): Parameter file name (optional)
    
    Returns:
        dict: Optimized parameters including both ratio and derived parameters, plus 'total_emd' if found
    """
    if filename is None:
        filename = f"simulation_optimized_parameters_{mechanism}.txt"
    
    try:
        params = {}
        total_emd = None
        
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        # Find the parameters section
        param_section = False
        derived_section = False
        mutant_section = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # Extract Total Validation EMD
            if "Total Validation EMD:" in line:
                try:
                    total_emd = float(line.split(':')[1].strip())
                    params['total_emd'] = total_emd
                except:
                    pass
            
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
        if 'r21' in params:
            params['n1'] = max(params['r21'] * params['n2'], 1)
            params['n3'] = max(params['r23'] * params['n2'], 1)
            params['N1'] = max(params['R21'] * params['N2'], 1)
            params['N3'] = max(params['R23'] * params['N2'], 1)
            print("Calculated derived parameters from ratios")
        
        print(f"Loaded parameters from {filename}")
        if total_emd is not None:
            print(f"Total Validation EMD: {total_emd:.2f}")
        print(f"Available parameters: {list(params.keys())}")
        return params
    
    except Exception as e:
        print(f"Error loading parameters: {e}")
        return {}


def run_simulation_with_params(mechanism, params, mutant_type, alpha, beta_k, beta_tau, beta_tau2=None, beta_k1=None, beta_k2=None, beta_k3=None, num_simulations=500):
    """
    Run simulations with given parameters for a specific mutant type.
    
    Args:
        mechanism (str): Mechanism name
        params (dict): Base parameters
        mutant_type (str): Mutant type
    Args:
        mechanism (str): Mechanism name
        params (dict): Base parameters
        mutant_type (str): Mutant type
        alpha, beta_k, beta_tau (float): Mutant parameters
        beta_k1, beta_k2, beta_k3 (float): Simple mutant parameters
        beta_tau2 (float): Velcade mutant parameter (optional)
        num_simulations (int): Number of simulations
    
    Returns:
        tuple: (delta_t12_list, delta_t32_list)
    """
    try:
        # Create base parameters dict
        base_params = {
            'n1': params['n1'], 'n2': params['n2'], 'n3': params['n3'],
            'N1': params['N1'], 'N2': params['N2'], 'N3': params['N3']
        }
        
        # Add rate parameter - either k (simple mechanisms) or k_max (time-varying mechanisms)
        if 'k_max' in params:
            base_params['k_max'] = params['k_max']
        elif 'k' in params:
            base_params['k'] = params['k']
        
        # Add tau parameter if available
        if 'tau' in params:
            base_params['tau'] = params['tau']
        elif 'k_1' in params:
            base_params['k_1'] = params['k_1']
        
        # k_1 calculation is now handled by simulation_utils.calculate_k1_from_params
        
        # Add mechanism-specific parameters
        if mechanism == 'fixed_burst':
            base_params['burst_size'] = params['burst_size']
        elif mechanism == 'feedback_onion':
            base_params['n_inner'] = params['n_inner']
        elif mechanism == 'fixed_burst_feedback_onion':
            base_params['burst_size'] = params['burst_size']
            base_params['n_inner'] = params['n_inner']
        elif mechanism == 'time_varying_k_fixed_burst':
            base_params['burst_size'] = params['burst_size']
        elif mechanism == 'time_varying_k_feedback_onion':
            base_params['n_inner'] = params['n_inner']
        elif mechanism == 'time_varying_k_combined':
            base_params['burst_size'] = params['burst_size']
            base_params['n_inner'] = params['n_inner']
        
        # Apply mutant modifications
        mutant_params, n0_list = apply_mutant_params(base_params, mutant_type, alpha, beta_k, beta_k1, beta_k2, beta_k3, beta_tau, beta_tau2)
        
        # Extract rate parameters
        initial_state = [mutant_params['N1'], mutant_params['N2'], mutant_params['N3']]
        
        # Build rate_params based on mechanism type
        if mechanism == 'simple':
            rate_params = {
                'k': mutant_params['k']
            }
        elif mechanism == 'fixed_burst':
            rate_params = {
                'k': mutant_params['k'],
                'burst_size': mutant_params['burst_size']
            }
        elif mechanism == 'feedback_onion':
            rate_params = {
                'k': mutant_params['k'],
                'n_inner': mutant_params['n_inner']
            }
        elif mechanism == 'fixed_burst_feedback_onion':
            rate_params = {
                'k': mutant_params['k'],
                'burst_size': mutant_params['burst_size'],
                'n_inner': mutant_params['n_inner']
            }
        elif mechanism == 'time_varying_k':
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
        
        
        # Run simulations using simulation_utils which handles all mechanism types
        # This will automatically dispatch to the appropriate simulator (Gillespie, FastBeta, or FastFeedback)
        from simulation_utils import run_simulation_for_dataset
        
        delta_t12_array, delta_t32_array = run_simulation_for_dataset(
            mechanism=mechanism,
            params=mutant_params,
            n0_list=n0_list,
            num_simulations=num_simulations
        )
        
        # Convert to lists and return
        delta_t12_list = delta_t12_array.tolist() if isinstance(delta_t12_array, np.ndarray) else list(delta_t12_array)
        delta_t32_list = delta_t32_array.tolist() if isinstance(delta_t32_array, np.ndarray) else list(delta_t32_array)
        
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
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f'Simulation vs Experimental Data: {mechanism.upper()}', fontsize=16, y=0.98)
    
    dataset_names = ['wildtype', 'threshold', 'degrade', 'degradeAPC', 'velcade']
    dataset_titles = ['Wildtype', 'Threshold', 'Separase', 'APC', 'Velcade']
    
    alpha = params['alpha']
    alpha = params['alpha']
    beta_k = params.get('beta_k', None)
    beta_k1 = params.get('beta_k1', None)
    beta_k2 = params.get('beta_k2', None)
    beta_k3 = params.get('beta_k3', None)
    beta_tau = params.get('beta_tau', None)  # Default to None if not present
    beta_tau2 = params.get('beta_tau2', None)  # Default to None if not present
    
    for i, (dataset_name, title) in enumerate(zip(dataset_names, dataset_titles)):
        if dataset_name not in experimental_data:
            continue
        
        # Get experimental data
        exp_delta_t12 = experimental_data[dataset_name]['delta_t12']
        exp_delta_t32 = experimental_data[dataset_name]['delta_t32']
        
        # Run simulations
        print(f"Running simulations for {dataset_name}...")
        sim_delta_t12, sim_delta_t32 = run_simulation_with_params(
            mechanism, params, dataset_name, alpha, beta_k, beta_tau, beta_tau2, beta_k1, beta_k2, beta_k3, num_simulations
        )
        
        if not sim_delta_t12 or not sim_delta_t32:
            continue
        
        # Save raw data to CSV
        csv_data = {
            'exp_delta_t12': exp_delta_t12,
            'exp_delta_t32': exp_delta_t32,
            'sim_delta_t12': sim_delta_t12,
            'sim_delta_t32': sim_delta_t32
        }
        # Create DataFrame with padding for unequal lengths
        max_len = max(len(exp_delta_t12), len(exp_delta_t32), len(sim_delta_t12), len(sim_delta_t32))
        csv_df = pd.DataFrame({
            'exp_delta_t12': pd.Series(exp_delta_t12).reindex(range(max_len)),
            'exp_delta_t32': pd.Series(exp_delta_t32).reindex(range(max_len)),
            'sim_delta_t12': pd.Series(sim_delta_t12).reindex(range(max_len)),
            'sim_delta_t32': pd.Series(sim_delta_t32).reindex(range(max_len))
        })
        csv_filename = f'simulation_data_{mechanism}_{dataset_name}.csv'
        csv_df.to_csv(csv_filename, index=False)
        print(f"  Saved data to {csv_filename}")
        
        # Set up x_grid for PDF plotting
        x_min, x_max = -140, 140  # Default range for all mechanisms
        if dataset == "velcade":
            x_min, x_max = -350, 350
        if dataset in ["wildtype", "threshold"]:
            x_min, x_max = -100, 100    
        x_grid = np.linspace(x_min, x_max, 401)
        # Plot T1-T2 (top row) - matching TestDataPlot.py style
        # Calculate bins for 7 second intervals
        bin_width = 7.0
        all_data_12 = np.concatenate([exp_delta_t12, sim_delta_t12])
        bins_12 = np.arange(np.floor(all_data_12.min() / bin_width) * bin_width, 
                            np.ceil(all_data_12.max() / bin_width) * bin_width + bin_width, 
                            bin_width)
        
        ax_12 = axes[0, i]
        ax_12.hist(exp_delta_t12, bins=bins_12, alpha=0.6, label='Experimental data', color='lightblue', density=True)
        ax_12.hist(sim_delta_t12, bins=bins_12, alpha=0.6, label='Simulated data', color='orange', density=True)
        ax_12.set_xlim(x_min - 20, x_max + 20)
        ax_12.set_title(f'{title}')  # Only dataset name
        ax_12.set_xlabel('Time Difference (T1-T2)')
        ax_12.set_ylabel('Percentage of cells')
        ax_12.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y*100:.1f}'))
        ax_12.legend(fontsize=8)
        ax_12.grid(True, alpha=0.3)
        
        # Calculate EMD and Wasserstein p-value for T1-T2
        p_value_12, emd_12 = calculate_wasserstein_p_value(exp_delta_t12, sim_delta_t12, num_permutations=1000)
        
        # Add statistics with EMD and p-value
        exp_mean_12 = np.mean(exp_delta_t12)
        sim_mean_12 = np.mean(sim_delta_t12)
        ax_12.axvline(exp_mean_12, color='blue', linestyle='--', alpha=0.8)
        ax_12.axvline(sim_mean_12, color='red', linestyle='--', alpha=0.8)
        stats_text12 = f'Exp: μ={exp_mean_12:.1f}, σ={np.std(exp_delta_t12):.1f}\nSim: μ={sim_mean_12:.1f}, σ={np.std(sim_delta_t12):.1f}\nEMD={emd_12:.2f}, p={p_value_12:.3f}\nN={num_simulations}'
        ax_12.text(0.02, 0.98, stats_text12, transform=ax_12.transAxes, 
                   verticalalignment='top', fontsize=7,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot T3-T2 (bottom row) - matching TestDataPlot.py style
        # Calculate bins for 3.5 second intervals
        all_data_32 = np.concatenate([exp_delta_t32, sim_delta_t32])
        bins_32 = np.arange(np.floor(all_data_32.min() / bin_width) * bin_width, 
                            np.ceil(all_data_32.max() / bin_width) * bin_width + bin_width, 
                            bin_width)
        
        ax_32 = axes[1, i]
        ax_32.hist(exp_delta_t32, bins=bins_32, alpha=0.6, label='Experimental data', color='lightblue', density=True)
        ax_32.hist(sim_delta_t32, bins=bins_32, alpha=0.6, label='Simulated data', color='orange', density=True)
        ax_32.set_xlim(x_min- 20, x_max + 20)
        ax_32.set_title('')  # No subtitle for bottom row
        ax_32.set_xlabel('Time Difference (T3-T2)')
        ax_32.set_ylabel('Percentage of cells')
        ax_32.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y*100:.1f}'))
        ax_32.legend(fontsize=8)
        ax_32.grid(True, alpha=0.3)
        
        # Calculate EMD and Wasserstein p-value for T3-T2
        p_value_32, emd_32 = calculate_wasserstein_p_value(exp_delta_t32, sim_delta_t32, num_permutations=1000)
        
        # Add statistics with EMD and p-value
        exp_mean_32 = np.mean(exp_delta_t32)
        sim_mean_32 = np.mean(sim_delta_t32)
        ax_32.axvline(exp_mean_32, color='blue', linestyle='--', alpha=0.8)
        ax_32.axvline(sim_mean_32, color='red', linestyle='--', alpha=0.8)
        stats_text32 = f'Exp: μ={exp_mean_32:.1f}, σ={np.std(exp_delta_t32):.1f}\nSim: μ={sim_mean_32:.1f}, σ={np.std(sim_delta_t32):.1f}\nEMD={emd_32:.2f}, p={p_value_32:.3f}\nN={num_simulations}'
        ax_32.text(0.02, 0.98, stats_text32, transform=ax_32.transAxes, 
                   verticalalignment='top', fontsize=7,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Adjusted for title space
    
    # Save plot
    filename = f'simulation_fit_{mechanism}.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight', format='pdf')
    filename = f'simulation_fit_{mechanism}.svg'
    plt.savefig(filename, dpi=300, bbox_inches='tight', format='svg')
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
    fig.suptitle(f'{dataset_name.title()}', fontsize=14)  # Only dataset name in main title
    
    # Get experimental data
    exp_delta_t12 = experimental_data[dataset_name]['delta_t12']
    exp_delta_t32 = experimental_data[dataset_name]['delta_t32']
    
    alpha = params['alpha']
    beta_k = params.get('beta_k', None)
    beta_k1 = params.get('beta_k1', None)
    beta_k2 = params.get('beta_k2', None)
    beta_k3 = params.get('beta_k3', None)
    beta_tau = params.get('beta_tau', None)  # Default to None if not present
    beta_tau2 = params.get('beta_tau2', None)  # Default to None if not present
    
    # Run simulations
    print(f"Running {num_simulations} simulations for {dataset_name}...")
    sim_delta_t12, sim_delta_t32 = run_simulation_with_params(
        mechanism, params, dataset_name, alpha, beta_k, beta_tau, beta_tau2, beta_k1, beta_k2, beta_k3, num_simulations
    )
    
    if not sim_delta_t12 or not sim_delta_t32:
        print(f"Failed to generate simulation data for {dataset_name}")
        return
    
    # Save raw data to CSV
    max_len = max(len(exp_delta_t12), len(exp_delta_t32), len(sim_delta_t12), len(sim_delta_t32))
    csv_df = pd.DataFrame({
        'exp_delta_t12': pd.Series(exp_delta_t12).reindex(range(max_len)),
        'exp_delta_t32': pd.Series(exp_delta_t32).reindex(range(max_len)),
        'sim_delta_t12': pd.Series(sim_delta_t12).reindex(range(max_len)),
        'sim_delta_t32': pd.Series(sim_delta_t32).reindex(range(max_len))
    })
    csv_filename = f'simulation_data_{mechanism}_{dataset_name}.csv'
    csv_df.to_csv(csv_filename, index=False)
    print(f"Saved simulation data to {csv_filename}")
    
    # Set up x_grid for PDF plotting
    x_min, x_max = -140, 140  # Default range for all mechanisms
    if dataset == "velcade":
        x_min, x_max = -350, 350
    if dataset in ["wildtype", "threshold"]:
        x_min, x_max = -100, 100    
    x_grid = np.linspace(x_min, x_max, 401)
    
    # Calculate bins for 7 second intervals
    bin_width = 7.0
    all_data_12 = np.concatenate([exp_delta_t12, sim_delta_t12])
    bins_12 = np.arange(np.floor(all_data_12.min() / bin_width) * bin_width, 
                        np.ceil(all_data_12.max() / bin_width) * bin_width + bin_width, 
                        bin_width)
    
    # Plot Chrom1 - Chrom2
    ax1.hist(exp_delta_t12, bins=bins_12, density=True, alpha=0.85, label='Experimental data', color='lightgrey')
    ax1.hist(sim_delta_t12, bins=bins_12, density=True, alpha=0.85, label='Simulated data', color='lightskyblue')
    ax1.set_xlim(x_min - 20, x_max + 20)
    ax1.set_xlabel(r'$t_{ChrI} - t_{ChrII}$ (sec)')
    ax1.set_ylabel('Percentage of cells')
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y*100:.1f}'))
    ax1.set_title('chromosome I vs. II')  # No subtitle - dataset name is in main title
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Calculate EMD and Wasserstein p-value for T1-T2
    p_value_12, emd_12 = calculate_wasserstein_p_value(exp_delta_t12, sim_delta_t12, num_permutations=1000)
    
    # Add statistics for Chrom1 - Chrom2 with EMD and p-value
    stats_text12 = f'Exp: μ={np.mean(exp_delta_t12):.1f}, σ={np.std(exp_delta_t12):.1f}\nSim: μ={np.mean(sim_delta_t12):.1f}, σ={np.std(sim_delta_t12):.1f}\nEMD={emd_12:.2f}, p={p_value_12:.3f}\nN={num_simulations}'
    ax1.text(0.02, 0.98, stats_text12, transform=ax1.transAxes,
             verticalalignment='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Calculate bins for 3.5 second intervals for Chrom3 - Chrom2
    all_data_32 = np.concatenate([exp_delta_t32, sim_delta_t32])
    bins_32 = np.arange(np.floor(all_data_32.min() / bin_width) * bin_width, 
                        np.ceil(all_data_32.max() / bin_width) * bin_width + bin_width, 
                        bin_width)
    
    # Plot Chrom3 - Chrom2
    ax2.hist(exp_delta_t32, bins=bins_32, density=True, alpha=0.85, label='Experimental data', color='lightgrey')
    ax2.hist(sim_delta_t32, bins=bins_32, density=True, alpha=0.85, label='Simulated data', color='lightskyblue')
    ax2.set_xlim(x_min - 20, x_max + 20)
    ax2.set_xlabel(r'$t_{ChrI} - t_{ChrII}$ (sec)')
    ax2.set_ylabel('Percentage of cells')
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y*100:.1f}'))
    ax2.set_title('chromosome II vs. III')  # No subtitle
    ax2.legend()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    
    # Calculate EMD and Wasserstein p-value for T3-T2
    p_value_32, emd_32 = calculate_wasserstein_p_value(exp_delta_t32, sim_delta_t32, num_permutations=1000)
    
    # Add statistics for Chrom3 - Chrom2 with EMD and p-value
    stats_text32 = f'Exp: μ={np.mean(exp_delta_t32):.1f}, σ={np.std(exp_delta_t32):.1f}\nSim: μ={np.mean(sim_delta_t32):.1f}, σ={np.std(sim_delta_t32):.1f}\nEMD={emd_32:.2f}, p={p_value_32:.3f}\nN={num_simulations}'
    ax2.text(0.02, 0.98, stats_text32, transform=ax2.transAxes,
             verticalalignment='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    filename = f'simulation_fit_{mechanism}_{dataset_name}.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight', format='pdf')
    filename = f'simulation_fit_{mechanism}_{dataset_name}.svg'
    plt.savefig(filename, dpi=300, bbox_inches='tight', format='svg')
    
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
    
    # Handle both tau and k_1 parameter display
    if 'tau' in params:
        k_1_calc = params['k_max'] / params['tau']
        print(f"  Rates: k_max={params['k_max']:.4f}, tau={params['tau']:.4f} (k_1={k_1_calc:.6f})")
    elif 'k_1' in params:
        print(f"  Rates: k_1={params['k_1']:.6f}, k_max={params['k_max']:.4f}")
    elif 'k_max' in params:
        print(f"  Rates: k_max={params['k_max']:.4f}")
    elif 'k' in params:
        print(f"  Rates: k={params['k']:.4f}")
    
    if 'burst_size' in params:
        print(f"  Burst size: {params['burst_size']:.1f}")
    if 'n_inner' in params:
        print(f"  Inner threshold: {params['n_inner']:.1f}")
    if 'burst_size' in params and 'n_inner' in params:
        print(f"  Combined mechanism: burst_size={params['burst_size']:.1f}, n_inner={params['n_inner']:.1f}")
    elif 'burst_size' in params and 'n_inner' not in params:
        print(f"  Burst mechanism: burst_size={params['burst_size']:.1f}")
    
    
    print(f"\nMutant Modifiers:")
    print(f"  Threshold mutant (alpha): {params['alpha']:.3f}")
    if 'beta_k' in params:
        print(f"  Separase mutant (beta_k): {params['beta_k']:.3f}")
    if 'beta_k1' in params:
        print(f"  Separase mutant (beta_k1): {params['beta_k1']:.3f}")
    if 'beta_k2' in params:
        print(f"  APC mutant (beta_k2): {params['beta_k2']:.3f}")
    if 'beta_k3' in params:
        print(f"  Velcade mutant (beta_k3): {params['beta_k3']:.3f}")
    if 'beta_tau' in params:
        print(f"  APC mutant (beta_tau): {params['beta_tau']:.3f}")
    if 'beta_tau2' in params:
        print(f"  Velcade mutant (beta_tau2): {params['beta_tau2']:.3f}")
    
    # Show effective parameters for each mutant
    print(f"\nEffective Parameters by Mutant:")
    print(f"  Threshold: n1={params['alpha']*params['n1']:.1f}, n2={params['alpha']*params['n2']:.1f}, n3={params['alpha']*params['n3']:.1f}")
    
    # Handle both k and k_max for Separase mutant
    if 'k_max' in params and 'beta_k' in params:
        print(f"  Separase: k_max={params['beta_k']*params['k_max']:.4f}")
    elif 'k' in params and 'beta_k1' in params:
        print(f"  Separase: k={params['beta_k1']*params['k']:.4f}")
    
    # Calculate effective k_1 or tau for APC mutant (only if beta_tau exists)
    if 'beta_tau' in params:
        beta_tau_value = params['beta_tau']
        if 'tau' in params:
            effective_tau = beta_tau_value * params['tau']
            effective_k1 = params['k_max'] / effective_tau
            print(f"  APC: tau={effective_tau:.4f} (k_1={effective_k1:.6f})")
        elif 'k_1' in params:
            effective_k1 = beta_tau_value * params['k_1']
            print(f"  APC: k_1={effective_k1:.6f}")
        else:
            print(f"  APC: modifier={beta_tau_value:.3f}")
    elif 'beta_k2' in params and 'k' in params:
        # For simple mechanisms, beta_k2 affects k directly
        effective_k = params['beta_k2'] * params['k']
        print(f"  APC: k={effective_k:.4f}")
    
    # Calculate effective k_1 or tau for Velcade mutant if beta_tau2 exists
    if 'beta_tau2' in params:
        beta_tau2_value = params['beta_tau2']
        if 'tau' in params:
            effective_tau_vel = beta_tau2_value * params['tau']
            effective_k1_vel = params['k_max'] / effective_tau_vel
            print(f"  Velcade: tau={effective_tau_vel:.4f} (k_1={effective_k1_vel:.6f})")
        elif 'k_1' in params:
            effective_k1_vel = beta_tau2_value * params['k_1']
            print(f"  Velcade: k_1={effective_k1_vel:.6f}")
        else:
            print(f"  Velcade: modifier={beta_tau2_value:.3f}")
    elif 'beta_k3' in params and 'k' in params:
        # For simple mechanisms, beta_k3 affects k directly
        effective_k_vel = params['beta_k3'] * params['k']
        print(f"  Velcade: k={effective_k_vel:.4f}")


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
    mechanism = 'simple'  # Choose mechanism to test
    filename = 'simulation_optimized_parameters_simple.txt'
    dataset = 'velcade'  # Choose: 'wildtype', 'threshold', 'degrade', 'degradeAPC', 'velcade'
    
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