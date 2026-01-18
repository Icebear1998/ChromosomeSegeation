#!/usr/bin/env python3
"""
Demonstration script for Kernel Density Estimation (KDE).
Runs simulations with optimized parameters and visualizes the KDE fit.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
import pandas as pd
from MultiMechanismSimulationTimevary import MultiMechanismSimulationTimevary
from simulation_utils import apply_mutant_params, load_parameters
import warnings
warnings.filterwarnings('ignore')




def run_simulations(mechanism, params, mutant_type, num_simulations=1000):
    """Run simulations and return time differences."""
    # Create base parameters
    base_params = {
        'n1': params['n1'], 'n2': params['n2'], 'n3': params['n3'],
        'N1': params['N1'], 'N2': params['N2'], 'N3': params['N3'],
        'k_max': params['k_max']
    }
    
    if 'tau' in params:
        base_params['tau'] = params['tau']
    if 'k_1' in params:
        base_params['k_1'] = params['k_1']
    
    # Add mechanism-specific parameters
    if mechanism == 'time_varying_k_combined':
        base_params['burst_size'] = params['burst_size']
        base_params['n_inner'] = params['n_inner']
    
    # Apply mutant modifications
    alpha = params.get('alpha', 1.0)
    beta_k = params.get('beta_k', 1.0)
    beta_tau = params.get('beta_tau', 1.0)
    beta_tau2 = params.get('beta_tau2', 1.0)
    
    mutant_params, n0_list = apply_mutant_params(
        base_params, mutant_type, alpha, beta_k, beta_tau, beta_tau2
    )
    
    # Extract rate parameters for simulation
    initial_state = [mutant_params['N1'], mutant_params['N2'], mutant_params['N3']]
    rate_params = {
        'k_1': mutant_params['k_1'],
        'k_max': mutant_params['k_max'],
        'burst_size': mutant_params['burst_size'],
        'n_inner': mutant_params['n_inner']
    }
    
    # Run simulations
    delta_t12_list = []
    delta_t32_list = []
    
    print(f"Running {num_simulations} simulations for {mutant_type}...")
    for i in range(num_simulations):
        if (i + 1) % 200 == 0:
            print(f"  Completed {i + 1}/{num_simulations} simulations...")
        
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
    
    return np.array(delta_t12_list), np.array(delta_t32_list)


def load_experimental_data(mutant_type='wildtype'):
    """Load experimental data for a specific mutant type."""
    try:
        file_path = "Data/All_strains_SCStimes.xlsx"
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        
        dataset_mapping = {
            'wildtype': ('wildtype12', 'wildtype32'),
            'threshold': ('threshold12', 'threshold32'),
            'degrade': ('degRade12', 'degRade32'),
            'degradeAPC': ('degRadeAPC12', 'degRadeAPC32'),
            'velcade': ('degRadeVel12', 'degRadeVel32')
        }
        
        if mutant_type not in dataset_mapping:
            print(f"Warning: Unknown mutant type: {mutant_type}")
            return None, None
        
        col_12, col_32 = dataset_mapping[mutant_type]
        
        if col_12 in df.columns and col_32 in df.columns:
            delta_t12 = df[col_12].dropna().values
            delta_t32 = df[col_32].dropna().values
            return delta_t12, delta_t32
        else:
            print(f"Warning: Could not find columns for {mutant_type}")
            return None, None
    
    except Exception as e:
        print(f"Error loading experimental data: {e}")
        return None, None


def calculate_nll(pdf_func, exp_data):
    """Calculate negative log-likelihood of experimental data under a PDF."""
    try:
        # Evaluate PDF at experimental data points
        pdf_values = pdf_func(exp_data)
        
        # Avoid log(0) by adding small epsilon
        pdf_values = np.maximum(pdf_values, 1e-10)
        
        # Calculate negative log-likelihood
        nll = -np.sum(np.log(pdf_values))
        return nll
    except Exception as e:
        print(f"Error calculating NLL: {e}")
        return np.inf


def create_kde_plot(sim_data, exp_data, title="KDE Demonstration", save_filename=None):
    """Create a plot showing simulation data histogram, fitted KDE, and normal distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Process both delta_t12 and delta_t32
    datasets = [
        (sim_data[0], exp_data[0] if exp_data[0] is not None else None, 
         'Chrom1 - Chrom2 (T1-T2)', ax1),
        (sim_data[1], exp_data[1] if exp_data[1] is not None else None,
         'Chrom3 - Chrom2 (T3-T2)', ax2)
    ]
    
    for sim_d, exp_d, label, ax in datasets:
        # Remove any NaN or infinite values from simulation data
        sim_d = sim_d[np.isfinite(sim_d)]
        
        if len(sim_d) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Create histogram
        n_bins = 30
        counts, bins, patches = ax.hist(
            sim_d, bins=n_bins, density=True, alpha=0.6, 
            color='lightblue', edgecolor='black', label='Simulation Data'
        )
        
        # Fit KDE using scipy
        kde_scipy = gaussian_kde(sim_d)
        
        # Fit normal distribution
        mean_sim = np.mean(sim_d)
        std_sim = np.std(sim_d)
        normal_dist = norm(loc=mean_sim, scale=std_sim)
        
        # Create evaluation grid
        x_min, x_max = sim_d.min() - 20, sim_d.max() + 20
        x_grid = np.linspace(x_min, x_max, 500)
        
        # Evaluate KDE and normal distribution
        kde_scipy_values = kde_scipy(x_grid)
        normal_values = normal_dist.pdf(x_grid)
        
        # Calculate NLL against experimental data if available
        nll_kde = None
        nll_normal = None
        
        if exp_d is not None and len(exp_d) > 0:
            # Remove NaN/inf from experimental data
            exp_d_clean = exp_d[np.isfinite(exp_d)]
            if len(exp_d_clean) > 0:
                # NLL for KDE
                nll_kde = calculate_nll(kde_scipy, exp_d_clean)
                
                # NLL for normal distribution
                nll_normal = calculate_nll(normal_dist.pdf, exp_d_clean)
        
        # Plot KDE and normal curves
        kde_label = 'KDE (scipy)'
        normal_label = 'Normal fit'
        
        if nll_kde is not None:
            kde_label += f' (NLL: {nll_kde:.1f})'
        if nll_normal is not None:
            normal_label += f' (NLL: {nll_normal:.1f})'
        
        ax.plot(x_grid, kde_scipy_values, 'r-', linewidth=2, label=kde_label, alpha=0.8)
        ax.plot(x_grid, normal_values, 'g--', linewidth=2, label=normal_label, alpha=0.8)
        
        # Add statistics
        ax.axvline(mean_sim, color='orange', linestyle=':', linewidth=2, label=f'Mean: {mean_sim:.2f}')
        
        # Formatting
        ax.set_xlabel('Time Difference (seconds)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add text box with statistics
        stats_text = f'Mean: {mean_sim:.2f}\nStd: {std_sim:.2f}\nN: {len(sim_d)}'
        if exp_d is not None and len(exp_d) > 0:
            exp_d_clean = exp_d[np.isfinite(exp_d)]
            if len(exp_d_clean) > 0:
                stats_text += f'\nExp N: {len(exp_d_clean)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_filename:
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_filename}")
    
    plt.show()


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("KDE Demonstration Script")
    print("=" * 60)
    
    # Load parameters
    param_file = "simulation_optimized_parameters_R1_time_varying_k_combined.txt"
    print(f"\nLoading parameters from: {param_file}")
    params = load_parameters(param_file)
    
    print("\nLoaded parameters:")
    print(f"  n2={params['n2']:.2f}, N2={params['N2']:.2f}")
    print(f"  k_max={params['k_max']:.4f}, tau={params['tau']:.2f} min")
    print(f"  burst_size={params['burst_size']:.2f}, n_inner={params['n_inner']:.2f}")
    
    # Run simulations for wildtype
    mechanism = 'time_varying_k_combined'
    mutant_type = 'wildtype'
    num_simulations = 100
    
    print(f"\n{'=' * 60}")
    delta_t12, delta_t32 = run_simulations(
        mechanism, params, mutant_type, num_simulations
    )
    
    print(f"\nSimulation results:")
    print(f"  T1-T2: mean={np.mean(delta_t12):.2f}, std={np.std(delta_t12):.2f}")
    print(f"  T3-T2: mean={np.mean(delta_t32):.2f}, std={np.std(delta_t32):.2f}")
    
    # Load experimental data
    print(f"\n{'=' * 60}")
    print("Loading experimental data...")
    exp_delta_t12, exp_delta_t32 = load_experimental_data(mutant_type)
    
    if exp_delta_t12 is not None and exp_delta_t32 is not None:
        print(f"  Experimental T1-T2: {len(exp_delta_t12)} points")
        print(f"  Experimental T3-T2: {len(exp_delta_t32)} points")
    else:
        print("  Warning: Could not load experimental data")
    
    # Create KDE plot
    print(f"\n{'=' * 60}")
    print("Creating KDE visualization...")
    create_kde_plot(
        (delta_t12, delta_t32),
        (exp_delta_t12, exp_delta_t32),
        title=f"KDE vs Normal Fit: {mechanism} ({mutant_type})",
        save_filename="kde_demonstration.png"
    )
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

