#!/usr/bin/env python3
"""
Test boundary conditions: Compare simple vs fixed_burst mechanisms.

Tests whether:
1. simple ≈ fixed_burst (burst_size=1) for both MoM and KDE
2. How NLL changes as burst_size increases
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'SecondVersion'))

from MoMOptimization_join import joint_objective, get_mechanism_info
from simulation_utils import load_experimental_data
from SimulationOptimization_join import joint_objective_simple_mechanisms, run_simple_simulation_for_dataset
from simulation_kde import build_kde_from_simulations, evaluate_kde_pdf
import matplotlib.pyplot as plt
from scipy import stats

# Shared parameters for all tests
SHARED_PARAMS = {
    'n2': 1.5,
    'N2': 377.03251217126376,
    'k': 0.05306211492180258,
    'r21': 0.6169229301033301,
    'r23': 2.9056141068546766,
    'R21': 0.503170207083632,
    'R23': 3.9174717230831364,
    'alpha': 0.48695158598867494,
    'beta_k': 0.44862061532702135,
    'beta2_k': 0.4992748045927576,  # Only used in MoM
    'beta3_k': 0.24058909365510822  # Only used in MoM
}

def calculate_nll_mom(mechanism, params_vector, data_arrays):
    """Helper to calculate MoM NLL."""
    mechanism_info = get_mechanism_info(mechanism, 'separate')
    return joint_objective(
        params_vector,
        mechanism,
        mechanism_info,
        data_arrays['data_wt12'], data_arrays['data_wt32'],
        data_arrays['data_threshold12'], data_arrays['data_threshold32'],
        data_arrays['data_degrate12'], data_arrays['data_degrate32'],
        data_arrays['data_initial12'], data_arrays['data_initial32'],
        data_arrays['data_degrateAPC12'], data_arrays['data_degrateAPC32'],
        data_arrays['data_velcade12'], data_arrays['data_velcade32']
    )


def plot_mom_vs_kde_comparison(datasets):
    """
    Plot MoM PDF vs Simulation KDE to visualize the difference.
    """
    print("\nGenerating comparison plots...")
    
    # Prepare parameters
    base_params = {
        'n1': SHARED_PARAMS['r21'] * SHARED_PARAMS['n2'],
        'n2': SHARED_PARAMS['n2'],
        'n3': SHARED_PARAMS['r23'] * SHARED_PARAMS['n2'],
        'N1': SHARED_PARAMS['R21'] * SHARED_PARAMS['N2'],
        'N2': SHARED_PARAMS['N2'],
        'N3': SHARED_PARAMS['R23'] * SHARED_PARAMS['N2'],
        'k': SHARED_PARAMS['k']
    }
    n0_list = [base_params['n1'], base_params['n2'], base_params['n3']]
    
    # Run simulations for wildtype
    print("Running simulations for wildtype...")
    sim_delta_t12, sim_delta_t32 = run_simple_simulation_for_dataset(
        'simple', base_params, n0_list, num_simulations=1000
    )
    
    # Get experimental data
    exp_delta_t12 = datasets['wildtype']['delta_t12']
    exp_delta_t32 = datasets['wildtype']['delta_t32']
    
    # Calculate MoM statistics (mean and std)
    sys.path.append(os.path.join(os.path.dirname(__file__), 'SecondVersion'))
    from MoMCalculations import compute_moments_mom
    
    # For T1-T2
    mean_t12_mom, var_t12_mom = compute_moments_mom(
        'simple',
        int(base_params['n1']), int(base_params['N1']),
        int(base_params['n2']), int(base_params['N2']),
        base_params['k']
    )
    std_t12_mom = np.sqrt(var_t12_mom)
    
    # For T3-T2
    mean_t32_mom, var_t32_mom = compute_moments_mom(
        'simple',
        int(base_params['n3']), int(base_params['N3']),
        int(base_params['n2']), int(base_params['N2']),
        base_params['k']
    )
    std_t32_mom = np.sqrt(var_t32_mom)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MoM (Normal) vs Simulation KDE Comparison - Wildtype', fontsize=14)
    
    # T1-T2 comparison
    ax1 = axes[0, 0]
    
    # Experimental histogram
    ax1.hist(exp_delta_t12, bins=30, density=True, alpha=0.3, color='gray', label='Experimental data')
    
    # Simulation KDE
    kde_t12 = build_kde_from_simulations(sim_delta_t12)
    x_grid = np.linspace(min(exp_delta_t12.min(), sim_delta_t12.min()),
                         max(exp_delta_t12.max(), sim_delta_t12.max()), 200)
    kde_pdf_t12 = evaluate_kde_pdf(kde_t12, x_grid)
    ax1.plot(x_grid, kde_pdf_t12, 'b-', linewidth=2, label='Simulation KDE')
    
    # MoM normal PDF
    mom_pdf_t12 = stats.norm.pdf(x_grid, mean_t12_mom, std_t12_mom)
    ax1.plot(x_grid, mom_pdf_t12, 'r--', linewidth=2, label='MoM (Normal)')
    
    ax1.axvline(exp_delta_t12.mean(), color='gray', linestyle=':', alpha=0.7, label=f'Exp mean: {exp_delta_t12.mean():.1f}')
    ax1.axvline(sim_delta_t12.mean(), color='blue', linestyle=':', alpha=0.7, label=f'Sim mean: {sim_delta_t12.mean():.1f}')
    ax1.axvline(mean_t12_mom, color='red', linestyle=':', alpha=0.7, label=f'MoM mean: {mean_t12_mom:.1f}')
    
    ax1.set_xlabel('T1 - T2 (min)')
    ax1.set_ylabel('Density')
    ax1.set_title('T1 - T2 Distribution')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # T3-T2 comparison
    ax2 = axes[0, 1]
    
    # Experimental histogram
    ax2.hist(exp_delta_t32, bins=30, density=True, alpha=0.3, color='gray', label='Experimental data')
    
    # Simulation KDE
    kde_t32 = build_kde_from_simulations(sim_delta_t32)
    x_grid = np.linspace(min(exp_delta_t32.min(), sim_delta_t32.min()),
                         max(exp_delta_t32.max(), sim_delta_t32.max()), 200)
    kde_pdf_t32 = evaluate_kde_pdf(kde_t32, x_grid)
    ax2.plot(x_grid, kde_pdf_t32, 'b-', linewidth=2, label='Simulation KDE')
    
    # MoM normal PDF
    mom_pdf_t32 = stats.norm.pdf(x_grid, mean_t32_mom, std_t32_mom)
    ax2.plot(x_grid, mom_pdf_t32, 'r--', linewidth=2, label='MoM (Normal)')
    
    ax2.axvline(exp_delta_t32.mean(), color='gray', linestyle=':', alpha=0.7, label=f'Exp mean: {exp_delta_t32.mean():.1f}')
    ax2.axvline(sim_delta_t32.mean(), color='blue', linestyle=':', alpha=0.7, label=f'Sim mean: {sim_delta_t32.mean():.1f}')
    ax2.axvline(mean_t32_mom, color='red', linestyle=':', alpha=0.7, label=f'MoM mean: {mean_t32_mom:.1f}')
    
    ax2.set_xlabel('T3 - T2 (min)')
    ax2.set_ylabel('Density')
    ax2.set_title('T3 - T2 Distribution')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Q-Q plots
    ax3 = axes[1, 0]
    stats.probplot(exp_delta_t12, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot: T1-T2 (Experimental vs Normal)')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    stats.probplot(exp_delta_t32, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot: T3-T2 (Experimental vs Normal)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = 'mom_vs_kde_comparison.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved as: {filename}")
    
    # Print statistics
    print("\n" + "="*80)
    print("DISTRIBUTION STATISTICS")
    print("="*80)
    
    print("\nT1 - T2:")
    print(f"  Experimental: mean={exp_delta_t12.mean():.2f}, std={exp_delta_t12.std():.2f}")
    print(f"  Simulation:   mean={sim_delta_t12.mean():.2f}, std={sim_delta_t12.std():.2f}")
    print(f"  MoM:          mean={mean_t12_mom:.2f}, std={std_t12_mom:.2f}")
    
    print("\nT3 - T2:")
    print(f"  Experimental: mean={exp_delta_t32.mean():.2f}, std={exp_delta_t32.std():.2f}")
    print(f"  Simulation:   mean={sim_delta_t32.mean():.2f}, std={sim_delta_t32.std():.2f}")
    print(f"  MoM:          mean={mean_t32_mom:.2f}, std={std_t32_mom:.2f}")
    
    # Test normality
    _, p_t12 = stats.shapiro(exp_delta_t12[:5000] if len(exp_delta_t12) > 5000 else exp_delta_t12)
    _, p_t32 = stats.shapiro(exp_delta_t32[:5000] if len(exp_delta_t32) > 5000 else exp_delta_t32)
    
    print("\nNormality test (Shapiro-Wilk):")
    print(f"  T1-T2: p-value = {p_t12:.4f} {'(Normal ✓)' if p_t12 > 0.05 else '(Non-normal ✗)'}")
    print(f"  T3-T2: p-value = {p_t32:.4f} {'(Normal ✓)' if p_t32 > 0.05 else '(Non-normal ✗)'}")
    
    print("\n💡 Key insight:")
    if p_t12 < 0.05 or p_t32 < 0.05:
        print("  The data is NOT normally distributed!")
        print("  → This explains why KDE (which captures the true distribution)")
        print("    gives different NLL than MoM (which assumes normality)")
    else:
        print("  The data appears normally distributed.")
        print("  → MoM and KDE should give similar results.")
    
    return fig
    

def test_burst_size_comparison():
    """
    Compare simple vs fixed_burst for different burst_size values (MoM and KDE).
    """
    print("="*80)
    print("BURST SIZE COMPARISON TEST")
    print("="*80)
    
    # Load data
    print("\nLoading experimental data...")
    datasets = load_experimental_data()
    
    data_arrays = {
        'data_wt12': datasets['wildtype']['delta_t12'],
        'data_wt32': datasets['wildtype']['delta_t32'],
        'data_threshold12': datasets['threshold']['delta_t12'],
        'data_threshold32': datasets['threshold']['delta_t32'],
        'data_degrate12': datasets['degrade']['delta_t12'],
        'data_degrate32': datasets['degrade']['delta_t32'],
        'data_degrateAPC12': datasets['degradeAPC']['delta_t12'],
        'data_degrateAPC32': datasets['degradeAPC']['delta_t32'],
        'data_velcade12': datasets['velcade']['delta_t12'],
        'data_velcade32': datasets['velcade']['delta_t32'],
        'data_initial12': np.array([]),
        'data_initial32': np.array([])
    }
    
    print("Data loaded ✓")
    print(f"Using parameters: n2={SHARED_PARAMS['n2']:.2f}, N2={SHARED_PARAMS['N2']:.1f}, k={SHARED_PARAMS['k']:.5f}")
    
    # MoM: Simple baseline
    print("\n" + "-"*80)
    print("MoM APPROACH")
    print("-"*80)
    
    simple_param_vector_mom = [
        SHARED_PARAMS['n2'], SHARED_PARAMS['N2'], SHARED_PARAMS['k'],
        SHARED_PARAMS['r21'], SHARED_PARAMS['r23'], SHARED_PARAMS['R21'], SHARED_PARAMS['R23'],
        SHARED_PARAMS['alpha'], SHARED_PARAMS['beta_k'], SHARED_PARAMS['beta2_k'], SHARED_PARAMS['beta3_k']
    ]
    
    print("Calculating simple model...")
    nll_simple_mom = calculate_nll_mom('simple', simple_param_vector_mom, data_arrays)
    print(f"Simple NLL: {nll_simple_mom:.2f}")
    
    # MoM: Fixed burst with varying burst_size
    print("\nTesting fixed_burst with different burst sizes...")
    burst_sizes = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
    
    print(f"\n{'burst_size':>11} | {'NLL':>10} | {'Diff':>10} | Better?")
    print("-"*50)
    
    mom_results = []
    for bs in burst_sizes:
        fb_params = [
            SHARED_PARAMS['n2'], SHARED_PARAMS['N2'], SHARED_PARAMS['k'],
            SHARED_PARAMS['r21'], SHARED_PARAMS['r23'], SHARED_PARAMS['R21'], SHARED_PARAMS['R23'],
            bs,  # burst_size
            SHARED_PARAMS['alpha'], SHARED_PARAMS['beta_k'], SHARED_PARAMS['beta2_k'], SHARED_PARAMS['beta3_k']
        ]
        
        nll_fb = calculate_nll_mom('fixed_burst', fb_params, data_arrays)
        diff = nll_fb - nll_simple_mom
        better = "✓" if nll_fb < nll_simple_mom else "✗"
        
        print(f"{bs:11.1f} | {nll_fb:10.2f} | {diff:+10.2f} | {better}")
        mom_results.append({'bs': bs, 'nll': nll_fb, 'diff': diff})
    
    # KDE: Simple and fixed_burst
    print("\n" + "-"*80)
    print("SIMULATION KDE APPROACH")
    print("-"*80)
    
    num_simulations = 1000
    print(f"Using {num_simulations} simulations per evaluation")
    
    simple_param_vector_kde = [
        SHARED_PARAMS['n2'], SHARED_PARAMS['N2'], SHARED_PARAMS['k'],
        SHARED_PARAMS['r21'], SHARED_PARAMS['r23'], SHARED_PARAMS['R21'], SHARED_PARAMS['R23'],
        SHARED_PARAMS['alpha'], SHARED_PARAMS['beta_k']
    ]
    
    print("\nCalculating simple model...")
    nll_simple_kde = joint_objective_simple_mechanisms(
        simple_param_vector_kde, 'simple', datasets, num_simulations=num_simulations
    )
    print(f"Simple NLL: {nll_simple_kde:.2f}")
    
    print("\nTesting fixed_burst with different burst sizes...")
    print(f"\n{'burst_size':>11} | {'NLL':>10} | {'Diff':>10} | Better?")
    print("-"*50)
    
    kde_results = []
    for bs in burst_sizes:
        fb_params = [
            SHARED_PARAMS['n2'], SHARED_PARAMS['N2'], SHARED_PARAMS['k'],
            SHARED_PARAMS['r21'], SHARED_PARAMS['r23'], SHARED_PARAMS['R21'], SHARED_PARAMS['R23'],
            bs,  # burst_size
            SHARED_PARAMS['alpha'], SHARED_PARAMS['beta_k']
        ]
        
        nll_fb = joint_objective_simple_mechanisms(
            fb_params, 'fixed_burst', datasets, num_simulations=num_simulations
        )
        diff = nll_fb - nll_simple_kde
        better = "✓" if nll_fb < nll_simple_kde else "✗"
        
        print(f"{bs:11.1f} | {nll_fb:10.2f} | {diff:+10.2f} | {better}")
        kde_results.append({'bs': bs, 'nll': nll_fb, 'diff': diff})
    
    return {
        'nll_simple_mom': nll_simple_mom,
        'nll_simple_kde': nll_simple_kde,
        'mom_results': mom_results,
        'kde_results': kde_results
    }


if __name__ == "__main__":
    # Load datasets first for plotting
    datasets = load_experimental_data()
    
    # Generate comparison plots to understand MoM vs KDE difference
    plot_mom_vs_kde_comparison(datasets)
    
    # Run the burst size comparison
    results = test_burst_size_comparison()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nSimple model NLL:")
    print(f"  MoM:            {results['nll_simple_mom']:.2f}")
    print(f"  Simulation KDE: {results['nll_simple_kde']:.2f}")
    print(f"  Difference:     {abs(results['nll_simple_mom'] - results['nll_simple_kde']):.2f}")
    
    print(f"\nBurst size = 1.0 (should ≈ simple):")
    mom_bs1 = results['mom_results'][0]
    kde_bs1 = results['kde_results'][0]
    print(f"  MoM fixed_burst:   {mom_bs1['nll']:.2f} (diff: {mom_bs1['diff']:+.2f})")
    print(f"  KDE fixed_burst:   {kde_bs1['nll']:.2f} (diff: {kde_bs1['diff']:+.2f})")
    
    # Check if burst_size=1 matches simple
    mom_match = abs(mom_bs1['diff']) < 0.01
    kde_match = abs(kde_bs1['diff']) < 10.0  # Looser tolerance for stochastic KDE
    
    if mom_match and kde_match:
        print("\n✅ PASS: burst_size=1 matches simple model in both approaches")
    elif mom_match:
        print("\n⚠️  MoM matches but KDE differs (may be stochastic noise)")
    else:
        print("\n❌ FAIL: burst_size=1 does not match simple model")
    
    # Find best burst_size for each approach
    best_mom = min(results['mom_results'], key=lambda x: x['nll'])
    best_kde = min(results['kde_results'], key=lambda x: x['nll'])
    
    print(f"\nBest burst_size:")
    print(f"  MoM: burst_size={best_mom['bs']:.1f}, NLL={best_mom['nll']:.2f} (Δ={best_mom['diff']:+.2f})")
    print(f"  KDE: burst_size={best_kde['bs']:.1f}, NLL={best_kde['nll']:.2f} (Δ={best_kde['diff']:+.2f})")
    
    if best_mom['nll'] < results['nll_simple_mom'] or best_kde['nll'] < results['nll_simple_kde']:
        print("\n💡 Fixed burst mechanism improves fit over simple model!")
    else:
        print("\n💡 Simple model performs as well as fixed burst mechanism.")
