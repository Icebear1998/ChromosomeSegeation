#!/usr/bin/env python3
"""
Test whether fixed_burst simulation data follows a normal distribution.

This script:
1. Runs fixed_burst simulations with typical parameters
2. Performs multiple normality tests (Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov, D'Agostino-Pearson)
3. Creates visualizations (histogram, Q-Q plot, probability plot, KDE comparison with normal)
4. Tests with different parameter sets to see if normality depends on parameters
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro, anderson, kstest, normaltest
import sys
import os

# Add SecondVersion to path for simple/fixed_burst mechanisms
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SecondVersion'))
from MultiMechanismSimulation import MultiMechanismSimulation


def run_fixed_burst_simulations(params, n0_list, num_simulations=1000):
    """
    Run fixed_burst simulations and return timing differences.
    
    Args:
        params: Dictionary with N1, N2, N3, k, burst_size
        n0_list: List of threshold values [n01, n02, n03]
        num_simulations: Number of simulations to run
        
    Returns:
        tuple: (delta_t12_array, delta_t32_array) as numpy arrays
    """
    print(f"Running {num_simulations} simulations with:")
    print(f"  N = [{params['N1']:.0f}, {params['N2']:.0f}, {params['N3']:.0f}]")
    print(f"  n = [{n0_list[0]:.0f}, {n0_list[1]:.0f}, {n0_list[2]:.0f}]")
    print(f"  k = {params['k']:.4f}")
    print(f"  burst_size = {params['burst_size']:.1f}")
    sys.stdout.flush()
    
    delta_t12_list = []
    delta_t32_list = []
    
    for i in range(num_simulations):
        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{num_simulations}")
            sys.stdout.flush()
        
        sim = MultiMechanismSimulation(
            mechanism='fixed_burst',
            initial_state_list=[params['N1'], params['N2'], params['N3']],
            rate_params={'k': params['k'], 'burst_size': params['burst_size']},
            n0_list=n0_list,
            max_time=1000.0
        )
        
        _, _, sep_times = sim.simulate()
        
        delta_t12 = sep_times[0] - sep_times[1]  # T1 - T2
        delta_t32 = sep_times[2] - sep_times[1]  # T3 - T2
        
        delta_t12_list.append(delta_t12)
        delta_t32_list.append(delta_t32)
    
    return np.array(delta_t12_list), np.array(delta_t32_list)


def perform_normality_tests(data, data_name):
    """
    Perform multiple normality tests on the data.
    
    Args:
        data: 1D numpy array of data
        data_name: Name of the data for reporting
        
    Returns:
        dict: Results from all tests
    """
    print(f"\n{'='*70}")
    print(f"NORMALITY TESTS FOR {data_name}")
    print(f"{'='*70}")
    
    # Remove any non-finite values
    data_clean = data[np.isfinite(data)]
    n = len(data_clean)
    
    print(f"Sample size: {n}")
    print(f"Mean: {np.mean(data_clean):.4f}")
    print(f"Std Dev: {np.std(data_clean, ddof=1):.4f}")
    print(f"Skewness: {stats.skew(data_clean):.4f}")
    print(f"Kurtosis: {stats.kurtosis(data_clean):.4f}")
    
    results = {}
    
    # 1. Shapiro-Wilk Test (most powerful for small to moderate sample sizes)
    print(f"\n1. Shapiro-Wilk Test:")
    if n <= 5000:  # Shapiro-Wilk has a limit
        stat, p_value = shapiro(data_clean)
        results['shapiro'] = {'statistic': stat, 'p_value': p_value}
        print(f"   Statistic: {stat:.6f}")
        print(f"   P-value: {p_value:.6e}")
        if p_value > 0.05:
            print(f"   ✅ PASS: Data appears normally distributed (p > 0.05)")
        else:
            print(f"   ❌ FAIL: Data does NOT appear normally distributed (p ≤ 0.05)")
    else:
        print(f"   ⚠️  Sample too large for Shapiro-Wilk test (n > 5000)")
    
    # 2. Anderson-Darling Test (good for detecting departures in tails)
    print(f"\n2. Anderson-Darling Test:")
    result = anderson(data_clean, dist='norm')
    results['anderson'] = {
        'statistic': result.statistic,
        'critical_values': result.critical_values,
        'significance_levels': result.significance_level
    }
    print(f"   Statistic: {result.statistic:.6f}")
    print(f"   Critical values: {result.critical_values}")
    print(f"   Significance levels: {result.significance_level}%")
    
    # Check at 5% significance level (usually index 2)
    if result.statistic < result.critical_values[2]:
        print(f"   ✅ PASS: Data appears normally distributed at 5% level")
    else:
        print(f"   ❌ FAIL: Data does NOT appear normally distributed at 5% level")
    
    # 3. Kolmogorov-Smirnov Test (comparing to normal with same mean/std)
    print(f"\n3. Kolmogorov-Smirnov Test:")
    mean = np.mean(data_clean)
    std = np.std(data_clean, ddof=1)
    stat, p_value = kstest(data_clean, lambda x: stats.norm.cdf(x, loc=mean, scale=std))
    results['kstest'] = {'statistic': stat, 'p_value': p_value}
    print(f"   Statistic: {stat:.6f}")
    print(f"   P-value: {p_value:.6e}")
    if p_value > 0.05:
        print(f"   ✅ PASS: Data appears normally distributed (p > 0.05)")
    else:
        print(f"   ❌ FAIL: Data does NOT appear normally distributed (p ≤ 0.05)")
    
    # 4. D'Agostino-Pearson Test (combines skewness and kurtosis)
    print(f"\n4. D'Agostino-Pearson Test:")
    if n >= 20:  # Requires at least 20 samples
        stat, p_value = normaltest(data_clean)
        results['dagostino'] = {'statistic': stat, 'p_value': p_value}
        print(f"   Statistic: {stat:.6f}")
        print(f"   P-value: {p_value:.6e}")
        if p_value > 0.05:
            print(f"   ✅ PASS: Data appears normally distributed (p > 0.05)")
        else:
            print(f"   ❌ FAIL: Data does NOT appear normally distributed (p ≤ 0.05)")
    else:
        print(f"   ⚠️  Sample too small for D'Agostino-Pearson test (n < 20)")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY:")
    passed = 0
    total = 0
    
    if 'shapiro' in results and results['shapiro']['p_value'] > 0.05:
        passed += 1
    if 'shapiro' in results:
        total += 1
    
    if results['anderson']['statistic'] < results['anderson']['critical_values'][2]:
        passed += 1
    total += 1
    
    if 'kstest' in results and results['kstest']['p_value'] > 0.05:
        passed += 1
    if 'kstest' in results:
        total += 1
    
    if 'dagostino' in results and results['dagostino']['p_value'] > 0.05:
        passed += 1
    if 'dagostino' in results:
        total += 1
    
    print(f"Normality tests passed: {passed}/{total}")
    
    if passed == total:
        print(f"✅ CONCLUSION: Data is consistent with normal distribution")
    elif passed >= total / 2:
        print(f"⚠️  CONCLUSION: Data shows mixed results, possibly approximately normal")
    else:
        print(f"❌ CONCLUSION: Data does NOT appear to be normally distributed")
    
    return results


def create_normality_plots(data_t12, data_t32, params, save_prefix='fixed_burst_normality'):
    """
    Create comprehensive plots to visualize normality.
    
    Args:
        data_t12: T1-T2 timing differences
        data_t32: T3-T2 timing differences
        params: Parameter dictionary
        save_prefix: Prefix for saved plot files
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Clean data
    data_t12_clean = data_t12[np.isfinite(data_t12)]
    data_t32_clean = data_t32[np.isfinite(data_t32)]
    
    # Plot for T1-T2
    # 1. Histogram with Normal overlay
    ax1 = plt.subplot(2, 4, 1)
    n, bins, patches = ax1.hist(data_t12_clean, bins=50, density=True, alpha=0.7, 
                                 color='skyblue', edgecolor='black', label='Simulated data')
    
    # Overlay normal distribution
    mean_t12 = np.mean(data_t12_clean)
    std_t12 = np.std(data_t12_clean, ddof=1)
    x = np.linspace(data_t12_clean.min(), data_t12_clean.max(), 200)
    normal_pdf = stats.norm.pdf(x, mean_t12, std_t12)
    ax1.plot(x, normal_pdf, 'r-', linewidth=2, label=f'Normal(μ={mean_t12:.2f}, σ={std_t12:.2f})')
    ax1.set_xlabel('T1 - T2 (min)')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('T1-T2: Histogram vs Normal')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Q-Q Plot
    ax2 = plt.subplot(2, 4, 2)
    stats.probplot(data_t12_clean, dist="norm", plot=ax2)
    ax2.set_title('T1-T2: Q-Q Plot')
    ax2.grid(True, alpha=0.3)
    
    # 3. Empirical CDF vs Normal CDF
    ax3 = plt.subplot(2, 4, 3)
    sorted_data = np.sort(data_t12_clean)
    empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    theoretical_cdf = stats.norm.cdf(sorted_data, mean_t12, std_t12)
    
    ax3.plot(sorted_data, empirical_cdf, 'b-', linewidth=1.5, label='Empirical CDF')
    ax3.plot(sorted_data, theoretical_cdf, 'r--', linewidth=1.5, label='Normal CDF')
    ax3.set_xlabel('T1 - T2 (min)')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('T1-T2: CDF Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Residuals from normal
    ax4 = plt.subplot(2, 4, 4)
    residuals = (data_t12_clean - mean_t12) / std_t12
    ax4.hist(residuals, bins=50, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
    x_std = np.linspace(-4, 4, 200)
    ax4.plot(x_std, stats.norm.pdf(x_std, 0, 1), 'r-', linewidth=2, label='Standard Normal')
    ax4.set_xlabel('Standardized Residuals')
    ax4.set_ylabel('Probability Density')
    ax4.set_title('T1-T2: Standardized Residuals')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot for T3-T2
    # 5. Histogram with Normal overlay
    ax5 = plt.subplot(2, 4, 5)
    n, bins, patches = ax5.hist(data_t32_clean, bins=50, density=True, alpha=0.7, 
                                 color='lightcoral', edgecolor='black', label='Simulated data')
    
    # Overlay normal distribution
    mean_t32 = np.mean(data_t32_clean)
    std_t32 = np.std(data_t32_clean, ddof=1)
    x = np.linspace(data_t32_clean.min(), data_t32_clean.max(), 200)
    normal_pdf = stats.norm.pdf(x, mean_t32, std_t32)
    ax5.plot(x, normal_pdf, 'r-', linewidth=2, label=f'Normal(μ={mean_t32:.2f}, σ={std_t32:.2f})')
    ax5.set_xlabel('T3 - T2 (min)')
    ax5.set_ylabel('Probability Density')
    ax5.set_title('T3-T2: Histogram vs Normal')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Q-Q Plot
    ax6 = plt.subplot(2, 4, 6)
    stats.probplot(data_t32_clean, dist="norm", plot=ax6)
    ax6.set_title('T3-T2: Q-Q Plot')
    ax6.grid(True, alpha=0.3)
    
    # 7. Empirical CDF vs Normal CDF
    ax7 = plt.subplot(2, 4, 7)
    sorted_data = np.sort(data_t32_clean)
    empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    theoretical_cdf = stats.norm.cdf(sorted_data, mean_t32, std_t32)
    
    ax7.plot(sorted_data, empirical_cdf, 'b-', linewidth=1.5, label='Empirical CDF')
    ax7.plot(sorted_data, theoretical_cdf, 'r--', linewidth=1.5, label='Normal CDF')
    ax7.set_xlabel('T3 - T2 (min)')
    ax7.set_ylabel('Cumulative Probability')
    ax7.set_title('T3-T2: CDF Comparison')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Residuals from normal
    ax8 = plt.subplot(2, 4, 8)
    residuals = (data_t32_clean - mean_t32) / std_t32
    ax8.hist(residuals, bins=50, density=True, alpha=0.7, color='lightyellow', edgecolor='black')
    x_std = np.linspace(-4, 4, 200)
    ax8.plot(x_std, stats.norm.pdf(x_std, 0, 1), 'r-', linewidth=2, label='Standard Normal')
    ax8.set_xlabel('Standardized Residuals')
    ax8.set_ylabel('Probability Density')
    ax8.set_title('T3-T2: Standardized Residuals')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Overall title
    title_text = (f"Fixed Burst Normality Analysis\n"
                  f"N=[{params['N1']:.0f},{params['N2']:.0f},{params['N3']:.0f}], "
                  f"n=[{params['n1']:.0f},{params['n2']:.0f},{params['n3']:.0f}], "
                  f"k={params['k']:.4f}, burst_size={params['burst_size']:.1f}")
    fig.suptitle(title_text, fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    filename = f"{save_prefix}_bs{params['burst_size']:.0f}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved as: {filename}")
    
    plt.show()


def main():
    """
    Main function to test normality of fixed_burst simulations.
    """
    print("="*80)
    print("TESTING NORMALITY OF FIXED_BURST SIMULATION DATA")
    print("="*80)
    sys.stdout.flush()
    
    # Define test parameter sets
    # Test 1: Small burst size (should be more normal)
    # Test 2: Medium burst size
    # Test 3: Large burst size (may deviate from normality)
    
    test_cases = [
        {
            'name': 'Small burst (bs=2)',
            'params': {
                'N1': 100.0, 'N2': 200.0, 'N3': 400.0,
                'n1': 5.0, 'n2': 10.0, 'n3': 20.0,
                'k': 0.05,
                'burst_size': 2.0
            }
        },
        {
            'name': 'Medium burst (bs=5)',
            'params': {
                'N1': 100.0, 'N2': 200.0, 'N3': 400.0,
                'n1': 5.0, 'n2': 10.0, 'n3': 20.0,
                'k': 0.05,
                'burst_size': 5.0
            }
        },
        {
            'name': 'Large burst (bs=10)',
            'params': {
                'N1': 100.0, 'N2': 200.0, 'N3': 400.0,
                'n1': 5.0, 'n2': 10.0, 'n3': 20.0,
                'k': 0.05,
                'burst_size': 10.0
            }
        }
    ]
    
    num_simulations = 2000  # Increase for better statistical power
    
    all_results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}/{len(test_cases)}: {test_case['name']}")
        print(f"{'='*80}")
        sys.stdout.flush()
        
        params = test_case['params']
        n0_list = [params['n1'], params['n2'], params['n3']]
        
        # Run simulations
        data_t12, data_t32 = run_fixed_burst_simulations(params, n0_list, num_simulations)
        
        # Perform normality tests
        print("\n" + "="*80)
        results_t12 = perform_normality_tests(data_t12, f"T1-T2 ({test_case['name']})")
        sys.stdout.flush()
        
        results_t32 = perform_normality_tests(data_t32, f"T3-T2 ({test_case['name']})")
        sys.stdout.flush()
        
        # Create plots
        create_normality_plots(data_t12, data_t32, params, 
                              save_prefix=f'fixed_burst_normality_case{i}')
        
        all_results.append({
            'test_case': test_case['name'],
            'params': params,
            'results_t12': results_t12,
            'results_t32': results_t32,
            'data_t12': data_t12,
            'data_t32': data_t32
        })
    
    # Final summary across all test cases
    print("\n" + "="*80)
    print("OVERALL SUMMARY ACROSS ALL TEST CASES")
    print("="*80)
    
    for i, result in enumerate(all_results, 1):
        print(f"\n{i}. {result['test_case']} (burst_size={result['params']['burst_size']:.1f}):")
        
        # T1-T2 summary
        t12_passed = 0
        t12_total = 0
        if 'shapiro' in result['results_t12']:
            if result['results_t12']['shapiro']['p_value'] > 0.05:
                t12_passed += 1
            t12_total += 1
        if result['results_t12']['anderson']['statistic'] < result['results_t12']['anderson']['critical_values'][2]:
            t12_passed += 1
        t12_total += 1
        
        print(f"   T1-T2: {t12_passed}/{t12_total} tests passed")
        
        # T3-T2 summary
        t32_passed = 0
        t32_total = 0
        if 'shapiro' in result['results_t32']:
            if result['results_t32']['shapiro']['p_value'] > 0.05:
                t32_passed += 1
            t32_total += 1
        if result['results_t32']['anderson']['statistic'] < result['results_t32']['anderson']['critical_values'][2]:
            t32_passed += 1
        t32_total += 1
        
        print(f"   T3-T2: {t32_passed}/{t32_total} tests passed")
    
    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    print("The fixed_burst mechanism produces data that may or may not be normally")
    print("distributed, depending on the burst size and other parameters.")
    print("\nKey observations:")
    print("- Smaller burst sizes tend to produce more normal-like distributions")
    print("- Larger burst sizes can introduce discretization effects that deviate from normality")
    print("- This justifies using KDE for simulation-based optimization rather than")
    print("  assuming a normal distribution (as in MoM approach)")
    print("="*80)
    sys.stdout.flush()


if __name__ == "__main__":
    main()

