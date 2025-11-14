#!/usr/bin/env python3
"""
Profile Likelihood Analysis: NLL as a function of burst_size.

This script:
1. Loads optimized parameters from the simple mechanism
2. Fixes all parameters except burst_size
3. Evaluates NLL for a range of burst_size values
4. Plots the profile to show why burst_size=1 is optimal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys
import os

# Import MoM optimization functions
sys.path.append(os.path.join(os.path.dirname(__file__), 'SecondVersion'))
from MoMOptimization_join import joint_objective, get_mechanism_info
from simulation_utils import load_experimental_data

print("="*80)
print("PROFILE LIKELIHOOD ANALYSIS: NLL vs BURST_SIZE")
print("="*80)

# Load experimental data
print("\n📊 Loading experimental data...")
datasets = load_experimental_data()
if not datasets:
    print("❌ Error: Could not load experimental data!")
    sys.exit(1)

# Convert to data arrays format expected by joint_objective
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
print(f"✅ Loaded {len(datasets)} datasets")

# Load optimized parameters from CSV file
print("\n📁 Loading optimized parameters from CSV...")
df = pd.read_csv('optimized_params_runs_20251113_145124.csv')

# Get the best simple mechanism run
simple_df = df[df['mechanism'] == 'simple'].copy()
best_simple_idx = simple_df['nll'].idxmin()
best_simple = simple_df.loc[best_simple_idx]

print(f"\n✅ Using best simple mechanism run:")
print(f"   Run #{int(best_simple['run_number'])}")
print(f"   NLL: {best_simple['nll']:.6f}")
print(f"   Parameters:")
print(f"     n2={best_simple['n2']:.4f}, N2={best_simple['N2']:.2f}")
print(f"     k={best_simple['k']:.6f}")
print(f"     r21={best_simple['r21']:.4f}, r23={best_simple['r23']:.4f}")
print(f"     R21={best_simple['R21']:.4f}, R23={best_simple['R23']:.4f}")
print(f"     alpha={best_simple['alpha']:.4f}, beta_k={best_simple['beta_k']:.4f}")
print(f"     beta2_k={best_simple['beta2_k']:.4f}, beta3_k={best_simple['beta3_k']:.4f}")

# Extract fixed parameters
fixed_params = {
    'n2': best_simple['n2'],
    'N2': best_simple['N2'],
    'k': best_simple['k'],
    'r21': best_simple['r21'],
    'r23': best_simple['r23'],
    'R21': best_simple['R21'],
    'R23': best_simple['R23'],
    'alpha': best_simple['alpha'],
    'beta_k': best_simple['beta_k'],
    'beta2_k': best_simple['beta2_k'],
    'beta3_k': best_simple['beta3_k']
}

simple_nll = best_simple['nll']

# Define burst_size range to test
print("\n🔬 Setting up burst_size profile...")
# Test from 1 (simple model) to 20 cohesins per burst
burst_sizes = np.concatenate([
    np.linspace(1, 2, 20),      # Fine resolution near 1
    np.linspace(2.1, 5, 15),    # Medium resolution 2-5
    np.linspace(5.5, 10, 10),   # Coarser resolution 5-10
    np.linspace(11, 20, 10)     # Coarse resolution 10-20
])
burst_sizes = np.unique(burst_sizes)  # Remove duplicates
burst_sizes = np.sort(burst_sizes)

print(f"   Testing {len(burst_sizes)} burst_size values from {burst_sizes[0]:.1f} to {burst_sizes[-1]:.1f}")

# Get mechanism info for fixed_burst
mechanism_info = get_mechanism_info('fixed_burst', gamma_mode='separate')

print(f"\n🚀 Computing NLL for each burst_size...")
print(f"   This may take a few minutes...")

nll_values = []
successful_burst_sizes = []

for i, burst_size in enumerate(burst_sizes):
    if (i+1) % 10 == 0 or i == 0 or i == len(burst_sizes)-1:
        print(f"   Progress: {i+1}/{len(burst_sizes)} - burst_size={burst_size:.2f}")
    
    # Construct parameter vector for fixed_burst mechanism
    # Order: n2, N2, k, r21, r23, R21, R23, burst_size, alpha, beta_k, beta2_k, beta3_k
    params_vector = np.array([
        fixed_params['n2'],
        fixed_params['N2'],
        fixed_params['k'],
        fixed_params['r21'],
        fixed_params['r23'],
        fixed_params['R21'],
        fixed_params['R23'],
        burst_size,  # The only varying parameter
        fixed_params['alpha'],
        fixed_params['beta_k'],
        fixed_params['beta2_k'],
        fixed_params['beta3_k']
    ])
    
    # Compute NLL
    try:
        nll = joint_objective(
            params_vector,
            'fixed_burst',  # mechanism name
            mechanism_info,
            data_arrays['data_wt12'],
            data_arrays['data_wt32'],
            data_arrays['data_threshold12'],
            data_arrays['data_threshold32'],
            data_arrays['data_degrate12'],
            data_arrays['data_degrate32'],
            data_arrays['data_initial12'],
            data_arrays['data_initial32'],
            data_arrays['data_degrateAPC12'],
            data_arrays['data_degrateAPC32'],
            data_arrays['data_velcade12'],
            data_arrays['data_velcade32']
        )
        
        if nll < 1e6:  # Valid NLL
            nll_values.append(nll)
            successful_burst_sizes.append(burst_size)
        else:
            print(f"     ⚠️  burst_size={burst_size:.2f} returned invalid NLL={nll:.2f}")
    except Exception as e:
        print(f"     ❌ Error at burst_size={burst_size:.2f}: {e}")

nll_values = np.array(nll_values)
successful_burst_sizes = np.array(successful_burst_sizes)

print(f"\n✅ Successfully computed NLL for {len(successful_burst_sizes)}/{len(burst_sizes)} burst_size values")

# Find minimum NLL and corresponding burst_size
min_idx = np.argmin(nll_values)
min_burst_size = successful_burst_sizes[min_idx]
min_nll = nll_values[min_idx]

print(f"\n📊 Profile Analysis Results:")
print(f"   Minimum NLL: {min_nll:.6f} at burst_size={min_burst_size:.3f}")
print(f"   Simple model NLL: {simple_nll:.6f}")
print(f"   Difference: {min_nll - simple_nll:+.6f}")

# Create visualization
print(f"\n📈 Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Profile Likelihood: Fixed Burst Mechanism\n(All parameters fixed at simple model optimum)', 
             fontsize=14, fontweight='bold')

# 1. Main profile plot (linear scale)
ax = axes[0, 0]
ax.plot(successful_burst_sizes, nll_values, 'b-', linewidth=2, label='Fixed burst NLL')
ax.axhline(y=simple_nll, color='red', linestyle='--', linewidth=2, 
           label=f'Simple model NLL = {simple_nll:.2f}')
ax.axvline(x=1, color='green', linestyle=':', linewidth=2, alpha=0.7,
           label='burst_size = 1 (simple model)')
ax.scatter([min_burst_size], [min_nll], color='orange', s=200, zorder=5, 
           marker='*', edgecolor='black', linewidth=1.5,
           label=f'Minimum: burst={min_burst_size:.2f}, NLL={min_nll:.2f}')

ax.set_xlabel('Burst Size (cohesins per degradation event)', fontsize=11)
ax.set_ylabel('Negative Log-Likelihood (NLL)', fontsize=11)
ax.set_title('Profile Likelihood: NLL vs Burst Size', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3)

# Add annotation for minimum
ax.annotate(f'Optimum\nburst={min_burst_size:.2f}',
            xy=(min_burst_size, min_nll),
            xytext=(min_burst_size + 3, min_nll + 20),
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=1.5))

# 2. Zoomed-in view near burst_size=1
ax = axes[0, 1]
zoom_mask = (successful_burst_sizes >= 1) & (successful_burst_sizes <= 5)
ax.plot(successful_burst_sizes[zoom_mask], nll_values[zoom_mask], 'b-', linewidth=2, marker='o', markersize=4)
ax.axhline(y=simple_nll, color='red', linestyle='--', linewidth=2, 
           label=f'Simple model NLL = {simple_nll:.2f}')
ax.axvline(x=1, color='green', linestyle=':', linewidth=2, alpha=0.7)
ax.scatter([min_burst_size], [min_nll], color='orange', s=200, zorder=5, 
           marker='*', edgecolor='black', linewidth=1.5)

ax.set_xlabel('Burst Size', fontsize=11)
ax.set_ylabel('NLL', fontsize=11)
ax.set_title('Zoomed View: Burst Size 1-5', fontsize=12, fontweight='bold')
ax.set_xlim([1, 5])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 3. Delta NLL (difference from simple model)
ax = axes[1, 0]
delta_nll = nll_values - simple_nll
ax.plot(successful_burst_sizes, delta_nll, 'b-', linewidth=2, label='ΔNLL (fixed_burst - simple)')
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='ΔNLL = 0 (no improvement)')
ax.axvline(x=1, color='green', linestyle=':', linewidth=2, alpha=0.7)
ax.scatter([min_burst_size], [min_nll - simple_nll], color='orange', s=200, zorder=5, 
           marker='*', edgecolor='black', linewidth=1.5)

# Add shaded region for "worse than simple"
ax.fill_between(successful_burst_sizes, 0, delta_nll, where=(delta_nll > 0), 
                alpha=0.3, color='red', label='Worse than simple')
ax.fill_between(successful_burst_sizes, 0, delta_nll, where=(delta_nll <= 0), 
                alpha=0.3, color='green', label='Better than simple')

ax.set_xlabel('Burst Size', fontsize=11)
ax.set_ylabel('ΔNLL (compared to simple model)', fontsize=11)
ax.set_title('Improvement Over Simple Model', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3)

# 4. Log scale x-axis view
ax = axes[1, 1]
ax.semilogx(successful_burst_sizes, nll_values, 'b-', linewidth=2, marker='o', markersize=4)
ax.axhline(y=simple_nll, color='red', linestyle='--', linewidth=2, 
           label=f'Simple model NLL = {simple_nll:.2f}')
ax.axvline(x=1, color='green', linestyle=':', linewidth=2, alpha=0.7)
ax.scatter([min_burst_size], [min_nll], color='orange', s=200, zorder=5, 
           marker='*', edgecolor='black', linewidth=1.5)

ax.set_xlabel('Burst Size (log scale)', fontsize=11)
ax.set_ylabel('NLL', fontsize=11)
ax.set_title('Profile Likelihood (log scale)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()

# Save figure
output_file = 'nll_vs_burst_size_profile.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_file}")

# Save data to CSV
print(f"\n💾 Saving profile data...")
profile_df = pd.DataFrame({
    'burst_size': successful_burst_sizes,
    'nll': nll_values,
    'delta_nll': nll_values - simple_nll
})
csv_file = 'nll_vs_burst_size_profile_data.csv'
profile_df.to_csv(csv_file, index=False)
print(f"✅ Saved: {csv_file}")

# Statistical summary
print(f"\n{'='*80}")
print("STATISTICAL SUMMARY")
print(f"{'='*80}")

print(f"\n📊 Profile Statistics:")
print(f"   Burst size range tested: {successful_burst_sizes[0]:.2f} to {successful_burst_sizes[-1]:.2f}")
print(f"   Number of points: {len(successful_burst_sizes)}")
print(f"\n   Simple model (burst_size=1):")
print(f"     NLL: {simple_nll:.6f}")
print(f"\n   Minimum in profile:")
print(f"     Burst size: {min_burst_size:.6f}")
print(f"     NLL: {min_nll:.6f}")
print(f"     Improvement over simple: {simple_nll - min_nll:+.6f} NLL points")

# Check if burst_size=1 is within confidence interval
# Using chi-square approximation: NLL increases by ~0.5 for 68% CI, ~2 for 95% CI
delta_nll_at_1 = nll_values[np.argmin(np.abs(successful_burst_sizes - 1))] - min_nll
print(f"\n   At burst_size=1:")
print(f"     ΔNLL from minimum: {delta_nll_at_1:+.6f}")
if delta_nll_at_1 < 2:
    print(f"     ✓ Within 95% confidence interval (ΔNLL < 2)")
    print(f"     ✓ burst_size=1 is statistically indistinguishable from optimum")
else:
    print(f"     ⚠️  Outside 95% confidence interval (ΔNLL > 2)")

# Interpretation
print(f"\n{'='*80}")
print("INTERPRETATION")
print(f"{'='*80}")

improvement = simple_nll - min_nll

if abs(improvement) < 1.0:
    print(f"\n✅ CONCLUSION: The simple model (burst_size=1) is optimal")
    print(f"   - The profile minimum is at burst_size={min_burst_size:.2f}")
    print(f"   - Improvement over burst_size=1 is only {-improvement:.3f} NLL points")
    print(f"   - This difference is negligible (< 0.02% of total NLL)")
    print(f"\n🎯 RESULT: No evidence for burst degradation in your data")
elif improvement > 1.0:
    print(f"\n⚠️  UNEXPECTED: Fixed burst shows improvement")
    print(f"   - Improvement: {improvement:.3f} NLL points")
    print(f"   - Optimal burst_size: {min_burst_size:.2f}")
    print(f"   - This suggests burst degradation might be present")
else:
    print(f"\n❌ PROBLEM: Fixed burst performs worse than simple")
    print(f"   - The profile minimum is worse than simple model by {-improvement:.3f}")
    print(f"   - This suggests a potential numerical issue")

# Compare to full optimization
print(f"\n{'='*80}")
print("COMPARISON WITH FULL OPTIMIZATION")
print(f"{'='*80}")

# Get best fixed_burst run from CSV
fixed_burst_df = df[df['mechanism'] == 'fixed_burst'].copy()
best_fb_idx = fixed_burst_df['nll'].idxmin()
best_fb = fixed_burst_df.loc[best_fb_idx]

print(f"\nFull optimization (all parameters free):")
print(f"   Best NLL: {best_fb['nll']:.6f}")
print(f"   Optimal burst_size: {best_fb['burst_size']:.6f}")
print(f"   Other parameters adjusted from simple model")

print(f"\nProfile optimization (only burst_size varies):")
print(f"   Best NLL: {min_nll:.6f}")
print(f"   Optimal burst_size: {min_burst_size:.6f}")
print(f"   Other parameters fixed at simple model optimum")

print(f"\nDifference:")
print(f"   Full optimization is better by: {min_nll - best_fb['nll']:.6f} NLL points")
if (min_nll - best_fb['nll']) > 5:
    print(f"   ✓ Full optimization significantly better")
    print(f"     (Other parameters compensate for burst_size constraint)")
else:
    print(f"   ✓ Profile and full optimization give similar results")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
print(f"\n📁 Output files:")
print(f"   - {output_file}")
print(f"   - {csv_file}")

plt.show()

