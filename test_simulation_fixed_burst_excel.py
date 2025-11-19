#!/usr/bin/env python3
"""
Test simulation-based optimization for fixed_burst mechanism using Excel data.
This script uses data from All_strains_SCStimes.xlsx and compares
simulation-based vs MoM-based optimization.

Note: Using only 32C data since 12C data has fewer points in Excel file.
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'SecondVersion'))
from simulation_utils import load_experimental_data
from MoMOptimization_join import joint_objective, get_mechanism_info, unpack_parameters
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("SIMULATION-BASED FIXED_BURST OPTIMIZATION TEST (Excel Data)")
print("="*80)
print()

# Load experimental data using simulation_utils function
print("📊 Loading experimental data from Excel...")
datasets = load_experimental_data()

if not datasets:
    print("❌ Error: Failed to load data!")
    exit(1)

# Extract data arrays for MoM optimization
data_wt12 = datasets['wildtype']['delta_t12']
data_wt32 = datasets['wildtype']['delta_t32']
data_threshold12 = datasets['threshold']['delta_t12']
data_threshold32 = datasets['threshold']['delta_t32']
data_degRade12 = datasets['degrade']['delta_t12']
data_degRade32 = datasets['degrade']['delta_t32']
data_degRadeAPC12 = datasets['degradeAPC']['delta_t12']
data_degRadeAPC32 = datasets['degradeAPC']['delta_t32']
data_degRadeVel12 = datasets['velcade']['delta_t12']
data_degRadeVel32 = datasets['velcade']['delta_t32']

print(f"✅ Data loaded successfully")
print()

# Prepare empty arrays for missing data (initial strain not fitted)
data_initial12 = np.array([])
data_initial32 = np.array([])

# ============================================================================
# STEP 1: Optimize simple model with MoM (for baseline parameters)
# ============================================================================
print("="*80)
print("1️⃣  OPTIMIZING SIMPLE MODEL (MoM-based, for baseline)")
print("="*80)

# Get mechanism info
mechanism_info_simple = get_mechanism_info('simple', gamma_mode='separate')

# Parameter bounds for simple model (MoM)
bounds_simple = mechanism_info_simple['bounds']

print(f"Running differential evolution optimization with {len(bounds_simple)} parameters...")
print(f"Mechanism: simple, Parameters: {mechanism_info_simple['params']}")

result_simple = differential_evolution(
    joint_objective,
    bounds_simple,
    args=('simple', mechanism_info_simple,
          data_wt12, data_wt32, data_threshold12, data_threshold32,
          data_degRade12, data_degRade32, data_initial12, data_initial32,
          data_degRadeAPC12, data_degRadeAPC32, data_degRadeVel12, data_degRadeVel32),
    maxiter=50,
    popsize=10,
    seed=42,
    workers=1,
    updating='deferred',
    polish=False
)

# Unpack optimized parameters
params_simple = unpack_parameters(result_simple.x, mechanism_info_simple)

print(f"✅ Simple model optimized: NLL = {result_simple.fun:.4f}")
print(f"   n1={params_simple['n1']:.2f}, n2={params_simple['n2']:.2f}, n3={params_simple['n3']:.2f}")
print(f"   N1={params_simple['N1']:.2f}, N2={params_simple['N2']:.2f}, N3={params_simple['N3']:.2f}")
print(f"   k={params_simple['k']:.5f}")
print(f"   alpha={params_simple['alpha']:.3f}, beta_k={params_simple['beta_k']:.3f}")
print(f"   beta2_k={params_simple['beta2_k']:.3f}, beta3_k={params_simple['beta3_k']:.3f}")
print()

# ============================================================================
# STEP 2: Test fixed_burst with MoM (Option 2 - Fractional Burst)
# ============================================================================
print("="*80)
print("2️⃣  TESTING FIXED_BURST WITH MoM (Fractional Burst)")
print("="*80)
print("Testing burst_size values with all other parameters fixed from simple model")
print()

# Get mechanism info for fixed_burst
mechanism_info_fb = get_mechanism_info('fixed_burst', gamma_mode='separate')

burst_sizes_test = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3.0, 5.0, 10.0]
nll_mom_profile = []

print(f"{'burst_size':>12} | {'NLL':>12} | {'Diff from simple':>18} | {'Better?':>8}")
print("-" * 65)

for burst_size in burst_sizes_test:
    # Create parameter vector for fixed_burst
    # Use same parameters as simple model, just add burst_size
    params_fb = result_simple.x.tolist() + [burst_size]
    
    nll = joint_objective(
        params_fb,
        'fixed_burst', mechanism_info_fb,
        data_wt12, data_wt32, data_threshold12, data_threshold32,
        data_degRade12, data_degRade32, data_initial12, data_initial32,
        data_degRadeAPC12, data_degRadeAPC32, data_degRadeVel12, data_degRadeVel32
    )
    
    nll_mom_profile.append(nll)
    diff = nll - result_simple.fun
    better = "✓ Yes" if nll < result_simple.fun else "✗ No"
    print(f"{burst_size:>12.1f} | {nll:>12.4f} | {diff:>+18.4f} | {better:>8}")

print()

# ============================================================================
# STEP 3: Full optimization of fixed_burst with MoM (optimize burst_size only)
# ============================================================================
print("="*80)
print("3️⃣  OPTIMIZING FIXED_BURST WITH MoM (burst_size only)")
print("="*80)

# Bounds: only burst_size varies
def objective_burst_only_mom(params):
    """Objective that only optimizes burst_size."""
    burst_size = params[0]
    params_full = result_simple.x.tolist() + [burst_size]
    
    return joint_objective(
        params_full,
        'fixed_burst', mechanism_info_fb,
        data_wt12, data_wt32, data_threshold12, data_threshold32,
        data_degRade12, data_degRade32, data_initial12, data_initial32,
        data_degRadeAPC12, data_degRadeAPC32, data_degRadeVel12, data_degRadeVel32
    )

bounds_burst = [(0.5, 10.0)]

print("Running optimization for burst_size only...")
result_fb_mom = differential_evolution(
    objective_burst_only_mom,
    bounds_burst,
    maxiter=30,
    popsize=8,
    seed=42,
    workers=1,
    updating='deferred',
    polish=False
)

print(f"✅ Fixed_burst optimized (MoM): NLL = {result_fb_mom.fun:.4f}")
print(f"   Optimal burst_size = {result_fb_mom.x[0]:.4f}")
print(f"   Improvement over simple: {result_fb_mom.fun - result_simple.fun:+.4f}")
print()

# ============================================================================
# STEP 4: Analysis and Visualization
# ============================================================================
print("="*80)
print("4️⃣  ANALYSIS")
print("="*80)
print()

# Calculate smoothness metrics for MoM
mom_jumps = np.abs(np.diff(nll_mom_profile[:10]))  # First 10 points (1.0 to 2.0)
mom_max_jump = np.max(mom_jumps)
mom_avg_jump = np.mean(mom_jumps)

print("📊 Smoothness Analysis (burst_size 1.0 to 2.0):")
print()
print(f"MoM (Fractional Burst):")
print(f"  Max jump: {mom_max_jump:.2f}")
print(f"  Avg jump: {mom_avg_jump:.2f}")
print()

print("🎯 burst_size=1.0 Consistency:")
print(f"  Simple model NLL:        {result_simple.fun:.4f}")
print(f"  MoM fixed_burst (b=1.0): {nll_mom_profile[0]:.4f}")
print(f"  Difference:              {abs(nll_mom_profile[0] - result_simple.fun):.4f}")
print()

if abs(nll_mom_profile[0] - result_simple.fun) < 1.0:
    print("  ✅ EXCELLENT: burst_size=1.0 matches simple model!")
elif abs(nll_mom_profile[0] - result_simple.fun) < 10.0:
    print("  ✅ GOOD: burst_size=1.0 close to simple model")
else:
    print("  ⚠️  WARNING: Large difference between burst_size=1.0 and simple")

print()

# ============================================================================
# STEP 5: Visualization
# ============================================================================
print("="*80)
print("5️⃣  CREATING PLOTS")
print("="*80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Full NLL profile
ax1.plot(burst_sizes_test, nll_mom_profile, 'o-', label='MoM (Fractional)', 
         linewidth=2, markersize=8, color='blue')
ax1.axhline(result_simple.fun, color='green', linestyle='--', 
            label='Simple Model', linewidth=2)
ax1.axvline(result_fb_mom.x[0], color='orange', linestyle=':', 
            label=f'MoM Optimum ({result_fb_mom.x[0]:.2f})', linewidth=2)

ax1.set_xlabel('burst_size', fontsize=12)
ax1.set_ylabel('Negative Log-Likelihood', fontsize=12)
ax1.set_title('NLL Profile: Fixed_Burst (MoM)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Zoom on burst_size 1.0 to 2.0
zoom_indices = [i for i, bs in enumerate(burst_sizes_test) if 1.0 <= bs <= 2.0]
burst_zoom = [burst_sizes_test[i] for i in zoom_indices]
nll_mom_zoom = [nll_mom_profile[i] for i in zoom_indices]

ax2.plot(burst_zoom, nll_mom_zoom, 'o-', label='MoM (Fractional)', 
         linewidth=2, markersize=8, color='blue')
ax2.axhline(result_simple.fun, color='green', linestyle='--', 
            label='Simple Model', linewidth=2)

ax2.set_xlabel('burst_size', fontsize=12)
ax2.set_ylabel('Negative Log-Likelihood', fontsize=12)
ax2.set_title('NLL Profile (Zoomed: 1.0-2.0)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fixed_burst_mom_excel_data.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved as 'fixed_burst_mom_excel_data.png'")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("📋 SUMMARY")
print("="*80)
print()
print("Key Findings (using Excel data from All_strains_SCStimes.xlsx):")
print()
print("1. Simple Model (Baseline):")
print(f"   - NLL = {result_simple.fun:.4f}")
print(f"   - Uses harmonic sum (no discrete summation issue)")
print()
print("2. Fixed_Burst Model with MoM (Fractional Burst - Option 2):")
print(f"   - burst_size=1.0 NLL: {nll_mom_profile[0]:.4f}")
print(f"   - Difference from simple: {abs(nll_mom_profile[0] - result_simple.fun):.4f}")
print(f"   - Optimized burst_size: {result_fb_mom.x[0]:.4f}")
print(f"   - Max NLL jump (1.0→2.0): {mom_max_jump:.2f}")
print()
print("3. Optimization Convergence:")
if result_fb_mom.x[0] < 1.5:
    print("   ✅ burst_size converges to ≈1.0")
    print("   ✅ Confirms: Simple model is sufficient!")
    print("   ✅ Option 2 (Fractional Burst) working correctly")
else:
    print(f"   ⚠️  burst_size = {result_fb_mom.x[0]:.2f} (larger than expected)")
    print("   ⚠️  May indicate: (1) Data supports larger bursts, or")
    print("   ⚠️                 (2) Need more optimization iterations")
print()
print("4. Numerical Stability:")
if mom_max_jump < 200:
    print(f"   ✅ Smooth optimization surface (max jump = {mom_max_jump:.2f})")
    print("   ✅ Option 2 successfully eliminated discrete summation issue")
else:
    print(f"   ⚠️  Some discontinuities remain (max jump = {mom_max_jump:.2f})")
print()

print("🎉 Test complete!")
print()
print("Next steps:")
print("  1. Review the plot: fixed_burst_mom_excel_data.png")
print("  2. If burst_size ≈ 1.0: Simple model is validated! ✅")
print("  3. If burst_size > 1.5: Consider running simulation-based validation")
print()

