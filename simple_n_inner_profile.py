#!/usr/bin/env python3
"""
Simple profile likelihood plot: NLL vs n_inner for feedback_onion mechanism
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'SecondVersion'))
from MoMOptimization_join import joint_objective, get_mechanism_info
from simulation_utils import load_experimental_data

print("Loading data and computing NLL profile...")

# Load experimental data
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

# Load all simple model runs
df = pd.read_csv('optimized_params_runs_20251113_145124.csv')
simple_df = df[df['mechanism'] == 'simple'].copy()

# Test n_inner values from N (no feedback) to very large (strong feedback)
# First need to get typical N values from simple model
typical_N2 = simple_df['N2'].mean()

# Create n_inner range: from typical_N2 to 10*typical_N2 (and beyond for very weak feedback)
n_inner_values = np.concatenate([
    np.linspace(typical_N2, 2*typical_N2, 30),  # Fine resolution near N
    np.linspace(2*typical_N2, 5*typical_N2, 20),  # Medium resolution
])
n_inner_values = np.unique(n_inner_values)
n_inner_values = np.sort(n_inner_values)

mechanism_info = get_mechanism_info('feedback_onion', gamma_mode='separate')

# Compute profile for ALL simple model runs
all_profiles = []
colors = plt.cm.rainbow(np.linspace(0, 1, len(simple_df)))

for idx, (row_idx, simple_run) in enumerate(simple_df.iterrows()):
    print(f"Computing profile {idx+1}/{len(simple_df)} (Run #{int(simple_run['run_number'])}, NLL={simple_run['nll']:.2f})...")
    
    fixed_params = {
        'n2': simple_run['n2'],
        'N2': simple_run['N2'],
        'k': simple_run['k'],
        'r21': simple_run['r21'],
        'r23': simple_run['r23'],
        'R21': simple_run['R21'],
        'R23': simple_run['R23'],
        'alpha': simple_run['alpha'],
        'beta_k': simple_run['beta_k'],
        'beta2_k': simple_run['beta2_k'],
        'beta3_k': simple_run['beta3_k']
    }
    
    nll_values = []
    successful_n_inner = []
    
    for n_inner in n_inner_values:
        # Order for feedback_onion: n2, N2, k, r21, r23, R21, R23, n_inner, alpha, beta_k, beta2_k, beta3_k
        params_vector = np.array([
            fixed_params['n2'], fixed_params['N2'], fixed_params['k'],
            fixed_params['r21'], fixed_params['r23'], fixed_params['R21'], fixed_params['R23'],
            n_inner,
            fixed_params['alpha'], fixed_params['beta_k'],
            fixed_params['beta2_k'], fixed_params['beta3_k']
        ])
        
        try:
            nll = joint_objective(
                params_vector, 'feedback_onion', mechanism_info,
                data_arrays['data_wt12'], data_arrays['data_wt32'],
                data_arrays['data_threshold12'], data_arrays['data_threshold32'],
                data_arrays['data_degrate12'], data_arrays['data_degrate32'],
                data_arrays['data_initial12'], data_arrays['data_initial32'],
                data_arrays['data_degrateAPC12'], data_arrays['data_degrateAPC32'],
                data_arrays['data_velcade12'], data_arrays['data_velcade32']
            )
            if nll < 1e6:
                nll_values.append(nll)
                successful_n_inner.append(n_inner)
        except:
            pass
    
    all_profiles.append({
        'n_inner_values': np.array(successful_n_inner),
        'nll_values': np.array(nll_values),
        'color': colors[idx],
        'run_number': int(simple_run['run_number']),
        'simple_nll': simple_run['nll'],
        'N2': fixed_params['N2']
    })

print(f"\nCreating plot...")

# Create plot
plt.figure(figsize=(12, 7))

# Plot all profiles with different colors
for i, profile in enumerate(all_profiles):
    if i == 0:
        plt.plot(profile['n_inner_values'], profile['nll_values'], 
                color=profile['color'], linewidth=2, alpha=0.7,
                label=f'Simple model profiles (20 runs)')
    else:
        plt.plot(profile['n_inner_values'], profile['nll_values'], 
                color=profile['color'], linewidth=2, alpha=0.7)

# Find and highlight the best profile
best_profile = min(all_profiles, key=lambda x: x['simple_nll'])
plt.plot(best_profile['n_inner_values'], best_profile['nll_values'], 
        color='red', linewidth=3, linestyle='--', alpha=0.9,
        label=f'Best run (#{best_profile["run_number"]}, NLL={best_profile["simple_nll"]:.2f})')

# Add vertical line at typical N2 to show where feedback becomes negligible
plt.axvline(x=typical_N2, color='gray', linestyle=':', linewidth=2, alpha=0.5,
           label=f'Mean N2={typical_N2:.0f} (feedback threshold)')

plt.xlabel('n_inner (inner cohesin threshold)', fontsize=12)
plt.ylabel('Negative Log-Likelihood (NLL)', fontsize=12)
plt.title('Profile Likelihood: NLL vs n_inner for Feedback Onion Model\n(Each line uses different optimized parameters from simple model)', 
          fontsize=13, fontweight='bold')
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Add annotation
plt.text(0.02, 0.98, 
         'n_inner >> N → no feedback (should match simple model)\nn_inner ≈ N → feedback active', 
         transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig('nll_vs_n_inner.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: nll_vs_n_inner.png")
print(f"Total profiles computed: {len(all_profiles)}")
print(f"NLL range across all runs: {min(p['simple_nll'] for p in all_profiles):.2f} to {max(p['simple_nll'] for p in all_profiles):.2f}")
print(f"n_inner range tested: {n_inner_values.min():.0f} to {n_inner_values.max():.0f}")
print(f"Typical N2 value: {typical_N2:.0f}")

plt.show()

