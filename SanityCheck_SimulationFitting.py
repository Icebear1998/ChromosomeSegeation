#!/usr/bin/env python3
"""
Simulation-based Fitting Sanity Check

This script validates the end-to-end simulation-based fitting pipeline by:
1) Generating synthetic datasets using MultiMechanismSimulationTimevary with known parameters
2) Running simulation-based joint optimization to recover parameters
3) Plotting results: data histograms with KDE overlays from true and recovered parameters, and parameter comparison plots

Outputs are saved under Results/SimulationSanityCheck/.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# Local imports
from MultiMechanismSimulationTimevary import MultiMechanismSimulationTimevary
from SimulationOptimization_join import run_optimization as run_sim_optimization
from SimulationOptimization_independent import run_independent_optimization as run_sim_independent

# Ensure reproducibility
np.random.seed(42)


def derive_wildtype_params_from_ratios(params: Dict[str, float]) -> Dict[str, float]:
	"""Given ratio-based params (n2,N2,r21,r23,R21,R23, k_1,k_max, ...), derive n1,n3,N1,N3."""
	n1 = max(params['r21'] * params['n2'], 1)
	n3 = max(params['r23'] * params['n2'], 1)
	N1 = max(params['R21'] * params['N2'], 1)
	N3 = max(params['R23'] * params['N2'], 1)
	out = dict(params)
	out.update({'n1': n1, 'n3': n3, 'N1': N1, 'N3': N3})
	return out


def get_test_params(mechanism: str) -> Dict[str, float]:
	"""Provide a reasonable true parameter set for a selected mechanism (ratio-based)."""
	base = {
		'n2': 22.0,
		'N2': 180.0,
		'r21': 1.2,
		'r23': 1.4,
		'R21': 0.9,
		'R23': 1.3,
		'k_1': 0.006,
		'k_max': 0.12,
		'alpha': 0.7,
		'beta_k': 0.8,    # affects k_max
		'beta2_k': 0.6    # affects k_1
	}
	if mechanism == 'time_varying_k':
		return base
	elif mechanism == 'time_varying_k_fixed_burst':
		base['burst_size'] = 6.0
		return base
	elif mechanism == 'time_varying_k_feedback_onion':
		base['n_inner'] = 30.0
		return base
	elif mechanism == 'time_varying_k_burst_onion':
		base['burst_size'] = 5.0
		return base
	elif mechanism == 'time_varying_k_combined':
		base['burst_size'] = 5.0
		base['n_inner'] = 30.0
		return base
	else:
		raise ValueError(f"Unknown mechanism: {mechanism}")


def _rate_params_for_mechanism(mechanism: str, params: Dict[str, float]) -> Dict[str, float]:
	"""Build rate_params for MultiMechanismSimulationTimevary based on mechanism."""
	rp = {'k_1': params['k_1'], 'k_max': params['k_max']}
	if mechanism == 'time_varying_k_fixed_burst':
		rp['burst_size'] = params['burst_size']
	elif mechanism == 'time_varying_k_feedback_onion':
		rp['n_inner'] = params['n_inner']
	elif mechanism == 'time_varying_k_combined':
		rp['burst_size'] = params['burst_size']
		rp['n_inner'] = params['n_inner']
	elif mechanism == 'time_varying_k_burst_onion':
		rp['burst_size'] = params['burst_size']
	return rp


def _apply_mutant(mechanism: str, wt_params: Dict[str, float], strain: str) -> Tuple[Dict[str, float], list]:
	"""Return (sim_params, n0_list) for a given strain from wildtype params.
	- threshold multiplies n by alpha
	- degrate scales k_max by beta_k
	- degrateAPC scales k_1 by beta2_k
	"""
	p = dict(wt_params)
	n1, n2, n3 = p['n1'], p['n2'], p['n3']
	N1, N2, N3 = p['N1'], p['N2'], p['N3']
	k_1, k_max = p['k_1'], p['k_max']
	alpha = p.get('alpha', 1.0)
	beta_k = p.get('beta_k', 1.0)
	beta2_k = p.get('beta2_k', 1.0)
	if strain == 'threshold':
		n1, n2, n3 = max(n1 * alpha, 1), max(n2 * alpha, 1), max(n3 * alpha, 1)
	elif strain == 'degrate':
		k_max = max(k_max * beta_k, 1e-6)
	elif strain == 'degrateAPC':
		k_1 = max(k_1 * beta2_k, 1e-6)
	p.update({'k_1': k_1, 'k_max': k_max})
	rp = _rate_params_for_mechanism(mechanism, p)
	n0_list = [int(round(n1)), int(round(n2)), int(round(n3))]
	return {'N1': int(round(N1)), 'N2': int(round(N2)), 'N3': int(round(N3)), **rp}, n0_list


def simulate_dataset(mechanism: str, params: Dict[str, float], num_simulations: int, max_time: float):
	"""Simulate a dataset (delta_t12, delta_t32 arrays) for the given params across strains."""
	datasets = {}
	for strain in ['wildtype', 'threshold', 'degrate', 'degrateAPC']:
		sim_params, n0_list = _apply_mutant(mechanism, params, strain)
		initial_state = [sim_params['N1'], sim_params['N2'], sim_params['N3']]
		rate_params = {k: v for k, v in sim_params.items() if k not in ['N1', 'N2', 'N3']}
		dt12, dt32 = [], []
		for _ in range(num_simulations):
			try:
				sim = MultiMechanismSimulationTimevary(
					mechanism=mechanism,
					initial_state_list=initial_state,
					rate_params=rate_params,
					n0_list=n0_list,
					max_time=max_time
				)
				_, _, sep = sim.simulate()
				dt12.append(sep[0] - sep[1])
				dt32.append(sep[2] - sep[1])
			except Exception as e:
				print(f"Sim warning ({strain}): {e}")
		datasets[strain] = {
			'delta_t12': np.array(dt12, dtype=float),
			'delta_t32': np.array(dt32, dtype=float)
		}
	print("Synthetic data generated for all strains.")
	return datasets


def kde_curve(samples: np.ndarray, x_grid: np.ndarray):
	"""Return KDE evaluated on x_grid for the given samples (guards against small sample sizes)."""
	from scipy.stats import gaussian_kde
	if samples is None or len(samples) < 10 or np.all(samples == samples[0]):
		return np.zeros_like(x_grid)
	try:
		kde = gaussian_kde(samples)
		return kde(x_grid)
	except Exception:
		return np.zeros_like(x_grid)


def _get_strain_adjusted_params(base_params: Dict[str, float], strain: str) -> Tuple:
	"""Return adjusted (n1,n2,n3,N1,N2,N3,k_1,k_max) for a given strain based on mutant multipliers."""
	n1 = base_params.get('n1')
	n2 = base_params.get('n2') 
	n3 = base_params.get('n3')
	N1 = base_params.get('N1')
	N2 = base_params.get('N2')
	N3 = base_params.get('N3')
	k_1 = base_params.get('k_1')
	k_max = base_params.get('k_max')
	alpha = base_params.get('alpha', 1.0)
	beta_k = base_params.get('beta_k', 1.0)
	beta2_k = base_params.get('beta2_k', 1.0)
	
	if any(v is None for v in [n1, n2, n3, N1, N2, N3, k_1, k_max]):
		return None
		
	if strain == 'threshold':
		n1, n2, n3 = max(n1 * alpha, 1), max(n2 * alpha, 1), max(n3 * alpha, 1)
	elif strain == 'degrate':
		k_max = max(k_max * beta_k, 1e-6)
	elif strain == 'degrateAPC':
		k_1 = max(k_1 * beta2_k, 1e-6)
	
	return n1, n2, n3, N1, N2, N3, k_1, k_max


def plot_hist_with_kde_overlays(datasets_true: Dict, datasets_rec_joint: Dict, datasets_rec_indep: Dict, out_dir: str, tag: str, true_params: Dict = None, rec_params_joint: Dict = None, rec_params_indep: Dict = None):
	"""For each strain and pair, plot empirical histogram (true) and KDE overlays (true vs joint vs independent recovered)."""
	os.makedirs(out_dir, exist_ok=True)
	for strain in ['wildtype', 'threshold', 'degrate', 'degrateAPC']:
		for key, title_pair in [('delta_t12', 'T1-T2'), ('delta_t32', 'T3-T2')]:
			true_vals = datasets_true[strain][key]
			if true_vals.size == 0:
				continue
			xmin, xmax = float(np.nanmin(true_vals)), float(np.nanmax(true_vals))
			if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
				continue
			x_grid = np.linspace(xmin, xmax, 400)
			plt.figure(figsize=(10, 5))
			plt.hist(true_vals, bins=50, density=True, alpha=0.35, color='gray', edgecolor='none', label='Synthetic (empirical)')
			# True KDE
			y_true = kde_curve(true_vals, x_grid)
			plt.plot(x_grid, y_true, color='C2', lw=2, label='True KDE')
			# Joint recovered KDE (if available)
			if datasets_rec_joint is not None and strain in datasets_rec_joint:
				rec_vals = datasets_rec_joint[strain][key]
				y_rec = kde_curve(rec_vals, x_grid)
				plt.plot(x_grid, y_rec, color='C0', lw=2, label='Joint fit KDE')
			# Independent recovered KDE (if available)
			if datasets_rec_indep is not None and strain in datasets_rec_indep:
				rec_vals = datasets_rec_indep[strain][key]
				y_rec = kde_curve(rec_vals, x_grid)
				plt.plot(x_grid, y_rec, color='C1', lw=2, label='Independent fit KDE')
			
			# Build title with parameter comparison
			title = f"{strain} – {title_pair}"
			
			if true_params is not None:
				# Get strain-adjusted parameters
				true_adj = _get_strain_adjusted_params(true_params, strain)
				
				if true_adj is not None:
					true_n1, true_n2, true_n3, true_N1, true_N2, true_N3, true_k1, true_kmax = true_adj
					
					# Show relevant parameters for this timing pair
					if key == 'delta_t12':  # T1-T2
						title += f"\nTrue: n1={true_n1:.0f}, N1={true_N1:.0f}, k_1={true_k1:.3f}, k_max={true_kmax:.3f}"
					else:  # T3-T2
						title += f"\nTrue: n3={true_n3:.0f}, N3={true_N3:.0f}, k_1={true_k1:.3f}, k_max={true_kmax:.3f}"
					
					# Add recovered parameters from both methods
					for method_name, rec_params in [('Joint fit', rec_params_joint), ('Independent fit', rec_params_indep)]:
						if rec_params is not None:
							rec_adj = _get_strain_adjusted_params(rec_params, strain)
							if rec_adj is not None:
								rec_n1, rec_n2, rec_n3, rec_N1, rec_N2, rec_N3, rec_k1, rec_kmax = rec_adj
								if key == 'delta_t12':  # T1-T2
									title += f"\n{method_name}: n1={rec_n1:.0f}, N1={rec_N1:.0f}, k_1={rec_k1:.3f}, k_max={rec_kmax:.3f}"
								else:  # T3-T2
									title += f"\n{method_name}: n3={rec_n3:.0f}, N3={rec_N3:.0f}, k_1={rec_k1:.3f}, k_max={rec_kmax:.3f}"
			
			plt.title(title, fontsize=10)
			plt.xlabel('Delta time')
			plt.ylabel('Density')
			plt.legend()
			fname = os.path.join(out_dir, f"hist_kde_{strain}_{'12' if key=='delta_t12' else '32'}_{tag}.png")
			plt.tight_layout()
			plt.savefig(fname, dpi=150)
			plt.close()


def plot_param_bars(true_params: Dict[str, float], rec_params: Dict[str, float], out_dir: str, method_tag: str):
	os.makedirs(out_dir, exist_ok=True)
	keys_order = ['n2','N2','r21','r23','R21','R23','k_1','k_max','burst_size','n_inner','alpha','beta_k','beta2_k','n1','n3','N1','N3']
	keys = [k for k in keys_order if k in true_params and rec_params and k in rec_params]
	if not keys:
		return
	true_vals = [true_params[k] for k in keys]
	rec_vals = [rec_params[k] for k in keys]
	x = np.arange(len(keys))
	w = 0.38
	plt.figure(figsize=(max(8, len(keys)*0.6), 4.5))
	plt.bar(x - w/2, true_vals, w, label='True', color='C2', alpha=0.75)
	plt.bar(x + w/2, rec_vals, w, label='Recovered', color='C0', alpha=0.75)
	plt.xticks(x, keys, rotation=45, ha='right')
	plt.ylabel('Value')
	plt.title(f'Parameter Comparison – {method_tag}')
	plt.legend()
	plt.tight_layout()
	fname = os.path.join(out_dir, f"params_{method_tag}.png")
	plt.savefig(fname, dpi=150)
	plt.close()


def main():
	print("Simulation-based Fitting Sanity Check")
	print("=" * 60)
	# Config
	mechanism = 'time_varying_k_combined'  # Change to any supported mechanism
	num_simulations_gen = 200  # per strain for synthetic generation
	num_simulations_fit = 50  # per evaluation in optimization (SimulationOptimization_join defaults will be used)
	max_time = 200
	out_dir = os.path.join('Results', 'SimulationSanityCheck')
	os.makedirs(out_dir, exist_ok=True)

	# True parameters (ratio-based), derive WT absolute n/N
	true_ratio = get_test_params(mechanism)
	true_full = derive_wildtype_params_from_ratios(true_ratio)
	print("True parameters (ratio-based -> derived):")
	print({k: round(v, 3) for k, v in true_full.items()})

	# Step 1: Generate synthetic data using simulation engine (as experimental data)
	print("\nGenerating synthetic datasets...")
	synthetic_datasets = simulate_dataset(mechanism, true_full, num_simulations_gen, max_time)

	# Step 2: Run simulation-based joint optimization to recover params
	print("\nRunning simulation-based joint optimization...")
	results_joint = run_sim_optimization(
		mechanism=mechanism,
		datasets=synthetic_datasets,
		max_iterations=40,
		num_simulations=num_simulations_fit,
		selected_strains=None
	)
	recovered_joint = results_joint['params'] if results_joint and results_joint.get('success') else None
	recovered_joint_full = derive_wildtype_params_from_ratios(recovered_joint) if recovered_joint else None

	# Step 3: Run simulation-based independent optimization to recover params
	print("\nRunning simulation-based independent optimization...")
	results_indep = run_sim_independent(
		mechanism=mechanism,
		datasets=synthetic_datasets,
		max_iterations_wt=40,
		max_iterations_mut=20,
		num_simulations=num_simulations_fit,
		selected_strains=None
	)
	recovered_indep = results_indep['complete_params'] if results_indep and results_indep.get('success') else None
	recovered_indep_full = derive_wildtype_params_from_ratios(recovered_indep) if recovered_indep else None

	# Step 4: Re-simulate from recovered parameters for KDE overlays
	print("\nRe-simulating from recovered parameters for plotting...")
	rec_datasets_joint = simulate_dataset(mechanism, recovered_joint_full, num_simulations_gen, max_time) if recovered_joint_full else None
	rec_datasets_indep = simulate_dataset(mechanism, recovered_indep_full, num_simulations_gen, max_time) if recovered_indep_full else None

	# Step 5: Plots
	print("\nGenerating plots...")
	plot_hist_with_kde_overlays(synthetic_datasets, rec_datasets_joint, rec_datasets_indep, out_dir, tag=mechanism, 
	                           true_params=true_full, rec_params_joint=recovered_joint_full, rec_params_indep=recovered_indep_full)
	
	# Generate parameter comparison plots for both methods
	if recovered_joint_full:
		plot_param_bars(true_full, recovered_joint_full, out_dir, method_tag=f"{mechanism}_joint")
	if recovered_indep_full:
		plot_param_bars(true_full, recovered_indep_full, out_dir, method_tag=f"{mechanism}_independent")
	
	print(f"Plots saved under: {out_dir}")

	print("\nSummary:")
	print("True (subset):", {k: round(true_full[k], 3) for k in ['n2','N2','r21','r23','R21','R23','k_1','k_max'] if k in true_full})
	
	if recovered_joint_full:
		print("Joint Recovered (subset):", {k: round(recovered_joint_full[k], 3) for k in ['n2','N2','r21','r23','R21','R23','k_1','k_max'] if k in recovered_joint_full})
	else:
		print("Joint optimization: FAILED")
	
	if recovered_indep_full:
		print("Independent Recovered (subset):", {k: round(recovered_indep_full[k], 3) for k in ['n2','N2','r21','r23','R21','R23','k_1','k_max'] if k in recovered_indep_full})
	else:
		print("Independent optimization: FAILED")
		
	print("Done.")


if __name__ == "__main__":
	main()
