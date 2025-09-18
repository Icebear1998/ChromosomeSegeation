#!/usr/bin/env python3
"""
Analysis and Visualization Script for Parameter Recovery Study

This script analyzes the results from the ARC parameter recovery study to:
1. Visualize parameter recovery accuracy
2. Identify parameter correlations and sloppiness
3. Assess model identifiability
4. Generate comprehensive recovery reports

Usage:
    python analyze_parameter_recovery.py

The script will automatically find the most recent recovery results file
(pattern: recovery_results_*.csv) and generate all plots and reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import glob
from datetime import datetime

# Optional imports
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

class ParameterRecoveryAnalyzer:
    """
    Analyzes parameter recovery study results from ARC.
    """
    
    def __init__(self, results_file=None):
        """
        Initialize analyzer.
        
        Args:
            results_file (str, optional): CSV file with recovery results. If None, finds most recent.
        """
        if results_file is None:
            # Find most recent recovery results file
            pattern = "recovery_results_*.csv"
            files = glob.glob(pattern)
            if not files:
                raise FileNotFoundError(
                    "No parameter recovery results found. Please download results from ARC first.\n"
                    "Expected filename pattern: recovery_results_*.csv"
                )
            results_file = max(files, key=os.path.getmtime)
            print(f"Using results file: {results_file}")
        elif results_file == 'recovery_results.csv' and not os.path.exists(results_file):
            # If default file doesn't exist, try to find timestamped files
            pattern = "recovery_results_*.csv"
            files = glob.glob(pattern)
            if files:
                results_file = max(files, key=os.path.getmtime)
                print(f"Default file not found, using most recent: {results_file}")
            else:
                raise FileNotFoundError(f"Recovery results file not found: {results_file}")
        else:
            print(f"Using specified results file: {results_file}")
        
        self.results_file = results_file
        self.df = pd.read_csv(results_file)
        
        # Extract mechanism name from filename
        if 'time_varying_k' in results_file:
            self.mechanism = 'time_varying_k_combined'
        elif 'combined' in results_file:
            self.mechanism = 'time_varying_k_combined'
        else:
            # Try to extract mechanism name from filename more safely
            filename_parts = results_file.split('_')
            # Look for mechanism-like parts (not dates or "recovery" or "results")
            mechanism_parts = []
            for part in filename_parts:
                if (not part.isdigit() and  # Not a date/timestamp
                    part not in ['recovery', 'results'] and  # Not standard prefixes
                    not part.endswith('.csv') and  # Not file extension
                    len(part) > 2):  # Meaningful length
                    mechanism_parts.append(part)
            
            if mechanism_parts:
                self.mechanism = '_'.join(mechanism_parts)
            else:
                # For simple filenames like 'recovery_results.csv', use generic name
                self.mechanism = 'combined_recovery'
        
        # Identify parameter columns (exclude metadata columns)
        metadata_cols = ['run_id', 'converged', 'final_nll', 'elapsed_time']
        self.param_cols = [col for col in self.df.columns 
                          if col not in metadata_cols and not col.endswith('_truth')]
        
        # Identify ground truth columns
        self.truth_cols = [col for col in self.df.columns if col.endswith('_truth')]
        
        print(f"Loaded recovery results:")
        print(f"  Mechanism: {self.mechanism}")
        print(f"  Total runs: {len(self.df)}")
        print(f"  Successful runs: {self.df['converged'].sum()}")
        print(f"  Parameters: {len(self.param_cols)}")
        
        # Display parameter names
        if len(self.param_cols) > 0:
            print(f"  Parameter names: {', '.join(self.param_cols[:5])}")
            if len(self.param_cols) > 5:
                print(f"                   ... and {len(self.param_cols)-5} more")
    
    def get_successful_results(self):
        """Get only successful recovery results."""
        return self.df[self.df['converged']].copy()
    
    def calculate_recovery_statistics(self):
        """
        Calculate detailed recovery statistics for each parameter.
        
        Returns:
            pd.DataFrame: Recovery statistics
        """
        successful_df = self.get_successful_results()
        
        if len(successful_df) == 0:
            print("Warning: No successful recoveries found!")
            return pd.DataFrame()
        
        stats_list = []
        
        for param in self.param_cols:
            truth_col = f"{param}_truth"
            if truth_col not in self.df.columns:
                continue
            
            truth_value = self.df[truth_col].iloc[0]  # Ground truth is same for all rows
            recovered_values = successful_df[param]
            
            # Calculate statistics
            stats_dict = {
                'parameter': param,
                'truth_value': truth_value,
                'n_recovered': len(recovered_values),
                'mean_recovered': recovered_values.mean(),
                'std_recovered': recovered_values.std(),
                'min_recovered': recovered_values.min(),
                'max_recovered': recovered_values.max(),
                'median_recovered': recovered_values.median(),
                'q25_recovered': recovered_values.quantile(0.25),
                'q75_recovered': recovered_values.quantile(0.75),
            }
            
            # Calculate errors
            absolute_errors = np.abs(recovered_values - truth_value)
            relative_errors = absolute_errors / np.abs(truth_value) * 100
            
            stats_dict.update({
                'mean_abs_error': absolute_errors.mean(),
                'std_abs_error': absolute_errors.std(),
                'mean_rel_error_pct': relative_errors.mean(),
                'std_rel_error_pct': relative_errors.std(),
                'max_rel_error_pct': relative_errors.max(),
            })
            
            # Recovery quality assessment
            # Consider "good recovery" if within 10% of truth value
            good_recoveries = relative_errors <= 10
            stats_dict['good_recovery_rate'] = good_recoveries.mean()
            
            # Coefficient of variation (measure of sloppiness)
            if np.abs(stats_dict['mean_recovered']) > 1e-10:
                stats_dict['cv_recovered'] = stats_dict['std_recovered'] / np.abs(stats_dict['mean_recovered'])
            else:
                stats_dict['cv_recovered'] = np.inf
            
            stats_list.append(stats_dict)
        
        return pd.DataFrame(stats_list)
    
    def perform_pca_analysis(self):
        """
        Perform PCA analysis to identify sloppy and stiff parameter directions.
        
        Returns:
            tuple: (pca, scaler, transformed_data, eigenvalues, eigenvectors)
        """
        successful_df = self.get_successful_results()
        
        if len(successful_df) == 0:
            print("No successful recoveries for PCA analysis!")
            return None, None, None, None, None
        
        # Get parameter data for PCA
        param_data = successful_df[self.param_cols].values
        
        # Standardize the data (important for PCA)
        scaler = StandardScaler()
        param_data_scaled = scaler.fit_transform(param_data)
        
        # Perform PCA
        pca = PCA()
        transformed_data = pca.fit_transform(param_data_scaled)
        
        # Get eigenvalues and eigenvectors
        eigenvalues = pca.explained_variance_
        eigenvectors = pca.components_
        
        print(f"PCA Analysis Results:")
        print(f"  Number of parameters: {len(self.param_cols)}")
        print(f"  Number of samples: {len(successful_df)}")
        print(f"  Explained variance ratio: {pca.explained_variance_ratio_[:5]}")  # Show first 5
        
        return pca, scaler, transformed_data, eigenvalues, eigenvectors
    
    def plot_pca_analysis(self, save_plots=True):
        """
        Create PCA plots showing sloppy and stiff directions.
        
        Args:
            save_plots (bool): Whether to save plots to files
        """
        pca_results = self.perform_pca_analysis()
        if pca_results[0] is None:
            return
        
        pca, scaler, transformed_data, eigenvalues, eigenvectors = pca_results
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.2, 1, 1], 
                              hspace=0.3, wspace=0.3)
        
        # 1. Eigenvalue spectrum (sloppy vs stiff directions)
        ax1 = fig.add_subplot(gs[0, 0])
        
        n_components = min(len(eigenvalues), 15)  # Show up to 15 components
        component_indices = np.arange(1, n_components + 1)
        
        # Plot eigenvalues on log scale
        ax1.semilogy(component_indices, eigenvalues[:n_components], 'bo-', markersize=8, linewidth=2)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Eigenvalue (log scale)')
        ax1.set_title('PCA Eigenvalue Spectrum\n(Sloppy vs Stiff Directions)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Identify sloppy vs stiff threshold (e.g., where eigenvalue drops significantly)
        eigenvalue_ratios = eigenvalues[:-1] / eigenvalues[1:]
        if len(eigenvalue_ratios) > 0:
            max_ratio_idx = np.argmax(eigenvalue_ratios)
            ax1.axvline(max_ratio_idx + 1.5, color='red', linestyle='--', alpha=0.7, 
                       label=f'Sloppy/Stiff Boundary')
            ax1.legend()
        
        # Add annotations for sloppy and stiff regions
        if n_components > 3:
            ax1.annotate('Stiff Directions\n(Well-constrained)', 
                        xy=(2, eigenvalues[1]), xytext=(3, eigenvalues[0]/2),
                        arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
                        fontsize=10, ha='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            
            if n_components > 5:
                sloppy_idx = min(n_components-2, max(5, max_ratio_idx + 2))
                ax1.annotate('Sloppy Directions\n(Poorly-constrained)', 
                            xy=(sloppy_idx, eigenvalues[sloppy_idx-1]), 
                            xytext=(sloppy_idx+1, eigenvalues[sloppy_idx-1]*10),
                            arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                            fontsize=10, ha='center',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        
        # 2. Cumulative explained variance
        ax2 = fig.add_subplot(gs[0, 1])
        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        ax2.plot(component_indices, cumulative_variance[:n_components], 'go-', markersize=6, linewidth=2)
        ax2.axhline(0.95, color='red', linestyle='--', alpha=0.7, label='95% Variance')
        ax2.axhline(0.99, color='orange', linestyle='--', alpha=0.7, label='99% Variance')
        
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 1.05)
        
        # 3. Parameter loadings for first few components
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Show loadings for first 3 components
        n_show_components = min(3, len(eigenvalues))
        param_names_short = [name[:8] + '...' if len(name) > 8 else name for name in self.param_cols]
        
        x_pos = np.arange(len(self.param_cols))
        width = 0.25
        
        colors = ['blue', 'red', 'green']
        for i in range(n_show_components):
            offset = (i - n_show_components//2) * width
            ax3.bar(x_pos + offset, eigenvectors[i], width, 
                   label=f'PC{i+1} (λ={eigenvalues[i]:.3f})', 
                   alpha=0.7, color=colors[i])
        
        ax3.set_xlabel('Parameters')
        ax3.set_ylabel('Loading')
        ax3.set_title('Parameter Loadings\n(First 3 Components)', fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(param_names_short, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 2D PCA scatter plot (first two components)
        ax4 = fig.add_subplot(gs[1, :2])
        
        if transformed_data.shape[1] >= 2:
            scatter = ax4.scatter(transformed_data[:, 0], transformed_data[:, 1], 
                                 alpha=0.6, s=50, c='blue', edgecolors='black', linewidth=0.5)
            
            ax4.set_xlabel(f'PC1 (λ={eigenvalues[0]:.3f}, {pca.explained_variance_ratio_[0]*100:.1f}% var)')
            ax4.set_ylabel(f'PC2 (λ={eigenvalues[1]:.3f}, {pca.explained_variance_ratio_[1]*100:.1f}% var)')
            ax4.set_title('Parameter Space Projection\n(First Two Principal Components)', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # Add ellipse showing parameter uncertainty
            from matplotlib.patches import Ellipse
            
            # Calculate 95% confidence ellipse
            cov_matrix = np.cov(transformed_data[:, 0], transformed_data[:, 1])
            eigenvals_2d, eigenvecs_2d = np.linalg.eigh(cov_matrix)
            
            # Sort eigenvalues and eigenvectors
            order = eigenvals_2d.argsort()[::-1]
            eigenvals_2d = eigenvals_2d[order]
            eigenvecs_2d = eigenvecs_2d[:, order]
            
            # Calculate ellipse parameters
            angle = np.degrees(np.arctan2(eigenvecs_2d[1, 0], eigenvecs_2d[0, 0]))
            width, height = 2 * np.sqrt(eigenvals_2d) * 2.448  # 95% confidence
            
            ellipse = Ellipse(xy=(np.mean(transformed_data[:, 0]), np.mean(transformed_data[:, 1])),
                            width=width, height=height, angle=angle,
                            facecolor='none', edgecolor='red', linewidth=2, linestyle='--',
                            label='95% Confidence')
            ax4.add_patch(ellipse)
            ax4.legend()
        
        # 5. Sloppiness analysis summary
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        # Calculate sloppiness metrics
        eigenvalue_range = eigenvalues[0] / eigenvalues[-1] if len(eigenvalues) > 1 else 1
        n_sloppy = np.sum(eigenvalues < eigenvalues[0] / 100)  # Components with <1% of max eigenvalue
        n_stiff = len(eigenvalues) - n_sloppy
        
        # Create summary text
        summary_text = f"""PCA Sloppiness Analysis
        
Parameter Identifiability:
• Total parameters: {len(self.param_cols)}
• Stiff directions: {n_stiff}
• Sloppy directions: {n_sloppy}

Eigenvalue Analysis:
• Largest eigenvalue: {eigenvalues[0]:.3e}
• Smallest eigenvalue: {eigenvalues[-1]:.3e}
• Condition number: {eigenvalue_range:.1e}

Variance Explained:
• PC1: {pca.explained_variance_ratio_[0]*100:.1f}%
• PC2: {pca.explained_variance_ratio_[1]*100:.1f}%
• First 5 PCs: {np.sum(pca.explained_variance_ratio_[:5])*100:.1f}%

Interpretation:
• Condition number > 10⁶: Highly sloppy
• Condition number > 10³: Moderately sloppy
• Condition number < 10³: Well-conditioned
"""
        
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.suptitle(f'PCA Analysis: Sloppy vs Stiff Parameter Directions\n({self.mechanism})', 
                     fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_plots:
            filename = f"pca_sloppiness_analysis_{self.mechanism}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved PCA analysis plot: {filename}")
        
        plt.show()
        
        return pca, eigenvalues, eigenvectors
    
    def plot_synthetic_vs_recovery_comparison(self, save_plots=True):
        """
        Compare synthetic data vs data generated from a randomly selected recovery parameter set.
        Shows all four strains to demonstrate recovery quality.
        
        Args:
            save_plots (bool): Whether to save plots to files
        """
        successful_df = self.get_successful_results()
        
        if len(successful_df) == 0:
            print("No successful recoveries for synthetic vs recovery comparison!")
            return
        
        # Select a random successful recovery
        random_idx = np.random.choice(len(successful_df))
        selected_recovery = successful_df.iloc[random_idx]
        
        print(f"Selected recovery run #{selected_recovery['run_id']} for comparison")
        print(f"Recovery NLL: {selected_recovery['final_nll']:.4f}")
        
        # Extract recovered parameters
        recovered_params = {}
        for param in self.param_cols:
            recovered_params[param] = selected_recovery[param]
        
        # Extract ground truth parameters
        ground_truth_params = {}
        for param in self.param_cols:
            truth_col = f"{param}_truth"
            if truth_col in self.df.columns:
                ground_truth_params[param] = self.df[truth_col].iloc[0]
        
        # Generate synthetic data using both parameter sets
        try:
            # Import necessary functions
            from simulation_utils import apply_mutant_params, run_simulation_for_dataset
            
            # Convert parameters to format expected by simulation
            def params_to_simulation_format(params_dict):
                """Convert parameter dictionary to simulation format."""
                base_params = {
                    'n1': max(params_dict['r21'] * params_dict['n2'], 1),
                    'n2': params_dict['n2'], 
                    'n3': max(params_dict['r23'] * params_dict['n2'], 1),
                    'N1': max(params_dict['R21'] * params_dict['N2'], 1),
                    'N2': params_dict['N2'],
                    'N3': max(params_dict['R23'] * params_dict['N2'], 1),
                    'k_1': params_dict['k_max'] / params_dict['tau'],
                    'k_max': params_dict['k_max'],
                    'tau': params_dict['tau']
                }
                
                # Add mechanism-specific parameters
                if 'burst_size' in params_dict:
                    base_params['burst_size'] = params_dict['burst_size']
                if 'n_inner' in params_dict:
                    base_params['n_inner'] = params_dict['n_inner']
                    
                return base_params
            
            # Generate data for both parameter sets
            ground_truth_base = params_to_simulation_format(ground_truth_params)
            recovered_base = params_to_simulation_format(recovered_params)
            
            # Strain parameters
            alpha = ground_truth_params.get('alpha', 1.0)
            beta_k = ground_truth_params.get('beta_k', 1.0) 
            beta_tau = ground_truth_params.get('beta_tau', 1.0)
            
            alpha_rec = recovered_params.get('alpha', 1.0)
            beta_k_rec = recovered_params.get('beta_k', 1.0)
            beta_tau_rec = recovered_params.get('beta_tau', 1.0)
            
            strains = ['wildtype', 'threshold', 'degrate', 'degrateAPC']
            strain_labels = ['Wild-type', 'Threshold Mutant', 'Separase Mutant', 'APC Mutant']
            
            # Colors: light yellow for synthetic, light blue for recovery
            synthetic_color = 'lightyellow'
            recovery_color = 'lightblue'
            num_sims = 500  # Number of simulations for comparison
            
            # Create separate figures for T1-T2 and T3-T2
            fig1, axes1 = plt.subplots(2, 2, figsize=(15, 12))
            axes1 = axes1.flatten()
            
            fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))
            axes2 = axes2.flatten()
            
            for i, (strain, strain_label) in enumerate(zip(strains, strain_labels)):
                # Generate synthetic data (ground truth)
                gt_params, gt_n0_list = apply_mutant_params(
                    ground_truth_base, strain, alpha, beta_k, beta_tau
                )
                
                # Use the correct mechanism name for simulation
                # The analysis may detect mechanism name incorrectly from filename
                # For now, assume time_varying_k_combined since that's what the recovery data is from
                simulation_mechanism = 'time_varying_k_combined'
                
                gt_delta_t12, gt_delta_t32 = run_simulation_for_dataset(
                    simulation_mechanism, gt_params, gt_n0_list, num_sims
                )
                
                # Generate recovery data
                rec_params, rec_n0_list = apply_mutant_params(
                    recovered_base, strain, alpha_rec, beta_k_rec, beta_tau_rec
                )
                
                rec_delta_t12, rec_delta_t32 = run_simulation_for_dataset(
                    simulation_mechanism, rec_params, rec_n0_list, num_sims
                )
                
                # Plot T1-T2 comparison (Figure 1)
                ax_t12 = axes1[i]
                if gt_delta_t12 is not None and rec_delta_t12 is not None:
                    # Plot T1-T2 comparison with new colors
                    ax_t12.hist(gt_delta_t12, bins=30, alpha=0.7, density=True, 
                               color=synthetic_color, label='Synthetic (Ground Truth)', 
                               edgecolor='black', linewidth=1)
                    ax_t12.hist(rec_delta_t12, bins=30, alpha=0.7, density=True, 
                               color=recovery_color, label='Recovery Simulation', 
                               edgecolor='darkblue', linewidth=1)
                    
                    # Add statistics
                    gt_mean_t12 = np.mean(gt_delta_t12)
                    rec_mean_t12 = np.mean(rec_delta_t12)
                    gt_std_t12 = np.std(gt_delta_t12)
                    rec_std_t12 = np.std(rec_delta_t12)
                    
                    ax_t12.axvline(gt_mean_t12, color='orange', linestyle='-', linewidth=2, alpha=0.8)
                    ax_t12.axvline(rec_mean_t12, color='darkblue', linestyle='--', linewidth=2, alpha=0.8)
                    
                    # Calculate difference metrics
                    mean_diff_t12 = abs(rec_mean_t12 - gt_mean_t12)
                    std_diff_t12 = abs(rec_std_t12 - gt_std_t12)
                    
                    ax_t12.set_title(f'{strain_label}\nT1-T2 Distribution\nΔμ={mean_diff_t12:.3f}, Δσ={std_diff_t12:.3f}', 
                                   fontweight='bold')
                    ax_t12.set_xlabel('Time Difference (T1-T2)')
                    ax_t12.set_ylabel('Density')
                    ax_t12.legend()
                    ax_t12.grid(True, alpha=0.3)
                    
                    # Add text box with parameter comparison
                    param_text = f"""Ground Truth vs Recovery:
α: {alpha:.3f} vs {alpha_rec:.3f}
β_k: {beta_k:.3f} vs {beta_k_rec:.3f}
β_τ: {beta_tau:.3f} vs {beta_tau_rec:.3f}"""
                    
                    ax_t12.text(0.02, 0.98, param_text, transform=ax_t12.transAxes, 
                               fontsize=8, verticalalignment='top',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
                else:
                    ax_t12.text(0.5, 0.5, f'T1-T2 Simulation failed for {strain_label}', 
                               ha='center', va='center', transform=ax_t12.transAxes,
                               fontsize=12, color='red')
                    ax_t12.set_title(f'{strain_label} - T1-T2 Simulation Failed')
                
                # Plot T3-T2 comparison (Figure 2)
                ax_t32 = axes2[i]
                if gt_delta_t32 is not None and rec_delta_t32 is not None:
                    # Plot T3-T2 comparison with new colors
                    ax_t32.hist(gt_delta_t32, bins=30, alpha=0.7, density=True, 
                               color=synthetic_color, label='Synthetic (Ground Truth)', 
                               edgecolor='black', linewidth=1)
                    ax_t32.hist(rec_delta_t32, bins=30, alpha=0.7, density=True, 
                               color=recovery_color, label='Recovery Simulation', 
                               edgecolor='darkblue', linewidth=1)
                    
                    # Add statistics
                    gt_mean_t32 = np.mean(gt_delta_t32)
                    rec_mean_t32 = np.mean(rec_delta_t32)
                    gt_std_t32 = np.std(gt_delta_t32)
                    rec_std_t32 = np.std(rec_delta_t32)
                    
                    ax_t32.axvline(gt_mean_t32, color='orange', linestyle='-', linewidth=2, alpha=0.8)
                    ax_t32.axvline(rec_mean_t32, color='darkblue', linestyle='--', linewidth=2, alpha=0.8)
                    
                    # Calculate difference metrics
                    mean_diff_t32 = abs(rec_mean_t32 - gt_mean_t32)
                    std_diff_t32 = abs(rec_std_t32 - gt_std_t32)
                    
                    ax_t32.set_title(f'{strain_label}\nT3-T2 Distribution\nΔμ={mean_diff_t32:.3f}, Δσ={std_diff_t32:.3f}', 
                                   fontweight='bold')
                    ax_t32.set_xlabel('Time Difference (T3-T2)')
                    ax_t32.set_ylabel('Density')
                    ax_t32.legend()
                    ax_t32.grid(True, alpha=0.3)
                    
                    # Add text box with parameter comparison
                    ax_t32.text(0.02, 0.98, param_text, transform=ax_t32.transAxes, 
                               fontsize=8, verticalalignment='top',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
                else:
                    ax_t32.text(0.5, 0.5, f'T3-T2 Simulation failed for {strain_label}', 
                               ha='center', va='center', transform=ax_t32.transAxes,
                               fontsize=12, color='red')
                    ax_t32.set_title(f'{strain_label} - T3-T2 Simulation Failed')
            
            # Finalize Figure 1 (T1-T2)
            fig1.suptitle(f'Synthetic vs Recovery Data Comparison - T1-T2 Distributions\n'
                         f'Recovery Run #{selected_recovery["run_id"]} (NLL: {selected_recovery["final_nll"]:.4f})', 
                         fontsize=16, fontweight='bold')
            fig1.tight_layout()
            
            # Finalize Figure 2 (T3-T2)
            fig2.suptitle(f'Synthetic vs Recovery Data Comparison - T3-T2 Distributions\n'
                         f'Recovery Run #{selected_recovery["run_id"]} (NLL: {selected_recovery["final_nll"]:.4f})', 
                         fontsize=16, fontweight='bold')
            fig2.tight_layout()
            
            if save_plots:
                filename1 = f"synthetic_vs_recovery_T1T2_{self.mechanism}.png"
                filename2 = f"synthetic_vs_recovery_T3T2_{self.mechanism}.png"
                
                fig1.savefig(filename1, dpi=300, bbox_inches='tight')
                fig2.savefig(filename2, dpi=300, bbox_inches='tight')
                
                print(f"Saved T1-T2 comparison: {filename1}")
                print(f"Saved T3-T2 comparison: {filename2}")
            
            plt.show(fig1)
            plt.show(fig2)
            
        except ImportError:
            print("Warning: Could not import simulation_utils. Skipping synthetic vs recovery comparison.")
            print("Make sure simulation_utils.py is available in the current directory.")
        except Exception as e:
            print(f"Error generating synthetic vs recovery comparison: {e}")
            print("This comparison requires the simulation infrastructure to be available.")
    
    def plot_parameter_recovery(self, save_plots=True):
        """
        Create comprehensive parameter recovery plots.
        
        Args:
            save_plots (bool): Whether to save plots to files
        """
        successful_df = self.get_successful_results()
        
        if len(successful_df) == 0:
            print("No successful recoveries to plot!")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        if HAS_SEABORN:
            sns.set_palette("husl")
        
        n_params = len(self.param_cols)
        n_cols = min(4, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        # 1. Parameter recovery scatter plots
        fig1, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, param in enumerate(self.param_cols):
            truth_col = f"{param}_truth"
            if truth_col not in self.df.columns:
                continue
            
            ax = axes[i]
            truth_value = self.df[truth_col].iloc[0]
            recovered_values = successful_df[param]
            
            # Create histogram
            ax.hist(recovered_values, bins=20, alpha=0.7, density=True, edgecolor='black')
            ax.axvline(truth_value, color='red', linestyle='--', linewidth=2, label='Ground Truth')
            ax.axvline(recovered_values.mean(), color='blue', linestyle='-', linewidth=2, label='Mean Recovered')
            
            # Add statistics text
            mean_val = recovered_values.mean()
            std_val = recovered_values.std()
            rel_error = abs(mean_val - truth_value) / abs(truth_value) * 100
            
            ax.set_xlabel(param)
            ax.set_ylabel('Density')
            ax.set_title(f'{param}\nTruth: {truth_value:.4f}, Mean: {mean_val:.4f}\nRel. Error: {rel_error:.1f}%')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if save_plots:
            filename = f"parameter_recovery_distributions_{self.mechanism}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved distribution plot: {filename}")
        plt.show()
        
        # 2. Parameter correlation heatmap
        if n_params > 1:
            fig2, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Calculate correlation matrix for recovered parameters
            recovery_data = successful_df[self.param_cols]
            corr_matrix = recovery_data.corr()
            
            # Create heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            if HAS_SEABORN:
                sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                           square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
            else:
                # Fallback to matplotlib imshow
                masked_corr = np.ma.masked_where(mask, corr_matrix)
                im = ax.imshow(masked_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='equal')
                
                # Add correlation values as text
                for i in range(len(self.param_cols)):
                    for j in range(len(self.param_cols)):
                        if not mask[i, j]:
                            ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                                   ha='center', va='center', fontsize=8)
                
                ax.set_xticks(range(len(self.param_cols)))
                ax.set_yticks(range(len(self.param_cols)))
                ax.set_xticklabels(self.param_cols, rotation=45, ha='right')
                ax.set_yticklabels(self.param_cols)
                plt.colorbar(im, ax=ax, shrink=0.8)
            
            ax.set_title(f'Parameter Correlation Matrix\n({self.mechanism})', fontsize=14)
            plt.tight_layout()
            
            if save_plots:
                filename = f"parameter_correlations_{self.mechanism}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved correlation plot: {filename}")
            plt.show()
        
        # 3. NLL distribution and convergence analysis
        fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # NLL distribution
        nll_values = successful_df['final_nll']
        ax1.hist(nll_values, bins=20, alpha=0.7, edgecolor='black')
        ax1.axvline(nll_values.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {nll_values.mean():.2f}')
        ax1.axvline(nll_values.min(), color='blue', linestyle='--', linewidth=2,
                   label=f'Best: {nll_values.min():.2f}')
        
        ax1.set_xlabel('Final NLL')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Distribution of Final NLL Values\n({self.mechanism})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Convergence rate analysis
        total_runs = len(self.df)
        converged_runs = len(successful_df)
        convergence_rate = converged_runs / total_runs * 100
        
        ax2.bar(['Failed', 'Converged'], [total_runs - converged_runs, converged_runs], 
               color=['red', 'green'], alpha=0.7)
        ax2.set_ylabel('Number of Runs')
        ax2.set_title(f'Convergence Success Rate\n{convergence_rate:.1f}% ({converged_runs}/{total_runs})')
        ax2.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for i, v in enumerate([total_runs - converged_runs, converged_runs]):
            ax2.text(i, v + 0.5, f'{v}\n({v/total_runs*100:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        if save_plots:
            filename = f"nll_convergence_analysis_{self.mechanism}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved NLL/convergence analysis: {filename}")
        plt.show()
        
        # 4. Parameter recovery error summary
        stats_df = self.calculate_recovery_statistics()
        if not stats_df.empty:
            fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Relative error plot
            colors = ['green' if x <= 10 else 'orange' if x <= 25 else 'red' 
                     for x in stats_df['mean_rel_error_pct']]
            
            bars1 = ax1.bar(range(len(stats_df)), stats_df['mean_rel_error_pct'], 
                           yerr=stats_df['std_rel_error_pct'], capsize=5, alpha=0.7, color=colors)
            ax1.set_xlabel('Parameter')
            ax1.set_ylabel('Mean Relative Error (%)')
            ax1.set_title('Parameter Recovery Error')
            ax1.set_xticks(range(len(stats_df)))
            ax1.set_xticklabels(stats_df['parameter'], rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # Add horizontal reference lines
            ax1.axhline(10, color='orange', linestyle='--', alpha=0.7, label='10% (Good)')
            ax1.axhline(25, color='red', linestyle='--', alpha=0.7, label='25% (Fair)')
            ax1.legend()
            
            # Coefficient of variation (sloppiness measure)
            cv_colors = ['green' if x <= 0.1 else 'orange' if x <= 0.5 else 'red' 
                        for x in stats_df['cv_recovered']]
            
            bars2 = ax2.bar(range(len(stats_df)), stats_df['cv_recovered'], alpha=0.7, color=cv_colors)
            ax2.set_xlabel('Parameter')
            ax2.set_ylabel('Coefficient of Variation')
            ax2.set_title('Coefficient of Variation')
            ax2.set_xticks(range(len(stats_df)))
            ax2.set_xticklabels(stats_df['parameter'], rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Add horizontal reference lines
            # ax2.axhline(0.1, color='orange', linestyle='--', alpha=0.7, label='0.1 (Low sloppiness)')
            # ax2.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='0.5 (High sloppiness)')
            ax2.legend()
            
            plt.tight_layout()
            if save_plots:
                filename = f"recovery_error_summary_{self.mechanism}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved recovery summary: {filename}")
            plt.show()
    
    def generate_report(self, save_report=True):
        """
        Generate a comprehensive text report of the recovery study.
        
        Args:
            save_report (bool): Whether to save report to file
        """
        successful_df = self.get_successful_results()
        stats_df = self.calculate_recovery_statistics()
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PARAMETER RECOVERY STUDY ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Mechanism: {self.mechanism}")
        report_lines.append(f"Results file: {self.results_file}")
        report_lines.append("")
        
        # Overall statistics
        report_lines.append("OVERALL RESULTS:")
        report_lines.append("-" * 40)
        report_lines.append(f"Total recovery attempts: {len(self.df)}")
        report_lines.append(f"Successful recoveries: {len(successful_df)} ({100*len(successful_df)/len(self.df):.1f}%)")
        
        if len(successful_df) > 0:
            report_lines.append(f"Best final NLL: {successful_df['final_nll'].min():.4f}")
            report_lines.append(f"Mean final NLL: {successful_df['final_nll'].mean():.4f} ± {successful_df['final_nll'].std():.4f}")
            report_lines.append(f"Mean runtime per recovery: {successful_df['elapsed_time'].mean():.1f} ± {successful_df['elapsed_time'].std():.1f} seconds")
        report_lines.append("")
        
        # Parameter-by-parameter analysis
        if not stats_df.empty:
            report_lines.append("PARAMETER RECOVERY ANALYSIS:")
            report_lines.append("-" * 40)
            
            for _, row in stats_df.iterrows():
                param = row['parameter']
                report_lines.append(f"\n{param.upper()}:")
                report_lines.append(f"  Ground truth value: {row['truth_value']:.6f}")
                report_lines.append(f"  Mean recovered: {row['mean_recovered']:.6f} ± {row['std_recovered']:.6f}")
                report_lines.append(f"  Recovery range: [{row['min_recovered']:.6f}, {row['max_recovered']:.6f}]")
                report_lines.append(f"  Mean relative error: {row['mean_rel_error_pct']:.2f}% ± {row['std_rel_error_pct']:.2f}%")
                report_lines.append(f"  Good recovery rate (≤10% error): {row['good_recovery_rate']*100:.1f}%")
                report_lines.append(f"  Coefficient of variation: {row['cv_recovered']:.4f}")
                
                # Interpretations
                if row['mean_rel_error_pct'] <= 5:
                    report_lines.append(f"  → EXCELLENT recovery (≤5% error)")
                elif row['mean_rel_error_pct'] <= 10:
                    report_lines.append(f"  → GOOD recovery (≤10% error)")
                elif row['mean_rel_error_pct'] <= 25:
                    report_lines.append(f"  → FAIR recovery (≤25% error)")
                else:
                    report_lines.append(f"  → POOR recovery (>25% error)")
                
                if row['cv_recovered'] <= 0.1:
                    report_lines.append(f"  → LOW sloppiness (CV ≤ 0.1)")
                elif row['cv_recovered'] <= 0.5:
                    report_lines.append(f"  → MODERATE sloppiness (CV ≤ 0.5)")
                else:
                    report_lines.append(f"  → HIGH sloppiness (CV > 0.5)")
        
        # PCA Sloppiness Analysis
        report_lines.append("\n")
        report_lines.append("PCA SLOPPINESS ANALYSIS:")
        report_lines.append("-" * 40)
        
        pca_results = self.perform_pca_analysis()
        if pca_results[0] is not None:
            pca, scaler, transformed_data, eigenvalues, eigenvectors = pca_results
            
            # Calculate sloppiness metrics
            eigenvalue_range = eigenvalues[0] / eigenvalues[-1] if len(eigenvalues) > 1 else 1
            n_sloppy = np.sum(eigenvalues < eigenvalues[0] / 100)  # Components with <1% of max eigenvalue
            n_stiff = len(eigenvalues) - n_sloppy
            
            report_lines.append(f"Eigenvalue spectrum analysis:")
            report_lines.append(f"  Largest eigenvalue: {eigenvalues[0]:.3e}")
            report_lines.append(f"  Smallest eigenvalue: {eigenvalues[-1]:.3e}")
            report_lines.append(f"  Condition number (λ_max/λ_min): {eigenvalue_range:.1e}")
            report_lines.append(f"  Stiff parameter directions: {n_stiff}")
            report_lines.append(f"  Sloppy parameter directions: {n_sloppy}")
            
            # Variance explained by first few components
            cumulative_var = np.cumsum(pca.explained_variance_ratio_)
            report_lines.append(f"\nVariance explained:")
            report_lines.append(f"  First 3 components: {cumulative_var[2]*100:.1f}%")
            report_lines.append(f"  First 5 components: {cumulative_var[4]*100:.1f}%" if len(cumulative_var) > 4 else "")
            
            # Sloppiness interpretation
            if eigenvalue_range > 1e6:
                sloppiness_level = "HIGHLY SLOPPY"
                sloppiness_desc = "Model is highly overparameterized with many unidentifiable parameter combinations"
            elif eigenvalue_range > 1e3:
                sloppiness_level = "MODERATELY SLOPPY"
                sloppiness_desc = "Model shows some parameter sloppiness but is reasonably well-constrained"
            else:
                sloppiness_level = "WELL-CONDITIONED"
                sloppiness_desc = "Model parameters are well-constrained by the data"
            
            report_lines.append(f"\nSloppiness assessment: {sloppiness_level}")
            report_lines.append(f"  {sloppiness_desc}")
            
            # Identify most important parameter combinations
            if len(eigenvectors) > 0:
                # Find parameters with highest loadings in first (stiffest) component
                first_pc_loadings = np.abs(eigenvectors[0])
                top_param_indices = np.argsort(first_pc_loadings)[-3:]  # Top 3
                top_params = [self.param_cols[i] for i in reversed(top_param_indices)]
                
                report_lines.append(f"\nMost constrained parameter combination (PC1):")
                report_lines.append(f"  Primary parameters: {', '.join(top_params)}")
                
                # Find parameters with highest loadings in last (sloppiest) component if it exists
                if len(eigenvalues) > 1:
                    last_pc_loadings = np.abs(eigenvectors[-1])
                    sloppy_param_indices = np.argsort(last_pc_loadings)[-3:]  # Top 3
                    sloppy_params = [self.param_cols[i] for i in reversed(sloppy_param_indices)]
                    
                    report_lines.append(f"\nMost sloppy parameter combination (PC{len(eigenvalues)}):")
                    report_lines.append(f"  Primary parameters: {', '.join(sloppy_params)}")
        else:
            report_lines.append("PCA analysis not available - insufficient successful recoveries")

        # Model identifiability assessment
        report_lines.append("\n")
        report_lines.append("MODEL IDENTIFIABILITY ASSESSMENT:")
        report_lines.append("-" * 40)
        
        if stats_df.empty:
            report_lines.append("Cannot assess identifiability - no successful recoveries.")
        else:
            excellent_params = (stats_df['mean_rel_error_pct'] <= 5).sum()
            good_params = (stats_df['mean_rel_error_pct'] <= 10).sum()
            total_params = len(stats_df)
            
            report_lines.append(f"Parameters with excellent recovery (≤5% error): {excellent_params}/{total_params} ({100*excellent_params/total_params:.1f}%)")
            report_lines.append(f"Parameters with good recovery (≤10% error): {good_params}/{total_params} ({100*good_params/total_params:.1f}%)")
            
            low_sloppy = (stats_df['cv_recovered'] <= 0.1).sum()
            high_sloppy = (stats_df['cv_recovered'] > 0.5).sum()
            
            report_lines.append(f"Parameters with low sloppiness (CV ≤ 0.1): {low_sloppy}/{total_params} ({100*low_sloppy/total_params:.1f}%)")
            report_lines.append(f"Parameters with high sloppiness (CV > 0.5): {high_sloppy}/{total_params} ({100*high_sloppy/total_params:.1f}%)")
            
            # Overall assessment
            if good_params >= 0.8 * total_params and low_sloppy >= 0.6 * total_params:
                assessment = "GOOD - Most parameters are well-identified"
            elif good_params >= 0.6 * total_params:
                assessment = "MODERATE - Some parameters show sloppiness"
            else:
                assessment = "POOR - Many parameters are poorly identified"
            
            report_lines.append(f"\nOverall model identifiability: {assessment}")
        
        # Recommendations
        report_lines.append("\n")
        report_lines.append("RECOMMENDATIONS:")
        report_lines.append("-" * 40)
        
        if len(successful_df) == 0:
            report_lines.append("• No successful recoveries - check optimization settings or parameter bounds")
        elif len(successful_df) < 0.5 * len(self.df):
            report_lines.append("• Low success rate - consider increasing max_iterations or checking bounds")
        
        if not stats_df.empty:
            poor_params = stats_df[stats_df['mean_rel_error_pct'] > 25]['parameter'].tolist()
            if poor_params:
                report_lines.append(f"• Poor recovery for: {', '.join(poor_params)} - consider parameter constraints or model reparameterization")
            
            sloppy_params = stats_df[stats_df['cv_recovered'] > 0.5]['parameter'].tolist()
            if sloppy_params:
                report_lines.append(f"• High sloppiness for: {', '.join(sloppy_params)} - parameters may be practically unidentifiable")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Print report
        report_text = "\n".join(report_lines)
        print(report_text)
        
        # Save report
        if save_report:
            report_filename = f"recovery_analysis_report_{self.mechanism}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_filename, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved to: {report_filename}")
        
        return report_text

def main():
    """
    Main function to analyze parameter recovery results.
    """
    # Default settings for direct execution
    results_file = 'recovery_results.csv'  # Auto-detect most recent file
    show_plots = True
    save_files = True
    
    try:
        print("PARAMETER RECOVERY ANALYSIS")
        print("=" * 50)
        print("Auto-detecting most recent recovery results file...")
        
        # Initialize analyzer (will find most recent results file automatically)
        analyzer = ParameterRecoveryAnalyzer(results_file)
        
        print("\nGenerating recovery statistics...")
        stats_df = analyzer.calculate_recovery_statistics()
        
        if show_plots:
            print("\nCreating visualization plots...")
            analyzer.plot_parameter_recovery(save_plots=save_files)
            
            print("\nPerforming PCA sloppiness analysis...")
            analyzer.plot_pca_analysis(save_plots=save_files)
            
            print("\nGenerating synthetic vs recovery data comparison...")
            analyzer.plot_synthetic_vs_recovery_comparison(save_plots=save_files)
        
        print("\nGenerating comprehensive analysis report...")
        analyzer.generate_report(save_report=save_files)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        
        if save_files:
            print("\nGenerated files:")
            print("• Parameter recovery distribution plots (PNG)")
            print("• Parameter correlation matrix (PNG)")
            print("• NLL and convergence analysis (PNG)")  
            print("• Recovery error summary plots (PNG)")
            print("• PCA sloppiness analysis plot (PNG)")
            print("• Synthetic vs recovery data comparison (PNG)")
            print("• Comprehensive analysis report (TXT)")
        
        print(f"\nNext steps:")
        print("1. Review the plots and report for parameter identifiability insights")
        print("2. Identify which parameters are well-constrained vs. sloppy")
        print("3. Consider model reparameterization for poorly identified parameters")
        print("4. Use insights to improve your optimization strategy")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
