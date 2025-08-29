#!/usr/bin/env python3
"""
Analysis and Visualization Script for MoM Parameter Recovery Study

This script analyzes the results from parameter_recovery_mom.py to:
1. Visualize parameter recovery accuracy for MoM approach
2. Compare MoM vs simulation-based parameter identifiability  
3. Assess model sloppiness using Method of Moments
4. Generate comprehensive recovery reports

Usage:
    python analyze_parameter_recovery_mom.py [results_file.csv]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import glob
import argparse
from datetime import datetime

class MoMParameterRecoveryAnalyzer:
    """
    Analyzes MoM parameter recovery study results.
    """
    
    def __init__(self, results_file=None):
        """
        Initialize analyzer.
        
        Args:
            results_file (str, optional): CSV file with MoM recovery results. If None, finds most recent.
        """
        if results_file is None:
            # Find most recent MoM recovery results file
            pattern = "mom_parameter_recovery_*.csv"
            files = glob.glob(pattern)
            if not files:
                raise FileNotFoundError(
                    "No MoM parameter recovery results found. Please run parameter_recovery_mom.py first.\n"
                    "Expected filename pattern: mom_parameter_recovery_*.csv"
                )
            results_file = max(files, key=os.path.getmtime)
            print(f"Using MoM results file: {results_file}")
        
        self.results_file = results_file
        self.df = pd.read_csv(results_file)
        
        # Extract mechanism name from filename
        if 'fixed_burst_feedback_onion' in results_file:
            self.mechanism = 'fixed_burst_feedback_onion'
        elif 'feedback_onion' in results_file:
            self.mechanism = 'feedback_onion'
        elif 'fixed_burst' in results_file:
            self.mechanism = 'fixed_burst'
        else:
            self.mechanism = results_file.split('_')[3] if '_' in results_file else 'unknown'
        
        # Identify parameter columns (exclude metadata columns)
        metadata_cols = ['run_id', 'converged', 'final_nll', 'elapsed_time']
        self.param_cols = [col for col in self.df.columns 
                          if col not in metadata_cols and not col.endswith('_truth')]
        
        # Identify ground truth columns
        self.truth_cols = [col for col in self.df.columns if col.endswith('_truth')]
        
        print(f"Loaded MoM recovery results:")
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
            print("Warning: No successful MoM recoveries found!")
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
            good_recoveries = relative_errors <= 10
            stats_dict['good_recovery_rate'] = good_recoveries.mean()
            
            # Coefficient of variation (measure of sloppiness)
            if np.abs(stats_dict['mean_recovered']) > 1e-10:
                stats_dict['cv_recovered'] = stats_dict['std_recovered'] / np.abs(stats_dict['mean_recovered'])
            else:
                stats_dict['cv_recovered'] = np.inf
            
            stats_list.append(stats_dict)
        
        return pd.DataFrame(stats_list)
    
    def plot_parameter_recovery(self, save_plots=True):
        """
        Create comprehensive parameter recovery plots for MoM approach.
        
        Args:
            save_plots (bool): Whether to save plots to files
        """
        successful_df = self.get_successful_results()
        
        if len(successful_df) == 0:
            print("No successful MoM recoveries to plot!")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        n_params = len(self.param_cols)
        n_cols = min(4, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        # 1. Parameter recovery distribution plots
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
            ax.hist(recovered_values, bins=15, alpha=0.7, density=True, edgecolor='black', color='lightblue')
            ax.axvline(truth_value, color='red', linestyle='--', linewidth=2, label='Ground Truth')
            ax.axvline(recovered_values.mean(), color='blue', linestyle='-', linewidth=2, label='Mean Recovered')
            
            # Add statistics text
            mean_val = recovered_values.mean()
            std_val = recovered_values.std()
            rel_error = abs(mean_val - truth_value) / abs(truth_value) * 100
            
            ax.set_xlabel(param)
            ax.set_ylabel('Density')
            ax.set_title(f'{param} (MoM)\nTruth: {truth_value:.4f}, Mean: {mean_val:.4f}\nRel. Error: {rel_error:.1f}%')
            ax.legend(fontsize='small')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if save_plots:
            filename = f"mom_parameter_recovery_distributions_{self.mechanism}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved MoM distribution plot: {filename}")
        plt.show()
        
        # 2. Parameter correlation heatmap
        if n_params > 1:
            fig2, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Calculate correlation matrix for recovered parameters
            recovery_data = successful_df[self.param_cols]
            corr_matrix = recovery_data.corr()
            
            # Create heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
            
            ax.set_title(f'MoM Parameter Correlation Matrix\n({self.mechanism})', fontsize=14)
            plt.tight_layout()
            
            if save_plots:
                filename = f"mom_parameter_correlations_{self.mechanism}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved MoM correlation plot: {filename}")
            plt.show()
        
        # 3. NLL distribution and convergence analysis
        fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # NLL distribution
        nll_values = successful_df['final_nll']
        ax1.hist(nll_values, bins=15, alpha=0.7, edgecolor='black', color='lightgreen')
        ax1.axvline(nll_values.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {nll_values.mean():.2f}')
        ax1.axvline(nll_values.min(), color='blue', linestyle='--', linewidth=2,
                   label=f'Best: {nll_values.min():.2f}')
        
        ax1.set_xlabel('Final NLL')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'MoM NLL Distribution\n({self.mechanism})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Convergence rate analysis
        total_runs = len(self.df)
        converged_runs = len(successful_df)
        convergence_rate = converged_runs / total_runs * 100
        
        ax2.bar(['Failed', 'Converged'], [total_runs - converged_runs, converged_runs], 
               color=['red', 'green'], alpha=0.7)
        ax2.set_ylabel('Number of Runs')
        ax2.set_title(f'MoM Convergence Success Rate\n{convergence_rate:.1f}% ({converged_runs}/{total_runs})')
        ax2.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for i, v in enumerate([total_runs - converged_runs, converged_runs]):
            ax2.text(i, v + 0.5, f'{v}\n({v/total_runs*100:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        if save_plots:
            filename = f"mom_nll_convergence_analysis_{self.mechanism}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved MoM NLL/convergence analysis: {filename}")
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
            ax1.set_title('MoM Parameter Recovery Error')
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
            ax2.set_title('MoM Parameter Sloppiness (CV)')
            ax2.set_xticks(range(len(stats_df)))
            ax2.set_xticklabels(stats_df['parameter'], rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Add horizontal reference lines
            ax2.axhline(0.1, color='orange', linestyle='--', alpha=0.7, label='0.1 (Low sloppiness)')
            ax2.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='0.5 (High sloppiness)')
            ax2.legend()
            
            plt.tight_layout()
            if save_plots:
                filename = f"mom_recovery_error_summary_{self.mechanism}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved MoM recovery summary: {filename}")
            plt.show()
    
    def generate_report(self, save_report=True):
        """
        Generate a comprehensive text report of the MoM recovery study.
        
        Args:
            save_report (bool): Whether to save report to file
        """
        successful_df = self.get_successful_results()
        stats_df = self.calculate_recovery_statistics()
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MoM PARAMETER RECOVERY STUDY ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Mechanism: {self.mechanism}")
        report_lines.append(f"Method: Method of Moments (MoM)")
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
            report_lines.append("MoM PARAMETER RECOVERY ANALYSIS:")
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
                    report_lines.append(f"  → EXCELLENT MoM recovery (≤5% error)")
                elif row['mean_rel_error_pct'] <= 10:
                    report_lines.append(f"  → GOOD MoM recovery (≤10% error)")
                elif row['mean_rel_error_pct'] <= 25:
                    report_lines.append(f"  → FAIR MoM recovery (≤25% error)")
                else:
                    report_lines.append(f"  → POOR MoM recovery (>25% error)")
                
                if row['cv_recovered'] <= 0.1:
                    report_lines.append(f"  → LOW sloppiness in MoM (CV ≤ 0.1)")
                elif row['cv_recovered'] <= 0.5:
                    report_lines.append(f"  → MODERATE sloppiness in MoM (CV ≤ 0.5)")
                else:
                    report_lines.append(f"  → HIGH sloppiness in MoM (CV > 0.5)")
        
        # Model identifiability assessment
        report_lines.append("\n")
        report_lines.append("MoM MODEL IDENTIFIABILITY ASSESSMENT:")
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
                assessment = "GOOD - Most parameters are well-identified using MoM"
            elif good_params >= 0.6 * total_params:
                assessment = "MODERATE - Some parameters show sloppiness in MoM"
            else:
                assessment = "POOR - Many parameters are poorly identified using MoM"
            
            report_lines.append(f"\nOverall MoM model identifiability: {assessment}")
        
        # Recommendations
        report_lines.append("\n")
        report_lines.append("MoM-SPECIFIC RECOMMENDATIONS:")
        report_lines.append("-" * 40)
        
        if len(successful_df) == 0:
            report_lines.append("• No successful MoM recoveries - check theoretical PDF calculations")
        elif len(successful_df) < 0.5 * len(self.df):
            report_lines.append("• Low MoM success rate - consider refining MoM approximations")
        
        if not stats_df.empty:
            poor_params = stats_df[stats_df['mean_rel_error_pct'] > 25]['parameter'].tolist()
            if poor_params:
                report_lines.append(f"• Poor MoM recovery for: {', '.join(poor_params)} - MoM may not capture these parameters well")
            
            sloppy_params = stats_df[stats_df['cv_recovered'] > 0.5]['parameter'].tolist()
            if sloppy_params:
                report_lines.append(f"• High MoM sloppiness for: {', '.join(sloppy_params)} - parameters may be poorly constrained by moments")
        
        report_lines.append("• Compare with simulation-based recovery results to assess MoM approximation quality")
        report_lines.append("• Consider hybrid approaches combining MoM and simulation for better parameter estimation")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Print report
        report_text = "\n".join(report_lines)
        print(report_text)
        
        # Save report
        if save_report:
            report_filename = f"mom_recovery_analysis_report_{self.mechanism}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_filename, 'w') as f:
                f.write(report_text)
            print(f"\nMoM report saved to: {report_filename}")
        
        return report_text

def main():
    """
    Main function to analyze MoM parameter recovery results.
    """
    parser = argparse.ArgumentParser(description="Analyze MoM parameter recovery study results")
    parser.add_argument('results_file', nargs='?', default=None,
                       help='CSV file with MoM recovery results (auto-detect if not specified)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots (report only)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save plots and reports to files')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = MoMParameterRecoveryAnalyzer(args.results_file)
        
        print("\nGenerating MoM recovery statistics...")
        stats_df = analyzer.calculate_recovery_statistics()
        
        if not args.no_plots:
            print("\nCreating MoM visualization plots...")
            analyzer.plot_parameter_recovery(save_plots=not args.no_save)
        
        print("\nGenerating comprehensive MoM analysis report...")
        analyzer.generate_report(save_report=not args.no_save)
        
        print("\n" + "="*60)
        print("MoM ANALYSIS COMPLETE!")
        print("="*60)
        
        if not args.no_save:
            print("\nGenerated MoM analysis files:")
            print("• MoM parameter recovery distribution plots (PNG)")
            print("• MoM parameter correlation matrix (PNG)")
            print("• MoM NLL and convergence analysis (PNG)")  
            print("• MoM recovery error summary plots (PNG)")
            print("• Comprehensive MoM analysis report (TXT)")
        
        print(f"\nNext steps:")
        print("1. Compare MoM results with simulation-based recovery from ARC")
        print("2. Identify parameters where MoM approximation is accurate vs. inaccurate")
        print("3. Consider which approach (MoM vs simulation) better constrains each parameter")
        print("4. Use insights to improve parameter estimation strategy")
        
    except Exception as e:
        print(f"Error in MoM analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
