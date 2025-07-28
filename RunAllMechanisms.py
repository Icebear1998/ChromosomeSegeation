#!/usr/bin/env python3
"""
Comprehensive Mechanism Analysis Script

This script runs all implemented mechanisms with both joint and independent optimization strategies,
organizes results in structured folders, generates comparison plots, and performs AIC/BIC model selection.

Author: Chromosome Segregation Modeling Project
Date: 2024
"""

import os
import sys
import time
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess
import json

# Add current directory to path for imports
sys.path.append('.')

from TestDataPlot import load_parameters, plot_all_datasets_2x2, load_dataset, apply_mutant_params, extract_mechanism_params
from MoMCalculations import compute_pdf_for_mechanism


class MechanismRunner:
    """Class to manage running all mechanisms and organizing results."""
    
    def __init__(self, base_results_dir="ResultsAllRun"):
        self.base_results_dir = base_results_dir
        self.mechanisms = ['simple', 'fixed_burst', 'time_varying_k', 'feedback_onion', 'fixed_burst_feedback_onion']
        self.strategies = ['join', 'independent']
        self.datasets = ['wildtype', 'initial', 'threshold', 'degrate']  # Ordered as requested
        self.results_summary = []
        
        # Create base results directory
        if os.path.exists(self.base_results_dir):
            shutil.rmtree(self.base_results_dir)
        os.makedirs(self.base_results_dir)
        
        print(f"Created results directory: {self.base_results_dir}")
        
    def run_single_mechanism(self, mechanism, strategy):
        """Run a single mechanism with specified strategy."""
        print(f"\n{'='*60}")
        print(f"Running {mechanism} with {strategy} optimization")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Modify the optimization files to use the current mechanism
            if strategy == 'join':
                self.modify_optimization_file('MoMOptimization_join.py', mechanism)
                param_file = f"optimized_parameters_{mechanism}_join.txt"
            else:  # independent
                self.modify_optimization_file('MoMOptimization_independent.py', mechanism)
                param_file = f"optimized_parameters_{mechanism}_independent.txt"
            
            # Run the optimization with timeout protection
            try:
                if strategy == 'join':
                    result = subprocess.run([sys.executable, 'MoMOptimization_join.py'], 
                                          capture_output=True, text=True, timeout=3600)  # 1 hour timeout
                else:  # independent
                    result = subprocess.run([sys.executable, 'MoMOptimization_independent.py'], 
                                          capture_output=True, text=True, timeout=3600)  # 1 hour timeout
                
                end_time = time.time()
                runtime = end_time - start_time
                
                # Check if optimization was successful
                if result.returncode == 0 and os.path.exists(param_file):
                    print(f"✅ Successfully completed {mechanism} ({strategy}) in {runtime:.1f} seconds")
                    return param_file, runtime, True
                else:
                    print(f"❌ Failed to run {mechanism} ({strategy}) - Return code: {result.returncode}")
                    if result.stderr:
                        print(f"Error output: {result.stderr[:500]}...")  # Limit error output length
                    if result.stdout:
                        print(f"Standard output: {result.stdout[-500:]}")  # Show last 500 chars of stdout
                    return None, runtime, False
                    
            except subprocess.TimeoutExpired:
                end_time = time.time()
                runtime = end_time - start_time
                print(f"⏰ Timeout: {mechanism} ({strategy}) exceeded 1 hour limit")
                return None, runtime, False
                
            except subprocess.SubprocessError as e:
                end_time = time.time()
                runtime = end_time - start_time
                print(f"❌ Subprocess error for {mechanism} ({strategy}): {e}")
                return None, runtime, False
                
        except Exception as e:
            end_time = time.time()
            runtime = end_time - start_time
            print(f"❌ Unexpected error running {mechanism} ({strategy}): {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None, runtime, False
    
    def modify_optimization_file(self, filename, mechanism):
        """Modify optimization file to use specified mechanism."""
        # Read the file
        with open(filename, 'r') as f:
            content = f.read()
        
        # Replace mechanism line
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('mechanism = '):
                lines[i] = f"    mechanism = '{mechanism}'  # Auto-set by RunAllMechanisms.py"
                break
        
        # Write back
        with open(filename, 'w') as f:
            f.write('\n'.join(lines))
    
    def create_custom_plot(self, params, mechanism, output_path, num_sim=1000):
        """Create custom 2x4 plot with each strain's Chrom1-2 and Chrom3-2 side by side."""
        print(f"Generating plots for {mechanism}...")
        
        # Load experimental data
        df = pd.read_excel("Data/All_strains_SCStimes.xlsx")
        
        # Create 2x4 subplot layout (2 rows, 4 columns)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Chromosome Segregation Times - {mechanism.replace("_", " ").title()} Mechanism', 
                     fontsize=16, y=0.98)  # Moved title higher to avoid overlap
        
        # Set up x_grid for PDF plotting (matching TestDataPlot.py)
        x_min, x_max = -140, 140  # Default range from TestDataPlot.py
        x_grid = np.linspace(x_min, x_max, 401)
        
        # Dataset arrangement: 
        # Row 1: wildtype12, wildtype32, initial12, initial32
        # Row 2: threshold12, threshold32, degrate12, degrate32
        plot_config = [
            # Row 1
            ('wildtype', 'Chrom1-2', 0, 0),
            ('wildtype', 'Chrom3-2', 0, 1), 
            ('initial', 'Chrom1-2', 0, 2),
            ('initial', 'Chrom3-2', 0, 3),
            # Row 2  
            ('threshold', 'Chrom1-2', 1, 0),
            ('threshold', 'Chrom3-2', 1, 1),
            ('degrate', 'Chrom1-2', 1, 2),
            ('degrate', 'Chrom3-2', 1, 3)
        ]
        
        for dataset, chrom_pair, row, col in plot_config:
            try:
                # Load dataset-specific experimental data
                data12, data32 = load_dataset(df, dataset)
                
                # Apply mutant parameters
                n1, n2, n3, N1, N2, N3, k = apply_mutant_params(params, dataset)
                
                # Extract mechanism-specific parameters
                mech_params = extract_mechanism_params(params, mechanism)
                
                # Compute MoM PDFs
                pdf12 = compute_pdf_for_mechanism(
                    mechanism, x_grid, n1, N1, n2, N2, k, mech_params, pair12=True)
                pdf32 = compute_pdf_for_mechanism(
                    mechanism, x_grid, n3, N3, n2, N2, k, mech_params, pair12=False)
                
                # Run stochastic simulation (reduced for speed)
                from TestDataPlot import run_stochastic_simulation
                delta_t12, delta_t32 = run_stochastic_simulation(
                    mechanism, k, n1, n2, n3, N1, N2, N3, mech_params, num_sim=num_sim)
                
                # Select data and simulation results based on chromosome pair
                if chrom_pair == 'Chrom1-2':
                    exp_data = data12
                    sim_data = delta_t12
                    pdf_data = pdf12
                    bins = 15
                else:  # Chrom3-2
                    exp_data = data32
                    sim_data = delta_t32
                    pdf_data = pdf32
                    bins = 14
                
                # Plot the data - matching TestDataPlot.py style
                ax = axes[row, col]
                ax.hist(exp_data, bins=bins, density=True, alpha=0.4, label='Experimental data')
                ax.hist(sim_data, bins=bins, density=True, alpha=0.4, label='Simulated data')
                ax.plot(x_grid, pdf_data, 'r-', linewidth=2, label='MoM PDF')
                
                ax.set_xlim(x_min - 20, x_max + 20)  # Matching TestDataPlot.py range
                ax.set_xlabel('Time Difference')
                ax.set_ylabel('Density')
                ax.set_title(f'{chrom_pair} ({dataset}, {mechanism.replace("_", " ").title()})')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add statistics (matching TestDataPlot.py format)
                stats_text = f'Exp: μ={np.mean(exp_data):.1f}, σ={np.std(exp_data):.1f}\nSim: μ={np.mean(sim_data):.1f}, σ={np.std(sim_data):.1f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                print(f"✓ Successfully plotted {dataset} {chrom_pair}")
                
            except Exception as e:
                print(f"✗ Error plotting {dataset} {chrom_pair}: {e}")
                axes[row, col].text(0.5, 0.5, f'Error plotting {dataset} {chrom_pair}', 
                                   transform=axes[row, col].transAxes, ha='center', va='center')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)  # Increased space for main title
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {output_path}")
    
    def calculate_model_metrics(self, params, mechanism):
        """Calculate AIC, BIC, and other model metrics."""
        try:
            # Load experimental data
            df = pd.read_excel("Data/All_strains_SCStimes.xlsx")
            
            total_nll = 0
            total_data_points = 0
            
            # Calculate NLL for each dataset
            for dataset in ['wildtype', 'threshold', 'degrate', 'degrateAPC', 'initial']:
                try:
                    data12, data32 = load_dataset(df, dataset)
                    n1, n2, n3, N1, N2, N3, k = apply_mutant_params(params, dataset)
                    mech_params = extract_mechanism_params(params, mechanism)
                    
                    # Compute PDFs
                    pdf12 = compute_pdf_for_mechanism(
                        mechanism, data12, n1, N1, n2, N2, k, mech_params, pair12=True)
                    pdf32 = compute_pdf_for_mechanism(
                        mechanism, data32, n3, N3, n2, N2, k, mech_params, pair12=False)
                    
                    # Calculate NLL
                    if np.all(pdf12 > 0) and np.all(pdf32 > 0):
                        nll12 = -np.sum(np.log(pdf12))
                        nll32 = -np.sum(np.log(pdf32))
                        total_nll += nll12 + nll32
                        total_data_points += len(data12) + len(data32)
                
                except Exception as e:
                    print(f"Warning: Could not calculate metrics for {dataset}: {e}")
            
            # Count parameters
            num_params = self.count_parameters(mechanism)
            
            # Calculate AIC and BIC
            aic = 2 * num_params + 2 * total_nll
            bic = np.log(total_data_points) * num_params + 2 * total_nll
            
            return {
                'nll': total_nll,
                'num_params': num_params,
                'num_data_points': total_data_points,
                'aic': aic,
                'bic': bic
            }
            
        except Exception as e:
            print(f"Error calculating metrics for {mechanism}: {e}")
            return None
    
    def count_parameters(self, mechanism):
        """Count the number of parameters for each mechanism."""
        # Base parameters: n2, N2, k, r21, r23, R21, R23, alpha, beta_k, beta2_k, gamma (or gamma1,2,3)
        base_count = 7 + 5  # 7 wild-type + 5 mutant (alpha, beta_k, beta2_k, gamma or gamma1,2,3)
        
        mechanism_params = {
            'simple': 0,
            'fixed_burst': 1,  # burst_size
            'time_varying_k': 1,  # k_1
            'feedback_onion': 1,  # n_inner
            'fixed_burst_feedback_onion': 2,  # burst_size, n_inner
        }
        
        return base_count + mechanism_params.get(mechanism, 0)
    
    def organize_results(self, mechanism, strategy, param_file, runtime, success):
        """Organize results into structured folders."""
        # Create subfolder
        subfolder = f"{mechanism}_{strategy}"
        result_dir = os.path.join(self.base_results_dir, subfolder)
        os.makedirs(result_dir, exist_ok=True)
        
        if success and param_file and os.path.exists(param_file):
            try:
                # Copy parameter file
                dest_param_file = os.path.join(result_dir, f"parameters_{mechanism}_{strategy}.txt")
                shutil.copy2(param_file, dest_param_file)
                
                # Load parameters and create plot
                params = load_parameters(param_file)
                plot_path = os.path.join(result_dir, f"plot_{mechanism}_{strategy}.png")
                
                # Try to create plot, but don't fail if it doesn't work
                try:
                    self.create_custom_plot(params, mechanism, plot_path)
                    print(f"✅ Plot created successfully for {mechanism} ({strategy})")
                except Exception as e:
                    print(f"⚠️  Warning: Could not create plot for {mechanism} ({strategy}): {e}")
                    plot_path = None
                
                # Calculate model metrics
                try:
                    metrics = self.calculate_model_metrics(params, mechanism)
                    if metrics:
                        metrics_file = os.path.join(result_dir, f"metrics_{mechanism}_{strategy}.json")
                        with open(metrics_file, 'w') as f:
                            json.dump(metrics, f, indent=2)
                        print(f"✅ Metrics calculated successfully for {mechanism} ({strategy})")
                    else:
                        print(f"⚠️  Warning: Could not calculate metrics for {mechanism} ({strategy})")
                except Exception as e:
                    print(f"⚠️  Warning: Error calculating metrics for {mechanism} ({strategy}): {e}")
                    metrics = None
                
                # Add to results summary
                result_info = {
                    'mechanism': mechanism,
                    'strategy': strategy,
                    'runtime': runtime,
                    'success': success,
                    'param_file': dest_param_file,
                    'plot_file': plot_path,
                    'metrics': metrics
                }
                
                # Clean up original parameter file
                if os.path.exists(param_file):
                    os.remove(param_file)
                    
            except Exception as e:
                print(f"❌ Error organizing results for {mechanism} ({strategy}): {e}")
                result_info = {
                    'mechanism': mechanism,
                    'strategy': strategy,
                    'runtime': runtime,
                    'success': False,  # Mark as failed if we can't organize results
                    'param_file': None,
                    'plot_file': None,
                    'metrics': None
                }
                
        else:
            result_info = {
                'mechanism': mechanism,
                'strategy': strategy,
                'runtime': runtime,
                'success': success,
                'param_file': None,
                'plot_file': None,
                'metrics': None
            }
        
        self.results_summary.append(result_info)
        return result_info
    
    def run_all_mechanisms(self):
        """Run all mechanisms with all strategies."""
        print(f"Starting comprehensive mechanism analysis...")
        print(f"Mechanisms: {self.mechanisms}")
        print(f"Strategies: {self.strategies}")
        print(f"Results will be saved to: {self.base_results_dir}")
        
        total_runs = len(self.mechanisms) * len(self.strategies)
        current_run = 0
        successful_runs = 0
        failed_runs = 0
        
        for mechanism in self.mechanisms:
            for strategy in self.strategies:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] Running {mechanism} with {strategy} strategy")
                
                try:
                    # Run the mechanism
                    param_file, runtime, success = self.run_single_mechanism(mechanism, strategy)
                    
                    # Organize results
                    result_info = self.organize_results(mechanism, strategy, param_file, runtime, success)
                    
                    if success:
                        successful_runs += 1
                        print(f"📊 Progress: {successful_runs} successful, {failed_runs} failed out of {current_run} completed")
                    else:
                        failed_runs += 1
                        print(f"⚠️  Skipping failed run: {mechanism} ({strategy})")
                        print(f"📊 Progress: {successful_runs} successful, {failed_runs} failed out of {current_run} completed")
                        
                except Exception as e:
                    failed_runs += 1
                    print(f"❌ Critical error running {mechanism} ({strategy}): {e}")
                    print(f"⚠️  Skipping this run and continuing...")
                    
                    # Still add to results summary for tracking
                    result_info = {
                        'mechanism': mechanism,
                        'strategy': strategy,
                        'runtime': 0,
                        'success': False,
                        'param_file': None,
                        'plot_file': None,
                        'metrics': None
                    }
                    self.results_summary.append(result_info)
                    
                    print(f"📊 Progress: {successful_runs} successful, {failed_runs} failed out of {current_run} completed")
        
        print(f"\n{'='*60}")
        print("All mechanisms completed!")
        print(f"📊 FINAL SUMMARY:")
        print(f"   Total runs attempted: {total_runs}")
        print(f"   ✅ Successful runs: {successful_runs}")
        print(f"   ❌ Failed runs: {failed_runs}")
        print(f"   Success rate: {successful_runs/total_runs*100:.1f}%")
        print(f"{'='*60}")
        
        if successful_runs == 0:
            print("⚠️  WARNING: No successful runs! Cannot perform model comparison.")
            return False
        elif failed_runs > 0:
            print(f"⚠️  Note: {failed_runs} runs failed but analysis will continue with {successful_runs} successful runs.")
        
        return True
    
    def create_comparison_report(self):
        """Create AIC/BIC comparison report."""
        print("\nCreating model comparison report...")
        
        # Collect successful results with metrics
        valid_results = [r for r in self.results_summary if r['success'] and r['metrics']]
        
        if not valid_results:
            print("No valid results for comparison!")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        for result in valid_results:
            metrics = result['metrics']
            comparison_data.append({
                'Mechanism': result['mechanism'],
                'Strategy': result['strategy'],
                'Runtime (s)': result['runtime'],
                'Num_Params': metrics['num_params'],
                'NLL': metrics['nll'],
                'AIC': metrics['aic'],
                'BIC': metrics['bic'],
                'Data_Points': metrics['num_data_points']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Sort by AIC (lower is better)
        df_comparison = df_comparison.sort_values('AIC')
        
        # Calculate AIC and BIC weights
        min_aic = df_comparison['AIC'].min()
        min_bic = df_comparison['BIC'].min()
        
        df_comparison['Delta_AIC'] = df_comparison['AIC'] - min_aic
        df_comparison['Delta_BIC'] = df_comparison['BIC'] - min_bic
        df_comparison['AIC_Weight'] = np.exp(-0.5 * df_comparison['Delta_AIC'])
        df_comparison['BIC_Weight'] = np.exp(-0.5 * df_comparison['Delta_BIC'])
        
        # Normalize weights
        df_comparison['AIC_Weight'] /= df_comparison['AIC_Weight'].sum()
        df_comparison['BIC_Weight'] /= df_comparison['BIC_Weight'].sum()
        
        # Save comparison report
        report_file = os.path.join(self.base_results_dir, "model_comparison_report.csv")
        df_comparison.to_csv(report_file, index=False)
        
        # Create summary report
        summary_file = os.path.join(self.base_results_dir, "analysis_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("CHROMOSOME SEGREGATION MECHANISM ANALYSIS SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("MECHANISMS TESTED:\n")
            for mechanism in self.mechanisms:
                f.write(f"  - {mechanism}\n")
            
            f.write(f"\nSTRATEGIES TESTED:\n")
            for strategy in self.strategies:
                f.write(f"  - {strategy}\n")
            
            f.write(f"\nTOTAL RUNS: {len(self.results_summary)}\n")
            f.write(f"SUCCESSFUL RUNS: {len(valid_results)}\n")
            f.write(f"FAILED RUNS: {len(self.results_summary) - len(valid_results)}\n\n")
            
            f.write("MODEL RANKING (by AIC):\n")
            f.write("-" * 40 + "\n")
            for i, row in df_comparison.iterrows():
                f.write(f"{i+1:2d}. {row['Mechanism']} ({row['Strategy']}) - "
                       f"AIC: {row['AIC']:.1f}, BIC: {row['BIC']:.1f}\n")
            
            f.write(f"\nBEST MODEL (AIC): {df_comparison.iloc[0]['Mechanism']} "
                   f"({df_comparison.iloc[0]['Strategy']})\n")
            
            # Sort by BIC for BIC ranking
            df_bic_sorted = df_comparison.sort_values('BIC')
            f.write(f"BEST MODEL (BIC): {df_bic_sorted.iloc[0]['Mechanism']} "
                   f"({df_bic_sorted.iloc[0]['Strategy']})\n\n")
            
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 40 + "\n")
            f.write(df_comparison.to_string(index=False))
        
        print(f"Comparison report saved to: {report_file}")
        print(f"Analysis summary saved to: {summary_file}")
        
        # Print summary to console
        print(f"\n{'='*60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"Best model (AIC): {df_comparison.iloc[0]['Mechanism']} ({df_comparison.iloc[0]['Strategy']})")
        print(f"Best model (BIC): {df_bic_sorted.iloc[0]['Mechanism']} ({df_bic_sorted.iloc[0]['Strategy']})")
        print(f"\nTop 3 models by AIC:")
        for i in range(min(3, len(df_comparison))):
            row = df_comparison.iloc[i]
            print(f"  {i+1}. {row['Mechanism']} ({row['Strategy']}) - AIC: {row['AIC']:.1f}")


def main():
    """Main function to run the comprehensive analysis."""
    print("COMPREHENSIVE CHROMOSOME SEGREGATION MECHANISM ANALYSIS")
    print("="*60)
    print("This script will run all mechanisms with both optimization strategies")
    print("and generate organized results with AIC/BIC model comparison.")
    print("Note: Failed runs will be skipped automatically.")
    print("="*60)
    
    # Create runner instance
    runner = MechanismRunner()
    
    try:
        # Run all mechanisms
        success = runner.run_all_mechanisms()
        
        if success:
            # Create comparison report only if we have successful runs
            runner.create_comparison_report()
            
            print(f"\n{'='*60}")
            print("ANALYSIS COMPLETE!")
            print(f"Results saved to: {runner.base_results_dir}")
            print("Check the analysis_summary.txt file for detailed results.")
            
            # Count successful vs failed runs
            successful_count = sum(1 for r in runner.results_summary if r['success'])
            total_count = len(runner.results_summary)
            
            if successful_count < total_count:
                print(f"\n⚠️  Note: {total_count - successful_count} out of {total_count} runs failed.")
                print("Check individual run outputs above for details on failures.")
                print("The analysis was completed with the successful runs only.")
            
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print("ANALYSIS FAILED!")
            print("All optimization runs failed. Please check:")
            print("1. Required Python packages are installed")
            print("2. Data files are present and accessible")
            print("3. Optimization scripts are working correctly")
            print(f"Partial results may be available in: {runner.base_results_dir}")
            print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        print(f"Partial results may be available in: {runner.base_results_dir}")
        
        # Still try to create a report with any completed runs
        try:
            if runner.results_summary:
                print("Attempting to create report with completed runs...")
                runner.create_comparison_report()
        except:
            print("Could not create comparison report from partial results.")
            
    except Exception as e:
        print(f"\nCritical error during analysis: {e}")
        import traceback
        traceback.print_exc()
        print(f"Partial results may be available in: {runner.base_results_dir}")
        
        # Still try to create a report with any completed runs
        try:
            if runner.results_summary:
                print("Attempting to create report with completed runs...")
                runner.create_comparison_report()
        except:
            print("Could not create comparison report from partial results.")


if __name__ == "__main__":
    main() 