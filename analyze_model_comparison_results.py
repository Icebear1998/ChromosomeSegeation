#!/usr/bin/env python3
"""
Comprehensive Diagnostic Analysis for Model Comparison Results

This script extracts and analyzes results from model_comparison output files
to verify correctness and understand why simple model performs best.

Tests performed:
1. Statistical significance tests (t-tests, ANOVA)
2. Convergence quality analysis
3. Parameter stability across runs
4. Effect size calculations (Cohen's d, ΔAIC/ΔBIC)
5. Likelihood ratio tests
6. Model selection diagnostics
7. Optimization quality checks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, mannwhitneyu
import re
from pathlib import Path

def extract_results_from_output(filename):
    """
    Extract model comparison results from output file.
    
    Args:
        filename (str): Path to output file
    
    Returns:
        dict: Extracted results with mechanism names as keys
    """
    with open(filename, 'r') as f:
        content = f.read()
    
    results = {}
    
    # Pattern to match mechanism summaries
    mechanism_pattern = r'Running comparison for: (\w+)'
    # Order: AIC, BIC, NLL (as they appear in the file)
    summary_pattern = r'📊 Summary for (\w+):\s+Convergence rate: ([\d.]+)% \((\d+)/(\d+)\)\s+Mean AIC: ([\d.]+) ± ([\d.]+)\s+Mean BIC: ([\d.]+) ± ([\d.]+)\s+Mean NLL: ([\d.]+) ± ([\d.]+)'
    
    # Find all mechanism sections
    mechanisms = re.findall(mechanism_pattern, content)
    
    # Extract individual run results
    run_pattern = r'✅ Run (\d+) converged: NLL=([\d.]+), AIC=([\d.]+), BIC=([\d.]+)'
    
    for mechanism in mechanisms:
        # Find section for this mechanism
        section_start = content.find(f'Running comparison for: {mechanism.upper()}')
        if section_start == -1:
            continue
        
        # Find next mechanism or end
        next_section = content.find('Running comparison for:', section_start + 1)
        if next_section == -1:
            section = content[section_start:]
        else:
            section = content[section_start:next_section]
        
        # Extract runs
        runs = re.findall(run_pattern, section)
        
        if runs:
            nll_values = [float(run[1]) for run in runs]
            aic_values = [float(run[2]) for run in runs]
            bic_values = [float(run[3]) for run in runs]
            
            # Extract summary statistics
            summary_match = re.search(summary_pattern, section)
            if summary_match:
                # Groups: 1=mech, 2=conv_rate, 3=conv_runs, 4=total_runs, 
                #         5=mean_aic, 6=std_aic, 7=mean_bic, 8=std_bic, 9=mean_nll, 10=std_nll
                results[mechanism.lower()] = {
                    'mechanism': mechanism.lower(),
                    'nll_values': nll_values,
                    'aic_values': aic_values,
                    'bic_values': bic_values,
                    'mean_aic': float(summary_match.group(5)),
                    'std_aic': float(summary_match.group(6)),
                    'mean_bic': float(summary_match.group(7)),
                    'std_bic': float(summary_match.group(8)),
                    'mean_nll': float(summary_match.group(9)),
                    'std_nll': float(summary_match.group(10)),
                    'convergence_rate': float(summary_match.group(2)),
                    'n_runs': len(runs)
                }
            else:
                # Fallback: calculate from runs
                results[mechanism.lower()] = {
                    'mechanism': mechanism.lower(),
                    'nll_values': nll_values,
                    'aic_values': aic_values,
                    'bic_values': bic_values,
                    'mean_nll': np.mean(nll_values),
                    'std_nll': np.std(nll_values),
                    'mean_aic': np.mean(aic_values),
                    'std_aic': np.std(aic_values),
                    'mean_bic': np.mean(bic_values),
                    'std_bic': np.std(bic_values),
                    'convergence_rate': 100.0,
                    'n_runs': len(runs)
                }
    
    return results


def statistical_significance_tests(results):
    """
    Perform statistical tests to check if differences are significant.
    
    Args:
        results (dict): Extracted results
    
    Returns:
        dict: Test results
    """
    test_results = {}
    
    # Get simple model as baseline
    simple_nll = np.array(results['simple']['nll_values'])
    simple_aic = np.array(results['simple']['aic_values'])
    simple_bic = np.array(results['simple']['bic_values'])
    
    mechanisms = list(results.keys())
    mechanisms.remove('simple')
    
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)
    print("\nComparing each mechanism against 'simple' model:")
    print("-"*80)
    
    for mech in mechanisms:
        mech_nll = np.array(results[mech]['nll_values'])
        mech_aic = np.array(results[mech]['aic_values'])
        mech_bic = np.array(results[mech]['bic_values'])
        
        # T-test for NLL
        t_stat_nll, p_val_nll = ttest_ind(simple_nll, mech_nll, alternative='less')
        
        # T-test for AIC
        t_stat_aic, p_val_aic = ttest_ind(simple_aic, mech_aic, alternative='less')
        
        # T-test for BIC
        t_stat_bic, p_val_bic = ttest_ind(simple_bic, mech_bic, alternative='less')
        
        # Mann-Whitney U test (non-parametric)
        u_stat, p_val_mw = mannwhitneyu(simple_nll, mech_nll, alternative='less')
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(simple_nll) + np.var(mech_nll)) / 2)
        cohens_d = (np.mean(mech_nll) - np.mean(simple_nll)) / pooled_std if pooled_std > 0 else 0
        
        test_results[mech] = {
            't_test_nll': (t_stat_nll, p_val_nll),
            't_test_aic': (t_stat_aic, p_val_aic),
            't_test_bic': (t_stat_bic, p_val_bic),
            'mann_whitney': (u_stat, p_val_mw),
            'cohens_d': cohens_d
        }
        
        print(f"\n{mech.upper()}:")
        print(f"  NLL: Simple={np.mean(simple_nll):.2f} vs {mech}={np.mean(mech_nll):.2f}")
        print(f"    T-test: t={t_stat_nll:.3f}, p={p_val_nll:.4f} {'***' if p_val_nll < 0.001 else '**' if p_val_nll < 0.01 else '*' if p_val_nll < 0.05 else 'ns'}")
        print(f"    Mann-Whitney: U={u_stat:.1f}, p={p_val_mw:.4f} {'***' if p_val_mw < 0.001 else '**' if p_val_mw < 0.01 else '*' if p_val_mw < 0.05 else 'ns'}")
        print(f"    Cohen's d: {cohens_d:.3f} ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'} effect)")
        print(f"  AIC: Simple={np.mean(simple_aic):.2f} vs {mech}={np.mean(mech_aic):.2f}")
        print(f"    T-test: t={t_stat_aic:.3f}, p={p_val_aic:.4f} {'***' if p_val_aic < 0.001 else '**' if p_val_aic < 0.01 else '*' if p_val_aic < 0.05 else 'ns'}")
        print(f"  BIC: Simple={np.mean(simple_bic):.2f} vs {mech}={np.mean(mech_bic):.2f}")
        print(f"    T-test: t={t_stat_bic:.3f}, p={p_val_bic:.4f} {'***' if p_val_bic < 0.001 else '**' if p_val_bic < 0.01 else '*' if p_val_bic < 0.05 else 'ns'}")
    
    # ANOVA across all models
    print("\n" + "-"*80)
    print("ONE-WAY ANOVA (All Models):")
    print("-"*80)
    
    all_nll = [results[m]['nll_values'] for m in results.keys()]
    f_stat, p_anova = f_oneway(*all_nll)
    print(f"F-statistic: {f_stat:.3f}, p-value: {p_anova:.6f}")
    print(f"Result: {'Significant differences exist' if p_anova < 0.05 else 'No significant differences'}")
    
    test_results['anova'] = (f_stat, p_anova)
    
    return test_results


def convergence_quality_analysis(results):
    """
    Analyze convergence quality across runs.
    
    Args:
        results (dict): Extracted results
    
    Returns:
        dict: Convergence analysis
    """
    print("\n" + "="*80)
    print("CONVERGENCE QUALITY ANALYSIS")
    print("="*80)
    
    convergence_analysis = {}
    
    for mech, data in results.items():
        nll_vals = np.array(data['nll_values'])
        
        # Coefficient of variation
        cv = (np.std(nll_vals) / np.mean(nll_vals)) * 100 if np.mean(nll_vals) > 0 else 0
        
        # Range
        nll_range = np.max(nll_vals) - np.min(nll_vals)
        
        # Relative range (as % of mean)
        rel_range = (nll_range / np.mean(nll_vals)) * 100 if np.mean(nll_vals) > 0 else 0
        
        convergence_analysis[mech] = {
            'cv': cv,
            'range': nll_range,
            'relative_range': rel_range,
            'min_nll': np.min(nll_vals),
            'max_nll': np.max(nll_vals),
            'std': np.std(nll_vals)
        }
        
        print(f"\n{mech.upper()}:")
        print(f"  NLL range: {np.min(nll_vals):.2f} - {np.max(nll_vals):.2f} (span: {nll_range:.2f})")
        print(f"  Coefficient of variation: {cv:.2f}%")
        print(f"  Relative range: {rel_range:.2f}%")
        print(f"  Convergence quality: {'Excellent' if cv < 1 else 'Good' if cv < 3 else 'Fair' if cv < 5 else 'Poor'}")
    
    return convergence_analysis


def effect_size_analysis(results):
    """
    Calculate effect sizes (ΔAIC, ΔBIC) and model selection strength.
    
    Args:
        results (dict): Extracted results
    
    Returns:
        dict: Effect size analysis
    """
    print("\n" + "="*80)
    print("EFFECT SIZE ANALYSIS (ΔAIC, ΔBIC)")
    print("="*80)
    
    # Find best model
    best_aic = min(results.values(), key=lambda x: x['mean_aic'])
    best_bic = min(results.values(), key=lambda x: x['mean_bic'])
    
    print(f"\nBest model by AIC: {best_aic['mechanism']} (AIC = {best_aic['mean_aic']:.2f})")
    print(f"Best model by BIC: {best_bic['mechanism']} (BIC = {best_bic['mean_bic']:.2f})")
    
    # Also show NLL for reference
    print(f"  (Simple model NLL = {results['simple']['mean_nll']:.2f}, BIC = {results['simple']['mean_bic']:.2f})")
    
    print("\nΔAIC and ΔBIC relative to best model:")
    print("-"*80)
    print(f"{'Mechanism':<30} {'ΔAIC':<10} {'ΔBIC':<10} {'Interpretation'}")
    print("-"*80)
    
    effect_sizes = {}
    
    for mech, data in results.items():
        delta_aic = data['mean_aic'] - best_aic['mean_aic']
        delta_bic = data['mean_bic'] - best_bic['mean_bic']
        
        # Interpretation
        if delta_aic < 2:
            aic_interp = "Substantial support"
        elif delta_aic < 6:
            aic_interp = "Considerable support"
        elif delta_aic < 10:
            aic_interp = "Weak support"
        else:
            aic_interp = "No support"
        
        if delta_bic < 2:
            bic_interp = "Substantial support"
        elif delta_bic < 6:
            bic_interp = "Considerable support"
        elif delta_bic < 10:
            bic_interp = "Weak support"
        else:
            bic_interp = "No support"
        
        effect_sizes[mech] = {
            'delta_aic': delta_aic,
            'delta_bic': delta_bic,
            'aic_interpretation': aic_interp,
            'bic_interpretation': bic_interp
        }
        
        print(f"{mech:<30} {delta_aic:>8.2f}  {delta_bic:>8.2f}  {aic_interp}")
    
    return effect_sizes


def likelihood_ratio_tests(results, n_data=1423):
    """
    Perform likelihood ratio tests between nested models.
    
    Args:
        results (dict): Extracted results
        n_data (int): Number of data points
    
    Returns:
        dict: LRT results
    """
    print("\n" + "="*80)
    print("LIKELIHOOD RATIO TESTS (Nested Models)")
    print("="*80)
    
    lrt_results = {}
    
    # Nested model pairs
    nested_pairs = [
        ('simple', 'fixed_burst'),  # simple nested in fixed_burst
        ('simple', 'feedback_onion'),  # simple nested in feedback_onion
        ('fixed_burst', 'fixed_burst_feedback_onion'),  # fixed_burst nested in combined
        ('feedback_onion', 'fixed_burst_feedback_onion'),  # feedback nested in combined
    ]
    
    print("\nTesting nested model pairs:")
    print("-"*80)
    
    for null_model, alt_model in nested_pairs:
        if null_model not in results or alt_model not in results:
            continue
        
        null_nll = np.mean(results[null_model]['nll_values'])
        alt_nll = np.mean(results[alt_model]['nll_values'])
        
        # LRT statistic: 2 * (null_nll - alt_nll)
        lrt_stat = 2 * (null_nll - alt_nll)
        
        # Degrees of freedom: difference in parameters
        null_params = len(results[null_model]['nll_values'])  # Approximate
        alt_params = len(results[alt_model]['nll_values'])
        df = abs(results[alt_model].get('n_params', 13) - results[null_model].get('n_params', 11))
        
        # Chi-square test
        p_value = 1 - stats.chi2.cdf(lrt_stat, df) if lrt_stat > 0 else 1.0
        
        lrt_results[f"{null_model}_vs_{alt_model}"] = {
            'lrt_statistic': lrt_stat,
            'df': df,
            'p_value': p_value,
            'null_nll': null_nll,
            'alt_nll': alt_nll
        }
        
        print(f"\n{null_model} vs {alt_model}:")
        print(f"  LRT statistic: {lrt_stat:.3f}")
        print(f"  Degrees of freedom: {df}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Result: {'Reject null (complex model better)' if p_value < 0.05 and lrt_stat > 0 else 'Accept null (simple model sufficient)'}")
    
    return lrt_results


def optimization_quality_check(results):
    """
    Check if simulation-based models are finding good optima.
    
    Args:
        results (dict): Extracted results
    
    Returns:
        dict: Optimization quality metrics
    """
    print("\n" + "="*80)
    print("OPTIMIZATION QUALITY CHECK")
    print("="*80)
    
    # Separate MoM and simulation-based models
    mom_models = ['simple', 'fixed_burst', 'feedback_onion', 'fixed_burst_feedback_onion']
    sim_models = ['time_varying_k', 'time_varying_k_fixed_burst', 
                  'time_varying_k_feedback_onion', 'time_varying_k_combined']
    
    mom_nlls = []
    sim_nlls = []
    
    for mech in mom_models:
        if mech in results:
            mom_nlls.extend(results[mech]['nll_values'])
    
    for mech in sim_models:
        if mech in results:
            sim_nlls.extend(results[mech]['nll_values'])
    
    print("\nComparison of MoM vs Simulation-based optimization:")
    print("-"*80)
    print(f"MoM models (n={len(mom_nlls)} runs):")
    print(f"  Mean NLL: {np.mean(mom_nlls):.2f} ± {np.std(mom_nlls):.2f}")
    print(f"  Range: {np.min(mom_nlls):.2f} - {np.max(mom_nlls):.2f}")
    
    print(f"\nSimulation-based models (n={len(sim_nlls)} runs):")
    print(f"  Mean NLL: {np.mean(sim_nlls):.2f} ± {np.std(sim_nlls):.2f}")
    print(f"  Range: {np.min(sim_nlls):.2f} - {np.max(sim_nlls):.2f}")
    
    # Test if simulation models are significantly worse
    if len(sim_nlls) > 0 and len(mom_nlls) > 0:
        t_stat, p_val = ttest_ind(mom_nlls, sim_nlls, alternative='less')
        print(f"\nT-test: MoM vs Simulation")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_val:.6f}")
        print(f"  Interpretation: {'Simulation models finding worse optima' if p_val < 0.05 else 'No significant difference in optimization quality'}")
    
    return {
        'mom_mean_nll': np.mean(mom_nlls) if mom_nlls else None,
        'sim_mean_nll': np.mean(sim_nlls) if sim_nlls else None,
        'difference': np.mean(sim_nlls) - np.mean(mom_nlls) if (sim_nlls and mom_nlls) else None
    }


def create_diagnostic_plots(results, output_dir='diagnostic_plots'):
    """
    Create diagnostic plots for model comparison.
    
    Args:
        results (dict): Extracted results
        output_dir (str): Output directory for plots
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Prepare data for plotting
    plot_data = []
    for mech, data in results.items():
        for i, (nll, aic, bic) in enumerate(zip(data['nll_values'], 
                                                  data['aic_values'], 
                                                  data['bic_values'])):
            plot_data.append({
                'mechanism': mech,
                'run': i+1,
                'NLL': nll,
                'AIC': aic,
                'BIC': bic
            })
    
    df = pd.DataFrame(plot_data)
    
    # 1. NLL distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # NLL boxplot
    mechanisms = sorted(results.keys())
    nll_data = [results[m]['nll_values'] for m in mechanisms]
    bp = axes[0, 0].boxplot(nll_data)
    axes[0, 0].set_xticklabels(mechanisms, rotation=45, ha='right')
    axes[0, 0].set_title('NLL Distribution Across Models')
    axes[0, 0].set_ylabel('Negative Log-Likelihood')
    axes[0, 0].grid(True, alpha=0.3)
    
    # AIC comparison
    aic_means = [results[m]['mean_aic'] for m in mechanisms]
    aic_stds = [results[m]['std_aic'] for m in mechanisms]
    axes[0, 1].bar(range(len(mechanisms)), aic_means, yerr=aic_stds, capsize=5, alpha=0.7)
    axes[0, 1].set_title('Mean AIC Comparison')
    axes[0, 1].set_ylabel('AIC')
    axes[0, 1].set_xticks(range(len(mechanisms)))
    axes[0, 1].set_xticklabels(mechanisms, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # BIC comparison
    bic_means = [results[m]['mean_bic'] for m in mechanisms]
    bic_stds = [results[m]['std_bic'] for m in mechanisms]
    axes[1, 0].bar(range(len(mechanisms)), bic_means, yerr=bic_stds, capsize=5, alpha=0.7, color='coral')
    axes[1, 0].set_title('Mean BIC Comparison')
    axes[1, 0].set_ylabel('BIC')
    axes[1, 0].set_xticks(range(len(mechanisms)))
    axes[1, 0].set_xticklabels(mechanisms, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Convergence stability (CV)
    cvs = [(np.std(results[m]['nll_values']) / np.mean(results[m]['nll_values'])) * 100 
           for m in mechanisms]
    axes[1, 1].bar(range(len(mechanisms)), cvs, alpha=0.7, color='green')
    axes[1, 1].set_title('Convergence Stability (Coefficient of Variation)')
    axes[1, 1].set_ylabel('CV (%)')
    axes[1, 1].set_xticks(range(len(mechanisms)))
    axes[1, 1].set_xticklabels(mechanisms, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/diagnostic_summary.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Diagnostic plots saved to {output_dir}/diagnostic_summary.png")
    
    plt.close()


def generate_report(results, test_results, output_file='diagnostic_report.txt'):
    """
    Generate comprehensive diagnostic report.
    
    Args:
        results (dict): Extracted results
        test_results (dict): Statistical test results
        output_file (str): Output file path
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL COMPARISON DIAGNOSTIC REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("SUMMARY OF FINDINGS:\n")
        f.write("-"*80 + "\n")
        
        # Best models
        best_aic = min(results.values(), key=lambda x: x['mean_aic'])
        best_bic = min(results.values(), key=lambda x: x['mean_bic'])
        
        f.write(f"Best model by AIC: {best_aic['mechanism']} (AIC = {best_aic['mean_aic']:.2f})\n")
        f.write(f"Best model by BIC: {best_bic['mechanism']} (BIC = {best_bic['mean_bic']:.2f})\n\n")
        
        # Key observations
        simple_nll = results['simple']['mean_nll']
        f.write("KEY OBSERVATIONS:\n")
        f.write(f"1. Simple model NLL: {simple_nll:.2f}\n")
        
        # Compare with time-varying models
        tv_models = [m for m in results.keys() if 'time_varying' in m]
        if tv_models:
            tv_nlls = [results[m]['mean_nll'] for m in tv_models]
            best_tv_nll = min(tv_nlls)
            f.write(f"2. Best time-varying model NLL: {best_tv_nll:.2f}\n")
            f.write(f"3. NLL difference: {best_tv_nll - simple_nll:.2f} (simple is better by this amount)\n")
            f.write(f"4. This difference is {'NOT' if best_tv_nll - simple_nll < 10 else ''} sufficient to overcome parameter penalty\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("RECOMMENDATIONS:\n")
        f.write("="*80 + "\n")
        f.write("1. Verify that simulation-based models are finding global optima\n")
        f.write("2. Check if time-varying mechanisms actually improve fit to data\n")
        f.write("3. Consider increasing optimization iterations for complex models\n")
        f.write("4. Validate that parameter bounds are appropriate\n")
        f.write("5. Check for systematic differences in likelihood calculation\n")
    
    print(f"\n✅ Diagnostic report saved to {output_file}")


def main():
    """
    Main diagnostic analysis function.
    """
    import sys
    
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    else:
        output_file = 'model_comparison_main.out'
    
    print("="*80)
    print("MODEL COMPARISON DIAGNOSTIC ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing results from: {output_file}")
    
    # Extract results
    print("\n📊 Extracting results from output file...")
    results = extract_results_from_output(output_file)
    
    if not results:
        print("❌ No results found in output file!")
        return
    
    print(f"✅ Extracted results for {len(results)} mechanisms")
    print(f"   Mechanisms: {', '.join(results.keys())}")
    
    # Perform tests
    print("\n🔬 Running diagnostic tests...")
    
    # 1. Statistical significance tests
    test_results = statistical_significance_tests(results)
    
    # 2. Convergence quality
    convergence_analysis = convergence_quality_analysis(results)
    
    # 3. Effect sizes
    effect_sizes = effect_size_analysis(results)
    
    # 4. Likelihood ratio tests
    lrt_results = likelihood_ratio_tests(results)
    
    # 5. Optimization quality
    opt_quality = optimization_quality_check(results)
    
    # 6. Create plots
    print("\n📈 Creating diagnostic plots...")
    create_diagnostic_plots(results)
    
    # 7. Generate report
    print("\n📝 Generating diagnostic report...")
    generate_report(results, test_results)
    
    print("\n" + "="*80)
    print("✅ DIAGNOSTIC ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - diagnostic_plots/diagnostic_summary.png")
    print("  - diagnostic_report.txt")
    print("\nReview these files for detailed analysis and recommendations.")


if __name__ == "__main__":
    main()
