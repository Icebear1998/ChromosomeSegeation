#!/usr/bin/env python3
"""
Visualize parameter distributions from model comparison optimization runs.

This script creates comprehensive visualizations of parameter distributions
across different mechanisms, showing how parameter values vary across runs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pathlib import Path
import sys

# Try to import seaborn, but make it optional
try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_style("whitegrid")
except ImportError:
    HAS_SEABORN = False
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10

def load_optimization_results(csv_file):
    """
    Load optimization results from CSV file.
    
    Args:
        csv_file (str): Path to CSV file
    
    Returns:
        pd.DataFrame: Loaded data
    """
    df = pd.read_csv(csv_file)
    
    # Filter only successful and converged runs
    df = df[(df['success'] == True) & (df['converged'] == True)].copy()
    
    print(f"Loaded {len(df)} successful runs from {csv_file}")
    print(f"Mechanisms: {df['mechanism'].unique()}")
    print(f"Runs per mechanism: {df.groupby('mechanism').size()}")
    
    return df


def get_parameter_columns(df):
    """
    Get all parameter columns from the dataframe.
    
    Args:
        df (pd.DataFrame): Data frame
    
    Returns:
        list: Parameter column names
    """
    # Parameter columns are between 'message' and 'params_json'
    param_cols = ['n2', 'N2', 'k', 'k_max', 'tau', 'r21', 'r23', 'R21', 'R23', 
                  'burst_size', 'n_inner', 'alpha', 'beta_k', 'beta_tau', 'beta_tau2', 
                  'beta2_k', 'beta3_k']
    
    # Only return columns that exist and have at least some non-null values
    available_cols = []
    for col in param_cols:
        if col in df.columns and df[col].notna().any():
            available_cols.append(col)
    
    return available_cols


def plot_parameter_distributions(df, output_dir='parameter_distribution_plots'):
    """
    Create distribution plots for each parameter across mechanisms.
    
    Args:
        df (pd.DataFrame): Optimization results
        output_dir (str): Output directory for plots
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    param_cols = get_parameter_columns(df)
    mechanisms = sorted(df['mechanism'].unique())
    
    # Define colors for each mechanism
    colors = plt.cm.tab10(np.linspace(0, 1, len(mechanisms)))
    color_map = dict(zip(mechanisms, colors))
    
    # Create a large figure with subplots for all parameters
    n_params = len(param_cols)
    n_cols = 4
    n_rows = int(np.ceil(n_params / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten() if n_params > 1 else [axes]
    
    for idx, param in enumerate(param_cols):
        ax = axes[idx]
        
        # Plot distribution for each mechanism
        for mechanism in mechanisms:
            mech_data = df[df['mechanism'] == mechanism][param].dropna()
            
            if len(mech_data) > 0:
                # Plot KDE
                try:
                    kde = stats.gaussian_kde(mech_data)
                    x_range = np.linspace(mech_data.min(), mech_data.max(), 200)
                    ax.plot(x_range, kde(x_range), 
                           label=f"{mechanism} (n={len(mech_data)})",
                           color=color_map[mechanism], linewidth=2, alpha=0.8)
                except:
                    # If KDE fails (e.g., all values are the same), plot histogram instead
                    ax.hist(mech_data, bins=20, alpha=0.3, 
                           label=f"{mechanism} (n={len(mech_data)})",
                           color=color_map[mechanism], density=True)
        
        ax.set_xlabel(param)
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of {param}')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_params, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/all_parameters_distributions.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved comprehensive plot: {output_dir}/all_parameters_distributions.png")
    plt.close()


def plot_individual_parameter_distributions(df, output_dir='parameter_distribution_plots'):
    """
    Create individual plots for each parameter.
    
    Args:
        df (pd.DataFrame): Optimization results
        output_dir (str): Output directory for plots
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    param_cols = get_parameter_columns(df)
    mechanisms = sorted(df['mechanism'].unique())
    
    # Define colors for each mechanism
    colors = plt.cm.tab10(np.linspace(0, 1, len(mechanisms)))
    color_map = dict(zip(mechanisms, colors))
    
    for param in param_cols:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        has_data = False
        for mechanism in mechanisms:
            mech_data = df[df['mechanism'] == mechanism][param].dropna()
            
            if len(mech_data) > 0:
                has_data = True
                # Plot KDE
                try:
                    kde = stats.gaussian_kde(mech_data)
                    x_range = np.linspace(mech_data.min(), mech_data.max(), 200)
                    ax.plot(x_range, kde(x_range), 
                           label=f"{mechanism} (n={len(mech_data)}, μ={mech_data.mean():.3f})",
                           color=color_map[mechanism], linewidth=2.5, alpha=0.8)
                    
                    # Add rug plot (small vertical lines at data points)
                    ax.scatter(mech_data, np.zeros_like(mech_data) - 0.01 * ax.get_ylim()[1],
                             alpha=0.3, color=color_map[mechanism], marker='|', s=100)
                except Exception as e:
                    print(f"Warning: Could not create KDE for {mechanism} - {param}: {e}")
        
        if has_data:
            ax.set_xlabel(param, fontsize=14)
            ax.set_ylabel('Density', fontsize=14)
            ax.set_title(f'Distribution of {param} Across Mechanisms', fontsize=16, fontweight='bold')
            ax.legend(fontsize=10, loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{param}_distribution.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {output_dir}/{param}_distribution.png")
            plt.close()
        else:
            plt.close()


def plot_parameter_boxplots(df, output_dir='parameter_distribution_plots'):
    """
    Create boxplot comparisons for each parameter across mechanisms.
    
    Args:
        df (pd.DataFrame): Optimization results
        output_dir (str): Output directory for plots
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    param_cols = get_parameter_columns(df)
    mechanisms = sorted(df['mechanism'].unique())
    
    # Create a large figure with subplots for all parameters
    n_params = len(param_cols)
    n_cols = 3
    n_rows = int(np.ceil(n_params / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten() if n_params > 1 else [axes]
    
    for idx, param in enumerate(param_cols):
        ax = axes[idx]
        
        # Prepare data for boxplot
        data_to_plot = []
        labels = []
        
        for mechanism in mechanisms:
            mech_data = df[df['mechanism'] == mechanism][param].dropna()
            if len(mech_data) > 0:
                data_to_plot.append(mech_data)
                labels.append(mechanism)
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, patch_artist=True)
            
            # Set labels manually
            ax.set_xticklabels(labels, rotation=45, ha='right')
            
            # Color the boxes
            colors = plt.cm.tab10(np.linspace(0, 1, len(data_to_plot)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            ax.set_ylabel(param, fontsize=11)
            ax.set_title(f'{param} Comparison', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
    
    # Hide extra subplots
    for idx in range(n_params, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/all_parameters_boxplots.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved boxplot comparison: {output_dir}/all_parameters_boxplots.png")
    plt.close()


def plot_mechanism_specific_parameters(df, output_dir='parameter_distribution_plots'):
    """
    Create mechanism-specific parameter distribution plots.
    Shows all parameters for each mechanism in a single figure.
    
    Args:
        df (pd.DataFrame): Optimization results
        output_dir (str): Output directory for plots
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    param_cols = get_parameter_columns(df)
    mechanisms = sorted(df['mechanism'].unique())
    
    for mechanism in mechanisms:
        mech_data = df[df['mechanism'] == mechanism]
        
        # Filter to parameters that this mechanism uses
        mech_params = [p for p in param_cols if mech_data[p].notna().any()]
        
        if not mech_params:
            continue
        
        # Create subplots
        n_params = len(mech_params)
        n_cols = 4
        n_rows = int(np.ceil(n_params / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        axes = axes.flatten() if n_params > 1 else [axes]
        
        fig.suptitle(f'Parameter Distributions for {mechanism.upper()}', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        for idx, param in enumerate(mech_params):
            ax = axes[idx]
            param_data = mech_data[param].dropna()
            
            if len(param_data) > 0:
                # Plot histogram and KDE
                ax.hist(param_data, bins=15, alpha=0.6, color='skyblue', 
                       density=True, edgecolor='black')
                
                try:
                    kde = stats.gaussian_kde(param_data)
                    x_range = np.linspace(param_data.min(), param_data.max(), 200)
                    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
                except:
                    pass
                
                # Add statistics
                mean_val = param_data.mean()
                median_val = param_data.median()
                std_val = param_data.std()
                
                ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, 
                          label=f'Mean: {mean_val:.3f}')
                ax.axvline(median_val, color='orange', linestyle='--', linewidth=2,
                          label=f'Median: {median_val:.3f}')
                
                ax.set_xlabel(param, fontsize=11)
                ax.set_ylabel('Density', fontsize=11)
                ax.set_title(f'{param}\n(μ={mean_val:.3f}, σ={std_val:.3f})', 
                           fontsize=10)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for idx in range(n_params, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{mechanism}_all_parameters.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved mechanism-specific plot: {output_dir}/{mechanism}_all_parameters.png")
        plt.close()


def create_parameter_summary_table(df, output_dir='parameter_distribution_plots'):
    """
    Create a summary table of parameter statistics for each mechanism.
    
    Args:
        df (pd.DataFrame): Optimization results
        output_dir (str): Output directory
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    param_cols = get_parameter_columns(df)
    mechanisms = sorted(df['mechanism'].unique())
    
    summary_data = []
    
    for mechanism in mechanisms:
        mech_data = df[df['mechanism'] == mechanism]
        
        for param in param_cols:
            param_data = mech_data[param].dropna()
            
            if len(param_data) > 0:
                summary_data.append({
                    'Mechanism': mechanism,
                    'Parameter': param,
                    'N': len(param_data),
                    'Mean': param_data.mean(),
                    'Median': param_data.median(),
                    'Std': param_data.std(),
                    'Min': param_data.min(),
                    'Max': param_data.max(),
                    'CV (%)': (param_data.std() / param_data.mean() * 100) if param_data.mean() != 0 else 0
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_file = f'{output_dir}/parameter_statistics_summary.csv'
    summary_df.to_csv(csv_file, index=False, float_format='%.4f')
    print(f"\n✅ Saved parameter statistics: {csv_file}")
    
    # Create formatted table for each mechanism
    for mechanism in mechanisms:
        mech_summary = summary_df[summary_df['Mechanism'] == mechanism]
        if len(mech_summary) > 0:
            print(f"\n{'='*80}")
            print(f"Parameter Statistics for {mechanism.upper()}")
            print(f"{'='*80}")
            print(mech_summary.to_string(index=False))
    
    return summary_df


def plot_nll_aic_bic_distributions(df, output_dir='parameter_distribution_plots'):
    """
    Plot distributions of NLL, AIC, and BIC values across mechanisms.
    
    Args:
        df (pd.DataFrame): Optimization results
        output_dir (str): Output directory
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    mechanisms = sorted(df['mechanism'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(mechanisms)))
    color_map = dict(zip(mechanisms, colors))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['nll', 'aic', 'bic']
    titles = ['Negative Log-Likelihood', 'AIC', 'BIC']
    
    for ax, metric, title in zip(axes, metrics, titles):
        for mechanism in mechanisms:
            mech_data = df[df['mechanism'] == mechanism][metric].dropna()
            
            if len(mech_data) > 0:
                try:
                    kde = stats.gaussian_kde(mech_data)
                    x_range = np.linspace(mech_data.min(), mech_data.max(), 200)
                    ax.plot(x_range, kde(x_range), 
                           label=f"{mechanism} (μ={mech_data.mean():.1f})",
                           color=color_map[mechanism], linewidth=2.5, alpha=0.8)
                except:
                    pass
        
        ax.set_xlabel(title, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{title} Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/nll_aic_bic_distributions.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved NLL/AIC/BIC distributions: {output_dir}/nll_aic_bic_distributions.png")
    plt.close()


def main():
    """
    Main function to create all visualizations.
    """
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Find the most recent CSV file
        csv_files = list(Path('.').glob('optimized_params_runs_*.csv'))
        if not csv_files:
            print("❌ No CSV files found! Please provide a file path.")
            print("Usage: python visualize_parameter_distributions.py <csv_file>")
            return
        csv_file = max(csv_files, key=lambda p: p.stat().st_mtime)
        print(f"Using most recent file: {csv_file}")
    
    print("="*80)
    print("PARAMETER DISTRIBUTION VISUALIZATION")
    print("="*80)
    
    # Load data
    df = load_optimization_results(csv_file)
    
    if len(df) == 0:
        print("❌ No successful runs found in the data!")
        return
    
    # Create output directory
    output_dir = 'parameter_distribution_plots'
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"\n📊 Creating visualizations...")
    
    # 1. Comprehensive multi-parameter plot
    print("\n1️⃣ Creating comprehensive distribution plot...")
    plot_parameter_distributions(df, output_dir)
    
    # 2. Individual parameter plots
    print("\n2️⃣ Creating individual parameter plots...")
    plot_individual_parameter_distributions(df, output_dir)
    
    # 3. Boxplot comparisons
    print("\n3️⃣ Creating boxplot comparisons...")
    plot_parameter_boxplots(df, output_dir)
    
    # 4. Mechanism-specific plots
    print("\n4️⃣ Creating mechanism-specific plots...")
    plot_mechanism_specific_parameters(df, output_dir)
    
    # 5. NLL/AIC/BIC distributions
    print("\n5️⃣ Creating NLL/AIC/BIC distributions...")
    plot_nll_aic_bic_distributions(df, output_dir)
    
    # 6. Create summary statistics table
    print("\n6️⃣ Creating parameter statistics summary...")
    summary_df = create_parameter_summary_table(df, output_dir)
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\n📁 All plots saved to: {output_dir}/")
    print(f"📊 Files created:")
    print(f"   - all_parameters_distributions.png (comprehensive overview)")
    print(f"   - all_parameters_boxplots.png (boxplot comparison)")
    print(f"   - <param>_distribution.png (individual parameter plots)")
    print(f"   - <mechanism>_all_parameters.png (mechanism-specific plots)")
    print(f"   - nll_aic_bic_distributions.png (model fit metrics)")
    print(f"   - parameter_statistics_summary.csv (summary table)")


if __name__ == "__main__":
    main()

