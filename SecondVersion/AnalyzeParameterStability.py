
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from MoMOptimization_join import run_mom_optimization_single, unpack_parameters, get_mechanism_info

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.optimize")
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


def load_data():
    """Load data arrays once to avoid repeated file I/O."""
    try:
        df = pd.read_excel("Data/All_strains_SCStimes.xlsx")
        data_arrays = {
            'data_wt12': df['wildtype12'].dropna().values,
            'data_wt32': df['wildtype32'].dropna().values,
            'data_threshold12': df['threshold12'].dropna().values,
            'data_threshold32': df['threshold32'].dropna().values,
            'data_degrate12': df['degRade12'].dropna().values,
            'data_degrate32': df['degRade32'].dropna().values,
            'data_initial12': df['initialProteins12'].dropna().values if 'initialProteins12' in df.columns else np.array([]),
            'data_initial32': df['initialProteins32'].dropna().values if 'initialProteins32' in df.columns else np.array([]),
            'data_degrateAPC12': df['degRadeAPC12'].dropna().values,
            'data_degrateAPC32': df['degRadeAPC32'].dropna().values,
            'data_velcade12': df['degRadeVel12'].dropna().values,
            'data_velcade32': df['degRadeVel32'].dropna().values
        }
        return data_arrays
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def main():
    # ========== CONFIGURATION ==========
    mechanism = 'fixed_burst_feedback_onion'       # Options: 'simple', 'fixed_burst', 'feedback_onion', 'fixed_burst_feedback_onion'
    tol = 1e-2                 # Optimal tolerance found from analysis (adjust as needed)
    num_runs = 10               # Number of repetitions
    # ===================================

    print(f"Starting Parameter Stability Analysis")
    print(f"Mechanism: {mechanism}")
    print(f"Tolerance: {tol}")
    print(f"Runs: {num_runs}")

    data_arrays = load_data()
    if data_arrays is None:
        return

    # Get parameter names from mechanism info
    mech_info = get_mechanism_info(mechanism, 'unified')
    # Create a mapping of parameter name to bounds
    # mech_info['params'] corresponds to mech_info['bounds']
    param_bounds_map = dict(zip(mech_info['params'], mech_info['bounds']))

    # Use dummy params to get full key list including derived ones if unpack_parameters handles them,
    # but unpack_parameters logic adds derived ones (n1, n3, N1, N3).
    # We want to track optimized params + derived ones.
    # Let's run one dummy unpack to get all keys.
    dummy_x = [1.0] * len(mech_info['params'])
    dummy_dict = unpack_parameters(dummy_x, mech_info)
    param_keys = list(dummy_dict.keys())
    
    # Store results
    all_params = [] # List of dicts

    for i in range(num_runs):
        seed = 2 + i
        print(f"Run {i+1}/{num_runs} (seed={seed})...", end="", flush=True)
        
        res = run_mom_optimization_single(
            mechanism=mechanism,
            data_arrays=data_arrays,
            max_iterations=400,
            seed=seed,
            gamma_mode='unified',
            tol=tol
        )
        
        if res['success']:
            # Unpack full parameters including derived ones
            params = unpack_parameters(res['result'].x, res['mechanism_info'])
            # Add NLL for reference
            params['nll'] = res['nll']
            # Add run ID for tracking/coloring
            params['Run'] = f"Run {i+1}"
            all_params.append(params)
            print(f" Done. NLL: {res['nll']:.4f}")
        else:
            print(f" Failed.")

    if not all_params:
        print("No successful runs.")
        return

    # Convert to DataFrame
    df_results = pd.DataFrame(all_params)
    
    # Save statistics
    stats_file = f"SecondVersion/parameter_stats_{mechanism}.txt"
    with open(stats_file, 'w') as f:
        f.write(f"Parameter Stability Statistics ({num_runs} runs)\n")
        f.write(f"Mechanism: {mechanism}, tol: {tol}\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Parameter':<15} {'Mean':<10} {'Std':<10} {'CV(%)':<10}\n")
        f.write("-" * 50 + "\n")
        
        # Calculate stats for numeric columns only (exclude 'Run')
        numeric_cols = [c for c in df_results.columns if c != 'Run']
        stats = df_results[numeric_cols].agg(['mean', 'std'])
        for col in numeric_cols:
            mean = stats.loc['mean', col]
            std = stats.loc['std', col]
            cv = (std / mean * 100) if mean != 0 else 0.0
            f.write(f"{col:<15} {mean:<10.4f} {std:<10.4f} {cv:<10.2f}\n")
            
    print(f"\nStatistics saved to {stats_file}")

    # Plotting
    # Filter out unwanted columns for plotting if many derived ones are redundant or constant
    # Plot all numeric parameters except NLL
    plot_cols = [c for c in df_results.columns if c not in ['nll', 'Run']]
    n_params = len(plot_cols)
    
    # Dynamic grid layout
    cols = 4
    rows = (n_params + cols - 1) // cols
    
    plt.figure(figsize=(15, 4 * rows))
    
    # Create a consistent palette for runs
    unique_runs = df_results['Run'].unique()
    palette = sns.color_palette("husl", len(unique_runs))
    
    for idx, param in enumerate(plot_cols):
        ax = plt.subplot(rows, cols, idx + 1)
        
        # Box plot
        sns.boxplot(y=df_results[param], ax=ax, color='lightblue', showfliers=False)
        # Strip plot (jitter) to show individual points, colored by Run
        sns.stripplot(data=df_results, y=param, hue='Run', palette=palette, ax=ax, alpha=0.8, jitter=True, legend=False)
        
        # Set y-limits to parameter bounds if available
        if param in param_bounds_map:
            lower, upper = param_bounds_map[param]
            # Add a small margin if needed, but 'within its bounds' usually means visible bounds
            # Let's set the limits exactly to bounds to see if params hit the walls
            ax.set_ylim(lower, upper)
        
        ax.set_title(param)
        ax.set_ylabel('')
        
        # Add CV to title
        mean = df_results[param].mean()
        std = df_results[param].std()
        cv = (std / mean * 100) if mean != 0 else 0
        ax.set_xlabel(f"CV: {cv:.1f}%")

    plt.tight_layout()
    plot_file = f"SecondVersion/parameter_distribution_{mechanism}.png"
    plt.savefig(plot_file)
    print(f"Distribution plot saved to {plot_file}")

if __name__ == "__main__":
    main()
