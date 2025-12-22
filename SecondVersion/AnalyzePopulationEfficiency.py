
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from MoMOptimization_join import run_mom_optimization_single

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.optimize")
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

def load_data():
    """Load data arrays once to avoid repeated file I/O."""
    try:
        # Try finding file in likely locations
        file_path = "Data/All_strains_SCStimes.xlsx"
        # If not found, try copy if it exists (from previous debugging)
        # But for now assume standard path.
        
        df = pd.read_excel(file_path)
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

def analyze_population_efficiency():
    mechanisms = ['simple', 'fixed_burst', 'feedback_onion']
    
    pop_sizes = [5, 10, 15, 20, 30, 40]
    num_runs = 10
    
    # Use efficient tol/atol found previously or defaults
    # For now, use robust defaults or what was used in stability analysis
    tol = 1e-4
    atol = 0
    
    data_arrays = load_data()
    if data_arrays is None:
        return

    results = {
        'Mechanism': [],
        'PopSize': [],
        'NLL': [],
        'Time': [],
        'Success': []
    }

    print("Starting Population Efficiency Analysis...")
    print(f"Mechanism: {mechanisms}")
    print(f"PopSizes: {pop_sizes}")
    print(f"Runs per size: {num_runs}")

    for mech in mechanisms:
        print(f"\nAnalyzing mechanism: {mech}")
        for pop in pop_sizes:
            print(f"  Testing PopSize={pop}: ", end="", flush=True)
            for i in range(num_runs):
                seed = 42 + i + pop # Unique seed
                
                start_time = time.time()
                res = run_mom_optimization_single(
                    mechanism=mech,
                    data_arrays=data_arrays,
                    max_iterations=400,
                    seed=seed,
                    gamma_mode='unified',
                    tol=tol,
                    atol=atol,
                    popsize=pop
                )
                elapsed = time.time() - start_time
                
                results['Mechanism'].append(mech)
                results['PopSize'].append(pop)
                results['Time'].append(elapsed)
                results['Success'].append(res['success'])
                
                if res['success']:
                    results['NLL'].append(res['nll'])
                    print(".", end="", flush=True)
                else:
                    results['NLL'].append(np.nan)
                    print("x", end="", flush=True)
            print(f" Done.")

    df = pd.DataFrame(results)
    
    # Calculate stats for plotting
    summary = df.groupby(['Mechanism', 'PopSize']).agg({
        'NLL': ['mean', 'min', 'std'],
        'Time': ['mean', 'std']
    }).reset_index()
    # Flatten columns
    summary.columns = ['Mechanism', 'PopSize', 'NLL_mean', 'NLL_min', 'NLL_std', 'Time_mean', 'Time_std']
    
    # Save CSV
    csv_file = "SecondVersion/population_efficiency_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nResults saved to {csv_file}")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: NLL vs PopSize
    # We can plot mean with error bars, and maybe the min found
    for mech in mechanisms:
        mech_data = summary[summary['Mechanism'] == mech]
        ax1.errorbar(mech_data['PopSize'], mech_data['NLL_mean'], yerr=mech_data['NLL_std'], 
                     fmt='o-', label=f'{mech} (Mean)', capsize=5)
        # Also plot min NLL to see if large populations find better minima
        ax1.plot(mech_data['PopSize'], mech_data['NLL_min'], 'x--', label=f'{mech} (Min)', alpha=0.7)

    ax1.set_xlabel('Population Size')
    ax1.set_ylabel('Negative Log-Likelihood (NLL)')
    ax1.set_title('Optimization Quality vs Population Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time vs PopSize
    for mech in mechanisms:
        mech_data = summary[summary['Mechanism'] == mech]
        ax2.errorbar(mech_data['PopSize'], mech_data['Time_mean'], yerr=mech_data['Time_std'],
                     fmt='o-', label=f'{mech}')
                     
    ax2.set_xlabel('Population Size')
    ax2.set_ylabel('Time per Run (s)')
    ax2.set_title('Computational Cost vs Population Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = "SecondVersion/population_efficiency_plot.png"
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")

if __name__ == "__main__":
    analyze_population_efficiency()
