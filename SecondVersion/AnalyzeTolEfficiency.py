
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Import for MoM optimization
from MoMOptimization_join import run_mom_optimization_single

# Imports for Fast simulation-based optimization
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SimulationOptimization_join import run_optimization
from simulation_utils import load_experimental_data

def load_data_mom():
    """Load data arrays for MoM optimization."""
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
    # ========== CHOOSE WHICH METHOD TO TEST ==========
    # Comment/uncomment one of the following:
    
    # Option 1: Test MoM mechanisms (analytical, fast but approximate)
    # MoM_mechanisms = ['simple', 'fixed_burst', 'feedback_onion']
    # mechanisms = MoM_mechanisms
    # use_simulation = False
    
    # Option 2: Test Fast simulation mechanisms (exact but slower)
    Fast_mechanisms = ['simple', 'fixed_burst', 'feedback_onion', 'time_varying_k']
    mechanisms = Fast_mechanisms
    use_simulation = True
    # =================================================
    
    # Construct sequence of (tol, atol) pairs
    # Start with default
    tol_atol_pairs = [(0.01, 0)]
    
    # Generate subsequent pairs: tol=10^-k, atol=10^-(k-2) for k=3..8
    # k=3: tol=1e-3, atol=1e-1 (0.1)
    # k=4: tol=1e-4, atol=1e-2 (0.01)
    # ...
    for k in range(2, 6):
        tol = 10**(-k)
        atol = 10**(-(k-2))
        tol_atol_pairs.append((tol, atol))
        
    num_runs = 5
    max_interations = 5000
    num_simulations = 10000  # Only used for simulation-based methods
    
    # Load data based on which method we're using
    if use_simulation:
        datasets = load_experimental_data()
        if not datasets:
            print("Error: Could not load experimental data!")
            return
        data_arrays = None
    else:
        data_arrays = load_data_mom()
        if data_arrays is None:
            return
        datasets = None

    # Use x-axis labels instead of just tol values for plotting
    pair_labels = [f"({t:.1e}, {a:.1e})" for t, a in tol_atol_pairs]
    # For plotting numerically, we can just use index 0, 1, 2...
    x_indices = np.arange(len(tol_atol_pairs))

    results = {mech: {'avg_nlls': [], 'std_nlls': [], 'converged_counts': []} for mech in mechanisms}

    print("Starting tolerance analysis (tol, atol)...")
    print(f"Mechanisms: {mechanisms}")
    print(f"Pairs (tol, atol): {tol_atol_pairs}")

    for mech in mechanisms:
        print(f"\nAnalyzing mechanism: {mech}")
        for tol, atol in tol_atol_pairs:
            nlls = []
            converged_count = 0
            print(f"  Testing (tol={tol:.1e}, atol={atol:.1e}): ", end="", flush=True)
            for i in range(num_runs):
                seed = 42 + i  # Different seed for each run
                
                if use_simulation:
                    # Use Fast simulation-based optimization
                    np.random.seed(seed)
                    res = run_optimization(
                        mechanism=mech,
                        datasets=datasets,
                        max_iterations=max_interations,
                        num_simulations=num_simulations,
                        selected_strains=None
                    )
                else:
                    # Use MoM optimization
                    res = run_mom_optimization_single(
                        mechanism=mech,
                        data_arrays=data_arrays,
                        max_iterations=max_interations,
                        seed=seed,
                        gamma_mode='unified',
                        tol=tol,
                        atol=atol
                    )
                
                if res['success']:
                    nlls.append(res['nll'])
                    # Check if it actually converged (not just hit max iterations)
                    if res.get('converged', False):
                        converged_count += 1
                        print(".", end="", flush=True)
                    else:
                        print("o", end="", flush=True)  # Success but didn't converge
                else:
                    print("x", end="", flush=True)
            
            if nlls:
                avg_nll = np.mean(nlls)
                std_nll = np.std(nlls)
                results[mech]['avg_nlls'].append(avg_nll)
                results[mech]['std_nlls'].append(std_nll)
                results[mech]['converged_counts'].append(converged_count)
                print(f" Mean NLL: {avg_nll:.4f}, Converged: {converged_count}/{num_runs}")
            else:
                results[mech]['avg_nlls'].append(np.nan)
                results[mech]['std_nlls'].append(np.nan)
                results[mech]['converged_counts'].append(0)
                print(" All failed")

    # Plotting
    plt.figure(figsize=(12, 6))
    for mech in mechanisms:
        avg_nlls = results[mech]['avg_nlls']
        converged_counts = results[mech]['converged_counts']
        
        # Plot the line and normal points
        plt.plot(x_indices, avg_nlls, marker='o', label=mech, alpha=0.7)
        
        # Overlay markers for non-converged points
        for i, (x_idx, nll, conv_count) in enumerate(zip(x_indices, avg_nlls, converged_counts)):
            if not np.isnan(nll) and conv_count < num_runs:
                # Mark points where not all runs converged
                plt.plot(x_idx, nll, 'rx', markersize=12, markeredgewidth=2)
                # Add annotation showing convergence rate
                plt.annotate(f'{conv_count}/{num_runs}', 
                            xy=(x_idx, nll),
                            xytext=(0, -15),
                            textcoords='offset points',
                            ha='center',
                            fontsize=8,
                            color='red',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    plt.xticks(x_indices, pair_labels, rotation=45, ha='right')
    plt.xlabel('(tol, atol)')
    plt.ylabel('Average Negative Log-Likelihood (NLL)')
    plt.title('Optimization Efficiency: (tol, atol) vs NLL\n(Red X = max_iterations reached, yellow labels show convergence rate)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    method_name = 'simulation' if use_simulation else 'mom'
    output_file = f'SecondVersion/tol_efficiency_plot_{method_name}.png'
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")
    print("\nLegend: '.' = converged, 'o' = completed but didn't converge, 'x' = failed")
    print("Red X markers indicate points where < all runs converged (hit max_iterations)")

if __name__ == "__main__":
    main()
