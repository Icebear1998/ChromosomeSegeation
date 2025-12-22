
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MoMOptimization_join import run_mom_optimization_single

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
    mechanisms = ['simple', 'fixed_burst', 'feedback_onion']
    
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
        
    num_runs = 10
    
    data_arrays = load_data()
    if data_arrays is None:
        return

    # Use x-axis labels instead of just tol values for plotting
    pair_labels = [f"({t:.1e}, {a:.1e})" for t, a in tol_atol_pairs]
    # For plotting numerically, we can just use index 0, 1, 2...
    x_indices = np.arange(len(tol_atol_pairs))

    results = {mech: {'avg_nlls': [], 'std_nlls': []} for mech in mechanisms}

    print("Starting tolerance analysis (tol, atol)...")
    print(f"Mechanisms: {mechanisms}")
    print(f"Pairs (tol, atol): {tol_atol_pairs}")

    for mech in mechanisms:
        print(f"\nAnalyzing mechanism: {mech}")
        for tol, atol in tol_atol_pairs:
            nlls = []
            print(f"  Testing (tol={tol:.1e}, atol={atol:.1e}): ", end="", flush=True)
            for i in range(num_runs):
                seed = 42 + i # Different seed for each run
                res = run_mom_optimization_single(
                    mechanism=mech,
                    data_arrays=data_arrays,
                    max_iterations=400,
                    seed=seed,
                    gamma_mode='unified',
                    tol=tol,
                    atol=atol
                )
                if res['success']:
                    nlls.append(res['nll'])
                    print(".", end="", flush=True)
                else:
                    print("x", end="", flush=True)
            
            if nlls:
                avg_nll = np.mean(nlls)
                std_nll = np.std(nlls)
                results[mech]['avg_nlls'].append(avg_nll)
                results[mech]['std_nlls'].append(std_nll)
                print(f" Mean NLL: {avg_nll:.4f}")
            else:
                results[mech]['avg_nlls'].append(np.nan)
                results[mech]['std_nlls'].append(np.nan)
                print(" All failed")

    # Plotting
    plt.figure(figsize=(12, 6))
    for mech in mechanisms:
        avg_nlls = results[mech]['avg_nlls']
        plt.plot(x_indices, avg_nlls, marker='o', label=mech)
    
    plt.xticks(x_indices, pair_labels, rotation=45, ha='right')
    plt.xlabel('(tol, atol)')
    plt.ylabel('Average Negative Log-Likelihood (NLL)')
    plt.title('Optimization Efficiency: (tol, atol) vs NLL')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_file = 'SecondVersion/tol_efficiency_plot.png'
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")

if __name__ == "__main__":
    main()
