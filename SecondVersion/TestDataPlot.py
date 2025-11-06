import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MultiMechanismSimulation import MultiMechanismSimulation
from MoMCalculations import compute_pdf_for_mechanism


def build_rate_params(mechanism, k, mech_params):
    """
    Build rate_params dictionary for simulation based on mechanism.
    """
    if mechanism == 'simple':
        return {'k': k}
    elif mechanism == 'fixed_burst':
        rate_params = {'k': k}
        rate_params.update(mech_params)
        return rate_params
    elif mechanism == 'time_varying_k':
        rate_params = {'k': k}
        rate_params.update(mech_params)
        return rate_params
    elif mechanism == 'feedback':
        rate_params = {'k': k}
        rate_params.update(mech_params)
        return rate_params
    elif mechanism == 'feedback_linear':
        rate_params = {'k': k}
        rate_params.update(mech_params)
        return rate_params
    elif mechanism == 'feedback_onion':
        rate_params = {'k': k}
        rate_params.update(mech_params)
        return rate_params
    elif mechanism == 'feedback_zipper':
        rate_params = {'k': k}
        rate_params.update(mech_params)
        return rate_params
    elif mechanism == 'fixed_burst_feedback_linear':
        rate_params = {'k': k}
        rate_params.update(mech_params)
        return rate_params
    elif mechanism == 'fixed_burst_feedback_onion':
        rate_params = {'k': k}
        rate_params.update(mech_params)
        return rate_params
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")


def load_parameters(filename):
    """Load optimized parameters from file."""
    params = {}
    mechanism = None

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# Mechanism:"):
                mechanism = line.split(": ")[1]
            elif line.startswith("#") or not line:
                continue
            elif ": " in line:
                key, value = line.split(": ", 1)
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value

    if mechanism:
        params['mechanism'] = mechanism

    return params


def load_dataset(df, dataset):
    """Load experimental dataset."""
    if dataset == "wildtype":
        return df['wildtype12'].dropna().values, df['wildtype32'].dropna().values
    elif dataset == "threshold":
        return df['threshold12'].dropna().values, df['threshold32'].dropna().values
    elif dataset == "degrate":
        return df['degRade12'].dropna().values, df['degRade32'].dropna().values
    elif dataset == "degrateAPC":
        return df['degRadeAPC12'].dropna().values, df['degRadeAPC32'].dropna().values
    elif dataset == "velcade":
        return df['degRadeVel12'].dropna().values, df['degRadeVel32'].dropna().values
    elif dataset == "initial":
        return df['initialProteins12'].dropna().values, df['initialProteins32'].dropna().values
    else:
        raise ValueError(
            "Invalid dataset. Choose 'wildtype', 'threshold', 'degrate', 'degrateAPC', 'velcade', or 'initial'.")


def apply_mutant_params(params, dataset):
    """Apply mutant parameters based on dataset type."""
    n1, n2, n3 = params['n1'], params['n2'], params['n3']
    N1, N2, N3 = params['N1'], params['N2'], params['N3']
    k = params['k']

    if dataset == "threshold":
        alpha = params['alpha']
        n1 = max(params['n1'] * alpha, 1)
        n2 = max(params['n2'] * alpha, 1)
        n3 = max(params['n3'] * alpha, 1)
    elif dataset == "degrate":
        beta_k = params['beta_k']
        k = max(beta_k * params['k'], 0.001)
    elif dataset == "degrateAPC":
        beta2_k = params['beta2_k']
        k = max(beta2_k * params['k'], 0.001)
    elif dataset == "velcade":
        beta3_k = params['beta3_k']
        k = max(beta3_k * params['k'], 0.001)
    elif dataset == "initial":
        if 'gamma' in params:  # unified mode
            gamma = params['gamma']
            N1 = max(params['N1'] * gamma, 1)
            N2 = max(params['N2'] * gamma, 1)
            N3 = max(params['N3'] * gamma, 1)
        else:  # separate mode
            gamma1 = params['gamma1']
            gamma2 = params['gamma2']
            gamma3 = params['gamma3']
            N1 = max(params['N1'] * gamma1, 1)
            N2 = max(params['N2'] * gamma2, 1)
            N3 = max(params['N3'] * gamma3, 1)

    return n1, n2, n3, N1, N2, N3, k


def extract_mechanism_params(params, mechanism):
    """Extract mechanism-specific parameters from params dictionary."""
    mech_params = {}

    if mechanism == 'fixed_burst':
        mech_params['burst_size'] = params.get('burst_size', 8)
    elif mechanism == 'time_varying_k':
        mech_params['k_1'] = params.get('k_1', 0.001)
    elif mechanism == 'feedback':
        mech_params['feedbackSteepness'] = params.get(
            'feedbackSteepness', 0.04)
        mech_params['feedbackThreshold'] = params.get('feedbackThreshold', 75)
    elif mechanism == 'feedback_linear':
        mech_params['w1'] = params.get('w1')
        mech_params['w2'] = params.get('w2')
        mech_params['w3'] = params.get('w3')
    elif mechanism == 'feedback_onion':
        mech_params['n_inner'] = params.get('n_inner', 10)
    elif mechanism == 'feedback_zipper':
        mech_params['z1'] = params.get('z1', 40)
        mech_params['z2'] = params.get('z2', 50)
        mech_params['z3'] = params.get('z3', 60)
    elif mechanism == 'fixed_burst_feedback_linear':
        mech_params['burst_size'] = params.get('burst_size', 8)
        mech_params['w1'] = params.get('w1')
        mech_params['w2'] = params.get('w2')
        mech_params['w3'] = params.get('w3')
    elif mechanism == 'fixed_burst_feedback_onion':
        mech_params['burst_size'] = params.get('burst_size', 5)
        mech_params['n_inner'] = params.get('n_inner', 50)
    # simple mechanism doesn't need additional parameters

    return mech_params


def run_stochastic_simulation(mechanism, k, n1, n2, n3, N1, N2, N3, mech_params, max_time=500, num_sim=1500):
    """Run stochastic simulation for any mechanism."""
    initial_proteins = [N1, N2, N3]
    rate_params = build_rate_params(mechanism, k, mech_params)

    n0_list = [n1, n2, n3]

    simulation = MultiMechanismSimulation(
        mechanism=mechanism,
        initial_state_list=initial_proteins,
        rate_params=rate_params,
        n0_list=n0_list,
        max_time=max_time
    )

    delta_t12 = []
    delta_t32 = []
    for _ in range(num_sim):
        _, _, sep_times = simulation.simulate()
        delta_t12.append(sep_times[0] - sep_times[1])
        delta_t32.append(sep_times[2] - sep_times[1])

    return np.array(delta_t12), np.array(delta_t32)


def plot_results(params, dataset="wildtype", mechanism=None, data_file="Data/All_strains_SCStimes.xlsx", num_sim=1500):
    """
    Plot comparison between experimental data, simulated data, and MoM PDF.

    Args:
        params (dict): Parameter dictionary loaded from file
        dataset (str): Dataset type ('wildtype', 'threshold', 'degrate', 'initial')
        mechanism (str, optional): Mechanism to use. If None, uses mechanism from params
        data_file (str): Path to experimental data file
        num_sim (int): Number of simulations to run
    """
    # Determine mechanism
    if mechanism is None:
        mechanism = params.get('mechanism', 'simple')

    print(f"Plotting results for mechanism: {mechanism}, dataset: {dataset}")

    # Load experimental data
    df = pd.read_excel(data_file)
    data12, data32 = load_dataset(df, dataset)

    # Apply mutant parameters
    n1, n2, n3, N1, N2, N3, k = apply_mutant_params(params, dataset)

    # Extract mechanism-specific parameters
    mech_params = extract_mechanism_params(params, mechanism)

    # Validate parameters
    if mechanism == 'time_varying_k' and mech_params['k_1'] <= 0:
        raise ValueError(
            "k_1 must be greater than 0 for time_varying_k mechanism.")
    if mechanism == 'fixed_burst' and mech_params['burst_size'] <= 0:
        raise ValueError(
            "burst_size must be greater than 0 for fixed_burst mechanism.")
    if mechanism == 'feedback_onion':
        if mech_params['n_inner'] <= 0:
            raise ValueError(f"Invalid n_inner parameter: {mech_params['n_inner']}")
    if mechanism == 'feedback_zipper':
        if any(mech_params[param] <= 0 for param in ['z1', 'z2', 'z3']):
            raise ValueError(
                "z parameters must be greater than 0 for feedback_zipper mechanism.")
    if mechanism in ['fixed_burst_feedback_linear', 'fixed_burst_feedback_onion']:
        if mech_params['burst_size'] <= 0:
            raise ValueError(
                f"burst_size must be greater than 0 for {mechanism} mechanism.")
    if mechanism == 'fixed_burst_feedback_onion':
        if mech_params['burst_size'] <= 0:
            raise ValueError(f"Invalid burst_size parameter: {mech_params['burst_size']}")
        if mech_params['n_inner'] <= 0:
            raise ValueError(f"Invalid n_inner parameter: {mech_params['n_inner']}")

    print(
        f"Parameters: n1={n1:.1f}, n2={n2:.1f}, n3={n3:.1f}, N1={N1:.1f}, N2={N2:.1f}, N3={N3:.1f}, k={k:.4f}")
    if mech_params:
        print(f"Mechanism-specific: {mech_params}")

    # Set up x_grid for PDF plotting
    x_min, x_max = -140, 140  # Default range for all mechanisms
    if dataset == "velcade":
        x_min, x_max = -350, 350
    if dataset in ["wildtype", "threshold"]:
        x_min, x_max = -100, 100    
    x_grid = np.linspace(x_min, x_max, 401)

    # Compute MoM PDFs
    pdf12 = compute_pdf_for_mechanism(
        mechanism, x_grid, n1, N1, n2, N2, k, mech_params, pair12=True)
    pdf32 = compute_pdf_for_mechanism(
        mechanism, x_grid, n3, N3, n2, N2, k, mech_params, pair12=False)

    # Run stochastic simulation
    print(f"Running {num_sim} simulations...")
    delta_t12, delta_t32 = run_stochastic_simulation(
        mechanism, k, n1, n2, n3, N1, N2, N3, mech_params, num_sim=num_sim)

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Chrom1 - Chrom2
    ax1.hist(data12, bins=15, density=True,
             alpha=0.6, label='Experimental data', color='lightblue')
    ax1.hist(delta_t12, bins=15, density=True,
             alpha=0.6, label='Simulated data', color='orange')
    ax1.plot(x_grid, pdf12, 'r-', linewidth=2, label='MoM PDF')
    ax1.set_xlim(x_min - 20, x_max + 20)
    ax1.set_xlabel('Time Difference')
    ax1.set_ylabel('Density')
    ax1.set_title(
        f"Chrom1 - Chrom2 ({dataset}, {mechanism.replace('_', ' ').title()})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add statistics for Chrom1 - Chrom2
    stats_text12 = f'Exp: μ={np.mean(data12):.1f}, σ={np.std(data12):.1f}\nSim: μ={np.mean(delta_t12):.1f}, σ={np.std(delta_t12):.1f}'
    ax1.text(0.02, 0.98, stats_text12, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot Chrom3 - Chrom2
    ax2.hist(data32, bins=15, density=True,
             alpha=0.6, label='Experimental data', color='lightblue')
    ax2.hist(delta_t32, bins=15, density=True,
             alpha=0.6, label='Simulated data', color='orange')
    ax2.plot(x_grid, pdf32, 'r-', linewidth=2, label='MoM PDF')
    ax2.set_xlim(x_min - 20, x_max + 20)
    ax2.set_xlabel('Time Difference')
    ax2.set_ylabel('Density')
    ax2.set_title(
        f"Chrom3 - Chrom2 ({dataset}, {mechanism.replace('_', ' ').title()})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add statistics for Chrom3 - Chrom2
    stats_text32 = f'Exp: μ={np.mean(data32):.1f}, σ={np.std(data32):.1f}\nSim: μ={np.mean(delta_t32):.1f}, σ={np.std(delta_t32):.1f}'
    ax2.text(0.02, 0.98, stats_text32, transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

    # Print some statistics
    print(f"\nStatistics for {dataset} dataset:")
    print(
        f"Experimental data12 - Mean: {np.mean(data12):.2f}, Std: {np.std(data12):.2f}")
    print(
        f"Simulated data12   - Mean: {np.mean(delta_t12):.2f}, Std: {np.std(delta_t12):.2f}")
    print(
        f"Experimental data32 - Mean: {np.mean(data32):.2f}, Std: {np.std(data32):.2f}")
    print(
        f"Simulated data32   - Mean: {np.mean(delta_t32):.2f}, Std: {np.std(delta_t32):.2f}")


def plot_all_datasets(params_file, mechanism=None, num_sim=1000):
    """
    Plot all datasets in a 2x4 layout with each strain's Chrom1-2 and Chrom3-2 side by side.
    Row 1: wildtype12, wildtype32, initial12, initial32
    Row 2: threshold12, threshold32, degrate12, degrate32

    Args:
        params_file (str): Path to parameter file
        mechanism (str, optional): Mechanism to use. If None, uses mechanism from file
        num_sim (int): Number of simulations per dataset
    """
    params = load_parameters(params_file)

    if mechanism is None:
        mechanism = params.get('mechanism', 'simple')

    print(f"Plotting all datasets for mechanism: {mechanism}")
    print(f"Using parameters from: {params_file}")

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
    # Row 1
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
            
            # Run stochastic simulation
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
    plt.show()


def plot_all_datasets_2x2(params, mechanism=None, data_file="Data/All_strains_SCStimes.xlsx", num_sim=1500):
    """
    Plot all six datasets in a 2x6 layout: wildtype, threshold, degrate, degrateAPC, velcade, initial.
    Top row: Chrom1-Chrom2, Bottom row: Chrom3-Chrom2
    
    Args:
        params (dict): Parameter dictionary loaded from file
        mechanism (str, optional): Mechanism to use. If None, uses mechanism from params
        data_file (str): Path to experimental data file
        num_sim (int): Number of simulations to run
    """
    # Determine mechanism
    if mechanism is None:
        mechanism = params.get('mechanism', 'simple')

    print(f"Plotting all datasets for mechanism: {mechanism}")

    # Load experimental data
    df = pd.read_excel(data_file)
    
    # Dataset configuration
    datasets = ['wildtype', 'threshold', 'degrate', 'degrateAPC', 'velcade', 'initial']
    dataset_titles = ['wildtype', 'threshold', 'degrate', 'degrateAPC', 'velcade', 'initial']
    
    # Create 2x6 subplot layout (2 rows, 6 columns)
    fig, axes = plt.subplots(2, 6, figsize=(30, 10))
    fig.suptitle(f'Chromosome Segregation Times - {mechanism.replace("_", " ").title()} Mechanism', fontsize=16, y=0.95)
    
    # Set up x_grid for PDF plotting
    x_min, x_max = -350, 350
    x_grid = np.linspace(x_min, x_max, 401)
    
    for i, dataset in enumerate(datasets):
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
            
            # Run stochastic simulation
            delta_t12, delta_t32 = run_stochastic_simulation(
                mechanism, k, n1, n2, n3, N1, N2, N3, mech_params, num_sim=num_sim)
            
            # Plot Chrom1 - Chrom2 (top row)
            ax1 = axes[0, i]
            ax1.hist(data12, bins=15, density=True, alpha=0.6, label='Experimental data', color='lightblue')
            ax1.hist(delta_t12, bins=15, density=True, alpha=0.6, label='Simulated data', color='orange')
            ax1.plot(x_grid, pdf12, 'r-', linewidth=2, label='MoM PDF')
            ax1.set_xlim(-350, 350)
            ax1.set_xlabel('Time Difference')
            ax1.set_ylabel('Density')
            ax1.set_title(f'Chrom1 - Chrom2 ({dataset_titles[i]})')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text12 = f'Exp: μ={np.mean(data12):.1f}, σ={np.std(data12):.1f}\nSim: μ={np.mean(delta_t12):.1f}, σ={np.std(delta_t12):.1f}'
            ax1.text(0.02, 0.98, stats_text12, transform=ax1.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)
            
            # Plot Chrom3 - Chrom2 (bottom row)
            ax2 = axes[1, i]
            ax2.hist(data32, bins=15, density=True, alpha=0.6, label='Experimental data', color='lightblue')
            ax2.hist(delta_t32, bins=15, density=True, alpha=0.6, label='Simulated data', color='orange')
            ax2.plot(x_grid, pdf32, 'r-', linewidth=2, label='MoM PDF')
            ax2.set_xlim(-350, 350)
            ax2.set_xlabel('Time Difference')
            ax2.set_ylabel('Density')
            ax2.set_title(f'Chrom3 - Chrom2 ({dataset_titles[i]})')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text32 = f'Exp: μ={np.mean(data32):.1f}, σ={np.std(data32):.1f}\nSim: μ={np.mean(delta_t32):.1f}, σ={np.std(delta_t32):.1f}'
            ax2.text(0.02, 0.98, stats_text32, transform=ax2.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)
            
            print(f"✓ Successfully plotted {dataset} dataset")
            
        except Exception as e:
            print(f"✗ Error plotting {dataset}: {e}")
            # Clear the axes if there's an error
            axes[0, i].text(0.5, 0.5, f'Error plotting {dataset}', 
                           transform=axes[0, i].transAxes, ha='center', va='center')
            axes[1, i].text(0.5, 0.5, f'Error plotting {dataset}', 
                           transform=axes[1, i].transAxes, ha='center', va='center')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()


if __name__ == "__main__":
    # ========== CONFIGURATION ==========
    # Specify parameter file and mechanism
    # For mechanism-specific files: "optimized_parameters_{mechanism}.txt"
    # For independent optimization: "optimized_parameters_independent_{mechanism}.txt"

    # Change this to your parameter file
    params_file = "optimized_parameters_simple_join.txt"
    #params_file = "optimized_parameters_simple_join_test1.txt"
    # Set to None to use mechanism from file, or specify: 'simple', 'fixed_burst', 'time_varying_k', 'feedback_onion', 'fixed_burst_feedback_onion'
    mechanism = None  # Set to None to use mechanism from file
    
    dataset = "velcade"  # Choose: 'wildtype', 'threshold', 'degrate', 'degrateAPC', 'velcade', 'initial'

    # ========== SINGLE PLOT ==========
    # Plot single dataset
    try:
        params = load_parameters(params_file)
        plot_results(params, dataset=dataset,
                     mechanism=mechanism, num_sim=1500)
    except FileNotFoundError:
        print(f"Parameter file not found: {params_file}")
        print("Available parameter files should follow the pattern:")
        print("  - optimized_parameters_{mechanism}.txt")
        print("  - optimized_parameters_independent_{mechanism}.txt")
    except Exception as e:
        print(f"Error: {e}")

    # ========== ALL DATASETS ==========
    # Uncomment to plot all datasets in 2x4 layout (each strain's Chrom1-2 and Chrom3-2 side by side)
    # plot_all_datasets(params_file, mechanism=mechanism, num_sim=1000)

    # ========== ALL DATASETS 2x6 ==========
    # Plot all six datasets in a 2x6 layout (includes velcade dataset)
    # try:
    #     params = load_parameters(params_file)
    #     plot_all_datasets_2x2(params, mechanism=mechanism, data_file="Data/All_strains_SCStimes.xlsx", num_sim=1500)
    # except FileNotFoundError:
    #     print(f"Parameter file not found: {params_file}")
    #     print("Available parameter files should follow the pattern:")
    #     print("  - optimized_parameters_{mechanism}.txt")
    #     print("  - optimized_parameters_independent_{mechanism}.txt")
    # except Exception as e:
    #     print(f"Error: {e}")
