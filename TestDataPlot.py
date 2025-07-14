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
        return df['degRate12'].dropna().values, df['degRate32'].dropna().values
    elif dataset == "initial":
        return df['initialProteins12'].dropna().values, df['initialProteins32'].dropna().values
    else:
        raise ValueError(
            "Invalid dataset. Choose 'wildtype', 'threshold', 'degrate', or 'initial'.")


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
    elif dataset == "initial":
        gamma = params['gamma']
        N1 = max(params['N1'] * gamma, 1)
        N2 = max(params['N2'] * gamma, 1)
        N3 = max(params['N3'] * gamma, 1)

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
             alpha=0.4, label='Experimental data')
    ax1.hist(delta_t12, bins=15, density=True,
             alpha=0.4, label='Simulated data')
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
    ax2.hist(data32, bins=14, density=True,
             alpha=0.4, label='Experimental data')
    ax2.hist(delta_t32, bins=14, density=True,
             alpha=0.4, label='Simulated data')
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
    Plot all datasets for a given parameter file and mechanism.

    Args:
        params_file (str): Path to parameter file
        mechanism (str, optional): Mechanism to use. If None, uses mechanism from file
        num_sim (int): Number of simulations per dataset
    """
    params = load_parameters(params_file)

    if mechanism is None:
        mechanism = params.get('mechanism', 'simple')

    datasets = ['wildtype', 'threshold', 'degrate', 'initial']

    print(f"Plotting all datasets for mechanism: {mechanism}")
    print(f"Using parameters from: {params_file}")

    for dataset in datasets:
        print(f"\n{'='*50}")
        try:
            plot_results(params, dataset=dataset,
                         mechanism=mechanism, num_sim=num_sim)
        except Exception as e:
            print(f"Error plotting {dataset}: {e}")


if __name__ == "__main__":
    # ========== CONFIGURATION ==========
    # Specify parameter file and mechanism
    # For mechanism-specific files: "optimized_parameters_{mechanism}.txt"
    # For independent optimization: "optimized_parameters_independent_{mechanism}.txt"

    # Change this to your parameter file
    params_file = "fixed_burst_feedback_onion_join_fit2.txt"
    # Set to None to use mechanism from file, or specify: 'simple', 'fixed_burst', 'time_varying_k', 'feedback', 'feedback_linear', 'feedback_onion', 'feedback_zipper', 'fixed_burst_feedback_linear'
    mechanism = 'fixed_burst_feedback_onion'
    dataset = "wildtype"  # Choose: 'wildtype', 'threshold', 'degrate', 'initial'

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
    # Uncomment to plot all datasets
    # plot_all_datasets(params_file, mechanism=mechanism, num_sim=1000)
