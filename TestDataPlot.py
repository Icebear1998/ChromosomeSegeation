import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from Chromosome_Gillespie4 import run_simulations, generate_threshold_values

# Compute MoM moments for f_X


def compute_moments_mom(n1, N1, n2, N2, k):
    sum1_T1 = sum(1/m for m in range(int(n1) + 1, int(N1) + 1))
    sum1_T2 = sum(1/m for m in range(int(n2) + 1, int(N2) + 1))
    sum2_T1 = sum(1/(m**2) for m in range(int(n1) + 1, int(N1) + 1))
    sum2_T2 = sum(1/(m**2) for m in range(int(n2) + 1, int(N2) + 1))

    mean_T1 = sum1_T1 / k
    mean_T2 = sum1_T2 / k
    var_T1 = sum2_T1 / (k**2)
    var_T2 = sum2_T2 / (k**2)

    mean_X = mean_T1 - mean_T2
    var_X = var_T1 + var_T2
    return mean_X, var_X


def run_stochastic_simulation(k, n1, n2, n3, N1, N2, N3,
                              data12, data32,
                              max_time=500, num_sim=1500):
    """
    Run a Gillespie-like simulation with the given parameters and compare to experimental data.
    """
    initial_proteins = [N1, N2, N3]
    rates = [k, k, k]
    n0_total = n1 + n2 + n3
    n0_list = generate_threshold_values([n1, n2], n0_total, num_sim)

    simulations, separate_times = run_simulations(
        initial_proteins, rates, n0_list, max_time, num_sim
    )

    # Compute time differences
    delta_t12 = [sep[0] - sep[1] for sep in separate_times]
    delta_t32 = [sep[2] - sep[1] for sep in separate_times]

    return delta_t12, delta_t32


def load_parameters(filename="optimized_parameters.txt"):
    """
    Read optimized parameters from a text file.
    """
    params = {}
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            key, value = line.strip().split(": ")
            params[key] = float(value)
    return params


def plot_results(params, dataset="wildtype", data_file="Data/All_strains_SCStimes.xlsx"):
    """
    Plot experimental data, simulated data, and MoM normal PDF for the specified dataset.
    """
    # Load data
    df = pd.read_excel(data_file)
    if dataset == "wildtype":
        data12 = df['wildtype12'].dropna().values
        data32 = df['wildtype32'].dropna().values
        n1, n2, n3 = params['n1'], params['n2'], params['n3']
        N1, N2, N3 = params['N1'], params['N2'], params['N3']
        k = params['k']
    elif dataset == "threshold":
        data12 = df['threshold12'].dropna().values
        data32 = df['threshold32'].dropna().values
        alpha = params['alpha']
        n1 = max(params['n1'] - alpha, 1)
        n2 = max(params['n2'] - alpha, 1)
        n3 = max(params['n3'] - alpha, 1)
        N1, N2, N3 = params['N1'], params['N2'], params['N3']
        k = params['k']
    elif dataset == "degrate":
        data12 = df['degRate12'].dropna().values
        data32 = df['degRate32'].dropna().values
        beta_k = params['beta_k']
        n1, n2, n3 = params['n1'], params['n2'], params['n3']
        N1, N2, N3 = params['N1'], params['N2'], params['N3']
        k = max(beta_k * params['k'], 0.001)
    elif dataset == "initial":
        data12 = df['initialProteins12'].dropna().values
        data32 = df['initialProteins32'].dropna().values
        gamma = params['gamma']
        n1, n2, n3 = params['n1'], params['n2'], params['n3']
        N1 = max(params['N1'] - gamma, 1)
        N2 = max(params['N2'] - gamma, 1)
        N3 = max(params['N3'] - gamma, 1)
        k = params['k']
    else:
        raise ValueError(
            "Invalid dataset. Choose 'wildtype', 'threshold', 'degrate', or 'initial'.")

    # Define x_grid for plotting
    x_grid = np.linspace(-100, 140, 401)

    # Compute MoM normal PDFs
    mean12, var12 = compute_moments_mom(n1, N1, n2, N2, k)
    pdf12 = norm.pdf(x_grid, loc=mean12, scale=np.sqrt(var12))
    mean32, var32 = compute_moments_mom(n3, N3, n2, N2, k)
    pdf32 = norm.pdf(x_grid, loc=mean32, scale=np.sqrt(var32))

    # Run stochastic simulation
    delta_t12, delta_t32 = run_stochastic_simulation(
        k, n1, n2, n3, N1, N2, N3, data12, data32
    )

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].hist(data12, bins=16, density=True,
               alpha=0.4, label='Experimental data12')
    ax[0].hist(delta_t12, bins=16, density=True,
               alpha=0.4, label='Simulated data12')
    ax[0].plot(x_grid, pdf12, 'r-', label='MoM Normal pdf12')
    ax[0].set_xlim(min(x_grid)-20, max(x_grid)+20)
    ax[0].set_title(f"Chrom1 - Chrom2 ({dataset})")
    ax[0].legend()

    ax[1].hist(data32, bins=16, density=True,
               alpha=0.4, label='Experimental data32')
    ax[1].hist(delta_t32, bins=16, density=True,
               alpha=0.4, label='Simulated data32')
    ax[1].plot(x_grid, pdf32, 'r-', label='MoM Normal pdf32')
    ax[1].set_xlim(min(x_grid)-20, max(x_grid)+20)
    ax[1].set_title(f"Chrom3 - Chrom2 ({dataset})")
    ax[1].legend()

    plt.tight_layout()
    # plt.savefig(f"plot_{dataset}_join2.png")
    plt.show()


if __name__ == "__main__":
    # Load optimized parameters
    params = load_parameters("optimized_parameters_joinUpdate.txt")

    # Plot for a specific dataset (e.g., 'threshold')
    # Change to 'wildtype', 'degrate', or 'initial' as needed
    plot_results(params, dataset="initial")
