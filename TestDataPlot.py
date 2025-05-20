import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MultiMechanismSimulation import MultiMechanismSimulation
from MoMCalculations import compute_pdf_mom

def generate_threshold_values(n0_mean, n0_total, num_simulations):
    n01_list = n0_mean[0] * np.ones(num_simulations)
    n02_list = n0_mean[1] * np.ones(num_simulations)
    n03_list = n0_total - n01_list - n02_list
    return np.column_stack((n01_list, n02_list, n03_list))

def load_parameters(filename="optimized_parameters.txt"):
    params = {}
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            key, value = line.strip().split(": ")
            params[key] = float(value)
    return params

def load_dataset(df, dataset):
    if dataset == "wildtype":
        return df['wildtype12'].dropna().values, df['wildtype32'].dropna().values
    elif dataset == "threshold":
        return df['threshold12'].dropna().values, df['threshold32'].dropna().values
    elif dataset == "degrate":
        return df['degRate12'].dropna().values, df['degRate32'].dropna().values
    elif dataset == "initial":
        return df['initialProteins12'].dropna().values, df['initialProteins32'].dropna().values
    else:
        raise ValueError("Invalid dataset. Choose 'wildtype', 'threshold', 'degrate', or 'initial'.")

def apply_mutant_params(params, dataset):
    n1, n2, n3 = params['n1'], params['n2'], params['n3']
    N1, N2, N3 = params['N1'], params['N2'], params['N3']
    k = params['k']
    
    if dataset == "threshold":
        alpha = params['alpha']
        n1 = max(params['n1'] - alpha, 1)
        n2 = max(params['n2'] - alpha, 1)
        n3 = max(params['n3'] - alpha, 1)
    elif dataset == "degrate":
        beta_k = params['beta_k']
        k = max(beta_k * params['k'], 0.001)
    elif dataset == "initial":
        gamma = params['gamma']
        N1 = max(params['N1'] - gamma, 1)
        N2 = max(params['N2'] - gamma, 1)
        N3 = max(params['N3'] - gamma, 1)
    
    return n1, n2, n3, N1, N2, N3, k

def run_stochastic_simulation(mechanism, k, n1, n2, n3, N1, N2, N3, burst_size, max_time=500, num_sim=1500):
    initial_proteins = [N1, N2, N3]
    rate_params = {'k_list': [k, k, k]} if mechanism == 'simple' else {'lambda_list': [k, k, k], 'burst_size': burst_size}
    n0_total = n1 + n2 + n3
    n0_list = generate_threshold_values([n1, n2], n0_total, num_sim)

    simulations = []
    separate_times = []
    for i in range(num_sim):
        simulation = MultiMechanismSimulation(
            mechanism=mechanism,
            initial_state_list=initial_proteins,
            rate_params=rate_params,
            n0_list=n0_list[i],
            max_time=max_time
        )
        times, states, sep_times = simulation.simulate()
        simulations.append((times, states))
        separate_times.append(sep_times)

    delta_t12 = [sep[0] - sep[1] for sep in separate_times]
    delta_t32 = [sep[2] - sep[1] for sep in separate_times]
    return delta_t12, delta_t32

def plot_results(params, dataset="wildtype", mechanism="simple", data_file="Data/All_strains_SCStimes.xlsx"):
    df = pd.read_excel(data_file)
    data12, data32 = load_dataset(df, dataset)
    n1, n2, n3, N1, N2, N3, k = apply_mutant_params(params, dataset)
    burst_size = params.get('burst_size', 5)

    x_grid = np.linspace(-100, 100, 401)
    pdf12 = compute_pdf_mom(mechanism, x_grid, n1, N1, n2, N2, k, burst_size)
    pdf32 = compute_pdf_mom(mechanism, x_grid, n3, N3, n2, N2, k, burst_size)

    delta_t12, delta_t32 = run_stochastic_simulation(
        mechanism, k, n1, n2, n3, N1, N2, N3, burst_size
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(data12, bins=15, density=True, alpha=0.4, label='Experimental data12')
    ax1.hist(delta_t12, bins=15, density=True, alpha=0.4, label='Simulated data12')
    ax1.plot(x_grid, pdf12, 'r-', label='MoM pdf12')
    ax1.set_xlim(min(x_grid)-20, max(x_grid)+20)
    ax1.set_title(f"Chrom1 - Chrom2 ({dataset}, {mechanism})")
    ax1.legend()

    ax2.hist(data32, bins=14, density=True, alpha=0.4, label='Experimental data32')
    ax2.hist(delta_t32, bins=14, density=True, alpha=0.4, label='Simulated data32')
    ax2.plot(x_grid, pdf32, 'r-', label='MoM pdf32')
    ax2.set_xlim(min(x_grid)-20, max(x_grid)+20)
    ax2.set_title(f"Chrom3 - Chrom2 ({dataset}, {mechanism})")
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    params = load_parameters("Results/optimized_parameters_joinBurst.txt")
    mechanisms = ['simple', 'fixed_burst']
    datasets = ['wildtype', 'threshold', 'degrate', 'initial']
    
    plot_results(params, dataset=datasets[0], mechanism=mechanisms[1])