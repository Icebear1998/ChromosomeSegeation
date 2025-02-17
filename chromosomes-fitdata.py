import numpy as np
import pandas as pd
from scipy.optimize import basinhopping, minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
from chromosome_Gillespie4 import ProteinDegradationSimulation


def cost_function(params, initial_proteins, max_time, num_simulations, experimental_data12, experimental_data23):
    k, n01_mean, n02_mean, n03_mean = params

    initial_proteins1, initial_proteins2, initial_proteins3 = initial_proteins

    # Simulate data for all chromosomes
    n0_total = 10
    n03_mean = n0_total - n01_mean - n02_mean

    n01_list = np.random.normal(loc=n01_mean, scale=1, size=num_simulations)
    n02_list = np.random.normal(loc=n02_mean, scale=1, size=num_simulations)
    n03_list = n0_total - n01_list - n02_list

    n01_list = np.floor(np.clip(n01_list, 0.01, n0_total))
    n02_list = np.floor(np.clip(n02_list, 0.01, n0_total))
    n03_list = np.floor(np.clip(n03_list, 0.01, n0_total))
    n0_list = np.column_stack((n01_list, n02_list, n03_list))

    simulations = []
    seperate_times = []

    for i in range(num_simulations):
        simulation = ProteinDegradationSimulation(
            initial_state_list=[initial_proteins1, initial_proteins2, initial_proteins3], k_list=[k, k, k], n0_list=n0_list[i], max_time=max_time)
        times, states, sep_times = simulation.simulate()
        simulations.append((times, states))
        seperate_times.append(sep_times)

    delta_t_list = np.array([[sep[0] - sep[1], sep[2] - sep[1]]
                            for sep in seperate_times])

    if len(delta_t_list) == 0 or len(experimental_data12) == 0 or len(experimental_data23) == 0:
        return np.inf  # Return a high cost if the data is invalid

    # Calculate the mean and std for chromosome 1 vs chromosome 2
    model_mean12 = np.mean(delta_t_list[:, 0])
    model_std12 = np.std(delta_t_list[:, 0])
    exp_mean12 = np.mean(experimental_data12)
    exp_std12 = np.std(experimental_data12)

    # Calculate the mean and std for chromosome 3 vs chromosome 2
    model_mean23 = np.mean(delta_t_list[:, 1])
    model_std23 = np.std(delta_t_list[:, 1])
    exp_mean23 = np.mean(experimental_data23)
    exp_std23 = np.std(experimental_data23)

    # Combine the costs
    cost = (model_mean12 - exp_mean12)**2 + (model_std12 - exp_std12)**2 + \
        (model_mean23 - exp_mean23)**2 + (model_std23 - exp_std23)**2
    return cost


if __name__ == "__main__":
    # Parameters for the simulations
    initial_proteins_chromosome1 = 100  # Initial number of proteins for Chromosome 1
    initial_proteins_chromosome2 = 120  # Initial number of proteins for Chromosome 2
    initial_proteins_chromosome3 = 350  # Initial number of proteins for Chromosome 3
    initial_proteins = [initial_proteins_chromosome1,
                        initial_proteins_chromosome2, initial_proteins_chromosome3]
    max_time = 150  # Maximum simulation time
    num_simulations = 10  # Number of simulations to run
    # Wild-type parameters
    k1_wt = 0.1  # Degradation rate
    k2_wt = 0.1  # Degradation rate
    k3_wt = 0.1  # Degradation rate
    k_wt = k1_wt  # [k1_wt, k2_wt, k3_wt]

    # Mutant parameters
    k1_mut = 0.05  # Degradation rate
    k2_mut = 0.05  # Degradation rate
    k3_mut = 0.05  # Degradation rate
    k_mut = k1_mut  # [k1_mut, k2_mut, k3_mut]

    # Read experimental data from Excel file
    df = pd.read_excel('Chromosome_diff.xlsx')
    experimental_data_wt12 = df['SCSdiff_Wildtype12'].dropna().tolist()
    experimental_data_mut12 = df['SCSdiff_Mutant12'].dropna().tolist()
    experimental_data_wt23 = df['SCSdiff_Wildtype23'].dropna().tolist()
    experimental_data_mut23 = df['SCSdiff_Mutant23'].dropna().tolist()

    # Initial guess for the parameters
    # Initial guess for wild-type degradation rates and thresholds
    initial_guess_wt = [k_wt, 4, 3, 5]
    # Initial guess for mutant degradation rates and thresholds
    initial_guess_mut = [k_mut, 4, 3, 5]

    # Define bounds for the parameters
    bounds_wt = [(0.02, 0.4), (1, 5), (1, 5), (1, 5)]
    bounds_mut = [(0.02, 0.4), (1, 5), (1, 5), (1, 5)]

    # Optimize the parameters for wild-type using basinhopping
    minimizer_kwargs_wt = {"method": "L-BFGS-B", "bounds": bounds_wt, "args": (
        initial_proteins, max_time, num_simulations, experimental_data_wt12, experimental_data_wt23)}

    # result_wt = basinhopping(cost_function, initial_guess_wt,
    #                          minimizer_kwargs=minimizer_kwargs_wt, niter=5)

    # Optimize the parameters for wild-type using minimize with L-BFGS-B method
    result_wt = minimize(cost_function, initial_guess_wt, args=(
        initial_proteins, max_time, num_simulations, experimental_data_wt12, experimental_data_wt23), method='L-BFGS-B', bounds=bounds_wt)

    k_opt, n01_wt_opt, n02_wt_opt, n03_wt_opt = result_wt.x

    print(
        f'Optimized parameters for wild-type: k = {k_opt}, n01={n01_wt_opt}, n02={n02_wt_opt}, n03={n03_wt_opt}')

    # Optimize the parameters for mutant using basinhopping
    # minimizer_kwargs_mut = {"method": "L-BFGS-B", "bounds": bounds_mut, "args": (
    #     initial_proteins_chromosome1, initial_proteins_chromosome2, initial_proteins_chromosome3, max_time, num_simulations, experimental_data_mut12, experimental_data_mut23)}
    # result_mut = basinhopping(cost_function, initial_guess_mut,
    #                           minimizer_kwargs=minimizer_kwargs_mut, niter=100)

    # k1_mut_opt, k2_mut_opt, k3_mut_opt, n01_mut_opt, n02_mut_opt, n03_mut_opt = result_mut.x

    # print(
    #     f'Optimized parameters for mutant: k1={k1_mut_opt}, k2={k2_mut_opt}, k3={k3_mut_opt}, n01={n01_mut_opt}, n02={n02_mut_opt}, n03={n03_mut_opt}')
