import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize, basinhopping
import pandas as pd


class ProteinDegradationSimulation:
    def __init__(self, initial_proteins, k, max_time, n0):
        # Initialize the simulation with the initial number of proteins, degradation rate, maximum simulation time, and threshold
        self.initial_proteins = initial_proteins
        self.k = k
        self.max_time = max_time
        self.n0 = n0

    def propensity_function(self, state):
        # Calculate the propensity function for protein degradation
        return self.k * state

    def run(self):
        # Step 1: Initialize the state vector, N. Set the initial time at 0 and the initial number of molecules as appropriate.
        state = self.initial_proteins
        time = 0
        states = [state]
        times = [time]
        seperate_time = None

        while time < self.max_time and state > 0:
            # Step 2: Calculate the reaction propensities, ak(N).
            a = self.propensity_function(state)
            if a == 0:
                break

            # Step 3: Draw a uniform random number, r1.
            r1 = random.random()

            # Step 4: Get the time for the next reaction, τ = (1 / Σak(N)) * ln(1 / r1).
            tau = -np.log(r1) / a

            # Step 5: Draw a uniform random number, r2.
            r2 = random.random()

            # Step 6: Get which reaction is next. Since we have only one reaction, we always update the state.
            # Update state vector, N -> N - 1 (protein degradation).
            state -= 1

            # Step 7: Update time, t -> t + τ.
            time += tau

            # Store the new state and time
            states.append(state)
            times.append(time)

            # Check if the state has reached the threshold n0
            if state <= self.n0 and seperate_time is None:
                seperate_time = time

        return times, states, seperate_time


def calculate_delta_t(initial_proteins1, initial_proteins2, k1, k2, max_time, n01_mean, n02_mean, num_simulations):
    delta_t_list = []
    for _ in range(num_simulations):
        n01 = np.random.normal(n01_mean, 1)
        if n01 <= 0:
            n01 = 0.1
        n01 = np.floor(n01)
        n02 = np.random.normal(n02_mean, 1)
        if n02 <= 0:
            n02 = 0.1
        n02 = np.floor(n02)

        simulation1 = ProteinDegradationSimulation(
            initial_proteins1, k1, max_time, n01)
        _, _, seperate_time1 = simulation1.run()
        simulation2 = ProteinDegradationSimulation(
            initial_proteins2, k2, max_time, n02)
        _, _, seperate_time2 = simulation2.run()
        if seperate_time1 is not None and seperate_time2 is not None:
            delta_t_list.append(seperate_time2 - seperate_time1)
    return delta_t_list


def cost_function(params, initial_proteins1, initial_proteins2, initial_proteins3, max_time, num_simulations, experimental_data12, experimental_data23):
    k1, k2, k3, n01_mean, n02_mean, n03_mean = params

    # Simulate data for chromosome 1 vs chromosome 2
    model_delta_t12 = calculate_delta_t(
        initial_proteins1, initial_proteins2, k1, k2, max_time, n01_mean, n02_mean, num_simulations)
    if len(model_delta_t12) == 0 or len(experimental_data12) == 0:
        return np.inf  # Return a high cost if the data is invalid
    model_mean12 = np.mean(model_delta_t12)
    model_std12 = np.std(model_delta_t12)
    exp_mean12 = np.mean(experimental_data12)
    exp_std12 = np.std(experimental_data12)

    # Simulate data for chromosome 3 vs chromosome 2
    model_delta_t23 = calculate_delta_t(
        initial_proteins2, initial_proteins3, k2, k3, max_time, n02_mean, n03_mean, num_simulations)
    if len(model_delta_t23) == 0 or len(experimental_data23) == 0:
        return np.inf  # Return a high cost if the data is invalid
    model_mean23 = np.mean(model_delta_t23)
    model_std23 = np.std(model_delta_t23)
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
    max_time = 150  # Maximum simulation time
    num_simulations = 100  # Number of simulations to run

    # Read experimental data from Excel file
    df = pd.read_excel('Chromosome_diff.xlsx')
    experimental_data_wt12 = df['SCSdiff_Wildtype12'].dropna().tolist()
    experimental_data_mut12 = df['SCSdiff_Mutant12'].dropna().tolist()
    experimental_data_wt23 = df['SCSdiff_Wildtype23'].dropna().tolist()
    experimental_data_mut23 = df['SCSdiff_Mutant23'].dropna().tolist()

    # Initial guess for the parameters
    # Initial guess for wild-type degradation rates and thresholds
    initial_guess_wt = [0.3, 0.3, 0.3, 4, 3, 5]
    # Initial guess for mutant degradation rates and thresholds
    initial_guess_mut = [0.1, 0.1, 0.1, 4, 3, 5]

    # Define bounds for the parameters
    bounds_wt = [(0.1, 0.4), (0.2, 0.4), (0.2, 0.4), (1, 5), (1, 5), (1, 5)]
    bounds_mut = [(0.05, 0.2), (0.05, 0.2),
                  (0.05, 0.2), (1, 5), (1, 5), (1, 5)]

    # Optimize the parameters for wild-type using basinhopping
    minimizer_kwargs_wt = {"method": "L-BFGS-B", "bounds": bounds_wt, "args": (
        initial_proteins_chromosome1, initial_proteins_chromosome2, initial_proteins_chromosome3, max_time, num_simulations, experimental_data_wt12, experimental_data_wt23)}

    result_wt = basinhopping(cost_function, initial_guess_wt,
                             minimizer_kwargs=minimizer_kwargs_wt, niter=5)

    k1_wt_opt, k2_wt_opt, k3_wt_opt, n01_wt_opt, n02_wt_opt, n03_wt_opt = result_wt.x

    print(
        f'Optimized parameters for wild-type: k1={k1_wt_opt}, k2={k2_wt_opt}, k3={k3_wt_opt}, n01={n01_wt_opt}, n02={n02_wt_opt}, n03={n03_wt_opt}')

    # # Optimize the parameters for mutant using basinhopping
    # minimizer_kwargs_mut = {"method": "L-BFGS-B", "bounds": bounds_mut, "args": (
    #     initial_proteins_chromosome1, initial_proteins_chromosome2, initial_proteins_chromosome3, max_time, num_simulations, experimental_data_mut12, experimental_data_mut23)}
    # result_mut = basinhopping(cost_function, initial_guess_mut,
    #                           minimizer_kwargs=minimizer_kwargs_mut, niter=100)

    # k1_mut_opt, k2_mut_opt, k3_mut_opt, n01_mut_opt, n02_mut_opt, n03_mut_opt = result_mut.x

    # print(
    #     f'Optimized parameters for mutant: k1={k1_mut_opt}, k2={k2_mut_opt}, k3={k3_mut_opt}, n01={n01_mut_opt}, n02={n02_mut_opt}, n03={n03_mut_opt}')
