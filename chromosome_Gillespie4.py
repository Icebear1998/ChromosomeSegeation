import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd


class ProteinDegradationSimulation:
    def __init__(self, initial_state_list, k_list, n0_list, max_time):
        self.k_list = k_list
        self.state = initial_state_list.copy()
        self.n0_list = n0_list
        self.max_time = max_time
        self.time = 0
        self.times = [0]
        self.states = [initial_state_list.copy()]
        self.seperate_times = [None, None, None]

    def simulate(self):
        while self.time < self.max_time:
            # Calculate propensities
            propensities = [self.k_list[i] * self.state[i] for i in range(3)]
            total_propensity = sum(propensities)

            if total_propensity <= 0:
                break

            # Determine time to next reaction
            tau = np.random.exponential(1 / total_propensity)
            self.time += tau

            # Determine which reaction occurs
            r = np.random.uniform(0, total_propensity)
            cumulative_propensity = 0
            for i in range(3):
                cumulative_propensity += propensities[i]
                if r < cumulative_propensity:
                    self.state[i] -= 1
                    break

            # Record the time and states
            self.times.append(self.time)
            self.states.append(self.state.copy())

            # Calculate seperate times
            for i in range(3):
                if self.seperate_times[i] is None and self.state[i] <= self.n0_list[i]:
                    self.seperate_times[i] = self.time

        # Set seperate times to max_time if not reached
        for i in range(3):
            if self.seperate_times[i] is None:
                self.seperate_times[i] = self.max_time

        return self.times, self.states, self.seperate_times


def generate_threshold_values(n0_mean, n0_total, num_simulations):
    # n01_list = np.random.normal(loc=n0_mean[0], scale=1, size=num_simulations)
    # n02_list = np.random.normal(loc=n0_mean[1], scale=1, size=num_simulations)
    # n03_list = n0_total - n01_list - n02_list


    # n01_list = np.floor(np.clip(n01_list, 0.01, n0_total))
    # n02_list = np.floor(np.clip(n02_list, 0.01, n0_total))
    # n03_list = np.floor(np.clip(n03_list, 0.01, n0_total))

    n01_list = n0_mean[0] * np.ones(num_simulations)
    n02_list = n0_mean[1] * np.ones(num_simulations)
    n03_list = n0_total - n01_list - n02_list
    n0_list = np.column_stack((n01_list, n02_list, n03_list))
    return n0_list


def run_simulations(initial_proteins, k_list, n0_list, max_time, num_simulations):
    simulations = []
    seperate_times = []

    for i in range(num_simulations):
        simulation = ProteinDegradationSimulation(
            initial_proteins, k_list, n0_list[i], max_time)
        times, states, sep_times = simulation.simulate()
        simulations.append((times, states))
        seperate_times.append(sep_times)

    return simulations, seperate_times


def plot_comparison(simulations1, simulations2, max_time, label1, label2, ax, n01, n02):
    # Plot the results of the simulations
    # Plot individual runs with grey color and fade
    for times, states in simulations1:
        ax.step(times, states, where='post', color='grey', alpha=0.1)
    for times, states in simulations2:
        ax.step(times, states, where='post', color='grey', alpha=0.1)

    delta_t_list = None  # Initialize the time difference between the two thresholds
    # Calculate and plot the average for the first set of simulations
    # Create a time grid for interpolation
    all_times = np.linspace(0, max_time, 1000)
    # Initialize an array to store interpolated states
    all_states1 = np.zeros((len(simulations1), len(all_times)))

    for i, (times, states) in enumerate(simulations1):
        # Interpolate the states for each simulation onto the time grid
        all_states1[i, :] = np.interp(
            all_times, times, states, left=states[0], right=0)

    # Calculate the average number of proteins at each time point
    avg_states1 = np.mean(all_states1, axis=0)
    ax.plot(all_times, avg_states1, color='blue',
            label=f'Average {label1}')  # Plot the average

    # Calculate and plot the average for the second set of simulations
    # Initialize an array to store interpolated states
    all_states2 = np.zeros((len(simulations2), len(all_times)))

    for i, (times, states) in enumerate(simulations2):
        # Interpolate the states for each simulation onto the time grid
        all_states2[i, :] = np.interp(
            all_times, times, states, left=states[0], right=0)

    # Calculate the average number of proteins at each time point
    avg_states2 = np.mean(all_states2, axis=0)
    ax.plot(all_times, avg_states2, color='red',
            label=f'Average {label2}')  # Plot the average

    # Find the threshold time for each simulation
    threshold_times1 = []
    for i in range(len(simulations1)):
        threshold_index = np.argmax(all_states1[i, :] < n01)
        if threshold_index > 0:
            threshold_times1.append(all_times[threshold_index])

    threshold_times2 = []
    for i in range(len(simulations2)):
        threshold_index = np.argmax(all_states2[i, :] < n02)
        if threshold_index > 0:
            threshold_times2.append(all_times[threshold_index])

    if threshold_times1 and threshold_times2:
        avg_threshold_time1 = np.mean(threshold_times1)
        avg_threshold_time2 = np.mean(threshold_times2)

        delta_t = avg_threshold_time1 - avg_threshold_time2
        ax.axvline(avg_threshold_time1, color='blue', linestyle='--',
                   label=f'{label1} threshold at t={avg_threshold_time1:.2f}')
        ax.axvline(avg_threshold_time2, color='red', linestyle='--',
                   label=f'{label2} threshold at t={avg_threshold_time2:.2f}')
        ax.text((avg_threshold_time1 + avg_threshold_time2) / 2, n01 if n01 > n02 else n02,
                f'Δt={delta_t:.2f}', color='black', verticalalignment='bottom')

    ax.set_xlabel('Time')  # Label the x-axis
    ax.set_ylabel('Number of Proteins')  # Label the y-axis
    ax.set_title(f'Comparison: {label1} vs {label2}')  # Add a title
    ax.legend()  # Add a legend


def plot_experimental_vs_simulation(experimental_data, simulation_data, label, ax):
    # Plot the experimental data
    ax.hist(experimental_data, bins=30, density=True, alpha=0.5,
            label=f'Experimental {label}', color='blue')

    # Plot the simulation data
    ax.hist(simulation_data, bins=50, density=True, alpha=0.5,
            label=f'Simulation {label}', color='orange')

    ax.set_xlabel('Difference in Separation Time')
    ax.set_ylabel('Density')
    ax.set_title(f'Comparison of Experimental Data vs Simulation ({label})')
    ax.legend()


if __name__ == "__main__":
    # Parameters for the simulations
    initial_proteins_chromosome1 = 100  # Initial number of proteins for Chromosome 1
    initial_proteins_chromosome2 = 120  # Initial number of proteins for Chromosome 2
    initial_proteins_chromosome3 = 350  # Initial number of proteins for Chromosome 3
    initial_proteins = [initial_proteins_chromosome1,
                        initial_proteins_chromosome2, initial_proteins_chromosome3]
    max_time = 150  # Maximum simulation time
    num_simulations = 1000  # Number of simulations to run

    # Wild-type parameters
    k1_wt = 0.1  # Degradation rate
    k2_wt = 0.1  # Degradation rate
    k3_wt = 0.1  # Degradation rate
    k_wt = [k1_wt, k2_wt, k3_wt]

    # Mutant parameters
    k1_mut = 0.05  # Degradation rate
    k2_mut = 0.05  # Degradation rate
    k3_mut = 0.05  # Degradation rate
    k_mut = [k1_mut, k2_mut, k3_mut]

    n0_total = 10  # Total number of threshold proteins
    # Threshold number of proteins for Chromosome 1 (Wild-type)
    n01_wt_mean = 2.14
    # Threshold number of proteins for Chromosome 2 (Wild-type)
    n02_wt_mean = 3
    n03_wt_mean = n0_total - n01_wt_mean - n02_wt_mean
    n01_mut_mean = 3  # Threshold number of proteins for Chromosome 1 (Mutant)
    n02_mut_mean = 5  # Threshold number of proteins for Chromosome 2 (Mutant)
    n03_mut_mean = n0_total - n01_mut_mean - n02_mut_mean

    # Generate threshold values with Gaussian distribution
    n0_wt_list = generate_threshold_values(
        [n01_wt_mean, n02_wt_mean], n0_total, num_simulations)
    n0_mut_list = generate_threshold_values(
        [n01_mut_mean, n02_mut_mean], n0_total, num_simulations)

    # Run simulations for Wild-type
    simulations_wt, seperate_times_wt = run_simulations(
        initial_proteins, k_wt, n0_wt_list, max_time, num_simulations)

    # Run simulations for Mutant
    simulations_mut, seperate_times_mut = run_simulations(
        initial_proteins, k_mut, n0_mut_list, max_time, num_simulations)

    delta_t_list_wt = np.array(
        [[sep[0] - sep[1], sep[2] - sep[1]] for sep in seperate_times_wt])
    delta_t_list_mut = np.array(
        [[sep[0] - sep[1], sep[2] - sep[1]] for sep in seperate_times_mut])

    # Read experimental data from Excel file
    df = pd.read_excel('Chromosome_diff.xlsx')
    experimental_data_wt12 = df['SCSdiff_Wildtype12'].dropna().tolist()
    experimental_data_mut12 = df['SCSdiff_Mutant12'].dropna().tolist()
    experimental_data_wt23 = df['SCSdiff_Wildtype23'].dropna().tolist()
    experimental_data_mut23 = df['SCSdiff_Mutant23'].dropna().tolist()

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    # Plot comparison between experimental data and simulation for Wild-type
    plot_experimental_vs_simulation(
        experimental_data_wt12, delta_t_list_wt[:, 0], 'Wild-type (Chromosome 1 vs Chromosome 2)', axs[0])
    plot_experimental_vs_simulation(
        experimental_data_wt23, delta_t_list_wt[:, 1], 'Wild-type (Chromosome 3 vs Chromosome 2)', axs[1])

    plt.tight_layout()
    plt.show()  # Show the plot

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    # Plot comparison between experimental data and simulation for Mutant
    plot_experimental_vs_simulation(
        experimental_data_mut12, delta_t_list_mut[:, 0], 'Mutant (Chromosome 1 vs Chromosome 2)', axs[0])
    plot_experimental_vs_simulation(
        experimental_data_mut23, delta_t_list_mut[:, 1], 'Mutant (Chromosome 3 vs Chromosome 2)', axs[1])

    plt.tight_layout()
    plt.show()  # Show the plot

    # # Extract times and the first state component from all simulations
    # simulations_chromosome1_mut = [
    #     [times, [state[0] for state in states]] for times, states in simulations_mut]
    # simulations_chromosome2_mut = [
    #     [times, [state[1] for state in states]] for times, states in simulations_mut]
    # simulations_chromosome3_mut = [
    #     [times, [state[2] for state in states]] for times, states in simulations_mut]

    # # Extract times and the first state component from all simulations
    # simulations_chromosome1_wt = [
    #     [times, [state[0] for state in states]] for times, states in simulations_wt]
    # simulations_chromosome2_wt = [
    #     [times, [state[1] for state in states]] for times, states in simulations_wt]
    # simulations_chromosome3_wt = [
    #     [times, [state[2] for state in states]] for times, states in simulations_wt]

    # fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    # # Plot comparison between Chromosome 1 and Chromosome 2
    # plot_comparison(simulations_chromosome1_mut, simulations_chromosome2_mut, max_time,
    #                 'Chromosome 1', 'Chromosome 2', axs[0], n01_mut_mean, n02_mut_mean)

    # # Plot comparison between Chromosome 3 and Chromosome 2
    # plot_comparison(simulations_chromosome3_mut, simulations_chromosome2_mut, max_time,
    #                 'Chromosome 3', 'Chromosome 2', axs[1], n03_mut_mean, n02_mut_mean)

    # plt.tight_layout()
    # plt.show()  # Show the plot

    # fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    # # Plot comparison between Chromosome 1 and Chromosome 2
    # plot_comparison(simulations_chromosome1_wt, simulations_chromosome2_wt, max_time,
    #                 'Chromosome 1', 'Chromosome 2', axs[0], n01_wt_mean, n02_wt_mean)

    # # Plot comparison between Chromosome 3 and Chromosome 2
    # plot_comparison(simulations_chromosome3_wt, simulations_chromosome2_wt, max_time,
    #                 'Chromosome 3', 'Chromosome 2', axs[1], n03_wt_mean, n02_wt_mean)

    # plt.tight_layout()
    # plt.show()  # Show the plot
