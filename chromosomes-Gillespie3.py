import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

class ProteinDegradationSimulation:
    def __init__(self, initial_proteins, k, max_time, n0):
        # Initialize the simulation with the initial number of proteins, degradation rate, and maximum simulation time
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
        seperate_time = 0

        while time < self.max_time and state > 0:
            # 0 2: Calculate the reaction propensities, ak(N).
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

            if state == self.n0:
                seperate_time = time

        return times, states, seperate_time

def plot_comparison(simulations1, simulations2, max_time, label1, label2, ax, n01, n02):
    # Plot the results of the simulations
    # Plot individual runs with grey color and fade
    for times, states in simulations1:
        ax.step(times, states, where='post', color='grey', alpha=0.1)
    for times, states in simulations2:
        ax.step(times, states, where='post', color='grey', alpha=0.1)

    delta_t_list = None  # Initialize the time difference between the two thresholds 
    # Calculate and plot the average for the first set of simulations
    all_times = np.linspace(0, max_time, 1000)  # Create a time grid for interpolation
    all_states1 = np.zeros((len(simulations1), len(all_times)))  # Initialize an array to store interpolated states

    for i, (times, states) in enumerate(simulations1):
        # Interpolate the states for each simulation onto the time grid
        all_states1[i, :] = np.interp(all_times, times, states, left=states[0], right=0)

    avg_states1 = np.mean(all_states1, axis=0)  # Calculate the average number of proteins at each time point
    ax.plot(all_times, avg_states1, color='blue', label=f'Average {label1}')  # Plot the average

    # Calculate and plot the average for the second set of simulations
    all_states2 = np.zeros((len(simulations2), len(all_times)))  # Initialize an array to store interpolated states

    for i, (times, states) in enumerate(simulations2):
        # Interpolate the states for each simulation onto the time grid
        all_states2[i, :] = np.interp(all_times, times, states, left=states[0], right=0)

    avg_states2 = np.mean(all_states2, axis=0)  # Calculate the average number of proteins at each time point
    ax.plot(all_times, avg_states2, color='red', label=f'Average {label2}')  # Plot the average

    
    # Find the time when the average number of proteins first drops below n0 for both sets of simulations
    threshold_index1 = np.argmax(avg_states1 < n01)
    threshold_index2 = np.argmax(avg_states2 < n02)

    delta_t_list = all_states1[:,threshold_index1] - all_states2[:, threshold_index2]

    if threshold_index1 > 0 and threshold_index2 > 0:
        threshold_time1 = all_times[threshold_index1]
        threshold_time2 = all_times[threshold_index2]
        delta_t = threshold_time2 - threshold_time1
        ax.axvline(threshold_time1, color='blue', linestyle='--', label=f'{label1} threshold at t={threshold_time1:.2f}')
        ax.axvline(threshold_time2, color='red', linestyle='--', label=f'{label2} threshold at t={threshold_time2:.2f}')
        ax.text((threshold_time1 + threshold_time2) / 2, n01 if n01 > n02 else n02, f'Δt={delta_t:.2f}', color='black', verticalalignment='bottom')

    ax.set_xlabel('Time')  # Label the x-axis
    ax.set_ylabel('Number of Proteins')  # Label the y-axis
    ax.set_title(f'Comparison: {label1} vs {label2}')  # Add a title
    ax.legend()  # Add a legend

    return delta_t_list

if __name__ == "__main__":
    # Parameters for the simulations
    initial_proteins_chromosome1 = 100  # Initial number of proteins for Chromosome 1
    initial_proteins_chromosome2 = 120  # Initial number of proteins for Chromosome 2
    initial_proteins_chromosome3 = 350  # Initial number of proteins for Chromosome 3
    max_time = 150  # Maximum simulation time
    num_simulations = 100  # Number of simulations to run
    
    # Wild-type parameters
    k1_wt = 0.1  # Degradation rate
    k2_wt = 0.1  # Degradation rate
    k3_wt = 0.1  # Degradation rate

    # Mutant parameters
    k1_mut = 0.05  # Degradation rate
    k2_mut = 0.05  # Degradation rate
    k3_mut = 0.05  # Degradation rate

    n0_total = 10  # Total number of threshold proteins
    n01_wt_mean = 2.14  # Threshold number of proteins for Chromosome 1 (Wild-type)
    n02_wt_mean = 3  # Threshold number of proteins for Chromosome 2 (Wild-type)
    n03_wt_mean = n0_total - n01_wt_mean - n02_wt_mean
    n01_mut_mean = 3  # Threshold number of proteins for Chromosome 1 (Mutant)
    n02_mut_mean = 5  # Threshold number of proteins for Chromosome 2 (Mutant)
    n03_mut_mean = n0_total - n01_mut_mean - n02_mut_mean

    # Generate threshold values with Gaussian distribution
    np.random.seed(42)  # For reproducibility
    n01_wt_list = np.random.normal(loc=n01_wt_mean, scale=1, size=num_simulations)
    n02_wt_list = np.random.normal(loc=n02_wt_mean, scale=1, size=num_simulations)
    n03_wt_list = n0_total - n01_wt_list - n02_wt_list

    n01_mut_list = np.random.normal(loc=n01_mut_mean, scale=1, size=num_simulations)
    n02_mut_list = np.random.normal(loc=n02_mut_mean, scale=1, size=num_simulations)
    n03_mut_list = n0_total - n01_mut_list - n02_mut_list

    # Ensure that the threshold values are within a reasonable range
    n01_wt_list = np.floor(np.clip(n01_wt_list, 0.01, n0_total))
    n02_wt_list = np.floor(np.clip(n02_wt_list, 0.01, n0_total))
    n03_wt_list = np.floor(np.clip(n03_wt_list, 0.01, n0_total))

    n01_mut_list = np.floor(np.clip(n01_mut_list, 0.01, n0_total))
    n02_mut_list = np.floor(np.clip(n02_mut_list, 0.01, n0_total))
    n03_mut_list = np.floor(np.clip(n03_mut_list, 0.01, n0_total))

    # Run simulations for Chromosome 1 (Wild-type)
    simulations_chromosome1_wt = []
    seperate_times_chromosome1_wt = []
    for i in range(num_simulations):
        simulation = ProteinDegradationSimulation(initial_proteins_chromosome1, k1_wt, max_time, n01_wt_list[i])
        times, states, seperate_time = simulation.run()
        simulations_chromosome1_wt.append((times, states))
        seperate_times_chromosome1_wt.append(seperate_time)

    # Run simulations for Chromosome 2 (Wild-type)
    simulations_chromosome2_wt = []
    seperate_times_chromosome2_wt = []
    for i in range(num_simulations):
        simulation = ProteinDegradationSimulation(initial_proteins_chromosome2, k2_wt, max_time, n02_wt_list[i])
        times, states, seperate_time = simulation.run()
        simulations_chromosome2_wt.append((times, states))
        seperate_times_chromosome2_wt.append(seperate_time)

    # Run simulations for Chromosome 3 (Wild-type)
    simulations_chromosome3_wt = []
    seperate_times_chromosome3_wt = []
    for i in range(num_simulations):
        simulation = ProteinDegradationSimulation(initial_proteins_chromosome3, k3_wt, max_time, n03_wt_list[i])
        times, states, seperate_time = simulation.run()
        simulations_chromosome3_wt.append((times, states))
        seperate_times_chromosome3_wt.append(seperate_time)
    
    # Run simulations for Chromosome 1 (Mutant)
    simulations_chromosome1_mut = []
    seperate_times_chromosome1_mut = []
    for i in range(num_simulations):
        simulation = ProteinDegradationSimulation(initial_proteins_chromosome1, k1_mut, max_time, n01_mut_list[i])
        times, states, seperate_time = simulation.run()
        simulations_chromosome1_mut.append((times, states))
        seperate_times_chromosome1_mut.append(seperate_time)

    # Run simulations for Chromosome 2 (Mutant)
    simulations_chromosome2_mut = []
    seperate_times_chromosome2_mut = []
    for i in range(num_simulations):
        simulation = ProteinDegradationSimulation(initial_proteins_chromosome2, k2_mut, max_time, n02_mut_list[i])
        times, states, seperate_time = simulation.run()
        simulations_chromosome2_mut.append((times, states))
        seperate_times_chromosome2_mut.append(seperate_time)

    # Run simulations for Chromosome 3 (Mutant)
    simulations_chromosome3_mut = []
    seperate_times_chromosome3_mut = []
    for i in range(num_simulations):
        simulation = ProteinDegradationSimulation(initial_proteins_chromosome3, k3_mut, max_time, n03_mut_list[i])
        times, states, seperate_time = simulation.run()
        simulations_chromosome3_mut.append((times, states))
        seperate_times_chromosome3_mut.append(seperate_time)


    delta_t1_list_wt = np.array(seperate_times_chromosome1_wt) - np.array(seperate_times_chromosome2_wt)
    delta_t2_list_wt = np.array(seperate_times_chromosome3_wt) - np.array(seperate_times_chromosome2_wt)
    delta_t1_list_mut = np.array(seperate_times_chromosome1_mut) - np.array(seperate_times_chromosome2_mut)
    delta_t2_list_mut = np.array(seperate_times_chromosome3_mut) - np.array(seperate_times_chromosome2_mut)

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Plot comparison between Chromosome 1 and Chromosome 2
    plot_comparison(simulations_chromosome1_mut, simulations_chromosome2_mut, max_time, 'Chromosome 1', 'Chromosome 2', axs[0], n01_mut_mean, n02_mut_mean)

    # Plot comparison between Chromosome 2 and Chromosome 3
    plot_comparison(simulations_chromosome3_mut, simulations_chromosome2_mut, max_time, 'Chromosome 2', 'Chromosome 3', axs[1], n03_mut_mean, n02_mut_mean)

    plt.tight_layout()
    plt.show()  # Show the plot

    # Plot histograms of the differences and fit them with Gaussian curves
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    
    # Histogram for delta_t1
    mu1, std1 = norm.fit(delta_t1_list_wt)
    axs[0].hist(delta_t1_list_wt, bins=20, density=True, alpha=0.2, color='blue')
    xmin1, xmax1 = axs[0].get_xlim()
    x1 = np.linspace(xmin1, xmax1, 100)
    p1 = norm.pdf(x1, mu1, std1)
    axs[0].plot(x1, p1, 'blue', linewidth=2, label='Wild-type model')
    axs[0].plot([], [], ' ', label=f'k1(wt)={k1_wt:.2f}, k2(wt)={k2_wt:.2f}')
    axs[0].text(mu1, max(p1), f'μ={mu1:.2f}, σ={std1:.2f}', color='black', verticalalignment='bottom')

    mu1, std1 = norm.fit(delta_t1_list_mut)
    axs[0].hist(delta_t1_list_mut, bins=20, density=True, alpha=0.2, color='orange')
    xmin1, xmax1 = axs[0].get_xlim()
    x1 = np.linspace(xmin1, xmax1, 100)
    p1 = norm.pdf(x1, mu1, std1)
    axs[0].plot(x1, p1, 'orange', linewidth=2, label='Mutant model')
    axs[0].plot([], [], ' ', label=f'k1(mut)={k1_mut:.2f}, k2(mut)={k2_mut:.2f}')
    axs[0].text(mu1, max(p1), f'μ={mu1:.2f}, σ={std1:.2f}', color='black', verticalalignment='bottom')

    # Read experimental data from Excel file
    df = pd.read_excel('Chromosome_diff.xlsx')
    experimental_data_wt = df['SCSdiff_Wildtype12'].dropna().tolist()
    experimental_data_mut = df['SCSdiff_Mutant12'].dropna().tolist()   
    axs[0].hist(experimental_data_wt, bins=20, alpha=0.15, color='k', density=True)
    axs[0].hist(experimental_data_mut, bins=20, alpha=0.15, color='darkgray', density=True)
    mu1, std1 = norm.fit(experimental_data_wt)
    x1 = np.linspace(xmin1, xmax1, 100)
    p1 = norm.pdf(x1, mu1, std1)
    axs[0].plot(x1, p1, 'k', linewidth=2, label='Wild-type data')
    mu1, std1 = norm.fit(experimental_data_mut)
    x1 = np.linspace(xmin1, xmax1, 100)
    p1 = norm.pdf(x1, mu1, std1)
    axs[0].plot(x1, p1, 'darkgray', linewidth=2, label='Mutant data')



    axs[0].legend()
    axs[0].set_title(f'Histogram of Δt (Chromosome 1 vs Chromosome 2)\nN1={initial_proteins_chromosome1}, N2={initial_proteins_chromosome2}')

    # Histogram for delta_t2
    mu2, std2 = norm.fit(delta_t2_list_wt)
    axs[1].hist(delta_t2_list_wt, bins=20, density=True, alpha=0.2, color='blue')
    xmin2, xmax2 = axs[1].get_xlim()
    x2 = np.linspace(xmin2, xmax2, 100)
    p2 = norm.pdf(x2, mu2, std2)
    axs[1].plot(x2, p2, 'blue', linewidth=2, label='Wild-type model')
    axs[1].plot([], [], ' ', label=f'k2(wt)={k2_wt:.2f}, k3(wt)={k3_wt:.2f}')
    axs[1].text(mu2, max(p2), f'μ={mu2:.2f}, σ={std2:.2f}', color='black', verticalalignment='bottom')

    mu2, std2 = norm.fit(delta_t2_list_mut)
    axs[1].hist(delta_t2_list_mut, bins=20, density=True, alpha=0.2, color='orange')
    xmin2, xmax2 = axs[1].get_xlim()
    x2 = np.linspace(xmin2, xmax2, 100)
    p2 = norm.pdf(x2, mu2, std2)
    axs[1].plot(x2, p2, 'orange', linewidth=2, label='Mutant model')
    axs[1].plot([], [], ' ', label=f'k2(mut)={k2_mut:.2f}, k3(mut)={k3_mut:.2f}')
    axs[1].text(mu2, max(p2), f'μ={mu2:.2f}, σ={std2:.2f}', color='black', verticalalignment='bottom')

    # Read experimental data from Excel file
    df = pd.read_excel('Chromosome_diff.xlsx')
    experimental_data_wt = df['SCSdiff_Wildtype23'].dropna().tolist()
    experimental_data_mut = df['SCSdiff_Mutant23'].dropna().tolist()
    axs[1].hist(experimental_data_wt, bins=20, alpha=0.15, color='k', density=True)
    axs[1].hist(experimental_data_mut, bins=20, alpha=0.15, color='darkgray', density=True)
    mu2, std2 = norm.fit(experimental_data_wt)
    x2 = np.linspace(xmin2, xmax2, 100)
    p2 = norm.pdf(x2, mu2, std2)
    axs[1].plot(x2, p2, 'k', linewidth=2, label='Wild-type data')
    mu2, std2 = norm.fit(experimental_data_mut)
    x2 = np.linspace(xmin2, xmax2, 100)
    p2 = norm.pdf(x2, mu2, std2)
    axs[1].plot(x2, p2, 'darkgray', linewidth=2, label='Mutant data')

    axs[1].legend()
    axs[1].set_title(f'Histogram of Δt (Chromosome 2 vs Chromosome 3)\nN2={initial_proteins_chromosome2}, N3={initial_proteins_chromosome3}')

    plt.tight_layout()
    plt.show()  # Show the plot