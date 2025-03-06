import numpy as np
import random
import matplotlib.pyplot as plt


class ProteinDegradationSimulation:
    def __init__(self, initial_proteins, k, max_time):
        # Initialize the simulation with the initial number of proteins, degradation rate, and maximum simulation time
        self.initial_proteins = initial_proteins
        self.k = k
        self.max_time = max_time

    def propensity_function(self, state):
        # Calculate the propensity function for protein degradation
        return self.k * state

    def run(self):
        # Step 1: Initialize the state vector, N. Set the initial time at 0 and the initial number of molecules as appropriate.
        state = self.initial_proteins
        time = 0
        states = [state]
        times = [time]

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

        return times, states


def plot_simulation(simulations, max_time, label, ax, n0):
    # Plot the results of the simulations
    # Plot individual runs with grey color and fade
    for times, states in simulations:
        ax.step(times, states, where='post', color='grey', alpha=0.1)

    # Calculate and plot the average
    # Create a time grid for interpolation
    all_times = np.linspace(0, max_time, 1000)
    # Initialize an array to store interpolated states
    all_states = np.zeros((len(simulations), len(all_times)))

    for i, (times, states) in enumerate(simulations):
        # Interpolate the states for each simulation onto the time grid
        all_states[i, :] = np.interp(
            all_times, times, states, left=states[0], right=0)

    # Calculate the average number of proteins at each time point
    avg_states = np.mean(all_states, axis=0)
    ax.plot(all_times, avg_states, color='blue',
            label='Average')  # Plot the average

    # # Find the time when the average number of proteins first drops below n0
    # threshold_index = np.argmax(avg_states < n0)
    # if threshold_index > 0:
    #     threshold_time = all_times[threshold_index]
    #     ax.axvline(threshold_time, color='red', linestyle='--',
    #                label=f'Threshold at t={threshold_time:.2f}')
    #     ax.text(threshold_time, n0,
    #             f't={threshold_time:.2f}', color='red', verticalalignment='bottom')

    # Find the threshold time for each simulation
    threshold_times = []
    for i in range(len(simulations)):
        threshold_index = np.argmax(all_states[i, :] < n0)
        if threshold_index > 0:
            threshold_times.append(all_times[threshold_index])

    # Calculate the average threshold time
    if threshold_times:
        avg_threshold_time = np.mean(threshold_times)
        ax.axvline(avg_threshold_time, color='red', linestyle='--',
                   label=f'Threshold at t={avg_threshold_time:.2f}')
        ax.text(avg_threshold_time, n0,
                f't={avg_threshold_time:.2f}', color='red', verticalalignment='bottom')

    ax.set_xlabel('Time')  # Label the x-axis
    ax.set_ylabel('Number of Proteins')  # Label the y-axis
    ax.set_title(f'Protein Degradation Simulation ({label})')  # Add a title
    ax.legend()  # Add a legend


def plot_comparison(simulations1, simulations2, max_time, label1, label2, ax):
    # Plot the results of the simulations
    # Plot individual runs with grey color and fade
    for times, states in simulations1:
        ax.step(times, states, where='post', color='grey', alpha=0.1)
    for times, states in simulations2:
        ax.step(times, states, where='post', color='grey', alpha=0.1)

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

    ax.set_xlabel('Time')  # Label the x-axis
    ax.set_ylabel('Number of Proteins')  # Label the y-axis
    ax.set_title(f'Comparison: {label1} vs {label2}')  # Add a title
    ax.legend()  # Add a legend


if __name__ == "__main__":
    initial_proteins = 100  # Initial number of proteins
    k_values = [0.1, 0.2, 0.3, 0.4]  # Different degradation rates
    max_time = 150  # Maximum simulation time
    num_simulations = 100  # Number of simulations to run
    n0 = 5  # Threshold number of proteins

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    for i, k in enumerate(k_values):
        simulations = []
        for _ in range(num_simulations):
            # Run the simulation multiple times and store the results
            simulation = ProteinDegradationSimulation(
                initial_proteins, k, max_time)
            times, states = simulation.run()
            simulations.append((times, states))

        plot_simulation(simulations, max_time,
                        f'k={k}', axs[i], n0)  # Plot the results

    plt.tight_layout()
    plt.show()  # Show the plot

    # Additional graph for different initial_proteins values
    # Different initial protein counts
    initial_proteins_values = [50, 100, 150, 200]
    k = 0.1  # Fixed degradation rate

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    for i, initial_proteins in enumerate(initial_proteins_values):
        simulations = []
        for _ in range(num_simulations):
            # Run the simulation multiple times and store the results
            simulation = ProteinDegradationSimulation(
                initial_proteins, k, max_time)
            times, states = simulation.run()
            simulations.append((times, states))

        # Plot the results
        plot_simulation(simulations, max_time,
                        f'initial_proteins={initial_proteins}', axs[i], n0)

    plt.tight_layout()
    plt.show()  # Show the plot
