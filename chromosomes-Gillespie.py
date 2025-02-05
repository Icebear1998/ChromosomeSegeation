import numpy as np
import random
import matplotlib.pyplot as plt

class ChromosomeSimulation:
    def __init__(self, initial_state, k, max_time):
        self.state = initial_state
        self.k = k
        self.max_time = max_time
        self.time = 0
        self.states = []
        self.times = []

    def propensity_function(self):
        # Define the propensity function based on the current state
        return self.k * self.state

    def update_state(self):
        # Define how the state changes
        self.state += 1

    def run(self):
        while self.time < self.max_time:
            a = self.propensity_function()
            if a == 0:
                break
            tau = -np.log(random.random()) / a
            self.time += tau
            self.update_state()
            self.states.append(self.state)
            self.times.append(self.time)

class ProteinDegradationSimulation:
    def __init__(self, initial_proteins, k, max_time):
        # Initialize the simulation with the initial number of proteins, degradation rate, and maximum simulation time
        self.initial_proteins = initial_proteins
        self.k = k
        self.max_time = max_time

    def propensity_function(self, state):
        # Calculate the propensity function for protein degradation
        # The propensity is the rate at which proteins degrade, which is proportional to the current number of proteins
        return self.k * state

    def run(self):
        # Run the Gillespie simulation
        state = self.initial_proteins  # Start with the initial number of proteins
        time = 0  # Start at time 0
        states = [state]  # List to store the number of proteins at each step
        times = [time]  # List to store the time at each step

        while time < self.max_time and state > 0:
            # Continue the simulation until the maximum time is reached or there are no more proteins
            a = self.propensity_function(state)  # Calculate the propensity function
            if a == 0:
                break  # If the propensity is 0, stop the simulation
            tau = -np.log(random.random()) / a  # Sample the time to the next reaction
            time += tau  # Update the current time
            state -= 1  # Degrade one protein
            states.append(state)  # Store the new state
            times.append(time)  # Store the new time

        return times, states  # Return the times and states

def plot_simulation(simulations, max_time):
    # Plot the results of the simulations
    plt.figure(figsize=(10, 6))

    # Plot individual runs with grey color and fade
    for times, states in simulations:
        plt.step(times, states, where='post', color='grey', alpha=0.1)

    # Calculate and plot the average
    all_times = np.linspace(0, max_time, 1000)  # Create a time grid for interpolation
    all_states = np.zeros((len(simulations), len(all_times)))  # Initialize an array to store interpolated states

    for i, (times, states) in enumerate(simulations):
        # Interpolate the states for each simulation onto the time grid
        all_states[i, :] = np.interp(all_times, times, states, left=states[0], right=0)

    avg_states = np.mean(all_states, axis=0)  # Calculate the average number of proteins at each time point
    plt.plot(all_times, avg_states, color='blue', label='Average')  # Plot the average

    plt.xlabel('Time')  # Label the x-axis
    plt.ylabel('Number of Proteins')  # Label the y-axis
    plt.title('Protein Degradation Simulation')  # Add a title
    plt.legend()  # Add a legend
    plt.show()  # Show the plot

if __name__ == "__main__":
    # Example usage
    initial_proteins = 100  # Initial number of proteins
    k = 0.1  # Degradation rate
    max_time = 200  # Maximum simulation time
    num_simulations = 100  # Number of simulations to run

    simulations = []
    for _ in range(num_simulations):
        # Run the simulation multiple times and store the results
        simulation = ProteinDegradationSimulation(initial_proteins, k, max_time)
        times, states = simulation.run()
        simulations.append((times, states))

    plot_simulation(simulations, max_time)  # Plot the results