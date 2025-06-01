import numpy as np
import matplotlib.pyplot as plt
import math
from Chromosomes_Theory import f_tau_analytic
from Chromosomes_Gillespie import ProteinDegradationSimulation


def run_theoretical(t, k, n, N):
    """Calculate theoretical f_tau values"""
    return [f_tau_analytic(n, ti, k) for ti in t]


def run_stochastic(k, n, N, max_time, num_simulations=2000):
    """Run stochastic simulations and compute histogram"""
    initial_proteins = N
    separation_times = []

    for _ in range(num_simulations):
        simulation = ProteinDegradationSimulation(
            initial_proteins, k, max_time)
        times, states = simulation.run()

        # Find when protein count drops below n
        for i in range(len(states)):
            if states[i] <= n:
                separation_times.append(times[i])
                break

    return separation_times


def compare_methods(k_values, n_values, N, max_time):
    t = [
        np.linspace(30, 110, 200),
        np.linspace(10, 60, 200),
        np.linspace(5, 35, 200),
    ]

    fig, axs = plt.subplots(len(k_values), len(n_values), figsize=(15, 12))

    for i, k in enumerate(k_values):
        for j, n in enumerate(n_values):
            # Calculate theoretical values
            theoretical_values = run_theoretical(t[i], k, n, N)

            # Run stochastic simulations
            separation_times = run_stochastic(k, n, N, max_time)

            # Plot theoretical curve
            axs[i, j].plot(t[i], theoretical_values, 'r-',
                           label='Theoretical', linewidth=2)

            # Plot stochastic histogram
            hist_values, bins, _ = axs[i, j].hist(separation_times, bins=30, density=True,
                                                  alpha=0.5, label='Stochastic')

            # Calculate mean and standard deviation for stochastic results
            mean_time = np.mean(separation_times)
            std_time = np.std(separation_times)

            # Calculate theoretical mean
            theoretical_mean = np.sum(
                t[i] * theoretical_values) * (t[i][1] - t[i][0])

            # Display statistical comparison in the plot
            stats_text = f"Stochastic Mean: {mean_time:.2f}\n" \
                         f"Stochastic Std: {std_time:.2f}\n" \
                         f"Theoretical Mean: {theoretical_mean:.2f}\n" \
                         f"Difference: {abs(mean_time - theoretical_mean):.2f}"
            axs[i, j].text(0.05, 0.95, stats_text, transform=axs[i, j].transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

            axs[i, j].set_title(f'k={k}, n={n}, N={N}')
            axs[i, j].set_xlabel('Time')
            axs[i, j].set_ylabel('Density')
            axs[i, j].legend()
            axs[i, j].grid(True)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Parameters to compare
    k_values = [0.05, 0.1, 0.2]  # Different degradation rates
    n_values = [3, 4, 5]        # Different threshold values
    N = 100                     # Total number of proteins
    max_time = 100              # Maximum simulation time

    # Generate comparison plots
    fig = compare_methods(k_values, n_values, N, max_time)

    # Save the figure
    fig.savefig('method_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
