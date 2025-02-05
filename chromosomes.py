import numpy as np
import matplotlib.pyplot as plt
import math

N = 100

def function(n, t, k):
    return k * math.factorial(N) / (np.math.factorial(n) * np.math.factorial(N-n-1))\
              * (np.exp(-(n+1)*k*t))*(1-np.exp(-k*t))**(N-n-1)

def plot_chromosomes():
    t = np.linspace(0, 60, 100)
    k_values = [0.1, 0.2, 0.3, 0.4]  # Choose 4 values of k
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()  # Flatten the 2D array of axes for easy iteration
    for i, k in enumerate(k_values):
        for n in range(3, 10):
            axs[i].plot(t, function(n, t, k), label=f'n={n}')
        axs[i].legend()
        axs[i].set_title(f'Plot for k={k}')
    
    plt.tight_layout()
    plt.show()

plot_chromosomes()