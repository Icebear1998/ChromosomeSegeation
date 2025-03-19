import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Chromosomes_Theory import f_diff
from Chromosome_Gillespie4 import run_simulations, generate_threshold_values


df = pd.read_excel("Data/Chromosome_diff.xlsx")
data12 = df['Wildtype12'].dropna().values
data32 = df['Wildtype32'].dropna().values

x_grid = np.linspace(-80, 80, 301)

n2_opt = 5.00
N2_opt = 100
k_opt = 0.0365
r21_opt = 1
R21_opt = 1
r23_opt = 1
R23_opt = 1.5
n1_opt = r21_opt * n2_opt
N1_opt = R21_opt * N2_opt
n3_opt = r23_opt * n1_opt
N3_opt = R23_opt * N1_opt

pdf12 = np.array([
    f_diff(x, k_opt, n1_opt, N1_opt, n2_opt, N2_opt)
    for x in x_grid
])
area12 = np.trapz(pdf12, x_grid)
if area12 > 1e-15:
    pdf12 /= area12

pdf32 = np.array([
    f_diff(x, k_opt, n3_opt, N3_opt, n2_opt, N2_opt)
    for x in x_grid
])
area32 = np.trapz(pdf32, x_grid)
if area32 > 1e-15:
    pdf32 /= area32

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].hist(data12, bins=12, density=True, alpha=0.4, label='data12')
ax[0].plot(x_grid, pdf12, 'r-', label='model pdf12')
ax[0].set_title("Chrom1 - Chrom2")
ax[0].legend()

ax[1].hist(data32, bins=12, density=True, alpha=0.4, label='data32')
ax[1].plot(x_grid, pdf32, 'r-', label='model pdf32')
ax[1].set_title("Chrom3 - Chrom2")
ax[1].legend()

plt.tight_layout()
plt.show()
fig.savefig('Results/theoryfittest.png', dpi=300, bbox_inches='tight')

def run_stochastic_simulation_and_plot(k_opt, r21_opt, R21_opt, r23_opt, R23_opt,
                                       data12, data32,
                                       n2_opt, N2_opt,
                                       max_time=150, num_sim=500):
    """
    Use the best-fit (k_opt, r1_opt, R1_opt, r2_opt, R2_opt) to run a Gillespie-like 
    simulation. Then compare sim difference times to the experimental data hist.
    """
    n1_opt = r21_opt*n2_opt
    N1_opt = R21_opt*N2_opt
    n3_opt = r23_opt*n2_opt
    N3_opt = R23_opt*N2_opt

    initial_proteins1 = N1_opt
    initial_proteins2 = N2_opt
    initial_proteins3 = N3_opt  # or any known / used
    initial_proteins = [initial_proteins1,
                        initial_proteins2, initial_proteins3]

    # If your code requires rates for each chromosome:
    rates = [k_opt, k_opt, k_opt]

    n0_total = n1_opt + n2_opt + n3_opt
    n01_mean = n1_opt
    n02_mean = n2_opt
    # if you only track 2 thresholds, pass [n01_mean, n02_mean]
    n0_list = generate_threshold_values(
        [n01_mean, n02_mean], n0_total, num_sim)

    simulations, separate_times = run_simulations(
        initial_proteins, rates, n0_list, max_time, num_sim
    )

    # separate_times[i] => [sep1, sep2, sep3]
    # so Chrom1 - Chrom2 => sep[0] - sep[1], Chrom3 - Chrom2 => sep[2] - sep[1]
    delta_t12 = [sep[0] - sep[1] for sep in separate_times]
    delta_t32 = [sep[2] - sep[1] for sep in separate_times]

    # Plot sim vs data
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].hist(delta_t12, bins=12, density=True, alpha=0.4, label='Sim data12')
    ax[0].hist(data12, bins=12, density=True, alpha=0.4, label='Exp data12')
    ax[0].set_title("Stochastic Sim vs Exp (Chrom1–Chrom2)")
    ax[0].legend()

    ax[1].hist(delta_t32, bins=12, density=True, alpha=0.4, label='Sim data32')
    ax[1].hist(data32, bins=12, density=True, alpha=0.4, label='Exp data32')
    ax[1].set_title("Stochastic Sim vs Exp (Chrom3–Chrom2)")
    ax[1].legend()

    plt.tight_layout()
    plt.show()
    fig.savefig('Results/simfit.png', dpi=300, bbox_inches='tight')

run_stochastic_simulation_and_plot(
    k_opt, r21_opt, R21_opt,
    r23_opt, R23_opt,
    data12, data32,
    n2_opt, N2_opt
)

# Best negative log-likelihood: 1134.341497412577
# Best parameters:
#   n2=10.00, N2=98.96
#   k=0.0278, r1=0.5251, R1=0.50
#   r2=1.1022, R2=1.50

# Best negative log-likelihood: 1135.278478815829
#   n2=10.00, N2=100.15
#   k=0.0283, r1=0.5251, R1=0.50
#   r2=0.9060, R2=1.25

# Best negative log-likelihood: 1139.349823606604
# Best parameters:
#   n2=5.00, N2=100.00
#   k=0.0365, r1=1.0000, R1=1.00
#   r2=1.0000, R2=1.50

# Best negative log-likelihood: 1139.3498236066043
# Best parameters:
#   n2=5.00, N2=100.00
#   k=0.0365, r1=1.0000, R1=1.00
#   r2=1.0000, R2=1.50