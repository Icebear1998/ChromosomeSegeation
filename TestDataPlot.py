import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Chromosomes_Theory import f_diff
from NumericalOptimiration import run_stochastic_simulation_and_plot


df = pd.read_excel("Data/Chromosome_diff.xlsx")
data12 = df['SCSdiff_Wildtype12'].dropna().values
data32 = df['SCSdiff_Wildtype23'].dropna().values

x_grid = np.linspace(-80, 80, 301)

n2_opt = 10.00
N2_opt = 100
k_opt = 0.0283
r21_opt = 0.5
R21_opt = 0.50
r23_opt = 0.9
R23_opt = 1.25
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
