import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from Chromosome_Gillespie4 import run_simulations, generate_threshold_values

# Compute MoM moments for f_X
def compute_moments_mom(n1, N1, n2, N2, k):
    sum1_T1 = sum(1/m for m in range(int(n1) + 1, int(N1) + 1))
    sum1_T2 = sum(1/m for m in range(int(n2) + 1, int(N2) + 1))
    sum2_T1 = sum(1/(m**2) for m in range(int(n1) + 1, int(N1) + 1))
    sum2_T2 = sum(1/(m**2) for m in range(int(n2) + 1, int(N2) + 1))
    
    mean_T1 = sum1_T1 / k
    mean_T2 = sum1_T2 / k
    var_T1 = sum2_T1 / (k**2)
    var_T2 = sum2_T2 / (k**2)
    
    mean_X = mean_T1 - mean_T2
    var_X = var_T1 + var_T2
    return mean_X, var_X

def run_stochastic_simulation_and_plot(k_opt, r21_opt, R21_opt, r23_opt, R23_opt,
                                       data12, data32,
                                       n2_opt, N2_opt,
                                       max_time=200, num_sim=2000):
    """
    Use the best-fit (k_opt, r21_opt, R21_opt, r23_opt, R23_opt) to run a Gillespie-like
    simulation. Then compare sim difference times to the experimental data hist.
    """
    n1_opt = r21_opt * n2_opt
    N1_opt = R21_opt * N2_opt
    n3_opt = r23_opt * n2_opt  # Fixed: Use n2_opt instead of n1_opt
    N3_opt = R23_opt * N2_opt

    initial_proteins1 = N1_opt
    initial_proteins2 = N2_opt
    initial_proteins3 = N3_opt
    initial_proteins = [initial_proteins1,
                        initial_proteins2, initial_proteins3]

    rates = [k_opt, k_opt, k_opt]

    n0_total = n1_opt + n2_opt + n3_opt
    n01_mean = n1_opt
    n02_mean = n2_opt
    n0_list = generate_threshold_values(
        [n01_mean, n02_mean], n0_total, num_sim)

    simulations, separate_times = run_simulations(
        initial_proteins, rates, n0_list, max_time, num_sim
    )

    # Compute time differences
    delta_t12 = [sep[0] - sep[1] for sep in separate_times]
    delta_t32 = [sep[2] - sep[1] for sep in separate_times]

    return delta_t12, delta_t32

if __name__ == "__main__":
    # Load data
    df = pd.read_excel("Data/Chromosome_diff.xlsx")
    # data12 = df['Wildtype12'].dropna().values
    # data32 = df['Wildtype32'].dropna().values
    # data12 = df['DegRateMT12'].dropna().values
    # data32 = df['DegRateMT32'].dropna().values
    data12 = df['ThresholdMT12'].dropna().values
    data32 = df['ThresholdMT32'].dropna().values

    # Define x_grid for plotting
    x_grid = np.linspace(-100, 140, 401)

    # Optimized parameters (replace with your actual optimized values)
    # Parameters: n2 = 5.39, N2 = 82.25, k = 0.0207, r21 = 1.75, r23 = 1.77, R21 = 1.29, R23 = 2.39
    n2_opt = 5.39
    N2_opt = 82.25
    k_opt = 0.0207
    r21_opt = 1.75
    R21_opt = 1.29
    r23_opt = 1.77
    R23_opt = 2.39

    # Compute derived parameters
    n1_opt = r21_opt * n2_opt
    N1_opt = R21_opt * N2_opt
    n3_opt = r23_opt * n2_opt  # Fixed: Use n2_opt instead of n1_opt
    N3_opt = R23_opt * N2_opt

    # Chrom1–Chrom2: Compute MoM normal PDF
    mean12, var12 = compute_moments_mom(n1_opt, N1_opt, n2_opt, N2_opt, k_opt)
    pdf12 = norm.pdf(x_grid, loc=mean12, scale=np.sqrt(var12))

    # Chrom3–Chrom2: Compute MoM normal PDF
    mean32, var32 = compute_moments_mom(n3_opt, N3_opt, n2_opt, N2_opt, k_opt)
    pdf32 = norm.pdf(x_grid, loc=mean32, scale=np.sqrt(var32))

    # Run stochastic simulation and plot
    delta_t12, delta_t32 = run_stochastic_simulation_and_plot(
        k_opt, r21_opt, R21_opt,
        r23_opt, R23_opt,
        data12, data32,
        n2_opt, N2_opt
    )

    # Plot experimental data vs MoM normal PDF
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].hist(data12, bins=16, density=True, alpha=0.4, label='data12')
    ax[0].hist(delta_t12, bins=16, density=True, alpha=0.4, label='Sim data12')
    ax[0].plot(x_grid, pdf12, 'r-', label='MoM Normal pdf12')
    ax[0].set_xlim(min(x_grid)-20, max(x_grid)+20)
    ax[0].set_title("Chrom1 - Chrom2")
    ax[0].legend()

    ax[1].hist(data32, bins=16, density=True, alpha=0.4, label='data32')
    ax[1].hist(delta_t32, bins=16, density=True, alpha=0.4, label='Sim data32')
    ax[1].plot(x_grid, pdf32, 'r-', label='MoM Normal pdf32')
    ax[1].set_xlim(min(x_grid)-20, max(x_grid)+20)
    ax[1].set_title("Chrom3 - Chrom2")
    ax[1].legend()

    plt.tight_layout()
    plt.show()
    fig.savefig('Results/theoryfittest3.png', dpi=300, bbox_inches='tight')

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

# Best negative log-likelihood: 1133.3325581290803
# Best parameters:
#   n2=5.00, N2=100.11
#   k=0.0376, r1=0.5217, R1=0.50
#   r2=1.6931, R2=2.50

# Best negative log-likelihood: 1132.2204025967908
# Best parameters:
#   n2=2.00, N2=100.13
#   k=0.0570, r1=0.5000, R1=0.50
#   r2=1.5696, R2=2.75

# Best Solution After Local Optimization 1:
# Negative Log-Likelihood: 1132.1302
# Parameters: n2 = 4.94, N2 = 91.56, k = 0.0409, r21 = 0.57, r23 = 1.87, R21 = 0.48, R23 = 3.37
# Derived: n1 = 2.80, N1 = 44.25, n3 = 9.22, N3 = 308.48

# Best Solution After Local Optimization 1:
# Negative Log-Likelihood: 1132.2197
# Parameters: n2 = 2.87, N2 = 299.48, k = 0.0549, r21 = 0.64, r23 = 2.01, R21 = 0.50, R23 = 4.11
# Derived: n1 = 1.83, N1 = 150.09, n3 = 5.76, N3 = 1231.02

# Best Solution After Local Optimization 1:
# Negative Log-Likelihood: 1132.0759
# Parameters: n2 = 7.89, N2 = 130.25, k = 0.0330, r21 = 0.50, r23 = 1.72, R21 = 0.42, R23 = 2.63
# Derived: n1 = 3.91, N1 = 54.40, n3 = 13.58, N3 = 342.11

# Best Solution After Local Optimization:
# Negative Log-Likelihood: 1132.0761
# Parameters: n2 = 9.26, N2 = 82.92, k = 0.0283, r21 = 0.53, r23 = 1.89, R21 = 0.43, R23 = 2.53
# Derived: n1 = 4.88, N1 = 35.86, n3 = 17.51, N3 = 210.12

# Best Solution After Local Optimization 2:
# Negative Log-Likelihood: 1298.4244
# Parameters: n2 = 4.45, N2 = 98.54, k = 0.0213, r21 = 1.7, r23 = 0.52, R21 = 1.31, R23 = 0.96
# Derived: n1 = 6.17, N1 = 129.17, n3 = 2.82, N3 = 94.78

# Best Solution After Local Optimization 2:
# Negative Log-Likelihood: 1299.2995
# Parameters: n2 = 1.01, N2 = 81.96, k = 0.0323, r21 = 2.25, r23 = 1.17, R21 = 1.43, R23 = 2.26
# Derived: n1 = 2.28, N1 = 116.99, n3 = 1.19, N3 = 185.28

# Best Solution After Local Optimization 2:
# Negative Log-Likelihood: 1298.4367
# Parameters: n2 = 2.30, N2 = 397.22, k = 0.0280, r21 = 2.08, r23 = 0.64, R21 = 1.58, R23 = 1.24
# Derived: n1 = 4.77, N1 = 629.06, n3 = 1.48, N3 = 491.54

# Best Solution After Local Optimization 3:
# Negative Log-Likelihood: 361.8008
# Parameters: n2 = 5.39, N2 = 82.25, k = 0.0207, r21 = 1.75, r23 = 1.77, R21 = 1.29, R23 = 2.39
# Derived: n1 = 9.43, N1 = 106.33, n3 = 9.56, N3 = 196.87

# Best Solution After Local Optimization 3:
# Negative Log-Likelihood: 361.8050
# Parameters: n2 = 3.14, N2 = 399.83, k = 0.0267, r21 = 2.08, r23 = 1.88, R21 = 1.27, R23 = 2.38
# Derived: n1 = 6.51, N1 = 509.52, n3 = 5.91, N3 = 950.16

# Best Solution After Local Optimization 3:
# Negative Log-Likelihood: 361.8026
# Parameters: n2 = 1.68, N2 = 80.47, k = 0.0381, r21 = 1.98, r23 = 2.02, R21 = 1.35, R23 = 4.18
# Derived: n1 = 3.33, N1 = 108.66, n3 = 3.40, N3 = 336.73
