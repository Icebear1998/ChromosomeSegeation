import numpy as np
from math import log, exp
from scipy.integrate import quad
from scipy.special import gamma
from Chromosomes_Theory import f_diff_gamma
# Updated import from SALib
from SALib.sample import sobol
from SALib.analyze import sobol as sobol_analyze


def mean_diff(k, n1, N1, n2, N2):
    """
    Approximate the mean E[X] of X = tau1 - tau2 
    by integrating z*f_X(z) from zmin to zmax.
    """
    zmin, zmax = -50.0, 50.0
    num_points = 301
    zvals = np.linspace(zmin, zmax, num_points)

    total = 0.0
    for i in range(num_points - 1):
        z_mid = 0.5*(zvals[i] + zvals[i+1])
        fx = f_diff_gamma(z_mid, k, n1, N1, n2, N2)
        dz = (zvals[i+1] - zvals[i])
        total += z_mid * fx * dz
    return total


###############################################################################
# Define the SALib problem (k is now in log-space)
###############################################################################
problem = {
    "num_vars": 5,
    "names": ["log_k", "n1", "N1", "n2", "N2"],
    # Bounds for k are now in log-space
    "bounds": [
        [log(0.02), log(0.5)],  # log(k) bounds
        [1.0, 9.0],             # n1
        [100.0, 300.0],         # N1
        [1.0, 9.0],             # n2
        [100.0, 300.0]          # N2
    ]
}

###############################################################################
# Sobol sampling
###############################################################################
# Choose N as a power of 2 (e.g., 128 or 256)
N = 128
print(
    f"Generating Sobol samples with N={N} (must be 2^m for best convergence).")

# param_values will be a 2D array of shape [ (2D+2)*N , D ] where D=5
param_values = sobol.sample(problem, N, calc_second_order=True)

###############################################################################
# Evaluate the model for each parameter set
###############################################################################
Y = []
for row in param_values:
    log_k, n1_, N1_, n2_, N2_ = row
    k_ = exp(log_k)  # Convert log-scale back to linear scale

    # If you want to ensure N>n strictly, you can skip or clamp invalid combos:
    if (N1_ <= n1_) or (N2_ <= n2_) or (n1_ < 0) or (n2_ < 0):
        # Skip or set a fallback
        Y.append(0.0)
        continue

    val = mean_diff(k_, n1_, N1_, n2_, N2_)
    # If val is NaN or infinite, handle it:
    if not np.isfinite(val):
        val = 0.0
    Y.append(val)

Y = np.array(Y)

###############################################################################
# Perform the Sobol analysis
###############################################################################
Si = sobol_analyze.analyze(problem, Y, calc_second_order=True)

print("S1 (First-order) indices:", Si["S1"])
print("S1_conf (First-order conf):", Si["S1_conf"])
print("ST (Total-order) indices:", Si["ST"])
print("ST_conf (Total-order conf):", Si["ST_conf"])

print("\nSecond-order (S2) indices (pairwise):")
print(Si["S2"])
print("S2_conf:")
print(Si["S2_conf"])
