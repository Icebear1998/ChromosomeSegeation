import numpy as np
from math import log
from scipy.integrate import quad
from scipy.special import gamma

# Updated import from SALib
from SALib.sample import sobol
from SALib.analyze import sobol as sobol_analyze

def f_tau(t, k, n, N):
    """
    Single-chromosome PDF:
      f_tau(t) = k * [Gamma(N+1)/(Gamma(n+1)*Gamma(N-n))] 
                 * exp(- (n+1)*k * t)
                 * [1 - exp(-k*t)]^( (N-n-1) ),  for t >= 0
    
    We skip invalid combos or negative results.
    """
    if t < 0:
        return 0.0
    # Check domain for gamma arguments
    if (N < n) or (n < 0):
        return 0.0
    try:
        comb_factor = gamma(N + 1.0) / (gamma(n + 1.0) * gamma(N - n))
    except (ValueError, OverflowError):
        return 0.0
    
    val = k * comb_factor \
          * np.exp(-(n+1.0)*k*t) \
          * (1.0 - np.exp(-k*t))**( (N - n - 1.0) )
    if not np.isfinite(val) or (val < 0):
        return 0.0
    return val

def f_diff(z, k, n1, N1, n2, N2):
    """
    Difference PDF: X = tau1 - tau2
    f_X(z) = ∫ f_tau1(t)*f_tau2(t-z) dt, from t=max(0,z) to ∞.
    """
    lower_limit = max(0.0, z)
    
    def integrand(t):
        return f_tau(t, k, n1, N1) * f_tau(t - z, k, n2, N2)
    
    # Attempt integration; if it fails, we return a small number or 0
    try:
        val, _ = quad(integrand, lower_limit, np.inf, epsabs=1e-8, epsrel=1e-8)
    except (ValueError, OverflowError):
        return 0.0
    if not np.isfinite(val):
        return 0.0
    return val

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
        fx = f_diff(z_mid, k, n1, N1, n2, N2)
        dz = (zvals[i+1] - zvals[i])
        total += z_mid * fx * dz
    return total

###############################################################################
# Define the SALib problem
###############################################################################
problem = {
    "num_vars": 5,
    "names": ["k", "n1", "N1", "n2", "N2"],
    # Ensure your bounds are physically valid: we want N>n. 
    # For demonstration, we do a broad range, but you can refine:
    "bounds": [
        [0.02, 0.2],   # k
        [1.0, 9.0],    # n1
        [10.0, 20.0],  # N1   (ensures N1>n1 from these ranges, if we want)
        [1.0, 9.0],    # n2
        [10.0, 20.0]   # N2   (similarly ensures N2>n2)
    ]
}

###############################################################################
# Sobol sampling
###############################################################################
# Choose N as a power of 2 (e.g., 128 or 256)
N = 128
print(f"Generating Sobol samples with N={N} (must be 2^m for best convergence).")

# param_values will be a 2D array of shape [ (2D+2)*N , D ] where D=5
param_values = sobol.sample(problem, N, calc_second_order=True)

###############################################################################
# Evaluate the model for each parameter set
###############################################################################
Y = []
for row in param_values:
    k_, n1_, N1_, n2_, N2_ = row
    
    # If you want to ensure N>n strictly, you can skip or clamp invalid combos:
    if (N1_ <= n1_) or (N2_ <= n2_) or (n1_ < 0) or (n2_ < 0):
        # skip or set a fallback
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
