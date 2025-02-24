import numpy as np
import pandas as pd
from math import log
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import gamma

###############################################################################
# 1) Read difference data
###############################################################################
df = pd.read_excel("Chromosome_diff.xlsx")
data_12 = df["SCSdiff_Wildtype12"].dropna().values
data_23 = df["SCSdiff_Wildtype23"].dropna().values

###############################################################################
# 2) Fix base chromosome-1 parameters
###############################################################################
n1 = 4.0
N1 = 100.0

###############################################################################
# 3) Single-chromosome PDF f_tau(t; k, n, N)
###############################################################################
def f_tau(t, k, n, N):
    """PDF of time to separate (n+1)-th event out of N exponentials, generalized to real n,N."""
    if t < 0:
        return 0.0
    # gamma-based generalization of factorial
    try:
        comb_factor = gamma(N + 1.0) / (gamma(n + 1.0) * gamma(N - n))
    except ValueError:
        return 0.0
    val = (k * comb_factor
           * np.exp(-(n+1.0)*k*t)
           * (1.0 - np.exp(-k*t))**( (N - n - 1.0) ))
    if val < 0:
        return 0.0
    return val

###############################################################################
# 4) PDF of the difference X = tauA - tauB
###############################################################################
def f_diff(z, k, nA, NA, nB, NB):
    """PDF of X = tauA - tauB, via convolution."""
    lower_limit = max(0.0, z)
    def integrand(t):
        return f_tau(t, k, nA, NA) * f_tau(t - z, k, nB, NB)
    val, _ = quad(integrand, lower_limit, np.inf, epsabs=1e-8, epsrel=1e-8)
    return val

###############################################################################
# 5) Log-likelihood for a single difference dataset
###############################################################################
def loglike_diff(diff_data, k, nA, NA, nB, NB):
    total_ll = 0.0
    for x in diff_data:
        p = f_diff(x, k, nA, NA, nB, NB)
        if p <= 0:
            return -np.inf
        total_ll += log(p)
    return total_ll

###############################################################################
# 6) Combined log-likelihood for data_12 (chr1-chr2) + data_23 (chr2-chr3)
###############################################################################
def total_loglike(k, r2, r3, R2, R3, data_12, data_23, n1, N1):
    """
    We interpret n2 = r2*n1, N2 = R2*N1, n3 = r3*n1, N3 = R3*N1.
    Then we combine the log-likelihood for (1-2) and (2-3).
    """
    n2 = r2 * n1
    N2 = R2 * N1
    n3 = r3 * n1
    N3 = R3 * N1
    
    ll_12 = loglike_diff(data_12, k, n1, N1, n2, N2)
    if ll_12 == -np.inf:
        return -np.inf
    ll_23 = loglike_diff(data_23, k, n2, N2, n3, N3)
    return ll_12 + ll_23 if ll_23 != -np.inf else -np.inf

def negative_loglike_for_optimizer(x, R2, R3, data_12, data_23, n1, N1):
    """
    x = [k, r2, r3]  (the variables we'll optimize).
    R2, R3 are fixed in the current iteration of the grid.
    Return negative log-likelihood (so we can minimize).
    """
    k, r2, r3 = x
    # quick checks on positivity if you like
    if k <= 0 or r2 <= 0 or r3 <= 0:
        return np.inf
    
    ll = total_loglike(k, r2, r3, R2, R3, data_12, data_23, n1, N1)
    if ll == -np.inf:
        return np.inf
    return -ll  # negative for minimization

###############################################################################
# 7) Hybrid approach: grid over R2, R3 and do local optimization for k, r2, r3
###############################################################################
# Example R2,R3 ranges. You can choose fewer or finer steps, depending on speed:
R2_candidates = np.round(np.arange(0.5, 2.01, 0.1), 1)  # e.g. 0.5->2.0 by 0.1
R3_candidates = np.round(np.arange(0.5, 2.01, 0.1), 1)

best_loglike  = -np.inf
best_solution = None

for R2_ in R2_candidates:
    for R3_ in R3_candidates:
        
        # We'll run a local optimizer on x=[k, r2, r3].
        # Provide an initial guess:
        #   e.g. k=0.08, r2=1.0, r3=1.0
        x0 = [0.08, 1.0, 1.0]
        
        # Possibly provide bounds for [k, r2, r3], e.g.:
        bnds = [(0.02, 0.2), (0.25, 4.0), (0.25, 4.0)]
        
        # We do local optimization:
        result = minimize(
            fun=negative_loglike_for_optimizer,
            x0=x0,
            args=(R2_, R3_, data_12, data_23, n1, N1),
            method="L-BFGS-B",
            bounds=bnds
        )
        
        if not result.success:
            # The optimizer might fail for some R2_, R3_ combos, skip it
            continue
        
        # Evaluate the best log-likelihood from this local optimum
        k_opt, r2_opt, r3_opt = result.x
        current_ll = -result.fun  # because we returned negative log-likelihood
        if current_ll > best_loglike:
            best_loglike  = current_ll
            best_solution = (k_opt, r2_opt, r3_opt, R2_, R3_)

###############################################################################
# 8) Final best result
###############################################################################
if best_solution is None:
    print("No successful optimization found in the (R2,R3) grid.")
else:
    print("Best log-likelihood:", best_loglike)
    k_opt, r2_opt, r3_opt, R2_opt, R3_opt = best_solution
    print(f"Parameters found:\n"
          f"  k   = {k_opt}\n"
          f"  r2  = {r2_opt},  R2 = {R2_opt}\n"
          f"  r3  = {r3_opt},  R3 = {R3_opt}\n"
          f"  n2  = {r2_opt*n1:.3f}, N2 = {R2_opt*N1:.3f}\n"
          f"  n3  = {r3_opt*n1:.3f}, N3 = {R3_opt*N1:.3f}")
