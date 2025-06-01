import numpy as np
import matplotlib.pyplot as plt
from math import log
from scipy.integrate import quad
from scipy.special import gamma

###############################################################################
# 1) Single-chromosome PDF: f_tau(t; k, n, N)
###############################################################################
def f_tau(t, k, n, N):
    """
    PDF of time to separate (the (n+1)-th event among N exponentials with rate k),
    generalized to real n, N via gamma(...).
    """
    if t < 0:
        return 0.0
    
    # "Binomial" coefficient generalized
    try:
        comb_factor = gamma(N + 1.0) / (gamma(n + 1.0) * gamma(N - n))
    except ValueError:
        return 0.0
    
    val = k * comb_factor \
          * np.exp(-(n + 1.0)*k*t) \
          * (1.0 - np.exp(-k*t))**( (N - n - 1.0) )
    if val < 0:
        return 0.0
    return val

###############################################################################
# 2) Difference PDF: f_diff(z; k, n1,N1, n2,N2)
#    = ∫ f_tau1(t) * f_tau2(t - z) dt, from t=max(0,z) to ∞
###############################################################################
def f_diff(z, k, n1, N1, n2, N2):
    lower_limit = max(0.0, z)
    def integrand(t):
        return f_tau(t,      k, n1, N1) \
             * f_tau(t - z, k, n2, N2)
    val, _ = quad(integrand, lower_limit, np.inf, epsabs=1e-8, epsrel=1e-8)
    return val

###############################################################################
# 3) Function to compute f_X(z) for a given parameter dictionary
###############################################################################
def compute_fX(params, zvals):
    """
    Returns an array of f_X(z) for each z in zvals, given a dictionary of parameters:
      { "k":..., "n1":..., "N1":..., "n2":..., "N2":... }
    """
    k   = params["k"]
    n1  = params["n1"]
    N1  = params["N1"]
    n2  = params["n2"]
    N2  = params["N2"]
    
    pdf_vals = []
    for z in zvals:
        pdf_vals.append(f_diff(z, k, n1, N1, n2, N2))
    return np.array(pdf_vals)

###############################################################################
# 4) Baseline parameters
###############################################################################
params_baseline = {
    "k":  0.08,
    "n1": 4.0,
    "N1": 100.0,
    "n2": 4.0,
    "N2": 120.0
}

###############################################################################
# 5) OAT Sensitivity: For each parameter, vary ±10% & compare
###############################################################################
perturb_fraction = 0.10  # 10%

#  We'll define a range of z. For example, from -30 to +30:
zvals = np.linspace(-30, 30, 201)

# Compute baseline curve
fX_baseline = compute_fX(params_baseline, zvals)

# We'll store results for each parameter
params_list = list(params_baseline.keys())

results = {}
for p in params_list:
    base_val = params_baseline[p]
    
    # +10%
    plus_val = base_val * (1.0 + perturb_fraction)
    # -10%
    minus_val = base_val * (1.0 - perturb_fraction)
    
    # Construct dicts for +/-
    params_plus  = dict(params_baseline)
    params_plus[p] = plus_val
    fX_plus = compute_fX(params_plus, zvals)
    
    params_minus = dict(params_baseline)
    params_minus[p] = minus_val
    fX_minus = compute_fX(params_minus, zvals)
    
    results[p] = {
        "plus_val": plus_val,
        "minus_val": minus_val,
        "fX_plus": fX_plus,
        "fX_minus": fX_minus
    }

###############################################################################
# 6) Plot single parameter sensitivity
###############################################################################
nrows = len(params_list)
fig, axes = plt.subplots(nrows, 1, figsize=(8, 3*nrows), sharex=True)

if nrows == 1:
    axes = [axes]  # so we can iterate

for i, p in enumerate(params_list):
    ax = axes[i]
    
    ax.plot(zvals, fX_baseline, label="Baseline", color="black")
    ax.plot(zvals, results[p]["fX_plus"],
            label=f"{p} +{int(perturb_fraction*100)}%", linestyle="--")
    ax.plot(zvals, results[p]["fX_minus"],
            label=f"{p} -{int(perturb_fraction*100)}%", linestyle=":")
    
    ax.set_title(f"Sensitivity: {p}")
    ax.set_ylabel("f_X(z)")
    ax.legend()

axes[-1].set_xlabel("z (difference)")

plt.tight_layout()

# Save the plot to a file
plt.savefig('Results/paramana_single.png') # Saves as PNG by default
plt.show()

###############################################################################
# 7) Analyze combinations of parameters
###############################################################################
combinations = [
    ("n1", "n2"),
    ("n1", "N1"),
    ("N1", "N2")
]

results_comb = {}
for p1, p2 in combinations:
    base_val1 = params_baseline[p1]
    base_val2 = params_baseline[p2]
    
    # +10% for both
    plus_val1 = base_val1 * (1.0 + perturb_fraction)
    plus_val2 = base_val2 * (1.0 + perturb_fraction)
    
    # -10% for both
    minus_val1 = base_val1 * (1.0 - perturb_fraction)
    minus_val2 = base_val2 * (1.0 - perturb_fraction)
    
    # Construct dicts for combinations
    params_plus_plus = dict(params_baseline)
    params_plus_plus[p1] = plus_val1
    params_plus_plus[p2] = plus_val2
    fX_plus_plus = compute_fX(params_plus_plus, zvals)
    
    params_minus_minus = dict(params_baseline)
    params_minus_minus[p1] = minus_val1
    params_minus_minus[p2] = minus_val2
    fX_minus_minus = compute_fX(params_minus_minus, zvals)
    
    params_plus_minus = dict(params_baseline)
    params_plus_minus[p1] = plus_val1
    params_plus_minus[p2] = minus_val2
    fX_plus_minus = compute_fX(params_plus_minus, zvals)
    
    params_minus_plus = dict(params_baseline)
    params_minus_plus[p1] = minus_val1
    params_minus_plus[p2] = plus_val2
    fX_minus_plus = compute_fX(params_minus_plus, zvals)
    
    results_comb[(p1, p2)] = {
        "plus_plus": fX_plus_plus,
        "minus_minus": fX_minus_minus,
        "plus_minus": fX_plus_minus,
        "minus_plus": fX_minus_plus
    }

###############################################################################
# 8) Plot combinations of parameters
###############################################################################
nrows_comb = len(combinations)
fig_comb, axes_comb = plt.subplots(nrows_comb, 1, figsize=(8, 3*nrows_comb), sharex=True)

if nrows_comb == 1:
    axes_comb = [axes_comb]  # so we can iterate

for i, (p1, p2) in enumerate(combinations):
    ax = axes_comb[i]
    
    ax.plot(zvals, fX_baseline, label="Baseline", color="black")
    ax.plot(zvals, results_comb[(p1, p2)]["plus_plus"],
            label=f"{p1} +{int(perturb_fraction*100)}%, {p2} +{int(perturb_fraction*100)}%", linestyle="--")
    ax.plot(zvals, results_comb[(p1, p2)]["minus_minus"],
            label=f"{p1} -{int(perturb_fraction*100)}%, {p2} -{int(perturb_fraction*100)}%", linestyle=":")
    ax.plot(zvals, results_comb[(p1, p2)]["plus_minus"],
            label=f"{p1} +{int(perturb_fraction*100)}%, {p2} -{int(perturb_fraction*100)}%", linestyle="-.")
    ax.plot(zvals, results_comb[(p1, p2)]["minus_plus"],
            label=f"{p1} -{int(perturb_fraction*100)}%, {p2} +{int(perturb_fraction*100)}%", linestyle="-.")
    
    ax.set_title(f"Sensitivity: {p1} and {p2}")
    ax.set_ylabel("f_X(z)")
    ax.legend()

axes_comb[-1].set_xlabel("z (difference)")

plt.tight_layout()

# Save the plot to a file
plt.savefig('Results/paramana_comb.png') # Saves as PNG by default
plt.show()
