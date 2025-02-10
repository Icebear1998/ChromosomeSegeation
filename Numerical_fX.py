import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# Define the PDF of separation time f_tau(t; k, n, N)
def f_tau(t, k, n, N):
    """
    Compute the PDF f_tau(t, theta) for a given t and parameters theta = (k, n, N).
    """
    if t < 0:
        return 0
    coeff = k * np.math.factorial(N) / (np.math.factorial(n) * np.math.factorial(N - n - 1))
    return coeff * np.exp(-(n + 1) * k * t) * (1 - np.exp(-k * t))**(N - n - 1)

# Compute the PDF of X = tau1 - tau2 using numerical convolution
def f_X(x, k1, n1, N1, k2, n2, N2):
    """
    Compute the PDF f_X(x) using numerical integration.
    """
    # Adjust the integration limits to ensure t - x >= 0
    integrand = lambda t: f_tau(t, k1, n1, N1) * f_tau(t - x, k2, n2, N2)
    lower_limit = max(0, x)  # Ensure t >= x (since t - x >= 0)
    result, _ = integrate.quad(integrand, lower_limit, np.inf, limit=100)
    return result

# Define parameters for the two chromosomes
k1, n1, N1 = 0.1, 4, 100  # Parameters for tau1
k2, n2, N2 = 0.1, 5, 120  # Parameters for tau2

# Define the range of x values (difference in separation times)
x_values = np.linspace(-15, 15, 200)

# Compute f_X(x) over the range of x values
f_x_values = [f_X(x, k1, n1, N1, k2, n2, N2) for x in x_values]

# Normalize the PDF (ensure it integrates to 1)
f_x_values = np.array(f_x_values)
f_x_values /= integrate.trapz(f_x_values, x_values)

# Plot the result
plt.figure(figsize=(8, 5))
plt.plot(x_values, f_x_values, label="Numerically Integrated PDF")

# Add parameter annotations using axes coordinates
params_text_12 = f"k1={k1:.2f}, n1={n1:.2f}, N1={N1:.2f}\n" \
                 f"k2={k2:.2f}, n2={n2:.2f}, N2={N2:.2f}"
plt.text(0.05, 0.95, params_text_12, fontsize=10, transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

plt.xlabel("Difference in Separation Time (X = tau1 - tau2)")
plt.ylabel("Density")
plt.title("Numerical Estimation of f_X(x) (Corrected)")
plt.legend()
plt.grid()
plt.show()