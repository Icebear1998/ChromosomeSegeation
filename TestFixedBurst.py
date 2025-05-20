import numpy as np
import matplotlib.pyplot as plt
from MultiMechanismSimulation import MultiMechanismSimulation
from MoMCalculations import compute_moments_mom, compute_pdf_mom

def run_fixed_burst_simulation(Ns, ns, lambda_rate, burst_size, burst_var=None, max_time=500, num_sim=1000):
    initial_proteins = Ns
    rate_params = {'lambda_list': [lambda_rate, lambda_rate, lambda_rate], 'burst_size': burst_size, 'var_burst_size': burst_var}
    n0_list = ns  # Uniform thresholds
    
    simulation = MultiMechanismSimulation(
        mechanism='fixed_burst',
        initial_state_list=initial_proteins,
        rate_params=rate_params,
        n0_list=n0_list,
        max_time=max_time
    )
    
    delta_t12 = []
    delta_t32 = []
    for _ in range(num_sim):
        _, _, sep_times = simulation.simulate()
        delta_t12.append(sep_times[0] - sep_times[1])
        delta_t32.append(sep_times[2] - sep_times[1])
    
    return np.array(delta_t12), np.array(delta_t32)

def test_mom_matching():
    # Parameters
    N1 = 100  # Initial cohesins
    N2 = 250  # Initial cohesins
    N3 = 450  # Initial cohesins
    n1 = 5   # Threshold cohesins
    n2 = 10   # Threshold cohesins
    n3 = 12   # Threshold cohesins
    lambda_rate = 0.05  # Burst rate
    burst_size = 8      # Burst size
    burst_var = 4     # Burst variance
    max_time = 500
    num_sim = 1000
    
    # Run simulations
    delta_t12, delta_t32 = run_fixed_burst_simulation([N1, N2, N3], [n1, n2, n3], lambda_rate, burst_size, burst_var, max_time, num_sim)
    
    # Empirical moments
    emp_mean12 = np.mean(delta_t12)
    emp_var12 = np.var(delta_t12)
    emp_mean32 = np.mean(delta_t32)
    emp_var32 = np.var(delta_t32)
    
    # MoM moments
    mom_mean12, mom_var12 = compute_moments_mom('fixed_burst', n1, N1, n2, N2, lambda_rate, burst_size)
    mom_mean32, mom_var32 = compute_moments_mom('fixed_burst', n3, N3, n2, N2, lambda_rate, burst_size)
    
    # Print comparison
    print("T1 - T2:")
    print(f"Empirical Mean: {emp_mean12:.4f}, MoM Mean: {mom_mean12:.4f}")
    print(f"Empirical Variance: {emp_var12:.4f}, MoM Variance: {mom_var12:.4f}")
    print("T3 - T2:")
    print(f"Empirical Mean: {emp_mean32:.4f}, MoM Mean: {mom_mean32:.4f}")
    print(f"Empirical Variance: {emp_var32:.4f}, MoM Variance: {mom_var32:.4f}")
    
    # Plot histograms with MoM PDF
    x_grid = np.linspace(-40, 40, 401)
    pdf12 = compute_pdf_mom('fixed_burst', x_grid, n1, N1, n2, N2, lambda_rate, burst_size)
    pdf32 = compute_pdf_mom('fixed_burst', x_grid, n3, N3, n2, N2, lambda_rate, burst_size)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(delta_t12, bins=20, density=True, alpha=0.4, label='Simulated T1 - T2')
    ax1.plot(x_grid, pdf12, 'r-', label='MoM PDF')
    ax1.set_title('T1 - T2 (Fixed Burst)')
    ax1.set_xlabel('Time Difference')
    ax1.legend()
    
    ax2.hist(delta_t32, bins=20, density=True, alpha=0.4, label='Simulated T3 - T2')
    ax2.plot(x_grid, pdf32, 'r-', label='MoM PDF')
    ax2.set_title('T3 - T2 (Fixed Burst)')
    ax2.set_xlabel('Time Difference')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_mom_matching()



# import numpy as np
# import matplotlib.pyplot as plt
# from MultiMechanismSimulation import MultiMechanismSimulation
# from scipy.stats import norm

# def compute_moments_mom_random_normal(n_i, N_i, n_j, N_j, rate_i, rate_j, mean_burst_size, var_burst_size):
#     num_bursts_i = (N_i - n_i) / mean_burst_size - 1
#     num_bursts_j = (N_j - n_j) / mean_burst_size - 1
#     if num_bursts_i < 0 or num_bursts_j < 0:
#         raise ValueError("Number of bursts must be non-negative.")

#     mean_Ti = sum(1 / (rate_i * (N_i - m * mean_burst_size)) for m in range(int(np.floor(num_bursts_i))))
#     mean_Tj = sum(1 / (rate_j * (N_j - m * mean_burst_size)) for m in range(int(np.floor(num_bursts_j))))
#     var_Ti = sum(1 / (rate_i * (N_i - m * mean_burst_size))**2 for m in range(int(np.floor(num_bursts_i))))
#     var_Tj = sum(1 / (rate_j * (N_j - m * mean_burst_size))**2 for m in range(int(np.floor(num_bursts_j))))

#     mean_X = mean_Ti - mean_Tj
#     var_X = var_Ti + var_Tj
#     return mean_X, var_X

# def compute_pdf_mom_random_normal(x_grid, n_i, N_i, n_j, N_j, rate_i, rate_j, mean_burst_size, var_burst_size):
#     mean_X, var_X = compute_moments_mom_random_normal(n_i, N_i, n_j, N_j, rate_i, rate_j, mean_burst_size, var_burst_size)
#     return norm.pdf(x_grid, loc=mean_X, scale=np.sqrt(var_X))

# def run_random_normal_burst_simulation(N1, N2, N3, n1, n2, n3, lambda_rate, mean_burst_size, var_burst_size, max_time=500, num_sim=1000):
#     initial_proteins = [N1, N2, N3]
#     rate_params = {
#         'lambda_list': [lambda_rate, lambda_rate, lambda_rate],
#         'mean_burst_size': mean_burst_size,
#         'var_burst_size': var_burst_size
#     }
#     n0_list = [n1, n2, n3]
    
#     simulation = MultiMechanismSimulation(
#         mechanism='random_normal_burst',
#         initial_state_list=initial_proteins,
#         rate_params=rate_params,
#         n0_list=n0_list,
#         max_time=max_time
#     )
    
#     delta_t12 = []
#     delta_t32 = []
#     for _ in range(num_sim):
#         _, _, sep_times = simulation.simulate()
#         delta_t12.append(sep_times[0] - sep_times[1])
#         delta_t32.append(sep_times[2] - sep_times[1])
    
#     return np.array(delta_t12), np.array(delta_t32)

# def test_mom_matching():
#     N1 = 150
#     N2 = 250
#     N3 = 450
#     n1 = 5
#     n2 = 10
#     n3 = 16
#     lambda_rate = 0.05
#     mean_burst_size = 5
#     var_burst_size = 1
#     max_time = 500
#     num_sim = 2000
    
#     delta_t12, delta_t32 = run_random_normal_burst_simulation(
#         N1, N2, N3, n1, n2, n3, lambda_rate, mean_burst_size, var_burst_size, max_time, num_sim
#     )
    
#     emp_mean12 = np.mean(delta_t12)
#     emp_var12 = np.var(delta_t12)
#     emp_mean32 = np.mean(delta_t32)
#     emp_var32 = np.var(delta_t32)
    
#     mom_mean12, mom_var12 = compute_moments_mom_random_normal(
#         n1, N1, n2, N2, lambda_rate, lambda_rate, mean_burst_size, var_burst_size
#     )
#     mom_mean32, mom_var32 = compute_moments_mom_random_normal(
#         n3, N3, n2, N2, lambda_rate, lambda_rate, mean_burst_size, var_burst_size
#     )
    
#     print("T1 - T2:")
#     print(f"Empirical Mean: {emp_mean12:.4f}, MoM Mean: {mom_mean12:.4f}")
#     print(f"Empirical Variance: {emp_var12:.4f}, MoM Variance: {mom_var12:.4f}")
#     print("T3 - T2:")
#     print(f"Empirical Mean: {emp_mean32:.4f}, MoM Mean: {mom_mean32:.4f}")
#     print(f"Empirical Variance: {emp_var32:.4f}, MoM Variance: {mom_var32:.4f}")
    
#     x_grid = np.linspace(-50, 50, 401)
#     pdf12 = compute_pdf_mom_random_normal(
#         x_grid, n1, N1, n2, N2, lambda_rate, lambda_rate, mean_burst_size, var_burst_size
#     )
#     pdf32 = compute_pdf_mom_random_normal(
#         x_grid, n3, N3, n2, N2, lambda_rate, lambda_rate, mean_burst_size, var_burst_size
#     )
    
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
#     ax1.hist(delta_t12, bins=20, density=True, alpha=0.4, label='Simulated T1 - T2')
#     ax1.plot(x_grid, pdf12, 'r-', label='MoM PDF')
#     ax1.set_title('T1 - T2 (Random Normal Burst)')
#     ax1.set_xlabel('Time Difference')
#     ax1.legend()
    
#     ax2.hist(delta_t32, bins=20, density=True, alpha=0.4, label='Simulated T3 - T2')
#     ax2.plot(x_grid, pdf32, 'r-', label='MoM PDF')
#     ax2.set_title('T3 - T2 (Random Normal Burst)')
#     ax2.set_xlabel('Time Difference')
#     ax2.legend()
    
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     test_mom_matching()