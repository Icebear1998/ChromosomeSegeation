import numpy as np
import matplotlib.pyplot as plt
from MultiMechanismSimulation import MultiMechanismSimulation
from scipy.stats import norm

# Define MoM calculation function directly in this file
def compute_moments_mom_timevarying(n_i, N_i, n_j, N_j, k_0, k_1):
    """
    Compute Method of Moments mean and variance for f_X = T_i - T_j with time-varying
    degradation rate k(t) = k_0 + k_1 t.
    """
    # Compute deterministic time T_i^* and T_j^*
    def solve_time(N, n, k0, k1):
        if k1 == 0:
            return (np.log(N / n) / k0) if k0 > 0 else np.inf
        discriminant = k0**2 + 2 * k1 * np.log(N / n)
        if discriminant < 0:
            return np.inf
        return (-k0 + np.sqrt(discriminant)) / k1 if k1 > 0 else np.inf

    T_i_star = solve_time(N_i, n_i, k_0, k_1)
    T_j_star = solve_time(N_j, n_j, k_0, k_1)

    # Mean approximation using deterministic times
    mean_T_i = T_i_star
    mean_T_j = T_j_star

    # Variance approximation using effective rate at midpoint
    if T_i_star > 0 and k_0 > 0:
        k_mid_i = k_0 + k_1 * (T_i_star / 2)
        var_T_i = sum(1 / ((k_mid_i * m)**2) for m in range(int(n_i) + 1, int(N_i) + 1))
    else:
        var_T_i = np.inf

    if T_j_star > 0 and k_0 > 0:
        k_mid_j = k_0 + k_1 * (T_j_star / 2)
        var_T_j = sum(1 / ((k_mid_j * m)**2) for m in range(int(n_j) + 1, int(N_j) + 1))
    else:
        var_T_j = np.inf

    mean_X = mean_T_i - mean_T_j
    var_X = var_T_i + var_T_j

    return mean_X, var_X

def compute_moments_mom_timevarying2(n_i, N_i, n_j, N_j, k_0, k_1):
    def compute_tm(m, N, k0, k1):
        if k1 == 0:
            return (1 / k0) * np.log(N / m)
        discriminant = k0**2 + 2 * k1 * np.log(N / m)
        return (-k0 + np.sqrt(discriminant)) / k1

    def kt(t, k0, k1):
        return k0 + k1 * t

    def mom_time_varying_k(N, n, k0, k1):
        expected_time = 0
        variance = 0
        for m in range(n + 1, N + 1):
            t_m = compute_tm(m, N, k0, k1)
            rate = kt(t_m, k0, k1) * m
            tau_mean = 1 / rate
            expected_time += tau_mean
            variance += tau_mean**2
        return expected_time, variance
    
    mean_T_i, var_T_i = mom_time_varying_k(N_i, n_i, k_0, k_1)
    mean_T_j, var_T_j = mom_time_varying_k(N_j, n_j, k_0, k_1)
    mean_X = mean_T_i - mean_T_j
    var_X = var_T_i + var_T_j
    return mean_X, var_X

def compute_pdf_mom_timevarying(x_grid, n_i, N_i, n_j, N_j, k_0, k_1):
    mean_X, var_X = compute_moments_mom_timevarying2(n_i, N_i, n_j, N_j, k_0, k_1)
    return norm.pdf(x_grid, loc=mean_X, scale=np.sqrt(var_X))

def run_time_varying_k_simulation(N1, N2, N3, n1, n2, n3, k_0, k_1, max_time=500, num_sim=1000):
    initial_proteins = [N1, N2, N3]
    rate_params = {'k_0': k_0, 'k_1': k_1}
    n0_list = [n1, n2, n3]
    
    simulation = MultiMechanismSimulation(
        mechanism='time_varying_k',
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
    N1 = 150
    N2 = 250
    N3 = 450
    n1 = 5
    n2 = 10
    n3 = 16
    k_0 = 0.08
    k_1 = 0.008
    max_time = 500
    num_sim = 2000
    
    delta_t12, delta_t32 = run_time_varying_k_simulation(
        N1, N2, N3, n1, n2, n3, k_0, k_1, max_time, num_sim
    )
    
    emp_mean12 = np.mean(delta_t12)
    emp_var12 = np.var(delta_t12)
    emp_mean32 = np.mean(delta_t32)
    emp_var32 = np.var(delta_t32)
    
    mom_mean12, mom_var12 = compute_moments_mom_timevarying(
        n1, N1, n2, N2, k_0, k_1
    )
    mom_mean32, mom_var32 = compute_moments_mom_timevarying(
        n3, N3, n2, N2, k_0, k_1
    )
    
    print("T1 - T2:")
    print(f"Empirical Mean: {emp_mean12:.4f}, MoM Mean: {mom_mean12:.4f}")
    print(f"Empirical Variance: {emp_var12:.4f}, MoM Variance: {mom_var12:.4f}")
    print("T3 - T2:")
    print(f"Empirical Mean: {emp_mean32:.4f}, MoM Mean: {mom_mean32:.4f}")
    print(f"Empirical Variance: {emp_var32:.4f}, MoM Variance: {mom_var32:.4f}")
    
    x_grid = np.linspace(-50, 50, 401)
    pdf12 = compute_pdf_mom_timevarying(
        x_grid, n1, N1, n2, N2, k_0, k_1
    )
    pdf32 = compute_pdf_mom_timevarying(
        x_grid, n3, N3, n2, N2, k_0, k_1
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(delta_t12, bins=20, density=True, alpha=0.4, label='Simulated T1 - T2')
    #ax1.plot(x_grid, pdf12, 'r-', label='MoM PDF')
    ax1.set_title('T1 - T2 (Time-Varying k(t))')
    ax1.set_xlabel('Time Difference')
    ax1.legend()
    
    ax2.hist(delta_t32, bins=20, density=True, alpha=0.4, label='Simulated T3 - T2')
    #ax2.plot(x_grid, pdf32, 'r-', label='MoM PDF')
    ax2.set_title('T3 - T2 (Time-Varying k(t))')
    ax2.set_xlabel('Time Difference')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    #plt.savefig('time_varying_k_comparison.png')
    # Removed plt.show() as per matplotlib guidelines

if __name__ == "__main__":
    test_mom_matching()