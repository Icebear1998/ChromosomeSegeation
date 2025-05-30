import numpy as np
import matplotlib.pyplot as plt
from MultiMechanismSimulation import MultiMechanismSimulation
from scipy.stats import norm
from MoMCalculations import compute_pdf_mom, compute_moments_mom



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
    k_0 = 0.05
    k_1 = 0.001
    max_time = 500
    num_sim = 2000
    
    delta_t12, delta_t32 = run_time_varying_k_simulation(
        N1, N2, N3, n1, n2, n3, k_0, k_1, max_time, num_sim
    )
    
    emp_mean12 = np.mean(delta_t12)
    emp_var12 = np.var(delta_t12)
    emp_mean32 = np.mean(delta_t32)
    emp_var32 = np.var(delta_t32)
    
    mom_mean12, mom_var12 = compute_moments_mom(
        'time_varying_k', n1, N1, n2, N2, k_0, k_1 = k_1
    )
    mom_mean32, mom_var32 = compute_moments_mom(
        'time_varying_k', n3, N3, n2, N2, k_0, k_1 = k_1
    )
    
    print("T1 - T2:")
    print(f"Empirical Mean: {emp_mean12:.4f}, MoM Mean: {mom_mean12:.4f}")
    print(f"Empirical Variance: {emp_var12:.4f}, MoM Variance: {mom_var12:.4f}")
    print("T3 - T2:")
    print(f"Empirical Mean: {emp_mean32:.4f}, MoM Mean: {mom_mean32:.4f}")
    print(f"Empirical Variance: {emp_var32:.4f}, MoM Variance: {mom_var32:.4f}")
    
    x_grid = np.linspace(-30, 30, 401)
    pdf12 = compute_pdf_mom(
        'time_varying_k', x_grid, n1, N1, n2, N2, k_0, k_1 = k_1
    )
    pdf32 = compute_pdf_mom(
        'time_varying_k', x_grid, n3, N3, n2, N2, k_0, k_1 = k_1
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(delta_t12, bins=20, density=True, alpha=0.4, label='Simulated T1 - T2')
    ax1.plot(x_grid, pdf12, 'r-', label='MoM PDF')
    ax1.set_title('T1 - T2 (Time-Varying k(t))')
    ax1.set_xlabel('Time Difference')
    ax1.legend()
    
    ax2.hist(delta_t32, bins=20, density=True, alpha=0.4, label='Simulated T3 - T2')
    ax2.plot(x_grid, pdf32, 'r-', label='MoM PDF')
    ax2.set_title('T3 - T2 (Time-Varying k(t))')
    ax2.set_xlabel('Time Difference')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    #plt.savefig('time_varying_k_comparison.png')
    # Removed plt.show() as per matplotlib guidelines

if __name__ == "__main__":
    test_mom_matching()