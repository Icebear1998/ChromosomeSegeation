import numpy as np
import matplotlib.pyplot as plt
from MultiMechanismSimulation import MultiMechanismSimulation
from scipy.stats import norm
from MoMCalculations import compute_pdf_mom, compute_moments_mom
from simulation_kde import build_kde_from_simulations, evaluate_kde_pdf, calculate_kde_likelihood


def run_simulation(mechanism, N1, N2, N3, n1, n2, n3, rate_params, max_time=500, num_sim=1000):
    """
    Run simulation for any mechanism type.

    Args:
        mechanism (str): 'simple', 'fixed_burst', 'time_varying_k', or 'feedback'
        N1, N2, N3 (float): Initial protein counts
        n1, n2, n3 (float): Threshold protein counts
        rate_params (dict): Mechanism-specific rate parameters
        max_time (float): Maximum simulation time
        num_sim (int): Number of simulations

    Returns:
        tuple: (delta_t12, delta_t32) arrays
    """
    initial_proteins = [N1, N2, N3]
    n0_list = [n1, n2, n3]

    simulation = MultiMechanismSimulation(
        mechanism=mechanism,
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


def build_rate_params(mechanism, k, mechanism_params):
    """
    Build rate_params dictionary for simulation based on mechanism.

    Args:
        mechanism (str): Mechanism name
        k (float): Base degradation rate
        mechanism_params (dict): Mechanism-specific parameters

    Returns:
        dict: Complete rate_params for simulation
    """
    if mechanism == 'time_varying_k':
        # This mechanism does not use the base 'k', only 'k_1'.
        return mechanism_params.copy()

    rate_params = {'k': k}

    if mechanism != 'simple':
        rate_params.update(mechanism_params)
    # simple mechanism only needs k

    return rate_params


def compute_mom_parameters(mechanism, n1, n2, n3, N1, N2, N3, k, mechanism_params):
    """
    Compute MoM parameters based on mechanism type.

    Args:
        mechanism (str): Mechanism name
        n1, n2, n3 (float): Threshold protein counts
        N1, N2, N3 (float): Initial protein counts
        k (float): Base degradation rate
        mechanism_params (dict): Mechanism-specific parameters

    Returns:
        tuple: ((mom_mean12, mom_var12), (mom_mean32, mom_var32))
    """
    if mechanism == 'simple':
        mom_mean12, mom_var12 = compute_moments_mom(
            'simple', n1, N1, n2, N2, k)
        mom_mean32, mom_var32 = compute_moments_mom(
            'simple', n3, N3, n2, N2, k)
    elif mechanism == 'fixed_burst':
        mom_mean12, mom_var12 = compute_moments_mom(
            'fixed_burst', n1, N1, n2, N2, k, burst_size=mechanism_params['burst_size']
        )
        mom_mean32, mom_var32 = compute_moments_mom(
            'fixed_burst', n3, N3, n2, N2, k, burst_size=mechanism_params['burst_size']
        )
    elif mechanism == 'time_varying_k':
        mom_mean12, mom_var12 = compute_moments_mom(
            'time_varying_k', n1, N1, n2, N2, k, k_1=mechanism_params['k_1']
        )
        mom_mean32, mom_var32 = compute_moments_mom(
            'time_varying_k', n3, N3, n2, N2, k, k_1=mechanism_params['k_1']
        )
    elif mechanism == 'feedback_onion':
        mom_mean12, mom_var12 = compute_moments_mom(
            'feedback_onion', n1, N1, n2, N2, k, n_inner=mechanism_params['n_inner']
        )
        mom_mean32, mom_var32 = compute_moments_mom(
            'feedback_onion', n3, N3, n2, N2, k, n_inner=mechanism_params['n_inner']
        )
    elif mechanism == 'fixed_burst_feedback_onion':
        mom_mean12, mom_var12 = compute_moments_mom(
            'fixed_burst_feedback_onion', n1, N1, n2, N2, k, 
            burst_size=mechanism_params['burst_size'],
            n_inner=mechanism_params['n_inner']
        )
        mom_mean32, mom_var32 = compute_moments_mom(
            'fixed_burst_feedback_onion', n3, N3, n2, N2, k,
            burst_size=mechanism_params['burst_size'],
            n_inner=mechanism_params['n_inner']
        )

    return (mom_mean12, mom_var12), (mom_mean32, mom_var32)


def compute_pdf_parameters(mechanism, x_grid, n1, n2, n3, N1, N2, N3, k, mechanism_params):
    """
    Compute PDF parameters based on mechanism type.

    Args:
        mechanism (str): Mechanism name
        x_grid (array): Grid points for PDF evaluation
        n1, n2, n3 (float): Threshold protein counts
        N1, N2, N3 (float): Initial protein counts
        k (float): Base degradation rate
        mechanism_params (dict): Mechanism-specific parameters

    Returns:
        tuple: (pdf12, pdf32)
    """
    if mechanism == 'simple':
        pdf12 = compute_pdf_mom('simple', x_grid, n1, N1, n2, N2, k)
        pdf32 = compute_pdf_mom('simple', x_grid, n3, N3, n2, N2, k)
    elif mechanism == 'fixed_burst':
        pdf12 = compute_pdf_mom(
            'fixed_burst', x_grid, n1, N1, n2, N2, k, burst_size=mechanism_params['burst_size']
        )
        pdf32 = compute_pdf_mom(
            'fixed_burst', x_grid, n3, N3, n2, N2, k, burst_size=mechanism_params['burst_size']
        )
    elif mechanism == 'time_varying_k':
        pdf12 = compute_pdf_mom(
            'time_varying_k', x_grid, n1, N1, n2, N2, k, k_1=mechanism_params['k_1']
        )
        pdf32 = compute_pdf_mom(
            'time_varying_k', x_grid, n3, N3, n2, N2, k, k_1=mechanism_params['k_1']
        )
    elif mechanism == 'feedback_linear':
        pdf12 = compute_pdf_mom(
            'feedback_linear', x_grid, n1, N1, n2, N2, k, w1=mechanism_params['w1'], w2=mechanism_params['w2']
        )
        pdf32 = compute_pdf_mom(
            'feedback_linear', x_grid, n3, N3, n2, N2, k, w1=mechanism_params['w3'], w2=mechanism_params['w2']
        )
    elif mechanism == 'feedback_onion':
        pdf12 = compute_pdf_mom(
            'feedback_onion', x_grid, n1, N1, n2, N2, k, n_inner=mechanism_params['n_inner']
        )
        pdf32 = compute_pdf_mom(
            'feedback_onion', x_grid, n3, N3, n2, N2, k, n_inner=mechanism_params['n_inner']
        )
    elif mechanism == 'fixed_burst_feedback_onion':
        pdf12 = compute_pdf_mom(
            'fixed_burst_feedback_onion', x_grid, n1, N1, n2, N2, k,
            burst_size=mechanism_params['burst_size'],
            n_inner=mechanism_params['n_inner']
        )
        pdf32 = compute_pdf_mom(
            'fixed_burst_feedback_onion', x_grid, n3, N3, n2, N2, k,
            burst_size=mechanism_params['burst_size'],
            n_inner=mechanism_params['n_inner']
        )

    return pdf12, pdf32


def test_mom_matching(mechanism, n1, n2, n3, N1, N2, N3, k, mechanism_params=None, max_time=500, num_sim=2000, kde_bandwidth=None):
    print(f"\n=== Testing {mechanism.upper()} Mechanism ===")

    # Build rate_params for simulation
    rate_params = build_rate_params(mechanism, k, mechanism_params)

    print(
        f"Parameters: N1={N1}, N2={N2}, N3={N3}, n1={n1}, n2={n2}, n3={n3}, k={k}")
    if mechanism_params:
        print(f"Mechanism-specific: {mechanism_params}")

    # Run simulations
    delta_t12, delta_t32 = run_simulation(
        mechanism, N1, N2, N3, n1, n2, n3, rate_params, max_time, num_sim
    )

    # Empirical moments
    emp_mean12 = np.mean(delta_t12)
    emp_var12 = np.var(delta_t12)
    emp_mean32 = np.mean(delta_t32)
    emp_var32 = np.var(delta_t32)

    # MoM moments
    (mom_mean12, mom_var12), (mom_mean32, mom_var32) = compute_mom_parameters(
        mechanism, n1, n2, n3, N1, N2, N3, k, mechanism_params
    )

    # Build KDE from simulations
    kde12 = build_kde_from_simulations(delta_t12, bandwidth=kde_bandwidth)
    kde32 = build_kde_from_simulations(delta_t32, bandwidth=kde_bandwidth)

    # Print comparison
    print("T1 - T2:")
    print(f"  Empirical Mean: {emp_mean12:.4f}, MoM Mean: {mom_mean12:.4f}")
    print(f"  Empirical Variance: {emp_var12:.4f}, MoM Variance: {mom_var12:.4f}")
    print("T3 - T2:")
    print(f"  Empirical Mean: {emp_mean32:.4f}, MoM Mean: {mom_mean32:.4f}")
    print(f"  Empirical Variance: {emp_var32:.4f}, MoM Variance: {mom_var32:.4f}")

    # Calculate KDE likelihoods (self-consistency check)
    nll_kde12 = calculate_kde_likelihood(kde12, delta_t12)
    nll_kde32 = calculate_kde_likelihood(kde32, delta_t32)
    print(f"\nKDE NLL (self-consistency):")
    print(f"  T1-T2: {nll_kde12:.2f}")
    print(f"  T3-T2: {nll_kde32:.2f}")

    # Plot histograms with both KDE and MoM PDF
    x_min = min(np.min(delta_t12), np.min(delta_t32)) - 10
    x_max = max(np.max(delta_t12), np.max(delta_t32)) + 10
    x_grid = np.linspace(x_min, x_max, 500)
    
    # Get MoM PDFs
    pdf12_mom, pdf32_mom = compute_pdf_parameters(
        mechanism, x_grid, n1, n2, n3, N1, N2, N3, k, mechanism_params
    )
    
    # Get KDE PDFs
    pdf12_kde = evaluate_kde_pdf(kde12, x_grid)
    pdf32_kde = evaluate_kde_pdf(kde32, x_grid)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # T1 - T2 plot
    ax1.hist(delta_t12, bins=30, density=True, alpha=0.3, 
             color='blue', label='Simulation Histogram')
    ax1.plot(x_grid, pdf12_kde, 'b-', linewidth=2, 
             label=f'Simulation KDE (bw={kde_bandwidth})')
    ax1.plot(x_grid, pdf12_mom, 'r--', linewidth=2, 
             label='MoM PDF (Normal Approx)')
    ax1.axvline(emp_mean12, color='blue', linestyle=':', alpha=0.7, 
                label=f'Sim Mean: {emp_mean12:.2f}')
    ax1.axvline(mom_mean12, color='red', linestyle=':', alpha=0.7, 
                label=f'MoM Mean: {mom_mean12:.2f}')
    ax1.set_title(f'T1 - T2 ({mechanism.replace("_", " ").title()})')
    ax1.set_xlabel('Time Difference (min)')
    ax1.set_ylabel('Probability Density')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # T3 - T2 plot
    ax2.hist(delta_t32, bins=30, density=True, alpha=0.3, 
             color='blue', label='Simulation Histogram')
    ax2.plot(x_grid, pdf32_kde, 'b-', linewidth=2, 
             label=f'Simulation KDE (bw={kde_bandwidth})')
    ax2.plot(x_grid, pdf32_mom, 'r--', linewidth=2, 
             label='MoM PDF (Normal Approx)')
    ax2.axvline(emp_mean32, color='blue', linestyle=':', alpha=0.7, 
                label=f'Sim Mean: {emp_mean32:.2f}')
    ax2.axvline(mom_mean32, color='red', linestyle=':', alpha=0.7, 
                label=f'MoM Mean: {mom_mean32:.2f}')
    ax2.set_title(f'T3 - T2 ({mechanism.replace("_", " ").title()})')
    ax2.set_xlabel('Time Difference (min)')
    ax2.set_ylabel('Probability Density')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ========== PARAMETER SPECIFICATION ==========
    # Common parameters for all mechanisms
    N1, N2, N3 = 0.4231530419228542*980, 980, 4.651995192387707*980  # Initial protein counts
    n1, n2, n3 = 0.32130232636872025*5, 5, 2.7917061905867135*5      # Threshold protein counts
    k = 0.014478986712698283                     # Base degradation rate
    #""n2"": 5.0, ""N2"": 980.0, ""k"": 0.014478986712698283, ""r21"": 0.32130232636872025, ""r23"": 2.7917061905867135, ""R21"": 0.4231530419228542, ""R23"": 4.651995192387707, ""burst_size"": 5.0, ""alpha"": 0.19107985964250104, ""beta_k"": 0.4271498852413332, ""beta2_k"": 0.4836025058642737, ""beta3_k"": 0.2324759899149937
    # Mechanism-specific parameters (optional - will use defaults if not specified)
    mechanism_params = {
        'simple': {},  # No additional parameters
        'fixed_burst': {'burst_size': 5},
        'time_varying_k': {'k_1': 0.005, 'k_max': 0.05},
        'feedback_onion': {'n_inner': 20},
        'fixed_burst_feedback_onion': {
            'burst_size': 5,
            'n_inner': 50
        }
    }

    # ========== TEST CONFIGURATION ==========
    # Specify which mechanism(s) to test:
    # Options: 'simple', 'fixed_burst', 'time_varying_k', 'feedback_onion', 'fixed_burst_feedback_onion'
    mechanism = 'fixed_burst'  # Change this to test specific mechanisms
    
    # # KDE bandwidth (tune for best fit to your data)
    # # Typical values: 5-15 for chromosome timing data
    # # Lower = captures more detail, Higher = smoother
    # kde_bandwidth = 1.0

    # ========== RUN TESTS ==========
    test_mom_matching(
        mechanism, n1, n2, n3, N1, N2, N3, k,
        mechanism_params.get(mechanism), 
        max_time=1000, 
        num_sim=1000,
    )
