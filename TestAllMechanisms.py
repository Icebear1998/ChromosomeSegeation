import numpy as np
import matplotlib.pyplot as plt
from MultiMechanismSimulation import MultiMechanismSimulation
from scipy.stats import norm
from MoMCalculations import compute_pdf_mom, compute_moments_mom


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
    elif mechanism == 'feedback_linear':
        mom_mean12, mom_var12 = compute_moments_mom(
            'feedback_linear', n1, N1, n2, N2, k, w1=mechanism_params['w1'], w2=mechanism_params['w2']
        )
        mom_mean32, mom_var32 = compute_moments_mom(
            'feedback_linear', n3, N3, n2, N2, k, w1=mechanism_params['w3'], w2=mechanism_params['w2']
        )
    elif mechanism == 'feedback_onion':
        mom_mean12, mom_var12 = compute_moments_mom(
            'feedback_onion', n1, N1, n2, N2, k, n_inner=mechanism_params['n_inner']
        )
        mom_mean32, mom_var32 = compute_moments_mom(
            'feedback_onion', n3, N3, n2, N2, k, n_inner=mechanism_params['n_inner']
        )
    elif mechanism == 'feedback_zipper':
        mom_mean12, mom_var12 = compute_moments_mom(
            'feedback_zipper', n1, N1, n2, N2, k, z1=mechanism_params['z1'], z2=mechanism_params['z2']
        )
        mom_mean32, mom_var32 = compute_moments_mom(
            'feedback_zipper', n3, N3, n2, N2, k, z1=mechanism_params['z3'], z2=mechanism_params['z2']
        )
    elif mechanism == 'feedback':
        mom_mean12, mom_var12 = compute_moments_mom(
            'feedback', n1, N1, n2, N2, k,
            feedbackSteepness=mechanism_params['feedbackSteepness'],
            feedbackThreshold=mechanism_params['feedbackThreshold']
        )
        mom_mean32, mom_var32 = compute_moments_mom(
            'feedback', n3, N3, n2, N2, k,
            feedbackSteepness=mechanism_params['feedbackSteepness'],
            feedbackThreshold=mechanism_params['feedbackThreshold']
        )
    elif mechanism == 'fixed_burst_feedback_linear':
        mom_mean12, mom_var12 = compute_moments_mom(
            'fixed_burst_feedback_linear', n1, N1, n2, N2, k, 
            burst_size=mechanism_params['burst_size'],
            w1=mechanism_params['w1'], w2=mechanism_params['w2']
        )
        mom_mean32, mom_var32 = compute_moments_mom(
            'fixed_burst_feedback_linear', n3, N3, n2, N2, k,
            burst_size=mechanism_params['burst_size'],
            w1=mechanism_params['w3'], w2=mechanism_params['w2']
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
    elif mechanism == 'feedback_zipper':
        pdf12 = compute_pdf_mom(
            'feedback_zipper', x_grid, n1, N1, n2, N2, k, z1=mechanism_params['z1'], z2=mechanism_params['z2']
        )
        pdf32 = compute_pdf_mom(
            'feedback_zipper', x_grid, n3, N3, n2, N2, k, z1=mechanism_params['z3'], z2=mechanism_params['z2']
        )
    elif mechanism == 'feedback':
        pdf12 = compute_pdf_mom(
            'feedback', x_grid, n1, N1, n2, N2, k,
            feedbackSteepness=mechanism_params['feedbackSteepness'],
            feedbackThreshold=mechanism_params['feedbackThreshold']
        )
        pdf32 = compute_pdf_mom(
            'feedback', x_grid, n3, N3, n2, N2, k,
            feedbackSteepness=mechanism_params['feedbackSteepness'],
            feedbackThreshold=mechanism_params['feedbackThreshold']
        )
    elif mechanism == 'fixed_burst_feedback_linear':
        pdf12 = compute_pdf_mom(
            'fixed_burst_feedback_linear', x_grid, n1, N1, n2, N2, k,
            burst_size=mechanism_params['burst_size'],
            w1=mechanism_params['w1'], w2=mechanism_params['w2']
        )
        pdf32 = compute_pdf_mom(
            'fixed_burst_feedback_linear', x_grid, n3, N3, n2, N2, k,
            burst_size=mechanism_params['burst_size'],
            w1=mechanism_params['w3'], w2=mechanism_params['w2']
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


def test_mom_matching(mechanism, n1, n2, n3, N1, N2, N3, k, mechanism_params=None, max_time=500, num_sim=2000):
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

    # Print comparison
    print("T1 - T2:")
    print(f"Empirical Mean: {emp_mean12:.4f}, MoM Mean: {mom_mean12:.4f}")
    print(
        f"Empirical Variance: {emp_var12:.4f}, MoM Variance: {mom_var12:.4f}")
    print("T3 - T2:")
    print(f"Empirical Mean: {emp_mean32:.4f}, MoM Mean: {mom_mean32:.4f}")
    print(
        f"Empirical Variance: {emp_var32:.4f}, MoM Variance: {mom_var32:.4f}")

    # Plot histograms with MoM PDF
    x_min, x_max = (-240, 240)
    x_grid = np.linspace(x_min, x_max, 401)
    pdf12, pdf32 = compute_pdf_parameters(
        mechanism, x_grid, n1, n2, n3, N1, N2, N3, k, mechanism_params
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(delta_t12, bins=20, density=True,
             alpha=0.4, label='Simulated T1 - T2')
    ax1.plot(x_grid, pdf12, 'r-', label='MoM PDF')
    ax1.set_title(f'T1 - T2 ({mechanism.replace("_", " ").title()})')
    ax1.set_xlabel('Time Difference')
    ax1.legend()

    ax2.hist(delta_t32, bins=20, density=True,
             alpha=0.4, label='Simulated T3 - T2')
    ax2.plot(x_grid, pdf32, 'r-', label='MoM PDF')
    ax2.set_title(f'T3 - T2 ({mechanism.replace("_", " ").title()})')
    ax2.set_xlabel('Time Difference')
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ========== PARAMETER SPECIFICATION ==========
    # Common parameters for all mechanisms
    N1, N2, N3 = 100, 150, 200  # Initial protein counts
    n1, n2, n3 = 3, 5, 8      # Threshold protein counts
    k = 0.02                     # Base degradation rate

    # Mechanism-specific parameters (optional - will use defaults if not specified)
    mechanism_params = {
        'simple': {},  # No additional parameters
        'fixed_burst': {'burst_size': 8},
        'time_varying_k': {'k_1': 0.005},
        'feedback_linear': {'w1': 0.005591, 'w2': 0.004757, 'w3': 0.005733},
        'feedback_onion': {'n_inner': 10},
        'feedback': {'feedbackSteepness': 0.02, 'feedbackThreshold': 120},
        'fixed_burst_feedback_linear': {
            'burst_size': 3,
            'w1': 0.005,
            'w2': 0.01,
            'w3': 0.015
        },
        'feedback_zipper': {'z1': 40, 'z2': 50, 'z3': 60},
        'fixed_burst_feedback_onion': {
            'burst_size': 5,
            'n_inner': 50
        }
    }

    # ========== TEST CONFIGURATION ==========
    # Specify which mechanism(s) to test:
    # Options: 'simple', 'fixed_burst', 'time_varying_k', 'feedback', 'feedback_linear', 'feedback_onion', 'feedback_zipper', 'fixed_burst_feedback_linear', 'fixed_burst_feedback_onion', or 'all'
    mechanism = 'fixed_burst_feedback_onion'  # Change this to test specific mechanisms

    # ========== RUN TESTS ==========
    test_mom_matching(
        mechanism, n1, n2, n3, N1, N2, N3, k,
        mechanism_params.get(mechanism), max_time=800, num_sim=2000
    )
