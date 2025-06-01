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


def get_mechanism_defaults(mechanism):
    """
    Get default mechanism-specific parameters and x_range for plotting.
    
    Args:
        mechanism (str): 'simple', 'fixed_burst', 'time_varying_k', or 'feedback'
    
    Returns:
        dict: Default mechanism-specific parameters and x_range
    """
    if mechanism == 'simple':
        return {
            'mechanism_params': {},
            'x_range': (-50, 50)
        }
    elif mechanism == 'fixed_burst':
        return {
            'mechanism_params': {'burst_size': 8},
            'x_range': (-40, 40)
        }
    elif mechanism == 'time_varying_k':
        return {
            'mechanism_params': {'k_1': 0.001},
            'x_range': (-30, 30)
        }
    elif mechanism == 'feedback':
        return {
            'mechanism_params': {'feedbackSteepness': 0.04, 'feedbackThreshold': 75},
            'x_range': (-200, 200)
        }
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}. Available: 'simple', 'fixed_burst', 'time_varying_k', 'feedback'")


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
    
    if mechanism == 'fixed_burst':
        rate_params.update(mechanism_params)
    elif mechanism == 'time_varying_k':
        rate_params.update(mechanism_params)
    elif mechanism == 'feedback':
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
        mom_mean12, mom_var12 = compute_moments_mom('simple', n1, N1, n2, N2, k)
        mom_mean32, mom_var32 = compute_moments_mom('simple', n3, N3, n2, N2, k)
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
    
    return pdf12, pdf32


def test_mom_matching(mechanism, n1, n2, n3, N1, N2, N3, k, mechanism_params=None, max_time=500, num_sim=2000):
    """
    Test Method of Moments matching for a specified mechanism.
    
    Args:
        mechanism (str): 'simple', 'fixed_burst', 'time_varying_k', or 'feedback'
        n1, n2, n3 (float): Threshold protein counts
        N1, N2, N3 (float): Initial protein counts
        k (float): Base degradation rate
        mechanism_params (dict, optional): Mechanism-specific parameters. If None, uses defaults.
        max_time (float): Maximum simulation time
        num_sim (int): Number of simulations
    """
    print(f"\n=== Testing {mechanism.upper()} Mechanism ===")
    
    # Get mechanism defaults
    defaults = get_mechanism_defaults(mechanism)
    
    # Use provided mechanism_params or defaults
    if mechanism_params is None:
        mechanism_params = defaults['mechanism_params']
    else:
        # Merge with defaults to ensure all required parameters are present
        final_params = defaults['mechanism_params'].copy()
        final_params.update(mechanism_params)
        mechanism_params = final_params
    
    # Build rate_params for simulation
    rate_params = build_rate_params(mechanism, k, mechanism_params)
    
    print(f"Parameters: N1={N1}, N2={N2}, N3={N3}, n1={n1}, n2={n2}, n3={n3}, k={k}")
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
    print(f"Empirical Variance: {emp_var12:.4f}, MoM Variance: {mom_var12:.4f}")
    print("T3 - T2:")
    print(f"Empirical Mean: {emp_mean32:.4f}, MoM Mean: {mom_mean32:.4f}")
    print(f"Empirical Variance: {emp_var32:.4f}, MoM Variance: {mom_var32:.4f}")
    
    # Plot histograms with MoM PDF
    x_min, x_max = defaults['x_range']
    x_grid = np.linspace(x_min, x_max, 401)
    pdf12, pdf32 = compute_pdf_parameters(
        mechanism, x_grid, n1, n2, n3, N1, N2, N3, k, mechanism_params
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(delta_t12, bins=20, density=True, alpha=0.4, label='Simulated T1 - T2')
    ax1.plot(x_grid, pdf12, 'r-', label='MoM PDF')
    ax1.set_title(f'T1 - T2 ({mechanism.replace("_", " ").title()})')
    ax1.set_xlabel('Time Difference')
    ax1.legend()
    
    ax2.hist(delta_t32, bins=20, density=True, alpha=0.4, label='Simulated T3 - T2')
    ax2.plot(x_grid, pdf32, 'r-', label='MoM PDF')
    ax2.set_title(f'T3 - T2 ({mechanism.replace("_", " ").title()})')
    ax2.set_xlabel('Time Difference')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ========== PARAMETER SPECIFICATION ==========
    # Common parameters for all mechanisms
    N1, N2, N3 = 150, 200, 250  # Initial protein counts
    n1, n2, n3 = 5, 10, 12      # Threshold protein counts
    k = 0.05                     # Base degradation rate
    
    # Mechanism-specific parameters (optional - will use defaults if not specified)
    mechanism_params = {
        'simple': {},  # No additional parameters
        'fixed_burst': {'burst_size': 8},
        'time_varying_k': {'k_1': 0.001},
        'feedback': {'feedbackSteepness': 0.04, 'feedbackThreshold': 75}
    }
    
    # ========== TEST CONFIGURATION ==========
    # Specify which mechanism(s) to test:
    # Options: 'simple', 'fixed_burst', 'time_varying_k', 'feedback', or 'all'
    mechanism = 'feedback'  # Change this to test specific mechanisms
    
    # ========== RUN TESTS ==========
    test_mom_matching(
                mechanism, n1, n2, n3, N1, N2, N3, k, 
                mechanism_params.get(mechanism), max_time=500, num_sim=1000
            ) 