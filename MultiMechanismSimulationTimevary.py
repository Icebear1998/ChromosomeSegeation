import numpy as np
import warnings


class MultiMechanismSimulationTimevary:
    """
    Simplified simulation class for time-varying degradation mechanisms.
    All mechanisms share the same simulation core and differ only in:
    1. How they calculate burst sizes
    2. How they calculate propensities (with/without feedback)
    
    All mechanisms use time-varying k(t) = min(k_1 * t, k_max).
    """

    def __init__(self, mechanism, initial_state_list, rate_params, n0_list, max_time):
        """
        Initialize the simulation.

        Args:
            mechanism (str): 'time_varying_k', 'time_varying_k_fixed_burst', 'time_varying_k_feedback_onion', 'time_varying_k_burst_onion', 'time_varying_k_combined'
            initial_state_list (list): Initial cohesin counts [N1, N2, N3].
            rate_params (dict): 
                For 'time_varying_k': {'k_1': k_1, 'k_max': k_max} (optional k_max for maximum rate).
                For 'time_varying_k_fixed_burst': {'k_1': k_1, 'k_max': k_max, 'burst_size': b}.
                For 'time_varying_k_feedback_onion': {'k_1': k_1, 'k_max': k_max, 'n_inner': n_inner}.
                For 'time_varying_k_burst_onion': {'k_1': k_1, 'k_max': k_max, 'burst_size': b}.
                For 'time_varying_k_combined': {'k_1': k_1, 'k_max': k_max, 'burst_size': b, 'n_inner': n_inner}.
            n0_list (list): Threshold counts [n01, n02, n03].
            max_time (float): Maximum expected simulation time.
        """
        self.mechanism = mechanism
        self.initial_state_list = list(initial_state_list)
        self.rate_params = rate_params.copy()
        self.n0_list = list(n0_list)
        self.max_time = max_time
        
        # Get mechanism-specific functions
        self.calculate_burst_size = self._get_burst_calculator()
        self.calculate_propensities = self._get_propensity_calculator()
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Check that all required parameters are present."""
        valid_mechanisms = ['time_varying_k', 'time_varying_k_fixed_burst', 'time_varying_k_feedback_onion', 'time_varying_k_burst_onion', 'time_varying_k_combined']
        if self.mechanism not in valid_mechanisms:
            raise ValueError(f"Mechanism must be one of: {valid_mechanisms}")
        
        # Check required parameters
        if 'k_1' not in self.rate_params:
            raise ValueError("All time-varying mechanisms require 'k_1' in rate_params.")
        
        if self.mechanism == 'time_varying_k_fixed_burst' and 'burst_size' not in self.rate_params:
            raise ValueError("time_varying_k_fixed_burst mechanism requires 'burst_size' in rate_params.")
        
        if self.mechanism == 'time_varying_k_feedback_onion' and 'n_inner' not in self.rate_params:
            raise ValueError("time_varying_k_feedback_onion mechanism requires 'n_inner' in rate_params.")
        
        if self.mechanism == 'time_varying_k_burst_onion' and 'burst_size' not in self.rate_params:
            raise ValueError("time_varying_k_burst_onion mechanism requires 'burst_size' in rate_params.")
        
        if self.mechanism == 'time_varying_k_combined':
            if 'burst_size' not in self.rate_params:
                raise ValueError("time_varying_k_combined mechanism requires 'burst_size' in rate_params.")
            if 'n_inner' not in self.rate_params:
                raise ValueError("time_varying_k_combined mechanism requires 'n_inner' in rate_params.")
    
    def _get_burst_calculator(self):
        """Return the burst size calculation function for the current mechanism."""
        
        if self.mechanism == 'time_varying_k':
            def single_cohesin():
                return 1.0
            return single_cohesin
        
        elif self.mechanism in ['time_varying_k_fixed_burst', 'time_varying_k_burst_onion', 'time_varying_k_combined']:
            def fixed_burst():
                return self.rate_params['burst_size']
            return fixed_burst
        
        elif self.mechanism == 'time_varying_k_feedback_onion':
            def single_cohesin_feedback():
                return 1.0
            return single_cohesin_feedback
        
        else:
            raise ValueError(f"Unknown mechanism: {self.mechanism}")
    
    def _get_propensity_calculator(self):
        """Return the propensity calculation function for the current mechanism."""
        
        if self.mechanism in ['time_varying_k', 'time_varying_k_fixed_burst', 'time_varying_k_burst_onion']:
            # Standard propensities: k(t) * state[i]
            def standard_propensities(k_t, states):
                return [k_t * max(state, 0) for state in states]
            return standard_propensities
        
        elif self.mechanism in ['time_varying_k_feedback_onion', 'time_varying_k_combined']:
            # Onion feedback propensities: k(t) * W(N_i) * state[i]
            def feedback_onion_propensities(k_t, states):
                propensities = []
                for i, state in enumerate(states):
                    N_i = self.initial_state_list[i]
                    n_inner = self.rate_params['n_inner']
                    
                    # Calculate feedback weight
                    if N_i > n_inner:
                        W_m = (N_i / n_inner) ** (-1/3)
                    else:
                        W_m = 1.0
                    
                    propensity = k_t * W_m * max(state, 0)
                    propensities.append(propensity)
                return propensities
            return feedback_onion_propensities
        
        else:
            raise ValueError(f"Unknown mechanism: {self.mechanism}")
    
    def _calculate_effective_total_state(self, states):
        """Calculate the effective total state for time calculations."""
        if self.mechanism in ['time_varying_k_feedback_onion', 'time_varying_k_combined']:
            # Use weighted sum for feedback mechanisms
            total_effective_state = 0
            for i, state in enumerate(states):
                N_i = self.initial_state_list[i]
                n_inner = self.rate_params['n_inner']
                if N_i > n_inner:
                    W_i = (N_i / n_inner) ** (-1/3)
                else:
                    W_i = 1.0
                total_effective_state += W_i * state
            return total_effective_state
        else:
            # Use simple sum for standard mechanisms
            return sum(states)


    def simulate(self):
        """
        Run the simulation using the shared core algorithm.
        All mechanisms use the same simulation loop with mechanism-specific functions.
        
        Returns:
            tuple: (times, states, separate_times)
        """
        # Initialize simulation state
        current_state = [int(s) for s in self.initial_state_list]
        current_time = 0.0
        times = [0.0]
        states = [current_state.copy()]
        separate_times = [None, None, None]
        
        # Main simulation loop - SAME FOR ALL TIME-VARYING MECHANISMS
        while True:
            # Calculate current time-varying rate
            k_t = self._get_time_varying_rate_at_time(current_time)
            
            # Calculate propensities (mechanism-specific)
            propensities = self.calculate_propensities(k_t, current_state)
            total_propensity = sum(propensities)
            
            # Check if simulation should stop
            if total_propensity <= 0 or all(t is not None for t in separate_times):
                break
            
            # Calculate effective total state for time calculations
            total_effective_state = self._calculate_effective_total_state(current_state)
            if total_effective_state == 0:
                break
            
            # Calculate time to next event using time-varying rate
            tau = self._calculate_next_event_time_at_time(current_time, total_effective_state)
            if tau <= 0:
                current_time += 1e-10  # Small increment to avoid infinite loops
                continue
            
            current_time += tau
            
            # Recalculate propensities at new time for reaction selection
            k_t = self._get_time_varying_rate_at_time(current_time)
            propensities = self.calculate_propensities(k_t, current_state)
            total_propensity = sum(propensities)
            
            # Select which chromosome experiences degradation
            r = np.random.uniform(0, total_propensity)
            cumulative = 0.0
            selected_chromosome = 0
            
            for i, propensity in enumerate(propensities):
                cumulative += propensity
                if r < cumulative:
                    selected_chromosome = i
                    break
            
            # Apply degradation (mechanism-specific burst size)
            burst_size = self.calculate_burst_size()
            current_state[selected_chromosome] = max(0, current_state[selected_chromosome] - burst_size)
            
            # Record this event
            times.append(current_time)
            states.append(current_state.copy())
            
            # Check for chromosome separation
            for i in range(3):
                if separate_times[i] is None and current_state[i] <= round(self.n0_list[i]):
                    separate_times[i] = current_time
        
        # Finalize simulation
        for i in range(3):
            if separate_times[i] is None:
                separate_times[i] = current_time
        
        if current_time > self.max_time:
            warnings.warn(f"Simulation time ({current_time:.2f}) exceeded max_time ({self.max_time:.2f})")
        
        return times, states, separate_times
    
    def _get_time_varying_rate_at_time(self, time):
        """Calculate the time-varying rate k(t) = min(k_1 * t, k_max) at a specific time."""
        k_1 = self.rate_params['k_1']
        k_max = self.rate_params.get('k_max', float('inf'))
        
        k_linear = k_1 * time
        k_t = min(k_linear, k_max)
        k_t = max(k_t, 1e-10)  # Prevent zero propensity at t=0
        
        return k_t
    
    def _calculate_next_event_time_at_time(self, current_time, total_effective_state):
        """Calculate the time to next event using inhomogeneous Poisson process."""
        k_1 = self.rate_params['k_1']
        k_max = self.rate_params.get('k_max', float('inf'))
        
        # Handle edge case at t=0 where rate is zero
        if current_time == 0:
            return 1e-10  # Small time increment
        
        # Determine if we're in linear phase or constant phase
        t_max = k_max / k_1 if k_max != float('inf') else float('inf')
        
        if current_time >= t_max:
            # Constant phase: k(t) = k_max, use standard Gillespie
            k_t = self._get_time_varying_rate_at_time(current_time)
            total_propensity = k_t * total_effective_state
            tau = np.random.exponential(1 / total_propensity)
            return tau
        else:
            # Linear phase: k(t) = k_1 * t, use inhomogeneous Poisson process
            r_1 = np.random.uniform(0, 1)
            
            # Check if the next event would occur after t_max
            if k_max != float('inf'):
                # Calculate tau assuming linear phase continues
                a = 1
                b = 2 * current_time
                c = 2 * np.log(r_1) / (k_1 * total_effective_state)
                
                discriminant = b**2 - 4 * a * c
                if discriminant < 0:
                    return 1e-10  # Small time increment
                
                tau_linear = (-b + np.sqrt(discriminant)) / (2 * a)
                
                if tau_linear <= 0:
                    return 1e-10  # Small time increment
                
                # Check if this tau would take us beyond t_max
                if current_time + tau_linear > t_max:
                    # Jump to t_max and continue with constant rate
                    tau = t_max - current_time
                    return tau
                else:
                    # Stay in linear phase
                    return tau_linear
            else:
                # No maximum rate, pure linear phase
                a = 1
                b = 2 * current_time
                c = 2 * np.log(r_1) / (k_1 * total_effective_state)
                
                discriminant = b**2 - 4 * a * c
                if discriminant < 0:
                    return 1e-10  # Small time increment
                
                tau = (-b + np.sqrt(discriminant)) / (2 * a)
                
                if tau <= 0:
                    return 1e-10  # Small time increment
                
                return tau
    
    @staticmethod
    def get_available_mechanisms():
        """Get list of all available time-varying mechanisms."""
        return [
            'time_varying_k', 
            'time_varying_k_fixed_burst', 
            'time_varying_k_feedback_onion', 
            'time_varying_k_burst_onion', 
            'time_varying_k_combined'
        ]
    
    @staticmethod
    def get_mechanism_info(mechanism):
        """Get information about a specific mechanism."""
        info = {
            'time_varying_k': {
                'name': 'Time-Varying k',
                'description': 'Single cohesin degradation with k(t) = min(k_1 * t, k_max)',
                'parameters': 'k_1 (initial rate), k_max (maximum rate, optional)'
            },
            'time_varying_k_fixed_burst': {
                'name': 'Time-Varying k + Fixed Burst',
                'description': 'Fixed burst sizes with time-varying rate',
                'parameters': 'k_1 (initial rate), k_max (max rate), burst_size (burst size)'
            },
            'time_varying_k_feedback_onion': {
                'name': 'Time-Varying k + Onion Feedback',
                'description': 'Single cohesin with time-varying rate and onion feedback',
                'parameters': 'k_1 (initial rate), k_max (max rate), n_inner (inner threshold)'
            },
            'time_varying_k_burst_onion': {
                'name': 'Time-Varying k + Burst (No Feedback)',
                'description': 'Fixed bursts with time-varying rate (no feedback despite name)',
                'parameters': 'k_1 (initial rate), k_max (max rate), burst_size (burst size)'
            },
            'time_varying_k_combined': {
                'name': 'Time-Varying k + Burst + Onion Feedback',
                'description': 'Combined: time-varying rate, fixed bursts, and onion feedback',
                'parameters': 'k_1 (initial rate), k_max (max rate), burst_size (burst size), n_inner (inner threshold)'
            }
        }
        
        return info.get(mechanism, {
            'name': 'Unknown', 
            'description': 'Unknown mechanism', 
            'parameters': 'Unknown'
        })