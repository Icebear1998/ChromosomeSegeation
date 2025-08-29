import numpy as np
import warnings


class MultiMechanismSimulationTimevary:
    """
    A simulation class for modeling cohesin degradation during chromosome segregation
    with time-varying degradation rates integrated with other mechanisms.
    All mechanisms (except simple) use time-varying k(t) = min(k_1 * t, k_max).
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
        self.initial_state_list = initial_state_list.copy()
        self.rate_params = rate_params.copy()
        self.n0_list = n0_list.copy()
        self.max_time = max_time
        self.time = 0
        self.times = [0]
        self.states = [initial_state_list.copy()]
        self.separate_times = [None, None, None]

        # Validate mechanism and parameters
        valid_mechanisms = ['time_varying_k', 'time_varying_k_fixed_burst', 'time_varying_k_feedback_onion', 'time_varying_k_burst_onion', 'time_varying_k_combined']
        if self.mechanism not in valid_mechanisms:
            raise ValueError(f"Mechanism must be one of: {valid_mechanisms}")
        
        # Check required parameters
        if 'k_1' not in rate_params:
            raise ValueError("All time-varying mechanisms require 'k_1' in rate_params.")
        
        if self.mechanism == 'time_varying_k_fixed_burst' and 'burst_size' not in rate_params:
            raise ValueError("time_varying_k_fixed_burst mechanism requires 'burst_size' in rate_params.")
        
        if self.mechanism == 'time_varying_k_feedback_onion' and 'n_inner' not in rate_params:
            raise ValueError("time_varying_k_feedback_onion mechanism requires 'n_inner' in rate_params.")
        
        if self.mechanism == 'time_varying_k_burst_onion' and 'burst_size' not in rate_params:
            raise ValueError("time_varying_k_burst_onion mechanism requires 'burst_size' in rate_params.")
        
        if self.mechanism == 'time_varying_k_combined':
            if 'burst_size' not in rate_params:
                raise ValueError("time_varying_k_combined mechanism requires 'burst_size' in rate_params.")
            if 'n_inner' not in rate_params:
                raise ValueError("time_varying_k_combined mechanism requires 'n_inner' in rate_params.")

    def _get_time_varying_rate(self):
        """
        Calculate the current time-varying rate k(t) = min(k_1 * t, k_max).
        
        Returns:
            float: Current degradation rate
        """
        k_1 = self.rate_params['k_1']
        k_max = self.rate_params.get('k_max', float('inf'))  # Default: no maximum
        
        k_linear = k_1 * self.time
        k_t = min(k_linear, k_max)
        k_t = max(k_t, 1e-10)  # Prevent zero propensity at t=0
        
        return k_t

    def _calculate_next_event_time(self, total_state):
        """
        Calculate the time to the next event using inhomogeneous Poisson process
        for time-varying rate k(t) = min(k_1 * t, k_max).
        
        Args:
            total_state (int): Sum of cohesins across all chromosomes
            
        Returns:
            float: Time increment to next event
        """
        k_1 = self.rate_params['k_1']
        k_max = self.rate_params.get('k_max', float('inf'))
        
        # Handle edge case at t=0 where rate is zero
        if self.time == 0:
            self.time = 1e-10
            return 0  # Continue to next iteration
        
        # Determine if we're in linear phase or constant phase
        t_max = k_max / k_1 if k_max != float('inf') else float('inf')  # Time when k(t) reaches k_max
        
        if self.time >= t_max:
            # Constant phase: k(t) = k_max, use standard Gillespie
            k_t = self._get_time_varying_rate()
            total_propensity = k_t * total_state
            tau = np.random.exponential(1 / total_propensity)
            return tau
        else:
            # Linear phase: k(t) = k_1 * t, use inhomogeneous Poisson process
            r_1 = np.random.uniform(0, 1)
            
            # Check if the next event would occur after t_max
            if k_max != float('inf'):
                # Calculate tau assuming linear phase continues
                a = 1
                b = 2 * self.time
                c = 2 * np.log(r_1) / (k_1 * total_state)
                
                discriminant = b**2 - 4 * a * c
                if discriminant < 0:
                    return 1e-10  # Small time increment
                
                tau_linear = (-b + np.sqrt(discriminant)) / (2 * a)
                
                if tau_linear <= 0:
                    return 1e-10  # Small time increment
                
                # Check if this tau would take us beyond t_max
                if self.time + tau_linear > t_max:
                    # Jump to t_max and continue with constant rate
                    tau = t_max - self.time
                    return tau
                else:
                    # Stay in linear phase
                    return tau_linear
            else:
                # No maximum rate, pure linear phase
                a = 1
                b = 2 * self.time
                c = 2 * np.log(r_1) / (k_1 * total_state)
                
                discriminant = b**2 - 4 * a * c
                if discriminant < 0:
                    return 1e-10  # Small time increment
                
                tau = (-b + np.sqrt(discriminant)) / (2 * a)
                
                if tau <= 0:
                    return 1e-10  # Small time increment
                
                return tau

    def _simulate_time_varying_k(self):
        """
        Simulate the pure time-varying degradation model with k(t) = min(k_1 * t, k_max).
        Each cohesin degrades at rate k(t) * state[i], reducing state[i] by 1.
        """
        while True:
            # Calculate current rate
            k_t = self._get_time_varying_rate()
            
            # Calculate propensities: rate of degradation for each chromosome
            propensities = [k_t * self.state[i] for i in range(3)]
            total_propensity = sum(propensities)

            # Stop if no further degradation possible or all thresholds reached
            if total_propensity <= 0 or all(t is not None for t in self.separate_times):
                break

            # Calculate total state (sum of cohesins across all chromosomes)
            total_state = sum(self.state)
            if total_state == 0:
                break  # No molecules left to degrade

            # Calculate time to next event
            tau = self._calculate_next_event_time(total_state)
            if tau <= 0:
                continue  # Skip invalid time increments
            
            self.time += tau

            # Recompute propensities at new time for reaction selection
            k_t = self._get_time_varying_rate()
            propensities = [k_t * self.state[i] for i in range(3)]
            total_propensity = sum(propensities)

            # Choose which chromosome's cohesin degrades
            r = np.random.uniform(0, total_propensity)
            cumulative_propensity = 0
            for i in range(3):
                cumulative_propensity += propensities[i]
                if r < cumulative_propensity:
                    self.state[i] -= 1
                    break

            # Record time and state
            self.times.append(self.time)
            self.states.append(self.state.copy())

            # Check for separation times
            for i in range(3):
                if self.separate_times[i] is None and self.state[i] <= round(self.n0_list[i]):
                    self.separate_times[i] = self.time

    def _simulate_time_varying_k_fixed_burst(self):
        """
        Simulate time-varying k with fixed burst sizes.
        Cohesins degrade in bursts of size b at time-varying rates k(t) * state[i].
        """
        burst_size = self.rate_params['burst_size']
        
        while True:
            # Calculate current rate
            k_t = self._get_time_varying_rate()
            
            # Calculate propensities: rate of bursts for each chromosome
            propensities = [k_t * self.state[i] for i in range(3)]
            total_propensity = sum(propensities)

            # Stop if no further bursts possible or all thresholds reached
            if total_propensity <= 0 or all(t is not None for t in self.separate_times):
                break

            # Calculate total state (sum of cohesins across all chromosomes)
            total_state = sum(self.state)
            if total_state == 0:
                break  # No molecules left to degrade

            # Calculate time to next event
            tau = self._calculate_next_event_time(total_state)
            if tau <= 0:
                continue  # Skip invalid time increments
            
            self.time += tau

            # Recompute propensities at new time for reaction selection
            k_t = self._get_time_varying_rate()
            propensities = [k_t * self.state[i] for i in range(3)]
            total_propensity = sum(propensities)

            # Choose which chromosome experiences a burst
            r = np.random.uniform(0, total_propensity)
            cumulative_propensity = 0
            for i in range(3):
                cumulative_propensity += propensities[i]
                if r < cumulative_propensity:
                    # Remove burst_size cohesins, but not below 0
                    self.state[i] = max(0, self.state[i] - burst_size)
                    break

            # Record time and state
            self.times.append(self.time)
            self.states.append(self.state.copy())

            # Check for separation times
            for i in range(3):
                if self.separate_times[i] is None and self.state[i] <= round(self.n0_list[i]):
                    self.separate_times[i] = self.time

    def _simulate_time_varying_k_feedback_onion(self):
        """
        Simulate time-varying k with feedback onion mechanism.
        Each cohesin degrades at rate k(t) * W(N_i) * state[i], where
        W(N_i) = (N_i/n_inner)^(-1/3) for N_i > n_inner, else W(N_i) = 1.
        """
        while True:
            # Calculate current rate
            k_t = self._get_time_varying_rate()
            
            # Calculate propensities with onion feedback
            propensities = []
            for i in range(3):
                m = self.state[i]
                N_i = self.initial_state_list[i]  # Initial cohesin count for chromosome i
                n_inner = self.rate_params['n_inner']
                
                # Calculate W_m based on onion feedback mechanism
                if N_i > n_inner:
                    W_m = (N_i / n_inner) ** (-1/3)
                else:
                    W_m = 1.0
                
                propensity = k_t * W_m * m
                propensities.append(propensity)
            
            total_propensity = sum(propensities)

            # Stop if no further degradation possible or all thresholds reached
            if total_propensity <= 0 or all(t is not None for t in self.separate_times):
                break

            # Calculate total effective state for time calculation
            # Use sum of (W_i * state[i]) as the effective total state
            total_effective_state = 0
            for i in range(3):
                N_i = self.initial_state_list[i]
                n_inner = self.rate_params['n_inner']
                if N_i > n_inner:
                    W_i = (N_i / n_inner) ** (-1/3)
                else:
                    W_i = 1.0
                total_effective_state += W_i * self.state[i]
            
            if total_effective_state == 0:
                break  # No effective molecules left to degrade

            # Calculate time to next event
            tau = self._calculate_next_event_time(total_effective_state)
            if tau <= 0:
                continue  # Skip invalid time increments
            
            self.time += tau

            # Recompute propensities at new time for reaction selection
            k_t = self._get_time_varying_rate()
            propensities = []
            for i in range(3):
                m = self.state[i]
                N_i = self.initial_state_list[i]
                n_inner = self.rate_params['n_inner']
                
                if N_i > n_inner:
                    W_m = (N_i / n_inner) ** (-1/3)
                else:
                    W_m = 1.0
                
                propensity = k_t * W_m * m
                propensities.append(propensity)
            
            total_propensity = sum(propensities)

            # Choose which chromosome's cohesin degrades
            r = np.random.uniform(0, total_propensity)
            cumulative_propensity = 0
            for i in range(3):
                cumulative_propensity += propensities[i]
                if r < cumulative_propensity:
                    self.state[i] -= 1
                    break

            # Record time and state
            self.times.append(self.time)
            self.states.append(self.state.copy())

            # Check for separation times
            for i in range(3):
                if self.separate_times[i] is None and self.state[i] <= round(self.n0_list[i]):
                    self.separate_times[i] = self.time

    def _simulate_time_varying_k_burst_onion(self):
        """
        Simulate time-varying k with fixed burst sizes (NO onion feedback).
        
        This mechanism combines:
        1. Time-varying degradation rate: k(t) = min(k_1 * t, k_max)
        2. Fixed burst sizes: cohesins degrade in bursts of size b
        
        Each chromosome experiences bursts of size b at rate k(t) * state[i].
        Note: Despite the name, this does NOT include onion feedback.
        """
        burst_size = self.rate_params['burst_size']
        
        while True:
            # Calculate current time-varying rate
            k_t = self._get_time_varying_rate()
            
            # Calculate propensities for burst events (NO onion feedback)
            propensities = [k_t * self.state[i] for i in range(3)]
            total_propensity = sum(propensities)

            # Stop if no further bursts possible or all thresholds reached
            if total_propensity <= 0 or all(t is not None for t in self.separate_times):
                break

            # Calculate total state (sum of cohesins across all chromosomes)
            total_state = sum(self.state)
            if total_state == 0:
                break  # No molecules left to degrade

            # Calculate time to next event
            tau = self._calculate_next_event_time(total_state)
            if tau <= 0:
                continue  # Skip invalid time increments
            
            self.time += tau

            # Recompute propensities at new time for reaction selection
            k_t = self._get_time_varying_rate()
            propensities = [k_t * self.state[i] for i in range(3)]
            total_propensity = sum(propensities)

            # Choose which chromosome experiences a burst
            r = np.random.uniform(0, total_propensity)
            cumulative_propensity = 0
            for i in range(3):
                cumulative_propensity += propensities[i]
                if r < cumulative_propensity:
                    # Remove burst_size cohesins, but not below 0
                    self.state[i] = max(0, self.state[i] - burst_size)
                    break

            # Record time and state
            self.times.append(self.time)
            self.states.append(self.state.copy())

            # Check for separation times
            for i in range(3):
                if self.separate_times[i] is None and self.state[i] <= round(self.n0_list[i]):
                    self.separate_times[i] = self.time

    def _simulate_time_varying_k_combined(self):
        """
        Simulate the combined mechanism: time-varying k + fixed burst + feedback onion.
        
        This mechanism combines all three features:
        1. Time-varying degradation rate: k(t) = min(k_1 * t, k_max)
        2. Fixed burst sizes: cohesins degrade in bursts of size b
        3. Feedback onion mechanism: rate modified by W(N_i) = (N_i/n_inner)^(-1/3) for N_i > n_inner
        
        Each chromosome experiences bursts of size b at rate k(t) * W(N_i) * state[i].
        """
        burst_size = self.rate_params['burst_size']
        
        while True:
            # Calculate current time-varying rate
            k_t = self._get_time_varying_rate()
            
            # Calculate propensities with onion feedback for burst events
            propensities = []
            for i in range(3):
                m = self.state[i]
                N_i = self.initial_state_list[i]  # Initial cohesin count for chromosome i
                n_inner = self.rate_params['n_inner']
                
                # Calculate W_m based on onion feedback mechanism
                if N_i > n_inner:
                    W_m = (N_i / n_inner) ** (-1/3)
                else:
                    W_m = 1.0
                
                # Propensity for burst events: k(t) * W(N_i) * current_state
                propensity = k_t * W_m * m
                propensities.append(propensity)
            
            total_propensity = sum(propensities)

            # Stop if no further bursts possible or all thresholds reached
            if total_propensity <= 0 or all(t is not None for t in self.separate_times):
                break

            # Calculate total effective state for time calculation
            # Use sum of (W_i * state[i]) as the effective total state
            total_effective_state = 0
            for i in range(3):
                N_i = self.initial_state_list[i]
                n_inner = self.rate_params['n_inner']
                if N_i > n_inner:
                    W_i = (N_i / n_inner) ** (-1/3)
                else:
                    W_i = 1.0
                total_effective_state += W_i * self.state[i]
            
            if total_effective_state == 0:
                break  # No effective molecules left to degrade

            # Calculate time to next event using effective state
            tau = self._calculate_next_event_time(total_effective_state)
            if tau <= 0:
                continue  # Skip invalid time increments
            
            self.time += tau

            # Recompute propensities at new time for reaction selection
            k_t = self._get_time_varying_rate()
            propensities = []
            for i in range(3):
                m = self.state[i]
                N_i = self.initial_state_list[i]
                n_inner = self.rate_params['n_inner']
                
                # Calculate W_m based on onion feedback mechanism
                if N_i > n_inner:
                    W_m = (N_i / n_inner) ** (-1/3)
                else:
                    W_m = 1.0
                
                # Propensity for burst events: k(t) * W(N_i) * current_state
                propensity = k_t * W_m * m
                propensities.append(propensity)
            
            total_propensity = sum(propensities)

            # Choose which chromosome experiences a burst
            r = np.random.uniform(0, total_propensity)
            cumulative_propensity = 0
            for i in range(3):
                cumulative_propensity += propensities[i]
                if r < cumulative_propensity:
                    # Remove burst_size cohesins, but not below 0
                    self.state[i] = max(0, self.state[i] - burst_size)
                    break

            # Record time and state
            self.times.append(self.time)
            self.states.append(self.state.copy())

            # Check for separation times
            for i in range(3):
                if self.separate_times[i] is None and self.state[i] <= round(self.n0_list[i]):
                    self.separate_times[i] = self.time

    def simulate(self):
        """
        Run the simulation based on the specified mechanism.
        Issues a warning if final simulation time exceeds max_time.

        Returns:
            tuple: (times, states, separate_times)
                - times: List of event times.
                - states: List of cohesin states at each event.
                - separate_times: List of separation times [T1, T2, T3].
        """
        # Reset state (ensure integer states)
        self.state = [int(state) for state in self.initial_state_list]
        self.time = 0
        self.times = [0]
        self.states = [self.state.copy()]
        self.separate_times = [None, None, None]

        # Run appropriate simulation
        if self.mechanism == 'time_varying_k':
            self._simulate_time_varying_k()
        elif self.mechanism == 'time_varying_k_fixed_burst':
            self._simulate_time_varying_k_fixed_burst()
        elif self.mechanism == 'time_varying_k_feedback_onion':
            self._simulate_time_varying_k_feedback_onion()
        elif self.mechanism == 'time_varying_k_burst_onion':
            self._simulate_time_varying_k_burst_onion()
        elif self.mechanism == 'time_varying_k_combined':
            self._simulate_time_varying_k_combined()

        # Issue warning if simulation time exceeds max_time
        if self.time > self.max_time:
            warnings.warn(
                f"Simulation time ({self.time:.2f}) exceeded max_time ({self.max_time:.2f}). "
                f"Parameters: mechanism={self.mechanism}, "
                f"initial_state={self.initial_state_list}, "
                f"rate_params={self.rate_params}, "
                f"n0_list={self.n0_list}"
            )

        # Set unset separation times to final simulation time
        for i in range(3):
            if self.separate_times[i] is None:
                self.separate_times[i] = self.time

        return self.times, self.states, self.separate_times 