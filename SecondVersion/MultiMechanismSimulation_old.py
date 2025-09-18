import numpy as np
import warnings


class MultiMechanismSimulation:
    """
    A simulation class for modeling cohesin degradation during chromosome segregation
    with multiple mechanisms. Supports Simple (gradual degradation), Fixed Burst Sizes
    (bursty degradation with fixed burst size), Random Normal Burst (variable burst sizes),
    Geometric Burst (burst sizes from geometric distribution), Feedback Onion (onion-like 
    feedback), and Fixed Burst Feedback Onion (combined burst and onion feedback) models. 
    Issues a warning if simulation time exceeds max_time.
    """

    def __init__(self, mechanism, initial_state_list, rate_params, n0_list, max_time):
        """
        Initialize the simulation.

        Args:
            mechanism (str): 'simple', 'fixed_burst', 'random_normal_burst', 'geometric_burst', 'feedback_onion', or 'fixed_burst_feedback_onion'.
            initial_state_list (list): Initial cohesin counts [N1, N2, N3].
            rate_params (dict): For 'simple': {'k': k}.
                               For 'fixed_burst': {'k': k, 'burst_size': b}.
                               For 'random_normal_burst': {'k': k, 'burst_size': mean_burst_size, 'var_burst_size': var_burst_size}.
                               For 'geometric_burst': {'k': k, 'burst_size': mean_burst_size}.
                               For 'feedback_onion': {'k': k, 'n_inner': n_inner}.
                               For 'fixed_burst_feedback_onion': {'k': k, 'burst_size': b, 'n_inner': n_inner}.
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
        if self.mechanism not in ['simple', 'fixed_burst', 'random_normal_burst', 'geometric_burst', 'feedback_onion', 'fixed_burst_feedback_onion']:
            raise ValueError(
                "Mechanism must be 'simple', 'fixed_burst', 'random_normal_burst', 'geometric_burst', 'feedback_onion', or 'fixed_burst_feedback_onion'.")
        if self.mechanism == 'fixed_burst' and 'burst_size' not in rate_params:
            raise ValueError(
                "Fixed burst mechanism requires 'lambda_list' and 'burst_size' in rate_params.")
        if self.mechanism == 'geometric_burst':
            required_params = ['k', 'burst_size']
            missing_params = [param for param in required_params if param not in rate_params]
            if missing_params:
                raise ValueError(
                    f"Geometric burst mechanism requires {required_params} in rate_params. Missing: {missing_params}")
        if self.mechanism == 'feedback_onion':
            required_params = ['k', 'n_inner']
            missing_params = [param for param in required_params if param not in rate_params]
            if missing_params:
                raise ValueError(
                    f"Feedback onion mechanism requires {required_params} in rate_params. Missing: {missing_params}")
        if self.mechanism == 'fixed_burst_feedback_onion':
            required_params = ['k', 'burst_size', 'n_inner']
            missing_params = [param for param in required_params if param not in rate_params]
            if missing_params:
                raise ValueError(
                    f"Fixed burst feedback onion mechanism requires {required_params} in rate_params. Missing: {missing_params}")

    def _simulate_simple(self):
        """
        Simulate the Simple Model: gradual, independent cohesin degradation.
        Each cohesin degrades at rate k_i * state[i], reducing state[i] by 1.
        Continues until all chromosomes reach threshold or propensities are zero.
        """
        while True:
            # Calculate propensities: rate of degradation for each chromosome
            propensities = [self.rate_params['k'] * self.state[i]
                            for i in range(3)]
            total_propensity = sum(propensities)

            # Stop if no further degradation possible or all thresholds reached
            if total_propensity <= 0 or all(t is not None for t in self.separate_times):
                break

            # Time to next degradation event
            tau = np.random.exponential(1 / total_propensity)
            self.time += tau

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



    def _simulate_fixed_burst(self):
        """
        Simulate the Fixed Burst Sizes Model: cohesins degrade in bursts of size b
        at Poisson-distributed times with rate lambda_i * state[i].
        Continues until all chromosomes reach threshold or propensities are zero.
        """
        burst_size = self.rate_params['burst_size']
        while True:
            # Calculate propensities: rate of bursts, proportional to remaining cohesins
            propensities = [self.rate_params['k'] * self.state[i]
                            for i in range(3)]
            total_propensity = sum(propensities)

            # Stop if no further bursts possible or all thresholds reached
            if total_propensity <= 0 or all(t is not None for t in self.separate_times):
                break

            # Time to next burst
            tau = np.random.exponential(1 / total_propensity)
            self.time += tau

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

    def _simulate_random_normal_burst(self):
        """
        Simulate the Random Normal Burst Model: cohesins degrade in bursts with size
        drawn from a normal distribution with mean mean_burst_size and variance var_burst_size,
        at Poisson-distributed times with rate lambda_i * state[i].
        Continues until all chromosomes reach threshold or propensities are zero.
        """
        while True:
            # Calculate propensities: rate of bursts, proportional to remaining cohesins
            propensities = [self.rate_params['k'] *
                            max(self.state[i], 0) for i in range(3)]
            total_propensity = sum(propensities)

            # Stop if no further bursts possible or all thresholds reached
            if total_propensity <= 0 or all(t is not None for t in self.separate_times):
                break

            # Time to next burst
            tau = np.random.exponential(1 / total_propensity)
            self.time += tau

            # Choose which chromosome experiences a burst
            r = np.random.uniform(0, total_propensity)
            cumulative_propensity = 0
            for i in range(3):
                cumulative_propensity += propensities[i]
                if r < cumulative_propensity:
                    # Remove burst_size cohesins, but not below 0, drawn from normal distribution
                    burst_size = max(np.random.normal(self.rate_params['burst_size'], np.sqrt(
                        self.rate_params['var_burst_size'])), 0)
                    self.state[i] = max(0, self.state[i] - burst_size)
                    break

            # Record time and state
            self.times.append(self.time)
            self.states.append(self.state.copy())

            # Check for separation times
            for i in range(3):
                if self.separate_times[i] is None and self.state[i] <= round(self.n0_list[i]):
                    self.separate_times[i] = self.time

    def _simulate_geometric_burst(self):
        """
        Simulate the Geometric Burst Model: cohesins degrade in bursts with size
        drawn from a geometric distribution with mean burst_size.
        The geometric distribution models the number of trials until first success,
        at Poisson-distributed times with rate k * state[i].
        Continues until all chromosomes reach threshold or propensities are zero.
        """
        while True:
            # Calculate propensities: rate of bursts, proportional to remaining cohesins
            propensities = [self.rate_params['k'] *
                            max(self.state[i], 0) for i in range(3)]
            total_propensity = sum(propensities)

            # Stop if no further bursts possible or all thresholds reached
            if total_propensity <= 0 or all(t is not None for t in self.separate_times):
                break

            # Time to next burst
            tau = np.random.exponential(1 / total_propensity)
            self.time += tau

            # Choose which chromosome experiences a burst
            r = np.random.uniform(0, total_propensity)
            cumulative_propensity = 0
            for i in range(3):
                cumulative_propensity += propensities[i]
                if r < cumulative_propensity:
                    # Calculate geometric distribution parameter from mean
                    # For geometric distribution: mean = 1/p, so p = 1/mean
                    mean_burst_size = self.rate_params['burst_size']
                    p = 1.0 / mean_burst_size
                    # Ensure p is valid (between 0 and 1)
                    p = min(max(p, 1e-10), 1.0 - 1e-10)
                    
                    # Draw burst size from geometric distribution
                    # numpy.random.geometric uses parameter p (probability of success)
                    # and returns number of failures before first success + 1
                    burst_size = np.random.geometric(p)
                    
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

    def _simulate_feedback_onion(self):
        """
        Simulate the Feedback Onion Model: cohesin degradation with onion feedback mechanism.
        Each cohesin degrades at a rate k * W(N_i) * state[i], reducing state[i] by 1.
        W(N_i) = (N_i/n_inner)^(-1/3) for N_i > n_inner, W(N_i) = 1 for N_i <= n_inner.
        Continues until all chromosomes reach threshold or propensities are zero.
        """
        while True:
            # Calculate propensities: rate of degradation for each chromosome
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
                
                propensity = self.rate_params['k'] * W_m * m
                propensities.append(propensity)
            total_propensity = sum(propensities)

            # Stop if no further degradation possible or all thresholds reached
            if total_propensity <= 0 or all(t is not None for t in self.separate_times):
                break

            # Time to next degradation event
            tau = np.random.exponential(1 / total_propensity)
            self.time += tau

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


    def _simulate_fixed_burst_feedback_onion(self):
        """
        Simulate the Fixed Burst with Feedback Onion Model: cohesins degrade in bursts of size b
        at rates modified by onion feedback effects.
        The rate for each chromosome i is k * W(N_i) * state[i], where
        W(N_i) = (N_i/n_inner_i)^(-1/3) for N_i > n_inner_i, else W(N_i) = 1.
        Continues until all chromosomes reach threshold or propensities are zero.
        """
        burst_size = self.rate_params['burst_size']
        while True:
            # Calculate propensities: rate of bursts with onion feedback effect
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
                
                propensity = self.rate_params['k'] * W_m * m
                propensities.append(propensity)
            total_propensity = sum(propensities)

            # Stop if no further bursts possible or all thresholds reached
            if total_propensity <= 0 or all(t is not None for t in self.separate_times):
                break

            # Time to next burst
            tau = np.random.exponential(1 / total_propensity)
            self.time += tau

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
        if self.mechanism == 'simple':
            self._simulate_simple()
        elif self.mechanism == 'fixed_burst':
            self._simulate_fixed_burst()
        elif self.mechanism == 'random_normal_burst':
            self._simulate_random_normal_burst()
        elif self.mechanism == 'geometric_burst':
            self._simulate_geometric_burst()
        elif self.mechanism == 'feedback_onion':
            self._simulate_feedback_onion()
        elif self.mechanism == 'fixed_burst_feedback_onion':
            self._simulate_fixed_burst_feedback_onion()

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
