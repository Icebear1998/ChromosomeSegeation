import numpy as np
import warnings


class MultiMechanismSimulation:
    """
    A simulation class for modeling cohesin degradation during chromosome segregation
    with multiple mechanisms. Supports Simple (gradual degradation) and Fixed Burst Sizes
    (bursty degradation with fixed burst size and Poisson timing) models.
    Issues a warning if simulation time exceeds max_time instead of stopping early.
    """

    def __init__(self, mechanism, initial_state_list, rate_params, n0_list, max_time):
        """
        Initialize the simulation.

        Args:
            mechanism (str): 'simple', 'fixed_burst', 'random_normal_burst', 'time_varying_k', 'feedback_linear', 'feedback_onion', 'feedback', 'feedback_zipper', or 'fixed_burst_feedback_linear'.
            initial_state_list (list): Initial cohesin counts [N1, N2, N3].
            rate_params (dict): For 'simple': {'k_list': [k1, k2, k3]}.
                               For 'fixed_burst': {'lambda_list': [lambda1, lambda2, lambda3], 'burst_size': b}.
                               For 'fixed_burst_feedback_linear': {'k': k, 'burst_size': b, 'w1': w1, 'w2': w2, 'w3': w3}.
                               For 'feedback_onion': {'k': k, 'n_inner1': n_inner1, 'n_inner2': n_inner2, 'n_inner3': n_inner3}.
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
        if self.mechanism not in ['simple', 'fixed_burst', 'random_normal_burst', 'time_varying_k', 'feedback_linear', 'feedback_onion', 'feedback', 'feedback_zipper', 'fixed_burst_feedback_linear']:
            raise ValueError(
                "Mechanism must be 'simple', 'fixed_burst', 'random_normal_burst', 'time_varying_k', 'feedback_linear', 'feedback_onion', 'feedback', 'feedback_zipper', or 'fixed_burst_feedback_linear'.")
        if self.mechanism == 'fixed_burst' and 'burst_size' not in rate_params:
            raise ValueError(
                "Fixed burst mechanism requires 'lambda_list' and 'burst_size' in rate_params.")
        if self.mechanism == 'fixed_burst_feedback_linear':
            required_params = ['k', 'burst_size', 'w1', 'w2', 'w3']
            missing_params = [param for param in required_params if param not in rate_params]
            if missing_params:
                raise ValueError(
                    f"Fixed burst feedback linear mechanism requires {required_params} in rate_params. Missing: {missing_params}")
        if self.mechanism == 'feedback_onion':
            required_params = ['k', 'n_inner1', 'n_inner2', 'n_inner3']
            missing_params = [param for param in required_params if param not in rate_params]
            if missing_params:
                raise ValueError(
                    f"Feedback onion mechanism requires {required_params} in rate_params. Missing: {missing_params}")
        if self.mechanism == 'feedback_zipper':
            required_params = ['k', 'z1', 'z2', 'z3']
            missing_params = [param for param in required_params if param not in rate_params]
            if missing_params:
                raise ValueError(
                    f"Feedback zipper mechanism requires {required_params} in rate_params. Missing: {missing_params}")

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
                if self.separate_times[i] is None and self.state[i] <= self.n0_list[i]:
                    self.separate_times[i] = self.time

    def _simulate_k_change(self):
        """
        Simulate the time-varying degradation model with k(t) = k_0 + k_1 t.
        Each cohesin degrades at rate k(t) * state[i], reducing state[i] by 1.
        Continues until the chromosome reaches its threshold.
        """
        while True:
            # Calculate propensity: rate of degradation for the chromosome
            k_t = self.rate_params['k'] + self.rate_params['k_1'] * self.time
            k_t = max(k_t, 1e-10)

            # Calculate propensities: rate of degradation for each chromosome
            propensities = [k_t * self.state[i] for i in range(3)]
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
                if self.separate_times[i] is None and self.state[i] <= self.n0_list[i]:
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
                if self.separate_times[i] is None and self.state[i] <= self.n0_list[i]:
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
                if self.separate_times[i] is None and self.state[i] <= self.n0_list[i]:
                    self.separate_times[i] = self.time

    def _simulate_feedbackLinear(self):
        """
        Simulate the Feedback Model: cohesin degradation with feedback mechanism.
        Each cohesin degrades at a rate k_i * W(state[i]) * state[i], reducing state[i] by 1.
        W(m) is a sigmoidal function increasing as m decreases, reflecting reduced blocking.
        Continues until all chromosomes reach threshold or propensities are zero.
        """
        while True:
            # Calculate propensities: rate of degradation for each chromosome
            propensities = []
            for i in range(3):
                m = self.state[i]
                W_m = 1 - self.rate_params['w' + str(i+1)]*m
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
                if self.separate_times[i] is None and self.state[i] <= self.n0_list[i]:
                    self.separate_times[i] = self.time

    def _simulate_feedback(self):
        """
        Simulate the Feedback Model: cohesin degradation with feedback mechanism.
        Each cohesin degrades at a rate k_i * W(state[i]) * state[i], reducing state[i] by 1.
        W(m) is a sigmoidal function increasing as m decreases, reflecting reduced blocking.
        Continues until all chromosomes reach threshold or propensities are zero.
        """
        while True:
            # Calculate propensities: rate of degradation for each chromosome
            propensities = []
            for i in range(3):
                m = self.state[i]
                # Compute W(m) = 1 / (1 + e^(a * (m - m_threshold)))
                W_m = 1 / (1 + np.exp(self.rate_params['feedbackSteepness'] * (
                    m - self.rate_params['feedbackThreshold'])))
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
                if self.separate_times[i] is None and self.state[i] <= self.n0_list[i]:
                    self.separate_times[i] = self.time

    def _simulate_fixed_burst_feedback_linear(self):
        """
        Simulate the Fixed Burst with Feedback Linear Model: cohesins degrade in bursts of size b
        at rates modified by linear feedback effects.
        The rate for each chromosome i is k * (1 - w_i * state[i]) * state[i].
        Continues until all chromosomes reach threshold or propensities are zero.
        """
        burst_size = self.rate_params['burst_size']
        while True:
            # Calculate propensities: rate of bursts with feedback effect
            propensities = []
            for i in range(3):
                m = self.state[i]
                W_m = 1 - self.rate_params['w' + str(i+1)] * m
                if W_m <= 0:  # Ensure positive rate
                    W_m = 1e-10
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
                if self.separate_times[i] is None and self.state[i] <= self.n0_list[i]:
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
                n_inner = self.rate_params['n_inner' + str(i+1)]
                
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
                if self.separate_times[i] is None and self.state[i] <= self.n0_list[i]:
                    self.separate_times[i] = self.time

    def _simulate_feedback_zipper(self):
        """
        Simulate the Feedback Zipper Model: cohesin degradation with zipper feedback mechanism.
        Each cohesin degrades at a rate k * W(N_i) * state[i], reducing state[i] by 1.
        W(N_i) = z_i / N_i where z_i is the feedback parameter for chromosome i.
        Continues until all chromosomes reach threshold or propensities are zero.
        """
        while True:
            # Calculate propensities: rate of degradation for each chromosome
            propensities = []
            for i in range(3):
                m = self.state[i]
                N_i = self.initial_state_list[i]  # Initial cohesin count for chromosome i
                z = self.rate_params['z' + str(i+1)]
                
                # Calculate W_m based on zipper feedback mechanism: W_m = z / N_i
                W_m = z / N_i
                
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
                if self.separate_times[i] is None and self.state[i] <= self.n0_list[i]:
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
        # Reset state
        self.state = self.initial_state_list.copy()
        self.time = 0
        self.times = [0]
        self.states = [self.initial_state_list.copy()]
        self.separate_times = [None, None, None]

        # Run appropriate simulation
        if self.mechanism == 'simple':
            self._simulate_simple()
        elif self.mechanism == 'fixed_burst':
            self._simulate_fixed_burst()
        elif self.mechanism == 'random_normal_burst':
            self._simulate_random_normal_burst()
        elif self.mechanism == 'time_varying_k':
            self._simulate_k_change()
        elif self.mechanism == 'feedback':
            self._simulate_feedback()
        elif self.mechanism == 'feedback_linear':
            self._simulate_feedbackLinear()
        elif self.mechanism == 'feedback_onion':
            self._simulate_feedback_onion()
        elif self.mechanism == 'feedback_zipper':
            self._simulate_feedback_zipper()
        elif self.mechanism == 'fixed_burst_feedback_linear':
            self._simulate_fixed_burst_feedback_linear()

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
