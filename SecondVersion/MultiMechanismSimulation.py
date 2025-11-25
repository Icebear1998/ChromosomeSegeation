#!/usr/bin/env python3
"""
Simplified refactored version of MultiMechanismSimulation.py

Key improvements:
1. Single simulation core for all mechanisms
2. Simple mechanism selection via functions
3. Easy to add new mechanisms
4. Reduced code duplication
5. Better readability

The main insight: All mechanisms share the same simulation loop,
they only differ in how they calculate burst sizes.
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Callable


class MultiMechanismSimulation:
    """
    Simplified simulation class where all mechanisms share the same core simulation loop.
    The only difference between mechanisms is how they calculate burst sizes.
    """
    
    def __init__(self, mechanism: str, initial_state_list: List[float], 
                 rate_params: Dict[str, float], n0_list: List[float], max_time: float):
        """
        Initialize the simulation.
        
        Args:
            mechanism: Name of the mechanism ('simple', 'fixed_burst', etc.)
            initial_state_list: Initial cohesin counts [N1, N2, N3]
            rate_params: Parameters like {'k': 0.1, 'burst_size': 5.0}
            n0_list: Threshold counts [n01, n02, n03]
            max_time: Maximum simulation time
        """
        self.mechanism = mechanism
        self.initial_state_list = list(initial_state_list)
        self.rate_params = rate_params.copy()
        self.n0_list = list(n0_list)
        self.max_time = max_time
        
        # Get the burst size calculator for this mechanism
        self.calculate_burst_size = self._get_burst_calculator()
        self.calculate_propensities = self._get_propensity_calculator()
        
        # Validate parameters
        self._validate_parameters()
    
    def _get_burst_calculator(self) -> Callable[[], float]:
        """Return the burst size calculation function for the current mechanism."""
        
        if self.mechanism == 'simple':
            def simple_burst():
                return 1.0
            return simple_burst
        
        elif self.mechanism == 'fixed_burst':
            def fixed_burst():
                return self.rate_params['burst_size']
            return fixed_burst
        
        elif self.mechanism == 'random_normal_burst':
            def normal_burst():
                mean = self.rate_params['burst_size']
                std = np.sqrt(self.rate_params['var_burst_size'])
                return max(0, np.random.normal(mean, std))
            return normal_burst
        
        elif self.mechanism == 'geometric_burst':
            def geometric_burst():
                mean_burst = self.rate_params['burst_size']
                p = min(max(1.0 / mean_burst, 1e-10), 1.0 - 1e-10)
                return float(np.random.geometric(p))
            return geometric_burst
        
        elif self.mechanism == 'feedback_onion':
            def feedback_onion_burst():
                return 1.0
            return feedback_onion_burst
        
        elif self.mechanism == 'fixed_burst_feedback_onion':
            def fixed_burst_feedback_onion_burst():
                return self.rate_params['burst_size']
            return fixed_burst_feedback_onion_burst
        
        else:
            raise ValueError(f"Unknown mechanism: {self.mechanism}")
    
    def _get_propensity_calculator(self) -> Callable[[List[int]], List[float]]:
        """Return the propensity calculation function for the current mechanism."""
        
        if self.mechanism == 'feedback_onion':
            def feedback_onion_propensities(states):
                propensities = []
                for i, state in enumerate(states):
                    n_inner = self.rate_params['n_inner']
                    
                    # Calculate feedback weight based on CURRENT state (not initial)
                    # This matches the MoM calculation where W_m depends on current m
                    if state > n_inner:
                        W_m = (state / n_inner) ** (-1/3)
                    else:
                        W_m = 1.0
                    
                    propensity = self.rate_params['k'] * W_m * max(state, 0)
                    propensities.append(propensity)
                return propensities
            return feedback_onion_propensities
        
        elif self.mechanism == 'fixed_burst_feedback_onion':
            def fixed_burst_feedback_onion_propensities(states):
                propensities = []
                for i, state in enumerate(states):
                    n_inner = self.rate_params['n_inner']
                    
                    # Calculate feedback weight based on CURRENT state (not initial)
                    # This matches the MoM calculation where W_m depends on current m
                    if state > n_inner:
                        W_m = (state / n_inner) ** (-1/3)
                    else:
                        W_m = 1.0
                    
                    propensity = self.rate_params['k'] * W_m * max(state, 0)
                    propensities.append(propensity)
                return propensities
            return fixed_burst_feedback_onion_propensities
        
        else:
            # Default propensity calculation for most mechanisms
            def standard_propensities(states):
                return [self.rate_params['k'] * max(state, 0) for state in states]
            return standard_propensities
    
    def _validate_parameters(self):
        """Check that all required parameters are present."""
        required_params = {
            'simple': ['k'],
            'fixed_burst': ['k', 'burst_size'],
            'random_normal_burst': ['k', 'burst_size', 'var_burst_size'],
            'geometric_burst': ['k', 'burst_size'],
            'feedback_onion': ['k', 'n_inner'],
            'fixed_burst_feedback_onion': ['k', 'burst_size', 'n_inner']
        }
        
        if self.mechanism not in required_params:
            available = list(required_params.keys())
            raise ValueError(f"Unknown mechanism '{self.mechanism}'. Available: {available}")
        
        required = required_params[self.mechanism]
        missing = [param for param in required if param not in self.rate_params]
        if missing:
            raise ValueError(f"Mechanism '{self.mechanism}' requires {required}. Missing: {missing}")
    
    def simulate(self) -> Tuple[List[float], List[List[int]], List[float]]:
        """
        Run the simulation using the shared core algorithm.
        
        Returns:
            tuple: (times, states, separate_times)
        """
        # Initialize simulation state
        current_state = [int(s) for s in self.initial_state_list]
        current_time = 0.0
        times = [0.0]
        states = [current_state.copy()]
        separate_times = [None, None, None]
        
        # Main simulation loop - SAME FOR ALL MECHANISMS
        while True:
            # Calculate propensities (mechanism-specific)
            propensities = self.calculate_propensities(current_state)
            total_propensity = sum(propensities)
            
            # Check if simulation should stop
            if total_propensity <= 0 or all(t is not None for t in separate_times):
                break
            
            # Calculate time to next event
            tau = np.random.exponential(1.0 / total_propensity)
            current_time += tau
            
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
                if separate_times[i] is None and current_state[i] <= max(self.n0_list[i],1):
                    separate_times[i] = current_time
        
        # Finalize simulation
        for i in range(3):
            if separate_times[i] is None:
                separate_times[i] = current_time
        
        if current_time > self.max_time:
            warnings.warn(f"Simulation time ({current_time:.2f}) exceeded max_time ({self.max_time:.2f})")
        
        return times, states, separate_times
    
    @staticmethod
    def get_available_mechanisms() -> List[str]:
        """Get list of all available mechanisms."""
        return ['simple', 'fixed_burst', 'random_normal_burst', 'geometric_burst', 
                'feedback_onion', 'fixed_burst_feedback_onion']
    
    @staticmethod
    def get_mechanism_info(mechanism: str) -> Dict[str, str]:
        """Get information about a specific mechanism."""
        info = {
            'simple': {
                'name': 'Simple',
                'description': 'Single cohesin degradation per event',
                'parameters': 'k (degradation rate)'
            },
            'fixed_burst': {
                'name': 'Fixed Burst',
                'description': 'Fixed number of cohesins per burst',
                'parameters': 'k (rate), burst_size (cohesins per burst)'
            },
            'random_normal_burst': {
                'name': 'Random Normal Burst',
                'description': 'Burst sizes from normal distribution',
                'parameters': 'k (rate), burst_size (mean), var_burst_size (variance)'
            },
            'geometric_burst': {
                'name': 'Geometric Burst',
                'description': 'Burst sizes from geometric distribution',
                'parameters': 'k (rate), burst_size (mean burst size)'
            },
            'feedback_onion': {
                'name': 'Feedback Onion',
                'description': 'Single cohesin with onion feedback',
                'parameters': 'k (rate), n_inner (inner threshold)'
            },
            'fixed_burst_feedback_onion': {
                'name': 'Fixed Burst + Onion Feedback',
                'description': 'Fixed bursts with onion feedback',
                'parameters': 'k (rate), burst_size (burst size), n_inner (inner threshold)'
            }
        }
        
        return info.get(mechanism, {'name': 'Unknown', 'description': 'Unknown mechanism', 'parameters': 'Unknown'})


def add_new_mechanism(mechanism_name: str, burst_calculator: Callable[[], float], 
                     required_params: List[str], propensity_calculator: Callable = None):
    """
    Easy way to add a new mechanism to the simulation.
    
    Args:
        mechanism_name: Name for the new mechanism
        burst_calculator: Function that returns burst size
        required_params: List of required parameter names
        propensity_calculator: Optional custom propensity function
    
    Example:
        # Add exponential burst mechanism
        def exp_burst():
            return np.random.exponential(rate_params['burst_size'])
        
        add_new_mechanism('exponential_burst', exp_burst, ['k', 'burst_size'])
    """
    # This is a simplified example of how you could extend the class
    # In practice, you'd modify the class methods above
    print(f"To add '{mechanism_name}', modify the _get_burst_calculator() method")
    print(f"Required parameters: {required_params}")
    print("This is a design pattern suggestion - actual implementation would modify the class")


if __name__ == "__main__":
    # Example usage
    print("Available mechanisms:", MultiMechanismSimulation.get_available_mechanisms())
    print()
    
    # Show info for each mechanism
    for mechanism in MultiMechanismSimulation.get_available_mechanisms():
        info = MultiMechanismSimulation.get_mechanism_info(mechanism)
        print(f"{info['name']}: {info['description']}")
        print(f"  Parameters: {info['parameters']}")
    print()
    
    # Test different mechanisms with the same simulation core
    mechanisms_to_test = [
        ('simple', {'k': 0.1}),
        ('fixed_burst', {'k': 0.1, 'burst_size': 5.0}),
        ('geometric_burst', {'k': 0.1, 'burst_size': 5.0})
    ]
    
    for mechanism, params in mechanisms_to_test:
        print(f"Testing {mechanism}:")
        
        sim = MultiMechanismSimulation(
            mechanism=mechanism,
            initial_state_list=[100, 80, 120],
            rate_params=params,
            n0_list=[10, 8, 12],
            max_time=100.0
        )
        
        times, states, separate_times = sim.simulate()
        print(f"  Completed in {times[-1]:.2f} time units")
        print(f"  {len(times)} total events")
        print(f"  Separation times: {[f'{t:.2f}' for t in separate_times]}")
        print()
    
    print("All mechanisms use the same simulation core!")
    print("Only the burst size calculation differs between mechanisms.")
