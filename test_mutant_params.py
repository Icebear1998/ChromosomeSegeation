
import sys
import unittest
import numpy as np
from simulation_utils import apply_mutant_params, get_parameter_names, get_parameter_bounds

class TestMutantParams(unittest.TestCase):
    def test_apply_mutant_params_simple(self):
        base_params = {
            'n1': 10, 'n2': 10, 'n3': 10,
            'N1': 100, 'N2': 100, 'N3': 100,
            'k': 1.0,
            'r21': 1.0, 'r23': 1.0,
            'R21': 1.0, 'R23': 1.0
        }
        
        # Test degrade (separase) -> beta_k1
        params, _ = apply_mutant_params(base_params, 'degrade', alpha=0.5, beta_k1=0.5, beta_k2=0.6, beta_k3=0.7)
        self.assertEqual(params['k'], 0.5 * 1.0, "Simple degrade should use beta_k1")
        
        # Test degradeAPC -> beta_k2
        params, _ = apply_mutant_params(base_params, 'degradeAPC', alpha=0.5, beta_k1=0.5, beta_k2=0.6, beta_k3=0.7)
        self.assertEqual(params['k'], 0.6 * 1.0, "Simple degradeAPC should use beta_k2")
        
        # Test velcade -> beta_k3
        params, _ = apply_mutant_params(base_params, 'velcade', alpha=0.5, beta_k1=0.5, beta_k2=0.6, beta_k3=0.7)
        self.assertEqual(params['k'], 0.7 * 1.0, "Simple velcade should use beta_k3")
        
    def test_apply_mutant_params_timevars(self):
        base_params = {
            'n1': 10, 'n2': 10, 'n3': 10,
            'N1': 100, 'N2': 100, 'N3': 100,
            'k_max': 1.0, 'tau': 10.0,
            'r21': 1.0, 'r23': 1.0,
            'R21': 1.0, 'R23': 1.0
        }
        
        # Test degrade (separase) -> beta_k
        params, _ = apply_mutant_params(base_params, 'degrade', alpha=0.5, beta_k=0.5, beta_tau=2.0, beta_tau2=3.0)
        self.assertEqual(params['k_max'], 0.5 * 1.0, "Time-varying degrade should use beta_k")
        
        # Test degradeAPC -> beta_tau
        params, _ = apply_mutant_params(base_params, 'degradeAPC', alpha=0.5, beta_k=0.5, beta_tau=2.0, beta_tau2=3.0)
        self.assertEqual(params['tau'], 2.0 * 10.0, "Time-varying degradeAPC should use beta_tau")
        
        # Test velcade -> beta_tau2
        params, _ = apply_mutant_params(base_params, 'velcade', alpha=0.5, beta_k=0.5, beta_tau=2.0, beta_tau2=3.0)
        self.assertEqual(params['tau'], 3.0 * 10.0, "Time-varying velcade should use beta_tau2")

    def test_parameter_names(self):
        simple_names = get_parameter_names('simple')
        self.assertIn('beta_k1', simple_names)
        self.assertIn('beta_k2', simple_names)
        self.assertIn('beta_k3', simple_names)
        self.assertNotIn('beta_k', simple_names)
        
        tv_names = get_parameter_names('time_varying_k')
        self.assertIn('beta_k', tv_names)
        self.assertNotIn('beta_k1', tv_names)

if __name__ == '__main__':
    unittest.main()
