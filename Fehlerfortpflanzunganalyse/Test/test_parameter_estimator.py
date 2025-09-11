import unittest
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import patch
import tempfile
import shutil

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parameter_estimator import estimate_parameters, fit_parameters, monte_carlo_simulation


class TestParameterEstimator(unittest.TestCase):
    """Unit tests for parameter_estimator.py functions"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample processed data
        self.sample_processed_data = pd.DataFrame({
            'reaction': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'c1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            'c2': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
            'c3': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'rates': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        })
        
        # Create simple test model info
        self.simple_model_info = {
            'name': 'test_model',
            'function': lambda x, a, b: a * x[0] + b,
            'param_names': ['a', 'b'],
            'param_units': ['U', 'U'],
            'initial_guess_func': lambda activities, substrate_data: [1.0, 0.1],
            'bounds_lower': [0, 0],
            'bounds_upper': [np.inf, np.inf],
            'description': 'Simple linear test model'
        }
        
        # Create sample data_info
        self.sample_data_info = {
            'test_key': 'test_value'
        }
        
        # Create sample calibration data
        self.sample_calibration_data = pd.DataFrame({
            'concentration': [0.0, 1.0, 2.0, 3.0, 4.0],
            'absorbance': [0.0, 0.1, 0.2, 0.3, 0.4]
        })
        
        # Create sample reaction data
        self.sample_reaction_data = {
            'reaction1': pd.DataFrame(np.random.rand(10, 8)),
            'reaction2': pd.DataFrame(np.random.rand(10, 8))
        }
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Clean up after each test method."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)


class TestEstimateParameters(TestParameterEstimator):
    """Tests for estimate_parameters function"""
    
    def test_estimate_parameters_success(self):
        """Test successful parameter estimation"""
        result = estimate_parameters(
            self.simple_model_info,
            self.sample_data_info,
            self.sample_processed_data,
            verbose=False
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('params', result)
        self.assertIn('r_squared', result)
        self.assertIn('model_name', result)
        
    def test_estimate_parameters_empty_data(self):
        """Test parameter estimation with empty data"""
        empty_data = pd.DataFrame()
        
        result = estimate_parameters(
            self.simple_model_info,
            self.sample_data_info,
            empty_data,
            verbose=False
        )
        
        self.assertIsInstance(result, dict)
        self.assertFalse(result.get('success', True))
        self.assertIn('error', result)
        
    def test_estimate_parameters_missing_columns(self):
        """Test parameter estimation with missing required columns"""
        incomplete_data = pd.DataFrame({
            'reaction': [1, 2, 3],
            'rates': [0.1, 0.2, 0.3]
            # Missing c1, c2, c3 columns
        })
        
        result = estimate_parameters(
            self.simple_model_info,
            self.sample_data_info,
            incomplete_data,
            verbose=False
        )
        
        self.assertIsInstance(result, dict)
        # Should handle missing columns gracefully by filling with zeros
        
    def test_estimate_parameters_verbose_mode(self):
        """Test parameter estimation in verbose mode"""
        with patch('builtins.print') as mock_print:
            estimate_parameters(
                self.simple_model_info,
                self.sample_data_info,
                self.sample_processed_data,
                verbose=True
            )
            
            # Check that print was called (verbose output)
            mock_print.assert_called()
            
    def test_estimate_parameters_custom_sigma(self):
        """Test parameter estimation with custom sigma value"""
        result = estimate_parameters(
            self.simple_model_info,
            self.sample_data_info,
            self.sample_processed_data,
            sigma=2.0,
            verbose=False
        )
        
        self.assertIsInstance(result, dict)


class TestFitParameters(TestParameterEstimator):
    """Tests for fit_parameters function"""
    
    def test_fit_parameters_success(self):
        """Test successful parameter fitting"""
        # Create simple test data
        x_data = [np.array([1, 2, 3, 4, 5])]
        y_data = np.array([2.1, 4.2, 6.1, 8.2, 10.1])  # approximately y = 2*x
        
        simple_model = {
            'name': 'linear',
            'function': lambda x, a: a * x[0],
            'param_names': ['slope'],
            'param_units': ['U'],
            'initial_guess_func': lambda activities, substrate_data: [1.0],
            'bounds_lower': [0],
            'bounds_upper': [np.inf],
            'description': 'Linear model y = a*x'
        }
        
        result = fit_parameters(x_data, y_data, simple_model, verbose=False)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get('success', False))
        self.assertIn('params', result)
        self.assertIn('param_errors', result)
        self.assertIn('r_squared', result)
        self.assertIn('correlation_matrix', result)
        
        # Check that fitted parameter is close to expected value (2.0)
        self.assertAlmostEqual(result['params'][0], 2.0, places=0)
        
    def test_fit_parameters_empty_activities(self):
        """Test fitting with empty activities array"""
        x_data = [np.array([1, 2, 3])]
        y_data = np.array([])
        
        result = fit_parameters(x_data, y_data, self.simple_model_info, verbose=False)
        
        self.assertIsInstance(result, dict)
        self.assertFalse(result.get('success', True))
        
    def test_fit_parameters_fitting_failure(self):
        """Test fitting when curve_fit fails"""
        # Create data and model that will definitely cause fitting to fail
        x_data = [np.array([1, 2, 3])]
        y_data = np.array([1, 2, 3])
        
        # Model with impossible constraints - upper bound lower than lower bound
        bad_model = {
            'name': 'bad_model',
            'function': lambda x, a: a * x[0],
            'param_names': ['a'],
            'param_units': ['U'],
            'initial_guess_func': lambda activities, substrate_data: [1.0],
            'bounds_lower': [10],    # Lower bound higher than upper bound
            'bounds_upper': [1],     # This will cause curve_fit to fail
            'description': 'Bad model for testing with impossible bounds'
        }
        
        result = fit_parameters(x_data, y_data, bad_model, verbose=False)
        
        self.assertIsInstance(result, dict)
        self.assertFalse(result.get('success', True))
        
    def test_fit_parameters_correlation_matrix(self):
        """Test correlation matrix calculation"""
        # Create data for two-parameter model
        x_data = [np.array([1, 2, 3, 4, 5])]
        y_data = np.array([3.1, 5.2, 7.1, 9.2, 11.1])  # approximately y = 2*x + 1
        
        two_param_model = {
            'name': 'linear_with_intercept',
            'function': lambda x, a, b: a * x[0] + b,
            'param_names': ['slope', 'intercept'],
            'param_units': ['U', 'U'],
            'initial_guess_func': lambda activities, substrate_data: [1.0, 0.0],
            'bounds_lower': [0, -np.inf],
            'bounds_upper': [np.inf, np.inf],
            'description': 'Linear model y = a*x + b'
        }
        
        result = fit_parameters(x_data, y_data, two_param_model, verbose=False)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get('success', False))
        self.assertIsNotNone(result.get('correlation_matrix'))
        
        # Correlation matrix should be 2x2 for two parameters
        corr_matrix = result['correlation_matrix']
        self.assertEqual(corr_matrix.shape, (2, 2))
        
        # Diagonal elements should be 1.0
        np.testing.assert_allclose(np.diag(corr_matrix), [1.0, 1.0], rtol=1e-10)


class TestMonteCarloSimulation(TestParameterEstimator):
    """Tests for monte_carlo_simulation function"""
    
    @patch('parameter_estimator.add_noise_calibration')
    @patch('parameter_estimator.calc_calibration_slope')
    @patch('parameter_estimator.add_noise_plate_reader_data')
    @patch('parameter_estimator.compute_processed_data')
    @patch('parameter_estimator.estimate_parameters')
    @patch('parameter_estimator.os.makedirs')
    @patch('builtins.open', create=True)
    @patch('parameter_estimator.pickle.dump')
    def test_monte_carlo_simulation_success(self, mock_pickle, mock_open, mock_makedirs,
                                          mock_estimate, mock_compute, mock_noise_plate,
                                          mock_calc_slope, mock_noise_calib):
        """Test successful Monte Carlo simulation"""
        
        # Setup mocks
        mock_noise_calib.return_value = self.sample_calibration_data
        mock_calc_slope.return_value = 0.1
        mock_noise_plate.return_value = self.sample_reaction_data
        mock_compute.return_value = self.sample_processed_data
        
        # Mock successful parameter estimation
        mock_estimate.return_value = {
            'success': True,
            'params': np.array([1.0, 2.0]),
            'param_errors': np.array([0.1, 0.2]),
            'r_squared': 0.95
        }
        
        # Setup model info for Monte Carlo
        mc_model_info = {
            'name': 'test_mc_model',
            'description': 'Test model for Monte Carlo',
            'param_names': ['param1', 'param2'],
            'param_units': ['U', 'U']
        }
        
        noise_level = {
            'calibration': 0.01,
            'reaction': 0.01,
            'concentration': 0.01
        }
        
        # Run Monte Carlo with few iterations for testing
        result = monte_carlo_simulation(
            self.sample_calibration_data,
            self.sample_reaction_data,
            mc_model_info,
            self.sample_data_info,
            noise_level,
            n_iterations=10,
            verbose=False
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('n_successful', result)
        self.assertIn('n_total', result)
        self.assertIn('success_rate', result)
        self.assertIn('model_name', result)
        self.assertEqual(result['n_total'], 10)
        
    @patch('parameter_estimator.add_noise_calibration')
    @patch('parameter_estimator.calc_calibration_slope')
    def test_monte_carlo_simulation_calibration_failure(self, mock_calc_slope, mock_noise_calib):
        """Test Monte Carlo simulation with calibration failure"""
        
        # Setup mocks to fail
        mock_noise_calib.side_effect = Exception("Calibration failed")
        
        mc_model_info = {
            'name': 'test_mc_model',
            'description': 'Test model for Monte Carlo',
            'param_names': ['param1'],
            'param_units': ['U']
        }
        
        noise_level = {
            'calibration': 0.01,
            'reaction': 0.01,
            'concentration': 0.01
        }
        
        # Should handle calibration failure gracefully
        result = monte_carlo_simulation(
            self.sample_calibration_data,
            self.sample_reaction_data,
            mc_model_info,
            self.sample_data_info,
            noise_level,
            n_iterations=5,
            verbose=False
        )
        
        self.assertIsInstance(result, dict)
        
    @patch('parameter_estimator.add_noise_calibration')
    @patch('parameter_estimator.calc_calibration_slope')
    @patch('parameter_estimator.add_noise_processed_data')
    @patch('parameter_estimator.compute_processed_data')
    def test_monte_carlo_simulation_processed_data_noise(self, mock_compute, mock_noise_processed,
                                                        mock_calc_slope, mock_noise_calib):
        """Test Monte Carlo simulation with processed data noise model"""
        
        # Setup mocks
        mock_noise_calib.return_value = self.sample_calibration_data
        mock_calc_slope.return_value = 0.1
        mock_compute.return_value = self.sample_processed_data
        mock_noise_processed.return_value = self.sample_processed_data
        
        mc_model_info = {
            'name': 'test_mc_model',
            'description': 'Test model for Monte Carlo',
            'param_names': ['param1'],
            'param_units': ['U']
        }
        
        noise_level = {
            'calibration': 0.01,
            'reaction': 0.01,
            'concentration': 0.01
        }
        
        # Test with processed_data noise model
        result = monte_carlo_simulation(
            self.sample_calibration_data,
            self.sample_reaction_data,
            mc_model_info,
            self.sample_data_info,
            noise_level,
            noise_model="processed_data",
            n_iterations=5,
            verbose=False
        )
        
        self.assertIsInstance(result, dict)
        mock_noise_processed.assert_called()
        
    def test_monte_carlo_simulation_insufficient_results(self):
        """Test Monte Carlo simulation with insufficient successful results"""
        
        mc_model_info = {
            'name': 'test_mc_model',
            'description': 'Test model for Monte Carlo',
            'param_names': ['param1'],
            'param_units': ['U']
        }
        
        noise_level = {
            'calibration': 0.01,
            'reaction': 0.01,
            'concentration': 0.01
        }
        
        # Mock everything to fail
        with patch('parameter_estimator.add_noise_calibration') as mock_noise_calib:
            mock_noise_calib.side_effect = Exception("Always fail")
            
            result = monte_carlo_simulation(
                self.sample_calibration_data,
                self.sample_reaction_data,
                mc_model_info,
                self.sample_data_info,
                noise_level,
                n_iterations=5,
                verbose=False
            )
            
            self.assertIsInstance(result, dict)
            self.assertLess(result.get('n_successful', 0), 10)  # Should have fewer than 10 successful


class TestParameterEstimatorIntegration(TestParameterEstimator):
    """Integration tests for parameter_estimator functions"""
    
    def test_full_workflow_simple_model(self):
        """Test the full workflow with a simple model"""
        
        # Create synthetic data that follows a known model
        np.random.seed(42)  # For reproducible results
        
        # Generate data: y = 2*x + noise
        x_true = np.linspace(1, 10, 20)
        y_true = 2.0 * x_true + 0.1 * np.random.randn(20)
        
        # Create DataFrame in expected format
        test_data = pd.DataFrame({
            'reaction': [1] * 20,
            'c1': x_true,
            'c2': np.ones(20),  # Constant
            'c3': np.zeros(20),  # Not used
            'rates': y_true
        })
        
        # Define simple linear model
        linear_model = {
            'name': 'linear_test',
            'function': lambda x, a: a * x[0],  # y = a * x1
            'param_names': ['slope'],
            'param_units': ['U'],
            'initial_guess_func': lambda activities, substrate_data: [1.0],
            'bounds_lower': [0],
            'bounds_upper': [np.inf],
            'description': 'Simple linear model for testing'
        }
        
        # Test parameter estimation
        result = estimate_parameters(
            linear_model,
            self.sample_data_info,
            test_data,
            verbose=False
        )
        
        self.assertTrue(result.get('success', False))
        self.assertAlmostEqual(result['params'][0], 2.0, places=0)  # Should recover slope ≈ 2
        self.assertGreater(result['r_squared'], 0.9)  # Should have good fit
        
    def test_error_handling_robustness(self):
        """Test error handling across all functions"""
        
        # Test with completely invalid data
        invalid_data = pd.DataFrame({
            'invalid_column': [1, 2, 3]
        })
        
        result = estimate_parameters(
            self.simple_model_info,
            self.sample_data_info,
            invalid_data,
            verbose=False
        )
        
        self.assertIsInstance(result, dict)
        self.assertFalse(result.get('success', True))


if __name__ == '__main__':
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEstimateParameters))
    suite.addTests(loader.loadTestsFromTestCase(TestFitParameters))
    suite.addTests(loader.loadTestsFromTestCase(TestMonteCarloSimulation))
    suite.addTests(loader.loadTestsFromTestCase(TestParameterEstimatorIntegration))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors))/result.testsRun*100:.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")
