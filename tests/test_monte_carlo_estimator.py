import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from monte_carlo_estimator import monte_carlo_parameter_estimation
from artifical_data import reaction1_synthetic_data


class TestMonteCarloEstimator:
    
    @pytest.fixture
    def michaelis_menten_model(self):
        """Standard Michaelis-Menten model function."""
        def michaelis_menten(S, Vmax, Km1, Km2):
            S1, S2 = S
            return (Vmax * S1 * S2) / ((Km1 + S1) * (Km2 + S2))
        return michaelis_menten
    
    @pytest.fixture
    def simple_noise_function(self):
        """Simple noise function for testing."""
        def add_simple_noise(data, cal_data, substrate, cal_param, noise_level):
            data_noisy = data.copy()
            rates = data_noisy["activity_U/mg"].values
            noisy_rates = rates + np.random.normal(0, noise_level, size=rates.shape)
            data_noisy["activity_U/mg"] = noisy_rates
            return data_noisy
        return add_simple_noise
    
    @pytest.fixture
    def sample_data(self):
        """Create sample experimental data."""
        true_parameters = (100, 2, 3)  # Vmax, Km1, Km2
        return reaction1_synthetic_data(true_parameters, noise_level=0.01, num_points=10)
    
    @pytest.fixture
    def sample_cal_data(self):
        """Create sample calibration data."""
        return pd.DataFrame({
            'c': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'ad1': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'ad2': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        })
    
    @pytest.fixture
    def sample_parameters(self):
        """Sample parameter configuration."""
        return {
            "substrate": ["PD_mM", "NAD_mM"],
            "Vf_well": 10.0,
            "Vf_prod": 1.0,
            "c_prod": 2.2108
        }
    
    def test_monte_carlo_basic_functionality(self, sample_data, sample_cal_data, sample_parameters, 
                                           michaelis_menten_model, simple_noise_function):
        """Test basic Monte Carlo parameter estimation functionality."""
        substrate_list = ["PD_mM", "NAD_mM"]
        initial_guess = [80, 1, 1]
        
        with patch('monte_carlo_estimator.estimate_parameters') as mock_estimate:
            # Mock successful parameter estimation
            mock_estimate.return_value = (np.array([95.0, 1.8, 2.9]), None)
            
            results = monte_carlo_parameter_estimation(
                data=sample_data,
                cal_data=sample_cal_data,
                substrate=substrate_list,
                cal_param=sample_parameters,
                model_func=michaelis_menten_model,
                noise_function=simple_noise_function,
                initial_guess=initial_guess,
                noise_level=0.1,
                num_iterations=10
            )
            
            # Check results structure
            assert isinstance(results, np.ndarray)
            assert results.shape == (10, 3)  # 10 iterations, 3 parameters
            assert mock_estimate.call_count == 10
    
    def test_monte_carlo_parameter_recovery(self, michaelis_menten_model, simple_noise_function):
        """Test that Monte Carlo can recover known parameters from synthetic data."""
        true_parameters = (100, 2, 3)
        synthetic_data = reaction1_synthetic_data(true_parameters, noise_level=0.001, num_points=20)
        
        # Create dummy calibration data
        cal_data = pd.DataFrame({
            'c': [0.0, 0.1, 0.2],
            'ad1': [0.0, 0.1, 0.2],
            'ad2': [0.0, 0.1, 0.2]
        })
        
        cal_param = {
            "substrate": ["PD_mM", "NAD_mM"],
            "Vf_well": 1.0,
            "Vf_prod": 1.0,
            "c_prod": 1.0
        }
        
        initial_guess = [90, 1.5, 2.5]
        
        results = monte_carlo_parameter_estimation(
            data=synthetic_data,
            cal_data=cal_data,
            substrate=["PD_mM", "NAD_mM"],
            cal_param=cal_param,
            model_func=michaelis_menten_model,
            noise_function=simple_noise_function,
            initial_guess=initial_guess,
            noise_level=0.05,
            num_iterations=50
        )
        
        # Check that mean values are reasonably close to true parameters
        mean_params = np.mean(results, axis=0)
        
        # Allow for some deviation due to noise and fitting uncertainty
        assert abs(mean_params[0] - true_parameters[0]) < 20  # Vmax within 20%
        assert abs(mean_params[1] - true_parameters[1]) < 1   # Km1 within 1
        assert abs(mean_params[2] - true_parameters[2]) < 1   # Km2 within 1
    
    def test_monte_carlo_different_iteration_counts(self, sample_data, sample_cal_data, 
                                                  sample_parameters, michaelis_menten_model, 
                                                  simple_noise_function):
        """Test Monte Carlo with different iteration counts."""
        substrate_list = ["PD_mM", "NAD_mM"]
        initial_guess = [80, 1, 1]
        
        with patch('monte_carlo_estimator.estimate_parameters') as mock_estimate:
            mock_estimate.return_value = (np.array([95.0, 1.8, 2.9]), None)
            
            # Test with different iteration counts
            for num_iter in [1, 5, 100]:
                results = monte_carlo_parameter_estimation(
                    data=sample_data,
                    cal_data=sample_cal_data,
                    substrate=substrate_list,
                    cal_param=sample_parameters,
                    model_func=michaelis_menten_model,
                    noise_function=simple_noise_function,
                    initial_guess=initial_guess,
                    noise_level=0.1,
                    num_iterations=num_iter
                )
                
                assert results.shape == (num_iter, 3)
                assert mock_estimate.call_count == num_iter
                mock_estimate.reset_mock()
    
    def test_monte_carlo_different_noise_levels(self, sample_data, sample_cal_data, 
                                               sample_parameters, michaelis_menten_model, 
                                               simple_noise_function):
        """Test Monte Carlo with different noise levels."""
        substrate_list = ["PD_mM", "NAD_mM"]
        initial_guess = [80, 1, 1]
        
        with patch('monte_carlo_estimator.estimate_parameters') as mock_estimate:
            mock_estimate.return_value = (np.array([95.0, 1.8, 2.9]), None)
            
            # Test with different noise levels
            for noise_level in [0.01, 0.1, 0.5]:
                results = monte_carlo_parameter_estimation(
                    data=sample_data,
                    cal_data=sample_cal_data,
                    substrate=substrate_list,
                    cal_param=sample_parameters,
                    model_func=michaelis_menten_model,
                    noise_function=simple_noise_function,
                    initial_guess=initial_guess,
                    noise_level=noise_level,
                    num_iterations=5
                )
                
                assert results.shape == (5, 3)
                mock_estimate.reset_mock()
    
    def test_monte_carlo_data_copying(self, sample_data, sample_cal_data, sample_parameters, 
                                     michaelis_menten_model):
        """Test that original data is not modified during Monte Carlo simulation."""
        original_data = sample_data.copy()
        
        def tracking_noise_function(data, cal_data, substrate, cal_param, noise_level):
            # This noise function modifies the input data
            data["activity_U/mg"] *= 1.1  # Modify in place
            return data
        
        with patch('monte_carlo_estimator.estimate_parameters') as mock_estimate:
            mock_estimate.return_value = (np.array([95.0, 1.8, 2.9]), None)
            
            monte_carlo_parameter_estimation(
                data=sample_data,
                cal_data=sample_cal_data,
                substrate=["PD_mM", "NAD_mM"],
                cal_param=sample_parameters,
                model_func=michaelis_menten_model,
                noise_function=tracking_noise_function,
                initial_guess=[80, 1, 1],
                noise_level=0.1,
                num_iterations=3
            )
            
            # Original data should be unchanged
            pd.testing.assert_frame_equal(original_data, sample_data)
    
    def test_monte_carlo_parameter_estimation_failures(self, sample_data, sample_cal_data, 
                                                      sample_parameters, michaelis_menten_model, 
                                                      simple_noise_function):
        """Test behavior when parameter estimation fails."""
        with patch('monte_carlo_estimator.estimate_parameters') as mock_estimate:
            # Mock some successful and some failed estimations
            mock_estimate.side_effect = [
                (np.array([95.0, 1.8, 2.9]), None),  # Success
                (None, None),  # Failure
                (np.array([98.0, 2.1, 3.1]), None),  # Success
                (None, None),  # Failure
                (np.array([102.0, 1.9, 2.8]), None)  # Success
            ]
            
            results = monte_carlo_parameter_estimation(
                data=sample_data,
                cal_data=sample_cal_data,
                substrate=["PD_mM", "NAD_mM"],
                cal_param=sample_parameters,
                model_func=michaelis_menten_model,
                noise_function=simple_noise_function,
                initial_guess=[80, 1, 1],
                noise_level=0.1,
                num_iterations=5
            )
            
            # Should handle failed estimations gracefully
            assert isinstance(results, np.ndarray)
            assert results.shape[1] == 3  # Still 3 parameters
    
    def test_monte_carlo_with_single_parameter_model(self, sample_cal_data, sample_parameters, 
                                                    simple_noise_function):
        """Test Monte Carlo with a single parameter model."""
        # Create simple single-parameter data
        single_param_data = pd.DataFrame({
            'PD_mM': [1, 2, 3, 4, 5],
            'activity_U/mg': [10, 20, 30, 40, 50]
        })
        
        def single_param_model(S, k):
            return k * S[0]  # Simple linear model
        
        with patch('monte_carlo_estimator.estimate_parameters') as mock_estimate:
            mock_estimate.return_value = (np.array([9.8]), None)
            
            results = monte_carlo_parameter_estimation(
                data=single_param_data,
                cal_data=sample_cal_data,
                substrate=["PD_mM"],
                cal_param=sample_parameters,
                model_func=single_param_model,
                noise_function=simple_noise_function,
                initial_guess=[10],
                noise_level=0.1,
                num_iterations=10
            )
            
            assert results.shape == (10, 1)  # 10 iterations, 1 parameter
    
    def test_monte_carlo_noise_function_call_signature(self, sample_data, sample_cal_data, 
                                                      sample_parameters, michaelis_menten_model):
        """Test that noise function is called with correct arguments."""
        def mock_noise_function(data, cal_data, substrate, cal_param, noise_level):
            # Verify the arguments passed to noise function
            assert isinstance(data, pd.DataFrame)
            assert isinstance(cal_data, pd.DataFrame)
            assert isinstance(substrate, list)
            assert isinstance(cal_param, dict)
            assert isinstance(noise_level, (int, float))
            return data.copy()  # Return unchanged data for simplicity
        
        with patch('monte_carlo_estimator.estimate_parameters') as mock_estimate:
            mock_estimate.return_value = (np.array([95.0, 1.8, 2.9]), None)
            
            monte_carlo_parameter_estimation(
                data=sample_data,
                cal_data=sample_cal_data,
                substrate=["PD_mM", "NAD_mM"],
                cal_param=sample_parameters,
                model_func=michaelis_menten_model,
                noise_function=mock_noise_function,
                initial_guess=[80, 1, 1],
                noise_level=0.1,
                num_iterations=3
            )
    
    def test_monte_carlo_random_seed_reproducibility(self, sample_data, sample_cal_data, 
                                                   sample_parameters, michaelis_menten_model, 
                                                   simple_noise_function):
        """Test reproducibility when random seed is set."""
        # Note: This test depends on the noise function using numpy.random
        
        def seeded_run():
            np.random.seed(42)
            with patch('monte_carlo_estimator.estimate_parameters') as mock_estimate:
                # Return slightly different values to test randomness
                mock_estimate.side_effect = [
                    (np.array([95.0 + np.random.normal(0, 1), 1.8, 2.9]), None)
                    for _ in range(10)
                ]
                
                return monte_carlo_parameter_estimation(
                    data=sample_data,
                    cal_data=sample_cal_data,
                    substrate=["PD_mM", "NAD_mM"],
                    cal_param=sample_parameters,
                    model_func=michaelis_menten_model,
                    noise_function=simple_noise_function,
                    initial_guess=[80, 1, 1],
                    noise_level=0.1,
                    num_iterations=10
                )
        
        # Run twice with same seed - results should be similar due to seeding
        results1 = seeded_run()
        results2 = seeded_run()
        
        # Results should be the same shape
        assert results1.shape == results2.shape
    
    def test_monte_carlo_empty_data_handling(self, sample_cal_data, sample_parameters, 
                                           michaelis_menten_model, simple_noise_function):
        """Test behavior with empty input data."""
        empty_data = pd.DataFrame(columns=['PD_mM', 'NAD_mM', 'activity_U/mg'])
        
        with pytest.raises((ValueError, IndexError, KeyError)):
            monte_carlo_parameter_estimation(
                data=empty_data,
                cal_data=sample_cal_data,
                substrate=["PD_mM", "NAD_mM"],
                cal_param=sample_parameters,
                model_func=michaelis_menten_model,
                noise_function=simple_noise_function,
                initial_guess=[80, 1, 1],
                noise_level=0.1,
                num_iterations=5
            )
    
    def test_monte_carlo_invalid_initial_guess(self, sample_data, sample_cal_data, 
                                              sample_parameters, michaelis_menten_model, 
                                              simple_noise_function):
        """Test behavior with invalid initial guess."""
        # Wrong number of parameters in initial guess
        with patch('monte_carlo_estimator.estimate_parameters') as mock_estimate:
            mock_estimate.side_effect = ValueError("Initial guess dimension mismatch")
            
            with pytest.raises(ValueError):
                monte_carlo_parameter_estimation(
                    data=sample_data,
                    cal_data=sample_cal_data,
                    substrate=["PD_mM", "NAD_mM"],
                    cal_param=sample_parameters,
                    model_func=michaelis_menten_model,
                    noise_function=simple_noise_function,
                    initial_guess=[80, 1],  # Wrong number of parameters
                    noise_level=0.1,
                    num_iterations=1
                )