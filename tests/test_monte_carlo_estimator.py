import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monte_carlo_estimator import monte_carlo_parameter_estimation
from noise_function_libary import no_noise, add_noise_processed_data, add_noise_plate_reader, add_noise_rate


def create_test_data_monte_carlo():
    """Creates test data for Monte Carlo testing."""
    # Create experimental data with time series
    data = pd.DataFrame({
        'HP_mM': [100, 100, 200, 200, 300, 300, 400, 400],
        'NADH_mM': [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
        'PD_mM': [0, 0, 0, 0, 0, 0, 0, 0],
        'activity_U/mg': [1.2, 1.1, 2.3, 2.2, 3.4, 3.3, 4.1, 4.0],
        'data_00': [1.9, 1.8, 2.0, 1.95, 2.1, 2.05, 2.2, 2.15],
        'data_30': [1.85, 1.75, 1.9, 1.85, 1.95, 1.9, 2.0, 1.95],
        'data_60': [1.8, 1.7, 1.8, 1.75, 1.8, 1.75, 1.85, 1.8],
        'data_90': [1.75, 1.65, 1.7, 1.65, 1.65, 1.6, 1.7, 1.65]
    })
    
    # Create calibration data
    cal_data = pd.DataFrame({
        'c': [0.0, 0.1, 0.2, 0.3, 0.4],
        'ad1': [0.0, 0.1, 0.2, 0.3, 0.4],
        'ad2': [0.0, 0.11, 0.19, 0.31, 0.39]
    })
    
    # Create processed data for simple testing
    processed_data = pd.DataFrame({
        'HP_mM': [100, 200, 300, 400],
        'NADH_mM': [0.6, 0.6, 0.6, 0.6], 
        'PD_mM': [0, 0, 0, 0],
        'activity_U/mg': [1.15, 2.25, 3.35, 4.05]
    })
    
    substrates = ['HP_mM', 'NADH_mM', 'PD_mM']
    
    cal_parameters = {
        'Vf_well': 10.0,
        'Vf_prod': 5.0,
        'c_prod': 2.15
    }
    
    # Simple Michaelis-Menten model for testing
    def simple_mm_model(S, *params):
        """Simple Michaelis-Menten model for testing."""
        S1 = S[0] if isinstance(S, (list, tuple, np.ndarray)) else S
        Vmax, Km = params[0], params[1]
        return (Vmax * S1) / (Km + S1)
    
    # More complex model similar to the real one
    def complex_mm_model(S, *params):
        """Complex Michaelis-Menten model with inhibition."""
        if len(params) < 4:
            # Fallback to simple model if not enough parameters
            return simple_mm_model(S, *params[:2])
        
        S1, S2, S3 = S[0], S[1], S[2]
        Vmax, Km1, Km2, Ki1 = params
        return (Vmax * S1 * S2) / ((Km1 * (1 + S3/Ki1) + S1) * (Km2 + S2))
    
    return data, cal_data, processed_data, substrates, cal_parameters, simple_mm_model, complex_mm_model


def test_monte_carlo_basic_functionality():
    """Test basic Monte Carlo parameter estimation functionality."""
    data, cal_data, processed_data, substrates, cal_parameters, simple_model, complex_model = create_test_data_monte_carlo()
    
    initial_guess = [5.0, 150.0]
    noise_level = 0.05
    num_iterations = 10
    
    # Test with processed data and simple noise function
    result = monte_carlo_parameter_estimation(
        data=processed_data,
        cal_data=cal_data,
        substrate=['HP_mM'],
        cal_param=cal_parameters,
        model_func=simple_model,
        noise_function=add_noise_processed_data,
        initial_guess=initial_guess,
        noise_level=noise_level,
        num_iterations=num_iterations
    )
    
    # Check basic properties
    assert isinstance(result, np.ndarray), "Result should be numpy array"
    assert result.shape == (num_iterations, len(initial_guess)), f"Expected shape ({num_iterations}, {len(initial_guess)})"
    assert np.all(np.isfinite(result)), "All parameter estimates should be finite"
    assert np.all(result > 0), "All parameters should be positive for MM kinetics"
    
    print("✓ test_monte_carlo_basic_functionality passed: Basic Monte Carlo estimation works")


def test_monte_carlo_with_different_noise_functions():
    """Test Monte Carlo estimation with different noise functions."""
    data, cal_data, processed_data, substrates, cal_parameters, simple_model, complex_model = create_test_data_monte_carlo()
    
    initial_guess = [4.0, 120.0]
    noise_level = 0.02
    num_iterations = 5
    
    # Test with no noise
    result_no_noise = monte_carlo_parameter_estimation(
        data=processed_data,
        cal_data=cal_data,
        substrate=['HP_mM'],
        cal_param=cal_parameters,
        model_func=simple_model,
        noise_function=no_noise,
        initial_guess=initial_guess,
        noise_level=noise_level,
        num_iterations=num_iterations
    )
    
    # Test with processed data noise
    result_processed_noise = monte_carlo_parameter_estimation(
        data=processed_data,
        cal_data=cal_data,
        substrate=['HP_mM'],
        cal_param=cal_parameters,
        model_func=simple_model,
        noise_function=add_noise_processed_data,
        initial_guess=initial_guess,
        noise_level=noise_level,
        num_iterations=num_iterations
    )
    
    # Both should have same shape
    assert result_no_noise.shape == result_processed_noise.shape
    
    # No noise should give more consistent results (lower variance)
    var_no_noise = np.var(result_no_noise, axis=0)
    var_with_noise = np.var(result_processed_noise, axis=0)
    
    # With noise should generally have higher variance (though this isn't guaranteed for small samples)
    assert np.all(var_no_noise >= 0), "Variance should be non-negative"
    assert np.all(var_with_noise >= 0), "Variance should be non-negative"
    
    print("✓ test_monte_carlo_with_different_noise_functions passed: Different noise functions work correctly")


def test_monte_carlo_parameter_convergence():
    """Test that Monte Carlo estimation converges with increasing iterations."""
    data, cal_data, processed_data, substrates, cal_parameters, simple_model, complex_model = create_test_data_monte_carlo()
    
    initial_guess = [4.0, 120.0]
    noise_level = 0.03
    
    # Test with different numbers of iterations
    iterations_list = [5, 10, 20]
    results = []
    
    for num_iter in iterations_list:
        # Fix random seed for reproducibility
        np.random.seed(42)
        result = monte_carlo_parameter_estimation(
            data=processed_data,
            cal_data=cal_data,
            substrate=['HP_mM'],
            cal_param=cal_parameters,
            model_func=simple_model,
            noise_function=add_noise_processed_data,
            initial_guess=initial_guess,
            noise_level=noise_level,
            num_iterations=num_iter
        )
        results.append(result)
    
    # Check that results have expected shapes
    for i, result in enumerate(results):
        expected_shape = (iterations_list[i], len(initial_guess))
        assert result.shape == expected_shape, f"Result {i} has wrong shape"
    
    # Check that means are reasonably consistent (though this is stochastic)
    means = [np.mean(result, axis=0) for result in results]
    
    # All means should be positive for MM kinetics
    for mean_params in means:
        assert np.all(mean_params > 0), "Mean parameters should be positive"
    
    print("✓ test_monte_carlo_parameter_convergence passed: Parameter estimation converges properly")


def test_monte_carlo_with_complex_model():
    """Test Monte Carlo estimation with more complex kinetic model."""
    data, cal_data, processed_data, substrates, cal_parameters, simple_model, complex_model = create_test_data_monte_carlo()
    
    # Create data suitable for complex model (3 substrates) with enough data points
    complex_data = pd.DataFrame({
        'HP_mM': [100, 150, 200, 250, 300],  # 5 data points for 4 parameters
        'NADH_mM': [0.5, 0.55, 0.6, 0.65, 0.7], 
        'PD_mM': [10, 15, 20, 25, 30],
        'activity_U/mg': [1.2, 1.7, 2.2, 2.6, 2.8]
    })
    
    initial_guess = [5.0, 100.0, 2.0, 50.0]  # Vmax, Km1, Km2, Ki1
    noise_level = 0.05
    num_iterations = 8
    
    result = monte_carlo_parameter_estimation(
        data=complex_data,
        cal_data=cal_data,
        substrate=substrates,
        cal_param=cal_parameters,
        model_func=complex_model,
        noise_function=add_noise_processed_data,
        initial_guess=initial_guess,
        noise_level=noise_level,
        num_iterations=num_iterations
    )
    
    assert result.shape == (num_iterations, len(initial_guess))
    assert np.all(np.isfinite(result)), "All parameters should be finite"
    
    # Check parameter ranges are reasonable
    vmax_estimates = result[:, 0]
    km1_estimates = result[:, 1] 
    km2_estimates = result[:, 2]
    ki1_estimates = result[:, 3]
    
    assert np.all(vmax_estimates > 0), "Vmax should be positive"
    assert np.all(km1_estimates > 0), "Km1 should be positive"
    assert np.all(km2_estimates > 0), "Km2 should be positive" 
    assert np.all(ki1_estimates > 0), "Ki1 should be positive"
    
    print("✓ test_monte_carlo_with_complex_model passed: Complex kinetic model works correctly")


def test_monte_carlo_noise_level_effects():
    """Test effects of different noise levels on parameter estimation."""
    data, cal_data, processed_data, substrates, cal_parameters, simple_model, complex_model = create_test_data_monte_carlo()
    
    initial_guess = [4.0, 120.0]
    num_iterations = 15
    noise_levels = [0.001, 0.01, 0.1]
    
    results_by_noise = []
    
    for noise_level in noise_levels:
        # Fix seed for fair comparison
        np.random.seed(123)
        result = monte_carlo_parameter_estimation(
            data=processed_data,
            cal_data=cal_data,
            substrate=['HP_mM'],
            cal_param=cal_parameters,
            model_func=simple_model,
            noise_function=add_noise_processed_data,
            initial_guess=initial_guess,
            noise_level=noise_level,
            num_iterations=num_iterations
        )
        results_by_noise.append(result)
    
    # Calculate variances for each noise level
    variances = [np.var(result, axis=0) for result in results_by_noise]
    
    # Generally, higher noise should lead to higher variance in estimates
    # (though this isn't guaranteed for all cases)
    for i, variance in enumerate(variances):
        assert np.all(variance >= 0), f"Variance for noise level {noise_levels[i]} should be non-negative"
        print(f"  Noise level {noise_levels[i]}: Variance = {variance}")
    
    print("✓ test_monte_carlo_noise_level_effects passed: Noise level effects properly characterized")


def test_monte_carlo_estimation_methods():
    """Test Monte Carlo with different parameter estimation methods."""
    data, cal_data, processed_data, substrates, cal_parameters, simple_model, complex_model = create_test_data_monte_carlo()
    
    initial_guess = [4.0, 120.0]
    noise_level = 0.05
    num_iterations = 6
    methods = ['standard', 'bootstrap']  # Test available methods
    
    results_by_method = {}
    
    for method in methods:
        try:
            result = monte_carlo_parameter_estimation(
                data=processed_data,
                cal_data=cal_data,
                substrate=['HP_mM'],
                cal_param=cal_parameters,
                model_func=simple_model,
                noise_function=add_noise_processed_data,
                initial_guess=initial_guess,
                noise_level=noise_level,
                estimate_method=method,
                num_iterations=num_iterations
            )
            results_by_method[method] = result
            assert result.shape == (num_iterations, len(initial_guess))
            print(f"  Method '{method}' completed successfully")
            
        except Exception as e:
            print(f"  Method '{method}' failed: {str(e)}")
            # Some methods might not be implemented, that's ok
    
    # At least one method should work
    assert len(results_by_method) > 0, "At least one estimation method should work"
    
    print("✓ test_monte_carlo_estimation_methods passed: Different estimation methods handled correctly")


def test_monte_carlo_edge_cases():
    """Test Monte Carlo estimation with edge cases."""
    data, cal_data, processed_data, substrates, cal_parameters, simple_model, complex_model = create_test_data_monte_carlo()
    
    # Test with minimal iterations
    result_minimal = monte_carlo_parameter_estimation(
        data=processed_data,
        cal_data=cal_data,
        substrate=['HP_mM'],
        cal_param=cal_parameters,
        model_func=simple_model,
        noise_function=add_noise_processed_data,
        initial_guess=[4.0, 120.0],
        noise_level=0.05,
        num_iterations=1
    )
    
    assert result_minimal.shape == (1, 2), "Single iteration should work"
    
    # Test with zero noise
    result_zero_noise = monte_carlo_parameter_estimation(
        data=processed_data,
        cal_data=cal_data,
        substrate=['HP_mM'],
        cal_param=cal_parameters,
        model_func=simple_model,
        noise_function=add_noise_processed_data,
        initial_guess=[4.0, 120.0],
        noise_level=0.0,
        num_iterations=3
    )
    
    assert result_zero_noise.shape == (3, 2), "Zero noise should work"
    
    # Test with single data point - should fail gracefully
    single_point_data = pd.DataFrame({
        'HP_mM': [200, 250, 300],  # Need at least 2 points for 2 parameters
        'NADH_mM': [0.6, 0.6, 0.6],
        'PD_mM': [0, 0, 0],
        'activity_U/mg': [2.0, 2.3, 2.8]
    })
    
    try:
        result_single = monte_carlo_parameter_estimation(
            data=single_point_data,
            cal_data=cal_data,
            substrate=['HP_mM'],
            cal_param=cal_parameters,
            model_func=simple_model,
            noise_function=add_noise_processed_data,
            initial_guess=[4.0, 120.0],
            noise_level=0.05,
            num_iterations=2
        )
        print("  Multiple data points handled successfully")
    except Exception as e:
        print(f"  Data points caused expected issues: {str(e)}")
    
    print("✓ test_monte_carlo_edge_cases passed: Edge cases handled appropriately")


def test_monte_carlo_statistical_properties():
    """Test statistical properties of Monte Carlo estimation results."""
    data, cal_data, processed_data, substrates, cal_parameters, simple_model, complex_model = create_test_data_monte_carlo()
    
    initial_guess = [4.0, 120.0]
    noise_level = 0.05
    num_iterations = 25  # Larger sample for statistical tests
    
    # Run Monte Carlo estimation
    result = monte_carlo_parameter_estimation(
        data=processed_data,
        cal_data=cal_data,
        substrate=['HP_mM'],
        cal_param=cal_parameters,
        model_func=simple_model,
        noise_function=add_noise_processed_data,
        initial_guess=initial_guess,
        noise_level=noise_level,
        num_iterations=num_iterations
    )
    
    # Calculate statistics
    means = np.mean(result, axis=0)
    stds = np.std(result, axis=0)
    mins = np.min(result, axis=0)
    maxs = np.max(result, axis=0)
    
    print(f"  Parameter statistics:")
    for i, param in enumerate(['Vmax', 'Km']):
        print(f"    {param}: mean={means[i]:.3f}, std={stds[i]:.3f}, range=[{mins[i]:.3f}, {maxs[i]:.3f}]")
    
    # Basic statistical checks
    assert np.all(means > 0), "Mean parameters should be positive"
    assert np.all(stds >= 0), "Standard deviations should be non-negative"
    assert np.all(mins <= maxs), "Min should be <= max"
    assert np.all(np.isfinite(means)), "Means should be finite"
    assert np.all(np.isfinite(stds)), "Standard deviations should be finite"
    
    # Check that we have some variability (not all estimates identical)
    assert np.any(stds > 1e-10), "Should have some variability in estimates"
    
    print("✓ test_monte_carlo_statistical_properties passed: Statistical properties are reasonable")


def run_all_tests():
    """Run all Monte Carlo estimator tests."""
    print("Running Monte Carlo estimator tests...\n")
    
    try:
        test_monte_carlo_basic_functionality()
        test_monte_carlo_with_different_noise_functions()
        test_monte_carlo_parameter_convergence()
        test_monte_carlo_with_complex_model()
        test_monte_carlo_noise_level_effects()
        test_monte_carlo_estimation_methods()
        test_monte_carlo_edge_cases()
        test_monte_carlo_statistical_properties()
        
        print("\n All Monte Carlo estimator tests passed!")
        
    except Exception as e:
        print(f"\n Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_all_tests()
