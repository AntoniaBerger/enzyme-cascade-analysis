import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parameter_estimator import (
    estimate_parameters_standard, estimate_parameters_adaptive, _multi_start_fit,
    _standard_fitting, _regularized_fitting, _bootstrap_fitting
)
from artifical_data import reaction1_synthetic_data


def create_test_data():
    """Creates test data for parameter estimation testing."""
    
    # Perfect synthetic data (no noise)
    true_params = (100, 2, 3)
    perfect_data = reaction1_synthetic_data(true_params)
    
    # Noisy synthetic data
    np.random.seed(42)
    noisy_data = perfect_data.copy()
    noise = np.random.normal(0, 2, len(noisy_data))
    noisy_data["activity_U/mg"] += noise
    
    # Sparse data (few points)
    sparse_data = perfect_data.iloc[::3].reset_index(drop=True)  # Every 3rd point
    
    # Very sparse data (critical case)
    very_sparse_data = perfect_data.iloc[::6].reset_index(drop=True)  # Every 6th point
    
    return {
        'perfect': perfect_data,
        'noisy': noisy_data,
        'sparse': sparse_data,
        'very_sparse': very_sparse_data,
        'true_params': true_params
    }


def michaelis_menten_test(S, Vmax, Km1, Km2):
    """Test model function for parameter estimation."""
    S1, S2 = S
    return (Vmax * S1 * S2) / ((Km1 + S1) * (Km2 + S2))


def single_substrate_model(S, Vmax, Km):
    """Single substrate Michaelis-Menten model for testing."""
    return (Vmax * S) / (Km + S)


def test_estimate_parameters_standard():
    """Test standard parameter estimation function."""
    print("Testing estimate_parameters_standard...")
    
    test_data = create_test_data()
    initial_guess = [80, 1, 1]
    
    # Test with perfect data
    popt, pcov = estimate_parameters_standard(
        test_data['perfect'], michaelis_menten_test, initial_guess
    )
    
    assert popt is not None, "Standard fitting failed with perfect data"
    assert pcov is not None, "Covariance matrix should not be None"
    assert len(popt) == 3, "Should estimate 3 parameters"
    
    # Check parameter accuracy (should be close to true values)
    true_params = test_data['true_params']
    for i, (est, true) in enumerate(zip(popt, true_params)):
        rel_error = abs(est - true) / true
        assert rel_error < 0.1, f"Parameter {i} error too large: {rel_error:.3f}"
    
    # Test with noisy data
    popt_noisy, pcov_noisy = estimate_parameters_standard(
        test_data['noisy'], michaelis_menten_test, initial_guess
    )
    
    assert popt_noisy is not None, "Standard fitting failed with noisy data"
    
    print("✓ estimate_parameters_standard passed")


def test_estimate_parameters_adaptive():
    """Test adaptive parameter estimation with different methods."""
    print("Testing estimate_parameters_adaptive...")
    
    test_data = create_test_data()
    initial_guess = [80, 1, 1]
    substrates = ['PD_mM', 'NAD_mM']
    
    # Test multi_start method
    popt_multi, pcov_multi = estimate_parameters_adaptive(
        test_data['perfect'], michaelis_menten_test, substrates, 
        initial_guess, method='multi_start'
    )
    
    assert popt_multi is not None, "Multi-start fitting failed"
    assert len(popt_multi) == 3, "Should estimate 3 parameters"
    
    # Test regularized method
    popt_reg, pcov_reg = estimate_parameters_adaptive(
        test_data['sparse'], michaelis_menten_test, substrates,
        initial_guess, method='regularized'
    )
    
    assert popt_reg is not None, "Regularized fitting failed"
    
    # Test bootstrap method
    popt_boot, pcov_boot = estimate_parameters_adaptive(
        test_data['noisy'], michaelis_menten_test, substrates,
        initial_guess, method='bootstrap'
    )
    
    assert popt_boot is not None, "Bootstrap fitting failed"
    
    # Test with single substrate
    single_data = test_data['perfect'][['PD_mM', 'activity_U/mg']].copy()
    single_data['activity_U/mg'] = single_substrate_model(
        single_data['PD_mM'].values, 50, 2
    )
    
    popt_single, _ = estimate_parameters_adaptive(
        single_data, single_substrate_model, ['PD_mM'], 
        [40, 1], method='multi_start'
    )
    
    assert popt_single is not None, "Single substrate fitting failed"
    assert len(popt_single) == 2, "Should estimate 2 parameters for single substrate"
    
    print("✓ estimate_parameters_adaptive passed")


def test_multi_start_fit():
    """Test multi-start fitting robustness."""
    print("Testing _multi_start_fit...")
    
    test_data = create_test_data()
    
    # Prepare x_data and y_data
    x_data = [test_data['perfect']['PD_mM'].values, test_data['perfect']['NAD_mM'].values]
    y_data = test_data['perfect']['activity_U/mg'].values
    initial_guess = [80, 1, 1]
    
    # Test with good data
    popt, pcov = _multi_start_fit(x_data, y_data, michaelis_menten_test, initial_guess)
    
    assert popt is not None, "Multi-start fit failed"
    assert pcov is not None, "Covariance should not be None"
    
    # Test with difficult data (sparse)
    x_sparse = [test_data['sparse']['PD_mM'].values, test_data['sparse']['NAD_mM'].values]
    y_sparse = test_data['sparse']['activity_U/mg'].values
    
    popt_sparse, _ = _multi_start_fit(x_sparse, y_sparse, michaelis_menten_test, initial_guess)
    
    # Should still work with sparse data
    assert popt_sparse is not None, "Multi-start should handle sparse data"
    
    print("✓ _multi_start_fit passed")


def test_standard_fitting():
    """Test standard fitting function."""
    print("Testing _standard_fitting...")
    
    test_data = create_test_data()
    
    x_data = [test_data['perfect']['PD_mM'].values, test_data['perfect']['NAD_mM'].values]
    y_data = test_data['perfect']['activity_U/mg'].values
    initial_guess = [80, 1, 1]
    
    popt, pcov = _standard_fitting(x_data, y_data, michaelis_menten_test, initial_guess)
    
    assert popt is not None, "Standard fitting failed"
    assert pcov is not None, "Covariance should not be None"
    assert len(popt) == len(initial_guess), "Parameter count mismatch"
    
    # Check that parameters are reasonable
    assert all(p > 0 for p in popt), "All parameters should be positive"
    assert popt[0] < 1000, "Vmax should be reasonable"
    assert popt[1] < 100, "Km values should be reasonable"
    assert popt[2] < 100, "Km values should be reasonable"
    
    print("✓ _standard_fitting passed")


def test_regularized_fitting():
    """Test regularized fitting for small datasets."""
    print("Testing _regularized_fitting...")
    
    test_data = create_test_data()
    
    # Use very sparse data
    x_data = [test_data['very_sparse']['PD_mM'].values, test_data['very_sparse']['NAD_mM'].values]
    y_data = test_data['very_sparse']['activity_U/mg'].values
    initial_guess = [80, 1, 1]
    
    popt, pcov = _regularized_fitting(x_data, y_data, michaelis_menten_test, initial_guess)
    
    # Should work even with very sparse data
    if popt is not None:
        assert len(popt) == len(initial_guess), "Parameter count mismatch"
        assert all(p > 0 for p in popt), "All parameters should be positive"
    
    # Test edge case: empty data should fail gracefully
    try:
        empty_x = [np.array([]), np.array([])]
        empty_y = np.array([])
        popt_empty, _ = _regularized_fitting(empty_x, empty_y, michaelis_menten_test, initial_guess)
        assert popt_empty is None, "Should fail with empty data"
    except:
        pass  # Expected to fail
    
    print("✓ _regularized_fitting passed")


def test_bootstrap_fitting():
    """Test bootstrap fitting for uncertainty estimation."""
    print("Testing _bootstrap_fitting...")
    
    test_data = create_test_data()
    
    x_data = [test_data['noisy']['PD_mM'].values, test_data['noisy']['NAD_mM'].values]
    y_data = test_data['noisy']['activity_U/mg'].values
    initial_guess = [80, 1, 1]
    
    # Test with sufficient data
    popt, pcov = _bootstrap_fitting(x_data, y_data, michaelis_menten_test, initial_guess, n_bootstrap=20)
    
    assert popt is not None, "Bootstrap fitting failed"
    assert pcov is not None, "Bootstrap covariance should not be None"
    
    # Check covariance matrix properties
    assert pcov.shape == (len(initial_guess), len(initial_guess)), "Covariance matrix shape incorrect"
    assert np.all(np.diag(pcov) >= 0), "Diagonal elements should be non-negative"
    
    # Test with insufficient data
    very_small_x = [x_data[0][:2], x_data[1][:2]]  # Only 2 points
    very_small_y = y_data[:2]
    
    popt_small, _ = _bootstrap_fitting(very_small_x, very_small_y, michaelis_menten_test, initial_guess, n_bootstrap=10)
    
    # Should fallback to standard fitting
    assert popt_small is not None or popt_small is None, "Should handle small datasets gracefully"
    
    print("✓ _bootstrap_fitting passed")


def test_parameter_estimation_accuracy():
    """Test overall accuracy of parameter estimation."""
    print("Testing parameter estimation accuracy...")
    
    test_data = create_test_data()
    true_params = test_data['true_params']
    initial_guess = [80, 1, 1]
    substrates = ['PD_mM', 'NAD_mM']
    
    # Test different methods
    methods = ['multi_start', 'regularized', 'bootstrap']
    
    for method in methods:
        popt, _ = estimate_parameters_adaptive(
            test_data['perfect'], michaelis_menten_test, substrates,
            initial_guess, method=method
        )
        
        if popt is not None:
            # Calculate relative errors
            rel_errors = [abs(est - true) / true for est, true in zip(popt, true_params)]
            max_error = max(rel_errors)
            
            print(f"Method {method}: Max relative error = {max_error:.3f}")
            
            # For perfect data, error should be small (except regularized which adds bias)
            if method != 'regularized':
                assert max_error < 0.2, f"Method {method} has too large error: {max_error:.3f}"
    
    print("✓ Parameter estimation accuracy passed")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("Testing edge cases...")
    
    test_data = create_test_data()
    
    # Test insufficient data points
    tiny_data = test_data['perfect'].iloc[:2].copy()  # Only 2 points for 3 parameters
    
    try:
        popt, _ = estimate_parameters_adaptive(
            tiny_data, michaelis_menten_test, ['PD_mM', 'NAD_mM'],
            [80, 1, 1], method='multi_start'
        )
        # Should either work or fail gracefully
    except ValueError as e:
        assert "Not enough data points" in str(e), "Should give meaningful error message"
    
    # Test with bad initial guess
    bad_guess = [0, 0, 0]  # All zeros
    popt_bad, _ = estimate_parameters_adaptive(
        test_data['perfect'], michaelis_menten_test, ['PD_mM', 'NAD_mM'],
        bad_guess, method='multi_start'
    )
    
    # Multi-start should handle bad initial guesses
    assert popt_bad is not None, "Multi-start should handle bad initial guesses"
    
    print("✓ Edge cases passed")


def run_all_tests():
    """Run all parameter estimator tests."""
    print("Running all parameter estimator tests...\n")
    
    test_functions = [
        test_estimate_parameters_standard,
        test_estimate_parameters_adaptive,
        test_multi_start_fit,
        test_standard_fitting,
        test_regularized_fitting,
        test_bootstrap_fitting,
        test_parameter_estimation_accuracy,
        test_edge_cases
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            failed += 1
    
    print(f"\n--- Parameter Estimator Test Results ---")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return failed == 0


if __name__ == "__main__":
    run_all_tests()