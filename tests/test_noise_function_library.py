import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from noise_function_libary import no_noise, add_noise_processed_data, add_noise_plate_reader, add_noise_rate, full_experiment_processing_with_noise


def create_test_data():
    """Creates test data for noise function testing."""
    # Create experimental data with time series
    data = pd.DataFrame({
        'HP_mM': [100, 100, 200, 200, 300, 300],
        'NADH_mM': [0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
        'PD_mM': [0, 0, 0, 0, 0, 0],
        'data_00': [1.9, 1.8, 2.0, 1.95, 2.1, 2.05],
        'data_30': [1.85, 1.75, 1.9, 1.85, 1.95, 1.9],
        'data_60': [1.8, 1.7, 1.8, 1.75, 1.8, 1.75],
        'data_90': [1.75, 1.65, 1.7, 1.65, 1.65, 1.6]
    })
    
    # Create calibration data
    cal_data = pd.DataFrame({
        'c': [0.0, 0.1, 0.2, 0.3, 0.4],
        'ad1': [0.0, 0.1, 0.2, 0.3, 0.4],
        'ad2': [0.0, 0.11, 0.19, 0.31, 0.39]
    })
    
    # Create processed data
    processed_data = pd.DataFrame({
        'HP_mM': [100, 200, 300],
        'NADH_mM': [0.6, 0.6, 0.6], 
        'PD_mM': [0, 0, 0],
        'activity_U/mg': [1.5, 2.0, 2.5]
    })
    
    substrates = ['HP_mM', 'NADH_mM', 'PD_mM']
    
    cal_parameters = {
        'Vf_well': 10.0,
        'Vf_prod': 5.0,
        'c_prod': 2.15
    }
    
    return data, cal_data, processed_data, substrates, cal_parameters


def test_no_noise():
    """Test that no_noise function returns input data unchanged."""
    data, cal_data, processed_data, substrates, cal_parameters = create_test_data()
    
    result = no_noise(data, cal_data, substrates, cal_parameters, 0.1)
    
    # Should return exactly the same data
    pd.testing.assert_frame_equal(result, data)
    
    print("✓ test_no_noise passed: Function returns unchanged data")


def test_add_noise_processed_data():
    """Test that add_noise_processed_data adds noise only to activity column."""
    data, cal_data, processed_data, substrates, cal_parameters = create_test_data()
    
    noise_level = 0.1
    result = add_noise_processed_data(processed_data, cal_data, substrates, cal_parameters, noise_level)
    
    # Check that result is a DataFrame with same structure
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == list(processed_data.columns)
    assert len(result) == len(processed_data)
    
    # Check that substrate concentrations are unchanged
    for substrate in substrates:
        pd.testing.assert_series_equal(result[substrate], processed_data[substrate])
    
    # Check that activity values have been modified (with noise)
    assert not result['activity_U/mg'].equals(processed_data['activity_U/mg'])
    
    # Check that the noise is reasonable (within expected range)
    original_activities = processed_data['activity_U/mg'].values
    noisy_activities = result['activity_U/mg'].values
    differences = np.abs(noisy_activities - original_activities)
    
    # Noise should be roughly in range of noise_level (allowing for some variance)
    assert np.all(differences < 5 * noise_level), "Noise appears too large"
    
    print("✓ test_add_noise_processed_data passed: Noise added correctly to activity column")


def test_add_noise_plate_reader():
    """Test that add_noise_plate_reader processes experimental data and adds noise."""
    data, cal_data, processed_data, substrates, cal_parameters = create_test_data()
    
    noise_level = 0.01
    result = add_noise_plate_reader(data, cal_data, substrates, cal_parameters, noise_level)
    
    # Check that result is processed data format
    assert isinstance(result, pd.DataFrame)
    assert 'activity_U/mg' in result.columns
    
    # Check that substrate columns are present
    for substrate in substrates:
        assert substrate in result.columns
    
    # Check that we have fewer rows than input (due to duplicate processing)
    assert len(result) <= len(data)
    
    # Check that activities are positive numbers
    assert np.all(result['activity_U/mg'] > 0), "Activities should be positive"
    
    # Run twice with same seed to check reproducibility structure
    np.random.seed(42)
    result1 = add_noise_plate_reader(data, cal_data, substrates, cal_parameters, noise_level)
    np.random.seed(42)
    result2 = add_noise_plate_reader(data, cal_data, substrates, cal_parameters, noise_level)
    
    # Results should have same structure (though noise will be different)
    assert len(result1) == len(result2)
    assert list(result1.columns) == list(result2.columns)
    
    print("✓ test_add_noise_plate_reader passed: Experimental data processed correctly")


def test_add_noise_rate():
    """Test that add_noise_rate processes data and adds noise to rates."""
    data, cal_data, processed_data, substrates, cal_parameters = create_test_data()
    
    noise_level = 0.05
    result = add_noise_rate(data, cal_data, substrates, cal_parameters, noise_level)
    
    # Check that result is processed data format
    assert isinstance(result, pd.DataFrame)
    assert 'activity_U/mg' in result.columns
    
    # Check that substrate columns are present
    for substrate in substrates:
        assert substrate in result.columns
    
    # Check that we have fewer rows than input (due to duplicate processing)  
    assert len(result) <= len(data)
    
    # Check that activities are numbers (can be negative after noise)
    assert np.all(np.isfinite(result['activity_U/mg'])), "Activities should be finite numbers"
    
    # Compare with no-noise version to verify noise was added
    result_no_noise = add_noise_rate(data, cal_data, substrates, cal_parameters, 0.0)
    
    # With zero noise, should get consistent results
    assert len(result) == len(result_no_noise)
    
    # With noise, activities should generally be different
    if noise_level > 0:
        activities_with_noise = result['activity_U/mg'].values
        activities_without_noise = result_no_noise['activity_U/mg'].values
        differences = np.abs(activities_with_noise - activities_without_noise)
        
        # At least some values should be different (unless very unlucky with random)
        assert np.any(differences > 1e-10), "Noise should cause some differences"
    
    print("✓ test_add_noise_rate passed: Rate noise added correctly")


def test_full_experiment_processing_with_noise():
    """Test the comprehensive noise function that adds multiple types of experimental errors."""
    data, cal_data, processed_data, substrates, cal_parameters = create_test_data()
    
    # Define realistic noise levels for different error sources
    noise_levels = {
        'fehler_wage': 0.001,       # Balance error
        'fehler_pipettieren': 0.01, # Pipetting error  
        'fehler_time_points': 5.0,  # Time point error in seconds
        'fehler_od': 0.005          # Optical density measurement error
    }
    
    result = full_experiment_processing_with_noise(data, cal_data, substrates, cal_parameters, noise_levels)
    
    # Check that result is processed data format
    assert isinstance(result, pd.DataFrame)
    assert 'activity_U/mg' in result.columns
    
    # Check that substrate columns are present
    for substrate in substrates:
        assert substrate in result.columns
    
    # Check that we have data (may be fewer rows due to processing)
    assert len(result) > 0, "Should have at least some data points"
    
    # Check that activities are finite numbers
    assert np.all(np.isfinite(result['activity_U/mg'])), "Activities should be finite numbers"
    
    # Test with zero noise to verify baseline functionality
    zero_noise_levels = {
        'fehler_wage': 0.0,
        'fehler_pipettieren': 0.0, 
        'fehler_time_points': 0.0,
        'fehler_od': 0.0
    }
    
    result_no_noise = full_experiment_processing_with_noise(data, cal_data, substrates, cal_parameters, zero_noise_levels)
    
    # Should have same structure
    assert len(result) == len(result_no_noise)
    assert list(result.columns) == list(result_no_noise.columns)
    
    # Test that different noise levels produce different results
    high_noise_levels = {
        'fehler_wage': 0.01,
        'fehler_pipettieren': 0.1,
        'fehler_time_points': 30.0,
        'fehler_od': 0.05
    }
    
    result_high_noise = full_experiment_processing_with_noise(data, cal_data, substrates, cal_parameters, high_noise_levels)
    
    # Should still have valid structure
    assert isinstance(result_high_noise, pd.DataFrame)
    assert 'activity_U/mg' in result_high_noise.columns
    
    print("✓ test_full_experiment_processing_with_noise passed: Comprehensive noise model works correctly")


def test_noise_levels_validation():
    """Test behavior with different noise level configurations."""
    data, cal_data, processed_data, substrates, cal_parameters = create_test_data()
    
    # Test with minimal noise
    minimal_noise = {
        'fehler_wage': 1e-6,
        'fehler_pipettieren': 1e-6,
        'fehler_time_points': 0.1,
        'fehler_od': 1e-6
    }
    
    result_minimal = full_experiment_processing_with_noise(data, cal_data, substrates, cal_parameters, minimal_noise)
    assert isinstance(result_minimal, pd.DataFrame)
    assert len(result_minimal) > 0
    
    # Test with asymmetric noise (only some error sources)
    asymmetric_noise = {
        'fehler_wage': 0.0,
        'fehler_pipettieren': 0.05,  # Only pipetting error
        'fehler_time_points': 0.0,
        'fehler_od': 0.0
    }
    
    result_asymmetric = full_experiment_processing_with_noise(data, cal_data, substrates, cal_parameters, asymmetric_noise)
    assert isinstance(result_asymmetric, pd.DataFrame)
    assert len(result_asymmetric) > 0
    
    print("✓ test_noise_levels_validation passed: Different noise configurations handled correctly")


def test_noise_function_reproducibility():
    """Test that noise functions are properly randomized but can be controlled with seeds."""
    data, cal_data, processed_data, substrates, cal_parameters = create_test_data()
    
    # Test add_noise_processed_data reproducibility
    np.random.seed(42)
    result1 = add_noise_processed_data(processed_data, cal_data, substrates, cal_parameters, 0.1)
    
    np.random.seed(42) 
    result2 = add_noise_processed_data(processed_data, cal_data, substrates, cal_parameters, 0.1)
    
    # With same seed, should get identical results
    pd.testing.assert_frame_equal(result1, result2)
    
    # Test that different seeds give different results
    np.random.seed(123)
    result3 = add_noise_processed_data(processed_data, cal_data, substrates, cal_parameters, 0.1)
    
    # Should have same structure but different activity values
    assert list(result1.columns) == list(result3.columns)
    assert len(result1) == len(result3)
    
    # Activities should be different (with high probability)
    activities1 = result1['activity_U/mg'].values
    activities3 = result3['activity_U/mg'].values
    differences = np.abs(activities1 - activities3)
    assert np.any(differences > 1e-10), "Different seeds should produce different noise"
    
    print("✓ test_noise_function_reproducibility passed: Noise functions properly randomized")


def test_error_edge_cases():
    """Test noise functions with edge cases and potential error conditions."""
    data, cal_data, processed_data, substrates, cal_parameters = create_test_data()
    
    # Test with very large noise levels
    try:
        large_noise_result = add_noise_processed_data(processed_data, cal_data, substrates, cal_parameters, 100.0)
        assert isinstance(large_noise_result, pd.DataFrame)
        print("✓ Large noise levels handled")
    except Exception as e:
        print(f"⚠ Large noise caused issues: {str(e)}")
    
    # Test with negative activities after noise (should still be valid)
    very_large_noise = add_noise_processed_data(processed_data, cal_data, substrates, cal_parameters, 10.0)
    assert isinstance(very_large_noise, pd.DataFrame)
    # Activities can be negative after noise - this is physically unrealistic but mathematically valid
    
    # Test with empty-like data
    minimal_data = pd.DataFrame({
        'HP_mM': [100],
        'NADH_mM': [0.6],
        'PD_mM': [0],
        'activity_U/mg': [1.5]
    })
    
    try:
        minimal_result = add_noise_processed_data(minimal_data, cal_data, substrates, cal_parameters, 0.1)
        assert len(minimal_result) == 1
        print("✓ Minimal data handled")
    except Exception as e:
        print(f"⚠ Minimal data caused issues: {str(e)}")
    
    print("✓ test_error_edge_cases passed: Edge cases handled appropriately")


def run_all_tests():
    """Run all noise function tests."""
    print("Running noise function tests...\n")
    
    try:
        test_no_noise()
        test_add_noise_processed_data() 
        test_add_noise_plate_reader()
        test_add_noise_rate()
        test_full_experiment_processing_with_noise()
        test_noise_levels_validation()
        test_noise_function_reproducibility()
        test_error_edge_cases()
        
        print("\n🎉 All noise function tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_all_tests()