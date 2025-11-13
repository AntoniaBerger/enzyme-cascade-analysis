import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from noise_function_libary import no_noise, add_noise_processed_data, add_noise_plate_reader, add_noise_rate


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


def run_all_tests():
    """Run all noise function tests."""
    print("Running noise function tests...\n")
    
    try:
        test_no_noise()
        test_add_noise_processed_data() 
        test_add_noise_plate_reader()
        test_add_noise_rate()
        
        print("\n🎉 All noise function tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_all_tests()