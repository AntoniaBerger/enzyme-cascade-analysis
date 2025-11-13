import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from process_data import (
    is_linear, get_time_points, get_concentration_data, get_absorbance_data,
    get_calibration_slope, get_reaction_slope, convert_ad_to_concentration,
    process_duplicates, process_duplicates2, get_processed_data, add_noise
)


def create_test_data():
    """Creates comprehensive test data for process_data functions."""
    # Create experimental data with time series
    data = pd.DataFrame({
        'Time_s': [0, 0, 30, 30, 60, 60],
        'HP_mM': [100, 100, 200, 200, 300, 300],
        'NADH_mM': [0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
        'PD_mM': [0, 0, 0, 0, 0, 0],
        'data_00': [2.0, 1.95, 2.1, 2.05, 2.2, 2.15],
        'data_30': [1.9, 1.85, 1.95, 1.9, 2.0, 1.95],
        'data_60': [1.8, 1.75, 1.8, 1.75, 1.8, 1.75],
        'data_90': [1.7, 1.65, 1.65, 1.6, 1.6, 1.55],
        'data_120': [1.6, 1.55, 1.5, 1.45, 1.4, 1.35]
    })
    
    # Create calibration data with clear linear relationship
    cal_data = pd.DataFrame({
        'c': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'ad1': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'ad2': [0.0, 0.11, 0.19, 0.31, 0.39, 0.51]
    })
    
    # Parameters for calculations
    cal_parameters = {
        'Vf_well': 10.0,
        'Vf_prod': 5.0,
        'c_prod': 2.15
    }
    
    concentration_columns = ['HP_mM', 'NADH_mM', 'PD_mM']
    
    return data, cal_data, cal_parameters, concentration_columns


def test_is_linear():
    """Test linear regression detection function."""
    # Perfect linear relationship
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 2, 4, 6, 8])  # y = 2x
    
    slope, r_squared = is_linear(x, y, threshold=0.99)
    
    assert slope is not False, "Should detect linear relationship"
    assert abs(slope - 2.0) < 0.001, f"Expected slope ~2.0, got {slope}"
    assert r_squared > 0.99, f"Expected R² > 0.99, got {r_squared}"
    
    # Non-linear relationship
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 8, 27, 64])  # y = x³
    
    slope, r_squared = is_linear(x, y, threshold=0.95)
    
    assert slope is False, "Should not detect linear relationship for cubic data"
    assert r_squared < 0.95, f"Expected R² < 0.95, got {r_squared}"
    
    print("✓ test_is_linear passed: Linear detection working correctly")


def test_get_time_points():
    """Test time point extraction from column names."""
    data, _, _, _ = create_test_data()
    
    time_points = get_time_points(data)
    expected_times = [0, 30, 60, 90, 120]
    
    assert time_points == expected_times, f"Expected {expected_times}, got {time_points}"
    assert all(isinstance(tp, int) for tp in time_points), "Time points should be integers"
    
    print("✓ test_get_time_points passed: Time points extracted correctly")


def test_get_concentration_data():
    """Test concentration data extraction and duplicate removal."""
    data, _, _, concentration_columns = create_test_data()
    
    conc_data = get_concentration_data(data, concentration_columns)
    
    # Should have 3 unique concentration combinations
    assert len(conc_data) == 3, f"Expected 3 unique combinations, got {len(conc_data)}"
    
    # Check that all concentration columns are present
    for col in concentration_columns:
        assert col in conc_data.columns, f"Missing column {col}"
    
    # Check continuous indexing (no gaps)
    expected_indices = list(range(len(conc_data)))
    actual_indices = list(conc_data.index)
    assert actual_indices == expected_indices, f"Index should be continuous: {actual_indices}"
    
    # Check unique values
    unique_hp = sorted(conc_data['HP_mM'].unique())
    assert unique_hp == [100, 200, 300], f"Expected [100, 200, 300], got {unique_hp}"
    
    print("✓ test_get_concentration_data passed: Concentrations extracted correctly")


def test_get_absorbance_data():
    """Test absorbance data extraction."""
    data, _, _, _ = create_test_data()
    
    abs_data = get_absorbance_data(data)
    
    # Should have only data_XX columns
    expected_columns = ['data_00', 'data_30', 'data_60', 'data_90', 'data_120']
    assert list(abs_data.columns) == expected_columns, f"Expected {expected_columns}, got {list(abs_data.columns)}"
    
    # Should have same number of rows as original data
    assert len(abs_data) == len(data), f"Expected {len(data)} rows, got {len(abs_data)}"
    
    # Check that values are numeric
    assert abs_data.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all(), "All values should be numeric"
    
    print("✓ test_get_absorbance_data passed: Absorbance data extracted correctly")


def test_get_calibration_slope():
    """Test calibration slope calculation."""
    _, cal_data, _, _ = create_test_data()
    
    slope = get_calibration_slope(cal_data)
    
    # Should detect linear relationship and return slope
    assert slope is not False, "Should detect linear calibration relationship"
    assert isinstance(slope, (int, float)), f"Slope should be numeric, got {type(slope)}"
    assert 0.9 < slope < 1.1, f"Expected slope ~1.0, got {slope}"  # Since cal data is roughly y=x
    
    print("✓ test_get_calibration_slope passed: Calibration slope calculated correctly")


def test_get_reaction_slope():
    """Test reaction slope calculation."""
    time_points = [0, 30, 60, 90, 120]
    # Decreasing absorbance (typical enzyme reaction)
    absorbance_values = [2.0, 1.8, 1.6, 1.4, 1.2]
    
    slope, r_value = get_reaction_slope(time_points, absorbance_values)
    
    assert slope is not False, "Should detect linear relationship"
    assert slope < 0, f"Expected negative slope, got {slope}"  # Decreasing absorbance
    assert r_value > 0.95, f"Expected high R², got {r_value}"
    
    print("✓ test_get_reaction_slope passed: Reaction slope calculated correctly")


def test_convert_ad_to_concentration():
    """Test conversion of absorbance slope to enzyme activity."""
    ad_slope = -0.005  # Negative slope (decreasing absorbance)
    cal_slope = 1.0    # 1:1 calibration
    parameters = {
        'Vf_well': 10.0,
        'Vf_prod': 5.0,
        'c_prod': 2.0
    }
    
    activity = convert_ad_to_concentration(ad_slope, cal_slope, parameters)
    
    assert isinstance(activity, (int, float)), f"Activity should be numeric, got {type(activity)}"
    assert activity > 0, f"Activity should be positive, got {activity}"
    
    # Test with different slope
    activity2 = convert_ad_to_concentration(-0.01, cal_slope, parameters)
    assert activity2 > activity, "Higher slope magnitude should give higher activity"
    
    print("✓ test_convert_ad_to_concentration passed: Activity conversion working correctly")


def test_process_duplicates():
    """Test duplicate processing (pairwise averaging)."""
    # Create test data with 4 rows (2 pairs)
    test_data = pd.DataFrame({
        'data_00': [2.0, 1.9, 2.1, 2.0],
        'data_30': [1.8, 1.7, 1.9, 1.8],
        'data_60': [1.6, 1.5, 1.7, 1.6]
    })
    
    result = process_duplicates(test_data)
    
    # Should have half the rows
    assert len(result) == 2, f"Expected 2 rows, got {len(result)}"
    
    # Check averaged values
    expected_first_row = [1.95, 1.75, 1.55]  # Average of rows 0 and 1
    actual_first_row = result.iloc[0].values
    
    np.testing.assert_allclose(actual_first_row, expected_first_row, rtol=1e-10)
    
    print("✓ test_process_duplicates passed: Pairwise averaging working correctly")


def test_process_duplicates2():
    """Test smart duplicate processing based on concentration matching."""
    data, _, _, concentration_columns = create_test_data()
    
    result = process_duplicates2(data, concentration_columns)
    
    # Should have 3 unique concentration combinations
    assert len(result) == 3, f"Expected 3 unique combinations, got {len(result)}"
    
    # Check that concentration columns are preserved
    for col in concentration_columns:
        assert col in result.columns, f"Missing concentration column {col}"
    
    # Check that absorbance columns are averaged
    absorbance_cols = [col for col in result.columns if 'data_' in col]
    assert len(absorbance_cols) > 0, "Should preserve absorbance columns"
    
    # Check unique concentrations
    unique_hp = sorted(result['HP_mM'].unique())
    assert unique_hp == [100, 200, 300], f"Expected [100, 200, 300], got {unique_hp}"
    
    print("✓ test_process_duplicates2 passed: Smart duplicate processing working correctly")


def test_add_noise():
    """Test noise addition to dataframe columns."""
    data = pd.DataFrame({
        'col1': [1.0, 2.0, 3.0, 4.0],
        'col2': [10.0, 20.0, 30.0, 40.0],
        'col3': [100.0, 200.0, 300.0, 400.0]
    })
    
    noise_level = 0.1
    keys = ['col1', 'col2']
    
    # Set seed for reproducible results
    np.random.seed(42)
    noisy_data = add_noise(data, keys, noise_level)
    
    # Should have same structure
    assert list(noisy_data.columns) == list(data.columns), "Columns should be unchanged"
    assert len(noisy_data) == len(data), "Number of rows should be unchanged"
    
    # col3 should be unchanged (not in keys)
    pd.testing.assert_series_equal(noisy_data['col3'], data['col3'])
    
    # col1 and col2 should be different (with noise)
    assert not noisy_data['col1'].equals(data['col1']), "col1 should have noise added"
    assert not noisy_data['col2'].equals(data['col2']), "col2 should have noise added"
    
    # Noise should be reasonable
    diff1 = np.abs(noisy_data['col1'] - data['col1'])
    assert np.all(diff1 < 5 * noise_level), "Noise in col1 seems too large"
    
    print("✓ test_add_noise passed: Noise addition working correctly")


def test_get_processed_data():
    """Test full data processing pipeline."""
    data, cal_data, cal_parameters, concentration_columns = create_test_data()
    
    # Process duplicates first
    processed_full_data = process_duplicates2(data, concentration_columns)
    
    # Get components
    time_points = get_time_points(processed_full_data)
    conc_data = processed_full_data[concentration_columns]
    abs_data = get_absorbance_data(processed_full_data)
    cal_slope = get_calibration_slope(cal_data)
    
    # Process the data
    result_df, regression_results = get_processed_data(
        time_points, conc_data, abs_data, cal_slope, 
        concentration_columns, cal_parameters
    )
    
    # Check structure
    assert isinstance(result_df, pd.DataFrame), "Result should be DataFrame"
    assert 'activity_U/mg' in result_df.columns, "Should have activity column"
    
    # Check concentration columns are present
    for col in concentration_columns:
        assert col in result_df.columns, f"Missing concentration column {col}"
    
    # Check activities are positive
    assert np.all(result_df['activity_U/mg'] > 0), "Activities should be positive"
    
    # Check regression results structure
    assert isinstance(regression_results, list), "Regression results should be list"
    assert len(regression_results) > 0, "Should have regression results"
    
    for result in regression_results:
        assert len(result) == 3, "Each result should have (index, slope, r_value)"
        index, slope, r_value = result
        assert isinstance(index, (int, np.integer)), "Index should be integer"
        
    print("✓ test_get_processed_data passed: Full processing pipeline working correctly")


def run_all_tests():
    """Run all process_data tests."""
    print("Running process_data function tests...\n")
    
    try:
        test_is_linear()
        test_get_time_points()
        test_get_concentration_data()
        test_get_absorbance_data()
        test_get_calibration_slope()
        test_get_reaction_slope()
        test_convert_ad_to_concentration()
        test_process_duplicates()
        test_process_duplicates2()
        test_add_noise()
        test_get_processed_data()
        
        print("\n🎉 All process_data tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_all_tests()