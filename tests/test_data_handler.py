import pytest
import pandas as pd
import numpy as np
import os
import pickle
import sys
from unittest.mock import patch
import tempfile
import shutil

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data_handler import save_results


class TestDataHandler:
    
    @pytest.fixture
    def sample_monte_carlo_data(self):
        """Create sample Monte Carlo data for testing."""
        np.random.seed(42)
        return np.random.rand(100, 3)  # 100 iterations, 3 parameters
    
    @pytest.fixture
    def sample_parameter_names(self):
        """Sample parameter names."""
        return ['Vmax', 'Km1', 'Km2']
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for file operations."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_save_results_basic_functionality(self, sample_monte_carlo_data, sample_parameter_names, temp_dir):
        """Test basic functionality of save_results function."""
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Call the function
            result_df = save_results(sample_monte_carlo_data, sample_parameter_names, "test_dataset")
            
            # Check return value
            assert isinstance(result_df, pd.DataFrame)
            assert list(result_df.columns) == sample_parameter_names
            assert result_df.index.name == "test_dataset"
            assert len(result_df) == 100
            
            # Check if files were created
            assert os.path.exists("test_dataset_results.csv")
            assert os.path.exists("test_dataset_results.pkl")
            
        finally:
            os.chdir(original_cwd)
    
    def test_save_results_default_dataset_name(self, sample_monte_carlo_data, sample_parameter_names, temp_dir):
        """Test save_results with default dataset name."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            result_df = save_results(sample_monte_carlo_data, sample_parameter_names)
            
            assert result_df.index.name == "monte_carlo_results"
            assert os.path.exists("monte_carlo_results_results.csv")
            assert os.path.exists("monte_carlo_results_results.pkl")
            
        finally:
            os.chdir(original_cwd)
    
    def test_save_results_data_integrity(self, sample_monte_carlo_data, sample_parameter_names, temp_dir):
        """Test that saved data maintains integrity."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            result_df = save_results(sample_monte_carlo_data, sample_parameter_names, "integrity_test")
            
            # Check CSV file integrity
            loaded_csv = pd.read_csv("integrity_test_results.csv", index_col=0)
            pd.testing.assert_frame_equal(result_df.reset_index(drop=True), loaded_csv.reset_index(drop=True))
            
            # Check pickle file integrity
            with open("integrity_test_results.pkl", 'rb') as f:
                loaded_pickle = pickle.load(f)
            pd.testing.assert_frame_equal(result_df, loaded_pickle)
            
        finally:
            os.chdir(original_cwd)
    
    def test_save_results_numpy_array_conversion(self, sample_parameter_names, temp_dir):
        """Test conversion of different numpy array types."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Test with different data types
            int_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            result_df = save_results(int_data, sample_parameter_names, "int_test")
            
            assert result_df.dtypes['Vmax'] in [np.int32, np.int64, int]
            assert len(result_df) == 3
            assert len(result_df.columns) == 3
            
        finally:
            os.chdir(original_cwd)
    
    def test_save_results_single_parameter(self, temp_dir):
        """Test save_results with single parameter."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            single_param_data = np.array([[1.5], [2.3], [3.7]])
            result_df = save_results(single_param_data, ['Vmax'], "single_param_test")
            
            assert list(result_df.columns) == ['Vmax']
            assert len(result_df) == 3
            assert result_df['Vmax'].tolist() == [1.5, 2.3, 3.7]
            
        finally:
            os.chdir(original_cwd)
    
    def test_save_results_empty_data(self, sample_parameter_names, temp_dir):
        """Test save_results with empty data."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            empty_data = np.array([]).reshape(0, 3)
            result_df = save_results(empty_data, sample_parameter_names, "empty_test")
            
            assert len(result_df) == 0
            assert list(result_df.columns) == sample_parameter_names
            assert os.path.exists("empty_test_results.csv")
            
        finally:
            os.chdir(original_cwd)
    
    def test_save_results_mismatched_dimensions(self, temp_dir):
        """Test save_results with mismatched data and parameter dimensions."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            data = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 array
            wrong_params = ['Vmax', 'Km1']  # Only 2 parameters for 3 columns
            
            with pytest.raises(ValueError):
                save_results(data, wrong_params, "mismatch_test")
                
        finally:
            os.chdir(original_cwd)
    
    def test_save_results_special_characters_in_dataset_name(self, sample_monte_carlo_data, sample_parameter_names, temp_dir):
        """Test save_results with special characters in dataset name."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Note: Some special characters might cause issues with file systems
            safe_special_name = "test_dataset-with_numbers123"
            result_df = save_results(sample_monte_carlo_data, sample_parameter_names, safe_special_name)
            
            assert result_df.index.name == safe_special_name
            assert os.path.exists(f"{safe_special_name}_results.csv")
            assert os.path.exists(f"{safe_special_name}_results.pkl")
            
        finally:
            os.chdir(original_cwd)
    
    @patch('pandas.DataFrame.to_csv')
    @patch('pandas.DataFrame.to_pickle')
    def test_save_results_file_writing_errors(self, mock_to_pickle, mock_to_csv, sample_monte_carlo_data, sample_parameter_names):
        """Test handling of file writing errors."""
        # Mock file writing to raise an exception
        mock_to_csv.side_effect = PermissionError("Permission denied")
        mock_to_pickle.side_effect = PermissionError("Permission denied")
        
        # The function should still return the DataFrame even if file writing fails
        with pytest.raises(PermissionError):
            save_results(sample_monte_carlo_data, sample_parameter_names, "error_test")
    
    def test_save_results_large_dataset(self, sample_parameter_names, temp_dir):
        """Test save_results with large dataset."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Create larger dataset
            large_data = np.random.rand(10000, 3)
            result_df = save_results(large_data, sample_parameter_names, "large_test")
            
            assert len(result_df) == 10000
            assert list(result_df.columns) == sample_parameter_names
            
            # Verify files exist and have reasonable sizes
            assert os.path.getsize("large_test_results.csv") > 1000  # CSV should be reasonably large
            assert os.path.getsize("large_test_results.pkl") > 500   # Pickle should be smaller but still substantial
            
        finally:
            os.chdir(original_cwd)
    
    def test_save_results_nan_values(self, sample_parameter_names, temp_dir):
        """Test save_results with NaN values."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            data_with_nan = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
            result_df = save_results(data_with_nan, sample_parameter_names, "nan_test")
            
            assert pd.isna(result_df.iloc[0, 2])  # Check NaN is preserved
            assert pd.isna(result_df.iloc[1, 1])
            
            # Verify NaN values are properly saved and loaded
            loaded_csv = pd.read_csv("nan_test_results.csv", index_col=0)
            assert pd.isna(loaded_csv.iloc[0, 2])
            assert pd.isna(loaded_csv.iloc[1, 1])
            
        finally:
            os.chdir(original_cwd)
    
    def test_save_results_extreme_values(self, sample_parameter_names, temp_dir):
        """Test save_results with extreme values."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            extreme_data = np.array([
                [1e-10, 1e10, 0.0],
                [np.inf, -np.inf, 1.0],
                [1.23456789e-15, 9.87654321e20, -1e-5]
            ])
            
            result_df = save_results(extreme_data, sample_parameter_names, "extreme_test")
            
            # Check that extreme values are preserved
            assert result_df.iloc[0, 0] == 1e-10
            assert result_df.iloc[0, 1] == 1e10
            assert np.isinf(result_df.iloc[1, 0])
            assert np.isinf(result_df.iloc[1, 1])
            
        finally:
            os.chdir(original_cwd)
    
    def test_save_results_parameter_name_validation(self, sample_monte_carlo_data, temp_dir):
        """Test save_results with various parameter name formats."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Test with different parameter name formats
            special_params = ['V_max', 'K_m1', 'K_m2']
            result_df = save_results(sample_monte_carlo_data, special_params, "special_params_test")
            
            assert list(result_df.columns) == special_params
            
            # Test with numeric-like parameter names
            numeric_params = ['param_1', 'param_2', 'param_3']
            result_df2 = save_results(sample_monte_carlo_data, numeric_params, "numeric_params_test")
            
            assert list(result_df2.columns) == numeric_params
            
        finally:
            os.chdir(original_cwd)
    
    def test_save_results_return_value_type(self, sample_monte_carlo_data, sample_parameter_names, temp_dir):
        """Test that save_results returns the correct type and structure."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            result = save_results(sample_monte_carlo_data, sample_parameter_names, "return_test")
            
            # Check return type
            assert isinstance(result, pd.DataFrame)
            
            # Check that returned DataFrame has correct structure
            assert result.shape == (100, 3)
            assert result.index.name == "return_test"
            assert all(col in result.columns for col in sample_parameter_names)
            
            # Check that data is accessible
            assert not result.empty
            assert result.iloc[0, 0] is not None
            
        finally:
            os.chdir(original_cwd)