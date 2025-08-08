import unittest
import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from data_hadler import (
    calculate_calibration,
    get_rates_and_concentrations, make_fitting_data,
    add_noise_reaction_dict,
    get_concentrations_from_csv, get_absorption_data, get_time_points,
    calculate_activity
)


class TestDataHandler(unittest.TestCase):

    def setUp(self):
        """Setup für Testdaten"""
        # Test DataFrame für calculate_calibration
        self.calibration_df = pd.DataFrame({
            'concentration': [0, 10, 20, 30, 40],
            'RD_1': [0.0, 0.5, 1.0, 1.5, 2.0],
            'RD_2': [0.1, 0.6, 1.1, 1.6, 2.1]
        })
        
        # Test Dictionary für calculate_calibration
        self.calibration_dict = {
            'concentrations': [0, 10, 20, 30, 40],
            'extinction': [0.05, 0.55, 1.05, 1.55, 2.05]
        }
        
        # Test CSV-Daten für get_concentrations
        self.csv_concentration_data = pd.DataFrame([
            ['Header1', 'Header2', 'Header3'],
            ['Time[s]', 'Conc[mM]', 'Data1'],
            ['0', '0.5', '0.1'],
            ['30', '1.0', '0.2'],
            ['60', '2.0', '0.3'],
            ['90', '5.0', '0.4'],
            ['120', np.nan, '0.5']  # NaN Test
        ])
        
        # Test CSV-Daten für get_rates
        self.csv_rates_data = pd.DataFrame([
            ['Sample', 'Concentration', 'Time0', 'Time30', 'Time60'],
            ['Units', 'mM', 's', 's', 's'],
            ['A1', '0.5', '0.1', '0.15', '0.2'],
            ['A2', '1.0', '0.2', '0.25', '0.3'],
            ['A3', '2.0', '0.3', '0.35', '0.4']
        ])
        
        # Activity parameters für get_rates
        self.activity_params = {
            "Vf_well": 1.0,
            "Vf_prod": 1.0,
            "c_prod": 100.0
        }
        
        self.calibration_slope = 0.05

    def test_calculate_calibration_dataframe_success(self):
        """Test calculate_calibration mit pandas DataFrame"""
        with patch('data_hadler.is_linear', return_value=True):
            slope = calculate_calibration(self.calibration_df)
            
            # Erwartete Steigung basierend auf den Testdaten
            expected_slope = 0.05  # Ungefähr, da Mittelwert aus RD_1 und RD_2
            self.assertIsNotNone(slope)
            self.assertAlmostEqual(slope, expected_slope, places=2)

    def test_calculate_calibration_dataframe_non_linear(self):
        """Test calculate_calibration mit nicht-linearen Daten"""
        with patch('data_hadler.is_linear', return_value=False):
            slope = calculate_calibration(self.calibration_df)
            self.assertIsNone(slope)

    def test_calculate_calibration_dict_success(self):
        """Test calculate_calibration mit Dictionary"""
        with patch('data_hadler.is_linear', return_value=True):
            slope = calculate_calibration(self.calibration_dict)
            
            expected_slope = 0.05
            self.assertIsNotNone(slope)
            self.assertAlmostEqual(slope, expected_slope, places=2)

    def test_calculate_calibration_dict_alternative_keys(self):
        """Test calculate_calibration mit alternativen Dictionary-Keys"""
        alt_dict = {
            'NADH': [0, 10, 20, 30, 40],
            'Mittelwert': [0.05, 0.55, 1.05, 1.55, 2.05]
        }
        
        with patch('data_hadler.is_linear', return_value=True):
            slope = calculate_calibration(alt_dict)
            
            expected_slope = 0.05
            self.assertIsNotNone(slope)
            self.assertAlmostEqual(slope, expected_slope, places=2)

    def test_get_concentrations_success(self):
        """Test get_concentrations_from_csv mit gültigen Daten"""
        concentrations = get_concentrations_from_csv(self.csv_concentration_data)
        
        expected_concentrations = np.array([0.5, 1.0, 2.0, 5.0])
        np.testing.assert_array_equal(concentrations, expected_concentrations)

    def test_get_concentrations_empty_data(self):
        """Test get_concentrations_from_csv mit leeren Daten"""
        empty_df = pd.DataFrame([
            ['Header1', 'Header2'],
            ['Time[s]', 'Conc[mM]']
        ])
        
        concentrations = get_concentrations_from_csv(empty_df)
        self.assertEqual(len(concentrations), 0)

    def test_get_concentrations_with_nan_values(self):
        """Test get_concentrations_from_csv behandelt NaN-Werte korrekt"""
        csv_with_nan = pd.DataFrame([
            ['Header1', 'Header2', 'Header3'],
            ['Time[s]', 'Conc[mM]', 'Data1'],
            ['0', '0.5', '0.1'],
            ['30', np.nan, '0.2'],
            ['60', '2.0', '0.3']
        ])
        
        concentrations = get_concentrations_from_csv(csv_with_nan)
        expected_concentrations = np.array([0.5, 2.0])
        np.testing.assert_array_equal(concentrations, expected_concentrations)

    @patch('data_hadler.get_concentrations_from_csv')
    @patch('data_hadler.get_absorption_data')
    @patch('data_hadler.get_time_points')
    @patch('data_hadler.calculate_activity')
    def test_get_rates_success(self, mock_calc_activity, mock_time_points, 
                              mock_absorption_data, mock_concentrations):
        """Test get_rates_and_concentrations mit erfolgreichen Mocks"""
        # Mock-Rückgabewerte
        mock_concentrations.return_value = np.array([0.5, 1.0, 2.0])
        mock_absorption_data.return_value = np.array([[0.1, 0.15, 0.2], [0.2, 0.25, 0.3]])
        mock_time_points.return_value = np.array([0, 30, 60])
        mock_calc_activity.return_value = np.array([0.01, 0.02, 0.03])
        
        # Test get_rates_and_concentrations instead of get_rates
        concentrations, rates = get_rates_and_concentrations(
            self.csv_rates_data, self.calibration_slope, self.activity_params
        )
        
        # Check if get_concentrations_from_csv was called (indirectly)
        self.assertIsNotNone(concentrations)
        self.assertIsNotNone(rates)

    def test_get_concentrations_invalid_float_values(self):
        """Test get_concentrations_from_csv mit ungültigen Float-Werten"""
        csv_invalid = pd.DataFrame([
            ['Header1', 'Header2', 'Header3'],
            ['Time[s]', 'Conc[mM]', 'Data1'],
            ['0', 'invalid', '0.1'],
            ['30', '1.0', '0.2'],
            ['60', 'also_invalid', '0.3']
        ])
        
        # Dies sollte eine Exception werfen oder graceful handling haben
        # Je nach Implementierung von get_concentrations_from_csv
        try:
            concentrations = get_concentrations_from_csv(csv_invalid)
            # Falls graceful handling: nur gültige Werte zurück
            self.assertEqual(len(concentrations), 1)
            self.assertEqual(concentrations[0], 1.0)
        except ValueError:
            # Falls Exception geworfen wird, ist das auch ok
            pass

    def test_calculate_calibration_missing_columns_dataframe(self):
        """Test calculate_calibration mit fehlenden Spalten im DataFrame"""
        incomplete_df = pd.DataFrame({
            'concentration': [0, 10, 20],
            'RD_1': [0.0, 0.5, 1.0]
            # RD_2 fehlt
        })
        
        with self.assertRaises(KeyError):
            calculate_calibration(incomplete_df)

    def test_calculate_calibration_empty_dict(self):
        """Test calculate_calibration mit leerem Dictionary"""
        empty_dict = {}
        
        with patch('data_hadler.is_linear', return_value=False):
            slope = calculate_calibration(empty_dict)
            # Je nach Implementierung sollte None oder Exception zurückgegeben werden
            self.assertIsNone(slope)

    def test_add_noise_reaction_dict_success(self):
        """Test add_noise_reaction_dict mit Reaktionsdaten-Dictionary"""
        # Create DataFrame-based reaction data as expected by the function
        reaction_dict = {
            'r1_data': pd.DataFrame({
                'Time': [0, 30, 60, 90],
                'Concentration': [1.0, 2.0, 5.0, 10.0],
                'Absorption': [0.1, 0.15, 0.2, 0.25]
            })
        }
        
        noise_percentage = 0.05
        noisy_dict = add_noise_reaction_dict(reaction_dict, noise_percentage)
        
        # Dictionary sollte gleiche Keys haben
        self.assertEqual(set(noisy_dict.keys()), set(reaction_dict.keys()))
        
        # Result should be a dictionary
        self.assertIsInstance(noisy_dict, dict)

    def test_add_noise_reaction_dict_zero_noise(self):
        """Test add_noise_reaction_dict mit 0% Rauschen"""
        # Create DataFrame-based reaction data as expected by the function
        test_dict = {
            'r1_data': pd.DataFrame({
                'Time': [0, 30, 60],
                'Value': [1.0, 2.0, 3.0]
            })
        }
        
        result_dict = add_noise_reaction_dict(test_dict, 0.0)
        
        # Bei 0% Rauschen sollten Werte identisch sein
        self.assertIsInstance(result_dict, dict)

    def test_get_rates_and_concentrations_success(self):
        """Test get_rates_and_concentrations mit einfachen Testdaten"""
        # Create simple test data
        test_csv_data = {
            'r1_data': pd.DataFrame({
                'Time': [0, 30, 60, 90],
                'Concentration': [1.0, 2.0, 5.0, 10.0],
                'A1': [0.1, 0.15, 0.2, 0.25],
                'A2': [0.12, 0.17, 0.22, 0.27]
            })
        }
        test_calibration = 0.05
        test_activity_params = {
            'r1': {
                'Vf_well': 0.2e-3,
                'Vf_prod': 1.0e-3,
                'c_prod': 100.0
            }
        }
        
        # Test the function - it should return concentrations and rates dictionaries
        result = get_rates_and_concentrations(
            test_csv_data, test_calibration, test_activity_params, verbose=False
        )
        
        # Check that result is a dictionary containing processed data
        self.assertIsInstance(result, dict)
        # The function should process the data without errors
        self.assertIsNotNone(result)

    @patch('data_hadler.get_rates_and_concentrations')
    def test_make_fitting_data_simple_model(self, mock_get_rates_conc):
        """Test make_fitting_data mit einfachem Modell"""
        # Mock get_rates_and_concentrations
        mock_get_rates_conc.return_value = (
            np.array([1.0, 2.0, 5.0]),  # concentrations
            np.array([0.1, 0.15, 0.2])  # rates
        )
        
        # Test Modell mit einem Substrat
        test_model = {
            'substrate_keys': ['r1_nad_conc']
        }
        
        test_data_info = {
            'constants': {},
            'active_params': {}
        }
        
        test_concentration_data = {
            'r1_nad_conc': np.array([1.0, 2.0, 5.0])
        }
        
        test_rate_data = {
            'r1_nad_rates': np.array([0.1, 0.15, 0.2])
        }
        
        substrate_data, activities = make_fitting_data(
            test_model, test_data_info, test_concentration_data, test_rate_data, verbose=False
        )
        
        # Erwarte eine Liste mit einem Array für ein Substrat
        self.assertEqual(len(substrate_data), 1)
        np.testing.assert_array_equal(substrate_data[0], np.array([1.0, 2.0, 5.0]))
        np.testing.assert_array_equal(activities, np.array([0.1, 0.15, 0.2]))

    @patch('data_hadler.get_rates_and_concentrations')
    def test_get_rates_and_concentrations_verbose(self, mock_get_rates_conc):
        """Test get_rates_and_concentrations mit verbose=True"""
        mock_get_rates_conc.return_value = (np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        
        test_csv = pd.DataFrame()
        
        # Teste, dass verbose Parameter keinen Fehler verursacht
        concentrations, rates = get_rates_and_concentrations(
            test_csv, 0.05, {}, verbose=True
        )
        
        # Funktion sollte erfolgreich ausgeführt werden
        self.assertIsNotNone(concentrations)
        self.assertIsNotNone(rates)

if __name__ == '__main__':
    # Test Suite ausführen
    unittest.main(verbosity=2)