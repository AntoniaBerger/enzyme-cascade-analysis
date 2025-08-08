import unittest
import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parameter_estimator import (
    estimate_parameters, fit_parameters, monte_carlo_simulation_r1,
    validate_parameters
)


class TestParameterEstimator(unittest.TestCase):

    def setUp(self):
        """Setup für Testdaten"""
        # Test-Daten für zwei-Substrat Michaelis-Menten
        self.test_concentrations = {
            "r1_nad_conc": np.array([0.5, 1.0, 2.0, 5.0]),
            "r1_nad_const": 500.0,
            "r1_pd_conc": np.array([100.0, 200.0, 500.0, 1000.0]),
            "r1_pd_const": 5.0,
        }
        
        self.test_rates = {
            "r1_nad_rates": np.array([0.1, 0.15, 0.2, 0.25]),
            "r1_pd_rates": np.array([0.08, 0.12, 0.18, 0.22]),
        }
        
        # Test-Model Definition
        def test_two_substrat_michaelis_menten(concentration_data, Vmax, Km1, Km2):
            S1_values, S2_values = concentration_data
            S1_values = np.asarray(S1_values)
            S2_values = np.asarray(S2_values)
            return (Vmax * S1_values * S2_values) / ((Km1 + S1_values) * (Km2 + S2_values))
        
        self.test_model = {
            "name": "two_substrat_michaelis_menten",
            "function": test_two_substrat_michaelis_menten,
            "param_names": ["Vmax", "Km_NAD", "Km_PD"],
            "param_units": ["U", "mM", "mM"],
            "substrate_keys": ["r1_nad_conc", "r1_pd_conc"],
            "initial_guess_func": lambda activities, substrate_data: [max(activities) if len(activities) > 0 else 1.0, 1.0, 1.0],
            "bounds_lower": [0, 0, 0],
            "bounds_upper": [np.inf, np.inf, np.inf],
            "description": "Test Two-Substrate Michaelis-Menten"
        }

    @patch('parameter_estimator.curve_fit')
    def test_fit_parameters_success(self, mock_curve_fit):
        """Test fit_parameters mit erfolgreichem Fitting"""
        # Mock curve_fit Rückgabe
        mock_params = [10.0, 2.0, 3.0]
        mock_covariance = np.array([[1.0, 0.1, 0.1], [0.1, 1.0, 0.1], [0.1, 0.1, 1.0]])
        mock_curve_fit.return_value = (mock_params, mock_covariance)
        
        # Verwende realistische Testdaten die zur Funktion passen
        S1_values = np.array([1.0, 2.0, 3.0])
        S2_values = np.array([5.0, 10.0, 15.0])
        substrate_data = [S1_values, S2_values]
        
        # Simuliere Aktivitäten basierend auf der Funktion mit bekannten Parametern
        Vmax_true = 10.0
        Km1_true = 2.0
        Km2_true = 3.0
        activities = (Vmax_true * S1_values * S2_values) / ((Km1_true + S1_values) * (Km2_true + S2_values))
        
        result = fit_parameters(substrate_data, activities, self.test_model)
        
        self.assertTrue(result['success'])
        np.testing.assert_array_almost_equal(result['params'], mock_params)
        # Da wir curve_fit mocken, wird ein Mock-R² berechnet
        self.assertIsInstance(result['r_squared'], (float, np.floating))

    @patch('parameter_estimator.curve_fit')
    def test_fit_parameters_failure(self, mock_curve_fit):
        """Test fit_parameters mit Fitting-Fehler"""
        mock_curve_fit.side_effect = RuntimeError("Fitting failed")
        
        substrate_data = [[1.0, 2.0], [5.0, 10.0]]
        activities = np.array([0.1, 0.2])
        
        result = fit_parameters(substrate_data, activities, self.test_model)
        
        self.assertFalse(result['success'])

    @patch('parameter_estimator.make_fitting_data')
    @patch('parameter_estimator.fit_parameters')
    def test_estimate_parameters_success(self, mock_fit_parameters, mock_make_fitting_data):
        """Test estimate_parameters mit erfolgreichen Mocks"""
        # Mock make_fitting_data
        mock_substrate_data = [[1.0, 2.0], [5.0, 10.0]]
        mock_activities = np.array([0.1, 0.2])
        mock_make_fitting_data.return_value = (mock_substrate_data, mock_activities)
        
        # Mock fit_parameters
        mock_result = {
            'success': True,
            'params': [10.0, 2.0, 3.0],
            'param_errors': [1.0, 0.5, 0.5],
            'r_squared': 0.95,
            'model_name': 'test_model',
            'description': 'Test Model'
        }
        mock_fit_parameters.return_value = mock_result
        
        # Use correct function signature: model_info, data_info, concentrations, rates
        data_info = {'constants': {}, 'active_params': {}}
        
        result = estimate_parameters(self.test_model, data_info, self.test_concentrations, self.test_rates)
        
        # Überprüfe, dass beide Funktionen aufgerufen wurden
        mock_make_fitting_data.assert_called_once_with(self.test_model, data_info, self.test_concentrations, self.test_rates, verbose=True)
        mock_fit_parameters.assert_called_once_with(mock_substrate_data, mock_activities, self.test_model, verbose=True)
        
        # Überprüfe Rückgabewert
        self.assertEqual(result, mock_result)

    @patch('parameter_estimator.make_fitting_data')
    def test_estimate_parameters_make_fitting_data_failure(self, mock_make_fitting_data):
        """Test estimate_parameters wenn make_fitting_data fehlschlägt"""
        mock_make_fitting_data.return_value = (None, None)
        
        data_info = {'constants': {}, 'active_params': {}}
        
        # Dies sollte zu einem Fehler führen, da fit_parameters None-Werte nicht verarbeiten kann
        with self.assertRaises((TypeError, AttributeError)):
            estimate_parameters(self.test_model, data_info, self.test_concentrations, self.test_rates)

    def test_fit_parameters_empty_activities(self):
        """Test fit_parameters mit leeren Aktivitätsdaten"""
        substrate_data = [[1.0, 2.0], [5.0, 10.0]]
        empty_activities = np.array([])
        
        result = fit_parameters(substrate_data, empty_activities, self.test_model)
        
        self.assertFalse(result['success'])

    def test_fit_parameters_mismatched_data_lengths(self):
        """Test fit_parameters mit unterschiedlichen Datenlängen"""
        substrate_data = [[1.0, 2.0], [5.0, 10.0]]  # 2 Datenpunkte
        activities = np.array([0.1, 0.2, 0.3])  # 3 Datenpunkte
        
        result = fit_parameters(substrate_data, activities, self.test_model)
        
        # Sollte fehlschlagen wegen unterschiedlicher Längen
        self.assertFalse(result['success'])

    def test_estimate_parameters_complete_workflow(self):
        """Test des kompletten estimate_parameters Workflows mit realen Daten"""
        # Erstelle realistische Testdaten
        S1_test = np.array([1.0, 2.0, 5.0, 10.0])
        S2_test = np.array([500.0, 500.0, 500.0, 500.0])
        
        # Simuliere Aktivitätsdaten basierend auf Michaelis-Menten
        Vmax_true = 20.0
        Km1_true = 2.5
        Km2_true = 300.0
        
        activities_test = (Vmax_true * S1_test * S2_test) / ((Km1_true + S1_test) * (Km2_true + S2_test))
        
        test_concentrations = {
            "r1_nad_conc": S1_test,
            "r1_nad_const": 500.0,
            "r1_pd_conc": np.array([]),  # Leer für diesen Test
            "r1_pd_const": 5.0,
        }
        
        test_rates = {
            "r1_nad_rates": activities_test,
            "r1_pd_rates": np.array([]),  # Leer für diesen Test
        }
        
        data_info = {'constants': {}, 'active_params': {}}
        
        # Da make_fitting_data aus data_hadler kommt, mocken wir es
        with patch('parameter_estimator.make_fitting_data') as mock_make_fitting_data:
            mock_make_fitting_data.return_value = ([S1_test, S2_test], activities_test)
            
            result = estimate_parameters(self.test_model, data_info, test_concentrations, test_rates)
            
            # Überprüfe, dass das Fitting erfolgreich war
            self.assertTrue(result['success'])
            self.assertGreater(result['r_squared'], 0.8)  # Sollte gutes Fitting sein
            
            # Überprüfe, dass Parameter in realistischen Bereichen liegen
            params = result['params']
            self.assertGreater(params[0], 0)  # Vmax > 0
            self.assertGreater(params[1], 0)  # Km1 > 0
            self.assertGreater(params[2], 0)  # Km2 > 0

    def test_fit_parameters_initial_guess_function(self):
        """Test dass die initial_guess_func korrekt aufgerufen wird"""
        substrate_data = [[1.0, 2.0, 3.0], [5.0, 10.0, 15.0]]
        activities = np.array([0.1, 0.2, 0.3])
        
        # Mock initial_guess_func um zu überprüfen, ob sie aufgerufen wird
        def mock_initial_guess(act, sub):
            return [max(act), 1.5, 2.5]
        
        test_model_copy = self.test_model.copy()
        test_model_copy['initial_guess_func'] = mock_initial_guess
        
        with patch('parameter_estimator.curve_fit') as mock_curve_fit:
            mock_curve_fit.return_value = ([10.0, 2.0, 3.0], np.eye(3))
            
            fit_parameters(substrate_data, activities, test_model_copy)
            
            # Überprüfe, dass curve_fit mit den richtigen Initial Guess aufgerufen wurde
            call_args = mock_curve_fit.call_args
            p0_used = call_args[1]['p0']  # p0 ist ein keyword argument
            
            expected_p0 = [0.3, 1.5, 2.5]  # max(activities) = 0.3
            np.testing.assert_array_almost_equal(p0_used, expected_p0)

    def test_fit_parameters_bounds_usage(self):
        """Test dass Parameter-Grenzen korrekt verwendet werden"""
        substrate_data = [[1.0, 2.0], [5.0, 10.0]]
        activities = np.array([0.1, 0.2])
        
        with patch('parameter_estimator.curve_fit') as mock_curve_fit:
            mock_curve_fit.return_value = ([10.0, 2.0, 3.0], np.eye(3))
            
            fit_parameters(substrate_data, activities, self.test_model)
            
            # Überprüfe, dass curve_fit mit den richtigen Grenzen aufgerufen wurde
            call_args = mock_curve_fit.call_args
            bounds_used = call_args[1]['bounds']
            
            expected_bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
            self.assertEqual(bounds_used, expected_bounds)

    @patch('parameter_estimator.estimate_parameters')
    @patch('parameter_estimator.add_noise_reaction_dict')
    @patch('parameter_estimator.add_noise_calibration')
    @patch('parameter_estimator.calculate_calibration')
    def test_monte_carlo_simulation_r1_success(self, mock_calc_cal, mock_add_noise_cal, 
                                              mock_add_noise_reaction, mock_estimate):
        """Test monte_carlo_simulation_r1 mit erfolgreichen Mocks"""
        # Mock functions
        mock_calc_cal.return_value = 0.05  # calibration slope
        mock_add_noise_cal.return_value = pd.DataFrame()  # noisy calibration
        mock_add_noise_reaction.return_value = {}  # noisy reaction data
        
        # Mock successful estimate_parameters
        mock_successful_result = {
            'success': True,
            'params': [10.0, 2.0, 3.0],
            'param_errors': [1.0, 0.5, 0.5],
            'r_squared': 0.95
        }
        mock_estimate.return_value = mock_successful_result
        
        # Test data
        calibration_data = pd.DataFrame()
        reaction_data = {}
        model_info = {'description': 'Test Model'}
        data_info = {'active_params': {}}
        noise_level = {'calibration': 0.05, 'reaction': 0.05}
        
        # Test mit wenigen Iterationen
        results = monte_carlo_simulation_r1(
            calibration_data, reaction_data, model_info, data_info,
            noise_level, n_iterations=2
        )
        
        # Überprüfe Rückgabe-Format
        self.assertIsInstance(results, dict)
        self.assertIn('successful_results', results)
        self.assertIn('failed_counts', results)
        self.assertIn('n_successful', results)
        self.assertIn('n_iterations', results)
        
        # Basic checks that function executed
        self.assertIsNotNone(results)

    def test_monte_carlo_simulation_r1_zero_iterations(self):
        """Test monte_carlo_simulation_r1 mit 0 Iterationen"""
        calibration_data = pd.DataFrame()
        reaction_data = {}
        model_info = {'description': 'Test Model'}
        data_info = {'active_params': {}}
        noise_level = {'calibration': 0.05, 'reaction': 0.05}
        
        results = monte_carlo_simulation_r1(
            calibration_data, reaction_data, model_info, data_info,
            noise_level, n_iterations=0
        )
        
        self.assertIsInstance(results, dict)
        self.assertEqual(results['n_successful'], 0)
        self.assertEqual(results['n_iterations'], 0)

    @patch('parameter_estimator.make_fitting_data')
    @patch('parameter_estimator.fit_parameters')
    def test_estimate_parameters_verbose_parameter(self, mock_fit_parameters, mock_make_fitting_data):
        """Test estimate_parameters mit verbose Parameter"""
        # Mock make_fitting_data
        mock_substrate_data = [[1.0, 2.0], [5.0, 10.0]]
        mock_activities = np.array([0.1, 0.2])
        mock_make_fitting_data.return_value = (mock_substrate_data, mock_activities)
        
        # Mock fit_parameters
        mock_result = {
            'success': True,
            'params': [10.0, 2.0, 3.0],
            'param_errors': [1.0, 0.5, 0.5],
            'r_squared': 0.95
        }
        mock_fit_parameters.return_value = mock_result
        
        data_info = {'constants': {}, 'active_params': {}}
        
        # Test mit verbose=True
        result = estimate_parameters(
            self.test_model, 
            data_info,
            self.test_concentrations, 
            self.test_rates, 
            verbose=True
        )
        
        # Überprüfe, dass make_fitting_data mit verbose aufgerufen wurde
        mock_make_fitting_data.assert_called_once_with(
            self.test_model, 
            data_info,
            self.test_concentrations, 
            self.test_rates, 
            verbose=True
        )
        
        self.assertEqual(result, mock_result)

    @patch('parameter_estimator.make_fitting_data')
    @patch('parameter_estimator.fit_parameters')  
    def test_fit_parameters_verbose_parameter(self, mock_fit_parameters, mock_make_fitting_data):
        """Test fit_parameters mit verbose Parameter"""
        substrate_data = [[1.0, 2.0], [5.0, 10.0]]
        activities = np.array([0.1, 0.2])
        
        with patch('parameter_estimator.curve_fit') as mock_curve_fit:
            mock_curve_fit.return_value = ([10.0, 2.0, 3.0], np.eye(3))
            
            # Test mit verbose=True
            with patch('builtins.print') as mock_print:
                result = fit_parameters(substrate_data, activities, self.test_model, verbose=True)
                
                # Überprüfe, dass verbose output erzeugt wurde
                mock_print.assert_called()
                
            self.assertTrue(result['success'])


if __name__ == '__main__':
    # Test Suite ausführen
    unittest.main(verbosity=2)