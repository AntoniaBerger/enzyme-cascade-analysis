import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

from data_hadler import  calculate_calibration, add_noise, get_rates_and_concentrations , make_fitting_data



def full_reaction_system(concentration_data, Vmax1, Vmax2, Vmax3, KmPD, KmNAD, KmLactol, KmNADH, KiPD, KiNAD, KiLactol):
    """
    Wrapper für curve_fit Kompatibilität - nimmt flache Parameter entgegen
    Berechnet die Enzymaktivität für das vollständige Drei-Reaktions-System
    
    ALLE DREI REAKTIONEN:
    - Reaktion 1: PD + NAD → Pyruvat + NADH
    - Reaktion 2: Lactol + NADH → ... (mit PD/NAD Inhibition)
    - Reaktion 3: Lactol + NAD → ... (mit Lactol Inhibition)
    """
    # Entpacke Substratkonzentrationen, Inhibitor-Konzentrationen und Reaktions-IDs
    # test if concentration_data has the right shape
    if len(concentration_data) != 4 or len(concentration_data[0]) == 0:
        raise ValueError("concentration_data must contain 4 arrays: S1, S2, Inhibitor, reaction_ids")
    
    S1, S2, Inhibitor, reaction_ids = concentration_data
    
    # Initialisiere Ergebnis-Array
    V_obs = np.zeros_like(S1, dtype=float)
    
    # Reaktion 1: PD + NAD → Pyruvat + NADH
    reaction_1_mask = (reaction_ids == 1)
    if np.any(reaction_1_mask):
        # S1 = NAD oder konstante NAD, S2 = PD oder konstante PD
        S1_r1 = S1[reaction_1_mask] # NAD
        S2_r1 = S2[reaction_1_mask] # PD
        
        V_obs[reaction_1_mask] = (Vmax1 * S1_r1 * S2_r1) / (
            (KmPD + S2_r1) *  (KmNAD + S1_r1)
        )
    
    # Reaktion 2: Lactol + NADH → ... (mit PD Inhibition)
    reaction_2_mask = (reaction_ids == 2)
    if np.any(reaction_2_mask):
        S1_r2 = S1[reaction_2_mask]  # Lactol
        S2_r2 = S2[reaction_2_mask]  # NADH
        PD_inhibitor = Inhibitor[reaction_2_mask]  # Variable PD-Konzentration als Inhibitor
        
        V_obs[reaction_2_mask] = (Vmax2 * S1_r2 * S2_r2) / (
            (KmLactol * (1 + PD_inhibitor/KiPD) + S1_r2) * (KmNADH * (1 + PD_inhibitor/KiNAD) + S2_r2)
        )
    # Reaktion 3: Lactol + NAD → ... (mit Lactol Selbst-Inhibition)
    reaction_3_mask = (reaction_ids == 3)
    if np.any(reaction_3_mask):
        S1_r3 = S1[reaction_3_mask]  # Lactol
        S2_r3 = S2[reaction_3_mask]  # NAD
        
        V_obs[reaction_3_mask] = (Vmax3 * S1_r3 * S2_r3) / (
            (KmLactol * (1 + S1_r3/KiLactol) + S1_r3) * (KmNAD + S2_r3)
        )
    
    return V_obs

# Setze Funktions-Referenzen in AVAILABLE_MODELS

def fit_parameters(substrate_data, activities, model_info):
    """
    Allgemeine Fitting-Funktion mit modularer Datenstruktur.
    """

    result = {
        'success': False,
        'params': {},
        'param_errors': {},
        'r_squared': 0.0,
        'model_name': model_info['name'],
        'description': model_info['description']
    }
    
    # Überprüfe zuerst, ob activities leer ist
    if len(activities) == 0:
        result['success'] = False
        return result
    
    # Initial guess berechnen
    p0 = model_info['initial_guess_func'](activities, substrate_data)
    print(f"Initial guess: {p0}")
    
    # Parameter-Grenzen für physikalisch sinnvolle Werte
    bounds_lower = model_info['bounds_lower']
    bounds_upper = model_info['bounds_upper']
    
    try:
        params, covariance = curve_fit(model_info['function'], substrate_data, activities, 
                                        p0=p0, bounds=(bounds_lower, bounds_upper), maxfev=5000)
        fitted_params = params
        param_errors = np.sqrt(np.diag(covariance))
            
        # R² berechnen
        y_pred = model_info['function'](substrate_data, *fitted_params)
            
        ss_res = np.sum((activities - y_pred) ** 2)
        ss_tot = np.sum((activities - np.mean(activities)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        result['success'] = True
        result['params'] = fitted_params
        result['param_errors'] = param_errors
        result['r_squared'] = r_squared

        return result
        
    except Exception as e:
        print(f"Fehler beim Fitten: {e}")
        result['success'] = False
        return result


# def monte_carlo_simulation(experiment_data, model_name, n_iterations=1000, noise_level=0.02):
#     """
#     Monte Carlo Simulation für beliebige Modelle mit modularer Datenstruktur.
#     experiment_data: Dict mit Konzentrationen und Aktivitäten 
#     model_name: String, Key aus AVAILABLE_MODELS
#     """
    
#     print("=== MONTE CARLO SIMULATION ===")
#     print(f"Modell: {model_name}")
#     print(f"Iterationen: {n_iterations}")
#     print(f"Rauschen: {noise_level*100:.1f}%")
#     print("="*50)

#     successful_iterations = 0
#     failed_reasons = {"daten": 0, "fitting": 0, "validation": 0}
    
#     for iteration in range(n_iterations):
#         try:
#             # Erstelle verrauschte Kopie der experiment_data
#             noisy_experiment_data = {}
#             for key, value in experiment_data.items():
#                 if key == 'activities':
#                     # Füge Rauschen zu den Aktivitäten hinzu
#                     noisy_experiment_data[key] = add_noise(value, noise_level)
#                 else:
#                     # Andere Felder unverändert
#                     noisy_experiment_data[key] = value.copy()

#         except Exception:
#             failed_reasons["daten"] += 1
#             continue
        
#         # Parameter schätzen mit verrauschten Daten
#         try:
#             result = schaetze_parameter(noisy_experiment_data, model_name, verbose=False)
#         except Exception:
#             result = None
        
#         if result is not None and result.get('success', False):
#             # Parameter extrahieren und validieren
#             params = result.get('params', {})
#             r_squared = result.get('r_squared', 0)
            
#             # Basis-Validierung: Parameter müssen positiv und realistisch sein
#             valid = True
#             for param_name in model_info['param_names']:
#                 param_value = params.get(param_name, 0)
#                 if param_value <= 0 or param_value > 10000:
#                     valid = False
#                     break
            
#             if valid and r_squared > 0.3:
#                 # Speichere gültige Parameter
#                 for param_name in model_info['param_names']:
#                     param_results[param_name].append(params[param_name])
#                 r_squared_results.append(r_squared)
#                 successful_iterations += 1
#             else:
#                 failed_reasons["validation"] += 1
#         else:
#             failed_reasons["fitting"] += 1
            
#         if (iteration + 1) % 100 == 0:
#             print(f"Iteration {iteration + 1}/{n_iterations} - Erfolgreiche: {successful_iterations}")
    
#     print("\nZusammenfassung der Fehlschläge:")
#     print(f"- Datenprobleme: {failed_reasons['daten']}")
#     print(f"- Fitting-Probleme: {failed_reasons['fitting']}")
#     print(f"- Validierungs-Probleme: {failed_reasons['validation']}")
    
#     if successful_iterations < 10:
#         print(f"Zu wenige erfolgreiche Iterationen: {successful_iterations}")
#         return None
    
#     # Statistische Auswertung
#     r_squared_results = np.array(r_squared_results)
#     param_names = model_info['param_names']
    
#     # Konvertiere zu numpy arrays
#     param_arrays = {name: np.array(param_results[name]) for name in param_names}

#     # Kovarianzmatrix und Korrelation berechnen
#     if len(param_arrays) >= 2:
#         parameter_matrix = np.column_stack([param_arrays[name] for name in param_names])
#         covariance_matrix = np.cov(parameter_matrix, rowvar=False)
#         correlation_matrix = np.corrcoef(parameter_matrix, rowvar=False)
#     else:
#         covariance_matrix = None
#         correlation_matrix = None
    
#     # Ergebnis-Dictionary erstellen
#     results = {
#         'n_successful': successful_iterations,
#         'model_name': model_name,
#         'param_names': param_names,
#         'R_squared_mean': np.mean(r_squared_results),
#         'R_squared_std': np.std(r_squared_results),
#         'r_squared_values': r_squared_results
#     }
    
#     # Parameter-spezifische Statistiken hinzufügen
#     for param_name in param_names:
#         if param_name in param_arrays:
#             param_data = param_arrays[param_name]
#             results[f'{param_name}_mean'] = np.mean(param_data)
#             results[f'{param_name}_std'] = np.std(param_data)
#             results[f'{param_name}_ci_lower'] = np.percentile(param_data, 2.5)
#             results[f'{param_name}_ci_upper'] = np.percentile(param_data, 97.5)
#             results[f'{param_name}_values'] = param_data
    
#     if covariance_matrix is not None:
#         results['covariance_matrix'] = covariance_matrix
#         results['correlation_matrix'] = correlation_matrix
    
#     # Ausgabe
#     print("\n=== MONTE CARLO ERGEBNISSE ===")
#     print(f"Erfolgreiche Iterationen: {successful_iterations}/{n_iterations}")
    
#     for param_name in param_names:
#         if f'{param_name}_mean' in results:
#             mean_val = results[f'{param_name}_mean']
#             std_val = results[f'{param_name}_std']
#             ci_lower = results[f'{param_name}_ci_lower']
#             ci_upper = results[f'{param_name}_ci_upper']
#             unit = model_info['param_units'][param_names.index(param_name)]
            
#             print(f"{param_name}: {mean_val:.4f} ± {std_val:.4f} {unit}")
#             print(f"{param_name} 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
#     print(f"R²: {results['R_squared_mean']:.4f} ± {results['R_squared_std']:.4f}")
    
#     return results

def monte_carlo_simulation_r1(calibration_data, reaction_data, model_info, noise_level, n_iterations = 1000):
    
    results = []

    for iteration in range(n_iterations):
        #1. verrausche daten 
        
        calibration_data_noisy = add_noise(calibration_data, noise_level=noise_level["calibration"])
        calibration_slope_noisy = calculate_calibration(calibration_data_noisy)

        reaction_data_noisy = add_noise(reaction_data, noise_level=noise_level["reaction"])

        reac_1_rates = {
        "r1_nad_rates": get_rates_and_concentrations(r1_nad_data, calibration_slope_noisy, reac1_activity_param["r1"])[0],
        "r1_pd_rates":  get_rates_and_concentrations(r1_pd_data, calibration_slope_noisy, reac1_activity_param["r1"])[0],
        }

        reac_1_concentrations = {
            "r1_nad_conc": get_rates_and_concentrations(r1_nad_data, calibration_slope_noisy, reac1_activity_param["r1"])[1],
            "r1_nad_const": 5.0,  # Konstante NAD Konzentration
            "r1_pd_conc":  get_rates_and_concentrations(r1_pd_data, calibration_slope_noisy, reac1_activity_param["r1"])[1],
            "r1_pd_const": 500.0,  # Konstante PD Konzentration
        }

        # 2. führe Parameter-Schätzung durch
        result = estimate_parameters(model_info, calibration_data_noisy, reaction_data_noisy)

        results.append(result)
    

    return results

def validate_parameters(params_dict, r_squared=None):
    """
    Validiert Parameter-Dictionary auf Plausibilität.
    params_dict: Dict mit Parameter-Namen als Keys und Werten
    r_squared: Optional, R²-Wert für zusätzliche Validierung
    """
    for param_name, value in params_dict.items():
        if not isinstance(value, (int, float)):
            return False
        if value <= 0 or value > 10000:
            return False
    
    if r_squared is not None and r_squared < 0.3:
        return False
    
    return True

def estimate_parameters(model_info, concentrations, rates): 
    """
    Schätzt die Parameter für ein gegebenes Modell basierend auf Konzentrationen und Raten.
    
    model_info: Dict mit Modell-Informationen (z.B. name, function, param_names, etc.)
    concentrations: Dict mit Substratkonzentrationen
    rates: Dict mit gemessenen Raten
    """
    
    substrate_data, activities = make_fitting_data(model_info, concentrations, rates)
    
    # Fitting durchführen
    result = fit_parameters(substrate_data, activities, model_info)

    return result

if __name__ == "__main__":

    
    BASE_PATH = r"C:\Users\berger\Documents\Projekts\enzyme-cascade-analysis\Fehlerfortpflanzunganalyse"
    
    calibration_data = pd.read_csv(os.path.join(BASE_PATH, 'Data', 'NADH_Kalibriergerade.csv'))
    calibration_slope = calculate_calibration(calibration_data)

    r1_path = os.path.join(BASE_PATH, 'Data', 'Reaction1')
    r1_nad_data = pd.read_csv(os.path.join(r1_path, 'r_1_NAD_PD_500mM.csv'), header=None)
    r1_pd_data = pd.read_csv(os.path.join(r1_path, 'r1_PD_NAD_5mM.csv'), header=None)


    reac_1_data = {
        "r1_nad": r1_nad_data,
        "r1_pd": r1_pd_data
    }

    reac1_activity_param = {
        "r1": {
            "Vf_well": 10.0,
            "Vf_prod": 1.0,
            "c_prod": 2.2108
        }
    }

    # Fix: get_rates function needs 3 parameters, not 2
    reac_1_rates = {
        "r1_nad_rates": get_rates_and_concentrations(r1_nad_data, calibration_slope, reac1_activity_param["r1"])[0],
        "r1_pd_rates":  get_rates_and_concentrations(r1_pd_data, calibration_slope, reac1_activity_param["r1"])[0],
    }

    reac_1_concentrations = {
        "r1_nad_conc": get_rates_and_concentrations(r1_nad_data, calibration_slope, reac1_activity_param["r1"])[1],
        "r1_nad_const": 5.0,  # Konstante NAD Konzentration
        "r1_pd_conc":  get_rates_and_concentrations(r1_pd_data, calibration_slope, reac1_activity_param["r1"])[1],
        "r1_pd_const": 500.0,  # Konstante PD Konzentration
    }

    def two_substrat_michaelis_menten(concentration_data, Vmax, Km1, Km2): 
        """Zwei-Substrat Michaelis-Menten Gleichung"""
        S1_values, S2_values = concentration_data

        # Sicherstellen, dass es numpy arrays sind
        S1_values = np.asarray(S1_values)
        S2_values = np.asarray(S2_values)
        
        # Element-weise Berechnung für alle Datenpunkte
        rates = (Vmax * S1_values * S2_values) / ((Km1 + S1_values) * (Km2 + S2_values))
        
        return rates

    reac_1_model = {
        "name": "two_substrat_michaelis_menten",
        "function": two_substrat_michaelis_menten,
        "param_names": ["Vmax", "Km_NAD", "Km_PD"],
        "param_units": ["U", "mM", "mM"],
        "substrate_keys": ["r1_nad_conc", "r1_pd_conc"],
        "initial_guess_func": lambda activities, substrate_data: [max(activities) if len(activities) > 0 else 1.0, 1.0, 1.0],
        "bounds_lower": [0, 0, 0],
        "bounds_upper": [np.inf, np.inf, np.inf],
        "description": "Zwei-Substrat Michaelis-Menten für Reaktion 1"
    }

    reac_1_parameters = estimate_parameters(reac_1_model, reac_1_concentrations, reac_1_rates)

    print("\n=== Reaktion 1 Parameter Schätzung ==="
          f"\nModell: {reac_1_model['description']}"
          f"\nErgebnis: {reac_1_parameters}"
          f"\nR²: {reac_1_parameters['r_squared']:.4f}"
          f"\nVmax: {reac_1_parameters['params'][0]:.4f} {reac_1_model['param_units'][0]}"
          f"\nKm1: {reac_1_parameters['params'][1]:.4f} {reac_1_model['param_units'][1]}"
          f"\nKm2: {reac_1_parameters['params'][2]:.4f} {reac_1_model['param_units'][2]}"
          f"\nVmax Fehler: {reac_1_parameters['param_errors'][0]:.4f} {reac_1_model['param_units'][0]}"
          f"\nKm1 Fehler: {reac_1_parameters['param_errors'][1]:.4f} {reac_1_model['param_units'][1]}"
          f"\nKm2 Fehler: {reac_1_parameters['param_errors'][2]:.4f} {reac_1_model['param_units'][2]}"
        )
