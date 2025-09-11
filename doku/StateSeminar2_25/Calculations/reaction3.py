# includes
import os
import sys
import shutil
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import pickle

BASE_PATH = r"C:\Users\berger\Documents\Projekts\enzyme-cascade-analysis\Fehlerfortpflanzunganalyse"

# Add BASE_PATH to Python path so we can import modules from there
sys.path.append(BASE_PATH)

from data_handler import calc_calibration_slope, compute_processed_data
from parameter_estimator import estimate_parameters, monte_carlo_simulation
from simulator import cadet_simulation_full_system
from plotter import plot_monte_carlo_results, create_monte_carlo_report, plot_fitting_quality, plot_parameter_convergence, plot_component_analysis,plot_corner_plot


BASE_PATH = r"C:\Users\berger\Documents\Projekts\enzyme-cascade-analysis\Fehlerfortpflanzunganalyse"

calibration_data = pd.read_csv(os.path.join(BASE_PATH, 'Data', 'NADH_Kalibriergerade.csv'))
calibration_slope = calc_calibration_slope(calibration_data)

r3_path = os.path.join(BASE_PATH, 'Data', 'Reaction3')
r3_lactol = pd.read_csv(os.path.join(r3_path, 'r_3_Lactol_NAD_5mM.csv'))
r3_nad = pd.read_csv(os.path.join(r3_path, 'r_3_NAD_Lactol_500mM.csv'))


reaction_3_data = {
    "r3": {
        "c1": r3_lactol,
        "c2": r3_nad
    }
}


reaction3_data_info = {
     "r3": {
        "Vf_well": 10.0,
        "Vf_prod": 10.0,
        "c_prod": 2.15,
        "c1_const": 500.0,
        "c2_const": 5.0
    },
    "x_dimension": 3,
    "y_dimension": 1
}

reaction3_dataframe = compute_processed_data(
    reaction_3_data,
    calibration_slope,
    reaction3_data_info
)

print("Anzahl der Datenpunkte pro Reaction")
print(reaction3_dataframe["reaction"].value_counts())

reaction3_dataframe.to_pickle(os.path.join(os.path.dirname(__file__), 'Results', 'reaction3_dataframe.pkl'))
reaction3_dataframe.to_csv(os.path.join(os.path.dirname(__file__), 'Results', 'reaction3_dataframe.csv'))

def reaction_3(concentration_data, Vmax3, KmLactol, KmNAD):
    """
    Wrapper für curve_fit Kompatibilität - nimmt flache Parameter entgegen
    Berechnet die Enzymaktivität für die dritte Reaktion

    ALLE DREI REAKTIONEN:
    - Reaktion 1: PD + NAD → Pyruvat + NADH
    """
    # Entpacke Substratkonzentrationen, Inhibitor-Konzentrationen und Reaktions-IDs
    S1, S2, Inhibitor, reaction_ids = concentration_data

    # Initialisiere Ergebnis-Array
    V_obs = np.zeros_like(S1, dtype=float)
    
    # Reaktion 1: PD + NAD → HD + NADH
    reaction_3_mask = (reaction_ids == 3)
    if np.any(reaction_3_mask):
        S1_r3 = S1[reaction_3_mask]  # Lactol
        S2_r3 = S2[reaction_3_mask]  # NAD
        V_obs[reaction_3_mask] = (Vmax3 * S1_r3 * S2_r3) / (
            (KmLactol  + S1_r3) * (KmNAD + S2_r3)
        )
    
    return V_obs

reaction3_model_info = {
    "name": "reaction_3",
    "function": reaction_3,
    "param_names": [
        "Vmax3", "KmLactol", "KmNAD"
    ],
    "param_units": [
        "U",
        "mM", "mM"
    ],
    "substrate_keys": ["S1", "S2", "Inhibitor", "reaction_ids"],
    "initial_guess_func": lambda activities, substrate_data: [
        max(activities) if len(activities) > 0 else 1.0,  # Vmax3
        62,  # KmLactol
        2.8,   # KmNAD
        ],
    "bounds_lower": [0]*3,
    "bounds_upper": [np.inf]*3,
    "description": "Komplettes Drei-Reaktions-System mit Inhibitionen"
}

reaction3_parameters = estimate_parameters(reaction3_model_info, reaction3_data_info, reaction3_dataframe,sigma = 0.01)

print("\n=== Parameter Schätzung für das vollständige System ==="
        f"\nModell: {reaction3_parameters['description']}"
        f"\nR²: {reaction3_parameters['r_squared']:.4f}"
        f"\ncovariance: {reaction3_parameters['correlation_matrix']}")

for i, param_name in enumerate(reaction3_model_info['param_names']):
    param_val = reaction3_parameters['params'][i]
    param_err = reaction3_parameters['param_errors'][i]
    unit = reaction3_model_info['param_units'][i]
    print(f"{param_name}: {param_val:.4f} ± {param_err:.4f} {unit}")


noise_level = {
    "calibration": 0.00,
    "reaction": 0.01,
    "concentration": 0.00
}
simulation_dir = "Results/Simulations"

if os.path.exists(simulation_dir):
    shutil.rmtree(simulation_dir)
    print(f"Simulations-Ordner gelöscht: {simulation_dir}")

monte_carlo_results = monte_carlo_simulation(
    calibration_data,
    reaction_3_data,
    reaction3_model_info,
    reaction3_data_info,
    noise_level,
    noise_model="plate_reader",
    n_iterations=1000
)

# Erstelle alle Ergebnisse mit demselben Zeitstempel in einem Ordner
results_dir = os.path.join(os.path.dirname(__file__), 'Results_reaction3')
timestamp = plot_monte_carlo_results(monte_carlo_results, reaction3_model_info, save_path=results_dir)
create_monte_carlo_report(monte_carlo_results, reaction3_model_info, save_path=results_dir, timestamp=timestamp)
plot_corner_plot(monte_carlo_results, reaction3_model_info, show_plots=True, save_path=results_dir, timestamp=timestamp)

monte_carlo_results = monte_carlo_simulation(
    calibration_data,
    reaction_3_data,
    reaction3_model_info,
    reaction3_data_info,
    noise_level,
    noise_model="processed_data",
    n_iterations=1000
)

# Erstelle alle Ergebnisse mit demselben Zeitstempel in einem Ordner
results_dir = os.path.join(os.path.dirname(__file__), 'Results_reaction3')
timestamp = plot_monte_carlo_results(monte_carlo_results, reaction3_model_info, save_path=results_dir)
create_monte_carlo_report(monte_carlo_results, reaction3_model_info, save_path=results_dir, timestamp=timestamp)
plot_corner_plot(monte_carlo_results, reaction3_model_info, show_plots=True, save_path=results_dir, timestamp=timestamp)

