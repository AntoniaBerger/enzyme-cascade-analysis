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

r1_path = os.path.join(BASE_PATH, 'Data', 'Reaction1')
r1_nad_data = pd.read_csv(os.path.join(r1_path, 'r_1_NAD_PD_500mM.csv'))
r1_pd_data = pd.read_csv(os.path.join(r1_path, 'r1_PD_NAD_5mM.csv'))


reaction_1_data = {
    "r1": {
        "c1": r1_nad_data,
        "c2": r1_pd_data
    }
}


reaction1_data_info = {
    "r1": {
        "Vf_well": 10.0,
        "Vf_prod": 1.0,
        "c_prod": 2.2108,
        "c1_const": 5.0,
        "c2_const": 500.0
    },
    "x_dimension": 2,
    "y_dimension": 1
}

reaction1_dataframe = compute_processed_data(
    reaction_1_data,
    calibration_slope,
    reaction1_data_info
)

print("Anzahl der Datenpunkte pro Reaction")
print(reaction1_dataframe["reaction"].value_counts())

reaction1_dataframe.to_pickle(os.path.join(os.path.dirname(__file__), 'Results', 'reaction1_dataframe.pkl'))
reaction1_dataframe.to_csv(os.path.join(os.path.dirname(__file__), 'Results', 'reaction1_dataframe.csv'))

def reaction_1(concentration_data, Vmax1, KmPD, KmNAD):
    """
    Wrapper für curve_fit Kompatibilität - nimmt flache Parameter entgegen
    Berechnet die Enzymaktivität für die erste Reaktion
    
    ALLE DREI REAKTIONEN:
    - Reaktion 1: PD + NAD → Pyruvat + NADH
    """
    # Entpacke Substratkonzentrationen, Inhibitor-Konzentrationen und Reaktions-IDs
    S1, S2, Inhibitor, reaction_ids = concentration_data

    # Initialisiere Ergebnis-Array
    V_obs = np.zeros_like(S1, dtype=float)
    
    # Reaktion 1: PD + NAD → HD + NADH
    reaction_1_mask = (reaction_ids == 1)
    if np.any(reaction_1_mask):
        # S1 = NAD oder konstante NAD, S2 = PD oder konstante PD
        S1_r1 = S1[reaction_1_mask] # NAD
        S2_r1 = S2[reaction_1_mask] # PD
        
        V_obs[reaction_1_mask] = (Vmax1 * S1_r1 * S2_r1) / (
            (KmPD + S2_r1) *  (KmNAD + S1_r1)
        )
    
    return V_obs


reaction1_model_info = {
    "name": "reaction_1",
    "function": reaction_1,
    "param_names": [
        "Vmax1", "KmPD", "KmNAD"
    ],
    "param_units": [
        "U",
        "mM", "mM"
    ],
    "substrate_keys": ["S1", "S2", "Inhibitor", "reaction_ids"],
    "initial_guess_func": lambda activities, substrate_data: [
        max(activities) if len(activities) > 0 else 1.0,  # Vmax1
        84.0,  # KmPD
        2.2   # KmNAD
        ],
    "bounds_lower": [0]*3,
    "bounds_upper": [np.inf]*3,
    "description": "Komplettes Drei-Reaktions-System mit Inhibitionen"
}

reaction1_parameters = estimate_parameters(reaction1_model_info, reaction1_data_info, reaction1_dataframe,sigma = 0.01)

print("\n=== Parameter Schätzung für das vollständige System ==="
        f"\nModell: {reaction1_parameters['description']}"
        f"\nR²: {reaction1_parameters['r_squared']:.4f}"
        f"\ncovariance: {reaction1_parameters['correlation_matrix']}")

for i, param_name in enumerate(reaction1_model_info['param_names']):
    param_val = reaction1_parameters['params'][i]
    param_err = reaction1_parameters['param_errors'][i]
    unit = reaction1_model_info['param_units'][i]
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
    reaction_1_data,
    reaction1_model_info,
    reaction1_data_info,
    noise_level,
    noise_model="plate_reader",
    n_iterations=1000
)

# Erstelle alle Ergebnisse mit demselben Zeitstempel in einem Ordner
results_dir = os.path.join(os.path.dirname(__file__), 'Results_reaction1')
timestamp = plot_monte_carlo_results(monte_carlo_results, reaction1_model_info, save_path=results_dir)
create_monte_carlo_report(monte_carlo_results, reaction1_model_info, save_path=results_dir, timestamp=timestamp)
plot_corner_plot(monte_carlo_results, reaction1_model_info, show_plots=True, save_path=results_dir, timestamp=timestamp)


monte_carlo_results = monte_carlo_simulation(
    calibration_data,
    reaction_1_data,
    reaction1_model_info,
    reaction1_data_info,
    noise_level,
    noise_model="processed_data",
    n_iterations=1000
)

results_dir = os.path.join(os.path.dirname(__file__), 'Results_reaction1')
timestamp = plot_monte_carlo_results(monte_carlo_results, reaction1_model_info, save_path=results_dir)
create_monte_carlo_report(monte_carlo_results, reaction1_model_info, save_path=results_dir, timestamp=timestamp)
plot_corner_plot(monte_carlo_results, reaction1_model_info, show_plots=True, save_path=results_dir, timestamp=timestamp)